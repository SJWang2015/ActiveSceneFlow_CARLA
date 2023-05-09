import sys 
import os 
import torch, numpy as np, torch.utils.data
import time
import datetime
import logging
from tqdm import tqdm 
from models import ThreeDFlow
# from models_kitti import ThreeDFlow_Kitti
from models_occlusion_kitti import ThreeDFlow_Kitti
from models import multiScaleLoss
from pathlib import Path
from collections import defaultdict
import transforms
from datasets.carla import CARLA3D
import cmd_args 
from main_utils import *
from utils import geometry
from evaluation_utils import evaluate_2d, evaluate_3d, evaluate_3d_mask

def main():

    #import ipdb; ipdb.set_trace()
    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '2'

    '''CREATE DIR'''
    experiment_dir = Path('./Evaluate_experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%sFlyingthings3d-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % ('models.py', log_dir))
    os.system('cp %s %s' % ('pointconv_util.py', log_dir))
    os.system('cp %s %s' % ('evaluate.py', log_dir))
    os.system('cp %s %s' % ('config_evaluate_carla.yaml', log_dir))

    '''LOG'''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'train_%s_sceneflow.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    blue = lambda x: '\033[94m' + x + '\033[0m'
    if args.dataset == "Kitti":
        model = ThreeDFlow_Kitti(args.is_training)
        num_point = args.num_points # 16384
    elif args.dataset == 'carla':
        model = ThreeDFlow(args.is_training)
        num_point = args.num_points # 16384
    else:
        model = ThreeDFlow(args.is_training)
        num_point = args.num_points # 8192
    
    if args.dataset == 'carla':
        val_dataset = CARLA3D(root_dir=args.data_root+'val/', nb_points=args.num_points,
         mode="test")
    else:
        val_dataset = datasets.__dict__[args.dataset](
            train=False,
            transform=transforms.ProcessData(args.data_process,
                                            num_point,
                                            args.allow_less_points),
            num_points= num_point,
            data_root = args.data_root
        )
    logger.info('val_dataset: ' + str(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    #load pretrained model
    pretrain = args.ckpt_dir + args.pretrain
    model.load_state_dict(torch.load(pretrain))
    print('load model %s'%pretrain)
    logger.info('load model %s'%pretrain)

    model.cuda()

    epe3ds = AverageMeter()
    acc3d_stricts = AverageMeter()
    acc3d_relaxs = AverageMeter()
    outliers = AverageMeter()

    fg_epe3ds = AverageMeter()
    fg_acc3d_stricts = AverageMeter()
    fg_acc3d_relaxs = AverageMeter()
    fg_outliers = AverageMeter()

    bg_epe3ds = AverageMeter()
    bg_acc3d_stricts = AverageMeter()
    bg_acc3d_relaxs = AverageMeter()
    bg_outliers = AverageMeter()

    # 2D
    epe2ds = AverageMeter()
    acc2ds = AverageMeter()

    total_loss = 0
    total_seen = 0
    total_epe = 0
    metrics = defaultdict(lambda:list())
    for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.9):
        # pos1, pos2, norm1, norm2, flow, path = data  

        #move to cuda 
        # pos1 = pos1.cuda()
        # pos2 = pos2.cuda() 
        # norm1 = norm1.cuda()
        # norm2 = norm2.cuda()
        # flow = flow.cuda() 
        pos1 = data['sequence'][0]
        pos2 = data['sequence'][1]
        # ego_flow = data['ground_truth'][0]
        flow = data['ground_truth'][1]

        frng_mask = data['mask'][0]
        frng_mask = frng_mask.permute(0,2,1).cuda()

        # fgrnd_inds_s = data['mask'][0]
        # fgrnd_inds_t = data['mask'][1]
        mask1 = torch.ones([pos1.shape[0], pos1.shape[2],1]).contiguous().cuda()
        # mask1 = fgrnd_inds_s.cuda()

        pos1 = pos1.squeeze(1).contiguous().cuda()
        pos2 = pos2.squeeze(1).contiguous().cuda()
        flow = flow.squeeze(1).contiguous().cuda()

        model = model.eval()
        with torch.no_grad():
            
            # pred_flows, gt_flows, pc1, pc2,raw_pc1,raw_pc2 = model(pos1, pos2, pos1, pos2, flow,mask1)
            pred_flows, gt_flows, pc1, pc2,_,_ = model(pos1, pos2,pos1, pos2,flow)
            
            loss = multiScaleLoss(pred_flows, gt_flows)

            full_flow = pred_flows[0].permute(0, 2, 1)
            epe3d = torch.norm(full_flow - gt_flows[0].permute(0, 2, 1), dim=2).mean()

        total_loss += loss.cpu().data * args.batch_size
        total_epe += epe3d.cpu().data * args.batch_size
        total_seen += args.batch_size

        # pc1_np = raw_pc1.cpu().numpy()
        # pc2_np = raw_pc2.cpu().numpy()
        sf_np = (gt_flows[0].permute(0, 2, 1)).cpu().numpy()
        pred_sf = full_flow.cpu().numpy()
        frng_mask = frng_mask.cpu().numpy()

        EPE3D, acc3d_strict, acc3d_relax, outlier = evaluate_3d(pred_sf, sf_np)

        fg_epe_3d, fg_acc_3d, fg_acc_3d_2, fg_outlier = evaluate_3d_mask(pred_sf, sf_np, frng_mask)
        bg_epe_3d, bg_acc_3d, bg_acc_3d_2, bg_outlier = evaluate_3d_mask(pred_sf, sf_np, 1.0-frng_mask)

        epe3ds.update(EPE3D)
        acc3d_stricts.update(acc3d_strict)
        acc3d_relaxs.update(acc3d_relax)
        outliers.update(outlier)

        fg_epe3ds.update(fg_epe_3d)
        fg_acc3d_stricts.update(fg_acc_3d)
        fg_acc3d_relaxs.update(fg_acc_3d_2)
        fg_outliers.update(fg_outlier)

        bg_epe3ds.update(bg_epe_3d)
        bg_acc3d_stricts.update(bg_acc_3d)
        bg_acc3d_relaxs.update(bg_acc_3d_2)
        bg_outliers.update(bg_outlier)

        # 2D evaluation metrics
        # flow_pred, flow_gt = geometry.get_batch_2d_flow(pc1_np,
        #                                                 pc1_np+sf_np,
        #                                                 pc1_np+pred_sf,
        #                                                 path)
        # EPE2D, acc2d = evaluate_2d(flow_pred, flow_gt)

        # epe2ds.update(EPE2D)
        # acc2ds.update(acc2d)

    mean_loss = total_loss / total_seen
    mean_epe = total_epe / total_seen
    str_out = '%s mean loss: %f mean epe: %f'%(blue('Evaluate'), mean_loss, mean_epe)
    print(str_out)
    logger.info(str_out)

    res_str = (' * EPE3D {epe3d_.avg:.4f}\t'
               'ACC3DS {acc3d_s.avg:.4f}\t'
               'ACC3DR {acc3d_r.avg:.4f}\t'
               'Outliers3D {outlier_.avg:.4f}'
            #    'EPE2D {epe2d_.avg:.4f}\t'
            #    'ACC2D {acc2d_.avg:.4f}'
               .format(
                       epe3d_=epe3ds,
                       acc3d_s=acc3d_stricts,
                       acc3d_r=acc3d_relaxs,
                       outlier_=outliers,
                    #    epe2d_=epe2ds,
                    #    acc2d_=acc2ds
                       ))
    print(res_str)

    res_str = (' * EPE3D {epe3d_.avg:.4f}\t'
               'ACC3DS {acc3d_s.avg:.4f}\t'
               'ACC3DR {acc3d_r.avg:.4f}\t'
               'Outliers3D {outlier_.avg:.4f}'
            #    'EPE2D {epe2d_.avg:.4f}\t'
            #    'ACC2D {acc2d_.avg:.4f}'
               .format(
                       epe3d_=fg_epe3ds,
                       acc3d_s=fg_acc3d_stricts,
                       acc3d_r=fg_acc3d_relaxs,
                       outlier_=fg_outliers,
                    #    epe2d_=epe2ds,
                    #    acc2d_=acc2ds
                       ))
    print(res_str)

    res_str = (' * EPE3D {epe3d_.avg:.4f}\t'
               'ACC3DS {acc3d_s.avg:.4f}\t'
               'ACC3DR {acc3d_r.avg:.4f}\t'
               'Outliers3D {outlier_.avg:.4f}'
            #    'EPE2D {epe2d_.avg:.4f}\t'
            #    'ACC2D {acc2d_.avg:.4f}'
               .format(
                       epe3d_=bg_epe3ds,
                       acc3d_s=bg_acc3d_stricts,
                       acc3d_r=bg_acc3d_relaxs,
                       outlier_=bg_outliers,
                    #    epe2d_=epe2ds,
                    #    acc2d_=acc2ds
                       ))
    print(res_str)
    logger.info(res_str)


if __name__ == '__main__':
    main()




