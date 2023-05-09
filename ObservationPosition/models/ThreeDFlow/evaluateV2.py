import sys 
import os 
import torch, numpy as np, torch.utils.data
import time
import datetime
import logging
from tqdm import tqdm 
from models import ThreeDFlow
from models_kitti import ThreeDFlow_Kitti
from models import multiScaleLoss
from pathlib import Path
from collections import defaultdict
import transforms
import datasets
import cmd_args 
from main_utils import *
from utils import geometry
from evaluation_utils import evaluate_2d, evaluate_3d
from torch.utils.data import DataLoader
from datasets.generic import Batch

def main():

    #import ipdb; ipdb.set_trace()
    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1,2,3'

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
    os.system('cp %s %s' % ('config_evaluate.yaml', log_dir))

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
        num_point = args.num_points*2 # 16384
    else:
        model = ThreeDFlow(args.is_training)
        num_point = args.num_points # 8192

    # val_dataset = datasets.__dict__[args.dataset](
    #     train=False,
    #     transform=transforms.ProcessData(args.data_process,
    #                                      num_point,
    #                                      args.allow_less_points),
    #     num_points= num_point,
    #     data_root = args.data_root
    # )
    # logger.info('val_dataset: ' + str(val_dataset))
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.workers,
    #     pin_memory=True,
    #     worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    # )
    
    # from datasets.flyingthings3d_hplflownet import FT3D
    # args.dataset_path = '/dataset/public_dataset_nas/flownet3d/FlyingThings3D_subset_processed_35m/'
    
    
    from datasets.flyingthings3d_flownet3d import FT3D
    args.dataset_path = '/dataset/public_dataset_nas/flownet3d/data_processed_maxcut_35_20k_2k_8192/data_processed_maxcut_35_20k_2k_8192/'
    
    test_dataset = FT3D(root_dir=args.dataset_path, nb_points=args.num_points, mode="test")
    # if args.dataset == 'HPLFlowNet':
    #     if args.dataset_cls == 'FT3D':
    #         from datasets.flyingthings3d_hplflownet import FT3D
    #         args.dataset_path = '/dataset/public_dataset_nas/flownet3d/FlyingThings3D_subset_processed_35m/'
    #         # lr_lambda = lambda epoch: 1.0 if epoch < 50 else 0.1
    #         # dataset = FT3D(root_dir=args.dataset_path, nb_points=args.n_points, mode="train")
    #         # test_dataset = FT3D(root_dir=args.dataset_path, nb_points=args.n_points, mode=mode)
    #     elif args.dataset_cls == 'Kitti':
    #         from datasets.kitti_hplflownet import Kitti
    #         args.dataset_path = '/home/wangsj/data/KITTI_processed_occ_final/'
    #         # dataset = Kitti(root_dir=args.dataset_path, nb_points=args.n_points, mode="train")
    #         # test_dataset = Kitti(root_dir=args.dataset_path, nb_points=args.n_points, mode=mode)
    # elif args.dataset == 'FlowNet3D':
    #     if args.dataset_cls == 'FT3D':
    #         from datasets.flyingthings3d_flownet3d import FT3D
    #         args.dataset_path = '/dataset/public_dataset_nas/flownet3d/data_processed_maxcut_35_20k_2k_8192/data_processed_maxcut_35_20k_2k_8192/'
    #         # lr_lambda = lambda epoch: 1.0 if epoch < 340 else 0.1
    #         # dataset = FT3D(root_dir=args.dataset_path, nb_points=args.n_points, mode="train")
    #         # test_dataset = FT3D(root_dir=args.dataset_path, nb_points=args.n_points, mode=mode)
    #     elif args.dataset_cls == 'Kitti':
    #         from datasets.kitti_flownet3d import Kitti
    #         args.dataset_path = '/home/wangsj/data/kitti_rm_ground'
    # elif args.dataset == 'Carla3D':
    #     from datasets.carla import CARLA3D
    #     # args.dataset_path = '/home2/public_workspace/carla_scene_flow/'
    # else:
    #     raise ValueError("Invalid dataset name: " + args.dataset)

    # Training dataset
    # ft3d_train = FT3D(root_dir=args.dataset_path, nb_points=args.n_points, mode="train", timeout=2)
    # use_test = True
    # mode = "test" if use_test else "val"
    # assert mode == "val" or mode == "test", "Problem with mode " + mode
    # if args.dataset_cls == 'FT3D':
    #     dataset = FT3D(root_dir=args.dataset_path, nb_points=args.n_points, mode="train")
    #     test_dataset = FT3D(root_dir=args.dataset_path, nb_points=args.n_points, mode=mode)
    # elif args.dataset_cls == 'Kitti':
    #     dataset = Kitti(root_dir=args.dataset_path, nb_points=args.n_points, mode="train")
    #     test_dataset = Kitti(root_dir=args.dataset_path, nb_points=args.n_points, mode="val")
    # elif args.dataset_cls == 'Carla3D':
    #     use_fg_inds = True
    #     args.random_dataset = False
    #     if args.random_dataset:
    #         dataset = CARLA3D(root_dir=args.dataset_path, nb_points=args.n_points, mode="train", use_fg_inds=use_fg_inds)
    #         test_dataset = CARLA3D(root_dir=args.dataset_path, nb_points=args.n_points, mode="test", use_fg_inds=use_fg_inds)
    #     else:
    #         dataset = CARLA3D(root_dir=args.dataset_path+'train/', nb_points=args.n_points, mode="train", use_fg_inds=use_fg_inds)
    #         test_dataset = CARLA3D(root_dir=args.dataset_path+'val/', nb_points=args.n_points, mode="test", use_fg_inds=use_fg_inds)
    # else:
    #     raise ValueError("Invalid dataset_cls name: " + args.dataset_cls)

    # train_loader = DataLoader(
    #     dataset,
    #     batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=True,
    #     collate_fn=Batch, drop_last=True, timeout=0, persistent_workers=True)
    val_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True,
        collate_fn=Batch, drop_last=False, timeout=0, persistent_workers=True)
    
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
    # 2D
    epe2ds = AverageMeter()
    acc2ds = AverageMeter()

    total_loss = 0
    total_seen = 0
    total_epe = 0
    metrics = defaultdict(lambda:list())
    for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.9):
        # pos1, pos2, norm1, norm2, flow, path = data  
        # pos1 = pos1.cuda()
        # pos2 = pos2.cuda() 
        # norm1 = norm1.cuda()
        # norm2 = norm2.cuda()
        # flow = flow.cuda() 
        pos1 = data['sequence'][0]
        pos2 = data['sequence'][1]
        mask1 = data['ground_truth'][0]
        # mask1 = torch.ones([pc1.shape[0], pc1.shape[1]]).contiguous()
        flow = data['ground_truth'][1]

        #move to cuda 
        pos1 = pos1.cuda()
        pos2 = pos2.cuda() 
        norm1 = pos1.cuda()
        norm2 = pos2.cuda()
        flow = flow.cuda() 

        model = model.eval()
        with torch.no_grad():
            
            pred_flows, gt_flows, pc1, pc2,raw_pc1,raw_pc2 = model(pos1, pos2, norm1, norm2,flow)
            
            loss = multiScaleLoss(pred_flows, gt_flows)

            full_flow = pred_flows[0].permute(0, 2, 1)
            epe3d = torch.norm(full_flow - gt_flows[0].permute(0, 2, 1), dim=2).mean()

        total_loss += loss.cpu().data * args.batch_size
        total_epe += epe3d.cpu().data * args.batch_size
        total_seen += args.batch_size

        pc1_np = raw_pc1.cpu().numpy()
        pc2_np = raw_pc2.cpu().numpy()
        sf_np = (gt_flows[0].permute(0, 2, 1)).cpu().numpy()
        pred_sf = full_flow.cpu().numpy()

        EPE3D, acc3d_strict, acc3d_relax, outlier = evaluate_3d(pred_sf, sf_np)

        epe3ds.update(EPE3D)
        acc3d_stricts.update(acc3d_strict)
        acc3d_relaxs.update(acc3d_relax)
        outliers.update(outlier)

        # # 2D evaluation metrics
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
    logger.info(res_str)


if __name__ == '__main__':
    main()




