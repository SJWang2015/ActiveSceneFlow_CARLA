"""
Evaluation
Author: Wenxuan Wu
Date: May 2020
"""

import argparse
import sys 
import os 

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import torch.nn.functional as F
import time
import torch.nn as nn
import pickle 
import datetime
import logging

from tqdm import tqdm 
from models import PointConvSceneFlowPWC8192selfglobalPointConv as PointConvSceneFlow
from models import multiScaleLoss
from pathlib import Path
from collections import defaultdict

import transforms
import datasets
from datasets.carla import CARLA3D
import cmd_args 
from main_utils import *
from utils import geometry
from evaluation_utils import evaluate_2d, evaluate_3d

def main():
    #import ipdb; ipdb.set_trace()
    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])
    # os.environ['CUDA_VISIBLE_DEVICES'] =  '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1'

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
    model = PointConvSceneFlow()
    random_dataset = False
    if args.dataset == 'carla':
        use_fg_inds = True
        if random_dataset:
            val_dataset = CARLA3D(root_dir=args.data_root, nb_points=args.num_points,
         mode="test", use_fg_inds=use_fg_inds)
        else:
            val_dataset = CARLA3D(root_dir=args.data_root+'val/', nb_points=args.num_points,
         mode="test")
    else:
        val_dataset = datasets.__dict__[args.dataset](
            train=False,
            transform=transforms.ProcessData(args.data_process,
                                            args.num_points,
                                            args.allow_less_points),
            num_points=args.num_points,
            data_root = args.data_root
        )

    print(args.data_root)
    logger.info('val_dataset: ' + str(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    '''GPU selection and multi-GPU'''
    if args.multi_gpu is not None:
        device_ids = [int(x) for x in args.multi_gpu.split(',')]
        torch.backends.cudnn.benchmark = True 
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids = device_ids)
    else:
        model.cuda()

    #load pretrained model
    pretrain = args.ckpt_dir + args.pretrain
    net_dict = model.state_dict()
    pretrained_dict = torch.load(pretrain)
    # for k, v in pretrained_dict.items():
    #     name = 'module.' + k[:]  # remove `module.`
    #     net_dict[name] = v
    model.load_state_dict(pretrained_dict, strict=True) 
    # model.load_state_dict(torch.load(pretrain))
    print('load model %s'%pretrain)
    logger.info('load model %s'%pretrain)

    # model.cuda()

    epe3ds = AverageMeter()
    acc3d_stricts = AverageMeter()
    acc3d_relaxs = AverageMeter()
    outliers = AverageMeter()
    # 2D
    epe2ds = AverageMeter()
    acc2ds = AverageMeter()
    
    fg_epe3ds = AverageMeter()
    fg_acc3d_stricts = AverageMeter()
    fg_acc3d_relaxs = AverageMeter()
    fg_outliers = AverageMeter()

    bg_epe3ds = AverageMeter()
    bg_acc3d_stricts = AverageMeter()
    bg_acc3d_relaxs = AverageMeter()
    bg_outliers = AverageMeter()

    total_loss = 0
    total_seen = 0
    total_epe = 0
    metrics = defaultdict(lambda:list())
    use_savefile = False
    for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.9):
        # # pos1, pos2, norm1, norm2, flow, path = data  

        # # #move to cuda 
        # # pos1 = pos1.cuda()
        # # pos2 = pos2.cuda() 
        # # norm1 = norm1.cuda()
        # # norm2 = norm2.cuda()
        # # flow = flow.cuda() 
        # pos1 = data['sequence'][0]
        # pos2 = data['sequence'][1]
        # ego_flow = data['ground_truth'][0]
        # flow = data['ground_truth'][1]
        # # fgrnd_inds_s = data['mask'][0]
        # # fgrnd_inds_t = data['mask'][1]
        # # mask1 = fgrnd_inds_s.cuda()
        
        # # pos1 = pos1.transpose(2, 1).contiguous().cuda()
        # # pos2 = pos2.transpose(2, 1).contiguous().cuda()
        # # flow = flow.transpose(2, 1).contiguous().cuda()
        # # ego_flow = ego_flow.transpose(2, 1).contiguous().cuda()
        # pos1 = pos1.squeeze(1).contiguous().cuda()
        # pos2 = pos2.squeeze(1).contiguous().cuda()
        # flow = flow.squeeze(1).contiguous().cuda()
        # ego_flow = ego_flow.squeeze(1).contiguous().cuda()
        # # pos2 = pos1 + ego_flow
        # mask1 = torch.ones([pos1.shape[0], pos1.shape[1], 1]).contiguous().cuda()
        pos1 = data['sequence'][0]
        pos2 = data['sequence'][1]
        # ego_flow = data['ground_truth'][0]
        flow = data['ground_truth'][1]
        
        frng_mask = data['mask'][0]
        frng_mask = frng_mask.permute(0,2,1).squeeze(-1).cuda()

        # fgrnd_inds_s = data['mask'][0]
        # fgrnd_inds_t = data['mask'][1]
        mask1 = torch.ones([pos1.shape[0], pos1.shape[2]]).contiguous().cuda()
        # mask1 = fgrnd_inds_s.cuda()
        pos1 = pos1.squeeze(1).contiguous().cuda()
        pos2 = pos2.squeeze(1).contiguous().cuda()
        flow = flow.squeeze(1).contiguous().cuda()
        # bg_mask = torch.norm(flow - ego_flow, dim=-1).cpu().numpy()
        # bg_flag = np.ones([pos1.shape[0],pos1.shape[1], 3 ])
        # for i in range(flow.shape[0]):
        #     bg_flag[i, bg_mask[i,:] > 1e-3, :] = 0
        # bg_flag = torch.tensor(bg_flag).cuda()

        model = model.eval()
        with torch.no_grad(): 
            pred_flows, fps_pc1_idxs, _, _, _ = model(pos1, pos2, pos1, pos2)
            
            loss = multiScaleLoss(pred_flows, flow, fps_pc1_idxs)
 
            full_flow = pred_flows[0].permute(0, 2, 1)
            # full_flow = full_flow * (1-bg_flag) + ego_flow * bg_flag 
            # epe3d = torch.norm(mask1.transpose(1,2) * (full_flow - flow), dim = 2).mean()
            epe3d = torch.norm((full_flow - flow), dim = 2).mean()
        
        if use_savefile:
            name_fmt = "{:0>6}".format(str(i)) + '.npz'
            np_src = pos1.squeeze(0).cpu().numpy()
            np_tgt = pos2.squeeze(0).cpu().numpy()
            gt = flow.squeeze(0).cpu().numpy()
            np_flow = full_flow.cpu().detach().squeeze(0).numpy()
            # np_ego_flow = ego_flow.cpu().detach().squeeze(0).numpy()
            # np_fgrnd_inds_s = fgrnd_inds_s.cpu().detach().squeeze(0).squeeze(0).numpy()
            # np_fgrnd_inds_t = fgrnd_inds_t.cpu().detach().squeeze(0).squeeze(0).numpy()
            # np.savez('./results/' + name_fmt, pos1=np_src, pos2=np_tgt, flow=np_flow, ego_flow=np_ego_flow, gt=gt, fgrnd_inds_s=np_fgrnd_inds_s, fgrnd_inds_t=np_fgrnd_inds_t)
            np.savez('./results/' + name_fmt, pos1=np_src, pos2=np_tgt, flow=np_flow, gt=gt)

        total_loss += loss.cpu().data * args.batch_size
        total_epe += epe3d.cpu().data * args.batch_size
        total_seen += args.batch_size

        pc1_np = pos1.cpu().numpy()
        pc2_np = pos2.cpu().numpy() 
        sf_np = flow.cpu().numpy()
        pred_sf = full_flow.cpu().numpy()

        EPE3D, acc3d_strict, acc3d_relax, outlier = evaluate_3d(pred_sf, sf_np, mask1.cpu().numpy())

        epe3ds.update(EPE3D)
        acc3d_stricts.update(acc3d_strict)
        acc3d_relaxs.update(acc3d_relax)
        outliers.update(outlier)
        
        fg_epe_3d, fg_acc_3d, fg_acc_3d_2, fg_outlier = evaluate_3d(pred_sf, sf_np, frng_mask.cpu().numpy())
        bg_epe_3d, bg_acc_3d, bg_acc_3d_2, bg_outlier = evaluate_3d(pred_sf, sf_np, (1.0-frng_mask).cpu().numpy())
        
        fg_epe3ds.update(fg_epe_3d)
        fg_acc3d_stricts.update(fg_acc_3d)
        fg_acc3d_relaxs.update(fg_acc_3d_2)
        fg_outliers.update(fg_outlier)

        bg_epe3ds.update(bg_epe_3d)
        bg_acc3d_stricts.update(bg_acc_3d)
        bg_acc3d_relaxs.update(bg_acc_3d_2)
        bg_outliers.update(bg_outlier)

        # # 2D evaluation metrics
        # flow_pred, flow_gt = geometry.get_batch_2d_flow(pc1_np,
        #                                                 pc1_np+sf_np,
        #                                                 pc1_np+pred_sf,
        #                                                 path)
        # EPE2D, acc2d = evaluate_2d(flow_pred, flow_gt)

        # epe2ds.update(EPE2D)
        # acc2ds.update(acc2d)
    epe2ds = 0.0
    acc2ds = 0.0
    mean_loss = total_loss / total_seen
    mean_epe = total_epe / total_seen
    str_out = '%s mean loss: %f mean epe: %f'%(blue('Evaluate'), mean_loss, mean_epe)
    # print("Total time consuming is %.4f." % (model.total_time))
    print(str_out)
    logger.info(str_out)

    res_str = (' * EPE3D {epe3d_.avg:.4f}\t'
               'ACC3DS {acc3d_s.avg:.4f}\t'
               'ACC3DR {acc3d_r.avg:.4f}\t'
               'Outliers3D {outlier_.avg:.4f}\t'
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




