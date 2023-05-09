# This is a sample Python script.
from __future__ import print_function

import argparse

import os
import sys
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.utils.data import DataLoader

import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
from math import log, sqrt, pi

# from utils import datasets as datsets
from utils.datasets.carla import Batch
from utils.datasets.generic import Batch
# from AEMFlow import AEMFlow, multiScaleLoss
# from lr_sp_loss import lowrank_loss,subspace_loss
from TFlowV3_Occlussion import TFlow, multiScaleLoss
version_str = './TFlowV3_Occlussion.py'
# from calc_coarse_flow import calc_coarse_flow_from_bev

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _init_(args):
    if args.rm_history and not args.eval:
        if os.path.exists(args.model_dir + args.exp_name):
            os.system('rm -r ' + args.model_dir + args.exp_name)
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        if not os.path.exists(args.model_dir + args.exp_name):
            os.makedirs(args.model_dir + args.exp_name)
        if not os.path.exists(args.model_dir + args.exp_name + '/' + 'models'):
            os.makedirs(args.model_dir + args.exp_name + '/' + 'models')
        os.system('cp main_transflow.py ' + args.model_dir + args.exp_name + '/' + 'main.py.backup')
        os.system('cp ' + version_str + ' ' + args.model_dir + args.exp_name + '/' + args.model + '.py.backup')
        os.system('cp ./utils/utils.py ' + args.model_dir + args.exp_name + '/' + 'utils.py.backup')
        os.system('cp ./utils/soflow.py ' + args.model_dir + args.exp_name + '/' + 'soflow.py.backup')
        os.system('cp ./utils/hnfflow.py ' + args.model_dir + args.exp_name + '/' + 'hnfflow.py.backup')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def criterion(pred_flow, flow, mask=None):
    if mask is None:
        loss = torch.mean(torch.sum((pred_flow - flow) * (pred_flow - flow), dim=1) / 2.0)
    else:
        loss = torch.mean(mask * torch.sum((pred_flow - flow) * (pred_flow - flow), dim=1) / 2.0)
    return loss


def error(sf_pred, sf_gt, mask=None):
    sf_pred = sf_pred.permute(0, 2, 1).cpu().numpy()
    sf_gt = sf_gt.permute(0, 2, 1).cpu().numpy()
    if mask is None:
        mask = np.ones([sf_pred.shape[0], sf_pred.shape[1]])
    else:
        mask = mask.squeeze(-1).cpu().numpy()
    
    l2_norm = np.linalg.norm(sf_gt - sf_pred, axis=-1)
    sf_norm = np.linalg.norm(sf_gt, axis=-1)
    if mask is not None:
        l2_norm *= mask
        sf_norm *= mask
    # EPE3D = l2_norm.mean()
    mask_sum = np.sum(mask, 1)
    EPE3D = np.sum(l2_norm, 1) / (mask_sum + 1e-10)
    EPE3D = np.mean(EPE3D)

    relative_err = l2_norm / (sf_norm + 1e-10)

    acc3d_strict = np.sum(np.logical_or((l2_norm < 0.05)*mask, (relative_err < 0.05)*mask), axis=1).astype(np.float)
    acc3d_relax = np.sum(np.logical_or((l2_norm < 0.1)*mask, (relative_err < 0.1)*mask), axis=1).astype(np.float)
    outlier = np.sum(np.logical_or((l2_norm >= 0.3)*mask, (relative_err >= 0.1)*mask), axis=1).astype(np.float)

    acc3d_strict = acc3d_strict[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc3d_strict = np.mean(acc3d_strict)
    acc3d_relax = acc3d_relax[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc3d_relax = np.mean(acc3d_relax)
    outlier = outlier[mask_sum > 0] / mask_sum[mask_sum > 0]
    outlier = np.mean(outlier)

    return EPE3D, acc3d_strict, acc3d_relax, outlier

# @torch.no_grad()
def test_one_epoch(args, net, test_loader, epoch=0):
    net.eval()
    total_loss = 0
    total_epe = 0
    total_acc3d = 0
    total_acc3d_2 = 0
    outliers = 0
    num_examples = 0
    use_savefile = False

    total_time = 0.0
    with tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9)as pbar:
        for i, data in enumerate(test_loader):
            # pc1, pc2, flow = data
            pc1 = data['sequence'][0]
            pc2 = data['sequence'][1]
            mask1 = data['ground_truth'][0]
            flow = data['ground_truth'][1]

            # pc1, pc2, flow, mask1 = data 
            pc1 = pc1.contiguous().cuda().float()
            pc2 = pc2.contiguous().cuda().float()
            pc1 = pc1.transpose(2, 1).contiguous()
            pc2 = pc2.transpose(2, 1).contiguous()
 
            flow = flow.contiguous().cuda().float()
            # flow,_,_,_ = torch.split(flow,2048,1) # 2048
            flow = flow.transpose(2, 1).contiguous()
            mask1 = mask1.cuda().float()#.transpose(2, 1).contiguous().float()
            if mask1 != None:
                mask1 = torch.ones([pc1.shape[0], pc1.shape[-1], 1]).contiguous().cuda()


            batch_size = pc1.size(0)
            num_examples += batch_size

            # resample 10 times
            pred_flow_sum = torch.zeros(pc1.shape[0], 3, pc1.shape[-1]).cuda()
            pred_c_flow_sum = torch.zeros(pc1.shape[0], 3, pc1.shape[-1]).cuda()

            repeat_num = 1 #args.repeat_num
            if repeat_num > 1:
                for j in range(repeat_num):
                    # print(i)
                    perm = torch.randperm(pc1.shape[2])
                    points1_perm = pc1[:, :, perm]
                    points2_perm = pc2[:, :, perm]
                    # feats1_perm = feats1[:, :, perm]
                    # feats2_perm = feats2[:, :, perm]
                    # ego_flow_perm = coarse_sf[:, :, perm]
                    # pred_c_flow_sum[:, :, perm] += ego_flow_perm
                    pred_c_flow_sum = pred_c_flow_sum
                    with torch.no_grad():
                        # flow_pred, fps_pc1_idxs = net(points1_perm, points2_perm, feats1_perm, feats2_perm)
                        flow_pred, fps_pc1_idxs = net(points1_perm, points2_perm)
                    pred_flow_sum[:, :, perm] += flow_pred[0]
                    pred_flow_sum = pred_flow_sum
            else:
                with torch.no_grad():
                    # coarse_sf = torch.zeros_like(flow)
                    # flow_pred, fps_pc1_idxs = net(pc1, pc2, feats1, feats2)
                    flow_pred, fps_pc1_idxs = net(pc1, pc2)
                pred_flow_sum += flow_pred[0]
   

            pred_flow_sum /= repeat_num

            if use_savefile:
                name_fmt = "{:0>6}".format(str(i)) + '.npz'
                np_src = pc1.permute(0, 2, 1).squeeze(0).cpu().numpy()
                np_tgt = pc2.permute(0, 2, 1).squeeze(0).cpu().numpy()
                gt = flow.permute(0, 2, 1).squeeze(0).cpu().numpy()
                np_flow = pred_flow_sum.permute(0, 2, 1).cpu().detach().squeeze(0).numpy()
                np.savez('./results_carla/' + name_fmt, pos1=np_src, pos2=np_tgt, flow=np_flow, gt=gt)
            # if flow is not None:
            if repeat_num > 1:
                loss = criterion(pred_flow_sum, flow)
            else:
                loss = multiScaleLoss(flow_pred, flow, mask1, fps_pc1_idxs)
                # loss = multiScaleLoss(flow_pred, flow, fgrnd_inds_s.cuda(), fps_pc1_idxs)

            total_loss += loss.item() * batch_size
            epe_3d, acc_3d, acc_3d_2, outlier = error(pred_flow_sum, flow, mask1)

            total_epe += epe_3d * batch_size
            total_acc3d += acc_3d * batch_size
            total_acc3d_2 += acc_3d_2 * batch_size
            outliers += outlier * batch_size
            pbar.set_postfix({blue('Loss'): '{0:1.5f}'.format(total_loss * 1.0 / num_examples)})  # 输入一个字典，显示实验指标
            pbar.update(1)


    textio.cprint('==Total TEST==')
    textio.cprint(
        'mean test loss: %f\tEPE 3D: %f\tACC 3D: %f\tACC 3D 2: %f\tOutlier: %f' % (total_loss * 1.0 / num_examples, total_epe * 1.0 / num_examples, total_acc3d * 1.0 / num_examples, total_acc3d_2 * 1.0 / num_examples, outliers * 1.0 / num_examples))

    boardio.add_scalar('Eval/EPE_3d', total_epe * 1.0 / num_examples, epoch)
    boardio.add_scalar('Eval/ACC_3d', total_acc3d * 1.0 / num_examples, epoch)
    boardio.add_scalar('Eval/ACC_3d_2', total_acc3d_2 * 1.0 / num_examples, epoch)
    boardio.add_scalar('Eval/Outlier', outliers * 1.0 / num_examples, epoch)
    boardio.add_scalar('Eval/Mean_loss', total_loss * 1.0 / num_examples, epoch)
   
    return total_loss * 1.0 / num_examples, total_epe * 1.0 / num_examples, total_acc3d * 1.0 / num_examples, total_acc3d_2 * 1.0 / num_examples, outliers * 1.0 / num_examples


def train_one_epoch(args, net, train_loader, opt, epoch=0):
    net.train()
    num_examples = 0
    total_loss = 0.0
    epoch_loss = 0.0
    data_size = len(train_loader)
    step = 100#args.step

    with tqdm(enumerate(train_loader), total=len(train_loader)) as pbar:
        for i, data in enumerate(train_loader):
            pc1 = data['sequence'][0]
            pc2 = data['sequence'][1]
            # feats1 = data['sequence'][2]
            # feats2 = data['sequence'][3]
            mask1 = data['ground_truth'][0]
            # # mask1 = torch.ones([pc1.shape[0], pc1.shape[1]]).contiguous().cuda()
            flow = data['ground_truth'][1]
            # pc1, pc2, flow, mask1 = data 
            pc1 = pc1.contiguous().cuda().float()
            pc2 = pc2.contiguous().cuda().float()
            pc1 = pc1.transpose(2, 1).contiguous()
            pc2 = pc2.transpose(2, 1).contiguous()
            flow = flow.contiguous().cuda().float()
            flow = flow.transpose(2, 1).contiguous()

            mask1 = torch.ones([pc1.shape[0], pc1.shape[-1], 1]).contiguous().cuda()

            batch_size = pc1.size(0)
            opt.zero_grad()
            num_examples += batch_size
            # pred_flows, fps_pc1_idxs = net(pc1, pc2, feats1, feats2)
            pred_flows, fps_pc1_idxs = net(pc1, pc2)

            loss = multiScaleLoss(pred_flows, flow, mask1, fps_pc1_idxs)
       
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch_size

            epoch_loss += loss.item() * batch_size

            if (i + 1) % step == 0:
                epoch_cnt = epoch * np.round(data_size / step) + (i + 1) / step - 1
                out_str = 'Train/Epoch100' + '_loss'
                boardio.add_scalar(out_str, epoch_loss / step / batch_size, epoch_cnt)
                epoch_loss = 0.0

            pbar.set_postfix({blue('Loss'): '{0:1.5f}'.format(total_loss * 1.0 / num_examples)})  # 输入一个字典，显示实验指标
            pbar.update(1)

    boardio.add_scalar('Train/Loss', total_loss * 1.0 / num_examples, epoch)
    for param_group in opt.param_groups:
        lr = float(param_group['lr'])
        break
    boardio.add_scalar('Train/learning_rate', lr, epoch)
    return total_loss * 1.0 / num_examples


def test(args, net, test_loader, boardio, textio):
    test_loss, epe, acc, acc_2, outlier = test_one_epoch(args, net, test_loader)
    textio.cprint('==FINAL TEST==')
    textio.cprint(
        'mean test loss: %f\tEPE 3D: %f\tACC 3D: %f\tACC 3D 2: %f\tOutlier: %f' % (test_loss, epe, acc, acc_2, outlier))

def exp_lr_scheduler(optimizer, global_step, init_lr, decay_steps, decay_rate, lr_clip, staircase=True):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if staircase:
        lr = init_lr * decay_rate**(global_step // decay_steps)
    else:
        lr = init_lr * decay_rate**(global_step / decay_steps)
    lr = max(lr, lr_clip)

    if global_step % decay_steps == 0:
        print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(args, net, train_loader, test_loader, boardio, textio):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    best_test_loss = np.inf
    for epoch in range(args.epochs):
        global_step = epoch * len(train_loader) * args.batch_size
        lr = exp_lr_scheduler(opt, global_step, args.lr, args.decay_steps, args.decay_rate, 0.000001, staircase=True)
        textio.cprint('==epoch: %d==' % epoch)
        textio.cprint('==global_step: %d==' % global_step)
        textio.cprint('==learning_rate: %f==' % lr)
        train_loss = train_one_epoch(args, net, train_loader, opt, epoch)
        textio.cprint('mean train EPE loss: %f' % train_loss)

        test_loss, epe, acc, acc_2, outlier = test_one_epoch(args, net, test_loader, epoch)
        textio.cprint(
            '%s: %f\tEPE 3D: %f\tACC 3D: %f\tACC 3D 2: %f\tOutlier: %f' % (
                blue('mean test loss'), test_loss, epe, acc, acc_2, outlier))
        if best_test_loss >= epe:
            best_test_loss = epe
            # textio.cprint('best test loss till now: %f' % best_test_loss)
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), '%s%s/models/model.best.t7' % (args.model_dir, args.exp_name))
            else:
                torch.save(net.state_dict(), '%s%s/models/model.best.t7' % (args.model_dir, args.exp_name))
        textio.cprint('%s till now: %f' % (red('best test loss'), best_test_loss))
        # scheduler.step()

    if torch.cuda.device_count() > 1:
        torch.save(net.module.state_dict(), '%s%s/models/model.best.final.t7' % (args.model_dir, args.exp_name))
    else:
        torch.save(net.state_dict(), '%s%s/models/model.best.final.t7' % (args.model_dir, args.exp_name))


def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N', help='Name of the experiment')
    parser.add_argument('--model', type=str, default='TFlow', metavar='N',
                        choices=['TFlow'],
                        help='Model to use, [TFlow]')
    parser.add_argument('--dataset', type=str, default='Carla3D', metavar='N',
                        help='Name of the dataset mode:[HPLFlowNet, FlowNet3D, Carla3D]')
    parser.add_argument('--dataset_cls', type=str, default='Carla3D',
                        metavar='N', choices=['Kitti', 'FT3D', 'Carla3D'],
                        help='dataset to use: [Kitti, FT3D, Carla]')
    parser.add_argument('--n_points', type=int, default=2048,
                        help='Point Number [default: 2048]')
    parser.add_argument('--repeat_num', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--decay_steps', type=float, default=8000, metavar='M',
                        help='SGD momentum (default: 200000)')
    parser.add_argument('--decay_rate', type=float, default=0.7, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model') 
    parser.add_argument('--dataset_path', type=str,
                        default='/home/wangsj/data/ActiveSF/record2022_0120_1056/SF2/', metavar='N',
                        help='dataset to use')
    parser.add_argument('--param_config', type=str, default='configs/config_train.yaml', metavar='N',
                        help='Dataset config file path')
    parser.add_argument('--model_dir', type=str, default='checkpoints/', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='load pretrained model for training')
    parser.add_argument('--rm_history', type=bool, default=False, metavar='N',
                        help='Whether to remove the history exp directories')
    parser.add_argument('--step', type=int, default=100, metavar='S',
                        help='the interval of tensorboard logs(default: 100)')
    parser.add_argument('--n_workers', type=int, default=1, metavar='S',
                        help='the number of worker loaders (default: 1)')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--random_dataset', action='store_true', default=False,
                        help='Whether to remove the history exp directories')
    args = parser.parse_args()

    if args.eval:
        args.rm_history = False

    # # CUDA settings

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'
    # device_ids = [0,1,2,3,4]
    device_ids = [0]
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # device_ids = [0]
    global device
    # device = torch.device("cuda:{}".format(5) if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # # CUDA settings
    if args.eval:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device_ids = [0]
    # gpus=[0,1]
    # torch.cuda.set_device('cuda:'+str(device_ids))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    # torch.autograd.set_detect_anomaly(True)

    global blue
    blue = lambda x: '\033[1;32m' + x + '\033[0m'
    global red
    red = lambda x: '\033[1;35m' + x + '\033[0m'
    global boardio

    _init_(args)
    if args.eval:
        file_dir = args.model_dir + args.exp_name + '/eval/'
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
    else:
        file_dir = args.model_dir + args.exp_name

    boardio = SummaryWriter(log_dir=file_dir)
    global textio
    textio = IOStream(file_dir + '/run.log')
    textio.cprint(str(args))
    # Path to dataset

    if args.dataset == 'HPLFlowNet':
        if args.dataset_cls == 'FT3D':
            from utils.datasets.flyingthings3d_hplflownet import FT3D
            args.dataset_path = '/dataset/public_dataset_nas/flownet3d/FlyingThings3D_subset_processed_35m/'
    elif args.dataset == 'FlowNet3D':
        if args.dataset_cls == 'Kitti':
            from utils.datasets.kitti_flownet3d import Kitti
            args.dataset_path = '/home2/wangsj/Dataset/kitti_rm_ground'
    elif args.dataset == 'Carla3D':
        from utils.datasets.carla import CARLA3D
        args.dataset_path = '/dataset/public_dataset_nas/carla_scene_flow2/'
    else:
        raise ValueError("Invalid dataset name: " + args.dataset)

    use_test = True
    mode = "test" if use_test else "val"
    assert mode == "val" or mode == "test", "Problem with mode " + mode
    if args.dataset_cls == 'FT3D':
        dataset = FT3D(root_dir=args.dataset_path, nb_points=args.n_points, mode="train")
        test_dataset = FT3D(root_dir=args.dataset_path, nb_points=args.n_points, mode=mode)
    elif args.dataset_cls == 'Kitti':
        dataset = Kitti(root_dir=args.dataset_path, nb_points=args.n_points, mode="train")
        test_dataset = Kitti(root_dir=args.dataset_path, nb_points=args.n_points, mode="val")
    elif args.dataset_cls == 'Carla3D':
        # npoints = 40000
        use_fg_inds = True
        use_hybrid_sample = True
        args.random_dataset = False
        mode = 'Full' #''Single' #Single' #   'Single' #  1
        i = 0
        dir_num = 0
        global step
        step = 45
        dir_num = 0
        global active_mode
        if i==0:
            active_mode = False
        else:
            active_mode = True
        
        str_dir_num = "%02d/"%(dir_num)
        if mode == 'Full':
            print('Total Dataset; Active mode: %s\t'%(active_mode))
        else:
            print('Dir Num: %s; Active mode: %s\t'%(str_dir_num, active_mode))
        
        use_hybrid_sample = True
        
        if args.random_dataset:
            dataset = CARLA3D(root_dir=args.dataset_path, nb_points=args.n_points, mode="train", use_fg_inds=use_fg_inds)
            test_dataset = CARLA3D(root_dir=args.dataset_path, nb_points=args.n_points, mode="test", use_fg_inds=use_fg_inds)
        else:
            # dataset = CARLA3D(root_dir=args.dataset_path+'train/', nb_points=args.n_points, mode="train", use_fg_inds=use_fg_inds)
            dataset_name = 'record2023_0430_1503' #30
            dataset = test_dataset = CARLA3D(root_dir=args.dataset_path+'active_val2/'+dataset_name, nb_points=args.n_points,
                    mode=mode, use_hybrid_sample=use_hybrid_sample, dir_num=dir_num, step=step)
            # test_dataset = CARLA3D(root_dir=args.dataset_path+'active_val3/', nb_points=args.n_points, mode="test", use_fg_inds=use_fg_inds)
            if mode == 'Full':
                if active_mode:
                    if step == 45:
                        dataset_name = 'record2023_0426_1659' #45
                        test_dataset = CARLA3D(root_dir=args.dataset_path+'active_val3/'+dataset_name, nb_points=args.n_points,
                mode=mode, use_hybrid_sample=use_hybrid_sample, dir_num=dir_num, step=step)
                else:
                    if step == 45:
                        dataset_name = 'record2023_0426_1659' #45
                        test_dataset = CARLA3D(root_dir=args.dataset_path+'val3/'+dataset_name, nb_points=args.n_points,
                mode=mode, use_hybrid_sample=use_hybrid_sample, dir_num=dir_num, step=step)

               
    else:
        raise ValueError("Invalid dataset_cls name: " + args.dataset_cls)

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=True,
        collate_fn=Batch, drop_last=True, timeout=0, persistent_workers=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True,
        collate_fn=Batch, drop_last=False, timeout=0, persistent_workers=True)
    

    if args.model == 'TFlow':
        net = TFlow(npoint = args.n_points).cuda()
        # net.apply(weights_init)

        if len(device_ids) > 1:
            net = nn.DataParallel(net, device_ids=device_ids)
        else:
            net = nn.DataParallel(net)
        print("Let's use ", len(device_ids), " GPUs!")

        if args.eval:
            if args.model_path == '':
                model_path = args.model_dir + args.exp_name + '/models/model.best.t7'
            else:
                model_path = args.model_path + '/model.best.t7'
            model_path = os.path.join(os.getcwd(), model_path)
            print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            net_dict = net.state_dict()
            pretrained_dict = torch.load(model_path)
            for k, v in pretrained_dict.items():
                name = 'module.' + k[:]  # remove `module.`
                net_dict[name] = v
            net.load_state_dict(net_dict, strict=True)
    else:
        raise Exception('Not implemented')

    if args.eval:
        test(args, net, test_loader, boardio, textio)
    else:
        if args.pretrained:
            if args.model_path == '':
                model_path = 'pretrained_model/model.best.t7'
                if len(device_ids) > 1:
                    pretrained_dict = torch.load(model_path, map_location={'cuda:0': 'cuda:0'})
                else:
                    pretrained_dict = torch.load(model_path)
            else:
                pretrained_dict = torch.load(args.model_path + 'model.best.t7')

            net_dict = net.state_dict()
            #
            for k, v in pretrained_dict.items():
                name = 'module.' + k[:]  # remove `module.`
                net_dict[name] = v
            net.load_state_dict(net_dict, strict=True)
            print('Update the neural network with the pretrained model.')
        train(args, net, train_loader, test_loader, boardio, textio)

    print('FINISH')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

