import numpy as np
# import matplotlib.pyplot as plt
import time

# from utils import odom_utils, flow_vis
from sklearn.neighbors import NearestNeighbors
from enum import IntEnum
# from apex import amp
from torch.cuda.amp import autocast as autocast

import torch
import torch.nn as nn


from utils.utils import index_points, PointNetSetAbstraction, PointNetSetUpConv
from utils.soflow import  PointConvTransFlowV2, PointWarping, UpsampleFlow

LEAKY_RATE = 0.1
use_bn = False

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bn),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x

class RefineFlowRegressor(nn.Module):
    def __init__(self, nsample=8, in_channel=128, feat_channel=128, mlp=[128, 128, 128], flow_mlp=[128, 128], use_flow=True):
        super(RefineFlowRegressor, self).__init__()
        self.use_flow = use_flow
        self.nsample = nsample
        self.cost = PointConvTransFlowV2(nsample, in_channel, feat_channel, mlp, flow_mlp, use_flow=use_flow)
        # warping
        self.warping = PointWarping()
     

    def forward(self, pc1, pc2, feats1, feats2, wraping_num=5, c_flow=None, flow_feats=None):
        if c_flow==None:
            pc2_warp = None
        else:
            pc2_warp = self.warping(pc1, pc2, c_flow, wraping_num)
        
        if self.use_flow:
            cost_fwd, cost_bwd, flow_feats, c_flow = self.cost(pc1, pc2, pc2_warp, feats1, feats2, c_flow, flow_feats)
        else:
            cost_fwd, cost_bwd, flow_feats, c_flow = self.cost(pc1, pc2, pc2_warp, feats1, feats2)
  

        return cost_fwd, cost_bwd, flow_feats, c_flow

class TFlow(nn.Module):
    def __init__(self, npoint=8192):
        super(TFlow, self).__init__()
        self.point_conv = nn.Sequential(Conv1d(3,32), Conv1d(32,32))

        self.sa1 = PointNetSetAbstraction(npoint=2048, radius=0.5, nsample=16, in_channel=32, mlp=[32,32,64],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=512, radius=2.0, nsample=16, in_channel=64, mlp=[64,64,128],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=256, radius=4.0, nsample=16, in_channel=128, mlp=[128,128,256],
                                          group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=128, radius=8.0, nsample=8, in_channel=256, mlp=[256,256,512],
                                          group_all=False)

        self.su3 = PointNetSetUpConv(nsample=16, radius=2.4, f1_channel=256, f2_channel=512, mlp=[256,256], mlp2=[256,256])
        self.flow3_r = RefineFlowRegressor(nsample=16, in_channel=256, feat_channel=0, mlp=[256,256], flow_mlp=[128,128], use_flow=False)

        self.su2 = PointNetSetUpConv(nsample=16, radius=2.4, f1_channel=128, f2_channel=256, mlp=[128,128], mlp2=[128,128])
        self.flow2_r = RefineFlowRegressor(nsample=16, in_channel=128+64, feat_channel=128, mlp=[128,128], flow_mlp=[128,128], use_flow=True)

        self.su1 = PointNetSetUpConv(nsample=16, radius=2.4, f1_channel=64, f2_channel=128, mlp=[64,64], mlp2=[64,64])
        self.flow1_r = RefineFlowRegressor(nsample=16, in_channel=64+32, feat_channel=128, mlp=[64,64], flow_mlp=[64,64], use_flow=True)
        
        self.su0 = PointNetSetUpConv(nsample=16, radius=2.4, f1_channel=32, f2_channel=64, mlp=[64,64], mlp2=[64,64])
        self.flow0_r = RefineFlowRegressor(nsample=16, in_channel=64+32, feat_channel=64, mlp=[64,64], flow_mlp=[64,64], use_flow=True)

        self.flow_up_sample = UpsampleFlow()
        self.deconv3_2 = Conv1d(256, 64)
        self.deconv2_1 = Conv1d(128, 32)
        self.deconv1_0 = Conv1d(64, 32)
        self.total_time = 0.0
        # self.iter_num=5

    # @autocast()
    def forward(self, pc1, pc2, feats1=None, feats2=None):
        '''
        pc1, pc2: [B,C,N]
        coarse_sf: [B,C,N]
        '''
        # time_start = time.time()

        if feats1 == None or feats2 == None:
            feats1 = self.point_conv(pc1)
            feats2 = self.point_conv(pc2)
        else:
            feats1 = self.point_conv(feats1)
            feats2 = self.point_conv(feats2)

        l1_pc1, l1_feats1, l1_fps_inds1 = self.sa1(pc1, feats1)
        l1_pc2, l1_feats2, _ = self.sa1(pc2, feats2)

        l2_pc1, l2_feats1, l2_fps_inds1 = self.sa2(l1_pc1, l1_feats1)
        l2_pc2, l2_feats2, _ = self.sa2(l1_pc2, l1_feats2)

        l3_pc1, l3_feats1, l3_fps_inds1 = self.sa3(l2_pc1, l2_feats1)
        l3_pc2, l3_feats2, _ = self.sa3(l2_pc2, l2_feats2)

        l4_pc1, l4_feats1, l4_fps_inds1 = self.sa4(l3_pc1, l3_feats1)
        l4_pc2, l4_feats2, _ = self.sa4(l3_pc2, l3_feats2)

        l3_4_feats1 = self.su3(l3_pc1, l4_pc1, l3_feats1, l4_feats1)
        l3_4_feats2 = self.su3(l3_pc2, l4_pc2, l3_feats2, l4_feats2)
        # l3_4_feats_sf = torch.cat([l3_4_feats_sf, l3_4_feats1], dim=1)

        c_feat_fwd_l3, c_feat_bwd_l3, l3_feats, l3_flow = self.flow3_r(l3_pc1, l3_pc2, l3_4_feats1, l3_4_feats2, 3)
        
        l2_3_feats1 = self.su2(l2_pc1, l3_pc1, l2_feats1, l3_4_feats1)
        l2_3_feats2 = self.su2(l2_pc2, l3_pc2, l2_feats2, l3_4_feats2)
        
        l2_coarse_sf = self.flow_up_sample(l2_pc1, l3_pc1, l3_flow, k=5)
        l2_3_feats_sf = self.flow_up_sample(l2_pc1,l3_pc1, l3_feats, k=5)
        # l2_3_feats_sf = torch.cat([l2_3_feats_sf, l2_3_feats1], dim=1)

        c_feat_fwd_l3_2 = self.flow_up_sample(l2_pc1, l3_pc1, c_feat_fwd_l3)
        c_feat_fwd_l3_2 = self.deconv3_2(c_feat_fwd_l3_2)
        c_feat_bwd_l3_2 = self.flow_up_sample(l2_pc1, l3_pc1, c_feat_bwd_l3)
        c_feat_bwd_l3_2 = self.deconv3_2(c_feat_bwd_l3_2)
        c_feat_fwd_l3_2 = torch.cat([l2_3_feats1, c_feat_fwd_l3_2], dim=1)
        c_feat_bwd_l3_2 = torch.cat([l2_3_feats2, c_feat_bwd_l3_2], dim=1)  

        c_feat_fwd_l2, c_feat_bwd_l2, l2_feats_sf, l2_flow = self.flow2_r(l2_pc1, l2_pc2, c_feat_fwd_l3_2, c_feat_bwd_l3_2, 5, l2_coarse_sf, l2_3_feats_sf)

        l1_2_feats1 = self.su1(l1_pc1, l2_pc1, l1_feats1, l2_3_feats1)
        l1_2_feats2 = self.su1(l1_pc2, l2_pc2, l1_feats2, l2_3_feats2)
        
        l1_coarse_sf = self.flow_up_sample(l1_pc1, l2_pc1, l2_flow, k=5)
        l1_2_feats_sf = self.flow_up_sample(l1_pc1, l2_pc1, l2_feats_sf, k=5)
        # l1_2_feats_sf = torch.cat([l1_2_feats_sf, l1_2_feats1], dim=1)

        c_feat_fwd_l2_1 = self.flow_up_sample(l1_pc1, l2_pc1, c_feat_fwd_l2)
        c_feat_fwd_l2_1 = self.deconv2_1(c_feat_fwd_l2_1)
        c_feat_bwd_l2_1 = self.flow_up_sample(l1_pc1, l2_pc1, c_feat_bwd_l2)
        c_feat_bwd_l2_1 = self.deconv2_1(c_feat_bwd_l2_1)
        c_feat_fwd_l2_1 = torch.cat([l1_2_feats1, c_feat_fwd_l2_1], dim=1)
        c_feat_bwd_l2_1 = torch.cat([l1_2_feats2, c_feat_bwd_l2_1], dim=1)

        c_feat_fwd_l1, c_feat_bwd_l1, l1_feats_sf, l1_flow = self.flow1_r(l1_pc1, l1_pc2, c_feat_fwd_l2_1, c_feat_bwd_l2_1, 7, l1_coarse_sf, l1_2_feats_sf)

        # flow = self.flow_up_sample(pc1, l1_pc1, l1_flow)
        
        l0_1_feats1 = self.su0(pc1, l1_pc1, feats1, l1_2_feats1)
        l0_1_feats2 = self.su0(pc2, l1_pc2, feats2, l1_2_feats2)
        l0_1_feats_sf = self.flow_up_sample(pc1, l1_pc1, l1_feats_sf, k=7)
        l0_coarse_sf = self.flow_up_sample(pc1, l1_pc1, l1_flow, k=7)

        c_feat_fwd_l2_1 = self.flow_up_sample(pc1, l1_pc1, c_feat_fwd_l1)
        c_feat_fwd_l2_1 = self.deconv1_0(c_feat_fwd_l2_1)
        c_feat_bwd_l2_1 = self.flow_up_sample(pc1, l1_pc1, c_feat_bwd_l1)
        c_feat_bwd_l2_1 = self.deconv1_0(c_feat_bwd_l2_1)
        c_feat_fwd_l2_1 = torch.cat([l0_1_feats1, c_feat_fwd_l2_1], dim=1)
        c_feat_bwd_l2_1 = torch.cat([l0_1_feats2, c_feat_bwd_l2_1], dim=1)

        _, _, _, flow = self.flow0_r(pc1, pc2, c_feat_fwd_l2_1, c_feat_bwd_l2_1, 7, l0_coarse_sf, l0_1_feats_sf)
            
        # time_end = time.time()
        # self.total_time = self.total_time + time_end - time_start
        # print('Total Time Consuming is %.4f, '%(self.total_time))

        flows = [flow, l1_flow, l2_flow, l3_flow]
        fps_inds = [l1_fps_inds1, l2_fps_inds1, l3_fps_inds1]


        return flows, fps_inds


def multiScaleLoss(pred_flows, gt_flow, mask, fps_idxs, alpha = [0.02, 0.04, 0.08, 0.16, 0.24]):
    #num of scale
    scale = 1.0
    num_scale = len(pred_flows)
    offset = len(fps_idxs) - num_scale + 1
    # mask = mask.repeat(1,1,3)
    #generate GT list and mask1s
    gt_flows = [gt_flow.permute(0,2,1).contiguous()]
    gt_masks = [mask]
    for i in range(1, len(fps_idxs) + 1):
        fps_idx = fps_idxs[i - 1]
        sub_gt_flow = index_points(gt_flows[-1], fps_idx.long()) / scale
        sub_mask = index_points(gt_masks[-1], fps_idx.long())
        gt_flows.append(sub_gt_flow)
        gt_masks.append(sub_mask)

    total_loss = torch.zeros(1).cuda()
    for i in range(num_scale):
        diff_flow = (pred_flows[i].permute(0, 2, 1) - gt_flows[i]) * gt_masks[i]
        total_loss += alpha[i] * torch.norm(diff_flow, dim=2).sum(dim=1).mean()

    return total_loss


if __name__ == "__main__":
    import os
    import sys
    import argparse

    os.environ["CUDA_VISIBLE_DEVICES"] = '6'
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--n_points', type=int, default=2048)
    args = parser.parse_args()

    sparse_input = torch.randn(2, 3, 512).cuda()
    input = torch.randn(2, 3, 2048).cuda()
    f_input = torch.randn(2, 128, 2048).cuda()

    model = TFlow(npoint=2048)
    total_num_paras = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is %d\t" % (total_num_paras))
    device_ids = [0]
    if len(device_ids) == 0:
        net = nn.DataParallel(model)
    else:
        net = nn.DataParallel(model, device_ids=device_ids)
    print("Let's use ", len(device_ids), " GPUs!")

    output = model(input, input)
    print(output[0][0].shape)
