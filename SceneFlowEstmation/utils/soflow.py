import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la
from lib import pointnet2_utils as pointutils
from .utils import *
import math
# from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from torch_scatter import scatter_softmax, scatter_sum
# from gflow import InvConvdLU

LEAKY_RATE = 0.1
use_bn = False


def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointutils.grouping_operation(points_flipped.float(), knn_idx.int()).permute(0, 2, 3, 1).contiguous()

    return new_points


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn, bias=True):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x
    
    
class CrossLayerLight(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, flow_mlp, bn = use_bn, use_leaky = True):
        super(CrossLayerLight,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        # self.cross_t12 = Conv1d(in_channel, mlp1[0], bn=bn, use_leaky=use_leaky)
        # self.cross_t21 = Conv1d(in_channel, mlp1[0], bn=bn, use_leaky=use_leaky)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
        self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

        self.pos2 = nn.Conv2d(3, mlp2[0], 1)
        self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
        self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

        self.mlp2 = nn.ModuleList()
        for i in range(1, len(mlp2)):
            self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        
        last_channel = mlp2[-1]
        self.flow_mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(flow_mlp):
            self.flow_mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1).contiguous()
        xyz2 = xyz2.permute(0, 2, 1).contiguous()
        points1 = points1.permute(0, 2, 1).contiguous()
        points2 = points2.permute(0, 2, 1).contiguous()

        # knn_idx = knn_point(self.nsample, xyz2, xyz1)[0] # B, N1, nsample
        _, knn_idx = pointutils.knn(self.nsample, xyz1, xyz2)
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2, sf=None):
        # _, feat1_new = self.fe1_layer(pc1, pc2, feat1, feat2)
        # _, feat2_new = self.fe1_layer(pc2, pc1, feat2, feat1)
        # _, feat1_final = self.fe2_layer(pc1, pc2, feat1_new, feat2_new)
        # flow1 = self.flow(feat1_final)

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross_t2(feat2_new)

        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)
        
        for conv in self.flow_mlp_convs:
            feat1_final = conv(feat1_final)

        re_sf = self.fc(feat1_final)
        re_sf = re_sf.clamp(-50.0, 50.0)

        if sf is not None:
            re_sf = re_sf + sf

        return feat1_new, feat2_new, feat1_final, re_sf.clamp(-50.0, 50.0)


class PointConvTransFlowV2(nn.Module):
    def __init__(self, nsample, in_channel, sf_channel, mlp, flow_mlp, bn=use_bn, use_leaky=True, use_flow=True):
        super(PointConvTransFlowV2, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.use_flow = use_flow
        self.mlp_convs = nn.ModuleList()
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel * 2
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.mlp_convs2 = nn.ModuleList()
        if bn:
            self.mlp_bns2 = nn.ModuleList()
        last_channel = in_channel * 2
        for out_channel in mlp:
            self.mlp_convs2.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns2.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet1 = nn.Sequential(nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//2, 1, kernel_size=1))
        self.softmax = nn.Softmax(dim=2)
   
        self.mlp_convs3 = nn.ModuleList()
        if bn:
            self.mlp_bns3 = nn.ModuleList()
        last_channel = last_channel + sf_channel + 3
        for out_channel in mlp:
            self.mlp_convs3.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns3.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.mlp_convs4 = nn.ModuleList()
        if bn:
            self.mlp_bns4 = nn.ModuleList()
        last_channel = last_channel * 2 + sf_channel  + 3
        for out_channel in mlp:
            self.mlp_convs4.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns4.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        
        last_channel = out_channel
        self.flow_mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(flow_mlp):
            self.flow_mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)
    
    @staticmethod
    def calculate_corr(fmap1, fmap2):
        corr = torch.matmul(fmap1, fmap2.permute(0,1,3,2).contiguous()) # B, N1, nsample, nsmaple
        corr = corr / torch.sqrt(torch.tensor(fmap1.shape[-1]).float()) # N, K , K
        return corr[:,:,0,:]
    
    def forward(self, xyz1, xyz2, xyz2w, points1, points2, sf=None, sf_feat=None):
        """
        Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
        Return:
            new_points: upsample points feature data, [B, D', N1]
        """
        # import ipdb; ipdb.set_trace()
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1).contiguous()
        xyz2 = xyz2.permute(0, 2, 1).contiguous()
        if xyz2w != None:
            xyz2w = xyz2w.permute(0, 2, 1).contiguous()
        else:
            xyz2w = xyz2
        points1 = points1.permute(0, 2, 1).contiguous()
        points2 = points2.permute(0, 2, 1).contiguous()

        # point-to-patch Volume
        # _, knn_idx = pointutils.knn(self.nsample, xyz1, xyz2) # B, N1, nsample
        if sf != None and self.use_flow:
            sf = sf.permute(0, 2, 1).contiguous()
            _, knn_idx = pointutils.knn(self.nsample, xyz1+sf, xyz2)
        else:
            _, knn_idx = pointutils.knn(self.nsample, xyz1, xyz2)
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx)  # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)

        new_points = torch.cat([grouped_points1, grouped_points2], dim=-1)  # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1)  # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points = self.relu(bn(conv(new_points)))
            else:
                new_points = self.relu(conv(new_points))

        _, knn_idxw = pointutils.knn(self.nsample, xyz1, xyz2w)
        neighbor_xyzw = index_points_group(xyz2, knn_idxw)
        direction_xyzw = neighbor_xyzw - xyz1.view(B, N1, 1, C)
        grouped_points2w = index_points_group(points2, knn_idxw)  # B, N1, nsample, D2
        
        new_pointsw = torch.cat([grouped_points1, grouped_points2w], dim=-1)  # B, N1, nsample, D1+D2+3
        new_pointsw = new_pointsw.permute(0, 3, 2, 1)  # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs2):
            if self.bn:
                bn = self.mlp_bns2[i]
                new_pointsw = self.relu(bn(conv(new_pointsw)))
            else:
                new_pointsw = self.relu(conv(new_pointsw))

        weight_qk = torch.matmul(new_points.permute(0,3,2,1).contiguous(), new_pointsw.permute(0,3,1,2).contiguous()) # B, N1, nsample, nsmaple
        weight_qk = torch.softmax(weight_qk, -2) * torch.softmax(weight_qk, -1)

        if sf_feat != None: 
            sf_feat = sf_feat.permute(0, 2, 1).contiguous()
            grouped_sf_feats = sf_feat.view(B, N1, 1, sf_feat.shape[-1]).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1).contiguous()
            new_points_cost = torch.cat([new_points, grouped_sf_feats, direction_xyz.permute(0, 3, 2, 1).contiguous()], dim=1)  # B, N1, nsample, D1+D2+3
            new_pointsw_cost = torch.cat([new_pointsw, grouped_sf_feats, direction_xyzw.permute(0, 3, 2, 1).contiguous()], dim=1)  # B, N1, nsample, D1+D2+3
        else:
            new_points_cost = torch.cat([new_points, direction_xyz.permute(0, 3, 2, 1).contiguous()], dim=1)  # B, N1, nsample, D1+D2+3
            new_pointsw_cost = torch.cat([new_pointsw, direction_xyzw.permute(0, 3, 2, 1).contiguous()], dim=1)  # B, N1, nsample, D1+D2+3
   
        for i, conv in enumerate(self.mlp_convs3):
            if self.bn:
                bn = self.mlp_bns3[i]
                new_points_cost = self.relu(bn(conv(new_points_cost)))
            else:
                new_points_cost = self.relu(conv(new_points_cost))

        for i, conv in enumerate(self.mlp_convs3):
            if self.bn:
                bn = self.mlp_bns3[i]
                new_pointsw_cost = self.relu(bn(conv(new_pointsw_cost)))
            else:
                new_pointsw_cost = self.relu(conv(new_pointsw_cost))

        # new_points_  =  new_points + torch.matmul(weight_qk, new_pointsw.permute(0,3,2,1).contiguous()).permute(0,3,2,1).contiguous() # B N S C
        # new_pointsw_ = new_pointsw + torch.matmul(new_points.permute(0,3,1,2).contiguous(), weight_qk).permute(0,2,3,1).contiguous() # (B N C S)        
        new_points_  =  torch.matmul(weight_qk, new_pointsw.permute(0,3,2,1).contiguous()).permute(0,3,2,1).contiguous() # B N S C
        new_pointsw_ = torch.matmul(new_points.permute(0,3,1,2).contiguous(), weight_qk).permute(0,2,3,1).contiguous() # (B N C S)  
        weight_feats = self.weightnet1(new_points_)  # B C nsample N1
        weight_featsw = self.weightnet1(new_pointsw_)  # B C nsample N1

        weights1 = self.softmax(weight_feats)

        knn_idxw_flatten = knn_idxw.view(B,-1).long()
        point_to_patch_costw_flatten = new_pointsw_cost.permute(0, 3, 2, 1).reshape(B,-1, new_points_cost.shape[1]) #[B,N,C]
        weight_bwd = scatter_softmax(weight_featsw.permute(0,3,2,1).reshape(B,-1, weight_featsw.shape[1]), knn_idxw_flatten, dim=1)
        weight_bwd_cpu = weight_bwd.cpu()
        if weight_bwd_cpu[0,0,0] == None:
            print('!!!Error...')

        point_to_patch_costw_flatten = point_to_patch_costw_flatten * weight_bwd
        point_to_patch_cost_bwd = scatter_sum(point_to_patch_costw_flatten, knn_idxw_flatten, dim=1)
        point_to_patch_cost_bwd_cpu = point_to_patch_cost_bwd.cpu()
        if point_to_patch_cost_bwd_cpu[0,0,0] == None:
            print('!!!Error...')

        point_to_patch_cost_fwd = torch.sum(weights1 * new_points_cost, dim=2) # B C N

        # weights for group cost
        grouped_cost_bwd = index_points_group(point_to_patch_cost_bwd, knn_idx)  # B, N1, nsample, D2
        grouped_cost_fwd = point_to_patch_cost_fwd.view(B, N1, 1, point_to_patch_cost_fwd.shape[1]).repeat(1, 1, self.nsample, 1)
        if sf_feat != None: 
            grouped_point_to_patch_cost = torch.cat([grouped_cost_fwd, grouped_cost_bwd, grouped_sf_feats.permute(0, 3, 2, 1).contiguous(), direction_xyz], dim=-1)  # B, N1, nsample, D1+D2+3
        else:
            grouped_point_to_patch_cost = torch.cat([grouped_cost_fwd, grouped_cost_bwd, direction_xyz], dim=-1)  # B, N1, nsample, 

        grouped_point_to_patch_cost = grouped_point_to_patch_cost.permute(0, 3, 2, 1)  # [B, D1+D2+3, nsample, N1]
        
        for i, conv in enumerate(self.mlp_convs4):
            if self.bn:
                bn = self.mlp_bns4[i]
                grouped_point_to_patch_cost = self.relu(bn(conv(grouped_point_to_patch_cost)))
            else:
                grouped_point_to_patch_cost = self.relu(conv(grouped_point_to_patch_cost))
        patch_to_patch_cost = torch.max(grouped_point_to_patch_cost, dim=2)[0]

        for conv in self.flow_mlp_convs:
            patch_to_patch_cost = conv(patch_to_patch_cost)

        re_sf = self.fc(patch_to_patch_cost)
        re_sf = re_sf.clamp(-50.0, 50.0)

        if sf is not None:
            re_sf = re_sf + sf.permute(0,2,1).contiguous()

        return point_to_patch_cost_fwd, point_to_patch_cost_bwd.permute(0,2,1).contiguous(), patch_to_patch_cost, re_sf.clamp(-50.0, 50.0)


class WeightNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8], bn=True):
        super(WeightNet, self).__init__()

        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))

    def forward(self, localized_xyz):
        # xyz : BxCxKxN
        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                weights = F.relu(bn(conv(weights)))
            else:
                weights = F.relu(conv(weights))

        return weights


def group(nsample, xyz, points):
    """
    Input:
        nsample: scalar
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    _, idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points_group(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points_group(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm


class PointConv2(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet=16, bn=use_bn, use_leaky=True):
        super(PointConv2, self).__init__()
        self.bn = bn
        self.nsample = nsample
        # self.weightnet = WeightNet(3, 1)
        # self.softmax = nn.Softmax(dim=1)
        # self.linear = nn.Linear(in_channel, out_channel)
        self.linear = nn.Conv2d(in_channel, out_channel, 1)
        if bn:
            self.bn_linear = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv without strides size, i.e., the input and output have the same number of points.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B, C, N = xyz.shape

        xyz_t = xyz.permute(0, 2, 1).contiguous()
        # points_t = points.permute(0, 2, 1).contiguous()

        # new_points, grouped_xyz_norm = group(self.nsample, xyz_t, points_t)
        # grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        _, idx = pointutils.knn(self.nsample, xyz_t, xyz_t)
        grouped_xyz = pointutils.grouping_operation(xyz, idx)
        grouped_points = pointutils.grouping_operation(points, idx)
        grouped_xyz_norm = grouped_xyz - xyz.view(B, C, N, 1)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=1)

        # weights = self.softmax(self.weightnet(grouped_xyz_norm))
        # # new_points = torch.matmul(input=new_points.permute(0, 2, 1, 3), other=weights.permute(0, 2, 3, 1)).view(B,N,-1)
        # new_points = torch.matmul(input=new_points.permute(0, 2, 1, 3), other=weights.permute(0, 2, 3, 1)).squeeze(-1).contiguous()
        new_points = self.linear(new_points)
        if self.bn:
            # new_points = self.bn_linear(new_points.permute(0, 2, 1))
            new_points = self.bn_linear(new_points)
        # else:
        #     # new_points = new_points.permute(0, 2, 1)
        #     new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)
        new_points = torch.max(new_points, dim=-1)[0]

        return new_points

class PointWarping(nn.Module):
    def forward(self, pos1, pos2, flow1=None, nsample=None):
        if flow1 is None:
            return pos2

        # move pos1 to pos2'
        pos1_to_2 = pos1 + flow1

        # interpolate flow
        B, C, N1 = pos1.shape
        _, _, N2 = pos2.shape
        pos1_to_2_t = pos1_to_2.permute(0, 2, 1).contiguous()  # B 3 N1
        pos2_t = pos2.permute(0, 2, 1).contiguous()  # B 3 N2
        # flow1_t = flow1.permute(0, 2, 1).contiguous()
        if nsample is None:
            nsample = 3
            _, knn_idx = pointutils.three_nn(pos2_t, pos1_to_2_t)
        else:
            _, knn_idx = pointutils.knn(nsample, pos2_t, pos1_to_2_t)
        grouped_pos_norm = pointutils.grouping_operation(pos1_to_2, knn_idx) - pos2.view(B, C, N2, 1)
        dist = torch.norm(grouped_pos_norm, dim=1).clamp(min=1e-10)
        norm = torch.sum(1.0 / dist, dim=2, keepdim=True)
        weight = (1.0 / dist) / norm

        grouped_flow1 = pointutils.grouping_operation(flow1, knn_idx)
        flow2 = torch.sum(weight.view(B, 1, N2, nsample) * grouped_flow1, dim=-1)
        warped_pos2 = pos2 - flow2  # B 3 N2

        return warped_pos2.clamp(-10.0, 10.0)


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x


class SceneFlowEstimatorPointConv(nn.Module):
    def __init__(self, feat_ch, cost_ch, flow_ch=3, channels=[128, 128], mlp=[128, 64], neighbors=9, clamp=[-20, 20],
                 use_leaky=True):
        super(SceneFlowEstimatorPointConv, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch + flow_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv2(neighbors, last_channel + 3, ch_out, bn=True, use_leaky=True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out

        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume, flow=None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        if flow is None:
            new_points = torch.cat([feats, cost_volume], dim=1)
        else:
            new_points = torch.cat([feats, cost_volume, flow], dim=1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow = self.fc(new_points)
        return new_points, flow.clamp(self.clamp[0], self.clamp[1])


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True, if_IN=False, IN_affine=False,
         if_BN=False):
    if isReLU:
        if if_IN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.InstanceNorm2d(out_planes, affine=IN_affine)
            )
        elif if_BN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(out_planes, affine=IN_affine)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
            )
    else:
        if if_IN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.InstanceNorm2d(out_planes, affine=IN_affine)
            )
        elif if_BN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.BatchNorm2d(out_planes, affine=IN_affine)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True)
            )


class UpsampleFlow(nn.Module):
    def forward(self, xyz, sparse_xyz, sparse_flow, k=3):
        '''
        :param xyz: [B,C,N]
        :param sparse_xyz: [B,C,N]
        :param sparse_flow: [B,C,N]
        :return: [B,C,N]
        '''
        # import ipdb; ipdb.set_trace()
        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz_t = xyz.permute(0, 2, 1).contiguous()  # B N 3
        sparse_xyz_t = sparse_xyz.permute(0, 2, 1).contiguous()  # B S 3
        # sparse_flow_t = sparse_flow.permute(0, 2, 1).contiguous() # B S 3
        # knn_idx = knn_point(3, sparse_xyz, xyz)
        if k ==3:
            _, knn_idx = pointutils.three_nn(xyz_t, sparse_xyz_t)
        else:
            _, knn_idx = pointutils.knn(k, xyz_t, sparse_xyz_t)
        grouped_xyz_norm = pointutils.grouping_operation(sparse_xyz, knn_idx) - xyz.view(B, C, N, 1)
        dist = torch.norm(grouped_xyz_norm, dim=1).clamp(min=1e-10)
        norm = torch.sum(1.0 / dist, dim=-1, keepdim=True)
        weight = (1.0 / dist) / norm
        # vv = torch.max(weight)
        # if vv > 1e4:
        #     print(torch.max(weight))

        grouped_flow = pointutils.grouping_operation(sparse_flow.float(), knn_idx)
        dense_flow = torch.sum(weight.view(B, 1, N, k) * grouped_flow, dim=-1)  # [B,C,N]
        # vv = torch.max(dense_flow)
        # if vv > 1e4:
        #     print(torch.max(dense_flow))
        return dense_flow.clamp(-100.0, 100.0)


class PointNetSetUpConv(nn.Module):
    def __init__(self, nsample, radius, f1_channel, f2_channel, mlp, mlp2, knn=True):
        super(PointNetSetUpConv, self).__init__()
        self.nsample = nsample
        self.radius = radius
        self.knn = knn
        self.mlp1_convs = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        last_channel = f2_channel + 3
        for out_channel in mlp:
            self.mlp1_convs.append(nn.Sequential(nn.Conv2d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm2d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel
        if len(mlp) != 0:
            last_channel = mlp[-1] + f1_channel
        else:
            last_channel = last_channel + f1_channel
        for out_channel in mlp2:
            self.mlp2_convs.append(nn.Sequential(nn.Conv1d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm1d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel

    def forward(self, pos1, pos2, feature1, feature2):
        """
            Feature propagation from xyz2 (less points) to xyz1 (more points)
        Inputs:
            xyz1: (batch_size, 3, npoint1)
            xyz2: (batch_size, 3, npoint2)
            feat1: (batch_size, channel1, npoint1) features for xyz1 points (earlier layers, more points)
            feat2: (batch_size, channel1, npoint2) features for xyz2 points
        Output:
            feat1_new: (batch_size, npoint2, mlp[-1] or mlp2[-1] or channel1+3)

            TODO: Add support for skip links. Study how delta(XYZ) plays a role in feature updating.
        """
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()
        B, C, N = pos1.shape
        if self.knn:
            _, idx = pointutils.knn(self.nsample, pos1_t, pos2_t)
        else:
            idx, _ = query_ball_point(self.radius, self.nsample, pos2_t, pos1_t)

        pos2_grouped = pointutils.grouping_operation(pos2, idx)
        pos_diff = pos2_grouped - pos1.view(B, -1, N, 1)  # [B,3,N1,S]

        feat2_grouped = pointutils.grouping_operation(feature2, idx)
        feat_new = torch.cat([feat2_grouped, pos_diff], dim=1)  # [B,C1+3,N1,S]
        for conv in self.mlp1_convs:
            feat_new = conv(feat_new)
        # max pooling
        feat_new = feat_new.max(-1)[0]  # [B,mlp1[-1],N1]
        # concatenate feature in early layer
        if feature1 is not None:
            feat_new = torch.cat([feat_new, feature1], dim=1)
        # feat_new = feat_new.view(B,-1,N,1)
        for conv in self.mlp2_convs:
            feat_new = conv(feat_new)

        return feat_new


class PointConv1DComposed(nn.Module):
    def __init__(self, nsample, in_channel, mlp, knn=True, bn=use_bn, use_leaky=True):
        super(PointConv1DComposed, self).__init__()
        self.in_channels = in_channel
        self.nsample = nsample
        self.knn = knn

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel  # TODO：
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        # relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, pos):
        """
                PointConv without strides size, i.e., the input and output have the same number of points.
                Input:
                    pos: input points position data, [B, C, N]
                Return:
                    new_pos: sampled points position data, [B, C, S]
                    new_points_concat: sample points feature data, [B, D', S]
                """
        B, C, N = pos.shape
        pos_t = pos.permute(0, 2, 1).contiguous()

        if self.knn:
            _, idx = pointutils.knn(self.nsample, pos_t, pos_t)
        else:
            idx, _ = query_ball_point(self.radius, self.nsample, pos_t, pos_t)

        new_points = pointutils.grouping_operation(pos, idx)  # [B,3,N,K]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = bn(conv(new_points))

        new_points = torch.max(new_points, -1)[0]
        return new_points


if __name__ == "__main__":
    import os
    import sys
    import argparse

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--n_points', type=int, default=2048)
    args = parser.parse_args()

    sparse_input = torch.randn(5, 3, 512).cuda()
    input = torch.randn(5, 3, 2048).cuda()
    f_input = torch.randn(5, 128, 2048).cuda()
    # label = torch.randn(4, 16)
    # (self, nsample, radius, in_channel, afn_mlp, fe_mlp, knn=True)
    model = SGUSceneFlowEstimatorMini(nsample=5, in_channel=128, flow_mlp=[128, 64, 64], ctx_mlp=[32, 16, 16]).cuda()
    # model = PointConvMiniSqueeze(nsample=10,in_channel=3, mlp=[16, 16], knn=True, bn=True, use_leaky=True).cuda()
    total_num_paras = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is %d\t" % (total_num_paras))
    device_ids = [0]
    if len(device_ids) == 0:
        net = nn.DataParallel(model)
    else:
        net = nn.DataParallel(model, device_ids=device_ids)
    print("Let's use ", len(device_ids), " GPUs!")
    # forward(self, pos1, pos2, feats1, feats2, sparse_pos1, sparse_flow):
    output = model(input, input, f_input, f_input, sparse_input, sparse_input)
    # output = model(sparse_input)
    print(output.shape)