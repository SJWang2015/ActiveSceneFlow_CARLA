import numpy as np
import matplotlib.pyplot as plt
import time

from utils import odom_utils, flow_vis
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
from torch.autograd import grad
from geomloss import SamplesLoss
from pykeops.torch.cluster import grid_cluster, cluster_ranges_centroids

from utils.utils import index_points, PointNetSetAbstraction, FlowEmbedding, PointNetFeaturePropogation
from utils.soflow import PointConvMiniSqueeze, PointConv, SGUSceneFlowEstimator, UpsampleFlow, PointNetSetUpConv#, PointNetSetAbstraction
from utils.hnfflow import RealNVPBijector
from lib import pointnet2_utils as pointutils

import cv2

# from AEMFlow import calc_coarseflow

import open3d as o3d
from open3d import *

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

side_range = [-120, 120]
fwd_range = [-120, 120]
RESELUTION = 0.3
min_height = -3.4
max_height = 0.01


def ComputeRoadWidthAndExtractIoUPts(pc1, np_pc1, ypr, vis_show=False):
    '''
    Extract the road width, centers, and boundraies.
    '''
    # drivable_areas = odom_utils.readPointCloud('./dataset/town02-map.bin') #road map
    drivable_areas = odom_utils.readPointCloud('./dataset/town02-map.bin') #road map
    # drivable_area = drivable_areas[:,:3]
    drivable_areas[:,-2] = min_height
    drivable_areas[:,:3] = (drivable_areas[:,:3] - ypr[:3])\
                    @ odom_utils.rotation_from_euler_zyx(-ypr[5], -ypr[4],ypr[3])#
    # roads_c = (roads_c - ypr[:3])\
    #                 @ odom_utils.rotation_from_euler_zyx(-ypr[5], -ypr[4],ypr[3])#
    drivable_area = (torch.tensor(drivable_areas[:,:3], dtype=torch.float32)).view(1, -1, 3).contiguous().cuda()
    road_id = drivable_areas[:,-1]
    road_id_set = np.unique(np.array(road_id, dtype=np.int32))
    road_inds = [[] for _ in range(road_id_set.shape[0])]
    for i, iCnt in enumerate(road_id):
        road_inds[int(iCnt)] += [[int(i)]]

    #指定实验道路
    # iou_id = np.argwhere(road_id == 1)

    drivable_centers = np.load('./dataset/np_c.npy')#odom_utils.readPointCloud('./dataset/town02-road-center.bin')
    drivable_center = drivable_centers[:,:3]
    drivable_center[:,1] *= -1.0
    drivable_center[:,-1] = min_height
    drivable_center[:,:3] = (drivable_center[:,:3] - ypr[:3])\
                     @ odom_utils.rotation_from_euler_zyx(-ypr[5], -ypr[4],ypr[3])#
    drivable_center_torch = (torch.tensor(drivable_center, dtype=torch.float32)).view(1, -1, 3).contiguous().cuda()
    
    center_id = drivable_centers[:,-1]
    cenetr_id_set = np.unique(np.array(center_id, dtype=np.int32))
    center_inds = [[] for _ in range(cenetr_id_set.shape[0])]
    for i, iCnt in enumerate(center_id):
        center_inds[int(iCnt)] += [[int(i)]]
    # road_id2 = []
    # road_id2 = drivable_centers[:,-1]
    # ref_drivable_center_inds = np.argwhere(drivable_centers[:,-1] == 1)
    
    dists, _ = pointutils.knn(1, drivable_area, drivable_center_torch)
    road_infos = np.zeros([road_id_set.shape[0], 2])
    np_tmp_dists = dists.squeeze().cpu().numpy()
    # np_idx = idx.squeeze(0).cpu().numpy()
    np_dists = np.zeros((drivable_center.shape[0], 2))
    np_dists[:, 0] = drivable_centers[:, -1]
    for i in road_id_set:
        # sub_road_inds = np.array(road_inds[i])
        sub_road_inds = road_inds[i]
        sub_dists = np_tmp_dists[sub_road_inds]
        road_infos[i,:] = np.array([[road_id_set[i], np.round(np.min(sub_dists))]])
        np_dists[center_inds[i], -1] = road_infos[i,-1]
        # print(np.min(sub_dists))
    
    # Extract the foreground points
    # pc1_t = pc1.transpose(1,2).contiguous().cuda()
    pc1[:,:,-1] = torch.mean(drivable_center_torch[:,:,-1])
    dists0, idx0 = pointutils.knn(1, pc1, drivable_center_torch)
    np_dists0 = dists0.squeeze(0).cpu().numpy()
    np_idx0 = idx0.squeeze(0).cpu().numpy().reshape(-1).tolist()
    foreground = np.logical_not((np_dists0[:,0] - np_dists[np_idx0,-1]) > -0.10)
    foreground = np.logical_and(foreground, np_pc1[:,-1] <= 0.1)
    # np_pc2 = np_pc1[foreground]
    background = np.logical_not((np_dists0[:,0] - np_dists[np_idx0,-1]) <= -0.10)
    background = np.logical_or(background, np_pc1[:,-1] > 0.1)
    # np_pc3 = np_pc1[background]
    
    pc_inds = np.array(range(0, pc1.shape[1]))
    f_inds = np.array(pc_inds)[foreground]
    b_inds = np.array(pc_inds)[background]
    np_pc2 = np_pc1[f_inds,:]
    np_pc3 = np_pc1[b_inds,:]
    # f_inds = torch.tensor(f_inds, dtype=torch.int32).cuda()
    # b_inds = torch.tensor(b_inds, dtype=torch.int32).cuda()

    if vis_show:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_pc3)
        pcd.paint_uniform_color([0,1,0])
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(np_pc2)
        pcd2.paint_uniform_color([1,0,0])
        pcd3 = o3d.geometry.PointCloud()
        pcd3.points = o3d.utility.Vector3dVector(drivable_areas[:,:3])
        pcd3.paint_uniform_color([0,0,1])
        pcd4 = o3d.geometry.PointCloud()
        pcd4.points = o3d.utility.Vector3dVector(drivable_centers[:,:3])
        pcd4.paint_uniform_color([0,1,1])
        vis_list = [pcd, pcd2]
        vis_list += [pcd3, pcd4]
        o3d.visualization.draw_geometries(vis_list)

    return f_inds, b_inds, np_pc2


def calc_coarse_flow_from_bev(raw_pc_list,
                          side_range=side_range,
                          fwd_range=fwd_range,
                          res=RESELUTION,
                          min_height = min_height,
                          max_height = max_height,
                          use_Transform = False,
                          use_fg_inds = True,
                          saveto=None):
    """ Creates an 2D birds eye view representation of the point cloud data.
        You can optionally save the image to specified filename.
    """
    B = raw_pc_list[0].shape[0]

    if len(raw_pc_list) == 5:
        raw_fgrnd_inds_s = raw_pc_list[3].cpu().numpy().astype(bool)
        raw_fgrnd_inds_t = raw_pc_list[4].cpu().numpy().astype(bool)
    raw_pc1 = raw_pc_list[0].cpu().numpy()
    raw_flow = raw_pc_list[1].cpu().numpy()
    raw_pc2 = raw_pc_list[2].cpu().numpy()
    
    # pc_list = [pc1, pc2]
    new_sf_set = []
    for iB in range(B):
        if use_fg_inds:
            fgrnd_inds_s = raw_fgrnd_inds_s[iB,:]
            fgrnd_inds_t = raw_fgrnd_inds_t[iB,:]
            pc1 = raw_pc1[iB,:, fgrnd_inds_s]
            flow = raw_flow[iB,:, fgrnd_inds_s]
            pc2 = raw_pc2[iB,:, fgrnd_inds_t]
        else:
            pc1 = (raw_pc1[iB,:, :]).T
            flow = (raw_flow[iB,:, :]).T
            pc2 = (raw_pc2[iB,:, :]).T

        pc_list = [pc1, pc1+flow, pc2]
    
        points_ = np.concatenate(tuple(pc_list), axis=0)

        imgs = []
        step = pc_list[0].shape[0]
        sub_imgs = []
        for i in range(len(pc_list)):
            # if i < (len(pc_list)-1):
            mask = np.zeros_like(points_)
            mask[i*step:(i+1)*step, :] = 1.0
            points = points_ * mask
            # else:
            #     points = pc_list[i]
            x_lidar = points[:, 0]
            y_lidar = points[:, 1]
        
            ff = np.logical_and((x_lidar > fwd_range[0]), (x_lidar < fwd_range[1]))
            ss = np.logical_and((y_lidar > -side_range[1]), (y_lidar < -side_range[0]))
            indices = np.argwhere(np.logical_and(ff,ss)).flatten()

            # for iB in range(B):
                # CONVERT TO PIXEL POSITION VALUES - Based on resolution
            x_img = (-y_lidar[indices]/res).astype(np.int32) # x axis is -y in LIDAR
            y_img = (x_lidar[indices]/res).astype(np.int32)  # y axis is -x in LIDAR

            # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
            # floor used to prevent issues with -ve vals rounding upwards
            x_img -= int(np.floor(side_range[0]/res))
            y_img -= int(np.floor(fwd_range[0]/res))

            pixel_values = np.clip(a = 255, a_min=0, a_max=255).astype(np.uint8)
            
            # FILL PIXEL VALUES IN IMAGE ARRAY
            x_max = int((side_range[1] - side_range[0])/res + 0.5 + 1)
            y_max = int((fwd_range[1] - fwd_range[0])/res + 0.5 + 1)
            im = np.zeros([y_max, x_max], dtype=np.uint8)
            im[-y_img, x_img] = pixel_values # -y because images start from top left
            sub_imgs.append(im)
        imgs.append(sub_imgs)

        # seed_rpns = [seed_center[seed_index, :], seed_areas[seed_index, :4], seed_index]
        ############
        # pred_pc2 = pc1 + flow
        pred_pc2 = pc1
        query_bbx_list = []
        seed_bbx_list = []
        
        # for iB in range(B):
        seed_rpns, query_rpns = RPN_torch([imgs[0][0], imgs[0][2]])
        query_areas = query_rpns[1]
        seed_areas = seed_rpns[1]
        new_sf = np.zeros_like(pc1)
        new_sf_flag = np.ones_like(pc1)
        for i in range(query_rpns[2].shape[0]):
            x_rb = query_areas[i,0] + query_areas[i,2]
            y_rb = query_areas[i,1] + query_areas[i,3]
            x_lt, y_lt = bev_to_lidar_coords(query_areas[i,0], query_areas[i,1])
            x_rb, y_rb = bev_to_lidar_coords(x_rb, y_rb)
            max_min_bnd = np.array([[x_rb, y_rb, min_height], [x_lt, y_lt, 0]])
            query_bbx_list.append(max_min_bnd)

            valid_query_x_flag  = np.logical_and(x_rb<pc2[:, 0], x_lt>pc2[:, 0])
            valid_query_y_flag  = np.logical_and(y_rb<pc2[:, 1], y_lt>pc2[:, 1])
            valid_query_xy_flag = np.logical_and(valid_query_x_flag, valid_query_y_flag)

            x_rb = seed_areas[i,0] + seed_areas[i,2]
            y_rb = seed_areas[i,1] + seed_areas[i,3]
            x_lt, y_lt = bev_to_lidar_coords(seed_areas[i,0], seed_areas[i,1])
            x_rb, y_rb = bev_to_lidar_coords(x_rb, y_rb)

            valid_seed_x_flag  = np.logical_and(x_rb<pred_pc2[:, 0], x_lt>pred_pc2[:, 0])
            valid_seed_y_flag  = np.logical_and(y_rb<pred_pc2[:, 1], y_lt>pred_pc2[:, 1])
            valid_seed_xy_flag = np.logical_and(valid_seed_x_flag, valid_seed_y_flag)
            new_sf_flag[valid_seed_xy_flag, :] = 0.0
            if (np.sum(valid_seed_xy_flag) > 0) and (np.sum(valid_query_xy_flag) > 0):
                mean_pc2 = np.mean(pc2[valid_query_xy_flag, :], axis=0)
                mean_pc1 = np.mean(pc1[valid_seed_xy_flag, :], axis=0)
                blob_sf = (mean_pc2 - mean_pc1).reshape(1,3)
                blob_sf[0,2] = 0.0
                if np.linalg.norm(blob_sf) > 10.0:
                    continue
                cnt = np.sum(valid_seed_xy_flag)
                # aa = np.repeat(blob_sf, repeats=cnt, axis=0)
                # bb = new_sf[:, valid_seed_xy_flag]
                new_sf[valid_seed_xy_flag, :] = np.repeat(blob_sf, repeats=cnt, axis=0)

                max_min_bnd = np.array([[x_rb, y_rb, min_height], [x_lt, y_lt, 0]])
                seed_bbx_list.append(max_min_bnd)
            # print("i:{}, x1: {}, y1: {}".format(i, x_lt,y_lt))
            # print("i:{}, x1: {}, y1: {}, x2: {}, y2: {}".format(i, x_lt,y_lt,x_rb,y_rb))
            # new_sf_set.append(new_sf)
        if use_fg_inds:
            raw_flow[iB, :, fgrnd_inds_s] = new_sf
        else:
            raw_flow[iB, :, :] = new_sf.T
        # new_sf += flow * new_sf_flag
        # pred_pc2 = pc1 + new_sf
        # return seed_bbx_list, query_bbx_list, pred_pc2
    return torch.from_numpy(raw_flow).cuda().contiguous()


def bev_to_lidar_coords(xx,yy, side_range=(-120, 120),fwd_range=(-120,120), RES=RESELUTION ):
    X0, Xn = 0, int((side_range[1]-side_range[0])//RES)+1
    Y0, Yn = 0, int((fwd_range[1]-fwd_range[0])//RES)+1
    y = Xn*RES-(xx+0.5)*RES + side_range[0]
    x = Yn*RES-(yy+0.5)*RES + fwd_range[0]
    return x,y

def RPN_torch(img_gray_list, kernel_size=3):
    '''
    Region Proposal Generator = max repsonse locations + detection_edges
    '''
    seed_img = img_gray_list[0]
    query_img = img_gray_list[1]
    # seed_img = cv2.medianBlur(seed_img, 3)
    # query_img = cv2.medianBlur(query_img, 3)
    # seed_img = cv2.boxFilter(seed_img,-1,(3,3),normalize=0)
    # query_img = cv2.boxFilter(query_img,-1,(3,3),normalize=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    query_img = cv2.dilate(query_img, kernel)
    seed_img = cv2.dilate(seed_img, kernel)
    # seed_img = cv2.medianBlur(seed_img, 5)
    # query_img = cv2.medianBlur(query_img, 5)
    # seed_img = cv2.boxFilter(seed_img,-1,(3,3),normalize=0)
    # query_img = cv2.boxFilter(query_img,-1,(3,3),normalize=0)
    
    # query = cv2.erode(query, kernel)
    # seed_img = cv2.erode(seed_img, kernel)
    # blur_query = cv2.GaussianBlur(query,(kernel_size, kernel_size), 0)
    # blur_seed_img = cv2.GaussianBlur(seed_img,(kernel_size, kernel_size), 0)

    _,_,query_areas,query_center = cv2.connectedComponentsWithStats(query_img)
    valid_area_inds = query_areas[:,-1] > 1
    query_areas = query_areas[valid_area_inds]
    query_center = query_center[valid_area_inds]
    _,_,seed_areas,seed_center = cv2.connectedComponentsWithStats(seed_img)
    valid_area_inds = seed_areas[:,-1] > 1
    seed_areas = seed_areas[valid_area_inds]
    seed_center = seed_center[valid_area_inds]
    query_area = query_areas[:,-1]
    seed_area = seed_areas[:,-1]
    
    dist_matrix = np.linalg.norm(query_center[:,np.newaxis,:] - seed_center[np.newaxis,...],axis=2,ord=2)
    
    dist_matrix[dist_matrix>50] = 100000
    # dist_matrix = dist_matrix + area_matrix
    query_index, seed_index = linear_sum_assignment(dist_matrix)
    area_matrix = (query_area[query_index] / seed_area[seed_index])
    # area_matrix_inds = np.logical_and(area_matrix>0.3, area_matrix<3.0)
    valid_area_inds = np.logical_not(seed_areas[seed_index,-1] > 40000)
    # area_matrix_inds = np.logical_and(valid_area_inds, area_matrix_inds)
    # query_index = query_index[area_matrix_inds]
    # seed_index = seed_index[area_matrix_inds]
    
    # # a = seed_areas[1,:4]
    if False:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(query_img, cmap='Greys_r')
        # for i,coord in enumerate(gray_center):
        for i in query_index[1:]:
            coord = query_center[i,:]
            plt.text(coord[0],coord[1],'%d' % i,c='r')
            x1 = coord[0] - 0.5 * query_areas[i,2]
            y1 = coord[1] - 0.5 * query_areas[i,3]
        
            # x2 = coord[0] + 0.5 * query_areas[i,2]
            # y2 = coord[1] + 0.5 * query_areas[i,3]
            c = np.array([i,0.2*i,0.8*i]) / query_index[-1]
            plt.gca().add_patch(plt.Rectangle([query_areas[i,0], y1], query_areas[i,2],query_areas[i,3],  edgecolor= 'g', fill=False, linewidth=2))
        plt.subplot(1,2,2)
        plt.imshow(seed_img, cmap='Greys_r')
        # for i,coord in enumerate(seed_center) :
        for i in seed_index[1:]:
            coord = seed_center[i,:]
            plt.text(coord[0],coord[1],'%d' % i,c='r')

            x1 = coord[0] - 0.5 * seed_areas[i,2]
            y1 = coord[1] - 0.5 * seed_areas[i,3]
            # coord[0] + 0.5 * seed_areas[i,2],
            # coord[1] + 0.5 *  seed_areas[i,3]
            c= np.array([i,0.5*i,i]) / seed_index[-1]
            plt.gca().add_patch(plt.Rectangle([x1, y1], seed_areas[i,2], seed_areas[i,3],  edgecolor='g', fill=False, linewidth=2))
            # figManager = plt.get_current_fig_manager()
            # figManager.window.showMaximized()
            plt.savefig('./bev.png',format='png')
            # plt.gca().add_patch(plt.Rectangle(coord[0] - 0.5 * seed_areas[i,2], coord[1] - 0.5 *  seed_areas[i,3],\
            #     coord[0] + 0.5 * seed_areas[i,2], coord[1] + 0.5 *  seed_areas[i,3],\
            #     edgecolor=np.array([i,0.5*i,i])/ col_index[-1], fill=False, linewidth=2))
        print('*'*20)
        for f,t in zip(query_index ,seed_index):
            print('%d -> %d'%(f,t))

        plt.show()

    query_rpns = [query_center[query_index[1:],:], query_areas[query_index[1:],:4], query_index[1:]]
    seed_rpns = [seed_center[seed_index[1:], :], seed_areas[seed_index[1:], :4], seed_index[1:]]

    return seed_rpns, query_rpns    


def error(pred, labels, mask=None):
    pred = pred.permute(0, 2, 1).cpu().numpy()
    labels = labels.permute(0, 2, 1).cpu().numpy()
    if mask is None:
        mask = np.ones([pred.shape[0], pred.shape[1]])
    else:
        mask = 1 - mask
        mask = mask.permute(0, 2, 1).cpu().numpy()
        valid_flag = mask == 1
        # pred = pred[valid_flag]
        pred = pred[valid_flag[:,:,0],:]
        labels = labels[valid_flag[:,:,0],:]
        mask = mask[valid_flag[:,:,0],:][:,0]
    err = np.sqrt(np.sum((pred - labels) ** 2, -1) + 1e-20)

    gtflow_len = np.sqrt(np.sum(labels * labels, -1) + 1e-20)  # B,N
    acc050 = np.sum(np.logical_or((err <= 0.05) * mask, (err / gtflow_len <= 0.05) * mask), axis=0)
    acc010 = np.sum(np.logical_or((err <= 0.1) * mask, (err / gtflow_len <= 0.1) * mask), axis=0)
    # outlier = np.sum(np.logical_or((err > 0.3) * mask, (err / gtflow_len > 0.1) * mask), axis=1)

    mask_sum = np.sum(mask, 0)
    acc050 = acc050[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc050 = np.mean(acc050)
    acc010 = acc010[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc010 = np.mean(acc010)
    if acc050 >= 0.999:
        outlier = np.sum(np.logical_and((err > 0.3) * mask, mask), axis=0)
    else:
        # outlier = np.sum(np.logical_or((err > 0.3)*mask, (err/gtflow_len > 0.1)*mask), axis=1)
        outlier = np.sum(np.logical_or((err > 0.3) * mask, (err / gtflow_len > 0.1) * mask * acc050), axis=0)
    outlier = outlier[mask_sum > 0] / mask_sum[mask_sum > 0]
    outlier = np.mean(outlier)

    epe = np.sum(err * mask, 0)[mask_sum > 0] / mask_sum[mask_sum > 0]
    epe = np.mean(epe)
    return epe, acc050, acc010, outlier


if __name__ == '__main__':
    import os
    import argparse
    from utils.datasets.carla import CARLA3D
    
    f_arr = np.linspace(0,30,30)
    iFrm = 0
    iStep = 7
    # for iRound in range(3):
    #     for _ in range(30):
    #         if (iFrm + iStep) > 30:
    #             iFrm = 0
    #             break
    #         else:
    #             iFrm = (iFrm + iStep) % 30
    #             print(iFrm)
    egoV_file_path ='./data/data_0315_1209_Vinfo.npz'
    egoV_infos = np.load(egoV_file_path)
    egoV_info_c= egoV_infos['cvInfo']
    egoV_info_n= egoV_infos['nvInfo']

    dataset_path = './results/record2022_0315_1209/rm_egoV_fg/SF/val/'
    npoints = 50000
    # dataset = CARLA3D(root_dir=dataset_path, nb_points=npoints, mode="train")
    test_dataset = CARLA3D(root_dir=dataset_path, nb_points=npoints, mode="test")
    # data = np.load('./dataset/000005.npz')
    rm_ground = True
    val_num = 156
    dir_id = 0
    min_p_cluster = 2
    # data = np.load(dataset_path+'/00/000015.npz')
    # np_pc1 = data['pos1']
    # np_pc2 = data['pos2']
    # ego_flow =data['ego_flow']
    # gt_flow = data['gt']
    # cluster_estimator = DBSCAN(min_samples=3, metric='euclidean', leaf_size=5, eps=0.2) 
    # B = 1
    for iCnt, data in enumerate(test_dataset): 
        if iCnt <= 32:
            continue
        # data = np.load(item)
        i = iCnt + val_num * (dir_id)

        ego_v_info = np.array([float(egoV_info_c[iCnt, 2]), float(egoV_info_c[iCnt, 3]), float(egoV_info_c[iCnt, 4]),
                                float(egoV_info_c[iCnt, 5]), float(egoV_info_c[iCnt, 6]), float(egoV_info_c[iCnt, 7])])

        ego_v_info_n2 = np.array([float(egoV_info_n[iCnt, 2]), float(egoV_info_n[iCnt, 3]), float(egoV_info_n[iCnt, 4]),
                        float(egoV_info_n[iCnt, 5]), float(egoV_info_n[iCnt, 6]), float(egoV_info_n[iCnt, 7])])
        pc1 = data['sequence'][0]
        pc2 = data['sequence'][1]
        ego_flow = data['ground_truth'][0]
        flow = data['ground_truth'][1]
        
        # pc1 = pc1.cuda().transpose(2, 1).contiguous()
        # pc2 = pc2.cuda().transpose(2, 1).contiguous()
        # flow = flow.cuda().transpose(2, 1).contiguous()
        # ego_flow = ego_flow.cuda().transpose(2, 1).contiguous()
        # bg_mask = torch.norm(flow - ego_flow, dim=1).cpu().numpy()
        # bg_flag = np.ones([pc1.shape[0], 3, pc1.shape[-1]])
        # for i in range(flow.shape[0]):
        #     bg_flag[i, :, bg_mask[i,:] > 1e-3] = 0
        # bg_flag = torch.tensor(bg_flag).cuda()
        pc1 = pc1.cuda()
        pc2 = pc2.cuda()
        flow = flow.cuda()
        ego_flow = ego_flow.cuda()

        # delta_coarse_sf = calc_coarseflow(pc1+ego_flow, pc2, ego_v_info)

        # epe_3d, acc_3d, acc_3d_2, outlier = error(delta_coarse_sf.transpose(2, 1).contiguous(), flow)
        # print("%s: %f; %s: %f; %s: %f; %s: %f;" % ("EPE", epe_3d, "ACC_S", acc_3d, "ACC_R", acc_3d_2, "Outlier", outlier))
        # continue

        np_pc1 = (data['sequence'][0].squeeze()).numpy()
        np_pc2 = (data['sequence'][1].squeeze()).numpy()
        ego_flow = (data['ground_truth'][0].squeeze()).numpy()
        gt_flow = (data['ground_truth'][1].squeeze()).numpy()

        
        fgrnd_inds_s, _, fgrnd_s = ComputeRoadWidthAndExtractIoUPts(pc1, np_pc1, ego_v_info)
        fgrnd_inds_t, _, fgrnd_t = ComputeRoadWidthAndExtractIoUPts(pc2, np_pc2, ego_v_info_n2)

        coarse_flow = ego_flow[fgrnd_inds_s, :]
        torch_fgrnd_s = torch.tensor(fgrnd_s, dtype=torch.float).unsqueeze(0).transpose(1,2).cuda().contiguous()
        torch_fgrnd_t = torch.tensor(fgrnd_t, dtype=torch.float).unsqueeze(0).transpose(1,2).cuda().contiguous()
        torch_coarse_flow = torch.tensor(coarse_flow, dtype=torch.float).unsqueeze(0).transpose(1,2).cuda().contiguous()
        new_sf = calc_coarse_flow_from_bev([torch_fgrnd_s, torch_coarse_flow, torch_fgrnd_t], use_fg_inds=False)
        np_pc5 = fgrnd_s + ego_flow[fgrnd_inds_s, :]
        np_new_sf = new_sf.cpu().numpy()
        ego_flow[fgrnd_inds_s, :] = np_new_sf[0].T

        np_pc3 = fgrnd_s + np_new_sf[0].T
        np_pc4 = fgrnd_s + gt_flow[fgrnd_inds_s, :] 
        np_pc5 = np_pc1 + gt_flow

        # np_pc5 = np_pc1 + pre_ego_flow
        # np_pc6 = np_pc1 + pre_gt_flow
        pcd_1 = o3d.geometry.PointCloud()
        pcd_1.points = o3d.utility.Vector3dVector(np_pc1)
        pcd_1.paint_uniform_color([1.        , 0.69803922, 0.4])
        # color1 = np.zeros(np_pc1.shape)

        pcd_2 = o3d.geometry.PointCloud()
        pcd_2.points = o3d.utility.Vector3dVector(np_pc2)
        pcd_2.paint_uniform_color([1.        , 0.69803922, 0.4])
        # color1 = np.zeros(np_pc1.shape)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(fgrnd_s)
        pcd.paint_uniform_color([0,1,0])
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(fgrnd_t)
        pcd2.paint_uniform_color([1,0,0])
        pcd3 = o3d.geometry.PointCloud()
        pcd3.points = o3d.utility.Vector3dVector(np_pc3)
        pcd3.paint_uniform_color([0,0,1])
        pcd4 = o3d.geometry.PointCloud()
        pcd4.points = o3d.utility.Vector3dVector(np_pc4)
        # pcd4.paint_uniform_color([1,0,0])
        pcd4.paint_uniform_color([0       , 1, 1])
        pcd5 = o3d.geometry.PointCloud()
        pcd5.points = o3d.utility.Vector3dVector(np_pc5)
        pcd5.paint_uniform_color([1.        , 0.69803922, 0.4])
        # pcd6 = o3d.geometry.PointCloud()
        # pcd6.points = o3d.utility.Vector3dVector(np_pc6)
        # pcd6.paint_uniform_color([0,0.6,1])
        # vis_list = [pcd, pcd2, pcd3, pcd4, pcd5]
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5,origin=[0, 0, -3.4])
        vis_list = [ pcd_1, pcd, mesh]
        o3d.visualization.draw_geometries(vis_list)

        vis_list = [pcd_2, pcd2, mesh]
        o3d.visualization.draw_geometries(vis_list)

        # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5)
        # vis_list = [pcd_1, pcd, mesh]
        # o3d.visualization.draw_geometries(vis_list)

        # vis_list = [pcd_2, pcd2, mesh]
        # o3d.visualization.draw_geometries(vis_list)
        vis_list = [ pcd2, pcd5, pcd4, mesh]
        o3d.visualization.draw_geometries(vis_list)

        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5)
        vis_list = [pcd, mesh]
        o3d.visualization.draw_geometries(vis_list)

        vis_list = [pcd2, mesh]
        o3d.visualization.draw_geometries(vis_list)

        # clusters_set = {}
        # clusters_center_set = {}
        # for b_idx in range(B):
        #     b_fgrnd_idx_s = []
        #     b_fgrnd_idx_t = []

        #     b_fgrnd_center_s = []
        #     b_fgrnd_center_t = []

        #     labels_s = cluster_estimator.fit_predict(fgrnd_s[:,:])  
        #     labels_t = cluster_estimator.fit_predict(fgrnd_t[:,:])  
        #     for class_label in np.unique(labels_s):
        #         if class_label != -1 and np.where(labels_s == class_label)[0].shape[0] >= min_p_cluster:
        #             sub_cls_inds = np.where(labels_s == class_label)[0]
        #             sub_cls_center = np.mean(np_pc1[sub_cls_inds,:], axis=0)
        #             b_fgrnd_idx_s.append(sub_cls_inds) 
        #             b_fgrnd_center_s.append(sub_cls_center) 
        #     clusters_set['src_cls_inds'] = b_fgrnd_idx_s
        #     clusters_center_set['src_cls_cenetr'] = b_fgrnd_center_s
        #     seed_center = np.array(clusters_center_set['src_cls_cenetr'])
        #     # seed_center[:,-1] = -3.4

        #     for class_label in np.unique(labels_t):
        #         if class_label != -1 and np.where(labels_t == class_label)[0].shape[0] >= min_p_cluster:
        #             sub_cls_inds = np.where(labels_t == class_label)[0]
        #             sub_cls_center = np.mean(np_pc2[sub_cls_inds, :], axis=0)
        #             b_fgrnd_idx_t.append(sub_cls_inds)   
        #             b_fgrnd_center_t.append(sub_cls_center)
        #     clusters_set['tgt_cls_inds'] = b_fgrnd_idx_t
        #     clusters_center_set['tgt_cls_cenetr'] = b_fgrnd_center_t
        #     query_center = np.array(clusters_center_set['tgt_cls_cenetr'])
        #     # query_center[:,-1] = -3.4

        #     dist_matrix = np.linalg.norm(query_center[:,np.newaxis,:] - seed_center[np.newaxis,...],axis=2,ord=2)
        #     # dist_matrix[dist_matrix>50] = 100000

        #     query_index, seed_index = linear_sum_assignment(dist_matrix)
        #     a_map = np.vstack([query_index, seed_index])
        #     print(a_map)
        #     # plt.plot(fgrnd_s[clusters_set['src_cls_inds'][0],0], fgrnd_s[clusters_set['src_cls_inds'][0],1], 'r.')
        #     # # plt.plot(fgrnd_t[:,0], fgrnd_t[:,1], 'b.')
        #     # plt.plot(seed_center[:,0], seed_center[:,1], 'r*')
        #     # plt.plot(query_center[:,0], query_center[:,1], 'b*')
        #     # plt.plot(fgrnd_t[:,0], fgrnd_t[:,1], 'b.')
        #     for i,item in enumerate(np.unique(labels_s)):
        #         print(item)
        #         plt.plot(fgrnd_s[clusters_set['src_cls_inds'][item],0], fgrnd_s[clusters_set['src_cls_inds'][item],1], 'r.')
        #         # plt.text(seed_center[i,0], seed_center[i,1],str(item),color ='red', fontsize = 15)
        #     # for i, item in enumerate(query_index):
        #         # plt.text(query_center[:,0], query_center[:,1],str(item),color ='blue', fontsize = 15)
        #     # plt.plot(seed_center[seed_index,0], seed_center[seed_index,1], 'r*')
        #     # plt.plot(query_center[query_index,0], query_center[query_index,1], 'b*')
        #     # for i, item in enumerate(query_index):
        #     #     plt.text(seed_center[seed_index[i],0], seed_center[seed_index[i],1],str(seed_index[i]),color ='red', fontsize = 15)
        #     #     plt.text(query_center[query_index[i],0], query_center[query_index[i],1],str(item),color ='blue', fontsize = 15)
            
        #     plt.show()
        
        continue
        # pre_ego_flow = (data['ground_truth'][2].squeeze()).numpy()
        # pre_gt_flow = (data['ground_truth'][3].squeeze()).numpy()
        # valid_flag = (data['ground_truth'][2].squeeze()).numpy().astype(bool)


        bg_mask = np.linalg.norm(gt_flow - ego_flow, axis=1)
        fg_flag = np.zeros([np_pc1.shape[0]])
        fg_flag[bg_mask > 1e-3] = 1
        valid_flag = fg_flag == 1
        
        ego_flow = ego_flow[valid_flag,:]
        gt_flow = gt_flow[valid_flag,:]
        # pre_ego_flow = pre_ego_flow[valid_flag,:]
        # pre_gt_flow = pre_gt_flow[valid_flag,:]
        np_pc1 = np_pc1[valid_flag,:]
        # mask = mask[valid_flag,:][:,0]
        
        np_pc3 = np_pc1 + ego_flow
        np_pc4 = np_pc1 + gt_flow

        # np_pc5 = np_pc1 + pre_ego_flow
        # np_pc6 = np_pc1 + pre_gt_flow

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_pc1)
        pcd.paint_uniform_color([1,0,0])
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(np_pc2)
        pcd2.paint_uniform_color([0,1,0])
        pcd3 = o3d.geometry.PointCloud()
        pcd3.points = o3d.utility.Vector3dVector(np_pc3)
        pcd3.paint_uniform_color([0,0,1])
        pcd4 = o3d.geometry.PointCloud()
        pcd4.points = o3d.utility.Vector3dVector(np_pc4)
        pcd4.paint_uniform_color([1,0,0])
        # pcd5 = o3d.geometry.PointCloud()
        # pcd5.points = o3d.utility.Vector3dVector(np_pc5)
        # pcd5.paint_uniform_color([0.6,0,1])
        # pcd6 = o3d.geometry.PointCloud()
        # pcd6.points = o3d.utility.Vector3dVector(np_pc6)
        # pcd6.paint_uniform_color([0,0.6,1])
        vis_list = [pcd2, pcd2, pcd3, pcd4]
        # vis_list += [pcd3, pcd4]
        o3d.visualization.draw_geometries(vis_list)

    # pc1 = np.load('./data/np_src.npy')
    # pc1_torch = torch.tensor(pc1, dtype=torch.float32).unsqueeze(0).contiguous().cuda()
    # ComputeRoadWidthAndExtractIoUPts(pc1_torch,pc1)
    # ot_map()

    