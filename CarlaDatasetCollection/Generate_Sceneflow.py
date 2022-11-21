import sys
import os
import copy
import math
import shutil
import glob

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import concat
from utils.calibration import Calibration
from utils import odom_utils, flow_vis
import re 
from sklearn.neighbors import NearestNeighbors

from HPR import *
image_folder = './passive_image/'
# image_folder = './active_image_261/'

global img_cnt
img_cnt = 0
prefixed_V_name = "vehicle.dodge.charger_police_2020_261"
# prefixed_V_name = "vehicle.mini.cooper_s_2021_263"
# prefixed_V_name = "vehicle.dodge.charger_police_2020_265"
# prefixed_V_name = "vehicle.volkswagen.t2_325"
str_time = '1714'
only_fg = False
# DATASET_PATH = 'dataset/record2022_0308_1719/'
# DATASET_PATH = 'dataset/record2022_0312_1451/' #v_id=8
DATASET_PATH = 'dataset/record2022_0315_1209/' #v_id=8
# DATASET_PATH = 'dataset/record2022_0316_1838/' #v_id=8
# DATASET_PATH = 'dataset/record2022_0316_0046/' #v_id=75

# DATASET_PATH2 = 'dataset/record2022_0316_0046/' #v_id=75
# DATASET_PATH2 = 'dataset/record2022_0316_0308/' #v_id=17
# DATASET_PATH2 = 'dataset/record2022_0316_0312/' #v_id=7
# DATASET_PATH2 = 'dataset/record2022_0316_0316/' #v_id=7
DATASET_PATH2 = 'dataset/record2022_0316_' + str_time + '/' #v_id=7
DATASET_PATH2 = 'dataset/record2022_0317_' + str_time + '/' #v_id=7
# DATASET_PATH2 = 'dataset/record2022_0312_1531/' #v_id=32
DATASET_RAWTRANS_PATH = DATASET_PATH + '/global_label'
RESULT_PATH = './results'

SUB_DATASET = 'vehicle.dodge.charger_police_2020_261'#'vehicle.nissan.micra_1045'
v_pattern = r"(?P<name>[vehicle]+.\w+.\w+)_(?P<v_id>\d+)"
v = re.findall(v_pattern, SUB_DATASET)[0]
DATASET_PC_PATH = DATASET_PATH + SUB_DATASET + '/velodyne/'
DATASET_LBL_PATH = DATASET_PATH + SUB_DATASET + '/label00/'
DATASET_RAWTRANS_PATH = DATASET_PATH + '/global_label'
DATASET_RAWTRANS_PATH2 = DATASET_PATH2 + '/global_label'
RESULT_PATH = './results/' + SUB_DATASET

# RESULT_PATH2 = './results/record2022_0308_0930/rm_egoV/'
RESULT_PATH2 = './results/record2022_0315_1209/rm_egoV_fg2/'
RESULT_PATH2 = './results/record2022_0315_1209/rm_egoV_fg2_correcyion/'
# RESULT_PATH2 = './results/record2022_0315_0046/rm_egoV_fg2_paper/'
RESULT_PATH2 = './results/record2022_0316_1838/rm_egoV_fg2_correcyion/'
RESULT_PATH3 = './results/record2022_0315_1209_' + str_time + '/rm_egoV_fg2/'
RESULT_PATH3 = './results/record2022_0316_1838_' + str_time + '/rm_egoV2/'
RESULT_PATH3 = './results/record2022_0315_1209_' + str_time + '/rm_egoV_fg2_correcyion/'
# RESULT_PATH3 = './results/record2022_0314_0305_2241/rm_egoV/'
trans_pattern = v[0] + " " + v[1] + r" (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
    r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
        r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"\

lidar_pattern = r"sensor.lidar.ray_cast (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
    r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
        r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 " + v[1]

glb_label_pattern = r"vehicle.(?P<name>\w+.\w+) (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
    r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
        r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<extbbx_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<extbbx_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<extbbx_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<bbx_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"
rgb_label_pattern = r"Car (?P<obj_id>>?\d+) (?P<size>\d+) (.+) (?P<v_id>\d+)"
lidar_label_pattern = r"(?P<v_id>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<lt_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
    r"(?P<lt_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<rb_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
        r"(?P<rb_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)\n"
# a = '262.0 564.0694003615954 198.16495985774483 588.6026152409875 218.249973158069'
# nb = re.findall(lidar_label_pattern, a)
# print(nb[0])
# label0_pattern = r"Car (?P<obj_id>>?\d+) (?P<size>\d+) (?P<alpha>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<lt_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#     r"(?P<lt_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<rb_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<rb_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#     r"(?P<extbbx_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<extbbx_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<extbbx_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#         r"(?P<vloc_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<vloc_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#         r"(?P<vloc_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<vbbx_delta>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<v_id>\d+)"

def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    R = o3d.geometry.get_rotation_matrix_from_xyz([0,np.pi/2,0])
    for p in pcd:
        # p.rotate(R,[0,0,0])
        vis.add_geometry(p)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()


def custom_draw_geometry(vis, geometry_list, map_file=None, recording=False, param_file='camera_view.json', save_fov=False):
    vis.clear_geometries()
    # R = o3d.geometry.get_rotation_matrix_from_xyz([-np.pi/2,0,0])
    for pcd in geometry_list:
        # pcd.rotate(R,[0,0,0])
        vis.add_geometry(pcd)
    param = o3d.io.read_pinhole_camera_parameters(param_file)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
    ctr = vis.get_view_control()
    # ctr.set_zoom(0.4)
    # ctr.set_up((0, -1, 0))
    # ctr.set_front((1, 0, 0))
    ctr.convert_from_pinhole_camera_parameters(param)
    # vis.register_animation_callback(rotate_view)
    vis.run()
    # time.sleep(5)
    if save_fov:
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters('camera_view.json', param)
    if recording:
        vis.capture_screen_image(map_file,True)
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        # if test == 1:
        o3d.io.write_pinhole_camera_parameters(param_file, param)


def get_matrix(location, rotation):
    T = np.matrix(np.eye(4))
    T[:3,:3] = o3d.geometry.get_rotation_matrix_from_xyz(rotation)
    T[0:3,3] = location.reshape(3,1)
    return T

def get_vehicle_bbox(other_vehilcle_label, sensor_center, sensor_rotation, flag=False):
    bbox_center = np.array(list(map(float,other_vehilcle_label[2:5]))) + np.array([0,0,float(other_vehilcle_label[-1])])
    bbox_center[1] *= -1
    sensor_world_matrix = get_matrix(sensor_center,-sensor_rotation+[0,0,np.pi*4/2])
    # sensor_world_matrix = get_matrix(sensor_center,-sensor_rotation+[0,0,np.pi*3/2])
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    bbox_center = np.append(bbox_center,1).reshape(4,1)
    bbox_center = np.dot(world_sensor_matrix, bbox_center).tolist()[:3]
    tmp = sensor_rotation-[0,0,np.radians(float(other_vehilcle_label[7]))]
    bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(tmp)
    # bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(sensor_rotation-[0,0,np.radians(float(other_vehilcle_label[7]))])
    bbox_delta = sensor_rotation[2] - np.radians(float(other_vehilcle_label[7])) - np.pi / 2
    # print(bbox_delta)
    bbox_extend = np.array([float(num)*2 for num in other_vehilcle_label[-4:-1]])
    if flag:
        bbox_extend += np.array([0.1,0.1,0.05])
    bbox = o3d.geometry.OrientedBoundingBox(bbox_center, bbox_R, bbox_extend)
    bbox.color = np.array([0.5, 1.0, 0.5])
    return bbox, bbox_extend, bbox_center, bbox_delta


def Compute_transf(src_vec, tgt_vec):
    '''
    Compute the transformation from target frame to source (origin) frame
    src_trans | tgt_trans = [x, y, z, x_roll, y_pitch, z_yaw]
    '''
    norm_t = np.linalg.norm(tgt_vec)
    norm_s = np.linalg.norm(src_vec)
    z = tgt_vec / norm_t
    u = src_vec - norm_s * z
    norm_u = (u / np.linalg.norm(u)).reshape(3,1)
    H = np.eye(3) - 2 * norm_u @ norm_u.T

    return H

def get_sensor_transform(vehicle_id, labels, sensor=None):
    if sensor == 'lidar':
        for label in labels:
            # print(label)
            if vehicle_id == label[-1] and 'lidar' in label[0]:
                sensor_center = np.array(list(map(float,label[2:5])))
                sensor_center[1] *= -1
                # sensor_rotation = np.array([-np.radians(float(label[6])),-np.radians(float(label[5])),np.radians(float(label[7]))])
                sensor_rotation = np.array([0,0,np.radians(float(label[7]))])
                sensor_rotation_test = np.array([np.radians(float(test)) for test in label[5:7]]+[0])
                return sensor_center, sensor_rotation, sensor_rotation_test
    elif sensor == 'camera':
        camera_info_list = []
        for label in labels:
            # if vehicle_id == label[-1] and 'camera.rgb' in label[0]:
            if vehicle_id == label[-1] and 'camera' in label[0]:
                sensor_center = np.array(list(map(float,label[2:5])))
                sensor_center[1] *= -1
                # sensor_rotation = np.array([-np.radians(float(label[6])),-np.radians(float(label[5])),np.radians(float(label[7]))])
                sensor_rotation = np.array([0,0,np.radians(float(label[7]))])
                camera_info_list.append([sensor_center,sensor_rotation,label[1]])
        return camera_info_list


def householder(vecA, vecB = [0,0,1]):
    vecB = np.asarray(vecB).reshape(3,1)
    vecA = np.asarray(vecA).reshape(3,1)
    norm_va = np.linalg.norm(vecA)
    va = vecA - norm_va * vecB
    u = va / np.linalg.norm(va)
    H = np.eye(3) - 2 * u @ u.T
    H2 = np.linalg.inv(H)
    return H2


def Get_SenceFlowArrow(np_pcd, sf, pt_color, np_pcd2=None, pt_color2=None):
    arrow_lens = np.linalg.norm(sf, axis=1)
    triMeshArrows = []
    np_pcd_trans = np.asarray(np_pcd)
    irows = np_pcd.shape[0]
    pc1_flag = True
    pc2_flag = False

    if np_pcd2 is not None:
        np_pcd2_trans = np.asarray(np_pcd2)
        pc2_flag = True
        if np_pcd.shape[0] < np_pcd2.shape[0]:
            irows = np_pcd2.shape[0]
            
    max_pc1_z = np.max(np_pcd[:,2])
    min_pc1_z = np.min(np_pcd[:,2])
    min_max_pc1_z = max_pc1_z - min_pc1_z
    max_pc2_z = np.max(np_pcd2[:,2])
    min_pc2_z = np.min(np_pcd2[:,2])
    min_max_pc2_z = max_pc2_z - min_pc2_z
    for item in range(sf.shape[0]):
        if pc1_flag:
            triMeshArrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.05, cone_radius=0.1, cone_height=0.2, cylinder_height=arrow_lens[item])
            triMeshSphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15, resolution=20)
            rot_mat = householder(sf[item, :])
            triMeshArrow.rotate(rot_mat, [0,0,0])
            triMeshArrow.translate(np_pcd_trans[item,:])
            triMeshSphere.translate(np_pcd_trans[item,:])
            triMeshArrow.paint_uniform_color([1,1,1])
            # pt_color[1] = (np_pcd[item,2] - min_pc1_z) / min_max_pc1_z
            # pt_color[0] = 1 - pt_color[1]
            # pt_color[2] = 1 - pt_color[1]*0.5
            triMeshSphere.paint_uniform_color(pt_color)
            if item == (np_pcd.shape[0]-1):
                pc1_flag = False
        if pc2_flag:
            triMeshSphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.15, resolution=20)
            triMeshSphere2.translate(np_pcd2_trans[item,:])
            # pt_color2[0] = (np_pcd2[item,2] - min_pc2_z) / min_max_pc2_z
            # pt_color2[2] = 1 - pt_color2[1]
            # pt_color2[2] = 1 - pt_color2[1]*0.3
            triMeshSphere2.paint_uniform_color(pt_color2)
            
            if item == (np_pcd2.shape[0]-1):
                pc2_flag = False
        # print(np.asarray(triMeshArrow.vertices))
        if pc1_flag and pc2_flag:
            triMeshArrows.append(triMeshArrow + triMeshSphere + triMeshSphere2)
        elif pc1_flag:
            triMeshArrows.append(triMeshArrow + triMeshSphere)
        elif pc2_flag:
            triMeshArrows.append(triMeshSphere2)
    return triMeshArrows

def write_labels(raw_data_date, ego_vehicle_name, frame_start, frame_end, frame_hz, index, camera_id=0):
    # ego_vehicle_name = ego_vehicle_label[0]+'_'+ ego_vehicle_label[1]
    '''
    index = 0: Camera FOV
    index = 1: LiDAR FOV
    '''
    dataset_path = 'dataset/' + raw_data_date + '/' + ego_vehicle_name
    raw_data_path = 'tmp/' + raw_data_date + ego_vehicle_name
    sensor_raw_path = os.listdir(raw_data_path)
    if not os.path.exists(dataset_path+'/label0'+str(index)):
        os.makedirs(dataset_path+'/label0'+str(index))
    
    # frame_start,frame_end,frame_hz = 10,-155,1
    
    for _tmp in sensor_raw_path:
        # if str(camera_id) in _tmp and 'label' in _tmp: 
        if 'label' in _tmp: 
            lbl_path = raw_data_path + '/' + _tmp #+ '/' + _frame[:-3] + 'txt' 
            print(lbl_path)
            frames = glob.glob(os.path.join(lbl_path, '*.txt'))
            frames.sort()
            for _frame in frames[frame_start:frame_end:frame_hz]:
                shutil.copy(_frame, dataset_path+'/label0'+str(index))
                
    
    
def Draw_SceneFlow(vis, np_list, use_fast_vis = True, use_pred_t = False, use_custom_color=True, use_flow=False, use_arrow=False, param_file='camera_view.json',img_cnt=0):
    pc1 = np_list[0]
    pc2 = np_list[1]
    pred_pc1 = np_list[2]
    
    if use_pred_t:
        pred_pc1 = pc1+np_list[3]
        pred_flow = np_list[3]
        pred_t_query_pt = o3d.geometry.PointCloud()
        pred_t_query_pt.points = o3d.utility.Vector3dVector(pred_flow)
        if use_custom_color:
            pred_t_query_pt.paint_uniform_color([1,1,1])
    else:
        pred_pc1 = np_list[2]

    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().point_size = 3.0
        vis.get_render_option().background_color = [0, 0, 0]

    # pred_pc1 = pc1 + flow
    pred_query_pt = o3d.geometry.PointCloud()
    pred_query_pt.points = o3d.utility.Vector3dVector(pred_pc1)
    # pred_query_pt.paint_uniform_color([0,0,1])
    
    query_pt = o3d.geometry.PointCloud()
    query_pt.points = o3d.utility.Vector3dVector(pc1)
    if use_custom_color:
        query_pt.paint_uniform_color([0,1,0])
        pred_query_pt.paint_uniform_color([0,0,1])
        
    if pc2 is not None:
        query_pt2 = o3d.geometry.PointCloud()
        query_pt2.points = o3d.utility.Vector3dVector(pc2)
        query_pt2.paint_uniform_color([1,0,0])
        if use_pred_t:
            # vis_list = [query_pt2, pred_query_pt, pred_t_query_pt, query_pt]
            vis_list = [query_pt2, pred_query_pt, query_pt]
        else:
            vis_list = [query_pt2, query_pt]
    else:
        vis_list = [pred_query_pt, pred_t_query_pt, query_pt]

    if use_flow:
        flow = np_list[3]
        corr_inds = np.arange(0, flow.shape[0], 1).reshape(-1,1)
        lines = np.hstack((corr_inds, corr_inds))
        flow_colors = np.expand_dims(flow, axis=0)
        colors = flow_vis.flow_to_color(flow_colors[:, :,:3], convert_to_bgr=False)[0]
        colors = colors / 255.0

        line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(query_pt, pred_query_pt, lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis_list += [line_set]
    if len(np_list)>=5:
        vis_list += np_list[4:]
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, -2.5])
    # bbx_pcd = o3d.geometry.PointCloud()
    # bbx_pcd.points = o3d.utility.Vector3dVector(bbx)
    # bbx_pcd.paint_uniform_color([1,0,1])
    # vis.add_geometry(bbx_pcd)
    # vis.add_geometry(mesh_frame)
    if use_fast_vis:
        for item in vis_list:
            vis.add_geometry(item)
    else:
        if use_flow:
            if use_arrow:
                flow = np_list[3]
                triMeshArrows = Get_SenceFlowArrow(pc1, flow, [0,1,0], pc2, [1,0,0])
                # o3d.visualization.draw_geometries(triMeshArrows)
                for item in triMeshArrows:
                    vis.add_geometry(item)
            else:
                for item in vis_list:
                    vis.add_geometry(item)
    param = o3d.io.read_pinhole_camera_parameters(param_file)
    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
    ctr = vis.get_view_control()
    # ctr.set_zoom(0.4)
    # ctr.set_up((0, -1, 0))
    # ctr.set_front((1, 0, 0))
    ctr.convert_from_pinhole_camera_parameters(param)
    # vis.register_animation_callback(rotate_view)
    vis.run()
    # dummy_fig = plt.gcf()
    img = vis.capture_screen_float_buffer(True)
    img = np.asarray(img)
    img = o3d.geometry.Image((img * 255).astype(np.uint8))
    image_name = image_folder + '{:04d}'.format(img_cnt) + '.png'
    o3d.io.write_image(image_name, img)
    # plt.axes().get_xaxis().set_visible(False)
    # plt.axes().get_yaxis().set_visible(False)  
    # plt.imshow(np.asarray(img))
    # dummy_fig.savefig('./test.png',format='png')
    vis.clear_geometries()

    # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # o3d.io.write_pinhole_camera_parameters(param_file, param)
    # vis.destroy_window()


def Compute_PairwiseSceneFlow(ind, step):
# def Compute_PairwiseSceneFlow(ind, next_ind, trans_pattern, lidar_pattern, DATASET_PC_PATH, DATASET_LBL_PATH):
    # ind = 46
    next_ind = ind + step
    # Translate frame coordinates from left-hand to right-hand
    v_l2r = np.array([1., -1., 1., 1., 1., 1.])
    s_l2r = np.array([1, 1., -1., 1., 1., 1., 1.])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, -2.5])

    
    pc_files = glob.glob(os.path.join(DATASET_PC_PATH, '*.bin'))#os.listdir(DATASET_PC_PATH)
    rawtrans_files = glob.glob(os.path.join(DATASET_RAWTRANS_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
    label_files = glob.glob(os.path.join(DATASET_LBL_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
    pc_files.sort()
    rawtrans_files.sort()
    label_files.sort()
    
    first_v_raw_trans = odom_utils.readRawData(rawtrans_files[ind], trans_pattern) * v_l2r
    first_sensor_raw_trans = odom_utils.readRawData(rawtrans_files[ind], lidar_pattern) * s_l2r

    src_pc = odom_utils.readPointCloud(pc_files[ind])[:, :3]
    
    #For the carla origin coordatinate 
    # first_v_raw_trans[2] += 2.5
    # src_pc += first_v_raw_trans[:3]
    arr = src_pc
    
    src_R_vec = first_sensor_raw_trans[4:]
    src_R = o3d.geometry.get_rotation_matrix_from_xyz(src_R_vec)
    src_R_inv = np.linalg.inv(src_R)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(src_pc)

    #For the carla origin coordatinate 
    # pcd.rotate(src_R_inv, first_v_raw_trans[:3])

    # For the first frame as the origin coordinate
    pcd.rotate(src_R_inv, np.zeros(3))

    pcd.paint_uniform_color([0,1,0])
    # vis_list = [mesh_frame, pcd]
    vis_list = [mesh_frame]
    # vis_list = []

    tgt_v_raw_trans = odom_utils.readRawData(rawtrans_files[next_ind], trans_pattern) * v_l2r
    tgt_sensor_raw_trans = odom_utils.readRawData(rawtrans_files[next_ind], lidar_pattern) * s_l2r
    
    tgt_pc = odom_utils.readPointCloud(pc_files[next_ind])[:, :3]
   

    #For the carla origin coordatinate 
    # tgt_v_raw_trans[2] += 2.5
    # tgt_pc += tgt_v_raw_trans[:3]

    tgt_R_vec = tgt_sensor_raw_trans[4:]
    tgt_R = o3d.geometry.get_rotation_matrix_from_xyz(tgt_R_vec)
    tgt_R_inv = np.linalg.inv(tgt_R)
    pcd_cur = o3d.geometry.PointCloud()
    pcd_cur.points = o3d.utility.Vector3dVector(tgt_pc)

    #For the carla origin coordatinate 
    # pcd_cur.rotate(tgt_R_inv, tgt_v_raw_trans[:3] )

    #For the first frame as the origin coordinate
    pcd_cur.rotate(tgt_R_inv, np.zeros(3))
    pcd_cur.translate(tgt_v_raw_trans[:3] - first_v_raw_trans[:3])

    pcd_cur.paint_uniform_color([1,0,0])

    # arr = ((arr-first_v_raw_trans[:3]) @ src_R_inv.T - tgt_v_raw_trans[:3] + first_v_raw_trans[:3]) @ tgt_R.T + tgt_v_raw_trans[:3]
    # flow = arr - src_pc
    
    pcd_cur2 = o3d.geometry.PointCloud()
    pcd_cur2.points = o3d.utility.Vector3dVector(src_pc)
    pcd_cur2.paint_uniform_color([0,1.0,0])
    pcd_cur3 = o3d.geometry.PointCloud()
    pcd_cur3.points = o3d.utility.Vector3dVector(tgt_pc)
    pcd_cur3.paint_uniform_color([1.0,0,0])
    # vis_list = [mesh_frame, pcd, pcd_cur, pcd_cur2, pcd_cur3]
    arr_ = (arr @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T
    # flow = arr_ - arr

    pcd_cur4 = o3d.geometry.PointCloud()
    pcd_cur4.points = o3d.utility.Vector3dVector(arr_)
    pcd_cur4.paint_uniform_color([0,1.0,0])

    oth_cars_info = []
    for i in [ind, next_ind]:
        with open(label_files[i], 'r') as file:
                log = file.read()
        det_car_info = []
        for r in re.findall(rgb_label_pattern, log):
            det_car_info.append(int(r[-1]))
        
        with open(rawtrans_files[i], 'r') as file:
            raw_log = file.read()
        oth_car_info = []
        for r in re.findall(glb_label_pattern, raw_log):
            # print(r[0])
            if int(r[1]) in det_car_info:
                oth_car_info.append([int(r[1])] + list(map(float, r[2:5])) + [np.radians(float(r[5])), np.radians(float(r[6])), np.radians(float(r[7]))] + list(map(float, r[-4:])))
        oth_cars_info.append(oth_car_info)
        # print(oth_car_info)
        
    show_flag = 1
    lcl_R = [src_R, tgt_R]
    lcl_R_inv = [src_R_inv, tgt_R_inv]
    lcl_rot_v = [first_v_raw_trans[3:], tgt_v_raw_trans[3:]]
    lcl_trans_v = [first_v_raw_trans[:3], tgt_v_raw_trans[:3]]
    colors = [[0,1,0], [1,0,0]]
    objs_bbox_R = []
    objs_flow = []
    objs_bbx = []
    if show_flag:
        for iv in range(len(oth_cars_info)):
            oth_v_loc = np.array(oth_cars_info[iv])[:,1:4] * np.array([1.0, -1.0, 1.0])
            oth_v_loc[:,-1] = oth_v_loc[:,-1] + np.array(oth_cars_info[iv])[:,-1] - 2.5
            oth_v_rot = np.array(oth_cars_info[iv])[:,4:7]
            oth_v_bbx_ext = np.array(oth_cars_info[iv])[:,-4:]

            # objs_center = []
            obj_flow = []
            obj_bbx = []
            obj_bbox_R = []

            for ibbx in range(oth_v_loc.shape[0]):
                bbox_rot = lcl_rot_v[iv] - oth_v_rot[ibbx, :] 
                # bbox_rot = lcl_rot_v[iv] - bbox_rot
                # print(bbox_rot)

                bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(bbox_rot)
                obj_bbox_R.append(bbox_rot)
                bbox_extend = oth_v_bbx_ext[ibbx, :-1] * 2.0
                bbox_center = lcl_R[iv] @ (oth_v_loc[ibbx,:]- lcl_trans_v[iv])
                print(bbox_center)
                obj_flow.append(np.hstack((np.array(oth_cars_info[iv][ibbx][0]), np.array(bbox_center))))
                # bbox = o3d.geometry.OrientedBoundingBox(bbox_center, lcl_R_inv[iv] @ bbox_R, bbox_extend)
                bbox = o3d.geometry.OrientedBoundingBox(bbox_center, bbox_R, bbox_extend)
                obj_bbx.append(bbox)
                bbox.color = colors[iv]
                vis_list.append(bbox)
            objs_flow.append(obj_flow)
            objs_bbx.append(obj_bbx)
            objs_bbox_R.append(obj_bbox_R)

        if len(objs_flow) <= 1:
            flow = arr_ - arr
        else:
            src_objs_flow = np.array(objs_flow[0])
            src_objs_bbox = dict(zip(src_objs_flow[:,0], objs_bbx[0]))
            src_objs_flow = dict(zip(src_objs_flow[:,0], src_objs_flow[:,1:]))
            src_bbox_R = dict(zip(src_objs_flow.keys(),objs_bbox_R[0]))
        
            tgt_objs_flow = np.array(objs_flow[1])
            tgt_objs_bbox = dict(zip(tgt_objs_flow[:,0], objs_bbx[1]))
            tgt_objs_flow = dict(zip(tgt_objs_flow[:,0], tgt_objs_flow[:,1:]))
            tgt_bbox_R = dict(zip(tgt_objs_flow.keys(),objs_bbox_R[1]))

        objs_flow = []
        objs_c = np.array(list(src_objs_flow.values()))
        subarr_ = (objs_c @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T
        subflow = subarr_ - objs_c
        src_rigid_flow = dict(zip(src_objs_flow.keys(), subflow))
        for k,v in src_objs_flow.items():   
            if k in tgt_objs_flow:
                obj_flow = tgt_objs_flow.get(k) - v
                delta_flow = obj_flow - src_rigid_flow.get(k)
                bbox = src_objs_bbox.get(k) 
                inds = bbox.get_point_indices_within_bounding_box(pcd_cur2.points)
                # flow[inds,:] = obj_flow
                arr_[inds,:] += delta_flow
                delat_rot = tgt_bbox_R.get(k) - src_bbox_R.get(k)
                bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(delat_rot)
                arr_[inds,:] = (arr_[inds,:] - tgt_objs_bbox.get(k).get_center()) @ bbox_R.T + tgt_objs_bbox.get(k).get_center()
            else:
                obj_flow = np.zeros(3)
            # print(obj_flow)
            
            objs_flow.append(obj_flow)
        # delta_flow = np.array(objs_flow) - subflow
        objs_flow = dict(zip(src_objs_flow.keys(), objs_flow))
    flow = arr_ - arr  
    # vis_list.append(pcd)
    pcd_cur5 = o3d.geometry.PointCloud()
    pcd_cur5.points = o3d.utility.Vector3dVector(arr_)
    pcd_cur5.paint_uniform_color([0,0,1.0])

    vis_list.append(pcd_cur5)
    vis_list.append(pcd_cur2)
    vis_list.append(pcd_cur3)
    o3d.visualization.draw_geometries(vis_list)

    if not(show_flag):
        use_vis = False
        use_flow = True
        vis = None
    
        if use_vis:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.get_render_option().point_size = 3.0
            vis.get_render_option().background_color = [0, 0, 0]
        if use_flow:
                # sf = sf[sample_idx1, :]
            np_list = [src_pc, tgt_pc, arr_, flow]
        else:
            np_list = [src_pc, tgt_pc, arr_,]
            # print(np.mean(np.linalg.norm(pos2- pc2, axis=-1), axis=-1))

        Draw_SceneFlow(vis, np_list + vis_list, use_fast_vis = not(use_flow), use_pred_t=False, use_custom_color=True, use_flow=use_flow)
    # pcd_cur4 = o3d.geometry.PointCloud()

    # pcd_cur4.points = o3d.utility.Vector3dVector(arr+flow)
    # pcd_cur4.paint_uniform_color([0,1.0,0])
    # vis_list = [mesh_frame, pcd_cur4, pcd_cur2, pcd_cur3]


    # # vis_list.append(pcd_cur)
    # o3d.visualization.draw_geometries(vis_list)
    # if not os.path.exists(RESULT_PATH):
    #     os.makedirs(RESULT_PATH)

    # o3d.io.write_point_cloud(os.path.join(RESULT_PATH,SUB_DATASET + ".pcd"), pcd)


# if __name__ == "__main__":
#     start = 0
#     drivable_area = odom_utils.readPointCloud('./dataset/town02-map.bin')[:, :3]
#     pcd_drivable_area = o3d.geometry.PointCloud()
#     pcd_drivable_area.points = o3d.utility.Vector3dVector(drivable_area)
#     pcd_drivable_area.paint_uniform_color([1.0,0,0.0])
#     # Compute_PairwiseSceneFlow(start, 5)
    
#     # Translate frame coordinates from left-hand to right-hand
#     v_l2r = np.array([1., -1., 1., 1., 1., 1.])
#     s_l2r = np.array([1, 1., -1., 1., 1., 1., 1.])
    
#     enable_animation = True
#     if enable_animation:
#         vis = o3d.visualization.Visualizer()
#         vis.create_window(width=960*2, height=640*2, left=5, top=5)
#         vis.get_render_option().background_color = np.array([0, 0, 0])
#         # vis.get_render_option().background_color = np.array([1, 1, 1])
#         vis.get_render_option().show_coordinate_frame = False
#         vis.get_render_option().point_size = 1.0
#         vis.get_render_option().line_width = 5.0
#     mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, -2.5])
    
#     # Generate the log data from the DIR:tmp/xxx/xxxxrgb_lable/xxxx 
#     build_lidar_fov = False

#     # Generate the scene flow data of the only camera field-of-view
#     use_rgb_fov = False
    
    
#     # trans_pattern = r"(?P<v_name>[vehicle]+.\w+.\w+) (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#     # r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#     #     r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#     #         r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"\
    
#     rawtrans_files = glob.glob(os.path.join(DATASET_RAWTRANS_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
#     rawtrans_files.sort()
#     # if not(use_rgb_fov):
#     #     for r in re.findall(trans_pattern, log):
#     #         # print(r[0])
#     #         det_car_info.append(int(float(r[0])))
#     print(os.path.abspath(os.curdir))
#     for sub_dir in os.listdir(DATASET_PATH):
#         if 'global_label' in sub_dir or 'img' in sub_dir:
#             continue 
#         # print(sub_dir)

#         SUB_DATASET = sub_dir #'vehicle.tesla.cybertruck_266'#'vehicle.citroen.c3_271'

#         v_pattern = r"(?P<name>[vehicle]+.\w+.\w+)_(?P<v_id>\d+)"
#         v = re.findall(v_pattern, SUB_DATASET)[0]
#         DATASET_PC_PATH = DATASET_PATH + SUB_DATASET + '/velodyne/'
#         DATASET_LBL_PATH = DATASET_PATH + SUB_DATASET + '/label00/'
#         # RESULT_PATH = './results/' + SUB_DATASET
#         trans_pattern = v[0] + " " + v[1] + r" (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#             r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                 r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                     r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"
#         lidar_pattern = r"sensor.lidar.ray_cast (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#             r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                 r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                     r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 " + v[1]
        
#         lidar_semantic_pattern = r"sensor.lidar.ray_cast_semantic (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#             r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                 r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                     r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 0 " + v[1]
        
#         # next_ind = ind + 2
#         pc_files = glob.glob(os.path.join(DATASET_PC_PATH, '*.bin'))#os.listdir(DATASET_PC_PATH)
#         # rawtrans_files = glob.glob(os.path.join(DATASET_RAWTRANS_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
#         label_files = glob.glob(os.path.join(DATASET_LBL_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
#         # print("Current DIR:" + DATASET_PC_PATH)
#         pc_files.sort()
#         # rawtrans_files.sort()
#         label_files.sort()
#         if build_lidar_fov:
#             frame_start,frame_end,frame_hz = 10,-5,1
#             write_labels('record2021_0716_2146/', sub_dir, frame_start,frame_end,frame_hz, index=1)
#         else:
#             step = 3
#             for ind in range(start, len(pc_files)-step, step):
#                 next_ind = ind + step
#                 if next_ind >= len(label_files):
#                         break
#                 print("Current DIR:" + DATASET_PC_PATH + ' Frame:' +str(ind))
#                 # Compute_PairwiseSceneFlow(ind, next_ind, trans_pattern, lidar_pattern, DATASET_PC_PATH, DATASET_LBL_PATH)
#                 vis_list = []
#                 first_v_raw_trans = odom_utils.readRawData(rawtrans_files[ind], trans_pattern) * v_l2r
#                 first_sensor_raw_trans = odom_utils.readRawData(rawtrans_files[ind], lidar_semantic_pattern) * s_l2r

#                 src_pc = odom_utils.readPointCloud(pc_files[ind])[:, :3]
#                 #For the carla origin coordatinate 
#                 # first_v_raw_trans[2] += 2.5
#                 # src_pc += first_v_raw_trans[:3]
#                 arr = src_pc
                
#                 src_R_vec = first_sensor_raw_trans[4:]
#                 src_R = o3d.geometry.get_rotation_matrix_from_xyz(src_R_vec)
#                 src_R_inv = np.linalg.inv(src_R)
#                 pcd = o3d.geometry.PointCloud()
#                 pcd.points = o3d.utility.Vector3dVector(src_pc)
#                 #For the carla origin coordatinate 
#                 # pcd.rotate(src_R_inv, first_v_raw_trans[:3])

#                 # For the first frame as the origin coordinate
#                 # pcd.rotate(src_R_inv, np.zeros(3))
#                 pcd.paint_uniform_color([0,1,0])
                
#                 vis_list = [mesh_frame, pcd]
                
#                 tgt_v_raw_trans = odom_utils.readRawData(rawtrans_files[next_ind], trans_pattern) * v_l2r
#                 tgt_sensor_raw_trans = odom_utils.readRawData(rawtrans_files[next_ind], lidar_semantic_pattern) * s_l2r
                
#                 tgt_pc = odom_utils.readPointCloud(pc_files[next_ind])[:, :3]

#                 #For the carla origin coordatinate 
#                 # tgt_v_raw_trans[2] += 2.5
#                 # tgt_pc += tgt_v_raw_trans[:3]

#                 tgt_R_vec = tgt_sensor_raw_trans[4:]
#                 tgt_R = o3d.geometry.get_rotation_matrix_from_xyz(tgt_R_vec)
#                 tgt_R_inv = np.linalg.inv(tgt_R)
#                 pcd_cur = o3d.geometry.PointCloud()
#                 pcd_cur.points = o3d.utility.Vector3dVector(tgt_pc)

#                 #For the carla origin coordatinate 
#                 # pcd_cur.rotate(tgt_R_inv, tgt_v_raw_trans[:3] )

#                 #For the first frame as the origin coordinate
#                 # pcd_cur.rotate(tgt_R_inv, np.zeros(3))
#                 # pcd_cur.translate(tgt_v_raw_trans[:3] - first_v_raw_trans[:3])
#                 pcd_cur.paint_uniform_color([1,0,0])
                
#                 arr_ = (src_pc @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T

#                 oth_cars_info = []
#                 for i in [ind, next_ind]:
#                     # if next_ind >= len(label_files):
#                     #     break
#                     with open(label_files[i], 'r') as file:
#                             log = file.read()
#                     det_car_info = []  
#                     if use_rgb_fov:
#                         for r in re.findall(rgb_label_pattern, log):
#                             det_car_info.append(int(r[-1]))
#                     # else:
#                     #     for r in re.findall(lidar_label_pattern, log):
#                     #         # print(r[0])
#                     #         det_car_info.append(int(float(r[0])))
                    
#                     with open(rawtrans_files[i], 'r') as file:
#                         raw_log = file.read()
#                     oth_car_info = []
#                     for r in re.findall(glb_label_pattern, raw_log):
#                         # print(r[0])
#                         if use_rgb_fov:
#                             if int(r[1]) in det_car_info:
#                                 oth_car_info.append([int(r[1])] + list(map(float, r[2:5])) + [np.radians(float(r[5])),\
#                                     np.radians(float(r[6])), np.radians(float(r[7]))] + list(map(float, r[-4:])))
#                         else:
#                             if r[1] != v[1]:
#                                 oth_car_info.append([int(r[1])] + list(map(float, r[2:5])) + [np.radians(float(r[5])),\
#                                     np.radians(float(r[6])), np.radians(float(r[7]))] + list(map(float, r[-4:])))
#                     if len(oth_car_info) > 0:
#                         oth_cars_info.append(oth_car_info)
#                     # print(oth_car_info)
                    
#                 show_flag = 1
#                 lcl_R = [src_R, tgt_R]
#                 lcl_R_inv = [src_R_inv, tgt_R_inv]
#                 lcl_rot_v = [first_v_raw_trans[3:], tgt_v_raw_trans[3:]]
#                 lcl_trans_v = [first_v_raw_trans[:3], tgt_v_raw_trans[:3]]
#                 colors = [[0,1,0], [1,0,0]]
#                 objs_center = []
#                 objs_flow = []
#                 objs_bbx = []
#                 objs_bbox_R = []
#                 # if show_flag:
#                 for iv in range(len(oth_cars_info)):
#                     if len(oth_cars_info) <= 1:
#                         flow = arr_ - arr
#                         # continue
#                     else:
#                         oth_v_loc = np.array(oth_cars_info[iv])[:,1:4] * np.array([1.0, -1.0, 1.0])
#                         oth_v_loc[:,-1] = oth_v_loc[:,-1] + np.array(oth_cars_info[iv])[:,-1] - 2.5
#                         oth_v_rot = np.array(oth_cars_info[iv])[:,4:7]
#                         oth_v_bbx_ext = np.array(oth_cars_info[iv])[:,-4:]

#                         # objs_center = []
#                         obj_flow = []
#                         obj_bbx = []
#                         obj_bbox_R = []

#                         for ibbx in range(oth_v_loc.shape[0]):
#                             # bbox_v_rot = -1.0 * oth_v_rot[ibbx, :] + np.array([0.0, 0.0, 2.0 * np.pi])
#                             bbox_v_rot = lcl_rot_v[iv] - oth_v_rot[ibbx, :]
#                             bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(bbox_v_rot)
#                             obj_bbox_R.append(bbox_v_rot)

#                             bbox_extend = oth_v_bbx_ext[ibbx, :-1] * 2.0
#                             bbox_center = lcl_R[iv] @ (oth_v_loc[ibbx,:]- lcl_trans_v[iv])
#                             # print(bbox_center)
#                             obj_flow.append(np.hstack((np.array(oth_cars_info[iv][ibbx][0]), np.array(bbox_center))))
#                             bbox = o3d.geometry.OrientedBoundingBox(bbox_center, bbox_R, bbox_extend)
#                             obj_bbx.append(bbox)
#                             bbox.color = colors[iv]
#                             # bbox.LineSet = 
#                             vis_list.append(bbox)
#                         objs_flow.append(obj_flow)
#                         objs_bbx.append(obj_bbx)
#                         objs_bbox_R.append(obj_bbox_R)
                
#                     if len(objs_flow) <= 1:
#                         flow = arr_ - arr
#                     else:
#                         src_objs_flow = np.array(objs_flow[0])
#                         src_objs_bbox = dict(zip(src_objs_flow[:,0], objs_bbx[0]))
#                         src_objs_flow = dict(zip(src_objs_flow[:,0], src_objs_flow[:,1:]))
#                         src_bbox_R = dict(zip(src_objs_flow.keys(),objs_bbox_R[0]))                    
                    
#                         tgt_objs_flow = np.array(objs_flow[1])
#                         tgt_objs_bbox = dict(zip(tgt_objs_flow[:,0], objs_bbx[1]))
#                         tgt_objs_flow = dict(zip(tgt_objs_flow[:,0], tgt_objs_flow[:,1:]))
#                         tgt_bbox_R = dict(zip(tgt_objs_flow.keys(),objs_bbox_R[1]))

#                         objs_flow = []
#                         objs_c = np.array(list(src_objs_flow.values()))
#                         subarr_ = (objs_c @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T
#                         subflow = subarr_ - objs_c
#                         src_rigid_flow = dict(zip(src_objs_flow.keys(), subflow))
#                         objs_bbox_R = dict(zip(src_objs_flow.keys(),objs_bbox_R))
#                         for k,v in src_objs_flow.items():   
#                             if k in tgt_objs_flow:
#                                 obj_flow = tgt_objs_flow.get(k) - v
#                                 delta_flow = obj_flow - src_rigid_flow.get(k)
#                                 bbox = src_objs_bbox.get(k) 
#                                 inds = bbox.get_point_indices_within_bounding_box(pcd.points)
#                                 # flow[inds,:] = obj_flow
#                                 arr_[inds,:] += delta_flow
#                                 delat_rot = tgt_bbox_R.get(k) - src_bbox_R.get(k)
#                                 bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(delat_rot)
#                                 arr_[inds,:] = (arr_[inds,:] - tgt_objs_bbox.get(k).get_center()) @ bbox_R.T + tgt_objs_bbox.get(k).get_center()
#                             else:
#                                 obj_flow = np.zeros(3)
#                             # print(obj_flow)
                            
#                             objs_flow.append(obj_flow)
#                         # delta_flow = np.array(objs_flow) - subflow
#                         objs_flow = dict(zip(src_objs_flow.keys(), objs_flow))
#                         flow = arr_ - arr  
#                 # vis_list.append(pcd)
#                 pcd_cur2 = o3d.geometry.PointCloud()
#                 pcd_cur2.points = o3d.utility.Vector3dVector(arr_)
#                 pcd_cur2.paint_uniform_color([0,0,1.0])

                
#                 # vis_list.append(pcd)
#                 # vis_list.append(pcd_cur)
#                 # vis_list.append(pcd_cur2)
#                 # o3d.visualization.draw_geometries(vis_list)
#                 # save_view_point(vis_list, 'camera_view.json')
#                 if not(enable_animation):
#                     vis = o3d.visualization.Visualizer()
#                     vis.create_window(width=960*2, height=640*2, left=5, top=5)
#                     vis.get_render_option().background_color = np.array([0,0,0])
#                     # vis.get_render_option().background_color = np.array([1, 1, 1])
#                     vis.get_render_option().show_coordinate_frame = False
#                     vis.get_render_option().point_size = 1.0
#                     vis.get_render_option().line_width = 3.0
#                 # custom_draw_geometry(vis, vis_list, map_file=None, recording=False,param_file='camera_view.json', save_fov=True)
#                 use_flow = True
#                 if use_flow:
#                     # sf = sf[sample_idx1, :]
#                     np_list = [src_pc, tgt_pc, arr_, flow]
#                 else:
#                     np_list = [src_pc, tgt_pc, arr_,]
#                 # import os
#                 # print(os.path.abspath(os.curdir))
                
#                 np_list += [pcd_drivable_area]
#                 Draw_SceneFlow(vis, np_list + vis_list, use_fast_vis = not(use_flow), use_pred_t=False, use_custom_color=True, use_flow=use_flow, param_file='camera_view.json')
        
# #For building the whole Town_xx map with CARLA origin coordinate
# if __name__ == "__main__":
#     ind = 0
#     drivable_area = odom_utils.readPointCloud('./dataset/town02-map.bin')[:, :3]
#     pcd_drivable_area = o3d.geometry.PointCloud()
#     pcd_drivable_area.points = o3d.utility.Vector3dVector(drivable_area)
#     pcd_drivable_area.paint_uniform_color([1.0,0,0.0])
#     # Translate frame coordinates from left-hand to right-hand
#     v_l2r = np.array([1., -1., 1., 1., 1., 1.])
#     s_l2r = np.array([1, 1., -1., 1., 1., 1., 1.])
#     mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, -2.5])
#     for sub_dir in os.listdir(DATASET_PATH):
#         if 'global_label' in sub_dir or 'img' in sub_dir:
#             continue 
#         # print(sub_dir)

#         SUB_DATASET = sub_dir #'vehicle.citroen.c3_271'
#         v_pattern = r"(?P<name>[vehicle]+.\w+.\w+)_(?P<v_id>\d+)"
#         v = re.findall(v_pattern, SUB_DATASET)[0]
#         DATASET_PC_PATH = DATASET_PATH + SUB_DATASET + '/velodyne/'
#         # RESULT_PATH = './results/' + SUB_DATASET
#         trans_pattern = v[0] + " " + v[1] + r" (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#             r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                 r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                     r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"
#         # lidar_pattern = r"sensor.lidar.ray_cast (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#         #     r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#         #         r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#         #             r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 " + v[1]
        
#         lidar_pattern = r"sensor.lidar.ray_cast_semantic (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#             r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                 r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                     r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 0 " + v[1]

#         # next_ind = ind + 2 
#         pc_files = glob.glob(os.path.join(DATASET_PC_PATH, '*.bin'))#os.listdir(DATASET_PC_PATH)
#         rawtrans_files = glob.glob(os.path.join(DATASET_RAWTRANS_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
#         pc_files.sort()
#         rawtrans_files.sort()
        
#         first_v_raw_trans = odom_utils.readRawData(rawtrans_files[ind], trans_pattern) * v_l2r
#         first_sensor_raw_trans = odom_utils.readRawData(rawtrans_files[ind], lidar_pattern) * s_l2r

#         src_pc = odom_utils.readPointCloud(pc_files[ind])[:, :3]
#         first_v_raw_trans[2] += 2.5
#         src_pc += first_v_raw_trans[:3]
#         src_R_vec = first_sensor_raw_trans[4:]
#         src_R = o3d.geometry.get_rotation_matrix_from_xyz(src_R_vec)
#         src_R = np.linalg.inv(src_R)

#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(src_pc)#
#         pcd.rotate(src_R, first_v_raw_trans[:3])
#         pcd.paint_uniform_color([0,1,0])
#         vis_list = [mesh_frame, pcd]
#         vis_list += [pcd_drivable_area]
        

#         for next_ind in range(ind, len(pc_files), 2):
#             tgt_v_raw_trans = odom_utils.readRawData(rawtrans_files[next_ind], trans_pattern) * v_l2r
#             tgt_sensor_raw_trans = odom_utils.readRawData(rawtrans_files[next_ind], lidar_pattern) * s_l2r
            
#             tgt_pc = odom_utils.readPointCloud(pc_files[next_ind])[:, :3]
#             tgt_v_raw_trans[2] += 2.5
#             tgt_pc += tgt_v_raw_trans[:3]
#             tgt_R_vec = tgt_sensor_raw_trans[4:]
#             tgt_R = o3d.geometry.get_rotation_matrix_from_xyz(tgt_R_vec)
#             tgt_R = np.linalg.inv(tgt_R)
            
#             pcd_cur = o3d.geometry.PointCloud()
#             pcd_cur.points = o3d.utility.Vector3dVector(tgt_pc)
#             pcd_cur.rotate(tgt_R, tgt_v_raw_trans[:3])
#             # pcd_cur.paint_uniform_color([0,0,1])
            
#             vis_list.append(pcd_cur)
#             pcd = pcd + pcd_cur


#         o3d.visualization.draw_geometries(vis_list)
#         if not os.path.exists(RESULT_PATH):
#             os.makedirs(RESULT_PATH)

#         o3d.io.write_point_cloud(os.path.join(RESULT_PATH,SUB_DATASET + ".pcd"), pcd)


## For Drivable Area Map with Active Scene Flow
# if __name__ == "__main__":
#     start = 5
#     drivable_areas = odom_utils.readPointCloud('./dataset/town02-map.bin') #road map
#     drivable_area = drivable_areas[:,:3]
#     road_id = drivable_areas[:,-1]
#     colors = []
#     for iCnt in road_id:
#         if iCnt == 0:
#             colors += [[1,0,0]]
#         elif iCnt == 1:
#             colors += [[0,1,0]]
#         elif iCnt == 2:
#             colors += [[0,0,1]]
#         elif iCnt == 3:
#             colors += [[1,1,0]]
#         else:
#             colors += [[1,0,1]]
#     #指定实验道路
#     iou_id = np.argwhere(road_id == 1)

#     drivable_centers = np.load('./dataset/np_c.npy')#odom_utils.readPointCloud('./dataset/town02-road-center.bin')
#     drivable_center = drivable_centers[:,:3]
#     drivable_center[:,1] *= -1.0
#     road_id2 = []
#     road_id2 = drivable_centers[:,-1]
#     ref_drivable_center_inds = np.argwhere(drivable_centers[:,-1] == 1)
#     colors2 = []
#     for iCnt in road_id2:
#         if iCnt < 1:
#             colors2 += [[1,0,0]]
#         elif iCnt >= 1  and iCnt < 2:
#             colors2 += [[0,1,0]]
#         elif iCnt >= 2 and iCnt < 3:
#             colors2 += [[0,0,1]]
#         elif iCnt >= 3 and iCnt < 4:
#             colors2 += [[1,1,0]]
#         else:
#             colors2 += [[1,0,1]]

    
#     # drivable_area[:,0] *= -1
#     pcd_drivable_area = o3d.geometry.PointCloud()
#     # pcd_drivable_area.points = o3d.utility.Vector3dVector(drivable_area)
#     drivable_area_bkp = drivable_area
#     pcd_drivable_area.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.dtype('f4')))

#     pcd_drivable_center = o3d.geometry.PointCloud()
#     # pcd_drivable_area.points = o3d.utility.Vector3dVector(drivable_center)
#     drivable_center_bkp = drivable_center
#     pcd_drivable_center.colors = o3d.utility.Vector3dVector(np.array(colors2, dtype=np.dtype('f4')))
                
#     # Compute_PairwiseSceneFlow(start, 5)

#     # Translate frame coordinates from left-hand to right-hand
#     v_l2r = np.array([1., -1., 1., 1., 1., 1.])
#     s_l2r = np.array([1, 1., -1., 1., 1., 1., 1.])
    
#     enable_animation = True
#     if enable_animation:
#         vis = o3d.visualization.Visualizer()
#         vis.create_window(width=960*2, height=640*2, left=5, top=5)
#         vis.get_render_option().background_color = np.array([0, 0, 0])
#         # vis.get_render_option().background_color = np.array([1, 1, 1])
#         vis.get_render_option().show_coordinate_frame = False
#         vis.get_render_option().point_size = 3.0
#         vis.get_render_option().line_width = 5.0
#     mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, -2.5])
    
#     # Generate the log data from the DIR:tmp/xxx/xxxxrgb_lable/xxxx 
#     build_lidar_fov = False

#     # Generate the scene flow data of the only camera field-of-view
#     use_rgb_fov = False
    
#     rm_ground = True
#     # trans_pattern = r"(?P<v_name>[vehicle]+.\w+.\w+) (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#     # r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#     #     r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#     #         r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"\
    
#     rawtrans_files = glob.glob(os.path.join(DATASET_RAWTRANS_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
#     rawtrans_files.sort()
#     # if not(use_rgb_fov):
#     #     for r in re.findall(trans_pattern, log):
#     #         # print(r[0])
#     #         det_car_info.append(int(float(r[0])))
#     # print(os.path.abspath(os.curdir))
#     sub_dir_cnt = 0
#     max_v = -100.0
#     for sub_dir in os.listdir(DATASET_PATH):
#         if 'global_label' in sub_dir or 'img' in sub_dir:
#             continue 
#         # print(sub_dir)

#         SUB_DATASET = sub_dir #'vehicle.tesla.cybertruck_266'#'vehicle.citroen.c3_271'

#         v_pattern = r"(?P<name>[vehicle]+.\w+.\w+)_(?P<v_id>\d+)"
#         v = re.findall(v_pattern, SUB_DATASET)[0]
#         DATASET_PC_PATH = DATASET_PATH + SUB_DATASET + '/velodyne/'
#         DATASET_LBL_PATH = DATASET_PATH + SUB_DATASET + '/label00/'
#         # RESULT_PATH = './results/' + SUB_DATASET
#         trans_pattern = v[0] + " " + v[1] + r" (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#             r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                 r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                     r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"
#         lidar_pattern = r"sensor.lidar.ray_cast (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#             r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                 r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                     r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 " + v[1]
        
#         lidar_semantic_pattern = r"sensor.lidar.ray_cast_semantic (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#             r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                 r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                     r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 0 " + v[1]

#         # next_ind = ind + 2
#         pc_files = glob.glob(os.path.join(DATASET_PC_PATH, '*.bin'))#os.listdir(DATASET_PC_PATH)
#         # rawtrans_files = glob.glob(os.path.join(DATASET_RAWTRANS_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
#         label_files = glob.glob(os.path.join(DATASET_LBL_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
#         # print("Current DIR:" + DATASET_PC_PATH)
#         pc_files.sort()
#         # rawtrans_files.sort()
#         label_files.sort()

#         total_cmd_num = len(pc_files)
#         frame_start,frame_end,frame_hz = 10,-5,1
#         spawn_points = []
#         cnt = 0
        
#         DATASET_SF_PATH = RESULT_PATH2 + '/SF2/' + "%02d"%(sub_dir_cnt) + '/'
#         if not os.path.exists(DATASET_SF_PATH):
#             os.makedirs(DATASET_SF_PATH)

#         if build_lidar_fov:
#             write_labels('record2021_0716_2146/', sub_dir, frame_start,frame_end, frame_hz, index=1)
#         else:
#             step = 1
#             for ind in range(start, len(pc_files)-step, step):
#                 next_ind = ind + step
#                 if next_ind >= len(label_files):
#                         break
#                 print("Current DIR:" + DATASET_PC_PATH + ' Frame:' +str(ind))
#                 # Compute_PairwiseSceneFlow(ind, next_ind, trans_pattern, lidar_pattern, DATASET_PC_PATH, DATASET_LBL_PATH)
#                 vis_list = []
#                 first_v_raw_trans = odom_utils.readRawData(rawtrans_files[ind], trans_pattern) * v_l2r
#                 first_sensor_raw_trans = odom_utils.readRawData(rawtrans_files[ind], lidar_semantic_pattern) * s_l2r

#                 src_pc = odom_utils.readPointCloud(pc_files[ind])[:, :3]
#                 if rm_ground:
#                     is_ground = np.logical_not(src_pc[:, 2] <= -2.4)
#                     src_pc = src_pc[is_ground, :] 
#                 # np.save('./data/np_src', src_pc)
#                 # For the carla origin coordatinate 
#                 # first_v_raw_trans[2] += 2.5
#                 # src_pc += first_v_raw_trans[:3] @ np.array([[0,-1,0],[1,0,0],[0,0,1]])
#                 drivable_area = (drivable_area_bkp - first_v_raw_trans[:3])\
#                      @ odom_utils.rotation_from_euler_zyx(-first_v_raw_trans[5], -first_v_raw_trans[4],first_v_raw_trans[3])#
#                 pcd_drivable_area.points = o3d.utility.Vector3dVector(drivable_area)
#                 # np.save('./data/np_drivable_area', np.hstack((drivable_area, (drivable_areas[:,-1]).reshape((-1,1)))))
                

#                 drivable_center = (drivable_center_bkp - first_v_raw_trans[:3])\
#                      @ odom_utils.rotation_from_euler_zyx(-first_v_raw_trans[5], -first_v_raw_trans[4],first_v_raw_trans[3])#
#                 pcd_drivable_center.points = o3d.utility.Vector3dVector(drivable_center)
#                 # pcd_drivable_area.paint_uniform_color([1.0,0,0.0])
#                 ref_drivable_center = drivable_centers[ref_drivable_center_inds[:,0],:3]

#                 # np.save('./data/np_drivable_center', np.hstack((drivable_center, (drivable_centers[:,-1]).reshape((-1,1)))))

#                 arr = src_pc
                
#                 src_R_vec = first_sensor_raw_trans[4:]
#                 src_R = o3d.geometry.get_rotation_matrix_from_xyz(src_R_vec)
#                 src_R_inv = np.linalg.inv(src_R)
#                 pcd = o3d.geometry.PointCloud()
#                 pcd.points = o3d.utility.Vector3dVector(src_pc)
#                 #For the carla origin coordatinate 
#                 # pcd.rotate(src_R_inv, first_v_raw_trans[:3])

#                 # For the first frame as the origin coordinate
#                 # pcd.rotate(src_R_inv, np.zeros(3))
#                 pcd.paint_uniform_color([0,1,0])

#                 # drivable_area =  drivable_area - (np.array([[spawn_points[ind][0], -spawn_points[ind][1], 0]]))
                
#                 vis_list = [mesh_frame, pcd, pcd_drivable_area, pcd_drivable_center]
#                 # vis_list = [mesh_frame, pcd]
                
#                 tgt_v_raw_trans = odom_utils.readRawData(rawtrans_files[next_ind], trans_pattern) * v_l2r
#                 tgt_sensor_raw_trans = odom_utils.readRawData(rawtrans_files[next_ind], lidar_semantic_pattern) * s_l2r
                
#                 tgt_pc = odom_utils.readPointCloud(pc_files[next_ind])[:, :3]
#                 if rm_ground:
#                     is_ground = np.logical_not(tgt_pc[:, 2] <= -2.4)
#                     tgt_pc = tgt_pc[is_ground, :] 

#                 #For the carla origin coordatinate 
#                 # tgt_v_raw_trans[2] += 2.5
#                 # tgt_pc += tgt_v_raw_trans[:3]

#                 tgt_R_vec = tgt_sensor_raw_trans[4:]
#                 tgt_R = o3d.geometry.get_rotation_matrix_from_xyz(tgt_R_vec)
#                 tgt_R_inv = np.linalg.inv(tgt_R)
#                 pcd_cur = o3d.geometry.PointCloud()
#                 pcd_cur.points = o3d.utility.Vector3dVector(tgt_pc)

#                 #For the carla origin coordatinate 
#                 # pcd_cur.rotate(tgt_R_inv, tgt_v_raw_trans[:3] )

#                 #For the first frame as the origin coordinate
#                 # pcd_cur.rotate(tgt_R_inv, np.zeros(3))
#                 # pcd_cur.translate(tgt_v_raw_trans[:3] - first_v_raw_trans[:3])
#                 pcd_cur.paint_uniform_color([1,0,0])
                
#                 arr_ = (src_pc @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T

#                 oth_cars_info = []
#                 for i in [ind, next_ind]:
#                     # if next_ind >= len(label_files):
#                     #     break
#                     with open(label_files[i], 'r') as file:
#                             log = file.read()
#                     det_car_info = []  
#                     if use_rgb_fov:
#                         for r in re.findall(rgb_label_pattern, log):
#                             det_car_info.append(int(r[-1]))
#                     # else:
#                     #     for r in re.findall(lidar_label_pattern, log):
#                     #         # print(r[0])
#                     #         det_car_info.append(int(float(r[0])))
                    
#                     with open(rawtrans_files[i], 'r') as file:
#                         raw_log = file.read()
#                     oth_car_info = []
#                     for r in re.findall(glb_label_pattern, raw_log):
#                         # print(r[0])
#                         if use_rgb_fov:
#                             if int(r[1]) in det_car_info:
#                                 oth_car_info.append([int(r[1])] + list(map(float, r[2:5])) + [np.radians(float(r[5])),\
#                                     np.radians(float(r[6])), np.radians(float(r[7]))] + list(map(float, r[-4:])))
#                         else:
#                             if r[1] != v[1]:
#                                 oth_car_info.append([int(r[1])] + list(map(float, r[2:5])) + [np.radians(float(r[5])),\
#                                     np.radians(float(r[6])), np.radians(float(r[7]))] + list(map(float, r[-4:])))
#                     if len(oth_car_info) > 0:
#                         oth_cars_info.append(oth_car_info)
#                     # print(oth_car_info)

#                 # print(first_v_raw_trans)
#                 # #######TEST FOR INVERSE TRANSFORM TO CARLR COORDINATE
#                 # inv_ego_pos = first_v_raw_trans[:3]
#                 # print(inv_ego_pos)
                    
#                 show_flag = 1
#                 lcl_R = [src_R, tgt_R]
#                 lcl_R_inv = [src_R_inv, tgt_R_inv]
#                 lcl_rot_v = [first_v_raw_trans[3:], tgt_v_raw_trans[3:]]
#                 lcl_trans_v = [first_v_raw_trans[:3], tgt_v_raw_trans[:3]]
#                 colors = [[0,1,0], [1,0,0]]
#                 objs_center = []
#                 objs_flow = []
#                 objs_bbx = []
#                 objs_bbox_R = []
#                 # if show_flag:
#                 for iv in range(len(oth_cars_info)):
#                     if len(oth_cars_info) <= 1:
#                         flow = arr_ - arr
#                         # continue
#                     else:
#                         oth_v_loc = np.array(oth_cars_info[iv])[:,1:4] * np.array([1.0, -1.0, 1.0])
#                         oth_v_loc[:,-1] = oth_v_loc[:,-1] + np.array(oth_cars_info[iv])[:,-1] - 2.5
#                         oth_v_rot = np.array(oth_cars_info[iv])[:,4:7]
#                         oth_v_bbx_ext = np.array(oth_cars_info[iv])[:,-4:]

#                         # objs_center = []
#                         obj_flow = []
#                         obj_bbx = []
#                         obj_bbox_R = []

#                         for ibbx in range(oth_v_loc.shape[0]):
#                             # bbox_v_rot = -1.0 * oth_v_rot[ibbx, :] + np.array([0.0, 0.0, 2.0 * np.pi])
#                             bbox_v_rot = lcl_rot_v[iv] - oth_v_rot[ibbx, :]
#                             bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(bbox_v_rot)
#                             #For the local (observation) coordinates
#                             obj_bbox_R.append(bbox_v_rot)
                       
#                             bbox_extend = oth_v_bbx_ext[ibbx, :-1] * 2.0
#                             #For the local (observation) coordinates
#                             bbox_center = lcl_R[iv] @ (oth_v_loc[ibbx,:] - lcl_trans_v[iv])
#                             #For the original coordinates
#                             # bbox_center = lcl_R[iv] @ oth_v_loc[ibbx,:]
#                             # print(bbox_center)
#                             obj_flow.append(np.hstack((np.array(oth_cars_info[iv][ibbx][0]), np.array(bbox_center))))
#                             #For the local (observation) coordinates
#                             bbox = o3d.geometry.OrientedBoundingBox(bbox_center, bbox_R, bbox_extend)
#                             #For the original coordinates
#                             # bbox = o3d.geometry.OrientedBoundingBox(bbox_center, bbox_R, bbox_extend)
#                             obj_bbx.append(bbox)
#                             bbox.color = colors[iv]
#                             # bbox.LineSet = 
#                             vis_list.append(bbox)
#                         objs_flow.append(obj_flow)
#                         objs_bbx.append(obj_bbx)
#                         objs_bbox_R.append(obj_bbox_R)
                
#                     if len(objs_flow) <= 1:
#                         flow = arr_ - arr
#                     else:
#                         src_objs_flow = np.array(objs_flow[0])
#                         src_objs_bbox = dict(zip(src_objs_flow[:,0], objs_bbx[0]))
#                         src_objs_flow = dict(zip(src_objs_flow[:,0], src_objs_flow[:,1:]))
#                         src_bbox_R = dict(zip(src_objs_flow.keys(),objs_bbox_R[0]))                    
                    
#                         tgt_objs_flow = np.array(objs_flow[1])
#                         tgt_objs_bbox = dict(zip(tgt_objs_flow[:,0], objs_bbx[1]))
#                         tgt_objs_flow = dict(zip(tgt_objs_flow[:,0], tgt_objs_flow[:,1:]))
#                         tgt_bbox_R = dict(zip(tgt_objs_flow.keys(),objs_bbox_R[1]))

#                         objs_flow = []
#                         objs_c = np.array(list(src_objs_flow.values()))
#                         subarr_ = (objs_c @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T
#                         subflow = subarr_ - objs_c
#                         src_rigid_flow = dict(zip(src_objs_flow.keys(), subflow))
#                         objs_bbox_R = dict(zip(src_objs_flow.keys(),objs_bbox_R))
#                         for k,v in src_objs_flow.items():   
#                             if k in tgt_objs_flow:
#                                 obj_flow = tgt_objs_flow.get(k) - v
#                                 delta_flow = obj_flow - src_rigid_flow.get(k)
#                                 bbox = src_objs_bbox.get(k) 
#                                 inds = bbox.get_point_indices_within_bounding_box(pcd.points)
#                                 # flow[inds,:] = obj_flow
#                                 arr_[inds,:] += delta_flow
#                                 delat_rot = tgt_bbox_R.get(k) - src_bbox_R.get(k)
#                                 bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(delat_rot)
#                                 arr_[inds,:] = (arr_[inds,:] - tgt_objs_bbox.get(k).get_center()) @ bbox_R.T + tgt_objs_bbox.get(k).get_center()
#                             else:
#                                 obj_flow = np.zeros(3)
#                             # print(obj_flow)
                            
#                             objs_flow.append(obj_flow)
#                         # delta_flow = np.array(objs_flow) - subflow
#                         objs_flow = dict(zip(src_objs_flow.keys(), objs_flow))
#                         flow = arr_ - arr  
#                 # vis_list.append(pcd)
#                 pcd_cur2 = o3d.geometry.PointCloud()
#                 pcd_cur2.points = o3d.utility.Vector3dVector(arr_)
#                 pcd_cur2.paint_uniform_color([0,0,1.0])

#                 # vis_list.append(pcd)
#                 # vis_list.append(pcd_cur)
#                 # vis_list.append(pcd_cur2)
#                 # o3d.visualization.draw_geometries(vis_list)
#                 # save_view_point(vis_list, 'camera_view.json')
#                 if not(enable_animation):
#                     vis = o3d.visualization.Visualizer()
#                     vis.create_window(width=960*2, height=640*2, left=5, top=5)
#                     vis.get_render_option().background_color = np.array([0,0,0])
#                     # vis.get_render_option().background_color = np.array([1, 1, 1])
#                     vis.get_render_option().show_coordinate_frame = False
#                     vis.get_render_option().point_size = 3.0
#                     vis.get_render_option().line_width = 5.0
#                 # custom_draw_geometry(vis, vis_list, map_file=None, recording=False,param_file='camera_view.json', save_fov=True)
#                 use_flow = True
#                 ego_flow = (first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T * np.ones(src_pc.shape)
#                 name_fmt = "%06d"%(ind)
#                 print(ego_flow[0,:3])
#                 if abs(ego_flow[0,0]) > max_v:
#                     max_v = abs(ego_flow[0,0])
#                 # np.savez(DATASET_SF_PATH + name_fmt, pos1=src_pc, pos2=tgt_pc, ego_flow=ego_flow, gt=flow) 
#                 if use_flow:
#                     # sf = sf[sample_idx1, :]
#                     np_list = [src_pc, tgt_pc, arr_, flow]
#                 else:
#                     np_list = [src_pc, tgt_pc, arr_,]
                
#                 # import os
#                 # print(os.path.abspath(os.curdir))
#                 src_pc2 = src_pc - np.mean(src_pc, axis=0)
#                 normp = np.sqrt(np.sum(src_pc2 * src_pc2, axis=1))
#                 diameter = np.max(normp)
#                 print("Define parameters used for hidden_point_removal")
                
#                 # camera = [-1.0, 0.0, 5]
#                 # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3).translate(camera)
#                 # print("Get all points that are visible from given view point")
#                 # radius = diameter * 10
#                 # _, pt_map = pcd3.hidden_point_removal(camera, radius)
#                 # pc = np.vstack([pc1, pc2])
#                 # camera = [-45.0, -20.0, drivable_area[0,2]]
#                 # camera = [45.0, -60.0, drivable_area[0,2]]
#                 # camera = [-35.0, -20.0, drivable_area[0,2]]
#                 # camera = [75.0, 1.5, drivable_area[0,2]]
#                 # arr2_ = arr_ + flow
#                 # pt_map = HPR(tgt, np.array([[0, 0, -2.5]]), 3)
#                 # np_arr2 = arr2_[pt_map,:3]
#                 # # print("Visualize result")
#                 # # pcd3 = pcd3.select_by_index(pt_map)
#                 # bool_in = in_convex_polyhedron(arr2_[pt_map,:3],np.array([[-5.0, 4.0, -2.5]]), vis) 
#                 # pcd3 = drivable_area[pt_map,:]
#                 # bool_in = in_convex_polyhedron(drivable_area[iou_id[:,0],:3], np.array([[0, 0, -2.5]]), vis) 
#                 # camera_pos = np.hstack((first_v_raw_trans[:2], -2.75)).reshape(1,3)
#                 # bool_in = in_convex_polyhedron(drivable_area[iou_id[:,0],:3],camera_pos, vis) 
#                 global bbx
#                 # bbx = np.array([[0,0,0], [4,0,0], [0,-4,0], [4,-1,0], [0,0,-2.5], [4,0,-2.5], [0,-4,-2.5], [4,-4,-2.5]]) * 2.0
#                 bbx = np.array([[0,0,0], [4,0,0], [0,-1,0], [4,-1,0], [0,0,-2.5], [4,0,-2.5], [0,-1,-2.5], [4,-1,-2.5]]) * 0.8
#                 # bbx = np.array([[0,0,0], [1,0,0], [0,-1,0], [1,-1,0], [0,0,-2.5], [1,0,-2.5], [0,-1,-2.5], [1,-1,-2.5]]) 
#                 # bool_in, _ = validDrivableArea(bbx, tgt_pc)
#                 bool_in = validateDrivableArea2(bbx, tgt_pc)

#                 sphere = o3d.geometry.TriangleMesh.create_sphere(2.5).translate(np.array([0, 0, -2.5]))
#                 if bool_in:
#                     sphere.paint_uniform_color([0.2,0.2,1.0])
#                 else:
#                     sphere.paint_uniform_color([1,0,0])
#                 vis_list += [sphere]

#                 # np_list = vis_list
#                 use_flow = False
#                 Draw_SceneFlow(vis, np_list+vis_list, use_fast_vis = not(use_flow), use_pred_t=False, use_custom_color=True, use_flow=use_flow, param_file='camera_view.json')
#         sub_dir_cnt += 1
def remove_EgoVehicle_points(pts, bbx):
    bool_in, inds = validateDrivableArea2(bbx, pts)
    if bool_in:
        valid_inds = np.ones([pts.shape[0],1])
        valid_inds[inds] = 0
        pts = pts[inds,:]
    return pts

# def Generate_Random_SceneFlow(start = 1, step=1, prefixed_V_name=None, show_active_egoV=False, save_pts=False, enable_animation = True, egoV_flag=False, reverse=False, egoV_file_path = './data/data_Vinfo0307'):
#     if step == 1:
#         SF_DIR = '/SF/'
#     elif step == 2:
#         SF_DIR = '/SF2/'
#     elif step == 3:
#         SF_DIR = '/SF3/'
#     else:
#         SF_DIR = 'TEST'
    
#     start = start
#     if reverse:
#         scale = -1
#     else:
#         scale = 1
#     drivable_areas = odom_utils.readPointCloud('./dataset/town02-map.bin') #road map
#     drivable_area = drivable_areas[:,:3]
#     road_id = drivable_areas[:,-1]
#     colors = []
#     for iCnt in road_id:
#         if iCnt == 0:
#             colors += [[1,0,0]]
#         elif iCnt == 1:
#             colors += [[0,1,0]]
#         elif iCnt == 2:
#             colors += [[0,0,1]]
#         elif iCnt == 3:
#             colors += [[1,1,0]]
#         else:
#             colors += [[1,0,1]]
#     #指定实验道路
#     iou_id = np.argwhere(road_id == 1)

#     drivable_centers = np.load('./dataset/np_c.npy')#odom_utils.readPointCloud('./dataset/town02-road-center.bin')
#     drivable_center = drivable_centers[:,:3]
#     drivable_center[:,1] *= -1.0
#     road_id2 = []
#     road_id2 = drivable_centers[:,-1]
#     ref_drivable_center_inds = np.argwhere(drivable_centers[:,-1] == 1)
#     colors2 = []
#     for iCnt in road_id2:
#         if iCnt < 1:
#             colors2 += [[1,0,0]]
#         elif iCnt >= 1  and iCnt < 2:
#             colors2 += [[0,1,0]]
#         elif iCnt >= 2 and iCnt < 3:
#             colors2 += [[0,0,1]]
#         elif iCnt >= 3 and iCnt < 4:
#             colors2 += [[1,1,0]]
#         else:
#             colors2 += [[1,0,1]]

    
#     # drivable_area[:,0] *= -1
#     pcd_drivable_area = o3d.geometry.PointCloud()
#     # pcd_drivable_area.points = o3d.utility.Vector3dVector(drivable_area)
#     drivable_area_bkp = drivable_area
#     pcd_drivable_area.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.dtype('f4')))

#     pcd_drivable_center = o3d.geometry.PointCloud()
#     # pcd_drivable_area.points = o3d.utility.Vector3dVector(drivable_center)
#     drivable_center_bkp = drivable_center
#     pcd_drivable_center.colors = o3d.utility.Vector3dVector(np.array(colors2, dtype=np.dtype('f4')))
                
#     # Compute_PairwiseSceneFlow(start, 5)

#     # Translate frame coordinates from left-hand to right-hand
#     v_l2r = np.array([1., -1., 1., 1., 1., 1.])
#     s_l2r = np.array([1, 1., -1., 1., 1., 1., 1.])
#     v2s = np.array([[0,-1,0],[1,0,0],[0,0,1]])

#     # file_name = "Town02_01_20_22_10_24_54_cmd_traj.npz"
#     # # file_name = "./Active_SF/Town03_07_15_21_23_12_39_cmd_traj.npz" 
#     # # file_name = "./Active_SF/Town03_07_13_21_21_53_00_cmd_traj.npz"
#     # # print(os.path.abspath(os.path.curdir))
#     # hist_data = np.load(file_name)
#     # vehicles = hist_data['vehicles']
#     # # self.vehicles = self.vehicles[:35,:]
#     # cmd_traj = hist_data['cmd_arr']
    
    
#     # enable_animation = True
#     if enable_animation:
#         vis = o3d.visualization.Visualizer()
#         vis.create_window(width=960*2, height=640*2, left=5, top=5)
#         vis.get_render_option().background_color = np.array([0, 0, 0])
#         # vis.get_render_option().background_color = np.array([1, 1, 1])
#         vis.get_render_option().show_coordinate_frame = False
#         vis.get_render_option().point_size = 3.0
#         vis.get_render_option().line_width = 5.0
#     mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, -2.5])
    
#     # Generate the log data from the DIR:tmp/xxx/xxxxrgb_lable/xxxx 
#     build_lidar_fov = False

#     # Generate the scene flow data of the only camera field-of-view
#     use_rgb_fov = False
    
#     rm_ground = False
#     # trans_pattern = r"(?P<v_name>[vehicle]+.\w+.\w+) (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#     # r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#     #     r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#     #         r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"\
    
#     rawtrans_files = glob.glob(os.path.join(DATASET_RAWTRANS_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
#     rawtrans_files.sort()
#     # if not(use_rgb_fov):
#     #     for r in re.findall(trans_pattern, log):
#     #         # print(r[0])
#     #         det_car_info.append(int(float(r[0])))
#     # print(os.path.abspath(os.curdir))
#     sub_dir_cnt = 0
#     max_v = -100.0
#     # v_info_cnt = 0
#     current_v_info = []
#     next_v_info = []
#     pred_v_info = []
#     v_pattern = r"vehicle.(?P<nonsense>[\s\S]+)_(?P<v_id>\d+)"
#     for sub_dir in os.listdir(DATASET_PATH):
#         if 'global_label' in sub_dir or 'img' in sub_dir:
#             continue 
#         # print(sub_dir)

#         if show_active_egoV:
#             SUB_DATASET = prefixed_V_name
#         else:
#             SUB_DATASET = sub_dir #'vehicle.tesla.cybertruck_266'#'vehicle.citroen.c3_271'
#         if sub_dir not in SUB_DATASET:
#             continue
#         v_pattern = r"[vehicle]+.(?P<name>\w+.\w+)_(?P<v_id>\d+)"
#         v = re.findall(v_pattern, SUB_DATASET)[0]
#         DATASET_PC_PATH = DATASET_PATH + SUB_DATASET + '/velodyne/'
#         DATASET_LBL_PATH = DATASET_PATH + SUB_DATASET + '/label00/'
#         # RESULT_PATH = './results/' + SUB_DATASET
#         trans_pattern = v[0] + " " + v[1] + r" (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#             r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                 r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                     r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"
#         lidar_pattern = r"sensor.lidar.ray_cast (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#             r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                 r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                     r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 " + v[1]
        
#         lidar_semantic_pattern = r"sensor.lidar.ray_cast_semantic (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#             r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                 r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                     r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 0 " + v[1]
#         glb_label_pattern2 = v[0] + " " + v[1] + r" (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#     r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#         r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#             r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<extbbx_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                 r"(?P<extbbx_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<extbbx_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
#                     r"(?P<bbx_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"
#         # next_ind = ind + 2
#         pc_files = glob.glob(os.path.join(DATASET_PC_PATH, '*.bin'))#os.listdir(DATASET_PC_PATH)
#         # rawtrans_files = glob.glob(os.path.join(DATASET_RAWTRANS_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
#         label_files = glob.glob(os.path.join(DATASET_LBL_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
#         # print("Current DIR:" + DATASET_PC_PATH)
#         pc_files.sort()
#         # rawtrans_files.sort()
#         label_files.sort()

#         total_cmd_num = len(pc_files)
#         frame_start,frame_end,frame_hz = 10,-5,1
#         spawn_points = []
#         cnt = 0

#         ego_v_info = re.findall(v_pattern, sub_dir)
#         with open(rawtrans_files[0], 'r') as file:
#                 raw_log = file.read()
#         r = re.findall(glb_label_pattern, raw_log)
#         for r in re.findall(glb_label_pattern, raw_log):
#             if ego_v_info[0][0] in r[0] and ego_v_info[0][1] in r[1]:
#                 l = float(r[8])
#                 w = float(r[9])
#                 h = float(r[10])
#                 egoV_bbox_extend = np.array([l, w, h+0.5]).reshape(3,1) * 2.0
        
        
#         # DATASET_SF_PATH = RESULT_PATH2 + '/SF/' + "%02d"%(sub_dir_cnt) + '/'
#         DATASET_SF_PATH = RESULT_PATH2 + SF_DIR + "%02d"%(sub_dir_cnt) + '/'
#         if not os.path.exists(DATASET_SF_PATH) and save_pts:
#             os.makedirs(DATASET_SF_PATH)

#         if build_lidar_fov:
#             write_labels('record2021_0716_2146/', sub_dir, frame_start,frame_end, frame_hz, index=1)
#         else:
#             # step = 1
#             for ind in range(start, len(pc_files)-step, step):
#                 next_ind = ind + step
#                 pred_ind  = ind - step
#                 if next_ind >= len(label_files):
#                         break
#                 if pred_ind < 0:
#                     break
#                 print("Current DIR:" + DATASET_PC_PATH + ' Frame:' +str(ind))
#                 # tmp = cmd_traj[ind, ]
#                 # saveVehicleInfo()
                
#                 # # For writting the vehicle info ln:1357-1558
#                 if egoV_flag:
#                     pred_v_raw_info = odom_utils.readRawData(rawtrans_files[pred_ind], glb_label_pattern2) 
#                     pred_v_raw_info[0:6] = pred_v_raw_info[0:6] * v_l2r
#                     pred_v_info.append([v[0], v[1]]+list(pred_v_raw_info))

#                     first_v_raw_info = odom_utils.readRawData(rawtrans_files[ind], glb_label_pattern2) 
#                     first_v_raw_info[0:6] = first_v_raw_info[0:6] * v_l2r
#                     first_v_raw_trans = odom_utils.readRawData(rawtrans_files[ind], trans_pattern) * v_l2r
#                     # first_sensor_raw_trans = odom_utils.readRawData(rawtrans_files[ind], lidar_semantic_pattern) * s_l2r
#                     current_v_info.append([v[0], v[1]]+list(first_v_raw_info))

#                     tgt_v_raw_info = odom_utils.readRawData(rawtrans_files[next_ind], glb_label_pattern2)
#                     tgt_v_raw_info[0:6] = tgt_v_raw_info[0:6] * v_l2r
#                     next_v_info.append([v[0], v[1]]+list(tgt_v_raw_info))
#                     # tgt_sensor_raw_trans = odom_utils.readRawData(rawtrans_files[next_ind], lidar_semantic_pattern) * s_l2r
#                     # ivFlag = 0
#                     # for i in [ind, next_ind]:
#                     #     # if next_ind >= len(label_files):
#                     #     #     break
#                     #     with open(label_files[i], 'r') as file:
#                     #             log = file.read()
#                     #     with open(rawtrans_files[i], 'r') as file:
#                     #         raw_log = file.read()
                
#                     #     for r in re.findall(glb_label_pattern, raw_log):
#                     #         # print(r)
#                     #         if ego_v_info[0][0] in r[0] and ego_v_info[0][1] in r[1]:
#                     #             if ivFlag % 2 == 0:
#                     #                 current_v_info.append(r)
#                     #             else:
#                     #                 next_v_info.append(r)
#                     #             ivFlag += 1

#                     continue
#                 # else:

#                 # Compute_PairwiseSceneFlow(ind, next_ind, trans_pattern, lidar_pattern, DATASET_PC_PATH, DATASET_LBL_PATH)
#                 vis_list = []
#                 first_v_raw_trans = odom_utils.readRawData(rawtrans_files[ind], trans_pattern) * v_l2r
#                 first_sensor_raw_trans = odom_utils.readRawData(rawtrans_files[ind], lidar_semantic_pattern) * s_l2r

#                 src_pc = odom_utils.readPointCloud(pc_files[ind])[:, :3]
#                 src_pc_bkp = src_pc + first_v_raw_trans[:3]
#                 if rm_ground:
#                     is_ground = np.logical_not(src_pc[:, 2] <= -2.4)
#                     src_pc = src_pc[is_ground, :] 
#                 # np.save('./data/np_src', src_pc)
#                 # For the carla origin coordatinate 
#                 # first_v_raw_trans[2] += 2.5
#                 # src_pc += first_v_raw_trans[:3] @ np.array([[0,-1,0],[1,0,0],[0,0,1]])
                
#                 # ## For the carla original coordinate
#                 # drivable_area = drivable_area_bkp @ np.array([[0,-1,0],[1,0,0],[0,0,1]])

#                 ## For the first car frame as the original coordinate
#                 drivable_area = (drivable_area_bkp - first_v_raw_trans[:3])\
#                      @ odom_utils.rotation_from_euler_zyx(-first_v_raw_trans[5], -first_v_raw_trans[4],first_v_raw_trans[3])#
#                 pcd_drivable_area.points = o3d.utility.Vector3dVector(drivable_area)
#                 # np.save('./data/np_drivable_area', np.hstack((drivable_area, (drivable_areas[:,-1]).reshape((-1,1)))))
                
#                 ## For the carla original coordinate
#                 # drivable_center = drivable_center_bkp @ np.array([[0,-1,0],[1,0,0],[0,0,1]])

#                 # For the first car frame as the original coordinate
#                 drivable_center = (drivable_center_bkp - first_v_raw_trans[:3])\
#                      @ odom_utils.rotation_from_euler_zyx(-first_v_raw_trans[5], -first_v_raw_trans[4],first_v_raw_trans[3])#
#                 pcd_drivable_center.points = o3d.utility.Vector3dVector(drivable_center)
#                 # pcd_drivable_area.paint_uniform_color([1.0,0,0.0])
#                 ref_drivable_center = drivable_centers[ref_drivable_center_inds[:,0],:3]

#                 # np.save('./data/np_drivable_center', np.hstack((drivable_center, (drivable_centers[:,-1]).reshape((-1,1)))))

#                 arr = src_pc
                
#                 src_R_vec = first_sensor_raw_trans[4:]
#                 src_R = o3d.geometry.get_rotation_matrix_from_xyz(src_R_vec)
#                 src_R_inv = np.linalg.inv(src_R)
#                 pcd = o3d.geometry.PointCloud()
#                 pcd.points = o3d.utility.Vector3dVector(src_pc)
#                 #For the carla origin coordatinate 
#                 # pcd.rotate(src_R_inv, first_v_raw_trans[:3])

#                 # For the first frame as the origin coordinate
#                 # pcd.rotate(src_R_inv, np.zeros(3))
#                 pcd.paint_uniform_color([0,1,0])

#                 # ego_pcd = o3d.geometry.PointCloud()
#                 # ego_pcd.points = o3d.utility.Vector3dVector(first_v_raw_trans[:3].reshape(-1,3))
#                 # ego_pcd.paint_uniform_color([0,1,1])
               
#                 # drivable_area =  drivable_area - (np.array([[spawn_points[ind][0], -spawn_points[ind][1], 0]]))
                
#                 vis_list = [mesh_frame, pcd_drivable_area, pcd_drivable_center]
#                 # vis_list = [mesh_frame, pcd]
                
#                 tgt_v_raw_trans = odom_utils.readRawData(rawtrans_files[next_ind], trans_pattern) * v_l2r
#                 tgt_sensor_raw_trans = odom_utils.readRawData(rawtrans_files[next_ind], lidar_semantic_pattern) * s_l2r
                
#                 tgt_pc = odom_utils.readPointCloud(pc_files[next_ind])[:, :3]
#                 tgt_pc_bkp = tgt_pc + tgt_v_raw_trans[:3]
#                 if rm_ground:
#                     is_ground = np.logical_not(tgt_pc[:, 2] <= -2.4)
#                     tgt_pc = tgt_pc[is_ground, :] 

#                 #For the carla origin coordatinate 
#                 # tgt_v_raw_trans[2] += 2.5
#                 # tgt_pc += tgt_v_raw_trans[:3] @ np.array([[0,-1,0],[1,0,0],[0,0,1]])

#                 tgt_R_vec = tgt_sensor_raw_trans[4:]
#                 tgt_R = o3d.geometry.get_rotation_matrix_from_xyz(tgt_R_vec)
#                 tgt_R_inv = np.linalg.inv(tgt_R)
#                 pcd_cur = o3d.geometry.PointCloud()
#                 pcd_cur.points = o3d.utility.Vector3dVector(tgt_pc)

#                 #For the carla origin coordatinate 
#                 # pcd_cur.rotate(tgt_R_inv, tgt_v_raw_trans[:3] )

#                 #For the first frame as the origin coordinate
#                 # pcd_cur.rotate(tgt_R_inv, np.zeros(3))
#                 # pcd_cur.translate(tgt_v_raw_trans[:3] - first_v_raw_trans[:3])
#                 pcd_cur.paint_uniform_color([1,0,0])
                
#                 arr_ = (src_pc @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T

#                 oth_cars_info = []
#                 for i in [ind, next_ind]:
#                     # if next_ind >= len(label_files):
#                     #     break
#                     with open(label_files[i], 'r') as file:
#                             log = file.read()
#                     det_car_info = []  
#                     if use_rgb_fov:
#                         for r in re.findall(rgb_label_pattern, log):
#                             det_car_info.append(int(r[-1]))
#                     # else:
#                     #     for r in re.findall(lidar_label_pattern, log):
#                     #         # print(r[0])
#                     #         det_car_info.append(int(float(r[0])))
                    
#                     with open(rawtrans_files[i], 'r') as file:
#                         raw_log = file.read()
#                     oth_car_info = []
#                     for r in re.findall(glb_label_pattern, raw_log):
#                         # print(r[0])
#                         if use_rgb_fov:
#                             if int(r[1]) in det_car_info:
#                                 oth_car_info.append([int(r[1])] + list(map(float, r[2:5])) + [np.radians(float(r[5])),\
#                                     np.radians(float(r[6])), np.radians(float(r[7]))] + list(map(float, r[-4:])))
#                         else:
#                             if r[1] != v[1]:
#                                 oth_car_info.append([int(r[1])] + list(map(float, r[2:5])) + [np.radians(float(r[5])),\
#                                     np.radians(float(r[6])), np.radians(float(r[7]))] + list(map(float, r[-4:])))
#                     if len(oth_car_info) > 0:
#                         oth_cars_info.append(oth_car_info)
#                     # print(oth_car_info)

#                 # print(first_v_raw_trans)
#                 # #######TEST FOR INVERSE TRANSFORM TO CARLR COORDINATE
#                 # inv_ego_pos = first_v_raw_trans[:3]
#                 # print(inv_ego_pos)
                    
#                 show_flag = 1
#                 lcl_R = [src_R, tgt_R]
#                 lcl_R_inv = [src_R_inv, tgt_R_inv]
#                 lcl_rot_v = [first_v_raw_trans[3:], tgt_v_raw_trans[3:]]
#                 lcl_trans_v = [first_v_raw_trans[:3], tgt_v_raw_trans[:3]]
#                 colors = [[0,1,0], [1,0,0]]
#                 objs_center = []
#                 objs_flow = []
#                 objs_bbx = []
#                 objs_bbox_R = []
#                 # fg_inds = []
#                 # if show_flag:
#                 for iv in range(len(oth_cars_info)):
#                     if len(oth_cars_info) <= 1:
#                         flow = arr_ - arr
#                         # continue
#                     else:
#                         oth_v_loc = np.array(oth_cars_info[iv])[:,1:4] * np.array([1.0, -1.0, 1.0])
#                         oth_v_loc[:,-1] = oth_v_loc[:,-1] + np.array(oth_cars_info[iv])[:,-1] - 2.5
#                         oth_v_rot = np.array(oth_cars_info[iv])[:,4:7]
#                         oth_v_bbx_ext = np.array(oth_cars_info[iv])[:,-4:]

#                         # objs_center = []
#                         obj_flow = []
#                         obj_bbx = []
#                         obj_bbox_R = []

#                         for ibbx in range(oth_v_loc.shape[0]):
#                             # bbox_v_rot = -1.0 * oth_v_rot[ibbx, :] + np.array([0.0, 0.0, 2.0 * np.pi])
#                             bbox_v_rot = lcl_rot_v[iv] - oth_v_rot[ibbx, :]
#                             bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(bbox_v_rot)
#                             #For the local (observation) coordinates
#                             obj_bbox_R.append(bbox_v_rot)
                       
#                             bbox_extend = oth_v_bbx_ext[ibbx, :-1] * 2.0
#                             #For the local (observation) coordinates
#                             bbox_center = lcl_R[iv] @ (oth_v_loc[ibbx,:] - lcl_trans_v[iv])
#                             #For the original coordinates
#                             # bbox_center = lcl_R[iv] @ oth_v_loc[ibbx,:]
#                             # print(bbox_center)
#                             obj_flow.append(np.hstack((np.array(oth_cars_info[iv][ibbx][0]), np.array(bbox_center))))
#                             #For the local (observation) coordinates
#                             bbox = o3d.geometry.OrientedBoundingBox(bbox_center, bbox_R, bbox_extend)
#                             #For the original coordinates
#                             # bbox = o3d.geometry.OrientedBoundingBox(bbox_center, bbox_R, bbox_extend)
#                             obj_bbx.append(bbox)
#                             bbox.color = colors[iv]
#                             # bbox.LineSet = 
#                             vis_list.append(bbox)
#                         objs_flow.append(obj_flow)
#                         objs_bbx.append(obj_bbx)
#                         objs_bbox_R.append(obj_bbox_R)
                
#                     if len(objs_flow) <= 1:
#                         flow = arr_ - arr
#                     else:
#                         src_objs_flow = np.array(objs_flow[0])
#                         src_objs_bbox = dict(zip(src_objs_flow[:,0], objs_bbx[0]))
#                         src_objs_flow = dict(zip(src_objs_flow[:,0], src_objs_flow[:,1:]))
#                         src_bbox_R = dict(zip(src_objs_flow.keys(),objs_bbox_R[0]))                    
                    
#                         tgt_objs_flow = np.array(objs_flow[1])
#                         tgt_objs_bbox = dict(zip(tgt_objs_flow[:,0], objs_bbx[1]))
#                         tgt_objs_flow = dict(zip(tgt_objs_flow[:,0], tgt_objs_flow[:,1:]))
#                         tgt_bbox_R = dict(zip(tgt_objs_flow.keys(),objs_bbox_R[1]))

#                         objs_flow = []
#                         objs_c = np.array(list(src_objs_flow.values()))
#                         subarr_ = (objs_c @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T
#                         subflow = subarr_ - objs_c
#                         src_rigid_flow = dict(zip(src_objs_flow.keys(), subflow))
#                         objs_bbox_R = dict(zip(src_objs_flow.keys(),objs_bbox_R))
#                         for k,v in src_objs_flow.items():   
#                             if k in tgt_objs_flow:
#                                 obj_flow = tgt_objs_flow.get(k) - v
#                                 delta_flow = obj_flow - src_rigid_flow.get(k)
#                                 bbox = src_objs_bbox.get(k) 
#                                 inds = bbox.get_point_indices_within_bounding_box(pcd.points)
#                                 # flow[inds,:] = obj_flow
#                                 arr_[inds,:] += delta_flow
#                                 delat_rot = tgt_bbox_R.get(k) - src_bbox_R.get(k)
#                                 bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(delat_rot)
#                                 arr_[inds,:] = (arr_[inds,:] - tgt_objs_bbox.get(k).get_center()) @ bbox_R.T + tgt_objs_bbox.get(k).get_center()
#                                 # fg_inds.append(inds)
#                             else:
#                                 obj_flow = np.zeros(3)
#                             # print(obj_flow)
                            
#                             objs_flow.append(obj_flow)
#                         # delta_flow = np.array(objs_flow) - subflow
#                         objs_flow = dict(zip(src_objs_flow.keys(), objs_flow))
#                         flow = 1.0 * (arr_ - arr)
#                 # vis_list.append(pcd)
#                 pcd_cur2 = o3d.geometry.PointCloud()
#                 pcd_cur2.points = o3d.utility.Vector3dVector(arr_)
#                 pcd_cur2.paint_uniform_color([0,0,1.0])

#                 # concat_fg_inds = []
#                 # for item in enumerate(fg_inds):
#                 #     if len(item) > 0:
#                 #         concat_fg_inds += item[1]

#                 # vis_list.append(pcd)
#                 # vis_list.append(pcd_cur)
#                 # vis_list.append(pcd_cur2)
#                 # o3d.visualization.draw_geometries(vis_list)
#                 # save_view_point(vis_list, 'camera_view.json')
#                 if not(enable_animation):
#                     vis = o3d.visualization.Visualizer()
#                     vis.create_window(width=960*2, height=640*2, left=5, top=5)
#                     vis.get_render_option().background_color = np.array([0,0,0])
#                     # vis.get_render_option().background_color = np.array([1, 1, 1])
#                     vis.get_render_option().show_coordinate_frame = False
#                     vis.get_render_option().point_size = 3.0
#                     vis.get_render_option().line_width = 5.0
#                 # custom_draw_geometry(vis, vis_list, map_file=None, recording=False,param_file='camera_view.json', save_fov=True)
#                 use_flow = True
#                 flow = flow * 1.0
#                 # bg_flag = np.ones([src_pc.shape[0],1])
#                 # bg_flag[concat_fg_inds] = -1
#                 # bg_flag2 = np.argwhere(bg_flag==1)
#                 # ego_flow =  np.zeros(src_pc.shape)
#                 # ego_flow[bg_flag2[:,0],:] = flow[bg_flag2[:,0],:]
#                 ego_flow = (src_pc @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T - src_pc
#                 # ego_flow = (first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T * np.ones(src_pc.shape)
#                 # delta_R = o3d.geometry.get_rotation_matrix_from_xyz(first_v_raw_trans[3:] - tgt_v_raw_trans[3:]) 
#                 # ego_flow = (tgt_v_raw_trans[:3] - first_v_raw_trans[:3] ) @ delta_R.T @ v2s * np.ones(src_pc.shape)
#                 # ego_flow = (first_sensor_raw_trans[:3] - tgt_sensor_raw_trans[:3]) @ v2s * np.ones(src_pc.shape)
#                 # ego_flow = (-tgt_v_raw_trans[:3] + first_v_raw_trans[:3]) @ v2s * np.ones(src_pc.shape)
#                 ego_flow_bkp = ego_flow + tgt_v_raw_trans[:3] - first_v_raw_trans[:3]
#                 flow_bkp = flow + tgt_v_raw_trans[:3] - first_v_raw_trans[:3] 
                
#                 # print(first_v_raw_trans)
#                 # print(tgt_v_raw_trans)
#                 # ## next_loc = first_loc - v @ tgt_R
#                 # pred_first_v_raw_trans = first_v_raw_trans[:3] - (first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T @ tgt_R 
#                 # print(pred_first_v_raw_trans)
#                 # # print(ego_flow_bkp[0,:3])
#                 # # print((tgt_v_raw_trans[:3] - first_v_raw_trans[:3])@v2s)
#                 # print(ego_flow[0,:3])
#                 # print(flow_bkp[0,:3])

#                 # ego_v_info = re.findall(v_pattern, sub_dir)
#                 # ivFlag = 0
#                 # for i in [ind, next_ind]:
#                 #     # if next_ind >= len(label_files):
#                 #     #     break
#                 #     with open(label_files[i], 'r') as file:
#                 #             log = file.read()
#                 #     with open(rawtrans_files[i], 'r') as file:
#                 #         raw_log = file.read()
               
#                 #     for r in re.findall(glb_label_pattern, raw_log):
#                 #         # print(r)
#                 #         if ego_v_info[0][0] in r[0] and ego_v_info[0][1] in r[1]:
#                 #             bbx = 
#                 #             if ivFlag % 2 == 0:
#                 #                 current_v_info.append(r)

#                 #             else:
#                 #                 next_v_info.append(r)
#                 #             ivFlag += 1
                
#                 bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(first_v_raw_trans[3:])
#                 ego_car_center = np.zeros([3,1])
#                 ego_car_center[2] = -0.4 * egoV_bbox_extend[2]
#                 ego_car_bbx_c = o3d.geometry.OrientedBoundingBox(ego_car_center, bbox_R, egoV_bbox_extend)
#                 vis_list.append(ego_car_bbx_c)
#                 inds = ego_car_bbx_c.get_point_indices_within_bounding_box(pcd.points)
#                 valid_inds = np.ones([src_pc.shape[0]])
#                 if len(inds) > 0:
#                     valid_inds[inds] = 0
#                     valid_inds = valid_inds.astype(bool)
#                     src_pc = src_pc[valid_inds,:]
#                     arr_ = arr_[valid_inds,:]
#                     ego_flow = ego_flow[valid_inds,:]
#                     flow = flow[valid_inds,:]

#                 bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(tgt_v_raw_trans[3:])
#                 ego_car_bbx_n = o3d.geometry.OrientedBoundingBox(ego_car_center, bbox_R, egoV_bbox_extend)
#                 vis_list.append(ego_car_bbx_n)
#                 inds = ego_car_bbx_n.get_point_indices_within_bounding_box(pcd_cur.points)
#                 valid_inds = np.ones([tgt_pc.shape[0]])
#                 if len(inds) > 0:
#                     valid_inds[inds] = 0
#                     valid_inds = valid_inds.astype(bool)
#                     tgt_pc = tgt_pc[valid_inds,:]
                

#                 name_fmt = "%06d"%(ind)
#                 if abs(ego_flow[0,0]) > max_v:
#                     max_v = abs(ego_flow[0,0])
#                 if save_pts:
#                     np.savez(DATASET_SF_PATH + name_fmt, pos1=src_pc, pos2=tgt_pc, ego_flow=ego_flow, gt=flow) 
#                 if use_flow:
#                     # sf = sf[sample_idx1, :]
#                     np_list = [src_pc, tgt_pc, arr_, flow]
#                     np_list = [src_pc, tgt_pc, arr_, ego_flow]   
#                     # np_list = [src_pc_bkp, tgt_pc_bkp, arr_ + tgt_v_raw_trans[:3], ego_flow_bkp]                 
#                     # np_list = [src_pc_bkp, tgt_pc_bkp, arr_ + tgt_v_raw_trans[:3], flow_bkp]
#                 else:
#                     np_list = [src_pc, tgt_pc, arr_,]
#                     # np_list = [src_pc_bkp, tgt_pc_bkp, arr_, flow]
                
#                 # import os
#                 # print(os.path.abspath(os.curdir))
#                 src_pc2 = src_pc - np.mean(src_pc, axis=0)
#                 normp = np.sqrt(np.sum(src_pc2 * src_pc2, axis=1))
#                 diameter = np.max(normp)
#                 print("Define parameters used for hidden_point_removal")
                
#                 # camera = [-1.0, 0.0, 5]
#                 # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3).translate(camera)
#                 # print("Get all points that are visible from given view point")
#                 # radius = diameter * 10
#                 # _, pt_map = pcd3.hidden_point_removal(camera, radius)
#                 # pc = np.vstack([pc1, pc2])
#                 # camera = [-45.0, -20.0, drivable_area[0,2]]
#                 # camera = [45.0, -60.0, drivable_area[0,2]]
#                 # camera = [-35.0, -20.0, drivable_area[0,2]]
#                 # camera = [75.0, 1.5, drivable_area[0,2]]
#                 # arr2_ = arr_ + flow
#                 # pt_map = HPR(tgt, np.array([[0, 0, -2.5]]), 3)
#                 # np_arr2 = arr2_[pt_map,:3]
#                 # # print("Visualize result")
#                 # # pcd3 = pcd3.select_by_index(pt_map)
#                 # bool_in = in_convex_polyhedron(arr2_[pt_map,:3],np.array([[-5.0, 4.0, -2.5]]), vis) 
#                 # pcd3 = drivable_area[pt_map,:]
#                 # bool_in = in_convex_polyhedron(drivable_area[iou_id[:,0],:3], np.array([[0, 0, -2.5]]), vis) 
#                 # camera_pos = np.hstack((first_v_raw_trans[:2], -2.75)).reshape(1,3)
#                 # bool_in = in_convex_polyhedron(drivable_area[iou_id[:,0],:3],camera_pos, vis) 
#                 global bbx
#                 # bbx = np.array([[0,0,0], [4,0,0], [0,-4,0], [4,-1,0], [0,0,-2.5], [4,0,-2.5], [0,-4,-2.5], [4,-4,-2.5]]) * 2.0
#                 bbx = np.array([[0,0,0], [4,0,0], [0,-1,0], [4,-1,0], [0,0,-2.5], [4,0,-2.5], [0,-1,-2.5], [4,-1,-2.5]]) * 0.8
#                 # bbx = np.array([[0,0,0], [1,0,0], [0,-1,0], [1,-1,0], [0,0,-2.5], [1,0,-2.5], [0,-1,-2.5], [1,-1,-2.5]]) 
#                 # bool_in, _ = validDrivableArea(bbx, tgt_pc)
#                 bool_in, _ = validateDrivableArea2(bbx, tgt_pc)

#                 sphere = o3d.geometry.TriangleMesh.create_sphere(3).translate(np.array([0, 0, -2.5]))
#                 if bool_in:
#                     sphere.paint_uniform_color([0.2,0.2,1.0])
#                 else:
#                     sphere.paint_uniform_color([1,0,0])
#                 vis_list += [sphere]

#                 # np_list = vis_list
#                 use_flow = False
#                 Draw_SceneFlow(vis, np_list+vis_list, use_fast_vis = not(use_flow), use_pred_t=True, use_custom_color=True, use_flow=use_flow, param_file='camera_view.json')
#         sub_dir_cnt += 1
#     if egoV_flag:
#         np.savez(egoV_file_path, cvInfo = current_v_info, nvInfo = next_v_info)

def Generate_Random_SceneFlow(start = 1, step=1, lidar_hight = 3.4, prefixed_V_name=None, show_active_egoV=False, save_pts=False, enable_animation = True, egoV_flag=False, rm_ground = False, show_fgrnds=True, egoV_file_path = './data/data_Vinfo0307'):
    if step == 1:
        SF_DIR = '/SF/'
    elif step == 2:
        SF_DIR = '/SF2/'
    elif step == 3:
        SF_DIR = '/SF3/'
    else:
        SF_DIR = 'TEST'
    
    start = start
    img_cnt = 0
    # show_fgrnds = True

    drivable_areas = odom_utils.readPointCloud('./dataset/town02-map.bin') #road map
    drivable_area = drivable_areas[:,:3]
    drivable_area[:,-1] = -1.0 * lidar_hight
    road_id = drivable_areas[:,-1]
    colors = []
    for iCnt in road_id:
        if iCnt == 0:
            colors += [[1,0,0]]
        elif iCnt == 1:
            colors += [[0,1,0]]
        elif iCnt == 2:
            colors += [[0,0,1]]
        elif iCnt == 3:
            colors += [[1,1,0]]
        else:
            colors += [[1,0,1]]
    #指定实验道路
    iou_id = np.argwhere(road_id == 1)

    drivable_centers = np.load('./dataset/np_c.npy')#odom_utils.readPointCloud('./dataset/town02-road-center.bin')
    drivable_center = drivable_centers[:,:3]
    drivable_center[:,-1] = -1.0 * lidar_hight
    drivable_center[:,1] *= -1.0
    road_id2 = []
    road_id2 = drivable_centers[:,-1]
    ref_drivable_center_inds = np.argwhere(drivable_centers[:,-1] == 1)
    colors2 = []
    for iCnt in road_id2:
        if iCnt < 1:
            colors2 += [[1,0,0]]
        elif iCnt >= 1  and iCnt < 2:
            colors2 += [[0,1,0]]
        elif iCnt >= 2 and iCnt < 3:
            colors2 += [[0,0,1]]
        elif iCnt >= 3 and iCnt < 4:
            colors2 += [[1,1,0]]
        else:
            colors2 += [[1,0,1]]

    
    # drivable_area[:,0] *= -1
    pcd_drivable_area = o3d.geometry.PointCloud()
    # pcd_drivable_area.points = o3d.utility.Vector3dVector(drivable_area)
    drivable_area_bkp = drivable_area
    pcd_drivable_area.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.dtype('f4')))

    pcd_drivable_center = o3d.geometry.PointCloud()
    # pcd_drivable_area.points = o3d.utility.Vector3dVector(drivable_center)
    drivable_center_bkp = drivable_center
    pcd_drivable_center.colors = o3d.utility.Vector3dVector(np.array(colors2, dtype=np.dtype('f4')))
                
    # Compute_PairwiseSceneFlow(start, 5)

    # Translate frame coordinates from left-hand to right-hand
    v_l2r = np.array([1., -1., 1., 1., 1., 1.])
    s_l2r = np.array([1, 1., -1., 1., 1., 1., 1.])
    v2s = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    # file_name = "Town02_01_20_22_10_24_54_cmd_traj.npz"
    # # file_name = "./Active_SF/Town03_07_15_21_23_12_39_cmd_traj.npz" 
    # # file_name = "./Active_SF/Town03_07_13_21_21_53_00_cmd_traj.npz"
    # # print(os.path.abspath(os.path.curdir))
    # hist_data = np.load(file_name)
    # vehicles = hist_data['vehicles']
    # # self.vehicles = self.vehicles[:35,:]
    # cmd_traj = hist_data['cmd_arr']
    
    
    # enable_animation = True
    if enable_animation:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=960*2, height=640*2, left=5, top=5)
        vis.get_render_option().background_color = np.array([0, 0, 0])
        # vis.get_render_option().background_color = np.array([1, 1, 1])
        vis.get_render_option().show_coordinate_frame = False
        vis.get_render_option().point_size = 3.0
        vis.get_render_option().line_width = 5.0
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, -1.0 * lidar_hight])
    
    # Generate the log data from the DIR:tmp/xxx/xxxxrgb_lable/xxxx 
    build_lidar_fov = False

    # Generate the scene flow data of the only camera field-of-view
    use_rgb_fov = False
    
    # rm_ground = False
    # trans_pattern = r"(?P<v_name>[vehicle]+.\w+.\w+) (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
    # r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
    #     r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
    #         r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"\
    
    rawtrans_files = glob.glob(os.path.join(DATASET_RAWTRANS_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
    rawtrans_files.sort()
    # if not(use_rgb_fov):
    #     for r in re.findall(trans_pattern, log):
    #         # print(r[0])
    #         det_car_info.append(int(float(r[0])))
    # print(os.path.abspath(os.curdir))
    sub_dir_cnt = 0
    max_v = -100.0
    # v_info_cnt = 0
    current_v_info = []
    next_v_info = []
    pred_v_info = []
    v_pattern = r"vehicle.(?P<nonsense>[\s\S]+)_(?P<v_id>\d+)"
    for sub_dir in os.listdir(DATASET_PATH):
        if 'global_label' in sub_dir or 'img' in sub_dir:
            continue 
        # print(sub_dir)

        if show_active_egoV:
            SUB_DATASET = prefixed_V_name
        else:
            SUB_DATASET = sub_dir #'vehicle.tesla.cybertruck_266'#'vehicle.citroen.c3_271'
        if sub_dir not in SUB_DATASET:
            continue
        v_pattern = r"[vehicle]+.(?P<name>\w+.\w+)_(?P<v_id>\d+)"
        v = re.findall(v_pattern, SUB_DATASET)[0]
        DATASET_PC_PATH = DATASET_PATH + SUB_DATASET + '/velodyne/'
        DATASET_LBL_PATH = DATASET_PATH + SUB_DATASET + '/label00/'
        # RESULT_PATH = './results/' + SUB_DATASET
        trans_pattern = v[0] + " " + v[1] + r" (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"
        lidar_pattern = r"sensor.lidar.ray_cast (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 " + v[1]
        
        lidar_semantic_pattern = r"sensor.lidar.ray_cast_semantic (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 0 " + v[1]
        glb_label_pattern2 = v[0] + " " + v[1] + r" (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
    r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
        r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<extbbx_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<extbbx_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<extbbx_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<bbx_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"
        # next_ind = ind + 2
        pc_files = glob.glob(os.path.join(DATASET_PC_PATH, '*.bin'))#os.listdir(DATASET_PC_PATH)
        # rawtrans_files = glob.glob(os.path.join(DATASET_RAWTRANS_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
        label_files = glob.glob(os.path.join(DATASET_LBL_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
        # print("Current DIR:" + DATASET_PC_PATH)
        pc_files.sort()
        # rawtrans_files.sort()
        label_files.sort()

        total_cmd_num = len(pc_files)
        frame_start,frame_end,frame_hz = 10,-5,1
        spawn_points = []
        cnt = 0

        ego_v_info = re.findall(v_pattern, sub_dir)
        with open(rawtrans_files[0], 'r') as file:
                raw_log = file.read()
        r = re.findall(glb_label_pattern, raw_log)
        for r in re.findall(glb_label_pattern, raw_log):
            if ego_v_info[0][0] in r[0] and ego_v_info[0][1] in r[1]:
                l = float(r[8])
                w = float(r[9])
                h = float(r[10])
                egoV_bbox_extend = np.array([l, w, h]).reshape(3,1) * 2.0
        
        
        # DATASET_SF_PATH = RESULT_PATH2 + '/SF/' + "%02d"%(sub_dir_cnt) + '/'
        DATASET_SF_PATH = RESULT_PATH2 + SF_DIR + "%02d"%(sub_dir_cnt) + '/'
        if not os.path.exists(DATASET_SF_PATH) and save_pts:
            os.makedirs(DATASET_SF_PATH)

        if build_lidar_fov:
            write_labels('record2021_0716_2146/', sub_dir, frame_start,frame_end, frame_hz, index=1)
        else:
            # step = 1
            for ind in range(start, len(pc_files)-step, step):
                raw_next_ind  = ind + step
                raw_pred_ind  = ind - step
                if raw_next_ind >= len(label_files):
                        break
                if raw_pred_ind < 0:
                    break
                print("Current DIR:" + DATASET_PC_PATH + ' Frame:' +str(ind))
                # tmp = cmd_traj[ind, ]
                # saveVehicleInfo()
                
                # # For writting the vehicle info ln:1357-1558
                if egoV_flag:
                    pred_v_raw_info = odom_utils.readRawData(rawtrans_files[raw_pred_ind], glb_label_pattern2) 
                    pred_v_raw_info[0:6] = pred_v_raw_info[0:6] * v_l2r
                    pred_v_info.append([v[0], v[1]]+list(pred_v_raw_info))

                    first_v_raw_info = odom_utils.readRawData(rawtrans_files[ind], glb_label_pattern2) 
                    first_v_raw_info[0:6] = first_v_raw_info[0:6] * v_l2r
                    first_v_raw_trans = odom_utils.readRawData(rawtrans_files[ind], trans_pattern) * v_l2r
                    # first_sensor_raw_trans = odom_utils.readRawData(rawtrans_files[ind], lidar_semantic_pattern) * s_l2r
                    current_v_info.append([v[0], v[1]]+list(first_v_raw_info))

                    tgt_v_raw_info = odom_utils.readRawData(rawtrans_files[raw_next_ind], glb_label_pattern2)
                    tgt_v_raw_info[0:6] = tgt_v_raw_info[0:6] * v_l2r
                    next_v_info.append([v[0], v[1]]+list(tgt_v_raw_info))
    
                    continue
                # else:

                # Compute_PairwiseSceneFlow(ind, next_ind, trans_pattern, lidar_pattern, DATASET_PC_PATH, DATASET_LBL_PATH)
                vis_list = []
                first_v_raw_trans = odom_utils.readRawData(rawtrans_files[ind], trans_pattern) * v_l2r
                first_sensor_raw_trans = odom_utils.readRawData(rawtrans_files[ind], lidar_semantic_pattern) * s_l2r

                raw_src_pc = odom_utils.readPointCloud(pc_files[ind])[:, :3]
                raw_src_pc_bkp = raw_src_pc + first_v_raw_trans[:3]
                if rm_ground:
                    is_ground = np.logical_not(raw_src_pc[:, 2] <= -1.0 * (lidar_hight-0.1))
                    raw_src_pc = raw_src_pc[is_ground, :] 
                # arr = src_pc
                # np.save('./data/np_src', src_pc)
                # For the carla origin coordatinate 
                # first_v_raw_trans[2] += 2.5
                # src_pc += first_v_raw_trans[:3] @ np.array([[0,-1,0],[1,0,0],[0,0,1]])
                
                # ## For the carla original coordinate
                # drivable_area = drivable_area_bkp @ np.array([[0,-1,0],[1,0,0],[0,0,1]])

                ## For the first car frame as the original coordinate
                drivable_area = (drivable_area_bkp - first_v_raw_trans[:3])\
                     @ odom_utils.rotation_from_euler_zyx(-first_v_raw_trans[5], -first_v_raw_trans[4],first_v_raw_trans[3])#
                pcd_drivable_area.points = o3d.utility.Vector3dVector(drivable_area)
                # np.save('./data/np_drivable_area', np.hstack((drivable_area, (drivable_areas[:,-1]).reshape((-1,1)))))
                
                ## For the carla original coordinate
                # drivable_center = drivable_center_bkp @ np.array([[0,-1,0],[1,0,0],[0,0,1]])

                # For the first car frame as the original coordinate
                drivable_center = (drivable_center_bkp - first_v_raw_trans[:3])\
                     @ odom_utils.rotation_from_euler_zyx(-first_v_raw_trans[5], -first_v_raw_trans[4],first_v_raw_trans[3])#
                pcd_drivable_center.points = o3d.utility.Vector3dVector(drivable_center)
                # pcd_drivable_area.paint_uniform_color([1.0,0,0.0])
                ref_drivable_center = drivable_centers[ref_drivable_center_inds[:,0],:3]

                # np.save('./data/np_drivable_center', np.hstack((drivable_center, (drivable_centers[:,-1]).reshape((-1,1)))))

                src_R_vec = first_sensor_raw_trans[4:]
                src_R = o3d.geometry.get_rotation_matrix_from_xyz(src_R_vec)
                src_R_inv = np.linalg.inv(src_R)
                tmp_pcd = o3d.geometry.PointCloud()
                tmp_pcd.points = o3d.utility.Vector3dVector(raw_src_pc)
                #For the carla origin coordatinate 
                # pcd.rotate(src_R_inv, first_v_raw_trans[:3])

                # For the first frame as the origin coordinate
                # pcd.rotate(src_R_inv, np.zeros(3))
                # tmp_pcd.paint_uniform_color([0,1,0])

                # ego_pcd = o3d.geometry.PointCloud()
                # ego_pcd.points = o3d.utility.Vector3dVector(first_v_raw_trans[:3].reshape(-1,3))
                # ego_pcd.paint_uniform_color([0,1,1])
               
                # drivable_area =  drivable_area - (np.array([[spawn_points[ind][0], -spawn_points[ind][1], 0]]))
                
                vis_list = [mesh_frame, pcd_drivable_area, pcd_drivable_center]
                # vis_list = [mesh_frame, pcd]

                raw_tgt_v_raw_trans = odom_utils.readRawData(rawtrans_files[raw_next_ind], trans_pattern) * v_l2r
                raw_tgt_sensor_raw_trans = odom_utils.readRawData(rawtrans_files[raw_next_ind], lidar_semantic_pattern) * s_l2r
                
                raw_tgt_pc = odom_utils.readPointCloud(pc_files[raw_next_ind])[:, :3]
                raw_tgt_pc_bkp = raw_tgt_pc + raw_tgt_v_raw_trans[:3]
                if rm_ground:
                    is_ground = np.logical_not(raw_tgt_pc[:, 2] <= -1.0 * (lidar_hight-0.1))
                    raw_tgt_pc = raw_tgt_pc[is_ground, :] 

                print(first_v_raw_trans[:3])
                print(raw_tgt_v_raw_trans[:3])
                #For the carla origin coordatinate 
                # tgt_v_raw_trans[2] += 2.5
                # tgt_pc += tgt_v_raw_trans[:3] @ np.array([[0,-1,0],[1,0,0],[0,0,1]])

                raw_tgt_R_vec = raw_tgt_sensor_raw_trans[4:]
                raw_tgt_R = o3d.geometry.get_rotation_matrix_from_xyz(raw_tgt_R_vec)
                raw_tgt_R_inv = np.linalg.inv(raw_tgt_R)
                tmp_pcd_cur = o3d.geometry.PointCloud()
                tmp_pcd_cur.points = o3d.utility.Vector3dVector(raw_tgt_pc)

                #For the carla origin coordatinate 
                # pcd_cur.rotate(tgt_R_inv, tgt_v_raw_trans[:3] )

                #For the first frame as the origin coordinate
                # pcd_cur.rotate(tgt_R_inv, np.zeros(3))
                # pcd_cur.translate(tgt_v_raw_trans[:3] - first_v_raw_trans[:3])
                # tmp_pcd_cur.paint_uniform_color([1,0,0])

                raw_tgt_v_raw_trans2 = odom_utils.readRawData(rawtrans_files[raw_pred_ind], trans_pattern) * v_l2r
                raw_tgt_sensor_raw_trans2 = odom_utils.readRawData(rawtrans_files[raw_pred_ind], lidar_semantic_pattern) * s_l2r
                
                raw_tgt_pc2 = odom_utils.readPointCloud(pc_files[raw_pred_ind])[:, :3]
                raw_tgt_pc_bkp2 = raw_tgt_pc2 + raw_tgt_v_raw_trans2[:3]
                if rm_ground:
                    is_ground = np.logical_not(raw_tgt_pc2[:, 2] <= -1.0 * (lidar_hight - 0.05))
                    raw_tgt_pc2 = raw_tgt_pc2[is_ground, :] 

                #For the carla origin coordatinate 
                # tgt_v_raw_trans[2] += 2.5
                # tgt_pc += tgt_v_raw_trans[:3] @ np.array([[0,-1,0],[1,0,0],[0,0,1]])

                raw_tgt_R_vec2 = raw_tgt_sensor_raw_trans2[4:]
                raw_tgt_R2 = o3d.geometry.get_rotation_matrix_from_xyz(raw_tgt_R_vec2)
                raw_tgt_R_inv2 = np.linalg.inv(raw_tgt_R2)
                pcd_pred = o3d.geometry.PointCloud()
                pcd_pred.points = o3d.utility.Vector3dVector(raw_tgt_pc2)

                #For the carla origin coordatinate 
                # pcd_cur.rotate(tgt_R_inv, tgt_v_raw_trans[:3] )

                #For the first frame as the origin coordinate
                # pcd_cur.rotate(tgt_R_inv, np.zeros(3))
                # pcd_cur.translate(tgt_v_raw_trans[:3] - first_v_raw_trans[:3])
                pcd_pred.paint_uniform_color([1,0,1])

                first_v_raw_rot = first_v_raw_trans[3:]
                # first_v_raw_rot[-1] *= -1.0
                src_ego_bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(first_v_raw_rot)
                # src_ego_bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(first_v_raw_trans[3:])
                ego_car_center = np.zeros([3,1])
                ego_car_center[2] = -1.0 * lidar_hight + 0.5 * egoV_bbox_extend[2]
                ego_car_bbx_c = o3d.geometry.OrientedBoundingBox(ego_car_center, src_ego_bbox_R, egoV_bbox_extend)
                # vis_list.append(ego_car_bbx_c)
                inds = ego_car_bbx_c.get_point_indices_within_bounding_box(tmp_pcd.points)
                valid_inds = np.ones([raw_src_pc.shape[0]])
                if len(inds) > 0:
                    valid_inds[inds] = 0
                    valid_inds = valid_inds.astype(bool)
                    raw_src_pc = raw_src_pc[valid_inds,:]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(raw_src_pc)
                #For the carla origin coordatinate 
                # pcd.rotate(src_R_inv, first_v_raw_trans[:3])

                # For the first frame as the origin coordinate
                # pcd.rotate(src_R_inv, np.zeros(3))
                pcd.paint_uniform_color([0,1,0])

                tgt_v_raw_rot = raw_tgt_v_raw_trans[3:]
                # first_v_raw_rot[-1] *= -1.0
                tgt_ego_bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(tgt_v_raw_rot)
                tgt_ego_bbx_center = (np.zeros([3]) @ src_R_inv.T + first_v_raw_trans[:3] - raw_tgt_v_raw_trans[:3]) @ raw_tgt_R.T
                tgt_ego_bbx_center[2] = -1.0 * lidar_hight + 0.5 * egoV_bbox_extend[2]
                tgt_ego_car_bbx_c = o3d.geometry.OrientedBoundingBox(tgt_ego_bbx_center, tgt_ego_bbox_R, egoV_bbox_extend)
                # vis_list.append(ego_car_bbx_c)
                tgt_inds = tgt_ego_car_bbx_c.get_point_indices_within_bounding_box(tmp_pcd_cur.points)
                tgt_valid_inds = np.ones([raw_tgt_pc.shape[0]])
                if len(tgt_inds) > 0:
                    tgt_valid_inds[tgt_inds] = 0
                    tgt_valid_inds = tgt_valid_inds.astype(bool)
                    raw_tgt_pc = raw_tgt_pc[tgt_valid_inds,:]
                pcd_cur = o3d.geometry.PointCloud()
                pcd_cur.points = o3d.utility.Vector3dVector(raw_tgt_pc)

                #For the carla origin coordatinate 
                # pcd_cur.rotate(tgt_R_inv, tgt_v_raw_trans[:3] )

                #For the first frame as the origin coordinate
                # pcd_cur.rotate(tgt_R_inv, np.zeros(3))
                # pcd_cur.translate(tgt_v_raw_trans[:3] - first_v_raw_trans[:3])
                pcd_cur.paint_uniform_color([1,0,0])

                ind_bkp = ind
                pre_ego_flow = []
                pre_flow = []
                ego_flow = []
                flow = []
                fgrnd_inds = np.zeros(raw_src_pc.shape[0])
                fg_inds = []
                fgrnd_inds_t = np.zeros(raw_tgt_pc.shape[0])
                fg_inds_t = []
                for i_ind, item in enumerate([raw_pred_ind, raw_next_ind]):
                    next_ind = item
                    if i_ind % 2 == 0:
                        tgt_v_raw_trans = raw_tgt_v_raw_trans2
                        tgt_sensor_raw_trans = raw_tgt_sensor_raw_trans2
                        tgt_R = raw_tgt_R2
                        tgt_R_inv = raw_tgt_R_inv2
                        tgt_pc = raw_tgt_pc2
                    else:
                        tgt_v_raw_trans = raw_tgt_v_raw_trans
                        tgt_sensor_raw_trans = raw_tgt_sensor_raw_trans
                        tgt_R = raw_tgt_R
                        tgt_R_inv = raw_tgt_R_inv
                        tgt_pc = raw_tgt_pc
                    
                    arr = raw_src_pc
                    src_pc = raw_src_pc
                    arr_ = (src_pc @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T

                    oth_cars_info = []
                    for i in [ind, next_ind]:
                        # if next_ind >= len(label_files):
                        #     break
                        with open(label_files[i], 'r') as file:
                                log = file.read()
                        det_car_info = []  
                        if use_rgb_fov:
                            for r in re.findall(rgb_label_pattern, log):
                                det_car_info.append(int(r[-1]))
                        # else:
                        #     for r in re.findall(lidar_label_pattern, log):
                        #         # print(r[0])
                        #         det_car_info.append(int(float(r[0])))
                        
                        with open(rawtrans_files[i], 'r') as file:
                            raw_log = file.read()
                        oth_car_info = []
                        for r in re.findall(glb_label_pattern, raw_log):
                            # print(r[0])
                            if use_rgb_fov:
                                if int(r[1]) in det_car_info:
                                    oth_car_info.append([int(r[1])] + list(map(float, r[2:5])) + [np.radians(float(r[5])),\
                                        np.radians(float(r[6])), np.radians(float(r[7]))] + list(map(float, r[-4:])))
                            else:
                                if r[1] != v[1]:
                                    oth_car_info.append([int(r[1])] + list(map(float, r[2:5])) + [np.radians(float(r[5])),\
                                        np.radians(float(r[6])), np.radians(float(r[7]))] + list(map(float, r[-4:])))
                        if len(oth_car_info) > 0:
                            oth_cars_info.append(oth_car_info)
                        # print(oth_car_info)

                    # print(first_v_raw_trans)
                    # #######TEST FOR INVERSE TRANSFORM TO CARLR COORDINATE
                    # inv_ego_pos = first_v_raw_trans[:3]
                    # print(inv_ego_pos)
                        
                    show_flag = 1
                    lcl_R = [src_R, tgt_R]
                    lcl_R_inv = [src_R_inv, tgt_R_inv]
                    lcl_rot_v = [first_v_raw_trans[3:], tgt_v_raw_trans[3:]]
                    lcl_trans_v = [first_v_raw_trans[:3], tgt_v_raw_trans[:3]]
                    colors = [[0,1,0], [1,0,0]]
                    colors2 = [[0,1,0], [0,1,1]]
                    objs_center = []
                    objs_flow = []
                    objs_bbx = []
                    objs_bbox_R = []
                    # fg_inds = []
                    # if show_flag:
                    
                    for iv in range(len(oth_cars_info)):
                        if len(oth_cars_info) <= 1:
                            flow = arr_ - arr
                            # continue
                        else:
                            oth_v_loc = np.array(oth_cars_info[iv])[:,1:4] * np.array([1.0, -1.0, 1.0])
                            oth_v_loc[:,-1] = oth_v_loc[:,-1] + np.array(oth_cars_info[iv])[:,-1] - lidar_hight
                            oth_v_rot = np.array(oth_cars_info[iv])[:,4:7]
                            oth_v_bbx_ext = np.array(oth_cars_info[iv])[:,-4:]

                            # objs_center = []
                            obj_flow = []
                            obj_bbx = []
                            obj_bbox_R = []

                            for ibbx in range(oth_v_loc.shape[0]):
                                # bbox_v_rot = -1.0 * oth_v_rot[ibbx, :] + np.array([0.0, 0.0, 2.0 * np.pi])
                                bbox_v_rot = lcl_rot_v[iv] - oth_v_rot[ibbx, :] #The relative rotation between the ego_vehcicle and others in the same frame
                                bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(bbox_v_rot)
                                # bbox_R = bbox_R @ o3d.geometry.get_rotation_matrix_from_xyz(lcl_rot_v[iv]) 
                                #For the local (observation) coordinates
                                obj_bbox_R.append(bbox_v_rot)
                        
                                bbox_extend = oth_v_bbx_ext[ibbx, :-1] * 2.0
                                #For the local (observation) coordinates
                                bbox_center = lcl_R[iv] @ (oth_v_loc[ibbx,:] - lcl_trans_v[iv]) #The relative shift between the ego_vehcicle and others in the same frame
                                #For the original coordinates
                                # bbox_center = lcl_R[iv] @ oth_v_loc[ibbx,:]
                                # print(bbox_center)
                                obj_flow.append(np.hstack((np.array(oth_cars_info[iv][ibbx][0]), np.array(bbox_center))))
                                #For the local (observation) coordinates
                                bbox = o3d.geometry.OrientedBoundingBox(bbox_center, bbox_R, bbox_extend)
                                #For the original coordinates
                                # bbox = o3d.geometry.OrientedBoundingBox(bbox_center, bbox_R, bbox_extend)
                                obj_bbx.append(bbox)
                                if i_ind % 2 == 0:
                                    bbox.color = colors2[iv]
                                else:
                                    bbox.color = colors[iv]
                                # bbox.LineSet = 
                                vis_list.append(bbox)
                            objs_flow.append(obj_flow)
                            objs_bbx.append(obj_bbx)
                            objs_bbox_R.append(obj_bbox_R)
                    
                        if len(objs_flow) <= 1:
                            flow = arr_ - arr
                        else:
                            src_objs_flow = np.array(objs_flow[0])
                            src_objs_bbox = dict(zip(src_objs_flow[:,0], objs_bbx[0]))
                            src_objs_flow = dict(zip(src_objs_flow[:,0], src_objs_flow[:,1:]))
                            src_bbox_R = dict(zip(src_objs_flow.keys(),objs_bbox_R[0]))                    
                        
                            tgt_objs_flow = np.array(objs_flow[1])
                            tgt_objs_bbox = dict(zip(tgt_objs_flow[:,0], objs_bbx[1]))
                            tgt_objs_flow = dict(zip(tgt_objs_flow[:,0], tgt_objs_flow[:,1:]))
                            tgt_bbox_R = dict(zip(tgt_objs_flow.keys(),objs_bbox_R[1]))

                            objs_flow = []
                            # objs_c = np.array(list(src_objs_flow.values()))
                            # subarr_ = (objs_c @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T
                            # subflow = subarr_ - objs_c
                            # subarr_ = (objs_c @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T
                            # subflow = subarr_ - objs_c
                            # src_rigid_flow = dict(zip(src_objs_flow.keys(), subflow))
                            objs_bbox_R = dict(zip(src_objs_flow.keys(),objs_bbox_R))
                            for k,v in src_objs_flow.items():   
                                if k in tgt_objs_flow:
                                    bbox = src_objs_bbox.get(k) 
                                    inds = bbox.get_point_indices_within_bounding_box(pcd.points)
                                    bbox_t = tgt_objs_bbox.get(k) 
                                    inds_t = bbox_t.get_point_indices_within_bounding_box(pcd_cur.points)
                                    # flow[inds,:] = obj_flow
                                    delat_rot = tgt_bbox_R.get(k) - src_bbox_R.get(k)
                                    bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(-delat_rot)
                                    obj_flow = tgt_objs_flow.get(k) - v @ bbox_R
                                    arr_[inds,:] = arr[inds,:] @ bbox_R + obj_flow 
                                    if i_ind % 2 != 0 and len(inds) > 0:
                                        if len(fg_inds) == 0:
                                            fg_inds = inds
                                            fg_inds_t = inds_t
                                        else:
                                            fg_inds += inds
                                            fg_inds_t += inds_t
                                        fgrnd_inds[inds] = 1
                                        fgrnd_inds_t[inds_t] = 1
                                else:
                                    obj_flow = np.zeros(3)
                                # print(obj_flow)
                                
                                objs_flow.append(obj_flow)
                            # delta_flow = np.array(objs_flow) - subflow
                            objs_flow = dict(zip(src_objs_flow.keys(), objs_flow))
                            flow = 1.0 * (arr_ - arr)
                    # vis_list.append(pcd)
                    pcd_cur2 = o3d.geometry.PointCloud()
                    pcd_cur2.points = o3d.utility.Vector3dVector(arr_)
                    pcd_cur2.paint_uniform_color([0,0,1.0])

                    # concat_fg_inds = []
                    # for item in enumerate(fg_inds):
                    #     if len(item) > 0:
                    #         concat_fg_inds += item[1]

                    # vis_list.append(pcd)
                    # vis_list.append(pcd_cur)
                    # vis_list.append(pcd_cur2)
                    # o3d.visualization.draw_geometries(vis_list)
                    # save_view_point(vis_list, 'camera_view.json')
                    
                    use_flow = True
                    flow = flow * 1.0
                    ego_flow = (src_pc @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T - src_pc
                    ego_flow_bkp = ego_flow + tgt_v_raw_trans[:3] - first_v_raw_trans[:3]
                    flow_bkp = flow + tgt_v_raw_trans[:3] - first_v_raw_trans[:3] 

                    if i_ind % 2 == 0:
                        pre_ego_flow = -1.0 * ego_flow
                        pre_flow = -1.0 * flow
                    
                    name_fmt = "%06d"%(ind)
                    if save_pts:
                        if only_fg:
                            np.savez(DATASET_SF_PATH + name_fmt, pos1=src_pc[fg_inds,:], pos2=raw_tgt_pc[fg_inds_t,:], pos3=raw_tgt_pc2, ego_flow=ego_flow[fg_inds,:], gt=flow[fg_inds,:], pre_ego_flow=pre_ego_flow[fg_inds,:], pre_gt=pre_flow[fg_inds,:], s_fg_mask = fgrnd_inds, t_fg_mask = fgrnd_inds_t) 
                        else:
                            np.savez(DATASET_SF_PATH + name_fmt, pos1=src_pc, pos2=raw_tgt_pc, pos3=raw_tgt_pc2, ego_flow=ego_flow, gt=flow, pre_ego_flow=pre_ego_flow, pre_gt=pre_flow, s_fg_mask = fgrnd_inds, t_fg_mask = fgrnd_inds_t) 
                    if show_fgrnds:
                        if i_ind % 2 != 0:
                            tgt_pc = tgt_pc[fg_inds_t,:]
                        src_pc = src_pc[fg_inds,:]
                        arr_ = arr_[fg_inds,:]
                        ego_flow = ego_flow[fg_inds,:]
                        flow = flow[fg_inds,:]

                
                    # tgt_v_raw_rot = tgt_v_raw_trans[3:]
                    # # tgt_v_raw_rot[-1] *= -1.0
                    # bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(tgt_v_raw_rot)
                    # # bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(tgt_v_raw_trans[3:])
                    # ego_car_bbx_n = o3d.geometry.OrientedBoundingBox(ego_car_center, bbox_R, egoV_bbox_extend)
                    # # vis_list.append(ego_car_bbx_n)
                    # if i_ind % 2 == 0:
                    #     inds = ego_car_bbx_n.get_point_indices_within_bounding_box(pcd_pred.points)
                    # else:
                    #     inds = ego_car_bbx_n.get_point_indices_within_bounding_box(pcd_cur.points)
                    # valid_inds = np.ones([tgt_pc.shape[0]])
                    # if len(inds) > 0:
                    #     valid_inds[inds] = 0
                    #     valid_inds = valid_inds.astype(bool)
                    #     tgt_pc = tgt_pc[valid_inds,:]
                

                # name_fmt = "%06d"%(ind)
                # if save_pts:
                #     np.savez(DATASET_SF_PATH + name_fmt, pos1=src_pc, pos2=tgt_pc, ego_flow=ego_flow, gt=flow) 

                    if use_flow:
                        # sf = sf[sample_idx1, :]
                        np_list = [src_pc, tgt_pc, arr_, flow]
                        # np_list = [src_pc, tgt_pc, arr_, ego_flow]   
                        # np_list = [src_pc_bkp, tgt_pc_bkp, arr_ + tgt_v_raw_trans[:3], ego_flow_bkp]                 
                        # np_list = [src_pc_bkp, tgt_pc_bkp, arr_ + tgt_v_raw_trans[:3], flow_bkp]
                    else:
                        np_list = [src_pc, tgt_pc, arr_,]
                        # np_list = [src_pc_bkp, tgt_pc_bkp, arr_, flow]
                    
                    # import os
                    # custom_draw_geometry(vis, vis_list, map_file=None, recording=False,param_file='camera_view.json', save_fov=True)
                    # print(os.path.abspath(os.curdir))
                    src_pc2 = src_pc - np.mean(src_pc, axis=0)
                    normp = np.sqrt(np.sum(src_pc2 * src_pc2, axis=1))
                    # diameter = np.max(normp)
                    print("Define parameters used for hidden_point_removal")
                    
                    # camera = [-1.0, 0.0, 5]
                    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3).translate(camera)
                    # print("Get all points that are visible from given view point")
                    # radius = diameter * 10
                    # _, pt_map = pcd3.hidden_point_removal(camera, radius)
                    # pc = np.vstack([pc1, pc2])
                    # camera = [-45.0, -20.0, drivable_area[0,2]]
                    # camera = [45.0, -60.0, drivable_area[0,2]]
                    # camera = [-35.0, -20.0, drivable_area[0,2]]
                    # camera = [75.0, 1.5, drivable_area[0,2]]
                    # arr2_ = arr_ + flow
                    # pt_map = HPR(tgt, np.array([[0, 0, -2.5]]), 3)
                    # np_arr2 = arr2_[pt_map,:3]
                    # # print("Visualize result")
                    # # pcd3 = pcd3.select_by_index(pt_map)
                    # bool_in = in_convex_polyhedron(arr2_[pt_map,:3],np.array([[-5.0, 4.0, -2.5]]), vis) 
                    # pcd3 = drivable_area[pt_map,:]
                    # bool_in = in_convex_polyhedron(drivable_area[iou_id[:,0],:3], np.array([[0, 0, -2.5]]), vis) 
                    # camera_pos = np.hstack((first_v_raw_trans[:2], -2.75)).reshape(1,3)
                    # bool_in = in_convex_polyhedron(drivable_area[iou_id[:,0],:3],camera_pos, vis) 
                    # global bbx
                    # # bbx = np.array([[0,0,0], [4,0,0], [0,-4,0], [4,-1,0], [0,0,-2.5], [4,0,-2.5], [0,-4,-2.5], [4,-4,-2.5]]) * 2.0
                    # bbx = np.array([[0,0,0], [4,0,0], [0,-1,0], [4,-1,0], [0,0,-2.5], [4,0,-2.5], [0,-1,-2.5], [4,-1,-2.5]]) * 0.8
                    # # bbx = np.array([[0,0,0], [1,0,0], [0,-1,0], [1,-1,0], [0,0,-2.5], [1,0,-2.5], [0,-1,-2.5], [1,-1,-2.5]]) 
                    # # bool_in, _ = validDrivableArea(bbx, tgt_pc)
                    # bool_in, _ = validateDrivableArea2(bbx, tgt_pc)

                    # sphere = o3d.geometry.TriangleMesh.create_sphere(1).translate(np.array([0, 0, -2.5]))
                    # if bool_in:
                    #     sphere.paint_uniform_color([0.2,0.2,1.0])
                    # else:
                    #     sphere.paint_uniform_color([1,0,0])
                    # vis_list += [sphere]

                    # np_list = vis_list
                    use_flow = False
                    if i_ind % 2:
                        if not(enable_animation):
                            vis = o3d.visualization.Visualizer()
                            vis.create_window(width=960*2, height=640*2, left=5, top=5)
                            vis.get_render_option().background_color = np.array([0,0,0])
                            # vis.get_render_option().background_color = np.array([1, 1, 1])
                            vis.get_render_option().show_coordinate_frame = False
                            vis.get_render_option().point_size = 3.0
                            vis.get_render_option().line_width = 5.0
                        Draw_SceneFlow(vis, np_list+vis_list, use_fast_vis = not(use_flow), use_pred_t=True, use_custom_color=True, use_flow=use_flow, param_file='camera_view.json',img_cnt=img_cnt)
                        img_cnt = img_cnt + 1
                
        sub_dir_cnt += 1
    if egoV_flag:
        np.savez(egoV_file_path, cvInfo = current_v_info, nvInfo = next_v_info)



# For scenario difference measure

def Generate_Active_SceneFlow(car_ids, start = 3, step = 3, lidar_hight = 3.4, prefixed_V_name=None, show_active_egoV=False, save_pts=False, enable_animation = True, egoV_flag=False, rm_ground = False, show_fgrnds = True, egoV_file_path = './data/data_Vinfo0307', GT_SUB_DATASET='vehicle.mini.cooper_s_2021_254', ACTIVE_SUB_DATASET='vehicle.mini.cooper_s_2021_1280'):
    if step == 1:
        SF_DIR = '/SF/'
    elif step == 2:
        SF_DIR = '/SF2/'
    elif step == 3:
        SF_DIR = '/SF3/'
    else:
        SF_DIR = 'TEST'

    # show_fgrnds = True
    
    start = start
    drivable_areas = odom_utils.readPointCloud('./dataset/town02-map.bin') #road map
    drivable_area = drivable_areas[:,:3]
    drivable_area[:,-1] = -1.0 * lidar_hight
    road_id = drivable_areas[:,-1]
    colors = []
    for iCnt in road_id:
        if iCnt == 0:
            colors += [[1,0,0]]
        elif iCnt == 1:
            colors += [[0,1,0]]
        elif iCnt == 2:
            colors += [[0,0,1]]
        elif iCnt == 3:
            colors += [[1,1,0]]
        else:
            colors += [[1,0,1]]
    #指定实验道路
    iou_id = np.argwhere(road_id == 1)

    drivable_centers = np.load('./dataset/np_c.npy')#odom_utils.readPointCloud('./dataset/town02-road-center.bin')
    drivable_center = drivable_centers[:,:3]
    drivable_center[:, -1] = -1.0 * lidar_hight
    drivable_center[:,1] *= -1.0
    road_id2 = []
    road_id2 = drivable_centers[:,-1]
    ref_drivable_center_inds = np.argwhere(drivable_centers[:,-1] == 1)
    colors2 = []
    for iCnt in road_id2:
        if iCnt < 1:
            colors2 += [[1,0,0]]
        elif iCnt >= 1  and iCnt < 2:
            colors2 += [[0,1,0]]
        elif iCnt >= 2 and iCnt < 3:
            colors2 += [[0,0,1]]
        elif iCnt >= 3 and iCnt < 4:
            colors2 += [[1,1,0]]
        else:
            colors2 += [[1,0,1]]

    
    # drivable_area[:,0] *= -1
    pcd_drivable_area = o3d.geometry.PointCloud()
    # pcd_drivable_area.points = o3d.utility.Vector3dVector(drivable_area)
    drivable_area_bkp = drivable_area
    pcd_drivable_area.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.dtype('f4')))

    pcd_drivable_center = o3d.geometry.PointCloud()
    # pcd_drivable_area.points = o3d.utility.Vector3dVector(drivable_center)
    drivable_center_bkp = drivable_center
    pcd_drivable_center.colors = o3d.utility.Vector3dVector(np.array(colors2, dtype=np.dtype('f4')))
                
    # Compute_PairwiseSceneFlow(start, 5)

    # Translate frame coordinates from left-hand to right-hand
    v_l2r = np.array([1., -1., 1., 1., 1., 1.])
    s_l2r = np.array([1, 1., -1., 1., 1., 1., 1.])
    v2s = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    # enable_animation = True
    if enable_animation:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=960*2, height=640*2, left=5, top=5)
        vis.get_render_option().background_color = np.array([0, 0, 0])
        # vis.get_render_option().background_color = np.array([1, 1, 1])
        vis.get_render_option().show_coordinate_frame = False
        vis.get_render_option().point_size = 3.0
        vis.get_render_option().line_width = 5.0
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, -1.0 * lidar_hight])
    
    # Generate the log data from the DIR:tmp/xxx/xxxxrgb_lable/xxxx 
    build_lidar_fov = False

    # Generate the scene flow data of the only camera field-of-view
    use_rgb_fov = False
    
    # rm_ground = False
    # trans_pattern = r"(?P<v_name>[vehicle]+.\w+.\w+) (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
    # r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
    #     r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
    #         r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"\
    
    rawtrans_files = glob.glob(os.path.join(DATASET_RAWTRANS_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
    rawtrans_files.sort()

    rawtrans_files2 = glob.glob(os.path.join(DATASET_RAWTRANS_PATH2, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
    rawtrans_files2.sort()

    # rawtrans_files2 = glob.glob(os.path.join(DATASET_RAWTRANS_PATH2, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
    # rawtrans_files2.sort()
    # if not(use_rgb_fov):
    #     for r in re.findall(trans_pattern, log):
    #         # print(r[0])
    #         det_car_info.append(int(float(r[0])))
    # print(os.path.abspath(os.curdir))
    sub_dir_cnt = 0
    max_v = -100.0
    # v_info_cnt = 0
    current_v_info = []
    next_v_info = []
    v_pattern = r"vehicle.(?P<nonsense>[\s\S]+)_(?P<v_id>\d+)"
    for sub_dir in os.listdir(DATASET_PATH):
        if 'global_label' in sub_dir or 'img' in sub_dir:
            continue 
        # print(sub_dir)

        SUB_DATASET = 'vehicle.mini.cooper_s_2021_1520' #sub_dir #'vehicle.tesla.cybertruck_266'#'vehicle.citroen.c3_271'
        SUB_DATASET = GT_SUB_DATASET #'vehicle.mini.cooper_s_2021_254' #sub_dir #'vehicle.tesla.cybertruck_266'#'vehicle.citroen.c3_271'
        # SUB_DATASET = 'vehicle.mini.cooper_s_2021_406' #sub_dir #'vehicle.tesla.cybertruck_266'#'vehicle.citroen.c3_271'
        SUB_DATASET2 = 'vehicle.mini.cooper_s_2021_402' #sub_dir #'vehicle.tesla.cybertruck_266'#'vehicle.citroen.c3_271'
        SUB_DATASET2 = ACTIVE_SUB_DATASET #'vehicle.mini.cooper_s_2021_254' #sub_dir #'vehicle.tesla.cybertruck_266'#'vehicle.citroen.c3_271'
        if sub_dir not in SUB_DATASET:
            continue
        v_pattern = r"[vehicle]+.(?P<name>\w+.\w+)_(?P<v_id>\d+)"
        v = re.findall(v_pattern, SUB_DATASET)[0]
        DATASET_PC_PATH = DATASET_PATH + SUB_DATASET + '/velodyne/'
        DATASET_LBL_PATH = DATASET_PATH + SUB_DATASET + '/label00/'

        v2 = re.findall(v_pattern, SUB_DATASET2)[0]
        DATASET_PC_PATH2 = DATASET_PATH2 + SUB_DATASET2 + '/velodyne/'
        DATASET_LBL_PATH2 = DATASET_PATH2 + SUB_DATASET2 + '/label00/'

        # RESULT_PATH = './results/' + SUB_DATASET
        trans_pattern = v[0] + " " + v[1] + r" (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"
        lidar_pattern = r"sensor.lidar.ray_cast (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 " + v[1]
        
        lidar_semantic_pattern = r"sensor.lidar.ray_cast_semantic (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 0 " + v[1]

        trans_pattern2 = v2[0] + " " + v2[1] + r" (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"
        lidar_pattern2 = r"sensor.lidar.ray_cast (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 " + v2[1]
        
        lidar_semantic_pattern2 = r"sensor.lidar.ray_cast_semantic (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 0 " + v2[1]
        
        glb_label_pattern2 = v[0] + " " + v[1] + r" (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
    r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
        r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<extbbx_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<extbbx_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<extbbx_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<bbx_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"

        # next_ind = ind + 2
        pc_files = glob.glob(os.path.join(DATASET_PC_PATH, '*.bin'))#os.listdir(DATASET_PC_PATH)
        # rawtrans_files = glob.glob(os.path.join(DATASET_RAWTRANS_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
        label_files = glob.glob(os.path.join(DATASET_LBL_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
        # print("Current DIR:" + DATASET_PC_PATH)
        pc_files.sort()
        # rawtrans_files.sort()
        label_files.sort()

        # next_ind = ind + 2
        pc_files2 = glob.glob(os.path.join(DATASET_PC_PATH2, '*.bin'))#os.listdir(DATASET_PC_PATH)
        # rawtrans_files = glob.glob(os.path.join(DATASET_RAWTRANS_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
        label_files2 = glob.glob(os.path.join(DATASET_LBL_PATH2, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
        # print("Current DIR:" + DATASET_PC_PATH)
        pc_files2.sort()
        # rawtrans_files.sort()
        label_files2.sort()

        total_cmd_num = len(pc_files)
        frame_start,frame_end,frame_hz = 10,-5,1
        spawn_points = []
        cnt = 0

        ego_v_info = re.findall(v_pattern, sub_dir)
        with open(rawtrans_files[0], 'r') as file:
                raw_log = file.read()
        r = re.findall(glb_label_pattern, raw_log)
        for r in re.findall(glb_label_pattern, raw_log):
            if ego_v_info[0][0] in r[0] and ego_v_info[0][1] in r[1]:
                l = float(r[8])
                w = float(r[9])
                h = float(r[10])
                egoV_bbox_extend = np.array([l, w, h]).reshape(3,1) * 2.0
        
        
        DATASET_SF_PATH = RESULT_PATH3 + SF_DIR + "%02d"%(sub_dir_cnt) + '/'
        if not os.path.exists(DATASET_SF_PATH) and save_pts:
            os.makedirs(DATASET_SF_PATH)

        if build_lidar_fov:
            write_labels('record2021_0716_2146/', sub_dir, frame_start,frame_end, frame_hz, index=1)
        else:
            # step = 1
            for ind in range(start, len(pc_files)-step, step):
                next_ind = ind #+ step
                if next_ind >= len(label_files):
                        break
                print("Current DIR:" + DATASET_PC_PATH + ' Frame:' +str(ind))
                # tmp = cmd_traj[ind, ]
                # saveVehicleInfo()
                
                # # For writting the vehicle info ln:1357-1558
                # ivFlag = 0
                # for i in [ind, next_ind]:
                #     # if next_ind >= len(label_files):
                #     #     break
                #     with open(label_files[i], 'r') as file:
                #             log = file.read()
                #     with open(rawtrans_files[i], 'r') as file:
                #         raw_log = file.read()
               
                #     for r in re.findall(glb_label_pattern, raw_log):
                #         # print(r)
                #         if ego_v_info[0][0] in r[0] and ego_v_info[0][1] in r[1]:
                #             if ivFlag % 2 == 0:
                #                 current_v_info.append(r)
                #             else:
                #                 next_v_info.append(r)
                #             ivFlag += 1

                # continue
   
                # Compute_PairwiseSceneFlow(ind, next_ind, trans_pattern, lidar_pattern, DATASET_PC_PATH, DATASET_LBL_PATH)
                vis_list = []
                first_v_raw_trans = odom_utils.readRawData(rawtrans_files[ind], trans_pattern) * v_l2r
                first_sensor_raw_trans = odom_utils.readRawData(rawtrans_files[ind], lidar_semantic_pattern) * s_l2r

                src_pc = odom_utils.readPointCloud(pc_files[ind])[:, :3]
                src_pc_bkp = src_pc + first_v_raw_trans[:3]
                if rm_ground:
                    is_ground = np.logical_not(src_pc[:, 2] <= -1.0 * (lidar_hight-0.1))
                    src_pc = src_pc[is_ground, :] 
                # np.save('./data/np_src', src_pc)
                # For the carla origin coordatinate 
                # first_v_raw_trans[2] += 2.5
                # src_pc += first_v_raw_trans[:3] @ np.array([[0,-1,0],[1,0,0],[0,0,1]])
                
                # ## For the carla original coordinate
                # drivable_area = drivable_area_bkp @ np.array([[0,-1,0],[1,0,0],[0,0,1]])

                ## For the first car frame as the original coordinate
                drivable_area = (drivable_area_bkp - first_v_raw_trans[:3])\
                     @ odom_utils.rotation_from_euler_zyx(-first_v_raw_trans[5], -first_v_raw_trans[4],first_v_raw_trans[3])#
                pcd_drivable_area.points = o3d.utility.Vector3dVector(drivable_area)
                # np.save('./data/np_drivable_area', np.hstack((drivable_area, (drivable_areas[:,-1]).reshape((-1,1)))))
                
                ## For the carla original coordinate
                # drivable_center = drivable_center_bkp @ np.array([[0,-1,0],[1,0,0],[0,0,1]])

                # For the first car frame as the original coordinate
                drivable_center = (drivable_center_bkp - first_v_raw_trans[:3])\
                     @ odom_utils.rotation_from_euler_zyx(-first_v_raw_trans[5], -first_v_raw_trans[4],first_v_raw_trans[3])#
                pcd_drivable_center.points = o3d.utility.Vector3dVector(drivable_center)
                # pcd_drivable_area.paint_uniform_color([1.0,0,0.0])
                ref_drivable_center = drivable_centers[ref_drivable_center_inds[:,0],:3]

                # np.save('./data/np_drivable_center', np.hstack((drivable_center, (drivable_centers[:,-1]).reshape((-1,1)))))

                arr = src_pc
                
                src_R_vec = first_sensor_raw_trans[4:]
                src_R = o3d.geometry.get_rotation_matrix_from_xyz(src_R_vec)
                src_R_inv = np.linalg.inv(src_R)
                tmp_pcd = o3d.geometry.PointCloud()
                tmp_pcd.points = o3d.utility.Vector3dVector(src_pc)
                #For the carla origin coordatinate 
                # pcd.rotate(src_R_inv, first_v_raw_trans[:3])

                # For the first frame as the origin coordinate
                # pcd.rotate(src_R_inv, np.zeros(3))
                # pcd.paint_uniform_color([0,1,0])

                # ego_pcd = o3d.geometry.PointCloud()
                # ego_pcd.points = o3d.utility.Vector3dVector(first_v_raw_trans[:3].reshape(-1,3))
                # ego_pcd.paint_uniform_color([0,1,1])
               
                # drivable_area =  drivable_area - (np.array([[spawn_points[ind][0], -spawn_points[ind][1], 0]]))
                
                vis_list = [mesh_frame, pcd_drivable_area, pcd_drivable_center]
                # vis_list = [mesh_frame, pcd]
                
                tgt_v_raw_trans = odom_utils.readRawData(rawtrans_files2[next_ind], trans_pattern2) * v_l2r

                print(first_v_raw_trans[:3])
                print(tgt_v_raw_trans[:3])
                # tgt_v_raw_trans2 = odom_utils.readRawData(rawtrans_files2[next_ind], trans_pattern2) * v_l2r
                # tgt_v_raw_trans0 = odom_utils.readRawData(rawtrans_files[next_ind], trans_pattern2) * v_l2r
                # print(first_v_raw_trans)
                # print(tgt_v_raw_trans0)
                # print(tgt_v_raw_trans)
                # print(tgt_v_raw_trans2)
                tgt_sensor_raw_trans = odom_utils.readRawData(rawtrans_files2[next_ind], lidar_semantic_pattern2) * s_l2r
                tgt_pc = odom_utils.readPointCloud(pc_files2[next_ind])[:, :3]

                tgt_pc_bkp = tgt_pc + tgt_v_raw_trans[:3]
                if rm_ground:
                    is_ground = np.logical_not(tgt_pc[:, 2] <= -1.0 * (lidar_hight - 0.05))
                    tgt_pc = tgt_pc[is_ground, :] 

                #For the carla origin coordatinate 
                # tgt_v_raw_trans[2] += 2.5
                # tgt_pc += tgt_v_raw_trans[:3] @ np.array([[0,-1,0],[1,0,0],[0,0,1]])

                tgt_R_vec = tgt_sensor_raw_trans[4:]
                tgt_R = o3d.geometry.get_rotation_matrix_from_xyz(tgt_R_vec)
                tgt_R_inv = np.linalg.inv(tgt_R)
                tmp_pcd_cur = o3d.geometry.PointCloud()
                tmp_pcd_cur.points = o3d.utility.Vector3dVector(tgt_pc)

                #For the carla origin coordatinate 
                # pcd_cur.rotate(tgt_R_inv, tgt_v_raw_trans[:3] )

                #For the first frame as the origin coordinate
                # pcd_cur.rotate(tgt_R_inv, np.zeros(3))
                # pcd_cur.translate(tgt_v_raw_trans[:3] - first_v_raw_trans[:3])
                # tmp_pcd_cur.paint_uniform_color([1,0,0])

                first_v_raw_rot = first_v_raw_trans[3:]
                # first_v_raw_rot[-1] *= -1.0
                src_ego_bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(first_v_raw_rot)
                # src_ego_bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(first_v_raw_trans[3:])
                ego_car_center = np.zeros([3,1])
                ego_car_center[2] = -1.0 * lidar_hight + 0.5 * egoV_bbox_extend[2]
                ego_car_bbx_c = o3d.geometry.OrientedBoundingBox(ego_car_center, src_ego_bbox_R, egoV_bbox_extend)
                # vis_list.append(ego_car_bbx_c)
                inds = ego_car_bbx_c.get_point_indices_within_bounding_box(tmp_pcd.points)
                valid_inds = np.ones([src_pc.shape[0]])
                if len(inds) > 0:
                    valid_inds[inds] = 0
                    valid_inds = valid_inds.astype(bool)
                    src_pc = src_pc[valid_inds,:]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(src_pc)
                #For the carla origin coordatinate 
                # pcd.rotate(src_R_inv, first_v_raw_trans[:3])

                # For the first frame as the origin coordinate
                # pcd.rotate(src_R_inv, np.zeros(3))
                pcd.paint_uniform_color([0,1,0])

                tgt_v_raw_rot = tgt_v_raw_trans[3:]
                # first_v_raw_rot[-1] *= -1.0
                tgt_ego_bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(tgt_v_raw_rot)
                tgt_ego_bbx_center = (np.zeros([3]) @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T
                tgt_ego_bbx_center[2] = -1.0 * lidar_hight + 0.5 * egoV_bbox_extend[2]
                tgt_ego_car_bbx_c = o3d.geometry.OrientedBoundingBox(tgt_ego_bbx_center, tgt_ego_bbox_R, egoV_bbox_extend)
                # vis_list.append(ego_car_bbx_c)
                tgt_inds = tgt_ego_car_bbx_c.get_point_indices_within_bounding_box(tmp_pcd_cur.points)
                tgt_valid_inds = np.ones([tgt_pc.shape[0]])
                if len(tgt_inds) > 0:
                    tgt_valid_inds[tgt_inds] = 0
                    tgt_valid_inds = tgt_valid_inds.astype(bool)
                    tgt_pc = tgt_pc[tgt_valid_inds,:]
                pcd_cur = o3d.geometry.PointCloud()
                pcd_cur.points = o3d.utility.Vector3dVector(tgt_pc)

                #For the carla origin coordatinate 
                # pcd_cur.rotate(tgt_R_inv, tgt_v_raw_trans[:3] )

                #For the first frame as the origin coordinate
                # pcd_cur.rotate(tgt_R_inv, np.zeros(3))
                # pcd_cur.translate(tgt_v_raw_trans[:3] - first_v_raw_trans[:3])
                pcd_cur.paint_uniform_color([1,0,0])
                
                arr_ = (src_pc @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T

                alternation_flag = 2
                oth_cars_info = []
                fgrnd_inds = np.zeros(src_pc.shape[0])
                fg_inds = []
                fgrnd_inds_t = np.zeros(tgt_pc.shape[0])
                fg_inds_t = []
                for  iflag, i in enumerate([ind, next_ind]):
                    # if next_ind >= len(label_files):
                    #     break
                    if iflag % alternation_flag == 0:
                        with open(label_files[i], 'r') as file:
                                log = file.read()
                    else:
                        with open(label_files2[i], 'r') as file:
                                log = file.read()
                    

                    det_car_info = []  
                    if use_rgb_fov:
                        for r in re.findall(rgb_label_pattern, log):
                            det_car_info.append(int(r[-1]))
                    # else:
                    #     for r in re.findall(lidar_label_pattern, log):
                    #         # print(r[0])
                    #         det_car_info.append(int(float(r[0])))
                    if iflag % alternation_flag == 0:
                        with open(rawtrans_files[i], 'r') as file:
                            raw_log = file.read()
                    else:
                        with open(rawtrans_files2[i], 'r') as file:
                            raw_log = file.read()
                    oth_car_info = []
                    for r in re.findall(glb_label_pattern, raw_log):
                        # print(r[0])
                        if use_rgb_fov:
                            if int(r[1]) in det_car_info:
                                oth_car_info.append([int(r[1])] + list(map(float, r[2:5])) + [np.radians(float(r[5])),\
                                    np.radians(float(r[6])), np.radians(float(r[7]))] + list(map(float, r[-4:])))
                        else:
                            if r[1] != v[1]:
                                oth_car_info.append([int(r[1])] + list(map(float, r[2:5])) + [np.radians(float(r[5])),\
                                    np.radians(float(r[6])), np.radians(float(r[7]))] + list(map(float, r[-4:])))
                    if len(oth_car_info) > 0:
                        oth_cars_info.append(oth_car_info)
                    # print(oth_car_info)

                # print(first_v_raw_trans)
                # #######TEST FOR INVERSE TRANSFORM TO CARLR COORDINATE
                # inv_ego_pos = first_v_raw_trans[:3]
                # print(inv_ego_pos)
                    
                show_flag = 1
                lcl_R = [src_R, tgt_R]
                lcl_R_inv = [src_R_inv, tgt_R_inv]
                lcl_rot_v = [first_v_raw_trans[3:], tgt_v_raw_trans[3:]]
                lcl_trans_v = [first_v_raw_trans[:3], tgt_v_raw_trans[:3]]
                colors = [[0,1,0], [1,0,0]]
                objs_center = []
                objs_flow = []
                objs_bbx = []
                objs_bbox_R = []
                # fg_inds = []
                # if show_flag:
                for iv in range(len(oth_cars_info)):
                    if len(oth_cars_info) <= 1:
                        flow = arr_ - arr
                        # continue
                    else:
                        oth_v_loc = np.array(oth_cars_info[iv])[:,1:4] * np.array([1.0, -1.0, 1.0])
                        oth_v_loc[:,-1] = oth_v_loc[:,-1] + np.array(oth_cars_info[iv])[:,-1] - lidar_hight
                        oth_v_rot = np.array(oth_cars_info[iv])[:,4:7]
                        oth_v_bbx_ext = np.array(oth_cars_info[iv])[:,-4:]

                        # objs_center = []
                        obj_flow = []
                        obj_bbx = []
                        obj_bbox_R = []

                        for ibbx in range(oth_v_loc.shape[0]):
                            # bbox_v_rot = -1.0 * oth_v_rot[ibbx, :] + np.array([0.0, 0.0, 2.0 * np.pi])
                            bbox_v_rot = lcl_rot_v[iv] - oth_v_rot[ibbx, :]
                            bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(bbox_v_rot)
                            #For the local (observation) coordinates
                            obj_bbox_R.append(bbox_v_rot)
                       
                            bbox_extend = oth_v_bbx_ext[ibbx, :-1] * 2.0
                            #For the local (observation) coordinates
                            bbox_center = lcl_R[iv] @ (oth_v_loc[ibbx,:] - lcl_trans_v[iv])
                            #For the original coordinates
                            # bbox_center = lcl_R[iv] @ oth_v_loc[ibbx,:]
                            # print(bbox_center)
                            obj_flow.append(np.hstack((np.array(oth_cars_info[iv][ibbx][0]), np.array(bbox_center))))
                            #For the local (observation) coordinates
                            bbox = o3d.geometry.OrientedBoundingBox(bbox_center, bbox_R, bbox_extend)
                            #For the original coordinates
                            # bbox = o3d.geometry.OrientedBoundingBox(bbox_center, bbox_R, bbox_extend)
                            obj_bbx.append(bbox)
                            bbox.color = colors[iv]
                            # bbox.LineSet = 
                            vis_list.append(bbox)
                        objs_flow.append(obj_flow)
                        objs_bbx.append(obj_bbx)
                        objs_bbox_R.append(obj_bbox_R)
                    
                    # fg_inds = np.zeros(src_pc.shape[0])
                    if len(objs_flow) <= 1:
                        flow = arr_ - arr
                    else:
                        src_objs_flow = np.array(objs_flow[0])
                        src_objs_bbox = dict(zip(src_objs_flow[:,0], objs_bbx[0]))
                        src_objs_flow = dict(zip(src_objs_flow[:,0], src_objs_flow[:,1:]))
                        src_bbox_R = dict(zip(src_objs_flow.keys(),objs_bbox_R[0]))                    
                    
                        tgt_objs_flow = np.array(objs_flow[1])
                        tgt_objs_bbox = dict(zip(tgt_objs_flow[:,0], objs_bbx[1]))
                        tgt_objs_flow = dict(zip(tgt_objs_flow[:,0], tgt_objs_flow[:,1:]))
                        tgt_bbox_R = dict(zip(tgt_objs_flow.keys(),objs_bbox_R[1]))

                        objs_flow = []
                        # objs_c = np.array(list(src_objs_flow.values()))
                        # subarr_ = (objs_c @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T
                        # subflow = subarr_ - objs_c
                        # subarr_ = (objs_c @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T
                        # subflow = subarr_ - objs_c
                        # src_rigid_flow = dict(zip(src_objs_flow.keys(), subflow))
                        objs_bbox_R = dict(zip(src_objs_flow.keys(),objs_bbox_R))
                        for k,v_info in src_objs_flow.items():   
                            tgt_k = k #car_ids.get(k)
                            # if tgt_k is not None:
                            #     tgt_k = float(tgt_k)                                
                            if tgt_k in tgt_objs_flow:
                                bbox = src_objs_bbox.get(k) 
                                inds = bbox.get_point_indices_within_bounding_box(pcd.points)
                                bbox_t = tgt_objs_bbox.get(k) 
                                inds_t = bbox_t.get_point_indices_within_bounding_box(pcd_cur.points)
                                # flow[inds,:] = obj_flow
                                delat_rot = tgt_bbox_R.get(tgt_k) - src_bbox_R.get(k)
                                bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(-delat_rot)
                                obj_flow = tgt_objs_flow.get(tgt_k) - v_info @ bbox_R
                                arr_[inds,:] = arr[inds,:] @ bbox_R + obj_flow 
                                if len(fg_inds) == 0:
                                    fg_inds = inds
                                    fg_inds_t = inds_t
                                else:
                                    fg_inds += inds
                                    fg_inds_t += inds_t
                                fgrnd_inds[inds] = 1
                                fgrnd_inds_t[inds_t] = 1
                            else:
                                obj_flow = np.zeros(3)
                            # print(obj_flow)
                            
                            objs_flow.append(obj_flow)
                        # delta_flow = np.array(objs_flow) - subflow
                        objs_flow = dict(zip(src_objs_flow.keys(), objs_flow))
                        flow = 1.0 * (arr_ - arr)
                # vis_list.append(pcd)
                pcd_cur2 = o3d.geometry.PointCloud()
                pcd_cur2.points = o3d.utility.Vector3dVector(arr_)
                pcd_cur2.paint_uniform_color([0,0,1.0])

                # concat_fg_inds = []
                # for item in enumerate(fg_inds):
                #     if len(item) > 0:
                #         concat_fg_inds += item[1]

                # vis_list.append(pcd)
                # vis_list.append(pcd_cur)
                # vis_list.append(pcd_cur2)
                # o3d.visualization.draw_geometries(vis_list)
                # save_view_point(vis_list, 'camera_view.json')
                if not(enable_animation):
                    vis = o3d.visualization.Visualizer()
                    vis.create_window(width=960*2, height=640*2, left=5, top=5)
                    vis.get_render_option().background_color = np.array([0,0,0])
                    # vis.get_render_option().background_color = np.array([1, 1, 1])
                    vis.get_render_option().show_coordinate_frame = False
                    vis.get_render_option().point_size = 3.0
                    vis.get_render_option().line_width = 5.0
                # custom_draw_geometry(vis, vis_list, map_file=None, recording=False,param_file='camera_view.json', save_fov=True)
                use_flow = True
                flow = flow * 1.0
                # bg_flag = np.ones([src_pc.shape[0],1])
                # bg_flag[concat_fg_inds] = -1
                # bg_flag2 = np.argwhere(bg_flag==1)
                # ego_flow =  np.zeros(src_pc.shape)
                # ego_flow[bg_flag2[:,0],:] = flow[bg_flag2[:,0],:]
                ego_flow = (src_pc @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T - src_pc
                # ego_flow = (first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T * np.ones(src_pc.shape)
                # delta_R = o3d.geometry.get_rotation_matrix_from_xyz(first_v_raw_trans[3:] - tgt_v_raw_trans[3:]) 
                # ego_flow = (tgt_v_raw_trans[:3] - first_v_raw_trans[:3] ) @ delta_R.T @ v2s * np.ones(src_pc.shape)
                # ego_flow = (first_sensor_raw_trans[:3] - tgt_sensor_raw_trans[:3]) @ v2s * np.ones(src_pc.shape)
                # ego_flow = (-tgt_v_raw_trans[:3] + first_v_raw_trans[:3]) @ v2s * np.ones(src_pc.shape)
                ego_flow_bkp = ego_flow + tgt_v_raw_trans[:3] - first_v_raw_trans[:3]
                flow_bkp = flow + tgt_v_raw_trans[:3] - first_v_raw_trans[:3] 
                
                name_fmt = "%06d"%(ind)
                if abs(ego_flow[0,0]) > max_v:
                    max_v = abs(ego_flow[0,0])
                if save_pts:
                    # np.savez(DATASET_SF_PATH + name_fmt, pos1=src_pc, pos2=tgt_pc, ego_flow=ego_flow, gt=flow) 
                    if only_fg:
                        np.savez(DATASET_SF_PATH + name_fmt, pos1=src_pc[fg_inds,:], pos2=tgt_pc[fg_inds_t,:], ego_flow=ego_flow[fg_inds,:], gt=flow[fg_inds,:], s_fg_mask = fgrnd_inds, t_fg_mask = fgrnd_inds_t) 
                    else:
                        np.savez(DATASET_SF_PATH + name_fmt, pos1=src_pc, pos2=tgt_pc, ego_flow=ego_flow, gt=flow, s_fg_mask = fgrnd_inds, t_fg_mask = fgrnd_inds_t) 

                if show_fgrnds:
                    tgt_pc = tgt_pc[fg_inds_t,:]
                    src_pc = src_pc[fg_inds,:]
                    arr_ = arr_[fg_inds,:]
                    ego_flow = ego_flow[fg_inds,:]
                    flow = flow[fg_inds,:]
                # bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(first_v_raw_trans[3:])
                # ego_car_center = np.zeros([3,1])
                # ego_car_center[2] = -1.0 * lidar_hight + 0.5 * egoV_bbox_extend[2] #-0.4 * egoV_bbox_extend[2]
                # ego_car_bbx_c = o3d.geometry.OrientedBoundingBox(ego_car_center, bbox_R, egoV_bbox_extend)
                # # vis_list.append(ego_car_bbx_c)
                # inds = ego_car_bbx_c.get_point_indices_within_bounding_box(pcd.points)
                # valid_inds = np.ones([src_pc.shape[0]])
                # if len(inds) > 0:
                #     valid_inds[inds] = 0
                #     valid_inds = valid_inds.astype(bool)
                #     src_pc = src_pc[valid_inds,:]
                #     arr_ = arr_[valid_inds,:]
                #     ego_flow = ego_flow[valid_inds,:]
                #     flow = flow[valid_inds,:]
                #     fg_inds = (fg_inds[valid_inds, :]).astype(bool)

                # bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(tgt_v_raw_trans[3:])
                # ego_car_bbx_n = o3d.geometry.OrientedBoundingBox(ego_car_center, bbox_R, egoV_bbox_extend)
                # # vis_list.append(ego_car_bbx_n)
                # inds = ego_car_bbx_n.get_point_indices_within_bounding_box(pcd_cur.points)
                # valid_inds = np.ones([tgt_pc.shape[0]])
                # if len(inds) > 0:
                #     valid_inds[inds] = 0
                #     valid_inds = valid_inds.astype(bool)
                #     tgt_pc = tgt_pc[valid_inds,:]
                
                if use_flow:
                    # sf = sf[sample_idx1, :]
                    np_list = [src_pc, tgt_pc, arr_, flow]
                    # np_list = [src_pc[], tgt_pc, arr_, flow]
                    # np_list = [src_pc, tgt_pc, arr_, ego_flow]   
                    # np_list = [src_pc_bkp, tgt_pc_bkp, arr_ + tgt_v_raw_trans[:3], ego_flow_bkp]                 
                    # np_list = [src_pc_bkp, tgt_pc_bkp, arr_ + tgt_v_raw_trans[:3], flow_bkp]
                else:
                    np_list = [src_pc, tgt_pc, arr_,]
                    # np_list = [src_pc_bkp, tgt_pc_bkp, arr_, flow]
                
                # # import os
                # # print(os.path.abspath(os.curdir))
                # src_pc2 = src_pc - np.mean(src_pc, axis=0)
                # normp = np.sqrt(np.sum(src_pc2 * src_pc2, axis=1))
                # diameter = np.max(normp)
                # print("Define parameters used for hidden_point_removal")
                
                # # camera = [-1.0, 0.0, 5]
                # # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3).translate(camera)
                # # print("Get all points that are visible from given view point")
                # # radius = diameter * 10
                # # _, pt_map = pcd3.hidden_point_removal(camera, radius)
                # # pc = np.vstack([pc1, pc2])
                # # camera = [-45.0, -20.0, drivable_area[0,2]]
                # # camera = [45.0, -60.0, drivable_area[0,2]]
                # # camera = [-35.0, -20.0, drivable_area[0,2]]
                # # camera = [75.0, 1.5, drivable_area[0,2]]
                # # arr2_ = arr_ + flow
                # # pt_map = HPR(tgt, np.array([[0, 0, -2.5]]), 3)
                # # np_arr2 = arr2_[pt_map,:3]
                # # # print("Visualize result")
                # # # pcd3 = pcd3.select_by_index(pt_map)
                # # bool_in = in_convex_polyhedron(arr2_[pt_map,:3],np.array([[-5.0, 4.0, -2.5]]), vis) 
                # # pcd3 = drivable_area[pt_map,:]
                # # bool_in = in_convex_polyhedron(drivable_area[iou_id[:,0],:3], np.array([[0, 0, -2.5]]), vis) 
                # # camera_pos = np.hstack((first_v_raw_trans[:2], -2.75)).reshape(1,3)
                # # bool_in = in_convex_polyhedron(drivable_area[iou_id[:,0],:3],camera_pos, vis) 
                # global bbx
                # # bbx = np.array([[0,0,0], [4,0,0], [0,-4,0], [4,-1,0], [0,0,-2.5], [4,0,-2.5], [0,-4,-2.5], [4,-4,-2.5]]) * 2.0
                # bbx = np.array([[0,0,0], [4,0,0], [0,-1,0], [4,-1,0], [0,0,-2.5], [4,0,-2.5], [0,-1,-2.5], [4,-1,-2.5]]) * 0.8
                # # bbx = np.array([[0,0,0], [1,0,0], [0,-1,0], [1,-1,0], [0,0,-2.5], [1,0,-2.5], [0,-1,-2.5], [1,-1,-2.5]]) 
                # # bool_in, _ = validDrivableArea(bbx, tgt_pc)
                # bool_in, _ = validateDrivableArea2(bbx, tgt_pc)

                # sphere = o3d.geometry.TriangleMesh.create_sphere(2).translate(np.array([0, 0, -2.5]))
                # if bool_in:
                #     sphere.paint_uniform_color([0.2,0.2,1.0])
                # else:
                #     sphere.paint_uniform_color([1,0,0])
                # vis_list += [sphere]

                # np_list = vis_list
                use_flow = False
                Draw_SceneFlow(vis, np_list+vis_list, use_fast_vis = not(use_flow), use_pred_t=True, use_custom_color=True, use_flow=use_flow, param_file='camera_view.json')
        sub_dir_cnt += 1
    # np.savez('./data/data_0308_1058_Vinfo', cvInfo = current_v_info, nvInfo = next_v_info)


def Generate_Random_SceneFlow_Paper(start = 3, step = 3, lidar_hight = 3.4, prefixed_V_name=None, show_active_egoV=False, save_pts=False, enable_animation = True, egoV_flag=False, rm_ground = False, show_fgrnds = True, egoV_file_path = './data/data_Vinfo0307', GT_SUB_DATASET='vehicle.mini.cooper_s_2021_254'):
    if step == 1:
        SF_DIR = '/SF/'
    elif step == 2:
        SF_DIR = '/SF2/'
    elif step == 3:
        SF_DIR = '/SF3/'
    else:
        SF_DIR = 'TEST'

    # show_fgrnds = True
    
    start = start
    drivable_areas = odom_utils.readPointCloud('./dataset/town02-map.bin') #road map
    drivable_area = drivable_areas[:,:3]
    drivable_area[:,-1] = -1.0 * lidar_hight
    road_id = drivable_areas[:,-1]
    colors = []
    for iCnt in road_id:
        if iCnt == 0:
            colors += [[1,0,0]]
        elif iCnt == 1:
            colors += [[0,1,0]]
        elif iCnt == 2:
            colors += [[0,0,1]]
        elif iCnt == 3:
            colors += [[1,1,0]]
        else:
            colors += [[1,0,1]]
    #指定实验道路
    iou_id = np.argwhere(road_id == 1)

    drivable_centers = np.load('./dataset/np_c.npy')#odom_utils.readPointCloud('./dataset/town02-road-center.bin')
    drivable_center = drivable_centers[:,:3]
    drivable_center[:, -1] = -1.0 * lidar_hight
    drivable_center[:,1] *= -1.0
    road_id2 = []
    road_id2 = drivable_centers[:,-1]
    ref_drivable_center_inds = np.argwhere(drivable_centers[:,-1] == 1)
    colors2 = []
    for iCnt in road_id2:
        if iCnt < 1:
            colors2 += [[1,0,0]]
        elif iCnt >= 1  and iCnt < 2:
            colors2 += [[0,1,0]]
        elif iCnt >= 2 and iCnt < 3:
            colors2 += [[0,0,1]]
        elif iCnt >= 3 and iCnt < 4:
            colors2 += [[1,1,0]]
        else:
            colors2 += [[1,0,1]]

    
    # drivable_area[:,0] *= -1
    pcd_drivable_area = o3d.geometry.PointCloud()
    # pcd_drivable_area.points = o3d.utility.Vector3dVector(drivable_area)
    drivable_area_bkp = drivable_area
    pcd_drivable_area.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.dtype('f4')))

    pcd_drivable_center = o3d.geometry.PointCloud()
    # pcd_drivable_area.points = o3d.utility.Vector3dVector(drivable_center)
    drivable_center_bkp = drivable_center
    pcd_drivable_center.colors = o3d.utility.Vector3dVector(np.array(colors2, dtype=np.dtype('f4')))
                
    # Compute_PairwiseSceneFlow(start, 5)

    # Translate frame coordinates from left-hand to right-hand
    v_l2r = np.array([1., -1., 1., 1., 1., 1.])
    s_l2r = np.array([1, 1., -1., 1., 1., 1., 1.])
    v2s = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    # enable_animation = True
    if enable_animation:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=960*2, height=640*2, left=5, top=5)
        vis.get_render_option().background_color = np.array([0, 0, 0])
        # vis.get_render_option().background_color = np.array([1, 1, 1])
        vis.get_render_option().show_coordinate_frame = False
        vis.get_render_option().point_size = 3.0
        vis.get_render_option().line_width = 5.0
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, -1.0 * lidar_hight])
    
    # Generate the log data from the DIR:tmp/xxx/xxxxrgb_lable/xxxx 
    build_lidar_fov = False

    # Generate the scene flow data of the only camera field-of-view
    use_rgb_fov = False
    
    # rm_ground = False
    # trans_pattern = r"(?P<v_name>[vehicle]+.\w+.\w+) (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
    # r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
    #     r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
    #         r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"\
    
    rawtrans_files = glob.glob(os.path.join(DATASET_RAWTRANS_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
    rawtrans_files.sort()

    rawtrans_files2 = glob.glob(os.path.join(DATASET_RAWTRANS_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
    rawtrans_files2.sort()

    # rawtrans_files2 = glob.glob(os.path.join(DATASET_RAWTRANS_PATH2, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
    # rawtrans_files2.sort()
    # if not(use_rgb_fov):
    #     for r in re.findall(trans_pattern, log):
    #         # print(r[0])
    #         det_car_info.append(int(float(r[0])))
    # print(os.path.abspath(os.curdir))
    sub_dir_cnt = 0
    max_v = -100.0
    # v_info_cnt = 0
    current_v_info = []
    next_v_info = []
    v_pattern = r"vehicle.(?P<nonsense>[\s\S]+)_(?P<v_id>\d+)"
    for sub_dir in os.listdir(DATASET_PATH):
        if 'global_label' in sub_dir or 'img' in sub_dir:
            continue 
        # print(sub_dir)

        SUB_DATASET = 'vehicle.mini.cooper_s_2021_1520' #sub_dir #'vehicle.tesla.cybertruck_266'#'vehicle.citroen.c3_271'
        SUB_DATASET = GT_SUB_DATASET #'vehicle.mini.cooper_s_2021_254' #sub_dir #'vehicle.tesla.cybertruck_266'#'vehicle.citroen.c3_271'
        # SUB_DATASET = 'vehicle.mini.cooper_s_2021_406' #sub_dir #'vehicle.tesla.cybertruck_266'#'vehicle.citroen.c3_271'
        SUB_DATASET2 = 'vehicle.mini.cooper_s_2021_402' #sub_dir #'vehicle.tesla.cybertruck_266'#'vehicle.citroen.c3_271'
        SUB_DATASET2 = GT_SUB_DATASET #'vehicle.mini.cooper_s_2021_254' #sub_dir #'vehicle.tesla.cybertruck_266'#'vehicle.citroen.c3_271'
        if sub_dir not in SUB_DATASET:
            continue
        v_pattern = r"[vehicle]+.(?P<name>\w+.\w+)_(?P<v_id>\d+)"
        v = re.findall(v_pattern, SUB_DATASET)[0]
        DATASET_PC_PATH = DATASET_PATH + SUB_DATASET + '/velodyne/'
        DATASET_LBL_PATH = DATASET_PATH + SUB_DATASET + '/label00/'

        v2 = re.findall(v_pattern, SUB_DATASET2)[0]
        DATASET_PC_PATH2 = DATASET_PATH + SUB_DATASET2 + '/velodyne/'
        DATASET_LBL_PATH2 = DATASET_PATH + SUB_DATASET2 + '/label00/'

        # RESULT_PATH = './results/' + SUB_DATASET
        trans_pattern = v[0] + " " + v[1] + r" (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"
        lidar_pattern = r"sensor.lidar.ray_cast (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 " + v[1]
        
        lidar_semantic_pattern = r"sensor.lidar.ray_cast_semantic (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 0 " + v[1]

        trans_pattern2 = v2[0] + " " + v2[1] + r" (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"
        lidar_pattern2 = r"sensor.lidar.ray_cast (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 " + v2[1]
        
        lidar_semantic_pattern2 = r"sensor.lidar.ray_cast_semantic (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 0 " + v2[1]
        
        glb_label_pattern2 = v[0] + " " + v[1] + r" (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
    r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
        r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<extbbx_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<extbbx_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<extbbx_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<bbx_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"

        # next_ind = ind + 2
        pc_files = glob.glob(os.path.join(DATASET_PC_PATH, '*.bin'))#os.listdir(DATASET_PC_PATH)
        # rawtrans_files = glob.glob(os.path.join(DATASET_RAWTRANS_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
        label_files = glob.glob(os.path.join(DATASET_LBL_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
        # print("Current DIR:" + DATASET_PC_PATH)
        pc_files.sort()
        # rawtrans_files.sort()
        label_files.sort()

        # next_ind = ind + 2
        pc_files2 = glob.glob(os.path.join(DATASET_PC_PATH, '*.bin'))#os.listdir(DATASET_PC_PATH)
        # rawtrans_files = glob.glob(os.path.join(DATASET_RAWTRANS_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
        label_files2 = glob.glob(os.path.join(DATASET_LBL_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
        # print("Current DIR:" + DATASET_PC_PATH)
        pc_files2.sort()
        # rawtrans_files.sort()
        label_files2.sort()

        total_cmd_num = len(pc_files)
        frame_start,frame_end,frame_hz = 10,-5,1
        spawn_points = []
        cnt = 0

        ego_v_info = re.findall(v_pattern, sub_dir)
        with open(rawtrans_files[0], 'r') as file:
                raw_log = file.read()
        r = re.findall(glb_label_pattern, raw_log)
        for r in re.findall(glb_label_pattern, raw_log):
            if ego_v_info[0][0] in r[0] and ego_v_info[0][1] in r[1]:
                l = float(r[8])
                w = float(r[9])
                h = float(r[10])
                egoV_bbox_extend = np.array([l, w, h]).reshape(3,1) * 2.0
        
        
        DATASET_SF_PATH = RESULT_PATH2 + SF_DIR + "%02d"%(sub_dir_cnt) + '/'
        if not os.path.exists(DATASET_SF_PATH) and save_pts:
            os.makedirs(DATASET_SF_PATH)

        if build_lidar_fov:
            write_labels('record2021_0716_2146/', sub_dir, frame_start,frame_end, frame_hz, index=1)
        else:
            # step = 1
            for ind in range(start, len(pc_files)-step, step):
                next_ind = ind + step
                if next_ind >= len(label_files):
                        break
                print("Current DIR:" + DATASET_PC_PATH + ' Frame:' +str(ind))
                # tmp = cmd_traj[ind, ]
                # saveVehicleInfo()
                
                # # For writting the vehicle info ln:1357-1558
                # ivFlag = 0
                # for i in [ind, next_ind]:
                #     # if next_ind >= len(label_files):
                #     #     break
                #     with open(label_files[i], 'r') as file:
                #             log = file.read()
                #     with open(rawtrans_files[i], 'r') as file:
                #         raw_log = file.read()
               
                #     for r in re.findall(glb_label_pattern, raw_log):
                #         # print(r)
                #         if ego_v_info[0][0] in r[0] and ego_v_info[0][1] in r[1]:
                #             if ivFlag % 2 == 0:
                #                 current_v_info.append(r)
                #             else:
                #                 next_v_info.append(r)
                #             ivFlag += 1

                # continue
   
                # Compute_PairwiseSceneFlow(ind, next_ind, trans_pattern, lidar_pattern, DATASET_PC_PATH, DATASET_LBL_PATH)
                tmp = ind 
                ind = next_ind
                next_ind = tmp
                vis_list = []
                first_v_raw_trans = odom_utils.readRawData(rawtrans_files[ind], trans_pattern) * v_l2r
                first_sensor_raw_trans = odom_utils.readRawData(rawtrans_files[ind], lidar_semantic_pattern) * s_l2r

                src_pc = odom_utils.readPointCloud(pc_files[ind])[:, :3]
                src_pc_bkp = src_pc + first_v_raw_trans[:3]
                if rm_ground:
                    is_ground = np.logical_not(src_pc[:, 2] <= -1.0 * (lidar_hight-0.1))
                    src_pc = src_pc[is_ground, :] 
                # np.save('./data/np_src', src_pc)
                # For the carla origin coordatinate 
                # first_v_raw_trans[2] += 2.5
                # src_pc += first_v_raw_trans[:3] @ np.array([[0,-1,0],[1,0,0],[0,0,1]])
                
                # ## For the carla original coordinate
                # drivable_area = drivable_area_bkp @ np.array([[0,-1,0],[1,0,0],[0,0,1]])

                ## For the first car frame as the original coordinate
                drivable_area = (drivable_area_bkp - first_v_raw_trans[:3])\
                     @ odom_utils.rotation_from_euler_zyx(-first_v_raw_trans[5], -first_v_raw_trans[4],first_v_raw_trans[3])#
                pcd_drivable_area.points = o3d.utility.Vector3dVector(drivable_area)
                # np.save('./data/np_drivable_area', np.hstack((drivable_area, (drivable_areas[:,-1]).reshape((-1,1)))))
                
                ## For the carla original coordinate
                # drivable_center = drivable_center_bkp @ np.array([[0,-1,0],[1,0,0],[0,0,1]])

                # For the first car frame as the original coordinate
                drivable_center = (drivable_center_bkp - first_v_raw_trans[:3])\
                     @ odom_utils.rotation_from_euler_zyx(-first_v_raw_trans[5], -first_v_raw_trans[4],first_v_raw_trans[3])#
                pcd_drivable_center.points = o3d.utility.Vector3dVector(drivable_center)
                # pcd_drivable_area.paint_uniform_color([1.0,0,0.0])
                ref_drivable_center = drivable_centers[ref_drivable_center_inds[:,0],:3]

                # np.save('./data/np_drivable_center', np.hstack((drivable_center, (drivable_centers[:,-1]).reshape((-1,1)))))

                arr = src_pc
                
                src_R_vec = first_sensor_raw_trans[4:]
                src_R = o3d.geometry.get_rotation_matrix_from_xyz(src_R_vec)
                src_R_inv = np.linalg.inv(src_R)
                tmp_pcd = o3d.geometry.PointCloud()
                tmp_pcd.points = o3d.utility.Vector3dVector(src_pc)
                #For the carla origin coordatinate 
                # pcd.rotate(src_R_inv, first_v_raw_trans[:3])

                # For the first frame as the origin coordinate
                # pcd.rotate(src_R_inv, np.zeros(3))
                # pcd.paint_uniform_color([0,1,0])

                # ego_pcd = o3d.geometry.PointCloud()
                # ego_pcd.points = o3d.utility.Vector3dVector(first_v_raw_trans[:3].reshape(-1,3))
                # ego_pcd.paint_uniform_color([0,1,1])
               
                # drivable_area =  drivable_area - (np.array([[spawn_points[ind][0], -spawn_points[ind][1], 0]]))
                
                vis_list = [mesh_frame, pcd_drivable_area, pcd_drivable_center]
                # vis_list = [mesh_frame, pcd]
                
                tgt_v_raw_trans = odom_utils.readRawData(rawtrans_files2[next_ind], trans_pattern2) * v_l2r

                print(first_v_raw_trans[:3])
                print(tgt_v_raw_trans[:3])
                # tgt_v_raw_trans2 = odom_utils.readRawData(rawtrans_files2[next_ind], trans_pattern2) * v_l2r
                # tgt_v_raw_trans0 = odom_utils.readRawData(rawtrans_files[next_ind], trans_pattern2) * v_l2r
                # print(first_v_raw_trans)
                # print(tgt_v_raw_trans0)
                # print(tgt_v_raw_trans)
                # print(tgt_v_raw_trans2)
                tgt_sensor_raw_trans = odom_utils.readRawData(rawtrans_files2[next_ind], lidar_semantic_pattern2) * s_l2r
                tgt_pc = odom_utils.readPointCloud(pc_files2[next_ind])[:, :3]

                tgt_pc_bkp = tgt_pc + tgt_v_raw_trans[:3]
                if rm_ground:
                    is_ground = np.logical_not(tgt_pc[:, 2] <= -1.0 * (lidar_hight - 0.05))
                    tgt_pc = tgt_pc[is_ground, :] 

                #For the carla origin coordatinate 
                # tgt_v_raw_trans[2] += 2.5
                # tgt_pc += tgt_v_raw_trans[:3] @ np.array([[0,-1,0],[1,0,0],[0,0,1]])

                tgt_R_vec = tgt_sensor_raw_trans[4:]
                tgt_R = o3d.geometry.get_rotation_matrix_from_xyz(tgt_R_vec)
                tgt_R_inv = np.linalg.inv(tgt_R)
                tmp_pcd_cur = o3d.geometry.PointCloud()
                tmp_pcd_cur.points = o3d.utility.Vector3dVector(tgt_pc)

                #For the carla origin coordatinate 
                # pcd_cur.rotate(tgt_R_inv, tgt_v_raw_trans[:3] )

                #For the first frame as the origin coordinate
                # pcd_cur.rotate(tgt_R_inv, np.zeros(3))
                # pcd_cur.translate(tgt_v_raw_trans[:3] - first_v_raw_trans[:3])
                # tmp_pcd_cur.paint_uniform_color([1,0,0])

                first_v_raw_rot = first_v_raw_trans[3:]
                # first_v_raw_rot[-1] *= -1.0
                src_ego_bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(first_v_raw_rot)
                # src_ego_bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(first_v_raw_trans[3:])
                ego_car_center = np.zeros([3,1])
                ego_car_center[2] = -1.0 * lidar_hight + 0.5 * egoV_bbox_extend[2]
                ego_car_bbx_c = o3d.geometry.OrientedBoundingBox(ego_car_center, src_ego_bbox_R, egoV_bbox_extend)
                # vis_list.append(ego_car_bbx_c)
                inds = ego_car_bbx_c.get_point_indices_within_bounding_box(tmp_pcd.points)
                valid_inds = np.ones([src_pc.shape[0]])
                if len(inds) > 0:
                    valid_inds[inds] = 0
                    valid_inds = valid_inds.astype(bool)
                    src_pc = src_pc[valid_inds,:]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(src_pc)
                #For the carla origin coordatinate 
                # pcd.rotate(src_R_inv, first_v_raw_trans[:3])

                # For the first frame as the origin coordinate
                # pcd.rotate(src_R_inv, np.zeros(3))
                pcd.paint_uniform_color([0,1,0])

                tgt_v_raw_rot = tgt_v_raw_trans[3:]
                # first_v_raw_rot[-1] *= -1.0
                tgt_ego_bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(tgt_v_raw_rot)
                tgt_ego_bbx_center = (np.zeros([3]) @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T
                tgt_ego_bbx_center[2] = -1.0 * lidar_hight + 0.5 * egoV_bbox_extend[2]
                tgt_ego_car_bbx_c = o3d.geometry.OrientedBoundingBox(tgt_ego_bbx_center, tgt_ego_bbox_R, egoV_bbox_extend)
                # vis_list.append(ego_car_bbx_c)
                tgt_inds = tgt_ego_car_bbx_c.get_point_indices_within_bounding_box(tmp_pcd_cur.points)
                tgt_valid_inds = np.ones([tgt_pc.shape[0]])
                if len(tgt_inds) > 0:
                    tgt_valid_inds[tgt_inds] = 0
                    tgt_valid_inds = tgt_valid_inds.astype(bool)
                    tgt_pc = tgt_pc[tgt_valid_inds,:]
                pcd_cur = o3d.geometry.PointCloud()
                pcd_cur.points = o3d.utility.Vector3dVector(tgt_pc)

                #For the carla origin coordatinate 
                # pcd_cur.rotate(tgt_R_inv, tgt_v_raw_trans[:3] )

                #For the first frame as the origin coordinate
                # pcd_cur.rotate(tgt_R_inv, np.zeros(3))
                # pcd_cur.translate(tgt_v_raw_trans[:3] - first_v_raw_trans[:3])
                pcd_cur.paint_uniform_color([1,0,0])
                
                arr_ = (src_pc @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T

                alternation_flag = 2
                oth_cars_info = []
                fgrnd_inds = np.zeros(src_pc.shape[0])
                fg_inds = []
                fgrnd_inds_t = np.zeros(tgt_pc.shape[0])
                fg_inds_t = []
                for  iflag, i in enumerate([ind, next_ind]):
                    # if next_ind >= len(label_files):
                    #     break
                    if iflag % alternation_flag == 0:
                        with open(label_files[i], 'r') as file:
                                log = file.read()
                    else:
                        with open(label_files2[i], 'r') as file:
                                log = file.read()
                    

                    det_car_info = []  
                    if use_rgb_fov:
                        for r in re.findall(rgb_label_pattern, log):
                            det_car_info.append(int(r[-1]))
                    # else:
                    #     for r in re.findall(lidar_label_pattern, log):
                    #         # print(r[0])
                    #         det_car_info.append(int(float(r[0])))
                    if iflag % alternation_flag == 0:
                        with open(rawtrans_files[i], 'r') as file:
                            raw_log = file.read()
                    else:
                        with open(rawtrans_files2[i], 'r') as file:
                            raw_log = file.read()
                    oth_car_info = []
                    for r in re.findall(glb_label_pattern, raw_log):
                        # print(r[0])
                        if use_rgb_fov:
                            if int(r[1]) in det_car_info:
                                oth_car_info.append([int(r[1])] + list(map(float, r[2:5])) + [np.radians(float(r[5])),\
                                    np.radians(float(r[6])), np.radians(float(r[7]))] + list(map(float, r[-4:])))
                        else:
                            if r[1] != v[1]:
                                oth_car_info.append([int(r[1])] + list(map(float, r[2:5])) + [np.radians(float(r[5])),\
                                    np.radians(float(r[6])), np.radians(float(r[7]))] + list(map(float, r[-4:])))
                    if len(oth_car_info) > 0:
                        oth_cars_info.append(oth_car_info)
                    # print(oth_car_info)

                # print(first_v_raw_trans)
                # #######TEST FOR INVERSE TRANSFORM TO CARLR COORDINATE
                # inv_ego_pos = first_v_raw_trans[:3]
                # print(inv_ego_pos)
                    
                show_flag = 1
                lcl_R = [src_R, tgt_R]
                lcl_R_inv = [src_R_inv, tgt_R_inv]
                lcl_rot_v = [first_v_raw_trans[3:], tgt_v_raw_trans[3:]]
                lcl_trans_v = [first_v_raw_trans[:3], tgt_v_raw_trans[:3]]
                colors = [[0,1,0], [1,0,0]]
                objs_center = []
                objs_flow = []
                objs_bbx = []
                objs_bbox_R = []
                # fg_inds = []
                # if show_flag:
                for iv in range(len(oth_cars_info)):
                    if len(oth_cars_info) <= 1:
                        flow = arr_ - arr
                        # continue
                    else:
                        oth_v_loc = np.array(oth_cars_info[iv])[:,1:4] * np.array([1.0, -1.0, 1.0])
                        oth_v_loc[:,-1] = oth_v_loc[:,-1] + np.array(oth_cars_info[iv])[:,-1] - lidar_hight
                        oth_v_rot = np.array(oth_cars_info[iv])[:,4:7]
                        oth_v_bbx_ext = np.array(oth_cars_info[iv])[:,-4:]

                        # objs_center = []
                        obj_flow = []
                        obj_bbx = []
                        obj_bbox_R = []

                        for ibbx in range(oth_v_loc.shape[0]):
                            # bbox_v_rot = -1.0 * oth_v_rot[ibbx, :] + np.array([0.0, 0.0, 2.0 * np.pi])
                            bbox_v_rot = lcl_rot_v[iv] - oth_v_rot[ibbx, :]
                            bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(bbox_v_rot)
                            #For the local (observation) coordinates
                            obj_bbox_R.append(bbox_v_rot)
                       
                            bbox_extend = oth_v_bbx_ext[ibbx, :-1] * 2.0
                            #For the local (observation) coordinates
                            bbox_center = lcl_R[iv] @ (oth_v_loc[ibbx,:] - lcl_trans_v[iv])
                            #For the original coordinates
                            # bbox_center = lcl_R[iv] @ oth_v_loc[ibbx,:]
                            # print(bbox_center)
                            obj_flow.append(np.hstack((np.array(oth_cars_info[iv][ibbx][0]), np.array(bbox_center))))
                            #For the local (observation) coordinates
                            bbox = o3d.geometry.OrientedBoundingBox(bbox_center, bbox_R, bbox_extend)
                            #For the original coordinates
                            # bbox = o3d.geometry.OrientedBoundingBox(bbox_center, bbox_R, bbox_extend)
                            obj_bbx.append(bbox)
                            bbox.color = colors[iv]
                            # bbox.LineSet = 
                            vis_list.append(bbox)
                        objs_flow.append(obj_flow)
                        objs_bbx.append(obj_bbx)
                        objs_bbox_R.append(obj_bbox_R)
                    
                    # fg_inds = np.zeros(src_pc.shape[0])
                    if len(objs_flow) <= 1:
                        flow = arr_ - arr
                    else:
                        src_objs_flow = np.array(objs_flow[0])
                        src_objs_bbox = dict(zip(src_objs_flow[:,0], objs_bbx[0]))
                        src_objs_flow = dict(zip(src_objs_flow[:,0], src_objs_flow[:,1:]))
                        src_bbox_R = dict(zip(src_objs_flow.keys(),objs_bbox_R[0]))                    
                    
                        tgt_objs_flow = np.array(objs_flow[1])
                        tgt_objs_bbox = dict(zip(tgt_objs_flow[:,0], objs_bbx[1]))
                        tgt_objs_flow = dict(zip(tgt_objs_flow[:,0], tgt_objs_flow[:,1:]))
                        tgt_bbox_R = dict(zip(tgt_objs_flow.keys(),objs_bbox_R[1]))

                        objs_flow = []
                        # objs_c = np.array(list(src_objs_flow.values()))
                        # subarr_ = (objs_c @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T
                        # subflow = subarr_ - objs_c
                        # subarr_ = (objs_c @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T
                        # subflow = subarr_ - objs_c
                        # src_rigid_flow = dict(zip(src_objs_flow.keys(), subflow))
                        objs_bbox_R = dict(zip(src_objs_flow.keys(),objs_bbox_R))
                        for k,v_info in src_objs_flow.items():   
                            tgt_k = k #car_ids.get(k)
                            # if tgt_k is not None:
                            #     tgt_k = float(tgt_k)                                
                            if tgt_k in tgt_objs_flow:
                                bbox = src_objs_bbox.get(k) 
                                inds = bbox.get_point_indices_within_bounding_box(pcd.points)
                                bbox_t = tgt_objs_bbox.get(k) 
                                inds_t = bbox_t.get_point_indices_within_bounding_box(pcd_cur.points)
                                # flow[inds,:] = obj_flow
                                delat_rot = tgt_bbox_R.get(tgt_k) - src_bbox_R.get(k)
                                bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(-delat_rot)
                                obj_flow = tgt_objs_flow.get(tgt_k) - v_info @ bbox_R
                                arr_[inds,:] = arr[inds,:] @ bbox_R + obj_flow 
                                if len(fg_inds) == 0:
                                    fg_inds = inds
                                    fg_inds_t = inds_t
                                else:
                                    fg_inds += inds
                                    fg_inds_t += inds_t
                                fgrnd_inds[inds] = 1
                                fgrnd_inds_t[inds_t] = 1
                            else:
                                obj_flow = np.zeros(3)
                            # print(obj_flow)
                            
                            objs_flow.append(obj_flow)
                        # delta_flow = np.array(objs_flow) - subflow
                        objs_flow = dict(zip(src_objs_flow.keys(), objs_flow))
                        flow = 1.0 * (arr_ - arr)
                # vis_list.append(pcd)
                pcd_cur2 = o3d.geometry.PointCloud()
                pcd_cur2.points = o3d.utility.Vector3dVector(arr_)
                pcd_cur2.paint_uniform_color([0,0,1.0])

                # concat_fg_inds = []
                # for item in enumerate(fg_inds):
                #     if len(item) > 0:
                #         concat_fg_inds += item[1]

                # vis_list.append(pcd)
                # vis_list.append(pcd_cur)
                # vis_list.append(pcd_cur2)
                # o3d.visualization.draw_geometries(vis_list)
                # save_view_point(vis_list, 'camera_view.json')
                if not(enable_animation):
                    vis = o3d.visualization.Visualizer()
                    vis.create_window(width=960*2, height=640*2, left=5, top=5)
                    vis.get_render_option().background_color = np.array([0,0,0])
                    # vis.get_render_option().background_color = np.array([1, 1, 1])
                    vis.get_render_option().show_coordinate_frame = False
                    vis.get_render_option().point_size = 3.0
                    vis.get_render_option().line_width = 5.0
                # custom_draw_geometry(vis, vis_list, map_file=None, recording=False,param_file='camera_view.json', save_fov=True)
                use_flow = True
                flow = flow * 1.0
                # bg_flag = np.ones([src_pc.shape[0],1])
                # bg_flag[concat_fg_inds] = -1
                # bg_flag2 = np.argwhere(bg_flag==1)
                # ego_flow =  np.zeros(src_pc.shape)
                # ego_flow[bg_flag2[:,0],:] = flow[bg_flag2[:,0],:]
                ego_flow = (src_pc @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T - src_pc
                # ego_flow = (first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T * np.ones(src_pc.shape)
                # delta_R = o3d.geometry.get_rotation_matrix_from_xyz(first_v_raw_trans[3:] - tgt_v_raw_trans[3:]) 
                # ego_flow = (tgt_v_raw_trans[:3] - first_v_raw_trans[:3] ) @ delta_R.T @ v2s * np.ones(src_pc.shape)
                # ego_flow = (first_sensor_raw_trans[:3] - tgt_sensor_raw_trans[:3]) @ v2s * np.ones(src_pc.shape)
                # ego_flow = (-tgt_v_raw_trans[:3] + first_v_raw_trans[:3]) @ v2s * np.ones(src_pc.shape)
                ego_flow_bkp = ego_flow + tgt_v_raw_trans[:3] - first_v_raw_trans[:3]
                flow_bkp = flow + tgt_v_raw_trans[:3] - first_v_raw_trans[:3] 
                
                name_fmt = "%06d"%(ind)
                if abs(ego_flow[0,0]) > max_v:
                    max_v = abs(ego_flow[0,0])
                if save_pts:
                    # np.savez(DATASET_SF_PATH + name_fmt, pos1=src_pc, pos2=tgt_pc, ego_flow=ego_flow, gt=flow) 
                    if only_fg:
                        np.savez(DATASET_SF_PATH + name_fmt, pos1=src_pc[fg_inds,:], pos2=tgt_pc[fg_inds_t,:], ego_flow=ego_flow[fg_inds,:], gt=flow[fg_inds,:], s_fg_mask = fgrnd_inds, t_fg_mask = fgrnd_inds_t) 
                    else:
                        np.savez(DATASET_SF_PATH + name_fmt, pos1=src_pc, pos2=tgt_pc, ego_flow=ego_flow, gt=flow, s_fg_mask = fgrnd_inds, t_fg_mask = fgrnd_inds_t) 

                if show_fgrnds:
                    tgt_pc = tgt_pc[fg_inds_t,:]
                    src_pc = src_pc[fg_inds,:]
                    arr_ = arr_[fg_inds,:]
                    ego_flow = ego_flow[fg_inds,:]
                    flow = flow[fg_inds,:]
                # bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(first_v_raw_trans[3:])
                # ego_car_center = np.zeros([3,1])
                # ego_car_center[2] = -1.0 * lidar_hight + 0.5 * egoV_bbox_extend[2] #-0.4 * egoV_bbox_extend[2]
                # ego_car_bbx_c = o3d.geometry.OrientedBoundingBox(ego_car_center, bbox_R, egoV_bbox_extend)
                # # vis_list.append(ego_car_bbx_c)
                # inds = ego_car_bbx_c.get_point_indices_within_bounding_box(pcd.points)
                # valid_inds = np.ones([src_pc.shape[0]])
                # if len(inds) > 0:
                #     valid_inds[inds] = 0
                #     valid_inds = valid_inds.astype(bool)
                #     src_pc = src_pc[valid_inds,:]
                #     arr_ = arr_[valid_inds,:]
                #     ego_flow = ego_flow[valid_inds,:]
                #     flow = flow[valid_inds,:]
                #     fg_inds = (fg_inds[valid_inds, :]).astype(bool)

                # bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(tgt_v_raw_trans[3:])
                # ego_car_bbx_n = o3d.geometry.OrientedBoundingBox(ego_car_center, bbox_R, egoV_bbox_extend)
                # # vis_list.append(ego_car_bbx_n)
                # inds = ego_car_bbx_n.get_point_indices_within_bounding_box(pcd_cur.points)
                # valid_inds = np.ones([tgt_pc.shape[0]])
                # if len(inds) > 0:
                #     valid_inds[inds] = 0
                #     valid_inds = valid_inds.astype(bool)
                #     tgt_pc = tgt_pc[valid_inds,:]
                
                if use_flow:
                    # sf = sf[sample_idx1, :]
                    np_list = [src_pc, tgt_pc, arr_, flow]
                    # np_list = [src_pc[], tgt_pc, arr_, flow]
                    # np_list = [src_pc, tgt_pc, arr_, ego_flow]   
                    # np_list = [src_pc_bkp, tgt_pc_bkp, arr_ + tgt_v_raw_trans[:3], ego_flow_bkp]                 
                    # np_list = [src_pc_bkp, tgt_pc_bkp, arr_ + tgt_v_raw_trans[:3], flow_bkp]
                else:
                    np_list = [src_pc, tgt_pc, arr_,]
                    # np_list = [src_pc_bkp, tgt_pc_bkp, arr_, flow]
                
                # import os
                # print(os.path.abspath(os.curdir))
                src_pc2 = src_pc - np.mean(src_pc, axis=0)
                normp = np.sqrt(np.sum(src_pc2 * src_pc2, axis=1))
                diameter = np.max(normp)
                print("Define parameters used for hidden_point_removal")
                
                # camera = [-1.0, 0.0, 5]
                # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3).translate(camera)
                # print("Get all points that are visible from given view point")
                # radius = diameter * 10
                # _, pt_map = pcd3.hidden_point_removal(camera, radius)
                # pc = np.vstack([pc1, pc2])
                # camera = [-45.0, -20.0, drivable_area[0,2]]
                # camera = [45.0, -60.0, drivable_area[0,2]]
                # camera = [-35.0, -20.0, drivable_area[0,2]]
                # camera = [75.0, 1.5, drivable_area[0,2]]
                # arr2_ = arr_ + flow
                # pt_map = HPR(tgt, np.array([[0, 0, -2.5]]), 3)
                # np_arr2 = arr2_[pt_map,:3]
                # # print("Visualize result")
                # # pcd3 = pcd3.select_by_index(pt_map)
                # bool_in = in_convex_polyhedron(arr2_[pt_map,:3],np.array([[-5.0, 4.0, -2.5]]), vis) 
                # pcd3 = drivable_area[pt_map,:]
                # bool_in = in_convex_polyhedron(drivable_area[iou_id[:,0],:3], np.array([[0, 0, -2.5]]), vis) 
                # camera_pos = np.hstack((first_v_raw_trans[:2], -2.75)).reshape(1,3)
                # bool_in = in_convex_polyhedron(drivable_area[iou_id[:,0],:3],camera_pos, vis) 
                global bbx
                # bbx = np.array([[0,0,0], [4,0,0], [0,-4,0], [4,-1,0], [0,0,-2.5], [4,0,-2.5], [0,-4,-2.5], [4,-4,-2.5]]) * 2.0
                bbx = np.array([[0,0,0], [4,0,0], [0,-1,0], [4,-1,0], [0,0,-2.5], [4,0,-2.5], [0,-1,-2.5], [4,-1,-2.5]]) * 0.8
                # bbx = np.array([[0,0,0], [1,0,0], [0,-1,0], [1,-1,0], [0,0,-2.5], [1,0,-2.5], [0,-1,-2.5], [1,-1,-2.5]]) 
                # bool_in, _ = validDrivableArea(bbx, tgt_pc)
                bool_in, _ = validateDrivableArea2(bbx, tgt_pc)

                sphere = o3d.geometry.TriangleMesh.create_sphere(2).translate(np.array([0, 0, -2.5]))
                if bool_in:
                    sphere.paint_uniform_color([0.2,0.2,1.0])
                else:
                    sphere.paint_uniform_color([1,0,0])
                vis_list += [sphere]

                # np_list = vis_list
                use_flow = False
                Draw_SceneFlow(vis, np_list+vis_list, use_fast_vis = not(use_flow), use_pred_t=True, use_custom_color=True, use_flow=use_flow, param_file='camera_view.json')
        sub_dir_cnt += 1
    # np.savez('./data/data_0308_1058_Vinfo', cvInfo = current_v_info, nvInfo = next_v_info)


def Generate_Active_SceneFlow_Paper(start = 3, step = 3, lidar_hight = 3.4, prefixed_V_name=None, show_active_egoV=False, save_pts=False, enable_animation = True, egoV_flag=False, rm_ground = False, show_fgrnds = True, egoV_file_path = './data/data_Vinfo0307', GT_SUB_DATASET='vehicle.mini.cooper_s_2021_254', ACTIVE_SUB_DATASET='vehicle.mini.cooper_s_2021_1280'):
    img_cnt = 0
    if step == 1:
        SF_DIR = '/SF/'
    elif step == 2:
        SF_DIR = '/SF2/'
    elif step == 3:
        SF_DIR = '/SF3/'
    else:
        SF_DIR = 'TEST'

    # show_fgrnds = True
    
    start = start
    drivable_areas = odom_utils.readPointCloud('./dataset/town02-map.bin') #road map
    drivable_area = drivable_areas[:,:3]
    drivable_area[:,-1] = -1.0 * lidar_hight
    road_id = drivable_areas[:,-1]
    colors = []
    for iCnt in road_id:
        if iCnt == 0:
            colors += [[1,0,0]]
        elif iCnt == 1:
            colors += [[0,1,0]]
        elif iCnt == 2:
            colors += [[0,0,1]]
        elif iCnt == 3:
            colors += [[1,1,0]]
        else:
            colors += [[1,0,1]]
    #指定实验道路
    iou_id = np.argwhere(road_id == 1)

    drivable_centers = np.load('./dataset/np_c.npy')#odom_utils.readPointCloud('./dataset/town02-road-center.bin')
    drivable_center = drivable_centers[:,:3]
    drivable_center[:, -1] = -1.0 * lidar_hight
    drivable_center[:,1] *= -1.0
    road_id2 = []
    road_id2 = drivable_centers[:,-1]
    ref_drivable_center_inds = np.argwhere(drivable_centers[:,-1] == 1)
    colors2 = []
    for iCnt in road_id2:
        if iCnt < 1:
            colors2 += [[1,0,0]]
        elif iCnt >= 1  and iCnt < 2:
            colors2 += [[0,1,0]]
        elif iCnt >= 2 and iCnt < 3:
            colors2 += [[0,0,1]]
        elif iCnt >= 3 and iCnt < 4:
            colors2 += [[1,1,0]]
        else:
            colors2 += [[1,0,1]]

    
    # drivable_area[:,0] *= -1
    pcd_drivable_area = o3d.geometry.PointCloud()
    # pcd_drivable_area.points = o3d.utility.Vector3dVector(drivable_area)
    drivable_area_bkp = drivable_area
    pcd_drivable_area.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.dtype('f4')))

    pcd_drivable_center = o3d.geometry.PointCloud()
    # pcd_drivable_area.points = o3d.utility.Vector3dVector(drivable_center)
    drivable_center_bkp = drivable_center
    pcd_drivable_center.colors = o3d.utility.Vector3dVector(np.array(colors2, dtype=np.dtype('f4')))
                
    # Compute_PairwiseSceneFlow(start, 5)

    # Translate frame coordinates from left-hand to right-hand
    v_l2r = np.array([1., -1., 1., 1., 1., 1.])
    s_l2r = np.array([1, 1., -1., 1., 1., 1., 1.])
    v2s = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    # enable_animation = True
    if enable_animation:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=960*2, height=640*2, left=5, top=5)
        vis.get_render_option().background_color = np.array([0, 0, 0])
        # vis.get_render_option().background_color = np.array([1, 1, 1])
        vis.get_render_option().show_coordinate_frame = False
        vis.get_render_option().point_size = 3.0
        vis.get_render_option().line_width = 5.0
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, -1.0 * lidar_hight])
    
    # Generate the log data from the DIR:tmp/xxx/xxxxrgb_lable/xxxx 
    build_lidar_fov = False

    # Generate the scene flow data of the only camera field-of-view
    use_rgb_fov = False
    
    # rm_ground = False
    # trans_pattern = r"(?P<v_name>[vehicle]+.\w+.\w+) (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
    # r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
    #     r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
    #         r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"\
    
    rawtrans_files = glob.glob(os.path.join(DATASET_RAWTRANS_PATH2, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
    rawtrans_files.sort()

    rawtrans_files2 = glob.glob(os.path.join(DATASET_RAWTRANS_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
    rawtrans_files2.sort()

    # rawtrans_files2 = glob.glob(os.path.join(DATASET_RAWTRANS_PATH2, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
    # rawtrans_files2.sort()
    # if not(use_rgb_fov):
    #     for r in re.findall(trans_pattern, log):
    #         # print(r[0])
    #         det_car_info.append(int(float(r[0])))
    # print(os.path.abspath(os.curdir))
    sub_dir_cnt = 0
    max_v = -100.0
    # v_info_cnt = 0
    current_v_info = []
    next_v_info = []
    v_pattern = r"vehicle.(?P<nonsense>[\s\S]+)_(?P<v_id>\d+)"
    for sub_dir in os.listdir(DATASET_PATH):
        if 'global_label' in sub_dir or 'img' in sub_dir:
            continue 
        # print(sub_dir)

        SUB_DATASET2 = 'vehicle.mini.cooper_s_2021_1520' #sub_dir #'vehicle.tesla.cybertruck_266'#'vehicle.citroen.c3_271'
        SUB_DATASET2 = GT_SUB_DATASET #'vehicle.mini.cooper_s_2021_254' #sub_dir #'vehicle.tesla.cybertruck_266'#'vehicle.citroen.c3_271'
        # SUB_DATASET = 'vehicle.mini.cooper_s_2021_406' #sub_dir #'vehicle.tesla.cybertruck_266'#'vehicle.citroen.c3_271'
        SUB_DATASET = 'vehicle.mini.cooper_s_2021_402' #sub_dir #'vehicle.tesla.cybertruck_266'#'vehicle.citroen.c3_271'
        SUB_DATASET = ACTIVE_SUB_DATASET #'vehicle.mini.cooper_s_2021_254' #sub_dir #'vehicle.tesla.cybertruck_266'#'vehicle.citroen.c3_271'
        if sub_dir not in SUB_DATASET:
            continue
        v_pattern = r"[vehicle]+.(?P<name>\w+.\w+)_(?P<v_id>\d+)"
        v = re.findall(v_pattern, SUB_DATASET)[0]
        DATASET_PC_PATH = DATASET_PATH2 + SUB_DATASET + '/velodyne/'
        DATASET_LBL_PATH = DATASET_PATH2 + SUB_DATASET + '/label00/'

        v2 = re.findall(v_pattern, SUB_DATASET2)[0]
        DATASET_PC_PATH2 = DATASET_PATH + SUB_DATASET2 + '/velodyne/'
        DATASET_LBL_PATH2 = DATASET_PATH + SUB_DATASET2 + '/label00/'

        # RESULT_PATH = './results/' + SUB_DATASET
        trans_pattern = v[0] + " " + v[1] + r" (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"
        lidar_pattern = r"sensor.lidar.ray_cast (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 " + v[1]
        
        lidar_semantic_pattern = r"sensor.lidar.ray_cast_semantic (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 0 " + v[1]

        trans_pattern2 = v2[0] + " " + v2[1] + r" (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"
        lidar_pattern2 = r"sensor.lidar.ray_cast (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 " + v2[1]
        
        lidar_semantic_pattern2 = r"sensor.lidar.ray_cast_semantic (?P<v_id>\d+) (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) 0 0 0 0 " + v2[1]
        
        glb_label_pattern2 = v[0] + " " + v[1] + r" (?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
    r"(?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
        r"(?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
            r"(?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<extbbx_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                r"(?P<extbbx_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) (?P<extbbx_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+) "\
                    r"(?P<bbx_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"

        # next_ind = ind + 2
        pc_files = glob.glob(os.path.join(DATASET_PC_PATH, '*.bin'))#os.listdir(DATASET_PC_PATH)
        # rawtrans_files = glob.glob(os.path.join(DATASET_RAWTRANS_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
        label_files = glob.glob(os.path.join(DATASET_LBL_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
        # print("Current DIR:" + DATASET_PC_PATH)
        pc_files.sort()
        # rawtrans_files.sort()
        label_files.sort()

        # next_ind = ind + 2
        pc_files2 = glob.glob(os.path.join(DATASET_PC_PATH2, '*.bin'))#os.listdir(DATASET_PC_PATH)
        # rawtrans_files = glob.glob(os.path.join(DATASET_RAWTRANS_PATH, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
        label_files2 = glob.glob(os.path.join(DATASET_LBL_PATH2, '*.txt'))#os.listdir(DATASET_RAWTRANS_PATH)
        # print("Current DIR:" + DATASET_PC_PATH)
        pc_files2.sort()
        # rawtrans_files.sort()
        label_files2.sort()

        total_cmd_num = len(pc_files)
        frame_start,frame_end,frame_hz = 10,-5,1
        spawn_points = []
        cnt = 0

        ego_v_info = re.findall(v_pattern, sub_dir)
        with open(rawtrans_files[0], 'r') as file:
                raw_log = file.read()
        r = re.findall(glb_label_pattern, raw_log)
        for r in re.findall(glb_label_pattern, raw_log):
            if ego_v_info[0][0] in r[0] and ego_v_info[0][1] in r[1]:
                l = float(r[8])
                w = float(r[9])
                h = float(r[10])
                egoV_bbox_extend = np.array([l, w, h]).reshape(3,1) * 2.0
        
        
        DATASET_SF_PATH = RESULT_PATH3 + SF_DIR + "%02d"%(sub_dir_cnt) + '/'
        if not os.path.exists(DATASET_SF_PATH) and save_pts:
            os.makedirs(DATASET_SF_PATH)

        if build_lidar_fov:
            write_labels('record2021_0716_2146/', sub_dir, frame_start,frame_end, frame_hz, index=1)
        else:
            # step = 1
            for ind in range(start, len(pc_files)-step, step):
                next_ind = ind #+ step
                if next_ind >= len(label_files):
                        break
                print("Current DIR:" + DATASET_PC_PATH + ' Frame:' +str(ind))
                # tmp = cmd_traj[ind, ]
                # saveVehicleInfo()
                
                # # For writting the vehicle info ln:1357-1558
                # ivFlag = 0
                # for i in [ind, next_ind]:
                #     # if next_ind >= len(label_files):
                #     #     break
                #     with open(label_files[i], 'r') as file:
                #             log = file.read()
                #     with open(rawtrans_files[i], 'r') as file:
                #         raw_log = file.read()
               
                #     for r in re.findall(glb_label_pattern, raw_log):
                #         # print(r)
                #         if ego_v_info[0][0] in r[0] and ego_v_info[0][1] in r[1]:
                #             if ivFlag % 2 == 0:
                #                 current_v_info.append(r)
                #             else:
                #                 next_v_info.append(r)
                #             ivFlag += 1

                # continue
                # tmp = next_ind
                # ind = next_ind
                # next_ind = ind
                # Compute_PairwiseSceneFlow(ind, next_ind, trans_pattern, lidar_pattern, DATASET_PC_PATH, DATASET_LBL_PATH)
                vis_list = []
                first_v_raw_trans = odom_utils.readRawData(rawtrans_files[ind], trans_pattern) * v_l2r
                first_sensor_raw_trans = odom_utils.readRawData(rawtrans_files[ind], lidar_semantic_pattern) * s_l2r

                src_pc = odom_utils.readPointCloud(pc_files[ind])[:, :3]
                src_pc_bkp = src_pc + first_v_raw_trans[:3]
                if rm_ground:
                    is_ground = np.logical_not(src_pc[:, 2] <= -1.0 * (lidar_hight-0.1))
                    src_pc = src_pc[is_ground, :] 
                # np.save('./data/np_src', src_pc)
                # For the carla origin coordatinate 
                # first_v_raw_trans[2] += 2.5
                # src_pc += first_v_raw_trans[:3] @ np.array([[0,-1,0],[1,0,0],[0,0,1]])
                
                # ## For the carla original coordinate
                # drivable_area = drivable_area_bkp @ np.array([[0,-1,0],[1,0,0],[0,0,1]])

                ## For the first car frame as the original coordinate
                drivable_area = (drivable_area_bkp - first_v_raw_trans[:3])\
                     @ odom_utils.rotation_from_euler_zyx(-first_v_raw_trans[5], -first_v_raw_trans[4],first_v_raw_trans[3])#
                pcd_drivable_area.points = o3d.utility.Vector3dVector(drivable_area)
                # np.save('./data/np_drivable_area', np.hstack((drivable_area, (drivable_areas[:,-1]).reshape((-1,1)))))
                
                ## For the carla original coordinate
                # drivable_center = drivable_center_bkp @ np.array([[0,-1,0],[1,0,0],[0,0,1]])

                # For the first car frame as the original coordinate
                drivable_center = (drivable_center_bkp - first_v_raw_trans[:3])\
                     @ odom_utils.rotation_from_euler_zyx(-first_v_raw_trans[5], -first_v_raw_trans[4],first_v_raw_trans[3])#
                pcd_drivable_center.points = o3d.utility.Vector3dVector(drivable_center)
                # pcd_drivable_area.paint_uniform_color([1.0,0,0.0])
                ref_drivable_center = drivable_centers[ref_drivable_center_inds[:,0],:3]

                # np.save('./data/np_drivable_center', np.hstack((drivable_center, (drivable_centers[:,-1]).reshape((-1,1)))))

                arr = src_pc
                
                src_R_vec = first_sensor_raw_trans[4:]
                src_R = o3d.geometry.get_rotation_matrix_from_xyz(src_R_vec)
                src_R_inv = np.linalg.inv(src_R)
                tmp_pcd = o3d.geometry.PointCloud()
                tmp_pcd.points = o3d.utility.Vector3dVector(src_pc)
                #For the carla origin coordatinate 
                # pcd.rotate(src_R_inv, first_v_raw_trans[:3])

                # For the first frame as the origin coordinate
                # pcd.rotate(src_R_inv, np.zeros(3))
                # pcd.paint_uniform_color([0,1,0])

                # ego_pcd = o3d.geometry.PointCloud()
                # ego_pcd.points = o3d.utility.Vector3dVector(first_v_raw_trans[:3].reshape(-1,3))
                # ego_pcd.paint_uniform_color([0,1,1])
               
                # drivable_area =  drivable_area - (np.array([[spawn_points[ind][0], -spawn_points[ind][1], 0]]))
                
                vis_list = [mesh_frame, pcd_drivable_area, pcd_drivable_center]
                # vis_list = [mesh_frame, pcd]
                
                tgt_v_raw_trans = odom_utils.readRawData(rawtrans_files2[next_ind], trans_pattern2) * v_l2r

                print(first_v_raw_trans[:3])
                print(tgt_v_raw_trans[:3])
                # tgt_v_raw_trans2 = odom_utils.readRawData(rawtrans_files2[next_ind], trans_pattern2) * v_l2r
                # tgt_v_raw_trans0 = odom_utils.readRawData(rawtrans_files[next_ind], trans_pattern2) * v_l2r
                # print(first_v_raw_trans)
                # print(tgt_v_raw_trans0)
                # print(tgt_v_raw_trans)
                # print(tgt_v_raw_trans2)
                tgt_sensor_raw_trans = odom_utils.readRawData(rawtrans_files2[next_ind], lidar_semantic_pattern2) * s_l2r
                tgt_pc = odom_utils.readPointCloud(pc_files2[next_ind])[:, :3]

                tgt_pc_bkp = tgt_pc + tgt_v_raw_trans[:3]
                if rm_ground:
                    is_ground = np.logical_not(tgt_pc[:, 2] <= -1.0 * (lidar_hight - 0.05))
                    tgt_pc = tgt_pc[is_ground, :] 

                #For the carla origin coordatinate 
                # tgt_v_raw_trans[2] += 2.5
                # tgt_pc += tgt_v_raw_trans[:3] @ np.array([[0,-1,0],[1,0,0],[0,0,1]])

                tgt_R_vec = tgt_sensor_raw_trans[4:]
                tgt_R = o3d.geometry.get_rotation_matrix_from_xyz(tgt_R_vec)
                tgt_R_inv = np.linalg.inv(tgt_R)
                tmp_pcd_cur = o3d.geometry.PointCloud()
                tmp_pcd_cur.points = o3d.utility.Vector3dVector(tgt_pc)

                #For the carla origin coordatinate 
                # pcd_cur.rotate(tgt_R_inv, tgt_v_raw_trans[:3] )

                #For the first frame as the origin coordinate
                # pcd_cur.rotate(tgt_R_inv, np.zeros(3))
                # pcd_cur.translate(tgt_v_raw_trans[:3] - first_v_raw_trans[:3])
                # tmp_pcd_cur.paint_uniform_color([1,0,0])

                first_v_raw_rot = first_v_raw_trans[3:]
                # first_v_raw_rot[-1] *= -1.0
                src_ego_bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(first_v_raw_rot)
                # src_ego_bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(first_v_raw_trans[3:])
                ego_car_center = np.zeros([3,1])
                ego_car_center[2] = -1.0 * lidar_hight + 0.5 * egoV_bbox_extend[2]
                ego_car_bbx_c = o3d.geometry.OrientedBoundingBox(ego_car_center, src_ego_bbox_R, egoV_bbox_extend)
                # vis_list.append(ego_car_bbx_c)
                inds = ego_car_bbx_c.get_point_indices_within_bounding_box(tmp_pcd.points)
                valid_inds = np.ones([src_pc.shape[0]])
                if len(inds) > 0:
                    valid_inds[inds] = 0
                    valid_inds = valid_inds.astype(bool)
                    src_pc = src_pc[valid_inds,:]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(src_pc)
                #For the carla origin coordatinate 
                # pcd.rotate(src_R_inv, first_v_raw_trans[:3])

                # For the first frame as the origin coordinate
                # pcd.rotate(src_R_inv, np.zeros(3))
                pcd.paint_uniform_color([0,1,0])

                tgt_v_raw_rot = tgt_v_raw_trans[3:]
                # first_v_raw_rot[-1] *= -1.0
                tgt_ego_bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(tgt_v_raw_rot)
                tgt_ego_bbx_center = (np.zeros([3]) @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T
                tgt_ego_bbx_center[2] = -1.0 * lidar_hight + 0.5 * egoV_bbox_extend[2]
                tgt_ego_car_bbx_c = o3d.geometry.OrientedBoundingBox(tgt_ego_bbx_center, tgt_ego_bbox_R, egoV_bbox_extend)
                # vis_list.append(ego_car_bbx_c)
                tgt_inds = tgt_ego_car_bbx_c.get_point_indices_within_bounding_box(tmp_pcd_cur.points)
                tgt_valid_inds = np.ones([tgt_pc.shape[0]])
                if len(tgt_inds) > 0:
                    tgt_valid_inds[tgt_inds] = 0
                    tgt_valid_inds = tgt_valid_inds.astype(bool)
                    tgt_pc = tgt_pc[tgt_valid_inds,:]
                pcd_cur = o3d.geometry.PointCloud()
                pcd_cur.points = o3d.utility.Vector3dVector(tgt_pc)

                #For the carla origin coordatinate 
                # pcd_cur.rotate(tgt_R_inv, tgt_v_raw_trans[:3] )

                #For the first frame as the origin coordinate
                # pcd_cur.rotate(tgt_R_inv, np.zeros(3))
                # pcd_cur.translate(tgt_v_raw_trans[:3] - first_v_raw_trans[:3])
                pcd_cur.paint_uniform_color([1,0,0])
                
                arr_ = (src_pc @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T

                alternation_flag = 2
                oth_cars_info = []
                fgrnd_inds = np.zeros(src_pc.shape[0])
                fg_inds = []
                fgrnd_inds_t = np.zeros(tgt_pc.shape[0])
                fg_inds_t = []
                for  iflag, i in enumerate([ind, next_ind]):
                    # if next_ind >= len(label_files):
                    #     break
                    if iflag % alternation_flag == 0:
                        with open(label_files[i], 'r') as file:
                                log = file.read()
                    else:
                        with open(label_files2[i], 'r') as file:
                                log = file.read()
                    

                    det_car_info = []  
                    if use_rgb_fov:
                        for r in re.findall(rgb_label_pattern, log):
                            det_car_info.append(int(r[-1]))
                    # else:
                    #     for r in re.findall(lidar_label_pattern, log):
                    #         # print(r[0])
                    #         det_car_info.append(int(float(r[0])))
                    if iflag % alternation_flag == 0:
                        with open(rawtrans_files[i], 'r') as file:
                            raw_log = file.read()
                    else:
                        with open(rawtrans_files2[i], 'r') as file:
                            raw_log = file.read()
                    oth_car_info = []
                    for r in re.findall(glb_label_pattern, raw_log):
                        # print(r[0])
                        if use_rgb_fov:
                            if int(r[1]) in det_car_info:
                                oth_car_info.append([int(r[1])] + list(map(float, r[2:5])) + [np.radians(float(r[5])),\
                                    np.radians(float(r[6])), np.radians(float(r[7]))] + list(map(float, r[-4:])))
                        else:
                            if r[1] != v[1]:
                                oth_car_info.append([int(r[1])] + list(map(float, r[2:5])) + [np.radians(float(r[5])),\
                                    np.radians(float(r[6])), np.radians(float(r[7]))] + list(map(float, r[-4:])))
                    if len(oth_car_info) > 0:
                        oth_cars_info.append(oth_car_info)
                    # print(oth_car_info)

                # print(first_v_raw_trans)
                # #######TEST FOR INVERSE TRANSFORM TO CARLR COORDINATE
                # inv_ego_pos = first_v_raw_trans[:3]
                # print(inv_ego_pos)
                    
                show_flag = 1
                lcl_R = [src_R, tgt_R]
                lcl_R_inv = [src_R_inv, tgt_R_inv]
                lcl_rot_v = [first_v_raw_trans[3:], tgt_v_raw_trans[3:]]
                lcl_trans_v = [first_v_raw_trans[:3], tgt_v_raw_trans[:3]]
                colors = [[0,1,0], [1,0,0]]
                objs_center = []
                objs_flow = []
                objs_bbx = []
                objs_bbox_R = []
                # fg_inds = []
                # if show_flag:
                for iv in range(len(oth_cars_info)):
                    if len(oth_cars_info) <= 1:
                        flow = arr_ - arr
                        # continue
                    else:
                        oth_v_loc = np.array(oth_cars_info[iv])[:,1:4] * np.array([1.0, -1.0, 1.0])
                        oth_v_loc[:,-1] = oth_v_loc[:,-1] + np.array(oth_cars_info[iv])[:,-1] - lidar_hight
                        oth_v_rot = np.array(oth_cars_info[iv])[:,4:7]
                        oth_v_bbx_ext = np.array(oth_cars_info[iv])[:,-4:]

                        # objs_center = []
                        obj_flow = []
                        obj_bbx = []
                        obj_bbox_R = []

                        for ibbx in range(oth_v_loc.shape[0]):
                            # bbox_v_rot = -1.0 * oth_v_rot[ibbx, :] + np.array([0.0, 0.0, 2.0 * np.pi])
                            bbox_v_rot = lcl_rot_v[iv] - oth_v_rot[ibbx, :]
                            bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(bbox_v_rot)
                            #For the local (observation) coordinates
                            obj_bbox_R.append(bbox_v_rot)
                       
                            bbox_extend = oth_v_bbx_ext[ibbx, :-1] * 2.0
                            #For the local (observation) coordinates
                            bbox_center = lcl_R[iv] @ (oth_v_loc[ibbx,:] - lcl_trans_v[iv])
                            #For the original coordinates
                            # bbox_center = lcl_R[iv] @ oth_v_loc[ibbx,:]
                            # print(bbox_center)
                            obj_flow.append(np.hstack((np.array(oth_cars_info[iv][ibbx][0]), np.array(bbox_center))))
                            #For the local (observation) coordinates
                            bbox = o3d.geometry.OrientedBoundingBox(bbox_center, bbox_R, bbox_extend)
                            #For the original coordinates
                            # bbox = o3d.geometry.OrientedBoundingBox(bbox_center, bbox_R, bbox_extend)
                            obj_bbx.append(bbox)
                            bbox.color = colors[iv]
                            # bbox.LineSet = 
                            vis_list.append(bbox)
                        objs_flow.append(obj_flow)
                        objs_bbx.append(obj_bbx)
                        objs_bbox_R.append(obj_bbox_R)
                    
                    # fg_inds = np.zeros(src_pc.shape[0])
                    if len(objs_flow) <= 1:
                        flow = arr_ - arr
                    else:
                        src_objs_flow = np.array(objs_flow[0])
                        src_objs_bbox = dict(zip(src_objs_flow[:,0], objs_bbx[0]))
                        src_objs_flow = dict(zip(src_objs_flow[:,0], src_objs_flow[:,1:]))
                        src_bbox_R = dict(zip(src_objs_flow.keys(),objs_bbox_R[0]))                    
                    
                        tgt_objs_flow = np.array(objs_flow[1])
                        tgt_objs_bbox = dict(zip(tgt_objs_flow[:,0], objs_bbx[1]))
                        tgt_objs_flow = dict(zip(tgt_objs_flow[:,0], tgt_objs_flow[:,1:]))
                        tgt_bbox_R = dict(zip(tgt_objs_flow.keys(),objs_bbox_R[1]))

                        objs_flow = []
                        # objs_c = np.array(list(src_objs_flow.values()))
                        # subarr_ = (objs_c @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T
                        # subflow = subarr_ - objs_c
                        # subarr_ = (objs_c @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T
                        # subflow = subarr_ - objs_c
                        # src_rigid_flow = dict(zip(src_objs_flow.keys(), subflow))
                        objs_bbox_R = dict(zip(src_objs_flow.keys(),objs_bbox_R))
                        for k,v_info in src_objs_flow.items():   
                            tgt_k = k #car_ids.get(k)
                            # if tgt_k is not None:
                            #     tgt_k = float(tgt_k)                                
                            if tgt_k in tgt_objs_flow:
                                bbox = src_objs_bbox.get(k) 
                                inds = bbox.get_point_indices_within_bounding_box(pcd.points)
                                bbox_t = tgt_objs_bbox.get(k) 
                                inds_t = bbox_t.get_point_indices_within_bounding_box(pcd_cur.points)
                                # flow[inds,:] = obj_flow
                                delat_rot = tgt_bbox_R.get(tgt_k) - src_bbox_R.get(k)
                                bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(-delat_rot)
                                obj_flow = tgt_objs_flow.get(tgt_k) - v_info @ bbox_R
                                arr_[inds,:] = arr[inds,:] @ bbox_R + obj_flow 
                                if len(fg_inds) == 0:
                                    fg_inds = inds
                                    fg_inds_t = inds_t
                                else:
                                    fg_inds += inds
                                    fg_inds_t += inds_t
                                fgrnd_inds[inds] = 1
                                fgrnd_inds_t[inds_t] = 1
                            else:
                                obj_flow = np.zeros(3)
                            # print(obj_flow)
                            
                            objs_flow.append(obj_flow)
                        # delta_flow = np.array(objs_flow) - subflow
                        objs_flow = dict(zip(src_objs_flow.keys(), objs_flow))
                        flow = 1.0 * (arr_ - arr)
                # vis_list.append(pcd)
                pcd_cur2 = o3d.geometry.PointCloud()
                pcd_cur2.points = o3d.utility.Vector3dVector(arr_)
                pcd_cur2.paint_uniform_color([0,0,1.0])

                # concat_fg_inds = []
                # for item in enumerate(fg_inds):
                #     if len(item) > 0:
                #         concat_fg_inds += item[1]

                # vis_list.append(pcd)
                # vis_list.append(pcd_cur)
                # vis_list.append(pcd_cur2)
                # o3d.visualization.draw_geometries(vis_list)
                # save_view_point(vis_list, 'camera_view.json')
                if not(enable_animation):
                    vis = o3d.visualization.Visualizer()
                    vis.create_window(width=960*2, height=640*2, left=5, top=5)
                    vis.get_render_option().background_color = np.array([0,0,0])
                    # vis.get_render_option().background_color = np.array([1, 1, 1])
                    vis.get_render_option().show_coordinate_frame = False
                    vis.get_render_option().point_size = 3.0
                    vis.get_render_option().line_width = 5.0
                # custom_draw_geometry(vis, vis_list, map_file=None, recording=False,param_file='camera_view.json', save_fov=True)
                use_flow = True
                flow = flow * 1.0
                # bg_flag = np.ones([src_pc.shape[0],1])
                # bg_flag[concat_fg_inds] = -1
                # bg_flag2 = np.argwhere(bg_flag==1)
                # ego_flow =  np.zeros(src_pc.shape)
                # ego_flow[bg_flag2[:,0],:] = flow[bg_flag2[:,0],:]
                ego_flow = (src_pc @ src_R_inv.T + first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T - src_pc
                # ego_flow = (first_v_raw_trans[:3] - tgt_v_raw_trans[:3]) @ tgt_R.T * np.ones(src_pc.shape)
                # delta_R = o3d.geometry.get_rotation_matrix_from_xyz(first_v_raw_trans[3:] - tgt_v_raw_trans[3:]) 
                # ego_flow = (tgt_v_raw_trans[:3] - first_v_raw_trans[:3] ) @ delta_R.T @ v2s * np.ones(src_pc.shape)
                # ego_flow = (first_sensor_raw_trans[:3] - tgt_sensor_raw_trans[:3]) @ v2s * np.ones(src_pc.shape)
                # ego_flow = (-tgt_v_raw_trans[:3] + first_v_raw_trans[:3]) @ v2s * np.ones(src_pc.shape)
                ego_flow_bkp = ego_flow + tgt_v_raw_trans[:3] - first_v_raw_trans[:3]
                flow_bkp = flow + tgt_v_raw_trans[:3] - first_v_raw_trans[:3] 
                
                name_fmt = "%06d"%(ind)
                if abs(ego_flow[0,0]) > max_v:
                    max_v = abs(ego_flow[0,0])
                if save_pts:
                    # np.savez(DATASET_SF_PATH + name_fmt, pos1=src_pc, pos2=tgt_pc, ego_flow=ego_flow, gt=flow) 
                    if only_fg:
                        np.savez(DATASET_SF_PATH + name_fmt, pos1=src_pc[fg_inds,:], pos2=tgt_pc[fg_inds_t,:], ego_flow=ego_flow[fg_inds,:], gt=flow[fg_inds,:], s_fg_mask = fgrnd_inds, t_fg_mask = fgrnd_inds_t) 
                    else:
                        np.savez(DATASET_SF_PATH + name_fmt, pos1=src_pc, pos2=tgt_pc, ego_flow=ego_flow, gt=flow, s_fg_mask = fgrnd_inds, t_fg_mask = fgrnd_inds_t) 

                if show_fgrnds:
                    tgt_pc = tgt_pc[fg_inds_t,:]
                    src_pc = src_pc[fg_inds,:]
                    arr_ = arr_[fg_inds,:]
                    ego_flow = ego_flow[fg_inds,:]
                    flow = flow[fg_inds,:]
                # bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(first_v_raw_trans[3:])
                # ego_car_center = np.zeros([3,1])
                # ego_car_center[2] = -1.0 * lidar_hight + 0.5 * egoV_bbox_extend[2] #-0.4 * egoV_bbox_extend[2]
                # ego_car_bbx_c = o3d.geometry.OrientedBoundingBox(ego_car_center, bbox_R, egoV_bbox_extend)
                # # vis_list.append(ego_car_bbx_c)
                # inds = ego_car_bbx_c.get_point_indices_within_bounding_box(pcd.points)
                # valid_inds = np.ones([src_pc.shape[0]])
                # if len(inds) > 0:
                #     valid_inds[inds] = 0
                #     valid_inds = valid_inds.astype(bool)
                #     src_pc = src_pc[valid_inds,:]
                #     arr_ = arr_[valid_inds,:]
                #     ego_flow = ego_flow[valid_inds,:]
                #     flow = flow[valid_inds,:]
                #     fg_inds = (fg_inds[valid_inds, :]).astype(bool)

                # bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(tgt_v_raw_trans[3:])
                # ego_car_bbx_n = o3d.geometry.OrientedBoundingBox(ego_car_center, bbox_R, egoV_bbox_extend)
                # # vis_list.append(ego_car_bbx_n)
                # inds = ego_car_bbx_n.get_point_indices_within_bounding_box(pcd_cur.points)
                # valid_inds = np.ones([tgt_pc.shape[0]])
                # if len(inds) > 0:
                #     valid_inds[inds] = 0
                #     valid_inds = valid_inds.astype(bool)
                #     tgt_pc = tgt_pc[valid_inds,:]
                
                if use_flow:
                    # sf = sf[sample_idx1, :]
                    np_list = [src_pc, tgt_pc, arr_, flow]
                    # np_list = [src_pc[], tgt_pc, arr_, flow]
                    # np_list = [src_pc, tgt_pc, arr_, ego_flow]   
                    # np_list = [src_pc_bkp, tgt_pc_bkp, arr_ + tgt_v_raw_trans[:3], ego_flow_bkp]                 
                    # np_list = [src_pc_bkp, tgt_pc_bkp, arr_ + tgt_v_raw_trans[:3], flow_bkp]
                else:
                    np_list = [src_pc, tgt_pc, arr_,]
                    # np_list = [src_pc_bkp, tgt_pc_bkp, arr_, flow]
                
                # import os
                # print(os.path.abspath(os.curdir))
                src_pc2 = src_pc - np.mean(src_pc, axis=0)
                normp = np.sqrt(np.sum(src_pc2 * src_pc2, axis=1))
                diameter = np.max(normp)
                print("Define parameters used for hidden_point_removal")
                
                # camera = [-1.0, 0.0, 5]
                # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3).translate(camera)
                # print("Get all points that are visible from given view point")
                # radius = diameter * 10
                # _, pt_map = pcd3.hidden_point_removal(camera, radius)
                # pc = np.vstack([pc1, pc2])
                # camera = [-45.0, -20.0, drivable_area[0,2]]
                # camera = [45.0, -60.0, drivable_area[0,2]]
                # camera = [-35.0, -20.0, drivable_area[0,2]]
                # camera = [75.0, 1.5, drivable_area[0,2]]
                # arr2_ = arr_ + flow
                # pt_map = HPR(tgt, np.array([[0, 0, -2.5]]), 3)
                # np_arr2 = arr2_[pt_map,:3]
                # # print("Visualize result")
                # # pcd3 = pcd3.select_by_index(pt_map)
                # bool_in = in_convex_polyhedron(arr2_[pt_map,:3],np.array([[-5.0, 4.0, -2.5]]), vis) 
                # pcd3 = drivable_area[pt_map,:]
                # bool_in = in_convex_polyhedron(drivable_area[iou_id[:,0],:3], np.array([[0, 0, -2.5]]), vis) 
                # camera_pos = np.hstack((first_v_raw_trans[:2], -2.75)).reshape(1,3)
                # bool_in = in_convex_polyhedron(drivable_area[iou_id[:,0],:3],camera_pos, vis) 
                global bbx
                # bbx = np.array([[0,0,0], [4,0,0], [0,-4,0], [4,-1,0], [0,0,-2.5], [4,0,-2.5], [0,-4,-2.5], [4,-4,-2.5]]) * 2.0
                bbx = np.array([[0,0,0], [4,0,0], [0,-1,0], [4,-1,0], [0,0,-2.5], [4,0,-2.5], [0,-1,-2.5], [4,-1,-2.5]]) * 0.8
                # bbx = np.array([[0,0,0], [1,0,0], [0,-1,0], [1,-1,0], [0,0,-2.5], [1,0,-2.5], [0,-1,-2.5], [1,-1,-2.5]]) 
                # bool_in, _ = validDrivableArea(bbx, tgt_pc)
                bool_in, _ = validateDrivableArea2(bbx, tgt_pc)

                sphere = o3d.geometry.TriangleMesh.create_sphere(2).translate(np.array([0, 0, -2.5]))
                if bool_in:
                    sphere.paint_uniform_color([0.2,0.2,1.0])
                else:
                    sphere.paint_uniform_color([1,0,0])
                vis_list += [sphere]

                # np_list = vis_list
                use_flow = False
                Draw_SceneFlow(vis, np_list+vis_list, use_fast_vis = not(use_flow), use_pred_t=True, use_custom_color=True, use_flow=use_flow, param_file='camera_view.json', img_cnt = img_cnt)
                img_cnt = img_cnt + 1
        sub_dir_cnt += 1
    # np.savez('./data/data_0308_1058_Vinfo', cvInfo = current_v_info, nvInfo = next_v_info)

if __name__ == "__main__":
    mode = 'random'
    # mode = 'random_paper'
    # mode = 'active_paper'
    # mode = 'active'
    prefixed_V_name = "vehicle.dodge.charger_police_2020_261"
    # prefixed_V_name = "vehicle.mini.cooper_s_2021_263"
    # prefixed_V_name = "vehicle.dodge.charger_police_2020_265"
    # prefixed_V_name = "vehicle.volkswagen.t2_325"
    if mode == 'random':
        Generate_Random_SceneFlow(start=1, step=1, lidar_hight = 3.4, prefixed_V_name=prefixed_V_name, show_active_egoV=True, save_pts=False, enable_animation = True, egoV_flag=False, rm_ground = True, show_fgrnds=False, egoV_file_path = './data/data_0316_1838_Vinfo')
    elif mode == 'random_paper':
        Generate_Random_SceneFlow_Paper(start=1, step=1, lidar_hight = 3.4, prefixed_V_name=prefixed_V_name,  show_active_egoV=True, save_pts=False, enable_animation = True, egoV_flag=False, rm_ground = True, show_fgrnds=False, egoV_file_path = './data/data_0316_1838_Vinfo',GT_SUB_DATASET=prefixed_V_name)
    elif mode == 'active_paper':
        Generate_Active_SceneFlow_Paper( start=1, step=1, lidar_hight = 3.4, prefixed_V_name=prefixed_V_name, show_active_egoV=True, save_pts=False, enable_animation = True, egoV_flag=False, rm_ground = True, show_fgrnds=False, egoV_file_path = './data/data_0314_1923_active_Vinfo',GT_SUB_DATASET=prefixed_V_name, ACTIVE_SUB_DATASET=prefixed_V_name)
    else:
        random_car_id = [271, 278, 255, 251, 264, 261, 253, 252, 262, 263, 270, 254, 247, 265, 266, 267] #0308_1719
        active_car_id = [271, 278, 255, 251, 264, 261, 253, 252, 262, 263, 270, 254, 247, 265, 266, 267] #0308_1719
        car_ids = dict(zip(random_car_id, active_car_id))
        Generate_Active_SceneFlow(car_ids, start=1, step=1, lidar_hight = 3.4, prefixed_V_name=prefixed_V_name, show_active_egoV=False, save_pts=False, enable_animation = True, egoV_flag=False, rm_ground = True, show_fgrnds=False, egoV_file_path = './data/data_0314_1923_active_Vinfo',GT_SUB_DATASET=prefixed_V_name, ACTIVE_SUB_DATASET=prefixed_V_name)