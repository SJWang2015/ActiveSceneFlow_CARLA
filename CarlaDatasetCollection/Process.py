import sys
import os
import copy
import math
import shutil
import glob


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from utils.calibration import Calibration

VIEW_WIDTH = 1242
VIEW_HEIGHT = 375
VIEW_FOV = 90
camera_intrinsic_matrix = np.identity(3)
camera_intrinsic_matrix[0, 2] = VIEW_WIDTH / 2.0
camera_intrinsic_matrix[1, 2] = VIEW_HEIGHT / 2.0
camera_intrinsic_matrix[0, 0] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
camera_intrinsic_matrix[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))

def get_ego_lidar_data_path(ego_vehicle_label, _frame_id):
    ego_vehicle_name = ego_vehicle_label[0] + '_' + ego_vehicle_label[1]
    ego_raw_data_path = raw_data_path + '/' + ego_vehicle_name
    raw_data_file = os.listdir(ego_raw_data_path)
    raw_data_file.sort()
    lidar_raw_data_file = ego_raw_data_path + '/' + raw_data_file[-1] + '/' + _frame_id + 'ply'
    return lidar_raw_data_file

def get_ego_camera_2Dlabel_path(ego_vehicle_label, camera_id, _frame_id):
    ego_vehicle_name = ego_vehicle_label[0] + '_' + ego_vehicle_label[1]
    ego_raw_data_path = raw_data_path + '/' + ego_vehicle_name
    raw_data_file = os.listdir(ego_raw_data_path)
    raw_data_file.sort()
    for _file in raw_data_file:
        if str(camera_id) in _file and 'label' in _file:
            camera_2Dlabel_file = ego_raw_data_path + '/' + _file + '/' + _frame_id + 'txt'
            return camera_2Dlabel_file

def get_raw_lidar_data(lidar_raw_data_file, sensor_rotation, use_seg=False):
    if use_seg:
        pcd = o3d.io.read_point_cloud(lidar_raw_data_file, format='xyzrgb')  #X,Y,Z, cosang, ObjID, ObjTag
    else:
        pcd = o3d.io.read_point_cloud(lidar_raw_data_file)
    R = o3d.geometry.get_rotation_matrix_from_xyz(np.array([np.pi,0,np.pi/2]))
    tmp_R = o3d.geometry.get_rotation_matrix_from_xyz(sensor_rotation)
    # pcd.rotate(R, center=(0,0,0))
    # pcd.rotate(tmp_R,center=(0,0,0))
    pointcloud = np.array(pcd.points, dtype=np.dtype('f4'))
    pointcloud = np.dot(pointcloud,np.array([1,0,0,0,-1,0,0,0,1]).reshape(3,3))
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    if not use_seg:
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
    return pcd

def process_pcd(pcd,calib,use_seg=False):
    img_shape = [1242,375]
    pointcloud = np.array(pcd.points, dtype=np.dtype('f4'))
    pts_rect = calib.lidar_to_rect(pointcloud[:, 0:3])
    fov_flag = get_fov_flag(pts_rect, img_shape, calib)
    tmp_pcd = o3d.geometry.PointCloud()
    tmp_pcd.points = o3d.utility.Vector3dVector(pointcloud[fov_flag])
    if not use_seg:
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
    return tmp_pcd

def get_fov_flag(pts_rect, img_shape, calib):
    '''
    Valid point should be in the image (and in the PC_AREA_SCOPE)
    :param pts_rect:
    :param img_shape:
    :return:
    '''
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[0])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[1])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    pts_valid_flag = np.logical_and(pts_valid_flag, pts_rect_depth <= 110)
    return pts_valid_flag

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

def get_vehicle_transform(vehilce_label):
    location = np.array(list(map(float,vehilce_label[2:5])))
    rotation = np.array([np.radians(float(vehilce_label[5])),np.radians(float(vehilce_label[6])),np.radians(float(vehilce_label[7]))])
    return location,rotation

def get_vehicle_bbox(other_vehilcle_label, sensor_center, sensor_rotation, flag=False):
    bbox_center = np.array(list(map(float,other_vehilcle_label[2:5]))) + np.array([0,0,float(other_vehilcle_label[-1])])
    bbox_center[1] *= -1
    sensor_world_matrix = get_matrix(sensor_center,-sensor_rotation+[0,0,np.pi*4/2])
    # sensor_world_matrix = get_matrix(sensor_center,-sensor_rotation+[0,0,np.pi*3/2])
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    bbox_center = np.append(bbox_center,1).reshape(4,1)
    bbox_center = np.dot(world_sensor_matrix, bbox_center).tolist()[:3]
    tmp = sensor_rotation-[0,0,np.radians(float(other_vehilcle_label[7]))]
    # tmp = [0,0,sensor_rotation[2]-np.radians(float(other_vehilcle_label[7]))]
    # print(sensor_rotation,[np.radians(float(test)) for test in other_vehilcle_label[5:8]])
    # tmp_tmp = [-sensor_rotation[1]+np.radians(float(other
    # _vehilcle_label[6])),-sensor_rotation[0]+np.radians(float(other_vehilcle_label[5]))]+[0]
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

def get_calib_file(raw_data_path, frame_id, ego_vehicle_label,index, calib_info):
    ego_vehicle_name = ego_vehicle_label[0]+'_'+ego_vehicle_label[1]
    dataset_path = 'dataset/' + raw_data_path[4:] + '/' + ego_vehicle_name
    if not os.path.exists(dataset_path+'/calib0'+str(index)):
        os.makedirs(dataset_path+'/calib0'+str(index))
    np.savetxt(dataset_path + '/calib0'+str(index)+'/' + frame_id, np.array(calib_info), fmt='%s', delimiter=' ')
    calib = Calibration(dataset_path + '/calib0'+str(index)+'/' + frame_id)
    return calib

def write_labels(raw_data_path, frame_id, ego_vehicle_label, tmp_bboxes, pointcloud, index, camera_id, calib_info, use_seg=False):
    ego_vehicle_name = ego_vehicle_label[0]+'_'+ego_vehicle_label[1]
    dataset_path = 'dataset/' + raw_data_path[4:] + '/' + ego_vehicle_name
    raw_data_path += '/' + ego_vehicle_name
    sensor_raw_path = os.listdir(raw_data_path)
    if not os.path.exists(dataset_path+'/label0'+str(index)):
        os.makedirs(dataset_path+'/label0'+str(index))
    if not os.path.exists(dataset_path+'/image0'+str(index)):
        os.makedirs(dataset_path+'/image0'+str(index))
    # if not os.path.exists(dataset_path+'/depth0'+str(index)):
    #     os.makedirs(dataset_path+'/depth0'+str(index))
    # if not os.path.exists(dataset_path+'/logdepth0'+str(index)):
    #     os.makedirs(dataset_path+'/logdepth0'+str(index))
    # if not os.path.exists(dataset_path+'/flow0'+str(index)):
    #     os.makedirs(dataset_path+'/flow0'+str(index))
    if not os.path.exists(dataset_path+'/calib0'+str(index)):
        os.makedirs(dataset_path+'/calib0'+str(index))
    if not os.path.exists(dataset_path+'/velodyne'):
        os.makedirs(dataset_path+'/velodyne')
    if use_seg:
        if not os.path.exists(dataset_path+'/velodyne_seg'):
            os.makedirs(dataset_path+'/velodyne_seg')
    
    for _tmp in sensor_raw_path:
        if str(camera_id) in _tmp and 'label' not in _tmp: 
            image_path = raw_data_path + '/' + _tmp + '/' + frame_id[:-3] + 'png' 
            print(image_path)
            if 'rgb' in _tmp:
                shutil.copy(image_path, dataset_path+'/image0'+str(index))
            elif 'camera.depth' in _tmp:
                shutil.copy(image_path, dataset_path+'/depth0'+str(index))
            elif 'logdepth' in _tmp:
                shutil.copy(image_path, dataset_path+'/logdepth0'+str(index))
            else:
                if 'camera.optical_flow' in _tmp:
                    image_path = raw_data_path + '/' + _tmp + '/' + frame_id[:-3] + 'npy' 
                    shutil.copy(image_path, dataset_path+'/flow0'+str(index))
            break
    
    # np.savetxt(dataset_path + '/calib0'+str(index)+'/' + frame_id, np.array(calib_info), fmt='%s', delimiter=' ')
    calib = Calibration(dataset_path + '/calib0'+str(index)+'/' + frame_id)
    for idx,tmp_bbox in enumerate(tmp_bboxes):
        coordinate = np.array(list(map(float,tmp_bbox[11:14]))).reshape(1,3)
        coordinate_camera = calib.lidar_to_rect(coordinate).flatten()
        tmp_bboxes[idx][11:14] = coordinate_camera
    np.savetxt(dataset_path + '/label0'+str(index)+'/' + frame_id, np.array(tmp_bboxes), fmt='%s', delimiter=' ')
    pointcloud = np.array(pointcloud.points, dtype=np.dtype('f4'))
    # pointcloud[:,2] = pointcloud[:,2] * -1
    # points_r = np.zeros((pointcloud.shape[0],1),dtype=np.dtype('f4'))
    points_R = np.exp(-0.05*np.sqrt(np.sum(pointcloud**2,axis=1))).reshape(-1,1)
    pointcloud = np.concatenate((pointcloud,points_R),axis=1)
    pointcloud.tofile(dataset_path + '/velodyne/' + frame_id[:-3] +'bin')
    if use_seg:
        seg_pointcloud = np.array(pointcloud.colors, dtype=np.dtype(np.int32))
        pointcloud.tofile(dataset_path + '/velodyne_seg/' + frame_id[:-3] +'bin')
    return image_path

def get_matrix(location, rotation):
    T = np.matrix(np.eye(4))
    T[:3,:3] = o3d.geometry.get_rotation_matrix_from_xyz(rotation)
    T[0:3,3] = location.reshape(3,1)
    return T
    
def get_matrix_from_origin_to_target(origin_location,origin_orientation,target_location,target_orientation):
    origin_matrix = get_matrix(origin_location,origin_orientation)
    target_matrix = get_matrix(target_location,target_orientation)
    target_matrix = np.linalg.inv(target_matrix)
    T = np.dot(target_matrix,origin_matrix)
    return T
    # delta_rotation = origin_orientation - target_orientation
    # delta_translate = origin_location - target_location
    # T = np.eye(4)
    # T[:3,:3] = o3d.geometry.get_rotation_matrix_from_xyz(delta_rotation)
    # T[0:3,3] = delta_translate
    # print(T)
    # return T

def transform_vehicle_to_lidar(vehicle_location, matrix):
    old_location = np.append(vehicle_location,1).reshape((4,1))
    new_location = np.dot(matrix, old_location).flatten()[:3]
    return new_location

def transform_lidar_to_camera(vehicle_bbox_points, intrinsic_matrix, extrinsic_matrix):
    bb_cords = np.transpose(np.concatenate((np.array(vehicle_bbox_points), np.ones([8,1])),axis=1))
    cords_x_y_z = np.dot(extrinsic_matrix, bb_cords)[:3,:]
    cords_x_minus_z_y = np.concatenate([cords_x_y_z[0, :], -cords_x_y_z[2, :], cords_x_y_z[1, :]])
    bbox = np.transpose(np.dot(intrinsic_matrix, cords_x_minus_z_y))
    camera_bbox = np.transpose(np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]]))
    return camera_bbox

def filter_label(label):
    label = label.reshape(3,8)
    x_min,y_min = np.min(label,axis=1)[:2]
    x_max,y_max = np.max(label,axis=1)[:2]
    depth_min = np.min(label,axis=1)[2]
    # print(depth_min)
    # print(y_min,y_max)
    # y_min -= 40*depth_min/100
    # y_max -= 30*depth_min/100
    # print(y_min,y_max)
    return [x_min,y_min,x_max,y_max]

def post(camera_intrinsic_matrix,index):
    info = []
    info.append('P'+str(index)+':')
    for row in camera_intrinsic_matrix:
        for column in row:
            info.append(str(column))
        info.append(str(0))
    return info

def process_matrix(velo_to_cam_matrix):
    test = np.identity(4)
    test =  np.concatenate([-test[1, :], -test[2, :], test[0, :], test[3,:]]).reshape(4,4)
    velo_to_cam_matrix = np.dot(velo_to_cam_matrix, test)
    info = []
    info.append('Tr_velo_to_cam:')
    for row in velo_to_cam_matrix.tolist():
        for column in row:
            info.append(str(column))
            if len(info) == 13:
                return info

def find_Ad_vehicles_location(vehicles,labels):
    results = {}
    # return {}
    # print(labels)
    for vehicle in vehicles:
        lidar_location,lidar_rotation,tmp_rotation = get_sensor_transform(vehicle[-3:],labels,'lidar')
        camera_location,camera_rotation,id = get_sensor_transform(vehicle[-3:],labels,'camera')[0]
        results[vehicle] = [lidar_location,lidar_rotation,tmp_rotation,camera_location,camera_rotation,id]
    return results

def judge_in_ROI_numbers(other_vehilcle_label,AD_vehicles_location,_frame_id):
    ans = 0

    for AD_vehicle,location in AD_vehicles_location.items():
        if other_vehilcle_label[1] in AD_vehicle:
            # print(other_vehilcle_label,AD_vehicle,'-----------------------')
            ans+=1
            continue
        AD_vehicle_path = raw_data_path + '/' + AD_vehicle
        raw_data_file = os.listdir(AD_vehicle_path)
        raw_data_file.sort()
        lidar_raw_data_file = AD_vehicle_path + '/' + raw_data_file[-1] + '/' + _frame_id + 'ply'
        pcd = get_raw_lidar_data(lidar_raw_data_file, location[2])
        ego_raw_data_path = raw_data_path + '/' + AD_vehicle
        raw_data_file = os.listdir(ego_raw_data_path)
        raw_data_file.sort()
        for _file in raw_data_file:
            if str(location[5]) in _file and 'label' in _file:
                camera_2Dlabel_file = ego_raw_data_path + '/' + _file + '/' + _frame_id + 'txt'
                break
        try:
            camera_2Dlabel = np.loadtxt(camera_2Dlabel_file).reshape(-1,5)
        except:
            camera_2Dlabel = np.array([])
        other_bbox, bbox_extend, bbox_location, bbox_delta = get_vehicle_bbox(other_vehilcle_label,location[0],location[1],True)
        tmp2 = pcd.crop(other_bbox)
        tmp_2Dlabel = []
        for _label in camera_2Dlabel:
            if other_vehilcle_label[1] == str(int(_label[0])):
                tmp_2Dlabel = _label[1:]
                if tmp_2Dlabel[0] < 0:
                    tmp_2Dlabel[0] = 0
                if tmp_2Dlabel[1] < 0:
                    tmp_2Dlabel[1] = 0
                if tmp_2Dlabel[2] > 1242:
                    tmp_2Dlabel[2] = 1242
                if tmp_2Dlabel[3] > 375:
                    tmp_2Dlabel[3] = 375
                break
        if len(tmp2.points) >=10 and len(tmp_2Dlabel) != 0:
            ans+=1
        else:
            pass
    return ans

if __name__ == "__main__":
    raw_data_path = 'tmp/record2021_0721_2016'
    # raw_data_path = str(sys.argv[1])
    label_file_path = raw_data_path + '/label/'
    frames = os.listdir(label_file_path)

    def _read_imageset_file(path):
        with open(path, 'r') as f:
            lines = f.readlines()
        return [int(line) for line in lines]
    # gt_split_file = 'dataset/record2020_0903_0142_/img_list_test.txt'
    # frames = _read_imageset_file(gt_split_file)
    global_label_file_path = 'dataset/' + raw_data_path[4:] + '/global_label/'
    frames.sort()

    if len(frames) < 10:
        print("there is no enough data.")
    # exit()
    print(len(frames))
    frame_start,frame_end,frame_hz = 10,-72,1
    for _frame in frames[frame_start:frame_end:frame_hz]:
        if not os.path.exists(global_label_file_path):
            os.makedirs(global_label_file_path)
        shutil.copy(label_file_path+_frame,global_label_file_path)
        labels = np.loadtxt(label_file_path + _frame, dtype='str', delimiter=' ')
        # record2021_0716_2146
        tmp_car_id = ['255','261','263','252','247','271','280','254','267','266','278']
        # record2021_0720_1647
        tmp_car_id = ['311','317','319','308','303','327','336','310','323','322','334']
        AD_vehicles = [v for v in os.listdir(raw_data_path) if 'vehicle' in v]
        AD_vehicles_location = find_Ad_vehicles_location(AD_vehicles,labels)
        for ego_vehicle_label in labels:
            tmp_name = ego_vehicle_label[0] + '_' + ego_vehicle_label[1]
            # tmp_car = ['246','248','249','250','251','253','256','257','258','259','260','262','264','265','268','269','270','272','273','274',\
            # '275','276','277','279','281','282','283','284','285']
            if 'vehicle' in ego_vehicle_label[0]:# and tmp_name in AD_vehicles_location.keys():# and '319' in ego_vehicle_label[1]:
                # if ego_vehicle_label[1] not in tmp_car:
                #     continue
                try:
                    lidar_center, lidar_rotation, lidar_rotation_test = get_sensor_transform(ego_vehicle_label[1], labels, sensor='lidar')
                except:
                    continue
                # camera_center, camera_rotation = get_sensor_transform(ego_vehicle_label[1], labels, sensor='camera')
                
                calib_info_list = []
                for index in range(4):
                    calib_info_list.append(' '.join(post(camera_intrinsic_matrix,index)))
                tmp_info = ['R0_rect:'] + list(map(str,[1,0,0,0,1,0,0,0,1]))
                calib_info_list.append(' '.join((copy.deepcopy(tmp_info))))
                tmp_info[0] = 'Tr_imu_to_velo:'
                calib_info_list.append(' '.join(tmp_info))
                calib_info_list.append(' '.join(tmp_info))

    
                camera_info_list = get_sensor_transform(ego_vehicle_label[1], labels, sensor='camera')
                lidar_raw_data_file = get_ego_lidar_data_path(ego_vehicle_label, _frame[:-3])
                pointCloud = get_raw_lidar_data(lidar_raw_data_file, lidar_rotation_test)
                for index, camera in enumerate(camera_info_list):
                    camera_center, camera_rotation, camera_id = camera
                    velo_to_cam_matrix = get_matrix_from_origin_to_target(lidar_center,lidar_rotation,camera_center,camera_rotation)
                    post_velo_to_cam = process_matrix(velo_to_cam_matrix)
                    calib_info_list[-2] = ' '.join(post_velo_to_cam)

                    calib = get_calib_file(raw_data_path, _frame, ego_vehicle_label, index, calib_info_list)
                    tmp_pointCloud = process_pcd(pointCloud,calib)

                    ego_vehicle_location, ego_vehicle_rotation = get_vehicle_transform(ego_vehicle_label)
                    tmp_labels,bboxes = [],[]
                    label2d = [] 
                    label2point = [[],[]]
                    tmp_p = []
                    camera_2Dlabel_file = get_ego_camera_2Dlabel_path(ego_vehicle_label, camera_id, _frame[:-3])
                    try:
                        camera_2Dlabel = np.loadtxt(camera_2Dlabel_file).reshape(-1,5)
                    except:
                        camera_2Dlabel = np.array([])
                    for other_vehilcle_label in labels:
                        if 'vehicle' in other_vehilcle_label[0]:
                            if ego_vehicle_label[1] == other_vehilcle_label[1]:# or ego_vehicle_label[1] != '247' or other_vehilcle_label[1] not in ['229','216']:
                                continue
                            # object_detected_number = judge_in_ROI_numbers(other_vehilcle_label,AD_vehicles_location,_frame[:-3])
                            object_detected_number = 0
                            # exit()
                            other_location, other_rotation = get_vehicle_transform(other_vehilcle_label)
                            other_bbox, bbox_extend, bbox_location, bbox_delta = get_vehicle_bbox(other_vehilcle_label,lidar_center,lidar_rotation)
                            tmp = tmp_pointCloud.crop(other_bbox)
                            other_bbox, bbox_extend, bbox_location, bbox_delta = get_vehicle_bbox(other_vehilcle_label,lidar_center,lidar_rotation,True)
                            tmp2 = tmp_pointCloud.crop(other_bbox)
                            # if len(tmp.points) != len(tmp2.points):
                            # if True:
                            #     print(len(tmp.points),len(tmp2.points),ego_vehicle_label[1],other_vehilcle_label[1],_frame)

                            tmp_2Dlabel = []
                            for _label in camera_2Dlabel:
                                if other_vehilcle_label[1] == str(int(_label[0])):
                                    tmp_2Dlabel = _label[1:]
                                    if tmp_2Dlabel[0] < 0:
                                        tmp_2Dlabel[0] = 0
                                    if tmp_2Dlabel[1] < 0:
                                        tmp_2Dlabel[1] = 0
                                    if tmp_2Dlabel[2] > 1242:
                                        tmp_2Dlabel[2] = 1242
                                    if tmp_2Dlabel[3] > 375:
                                        tmp_2Dlabel[3] = 375
                                    break
                            if len(tmp.points) >= 10 and len(tmp_2Dlabel) != 0:
                            # if True:
                                # camera_label = transform_lidar_to_camera(other_bbox.get_box_points(),camera_intrinsic_matrix, velo_to_cam_matrix)
                                pcd = o3d.geometry.PointCloud()
                                tmp_pcd = np.array(np.array(other_bbox.get_box_points()), dtype=np.dtype('f4'))
                                pcd.points = o3d.utility.Vector3dVector(tmp_pcd)
                                pcd.paint_uniform_color([1,0,0])
                                tmp_p.append(pcd)

                                # if not all(camera_label.reshape(8,3)[:, 2] > 0):
                                #     print('# filter objects behind camera')
                                #     continue
                                # camera_label = filter_label(camera_label)
                                # if camera_label[0] > 1242 or camera_label[1] > 375 or camera_label[2] < 0 or camera_label[3] < 0: 
                                #     print('# filter objects behind camera')
                                #     continue

                                label2d.append(tmp_2Dlabel)
                                # camera_label = [np.array(i)[0][0] for i in camera_label]
                                other_location = other_bbox.get_center()
                                bboxes.append(other_bbox)
                                # print(other_location,float(other_vehilcle_label[-1]))
                                other_location -= np.array([0,0,float(other_vehilcle_label[-1])])
                                # other_location = transform_vehicle_to_lidar(other_location, matrix_to_ego)
                                alpha = - bbox_delta - math.atan(other_location[0]/other_location[2]) - np.pi
                                def process_theta(alpha):
                                    while alpha > np.pi:
                                        alpha -= np.pi*2
                                    while alpha < -np.pi:
                                        alpha += np.pi*2
                                    return alpha
                                alpha = process_theta(alpha)
                                # print(-bbox_delta-np.pi+lidar_rotation[2]+np.pi/2,np.radians(float(other_vehilcle_label[7])))
                                bbox_delta = process_theta(-bbox_delta-np.pi)
                                if len(tmp2.points) > 10:
                                    size = 0
                                    if len(tmp2.points) + other_location[0] < 250:
                                        size = 1
                                    if len(tmp2.points) + other_location[0] < 125:
                                        size = 2
                                    tmp_labels.append(['Car',str(object_detected_number),str(size),str(alpha),tmp_2Dlabel[0],tmp_2Dlabel[1],tmp_2Dlabel[2],tmp_2Dlabel[3],
                                                        str(bbox_extend[2]), str(bbox_extend[1]), str(bbox_extend[0]),
                                                        str(other_location[0]), str(other_location[1]), str(other_location[2]), str(bbox_delta), str(other_vehilcle_label[1])])
                                # else:
                                #     # tmp_labels.append(['DontCare','0','0',str(alpha),tmp_2Dlabel[0],tmp_2Dlabel[1],tmp_2Dlabel[2],tmp_2Dlabel[3],
                                #     tmp_labels.append(['Car','0','0',str(alpha),0,0,0,0,
                                #                         str(bbox_extend[2]), str(bbox_extend[1]), str(bbox_extend[0]),
                                #                         str(other_location[0]), str(other_location[1]), str(other_location[2]), str(bbox_delta), str(other_vehilcle_label[1])])
                                

                    
                    if len(tmp_labels) == 0:
                        tmp_labels.append(['DontCare','-1','-1','-10','522.25','202.35','547.77','219.71','-1','-1','-1','-1000','-1000','-1000','-10','-10'])
                    img_path = write_labels(raw_data_path, _frame, ego_vehicle_label, tmp_labels, pointCloud, 0, camera_id, calib_info_list, use_seg=True)
                    # print(label2d)
                    show_flag = False
                    if show_flag:
                        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
                        o3d.visualization.draw_geometries(bboxes+[pointCloud,mesh]+tmp_p,width=960,height=640,point_show_normal=True)
                        
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        img = np.array(o3d.io.read_image(img_path))
                        plt.imshow(img)
                        for rect in label2d:
                            width = rect[2] - rect[0]
                            height = rect[3] - rect[1]
                            rect = plt.Rectangle((rect[0],rect[1]),width,height,fill=False,color='b')
                            ax.add_patch(rect)
                        plt.show()
                    
                    # plt.close()
                    # o3d.visualization.draw_geometries([img],width=1242,height=375)
                    # exit()
            else:
                continue
    global_label_file_path = 'dataset/' + raw_data_path[4:] + '/img_list.txt'
    frames = [frame[:-4] for frame in frames[frame_start:frame_end:frame_hz]]
    # if not os.path.exists(global_label_file_path):
    #     os.makedirs(global_label_file_path[:-13])
    np.savetxt(global_label_file_path, np.array(frames),fmt='%s', delimiter=' ')