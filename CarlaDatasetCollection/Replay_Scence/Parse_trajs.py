import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pkgutil
import encodings
import os
import math

def all_encodings():
    modnames = set([modname for importer, modname, ispkg in pkgutil.walk_packages(
        path=[os.path.dirname(encodings.__file__)], prefix='')])
    aliases = set(encodings.aliases.aliases.values())
    return modnames.union(aliases)

# text = b'\xbc'
# for enc in all_encodings():
#     try:
#         msg = text.decode(enc)
#         print('Decoding {t} with {enc} is {m}'.format(t=text, enc=enc, m=msg))
#     except Exception:
#         continue
#     if msg == 'ñ':
#         print('Decoding {t} with {enc} is {m}'.format(t=text, enc=enc, m=msg))

class Frame:
    def __init__(self):
        self.FrameVehicle = ''
        self.frame_id = ''


class Vehicle:
    def __init__(self, frame_id, id, x, y, z, rx, py, yz):
        self.vehicle_name = ""
        self.frame_id = frame_id
        self.vehicle_spawn_id = frame_id
        self.vehicle_spawn_x = x
        self.vehicle_spawn_y = y
        self.vehicle_spawn_z = z
        self.vehicle_id = id
        self.vehicle_x = x
        self.vehicle_y = y
        self.vehicle_z = z
        self.vehicle_rx = rx
        self.vehicle_ry = py
        self.vehicle_rz = yz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_files', type=str, nargs='+', default='./logfile.txt')
    args = parser.parse_args()
    ax1 = ax2 = None
    ax1_legends = []
    ax1_legends = []

    
    map_pattern = r"Map: (?P<map>\w+)"
    date_pattern = r"Date: (?P<date>\d+[/]\d+[/]\d+) (?P<time>\d+[:]\d+[:]\d+)"
    for i, log_file in enumerate(args.log_files):
        with open(log_file, 'r') as file:
            log = file.read()
        # vehicle_pattern = r"Frame (?P<frame>\d+) at (?P<time>\d+.\d+) seconds+[\n ](?P<contents>[\s\S]+)" \
        #     r"Frame (?P<frame2>\d+) at (?P<time2>\d+.\d+) seconds" 
        # Map: Town03
        # Date: 07/13/21 21:53:00
        # map_pattern = r"Map: (?P<map>\w+)"
        # data_pattern = r"Date: (?P<date>\d+[/]\d+[/]\d+) (?P<time>\d+[:]\d+[:]\d+)"
        vehicle_pattern = r"Frame (?P<frame>\d+) at (?P<time>\d+.\d+|\d+.\d+e[+-][0-9]+|\d+) seconds+[\n]"
        # vehicle_pattern2 = r"Create (?P<id>\d+): (?P<name>[vehicle]+.\w+.\w+) [(](?P<att_id>\d+)[)] at [(](?P<x>[+-]?\d+.\d+|[+-]\d+.\d+e[+-][0-9]+|[+-]\d+), (?P<y>[+-]?\d+.\d+|[+-]\d+.\d+e[+-][0-9]+|[+-]\d+), (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)[)\n]+"\
        #                     r"(?P<nonsense>[\s\S]+|[number_of_wheels = 4][\n])"
        # vehicle_pattern2 = r"Create (?P<id>\d+): (?P<name>[vehicle]+.\w+.\w+) [(](?P<att_id>\d+)[)] at [(](?P<x>[+-]?\d+.\d+|[+-]\d+.\d+e[+-][0-9]+|[+-]\d+), (?P<y>[+-]?\d+.\d+|[+-]\d+.\d+e[+-][0-9]+|[+-]\d+), (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)[)\n]+"
        vehicle_loc_rot = r"Id: (?P<id>\d+) Location: [(](?P<x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+), (?P<y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+), (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)[)] "\
                        r"Rotation [(](?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+), (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+), (?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"
        # a = "Id: 93 Location: (-0.137634, 20742.7, 6.23198) Rotation (7.09023e-05, 0.550138, -0.137634)"
        # vehicle_loc_rot = r"Rotation [(](?P<r_x>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+), (?P<r_y>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+), (?P<r_z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)"
        # vehicle_pattern2 = r"Create (?P<id>\d+): (?P<name>[vehicle]+.\w+.\w+) [(](?P<att_id>\d+)[)] at [(](?P<x>[+-]?\d+.\d+|[+-]\d+.\d+e[+-][0-9]+|[+-]\d+), (?P<y>[+-]?\d+.\d+|[+-]\d+.\d+e[+-][0-9]+|[+-]\d+), (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)[\s\S]+"  \
        #                     r"number_of_wheels = 4"
        # a = "Create 338: vehicle.charger2020.charger2020 (1) at (13203, 20117, 50)\n" + "dsfsdf\n" + "number_of_wheels = 4"   
        vehicle_pattern2 = r"Create (?P<id>\d+): (?P<name>[vehicle]+.\w+.\w+) [(](?P<att_id>\d+)[)] at [(](?P<x>[+-]?\d+.\d+|[+-]\d+.\d+e[+-][0-9]+|[+-]\d+), (?P<y>[+-]?\d+.\d+|[+-]\d+.\d+e[+-][0-9]+|[+-]\d+), (?P<z>[+-]?\d+.\d+|[+-]?\d+.\d+e[+-][0-9]+|[+-]?\d+)[)\n]+"\
                            r"(.|[object_type = \n]+)number_of_wheels = 4"      
        log_info = []
        map_info = re.findall(map_pattern, log)
        date_info = re.findall(date_pattern, log)
        log_info = [map_info[0], date_info[0][0], date_info[0][1]]
        vehicle_name = []
        vehicle_spawn_id = []
        vehicle_spawn_x = []
        vehicle_spawn_y = []
        vehicle_spawn_z = []
        # vehicle_id = []
        # vehicle_x = []
        # vehicle_y = []
        # vehicle_z = []
        # vehicle_rx = []
        # vehicle_ry = []
        # vehicle_rz = []
        valid_frame = []
        vehicle_list = []
        frame_vehicles = []
        frame = Frame()
        contents = re.findall(vehicle_pattern, log)
        for r in re.findall(vehicle_pattern2, log):
                vehicle_list.append(r[:6])
                # vehicle_spawn_id.append(int(sub_ct_begin[0]))
                vehicle_name.append(str(r[1]))
                vehicle_spawn_x.append(float(r[3]))
                vehicle_spawn_y.append(float(r[4]))
                vehicle_spawn_z.append(float(r[5]))
        width = len(vehicle_list)
        ref_vId = [vehicle_list[i][0] for i in range(width)]

        for i_c in range(len(contents)-1):
            print(i_c)
            sub_ct_begin = contents[i_c]
            sub_ct_end = contents[i_c + 1]
            vehicle_sub_pattern = r"Frame " + sub_ct_begin[0] + r" at (?P<time>\d+.\d+|\d+.\d+e[+-][0-9]+|\d+) seconds+[\n ](?P<contents>[\s\S]+)" \
            r"Frame " + sub_ct_end[0] + r" at (?P<time2>\d+.\d+|\d+.\d+e[+-][0-9]+|\d+) seconds" 
            sub_cnt = re.findall(vehicle_sub_pattern, log)
            vehicle_spawn_id.append(int(sub_ct_begin[0]))
            sub_list = []
            for rt in re.findall(vehicle_loc_rot, sub_cnt[0][1]):
                if rt[0] in ref_vId:
                    # print(sub_ct_begin[0])
                    v = Vehicle(sub_ct_begin[0], rt[0], rt[1], rt[2], rt[3], rt[4], rt[5], rt[6])
                    sub_list.append(v)
                else:
                    continue
                # vehicle_id.append(int(rt[0]))
                # vehicle_x.append(float(rt[1]))
                # vehicle_y.append(float(rt[2]))
                # vehicle_z.append(float(rt[3]))
                # vehicle_rx.append(float(rt[4]))
                # vehicle_ry.append(float(rt[5]))
                # vehicle_rz.append(float(rt[6]))
            valid_frame.append(len(sub_list))
            frame.FrameVehicle = sub_list
            frame.frame_id = sub_ct_begin[0]
            frame_vehicles.append(frame)
            sub_list = []
            frame = Frame()

        print(v)
        height = len(frame_vehicles)
        depth = 8 # id, frame_id, x,y,z,rx,py,yz
        cmd_arr = np.full([height, width, depth], np.nan)


        valid_frame = np.array(valid_frame)
        valid_inds = np.argwhere(valid_frame<len(vehicle_list))
        # print(valid_inds)

        # fig, ax = plt.subplots()
        # ax.plot(np.arange(height), valid_frame, 'o-')
        # ax.set(xlabel='Id', ylabel='Count Num',
        # title='About as simple as it gets, folks')
        # ax.grid()
        # plt.show()
        
        # for iR in range(height):
        #     if len(vehicle_list) == len(frame_vehicles[iR].FrameVehicle):
        #         for iC in range(width):
                    # f_v = frame_vehicles[iR].FrameVehicle[iC]
                    # f_iC = ref_vId.index(f_v.vehicle_id)
                    # f_iR = int(f_v.frame_id) - 1
        #             sub_arr = np.array([int(f_v.vehicle_id), int(f_v.frame_id), float(f_v.vehicle_x), float(f_v.vehicle_x), float(f_v.vehicle_x),
        #                                     float(f_v.vehicle_rx), float(f_v.vehicle_ry), float(f_v.vehicle_rz)])
        #             sub_arr = np.tile(sub_arr, (height, 1))
        #             cmd_arr[:, iC, :] = sub_arr
 
        for iR in range(height):
            for iC in range(width):
                if iC < len(frame_vehicles[iR].FrameVehicle):
                    f_v = frame_vehicles[iR].FrameVehicle[iC]
                    f_iC = ref_vId.index(f_v.vehicle_id)
                    f_iR = int(f_v.frame_id) - 1
                    cmd_arr[f_iR, f_iC, :] = [int(f_v.vehicle_id), int(f_v.frame_id), float(f_v.vehicle_x) * 0.01, float(f_v.vehicle_y) * 0.01, float(f_v.vehicle_z) * 0.01,
                                            float(f_v.vehicle_rx), float(f_v.vehicle_ry), float(f_v.vehicle_rz)]
                    # print(f_iR, f_iC, cmd_arr[f_iR, f_iC, :])
        for iC in range(width):
            for i, iRow in enumerate(valid_inds):
                iR = iRow[0]
                if np.isnan(cmd_arr[iR,iC,0]):
                    Left = 0
                    Right = height - 1
                    for sub_iR in range(iR-1, 0, -1):
                        if not(np.isnan(cmd_arr[sub_iR,iC,0])):
                            Left = sub_iR
                            break
                    for sub_iR in range(iR+1, height, 1):
                        if not(np.isnan(cmd_arr[sub_iR,iC,0])):
                            Right = sub_iR
                            break
                    
                    f_v = cmd_arr[Left, iC, :]
                    if np.isnan(float(f_v[2])):
                        f_v = cmd_arr[Right, iC, :]
                    
                    if Left == 0:
                        sub_arr = np.tile(f_v, (Right-Left+1, 1))
                    else:
                        sub_arr = np.tile(f_v, (Right-Left, 1))
                    cmd_arr[iR:(Right+1), iC, :] = sub_arr
            
        cmd_arr[:,:,1] = np.tile(np.arange(height) + 1, (width,1)).T

        # np.save('./cmd_traj.npy', cmd_arr)
        log_date = datetime.strptime(date_info[0][0], '%m/%d/%y').strftime('%m_%d_%y')
        log_time = datetime.strptime(date_info[0][1], '%H:%M:%S').strftime('%H_%M_%S')
        file_name = map_info[0] + "_" + log_date + "_" + log_time + '_cmd_traj.npz'
        # np.save(file_name, cmd_arr)
        np.savez(file_name, cmd_arr=cmd_arr, vehicles = vehicle_list)
        # np.savez('cmd_traj')
        print("File Name:%s" % file_name)
        print("FINISH")
        
        # print(recorder_str)
        

if __name__ == '__main__':
    main()