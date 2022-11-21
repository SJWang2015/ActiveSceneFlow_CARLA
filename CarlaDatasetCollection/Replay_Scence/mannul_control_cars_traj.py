# -- coding:UTF-8 --
#!/usr/bin/env python
# Author:Kin Zhang
# email: kin_eng@163.com

# from Active_SF.generate_waypoints import Vehicle
import glob
import os
import sys
import time
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from carla import VehicleLightState as vls

import argparse
import logging
from numpy import random

def initial_spectator(world, spectator_point):
    spectator = world.get_spectator()
    spectator_point_transform = carla.Transform(carla.Location( spectator_point[0][0],
                                                                spectator_point[0][1],
                                                                spectator_point[0][2]), 
                                                carla.Rotation( spectator_point[1][0],
                                                                spectator_point[1][1],
                                                                spectator_point[1][2]))
    spectator.set_transform(spectator_point_transform)

def main():
    # synchronous_master = True
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='10.16.5.40',
        # default = 'lab910.kevinlad.cn',
        # default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=11,
        type=int,
        help='number of vehicles (default: 11)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=50,
        type=int,
        help='number of walkers (default: 50)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='vehicles filter (default: "vehicle.*")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='pedestrians filter (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Enanble')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Random device seed')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enanble car lights')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    file_name = "Town03_07_13_21_21_53_00_cmd_traj.npz" 
    # file_name = "Town03_07_15_21_23_12_39_cmd_traj.npz" 
    # file_name = "Town02_07_16_21_11_10_19_cmd_traj.npz"
    hist_data = np.load(file_name)
    vehicles = hist_data['vehicles']
    cmd_traj = hist_data['cmd_arr']
    # cmd_traj[:,:,2:5] = cmd_traj[:,:,2:5] * 0.01 # Only for Town03_07_13_21_21_53_00_cmd_traj.npz
    # spectator_point = [(-300,-300,260),(-30,40,0)]
    map_name = "Town03"
    spectator_point = [(-160,-200,200),(-30,40,0)]
    # map_name = "Town02"
    # spectator_point = [(100,150,150),(-60,90,0)]
    bInitialized = False
    # spectator_point = [(cmd_traj[0,3,2], cmd_traj[0,3,3], cmd_traj[0,3,4]),(-60,90,0)]
    vehicles_list = []
    # walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    # map_name = "Town02"
    client.load_world(map_name)
    synchronous_master = False
    random.seed(args.seed if args.seed is not None else int(time.time()))
    try:
        world = client.get_world()
        

        spectator = world.get_spectator()
        spectator_point_transform = carla.Transform(carla.Location( spectator_point[0][0],
                                                                    spectator_point[0][1],
                                                                    spectator_point[0][2]), 
                                                    carla.Rotation( spectator_point[1][0],
                                                                    spectator_point[1][1],
                                                                    spectator_point[1][2]))
        spectator.set_transform(spectator_point_transform)

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)

        if args.sync:
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                world.apply_settings(settings)
            else:
                synchronous_master = False

        blueprint_library = world.get_blueprint_library()
        spawn_points = []
        # spawn_points = world.get_map().get_spawn_points()
        # number_of_spawn_points = len(spawn_points)

        # if len(vehicles) < number_of_spawn_points:
        #     random.shuffle(spawn_points)
        # elif len(vehicles) > number_of_spawn_points:
        #     msg = 'requested %d vehicles, but could only find %d spawn points'
        #     logging.warning(msg, len(vehicles), number_of_spawn_points)
            # len(vehicles) = number_of_spawn_points
        # 设置固定点
        spawn_points = []
        for iV in range(len(vehicles)):
            v_loc = carla.Location(x=cmd_traj[0,iV,2], y=cmd_traj[0,iV,3], z=cmd_traj[0,iV,4])
            v_rot = carla.Rotation(pitch=cmd_traj[0,iV,6], yaw=cmd_traj[0,iV,7], roll=cmd_traj[0,iV,5])
            v_tsf = carla.Transform(v_loc, v_rot)
            spawn_points.append(v_tsf)
        number_of_vehicles = len(spawn_points)
        
        
        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor
        ApplyTransform = carla.command.ApplyTransform
        # ego_vehicle_bp = blueprint_library.find('vehicle.audi.a2')
        # ego_vehicle_bp.set_attribute('color', '0, 0, 0')
        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break
            blueprint = blueprint_library.find(vehicles[n][1])
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, False, traffic_manager.get_port())))
        
        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)
        
        if len(vehicles_list) == 0:
            return

        for n, transform in enumerate(spawn_points):
            if n >= len(vehicles_list):
                break
            vehicle = world.get_actor(vehicles_list[n])
            nb = vehicle.get_transform()
            print(vehicle.type_id)
            print(nb)
            # vehicle.set_transform(nb)
        iR = 0
        iRound = 0
        while True:
            batch_control = []
            # next_points = []
            # for iR in range(cmd_traj.shape[0]):
            for iV in range(len(vehicles_list)):
                v_loc = carla.Location(x=cmd_traj[iR,iV,2], y=cmd_traj[iR,iV,3], z=cmd_traj[iR,iV,4])
                v_rot = carla.Rotation(pitch=cmd_traj[iR,iV,6], yaw=cmd_traj[iR,iV,7], roll=cmd_traj[iR,iV,5])
                v_tsf = carla.Transform(v_loc, v_rot)
                # next_points.append(v_tsf)
                vehicle = world.get_actor(vehicles_list[iV])
                batch_control.append(ApplyTransform(vehicle, v_tsf))
                # nb = vehicle.set_transform(v_tsf)
                    # batch_control.append(nb)
            
            # for t in thread_list:
            #     batch_control.append(t.return_control())
            for response in client.apply_batch_sync(batch_control, False):
                if response.error:
                    logging.error(response.error)
            if synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()
            
            # top view
            if not(bInitialized):
                for iV, _ in enumerate(vehicles_list):
                    spectator = world.get_spectator()
                    ego_vehicle = world.get_actor(vehicles_list[iV])
                    transform = ego_vehicle.get_transform()
                    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=80),
                                                            carla.Rotation(pitch=-90)))
                    bInitialized = True
                    spectator.set_transform(spectator_point_transform)
            # else:
                # spectator.set_transform(spectator_point_transform)
            
            iR = (iR + 1) % cmd_traj.shape[0]
            if iR == 0:
                if iRound >= 20:
                    print("TASK COMPLETED. ByeBye.")
                    return
                iRound += 1      
                print("The %dth round." % iRound)
    finally:
        # 如果设置了同步记得 设置回去异步否则下次无法开启同步
        if args.sync and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
        # client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
        time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')