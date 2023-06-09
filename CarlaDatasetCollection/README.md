# (Active & Passive) Scene Flow Dataset

1. Randomly generate motion scenes and record the log file;
 ```
python3 start_recording.py -n xxx
 ```

2. Run the following command to get the log file with ASCII encoding;
 ```
python3 Replay_Scene/Replay_log.py
 ```

3. Paraser the log file (in step 2) and output all the vehicles trajectories file (*.npz);
 ```
python3 Replay_Scene/Parse_trajs.py
 ```

4. (Optional) Test the trajectories file (*.npz) and replay the traffic scenarios;
 ```
python3 Replay_Scene/mannul_control_cars_traj.py
 ```

5. (Optional) Inspect the spawn point of the map; [Reference](https://github.com/zijianzhang/CARLA_INVS)
 ```
python3 Scenario_Traj.py spawn
 ```

Run data collection command; [Reference](https://github.com/zijianzhang/CARLA_INVS)
 ```
python3 Scenario_Traj.py record xx,xx[Just write one digit you like] xx,xx[Data Collection Vechicles]
 ```

6. Process the dataset into KITTI format;
 ```
Python3 Process.py [dir: the directory to store the output dataset ]
 ```

7. Annotate the scene flow dataset
 ```
Python3 Generate_Sceneflow.py [dir: the directory to store the output dataset ]
 ```
[Scene flow fields color palette](https://github.com/tomrunia/OpticalFlow_Visualization)
