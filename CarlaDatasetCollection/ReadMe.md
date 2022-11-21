# Active Scene Flow Dataset

1. 随机生成运动场景，并记录log
使用CARLA官方的代码生成含有多个运动目标的交通场景，并使用对应的代码生成log文件

2. 运行
 ```
python3 Replay_Scene/Replay_log.py
 ```
得到对应的ASCII编码的TXT文件；

3. 运行
 ```
python3 Replay_Scene/Parse_trajs.py
 ```
对step 2中生成的文件进行解析，生成所有汽车的运动轨迹的npz运动文件；

4. 运行
 ```
python3 Replay_Scene/mannul_control_cars_traj.py
 ```
用于测试step 3生成的运动轨迹的文件，实现场景回放；

5. 运行
 ```
python3 Scenario_Traj.py spawn
 ```
查看对应的场景下所有车辆的位置；[详细参考](https://github.com/zijianzhang/CARLA_INVS)

运行
 ```
python3 Scenario_Traj.py record xx,xx[随便写一个就行] xx,xx[采集车]
 ```
使用指定的采集车进行数据采集；[详细参考](https://github.com/zijianzhang/CARLA_INVS)

6. 运行
 ```
Python3 Process.py [dir:处理后数据的存储文件夹]
 ```

7. 运行
 ```
Python3 Generate_Sceneflow.py [dir:处理后数据的存储文件夹]
 ```
可以查看生成的场景流的数据，对应场景流运动的颜色调色板与[光流运动的调色板](https://github.com/tomrunia/OpticalFlow_Visualization)一致
