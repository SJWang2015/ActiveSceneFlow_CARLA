# ActiveSceneFlow_CARLA
The active scene flow estimation technology aims at higher quality of scene flow data through actively changing the ego-vehicle’s trajectory as well as sensor observation position.

## Abstract
<img src="https://github.com/SJWang2015/ActiveSceneFlow_CARLA/blob/main/media/motivation.png" width="1000" />
The active scene flow estimation technology aims at higher quality of scene flow data through actively changing the ego-vehicle's trajectory as well as sensor observation position. The major challenges for active 3D point cloud scene flow estimation for autonomous driving  (AD) include mis-registrations of point cloud frames, inaccurate scene prediction, and inefficient scene similarity evaluation. This paper presents a framework of active scene flow estimation for AD applications, which consists of three main modules: robust scene flow estimation, reliable reachable area detection, and efficient observation position decision. The novelty of this work is threefold: (1) developing a robust bi-direction attention-based mechanism neural network scene flow estimation method; (2) developing a reliable reachable area detection strategy through hidden points removing (HPR) based scene prediction and a legality checking scheme between the ego-vehicle and the road as well as the predicted scene; (3) developing an efficient observation position decision strategy through building a scene similarity measure, which can help evaluate the differences between two frames of point clouds from different views. The proposed method is trained and verified using a CARLA simulator based dataset, the FlyingThings3D and KiTTI datasets, which have the accurate scene flow ground-truth. The experiment results demonstrate the superior estimation performance and generalization capacity of our method for various AD scenes as well as different system configurations, compared with other SOTA methods.

##
This repository includes:
1. [Scene flow dataset](https://github.com/SJWang2015/ActiveSceneFlow_CARLA/blob/main/media/carla.gif) collection and anotation;
2. A reliable reachable area detection strategy through hidden points removing (HPR) based scene prediction and a legality checking scheme between
the ego-vehicle and the road as well as the predicted scene;
3. Actively select the optimal observation position based scene similarity metirc between 2 frames point clouds;
4. Scene flow estimation method.
network based refinement;
5. A [briefing video](https://github.com/SJWang2015/ActiveSceneFlow_CARLA/blob/main/media/1509_tip2_lr.mp4) for introducing this work in the media directory.

## Visualizaiton Videos
1. Scene flow dataset (Demo)          
<img src="https://github.com/SJWang2015/ActiveSceneFlow_CARLA/blob/main/media/carla.gif" width="600" />

2. Passive Trajectory VS Active Trajectory (Demo)

<img src="https://github.com/SJWang2015/ActiveSceneFlow_CARLA/blob/main/media/active_scene_flow.gif" width="600" />

## Citation
If you use this code for academic research, you are highly encouraged to cite the following paper:

```latex

@article{wang2023active,
  title={Active Scene Flow Estimation for Autonomous Driving via Real-Time Scene Prediction and Optimal Decision},
  author={Wang, Shuaijun and Gao, Rui and Han, Ruihua and Chen, Jianjun and Zhao, Zirui and Lyu, Zhijun and Hao, Qi},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2023},
  publisher={IEEE}
}
