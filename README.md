# ActiveSceneFlow_CARLA
The active scene flow estimation technology aims at higher quality of scene flow data through actively changing the ego-vehicle’s trajectory as well as sensor observation position.

## Abstract
<img src="https://github.com/SJWang2015/ActiveSceneFlow_CARLA/blob/main/media/motivation.png" width="1000" />
The active scene flow estimation technology aims at higher quality of scene flow data through actively changing the 
ego-vehicle’s trajectory as well as sensor observation position.
The major challenges for active 3D point cloud scene flow
estimation for autonomous driving (AD) include mis-registrations
of point cloud frames, inaccurate scene prediction, and inefficient
scene similarity evaluation. This paper presents a framework of
active scene flow estimation for AD applications, which consists
of three main modules: robust scene flow estimation, reliable
reachable area detection, and efficient observation position de-
cision. The novelty of this work is threefold: (1) developing
a robust scene flow estimation method with Kuhn-Munkres
(KM) scheme based coarse estimation and self-attention neural
network based refinement; (2) developing a reliable reachable
area detection strategy through hidden points removing (HPR)
based scene prediction and a legality checking scheme between
the ego-vehicle and the road as well as the predicted scene;
(3) developing an efficient observation position decision strategy
through building a scene similarity measure, which can help
evaluate the differences between two frames of point clouds from
different views. The proposed method is trained and verified
using a CARLA simulator based dataset, which has the accurate
scene flow ground-truth. The experiment results demonstrate the
superior estimation performance and generalization capacity of
our method for various AD scenes as well as different system
configurations, compared with other SOTA methods

##
This repository includes:
1. Scene flow dataset collection and anotation;
2. A reliable reachable area detection strategy through hidden points removing (HPR) based scene prediction and a legality checking scheme between
the ego-vehicle and the road as well as the predicted scene;
3. Actively select the optimal observation position based scene similarity metirc between 2 frames point clouds;
4. Scene flow estimation method with Kuhn-Munkres (KM) scheme based coarse estimation and self-attention neural
network based refinement.

