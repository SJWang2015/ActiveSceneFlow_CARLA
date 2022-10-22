# ActiveSceneFlow_CARLA
The active scene flow estimation technology aims at higher quality of scene flow data through actively changing the ego-vehicle’s trajectory as well as sensor observation position.

This repository includes:
1. Scene flow dataset collection and anotation;
2. A reliable reachable area detection strategy through hidden points removing (HPR) based scene prediction and a legality checking scheme between
the ego-vehicle and the road as well as the predicted scene;
3. Actively select the optimal observation position based scene similarity metirc between 2 frames point clouds;
4. Scene flow estimation method with Kuhn-Munkres (KM) scheme based coarse estimation and self-attention neural
network based refinement;

