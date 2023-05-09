import os
import glob
import numpy as np
from .generic import SceneFlowDataset


class CARLA3D(SceneFlowDataset):
    def __init__(self, root_dir, nb_points, mode, val_num=282, rm_ground=True, use_fg_inds=True):
        """
        Construct the FlyingThing3D datatset as in:
        Liu, X., Qi, C.R., Guibas, L.J.: FlowNet3D: Learning scene ﬂow in 3D
        point clouds. IEEE Conf. Computer Vision and Pattern Recognition
        (CVPR). pp. 529–537 (2019)

        Parameters
        ----------
        root_dir : str
            Path to root directory containing the datasets.
        nb_points : int
            Maximum number of points in point clouds.
        mode : str
            'train': training dataset.

            'val': validation dataset.

            'test': test dataset

        """

        super(CARLA3D, self).__init__(nb_points, rm_ground)
        self.mode = mode
        self.val_num = val_num
        self.root_dir = root_dir
        self.use_fg_inds = use_fg_inds
        self.filenames = self.get_file_list()

    def __len__(self):

        return len(self.filenames)

    def get_file_list(self):
        """
        Find and filter out paths to all examples in the dataset.

        """
        filenames = []
        for sub_dir in sorted(os.listdir(self.root_dir)):
            # if self.mode == "train" or self.mode == "val":
            sub_path = os.path.join(self.root_dir, sub_dir)
            # all_sub_paths = sorted(os.listdir(sub_path))
            # for sub_sub_dir in all_sub_paths:
                # pattern = sub_path + '/' + sub_sub_dir + "/" + "*.npz"
            pattern = sub_path + "/" + "*.npz"
            filenames += glob.glob(pattern)

        # Train / val / test split
        filenames = np.sort(filenames)
        # if self.mode == "train" or self.mode == "val":
        #     filenames = filenames[:-self.val_num]
        #     # ind_val = set(np.linspace(0, len(filenames) - 1, self.val_num).astype("int"))
        #     # ind_all = set(np.arange(len(filenames)).astype("int"))
        #     # ind_train = ind_all - ind_val
        #     # assert (
        #     #         len(ind_train.intersection(ind_val)) == 0
        #     # ), "Train / Val not split properly"
        #     # filenames = np.sort(filenames)
        #     # if self.mode == "train":
        #     #     filenames = filenames[list(ind_train)]
        #     # elif self.mode == "val":
        #     #     filenames = filenames[list(ind_val)]
        # if self.mode == "test":
        #     filenames = filenames[-self.val_num:]


        return filenames

    def load_sequence(self, idx):
        """
        Load a sequence of point clouds.

        Parameters
        ----------
        idx : int
            Index of the sequence to load.

        Returns
        -------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene
            flow. pc1 has size n x 3 and pc2 has size m x 3.

        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size n x 1 and pc1 has size n x 3. 
            flow is the ground truth scene flow between pc1 and pc2. mask is 
            binary with zeros indicating where the flow is not valid/occluded.

        """

        # Load data
        with np.load(self.filenames[idx]) as data:
            sequence = [data["pos1"], data["pos2"]]
            if "pre_ego_flow" not in data and "s_fg_mask" not in data and not self.use_fg_inds:
                ground_truth = [data["ego_flow"], data["gt"]]
            elif "pre_ego_flow" not in data:
                ground_truth = [data["ego_flow"], data["gt"]]
            else:
                ground_truth = [data["ego_flow"], data["gt"], data["pre_ego_flow"], data["pre_gt"]]
            
            if "s_fg_mask" in data and "t_fg_mask" in data:
                mask = [data["s_fg_mask"], data["t_fg_mask"]]
            else:
                mask = []


            return sequence, ground_truth, mask
        

        
