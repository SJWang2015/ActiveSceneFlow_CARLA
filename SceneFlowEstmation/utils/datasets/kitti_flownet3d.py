import os
import glob
import numpy as np
from .generic import SceneFlowDataset


class Kitti(SceneFlowDataset):
    def __init__(self, root_dir, nb_points, mode):
        """
        Construct the KITTI scene flow datatset as in:
        Liu, X., Qi, C.R., Guibas, L.J.: FlowNet3D: Learning scene ﬂow in 3D 
        point clouds. IEEE Conf. Computer Vision and Pattern Recognition 
        (CVPR). pp. 529–537 (2019) 

        Parameters
        ----------
        root_dir : str
            Path to root directory containing the datasets.
        nb_points : int
            Maximum number of points in point clouds.

        """

        super(Kitti, self).__init__(nb_points)
        self.root_dir = root_dir
        self.filenames = self.make_dataset()
        self.mode = mode

        # Train / val / test split
        assert len(self.filenames) == 150, "Problem with size of training set"
        # ind_val = set(np.sort(np.linspace(0, 149, 70).astype("int")))
        # ind_all = set(np.sort(np.arange(150).astype("int")))
        ind_val = set(np.linspace(0, 149, 70).astype("int"))
        ind_all = set(np.arange(150).astype("int"))
        ind_train = ind_all - ind_val
        assert (
                len(ind_train.intersection(ind_val)) == 0
        ), "Train / Val not split properly"
        filenames = np.sort(self.filenames)
        if self.mode == "train":
            self.filenames = filenames[list(ind_train)]
        elif self.mode == "val":
            self.filenames = filenames[list(ind_all)]

    def __len__(self):

        return len(self.filenames)

    def make_dataset(self):
        """
        Find and filter out paths to all examples in the dataset. 
        
        """

        filenames = glob.glob(os.path.join(self.root_dir, "*.npz"))
        assert len(filenames) == 150, "Problem with size of kitti dataset"

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
            sequence = [data["pos1"][:, (1, 2, 0)], data["pos2"][:, (1, 2, 0)]]
            ground_truth = [
                np.ones_like(data["pos1"][:, 0:1]),
                data["gt"][:, (1, 2, 0)],
            ]

        # Restrict to 35m
        loc = sequence[0][:, 2] < 30
        sequence[0] = sequence[0][loc]
        ground_truth[0] = ground_truth[0][loc]
        ground_truth[1] = ground_truth[1][loc]
        loc = sequence[1][:, 2] < 30
        sequence[1] = sequence[1][loc]

        return sequence, ground_truth
