import os
import glob
import numpy as np
from .generic import SceneFlowDataset
import torch
from torch.utils.data import Dataset

DEPTH_THRESHOLD = 35.0
class FT3D(Dataset):
    def __init__(self, root_dir, nb_points, mode, cache=None):
        """
        Construct the FlyingThing3D datatset as in:
        Gu, X., Wang, Y., Wu, C., Lee, Y.J., Wang, P., HPLFlowNet: Hierarchical
        Permutohedral Lattice FlowNet for scene ﬂow estimation on large-scale 
        point clouds. IEEE Conf. Computer Vision and Pattern Recognition 
        (CVPR). pp. 3254–3263 (2019) 

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
        self.mode = mode
        self.root_dir = root_dir
        self.DEPTH_THRESHOLD = DEPTH_THRESHOLD
        self.no_corr = True
        self.allow_less_points = False
        self.filenames = self.get_file_list()
        self.nb_points = nb_points
        if cache is None:
            self.cache = {}
        else:
            self.cache = cache

        self.cache_size = 30000

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if idx in self.cache:
            data = self.cache[idx]#{"sequence": self.cache[idx]["sequence"], "ground_truth": self.cache[idx]["ground_truth"]}
        else:
            sequence, ground_truth = self.to_torch(
                *self.subsample_points(*self.load_sequence(idx))
            )
            data = {"sequence": sequence, "ground_truth": ground_truth}

        if len(self.cache) < self.cache_size:
            self.cache[idx] = data
        return data

    def to_torch(self, sequence, ground_truth):
        """
        Convert numpy array and torch.Tensor.

        Parameters
        ----------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene
            flow. pc1 has size n x 3 and pc2 has size m x 3.

        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size n x 1 and pc1 has size n x 3.
            flow is the ground truth scene flow between pc1 and pc2. mask is
            binary with zeros indicating where the flow is not valid/occluded.

        Returns
        -------
        sequence : list(torch.Tensor, torch.Tensor)
            List [pc1, pc2] of point clouds between which to estimate scene
            flow. pc1 has size 1 x n x 3 and pc2 has size 1 x m x 3.

        ground_truth : list(torch.Tensor, torch.Tensor)
            List [mask, flow]. mask has size 1 x n x 1 and pc1 has size
            1 x n x 3. flow is the ground truth scene flow between pc1 and pc2.
            mask is binary with zeros indicating where the flow is not
            valid/occluded.

        """

        sequence = [torch.unsqueeze(torch.from_numpy(s), 0).float() for s in sequence]
        ground_truth = [
            torch.unsqueeze(torch.from_numpy(gt), 0).float() for gt in ground_truth
        ]

        return sequence, ground_truth

    def subsample_points(self, sequence, ground_truth):
        """
        Subsample point clouds randomly.

        Parameters
        ----------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene
            flow. pc1 has size 1 x N x 3 and pc2 has size 1 x M x 3.

        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size 1 x N x 1 and pc1 has size
            1 x N x 3. flow is the ground truth scene flow between pc1 and pc2.
            mask is binary with zeros indicating where the flow is not
            valid/occluded.

        Returns
        -------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene
            flow. pc1 has size 1 x n x 3 and pc2 has size 1 x m x 3. The n
            points are chosen randomly among the N available ones. The m points
            are chosen randomly among the M available ones. If N, M >=
            self.nb_point then n, m = self.nb_points. If N, M <
            self.nb_point then n, m = N, M.

        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size 1 x n x 1 and pc1 has size
            1 x n x 3. flow is the ground truth scene flow between pc1 and pc2.
            mask is binary with zeros indicating where the flow is not
            valid/occluded.

        """

        # Choose points in first scan
        # ind1 = np.random.permutation(sequence[0].shape[0])[: self.nb_points]
        # sequence[0] = sequence[0][ind1]
        # ground_truth = [g[ind1] for g in ground_truth]

        # # Choose point in second scan
        # ind2 = np.random.permutation(sequence[1].shape[0])[: self.nb_points]
        # sequence[1] = sequence[1][ind2]
        if self.DEPTH_THRESHOLD > 0:
            near_mask = np.logical_and(sequence[0][:, 2] < self.DEPTH_THRESHOLD, sequence[1][:, 2] < self.DEPTH_THRESHOLD)
        else:
            near_mask = np.ones(sequence[0].shape[0], dtype=np.bool)
        
        indices = np.where(near_mask)[0]
        if len(indices) == 0:
            print('indices = np.where(mask)[0], len(indices) == 0')
            return None, None, None

        if self.nb_points > 0:
            try:
                sampled_indices1 = np.random.choice(indices, size=self.nb_points, replace=False, p=None)
                if self.no_corr:
                    sampled_indices2 = np.random.choice(indices, size=self.nb_points, replace=False, p=None)
                else:
                    sampled_indices2 = sampled_indices1
            except ValueError:
                '''
                if not self.allow_less_points:
                    print('Cannot sample {} points'.format(self.num_points))
                    return None, None, None
                else:
                    sampled_indices1 = indices
                    sampled_indices2 = indices
                '''
                if not self.allow_less_points:
                    #replicate some points
                    sampled_indices1 = np.random.choice(indices, size=self.nb_points, replace=True, p=None)
                    if self.no_corr:
                        sampled_indices2 = np.random.choice(indices, size=self.nb_points, replace=True, p=None)
                    else:
                        sampled_indices2 = sampled_indices1
                else:
                    sampled_indices1 = indices
                    sampled_indices2 = indices
        else:
            sampled_indices1 = indices
            sampled_indices2 = indices

        # pc1 = pc1[sampled_indices1]
        # pc2 = pc2[sampled_indices2]
        # sf = sf[sampled_indices1]
        sequence[0] = sequence[0][sampled_indices1]
        ground_truth = [g[sampled_indices1] for g in ground_truth]
        sequence[1] = sequence[1][sampled_indices2]

        return sequence, ground_truth


    def get_file_list(self):
        """
        Find and filter out paths to all examples in the dataset. 
        """

        # Get list of filenames / directories
        if self.mode == "train" or self.mode == "val":
            pattern = "train/0*"
        elif self.mode == "test":
            pattern = "val/0*"
        else:
            raise ValueError("Mode " + str(self.mode) + " unknown.")
        filenames = glob.glob(os.path.join(self.root_dir, pattern))

        # HPLFlowNet/FlyingThings3D_subset_processed_35m/train/0004781 has only 8144 points, exclude it here.
        orig_filenames_len = len(filenames)
        if self.mode == "train" or self.mode == "val":
            filenames = list(filter(lambda p: "train/0004781" not in p, filenames))
            cur_filenames_len = len(filenames)
            if cur_filenames_len == orig_filenames_len - 1:
                # add the first one example to make the length unchanged.
                filenames.append(filenames[0])

        # Train / val / test split
        if self.mode == "train" or self.mode == "val":
            assert len(filenames) == 19640, "Problem with size of training set"
            ind_val = set(np.linspace(0, 19639, 2000).astype("int"))
            ind_all = set(np.arange(19640).astype("int"))
            ind_train = ind_all - ind_val
            assert (
                len(ind_train.intersection(ind_val)) == 0
            ), "Train / Val not split properly"
            filenames = np.sort(filenames)
            if self.mode == "train":
                filenames = filenames[list(ind_all)]
            elif self.mode == "val":
                filenames = filenames[list(ind_val)]
        else:
            assert len(filenames) == 3824, "Problem with size of test set"

        return list(filenames)

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
        sequence = []  # [Point cloud 1, Point cloud 2]
        for fname in ["pc1.npy", "pc2.npy"]:
            pc = np.load(os.path.join(self.filenames[idx], fname))
            pc[..., 0] *= -1
            pc[..., -1] *= -1
            sequence.append(pc)
        ground_truth = [
            np.ones_like(sequence[0][:, 0:1]),
            sequence[1] - sequence[0],
        ]  # [Occlusion mask, flow]
        return sequence, ground_truth
        # pos1 = torch.from_numpy(sequence[0])
        # pos2 = torch.from_numpy(sequence[1])
        # flow = torch.from_numpy(ground_truth)

        # return pos1, pos2, flow
