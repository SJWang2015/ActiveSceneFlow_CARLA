import os
import glob
import numpy as np
import torch
import numpy as np
from torch.utils.data import Dataset


class Batch:
    def __init__(self, batch):
        """
        Concatenate list of dataset.generic.SceneFlowDataset's item in batch 
        dimension.

        Parameters
        ----------
        batch : list
            list of dataset.generic.SceneFlowDataset's item.

        """

        self.data = {}
        batch_size = len(batch)
        # for key in ["sequence", "ground_truth", "mask"]:
        for key in ["sequence", "ground_truth"]:
            self.data[key] = []
            # if key in "sequence":
            #     data_size = 2
            # else:
            #     data_size = 6
            for ind_seq in range(2):
                tmp = []
                for ind_batch in range(batch_size):
                    tmp.append(batch[ind_batch][key][ind_seq])
                self.data[key].append(torch.cat(tmp, 0))

    def __getitem__(self, item):
        """
        Get 'sequence' or 'ground_thruth' from the batch.
        
        Parameters
        ----------
        item : str
            Accept two keys 'sequence' or 'ground_truth'.

        Returns
        -------
        list(torch.Tensor, torch.Tensor)
            item='sequence': returns a list [pc1, pc2] of point clouds between 
            which to estimate scene flow. pc1 has size B x n x 3 and pc2 has 
            size B x m x 3.
            
            item='ground_truth': returns a list [ego_flow, flow]. ego_flow has size 
            B x n x 3 and flow has size B x n x 3. flow is the ground truth 
            scene flow between pc1 and pc2. flow is the ground truth scene 
            flow. ego_flow is motion vector of the ego-vehicle.
        """
        return self.data[item]

    def to(self, *args, **kwargs):

        for key in self.data.keys():
            self.data[key] = [d.to(*args, **kwargs) for d in self.data[key]]

        return self

    def pin_memory(self):

        for key in self.data.keys():
            self.data[key] = [d.pin_memory() for d in self.data[key]]

        return self


class SceneFlowDataset(Dataset):
    def __init__(self, nb_points, rm_ground=False, use_fg_inds=True, use_hybrid_sample=False, cache=None):
        """
        Abstract constructor for scene flow datasets.
        
        Each item of the dataset is returned in a dictionary with two keys:
            (key = 'sequence', value=list(torch.Tensor, torch.Tensor)): 
            list [pc1, pc2] of point clouds between which to estimate scene 
            flow. pc1 has size 1 x n x 3 and pc2 has size 1 x m x 3.
            
            (key = 'ground_truth', value = list(torch.Tensor, torch.Tensor)): 
            list [mask, flow]. mask has size 1 x n x 1 and pc1 has size 
            1 x n x 3. flow is the ground truth scene flow between pc1 and pc2.
            mask is binary with zeros indicating where the flow is not 
            valid/occluded.

        Parameters
        ----------
        nb_points : int
            Maximum number of points in point clouds: m, n <= self.nb_points.

        """
        super(SceneFlowDataset, self).__init__()
        self.nb_points = nb_points
        self.rm_ground = rm_ground

        self.use_fg_inds = use_fg_inds
        self.hybrid_sample = use_hybrid_sample
        self.pre_segfrnt = True
        if cache is None:
            self.cache = {}
        else:
            self.cache = cache

        self.cache_size = 30000

    def __getitem__(self, idx):
        if idx in self.cache:
            if self.use_fg_inds:
                data = {"sequence": self.cache[idx]["sequence"], "ground_truth": self.cache[idx]["ground_truth"], "mask": self.cache[idx]["mask"]}
            else:
                data = {"sequence": self.cache[idx]["sequence"], "ground_truth": self.cache[idx]["ground_truth"]}
        else:
            if self.use_fg_inds:
                sequence, ground_truth, mask = self.to_torch(
                    *self.subsample_points(*self.load_sequence(idx))
                )
                data = {"sequence": sequence, "ground_truth": ground_truth, "mask": mask}
            else:
                sequence, ground_truth = self.to_torch(
                *self.subsample_points(*self.load_sequence(idx))
                )
                data = {"sequence": sequence, "ground_truth": ground_truth}

        if len(self.cache) < self.cache_size:
            self.cache[idx] = data
        return data

    def to_torch(self, sequence, ground_truth, mask=None):
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
        if self.use_fg_inds:
            mask = [
                torch.unsqueeze(torch.from_numpy(msk), 0).float() for msk in mask
            ]
            return sequence, ground_truth, mask
        else:
            return sequence, ground_truth
            
    def hybrid_sample_points(self, mask, num_pts=4500):
        '''
        Seprately sample 3D points from the original data
        
        '''
        bkg_num = self.nb_points - num_pts
        src_frnt_num = (np.sum(mask)).astype("int32")
        src_bkg_index = np.argwhere(mask == 0).squeeze()
        if src_frnt_num < num_pts:
            bkg_ind1 = np.random.choice(src_bkg_index.shape[0], self.nb_points-src_frnt_num, replace=False)
            src_frnt_index = np.argwhere(mask == 1)
            src_ind = np.hstack([src_frnt_index[:,0], bkg_ind1])
            # src_ind = src_bkg_index[bkg_ind1]
        else:
            src_frnt_index = np.argwhere(mask == 1).squeeze()
            frnt_ind1 = np.random.choice(src_frnt_index.shape[0], num_pts, replace=False)
            bkg_ind1 = np.random.choice(src_bkg_index.shape[0], bkg_num, replace=False)
            src_frnt_ind = src_frnt_index[frnt_ind1]
            src_bkg_ind = src_bkg_index[bkg_ind1]
            src_ind = np.hstack([src_frnt_ind, src_bkg_ind])
        mask = mask[src_ind]
        return src_ind, mask
    
    
    def subsample_points(self, sequence, ground_truth, mask):
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
        ind_mask = mask
        if self.rm_ground:
            not_ground1 = np.logical_not(sequence[0][:, -1] < -3.3)
            sequence[0] = sequence[0][not_ground1]
            ground_truth = [g[not_ground1] for g in ground_truth]
            # ground_truth[0] = ground_truth[0][not_ground1]
            # ground_truth[1] = ground_truth[1][not_ground1]
            not_ground2 = np.logical_not(sequence[1][:, -1] < -3.3)
            sequence[1] = sequence[1][not_ground2]
            if len(mask) >= 2 and self.use_fg_inds:
                mask[0] = mask[0][not_ground1]
                mask[1] = mask[1][not_ground2]
        
        if True:
            if self.hybrid_sample:
                ind_mask1, mask[0] = self.hybrid_sample_points(mask[0], num_pts=2000)
                ind_mask2, mask[1] = self.hybrid_sample_points(mask[1], num_pts=2000)
            else:
                ind_mask = None

            if self.pre_segfrnt:
                if ind_mask != None:
                    sequence[0] = sequence[0][ind_mask1,:]
                    sequence[1] = sequence[1][ind_mask2,:]
                    ground_truth[0] = ground_truth[0][ind_mask1,:]
                    ground_truth[1] = ground_truth[1][ind_mask1,:]
                else:
                    sequence[0] = sequence[0][(mask[0]).astype("bool")]
                    sequence[1] = sequence[1][(mask[1]).astype("bool")]
                    ground_truth[0] = ground_truth[0][(mask[0]).astype("bool")]
                    ground_truth[1] = ground_truth[1][(mask[0]).astype("bool")]
             
        # Choose points in first scan
        # ind1 = np.random.permutation(sequence[0].shape[0])[: self.nb_points]
        # print("pc1.shape:%d\tpc1.shape:%d\t"%(sequence[0].shape[0], sequence[1].shape[0]))
        if self.nb_points < sequence[0].shape[0]:
            ind1 = np.random.choice(sequence[0].shape[0], self.nb_points, replace=False)
        else:
            ind1 = np.random.choice(sequence[0].shape[0], self.nb_points, replace=True)
        sequence[0] = sequence[0][ind1]
        # Choose point in second scan
        # ind2 = np.random.permutation(sequence[1].shape[0])[: self.nb_points]
        if self.nb_points < sequence[1].shape[0]:
            ind2 = np.random.choice(sequence[1].shape[0], self.nb_points, replace=False)
        else:
            ind2 = np.random.choice(sequence[1].shape[0], self.nb_points, replace=True)
        sequence[1] = sequence[1][ind2]
        
        # ground_truth = [g[ind1] for g in ground_truth]
        if len(ground_truth[0].shape) == 1:
            ground_truth[0] = ground_truth[0][ind1]  
            # ground_truth[0] = np.ones(ground_truth[1].shape) * ground_truth[0].reshape(1,3)
            # ground_truth[1] = ground_truth[1][ind1]
            # ground_truth[2] = ground_truth[2][ind1]  
            # # ground_truth[2] = np.ones(ground_truth[1].shape) * ground_truth[0].reshape(1,3)
            # ground_truth[3] = ground_truth[3][ind1]
        else:
            ground_truth = [g[ind1] for g in ground_truth[:2]]
            # if len(mask) >= 2 and self.use_fg_inds and not self.pre_segfrnt:
            mask[0] = mask[0][ind1]
            mask[1] = mask[1][ind2]
            # else:
            #     mask = []

        
        # ground_truth[0] = ground_truth[0][ind1] 

        return sequence, ground_truth, mask

    def load_sequence(self, idx):
        """
        Abstract function to be implemented to load a sequence of point clouds.

        Parameters
        ----------
        idx : int
            Index of the sequence to load.

        Must return:
        -------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene 
            flow. pc1 has size N x 3 and pc2 has size M x 3.
            
        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size N x 1 and pc1 has size N x 3. 
            flow is the ground truth scene flow between pc1 and pc2. mask is 
            binary with zeros indicating where the flow is not valid/occluded.

        """

        raise NotImplementedError



class CARLA3D(SceneFlowDataset):
    def __init__(self, root_dir, nb_points, mode, val_num=282, rm_ground=False, use_fg_inds=True, use_hybrid_sample=True):
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

        super(CARLA3D, self).__init__(nb_points, rm_ground, use_hybrid_sample=use_hybrid_sample)
        self.mode = mode
        self.val_num = val_num
        self.root_dir = root_dir
        self.use_fg_inds = use_fg_inds
        # invalid_files = np.load('./invalid_filename_2023.npz')
        # if nb_points == 2048:
        #     self.invalid_files = invalid_files['invalid_name']
        # elif nb_points == 4096:
        #     self.invalid_files = invalid_files['invalid_name2']
        # elif nb_points == 8192:
        #     self.invalid_files = invalid_files['invalid_name3']   
        # else:
        #     self.invalid_files = []
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
            sub_path = os.path.join(self.root_dir, sub_dir) + "/rm_road/SF/"
            for sub_sub_dir in sorted(os.listdir(sub_path)): 
            # all_sub_paths = sorted(os.listdir(sub_path))
            # for sub_sub_dir in all_sub_paths:
                pattern = sub_path + '/' + sub_sub_dir + "/" + "*.npz"
                # pattern = sub_path + "/" + "*.npz"
                filenames += glob.glob(pattern)
                pattern = self.root_dir + "/" + "*.npz"
                pattern = sub_path + "/" + "*.npz"
                filenames += glob.glob(pattern)

        # pattern = self.root_dir + "/" + "*.npz"
        # filenames += glob.glob(pattern)

        # Train / val / test split
        filenames = np.sort(filenames)


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

            # file_name = self.filenames[idx]
            return sequence, ground_truth, mask
        

        
