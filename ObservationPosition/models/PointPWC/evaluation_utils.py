"""
Evaluation metrics
Borrowed from HPLFlowNet
Date: May 2020

@inproceedings{HPLFlowNet,
  title={HPLFlowNet: Hierarchical Permutohedral Lattice FlowNet for
Scene Flow Estimation on Large-scale Point Clouds},
  author={Gu, Xiuye and Wang, Yijie and Wu, Chongruo and Lee, Yong Jae and Wang, Panqu},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2019 IEEE International Conference on},
  year={2019}
}
"""

import numpy as np


def evaluate_3d(sf_pred, sf_gt, mask=None):
    """
    sf_pred: (N, 3)
    sf_gt: (N, 3)
    """
    l2_norm = np.linalg.norm(sf_gt - sf_pred, axis=-1)
    sf_norm = np.linalg.norm(sf_gt, axis=-1)
    if mask is not None:
      l2_norm *= mask
      sf_norm *= mask
    # EPE3D = l2_norm.mean()
    mask_sum = np.sum(mask, 1)
    EPE3D = np.sum(l2_norm, 1) / (mask_sum + 1e-4)
    EPE3D = np.mean(EPE3D)

    
    relative_err = l2_norm / (sf_norm + 1e-4)

    acc3d_strict = np.sum(np.logical_or((l2_norm < 0.05)*mask, (relative_err < 0.05)*mask), axis=1).astype(np.float)
    acc3d_relax = np.sum(np.logical_or((l2_norm < 0.1)*mask, (relative_err < 0.1)*mask), axis=1).astype(np.float)
    outlier = np.sum(np.logical_or((l2_norm > 0.3)*mask, (relative_err > 0.1)*mask), axis=1).astype(np.float)

    acc3d_strict = acc3d_strict[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc3d_strict = np.mean(acc3d_strict)
    acc3d_relax = acc3d_relax[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc3d_relax = np.mean(acc3d_relax)
    outlier = outlier[mask_sum > 0] / mask_sum[mask_sum > 0]
    outlier = np.mean(outlier)

    return EPE3D, acc3d_strict, acc3d_relax, outlier


def evaluate_2d(flow_pred, flow_gt):
    """
    flow_pred: (N, 2)
    flow_gt: (N, 2)
    """

    epe2d = np.linalg.norm(flow_gt - flow_pred, axis=-1)
    epe2d_mean = epe2d.mean()

    flow_gt_norm = np.linalg.norm(flow_gt, axis=-1)
    relative_err = epe2d / (flow_gt_norm + 1e-5)

    acc2d = (np.logical_or(epe2d < 3., relative_err < 0.05)).astype(np.float).mean()

    return epe2d_mean, acc2d
