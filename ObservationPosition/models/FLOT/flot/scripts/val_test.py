import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import argparse
import numpy as np
from tqdm import tqdm
from flot.models.scene_flow import FLOT
from torch.utils.data import DataLoader
from flot.datasets.generic import Batch


def compute_epe(est_flow, batch, mask=None):
    """
    Compute EPE, accuracy and number of outliers.

    Parameters
    ----------
    est_flow : torch.Tensor
        Estimated flow.
    batch : flot.datasets.generic.Batch
        Contains ground truth flow and mask.

    Returns
    -------
    EPE3D : float
        End point error.
    acc3d_strict : float
        Strict accuracy.
    acc3d_relax : float
        Relax accuracy.
    outlier : float
        Percentage of outliers.

    """

    # Extract occlusion mask
    # mask = batch["ground_truth"][0].cpu().numpy()[..., 0]
    if mask is None:
        mask = np.ones([batch["ground_truth"][0].cpu().numpy().shape[0], batch["ground_truth"][0].cpu().numpy().shape[1]])
    
    
    # Flow
    sf_gt = batch["ground_truth"][1].cpu().numpy() #* mask
    sf_pred = est_flow.cpu().numpy() #[mask > 0]
    #
    l2_norm = np.linalg.norm(sf_gt - sf_pred, axis=-1)
    sf_norm = np.linalg.norm(sf_gt, axis=-1)
    if mask is not None:
      l2_norm *= mask
      sf_norm *= mask
    
    mask_sum = np.sum(mask, 1)
    EPE3D = np.sum(l2_norm, 1) / (mask_sum + 1e-4)
    EPE3D = np.mean(EPE3D)

    #
    # mask_sum = np.sum(mask, 1)
    # sf_norm = np.linalg.norm(sf_gt, axis=-1)
    relative_err = l2_norm / (sf_norm + 1e-4)
    acc3d_strict = np.sum(
        (np.logical_or((l2_norm <= 0.05) * mask, (l2_norm <= 0.05) * mask)).astype(np.float), axis=1
    )
    # acc050 = np.mean(acc050)
    acc3d_relax = np.sum(
        (np.logical_or((l2_norm <= 0.1) * mask, (l2_norm <= 0.1) * mask)).astype(np.float), axis=1
    )
    # acc050 = np.mean(acc050)
    outlier = np.sum((np.logical_or((l2_norm > 0.1) * mask, (l2_norm > 0.1) * mask)).astype(np.float), axis=1)

    acc3d_strict = acc3d_strict[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc3d_strict = np.mean(acc3d_strict)
    acc3d_relax = acc3d_relax[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc3d_relax = np.mean(acc3d_relax)
    outlier = outlier[mask_sum > 0] / mask_sum[mask_sum > 0]
    outlier = np.mean(outlier)
    return EPE3D, acc3d_strict, acc3d_relax, outlier

def error(pred, labels, mask=None):
    pred = pred.permute(0, 2, 1).cpu().numpy()
    labels = labels.permute(0, 2, 1).cpu().numpy()
    if mask is None:
        mask = np.ones([pred.shape[0], pred.shape[1]])
    else:
        mask = mask.cpu().numpy()

    err = np.sqrt(np.sum((pred - labels) ** 2, 2) + 1e-20)

    gtflow_len = np.sqrt(np.sum(labels * labels, 2) + 1e-20)  # B,N
    acc050 = np.sum(np.logical_or((err <= 0.05) * mask, (err / gtflow_len <= 0.05) * mask), axis=1)
    acc010 = np.sum(np.logical_or((err <= 0.1) * mask, (err / gtflow_len <= 0.1) * mask), axis=1)
    # outlier = np.sum(np.logical_or((err > 0.3) * mask, (err / gtflow_len > 0.1) * mask), axis=1)

    mask_sum = np.sum(mask, 1)
    acc050 = acc050[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc050 = np.mean(acc050)
    acc010 = acc010[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc010 = np.mean(acc010)
    if acc050 >= 0.999:
        outlier = np.sum(np.logical_and((err > 0.3) * mask, mask), axis=1)
    else:
        # outlier = np.sum(np.logical_or((err > 0.3)*mask, (err/gtflow_len > 0.1)*mask), axis=1)
        outlier = np.sum(np.logical_or((err > 0.3) * mask, (err / gtflow_len > 0.1) * mask * acc050), axis=1)
    outlier = outlier[mask_sum > 0] / mask_sum[mask_sum > 0]
    outlier = np.mean(outlier)

    epe = np.sum(err * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    epe = np.mean(epe)
    return epe, acc050, acc010, outlier


def eval_model(scene_flow, testloader):
    """
    Compute performance metrics on test / validation set.

    Parameters
    ----------
    scene_flow : flot.models.FLOT
        FLOT model to evaluate.
    testloader : flot.datasets.generic.SceneFlowDataset
        Dataset  loader.
    no_refine : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    mean_epe : float
        Average EPE on dataset.
    mean_outlier : float
        Average percentage of outliers.
    mean_acc3d_relax : float
        Average relaxed accuracy.
    mean_acc3d_strict : TYPE
        Average strict accuracy.

    """

    # Init.
    running_epe = 0
    running_outlier = 0
    running_acc3d_relax = 0
    running_acc3d_strict = 0

    #
    use_savefile = False
    scene_flow = scene_flow.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for it, batch in enumerate(tqdm(testloader)):

        # Send data to GPU
        batch = batch.to(device, non_blocking=True)
        # ego_flow = batch['ground_truth'][0]
        flow = batch['ground_truth'][1]
        pcs = [batch["sequence"][0], batch["sequence"][1]]

        # Estimate flow
        with torch.no_grad():
            est_flow = scene_flow(pcs)
            # est_flow = est_flow * (1-bg_flag) + ego_flow * bg_flag 
        
        if use_savefile:
            name_fmt = "{:0>6}".format(str(it)) + '.npz'
            np_src = pcs[0].squeeze(0).cpu().numpy()
            np_tgt = pcs[1].squeeze(0).cpu().numpy()
            gt = flow.squeeze(0).cpu().numpy()
            np_flow = est_flow.cpu().detach().squeeze(0).numpy()
            np.savez('./results/' + name_fmt, pos1=np_src, pos2=np_tgt, flow=np_flow, gt=gt)
            

        # Perf. metrics
        EPE3D, acc3d_strict, acc3d_relax, outlier = compute_epe(est_flow, batch)
        running_epe += EPE3D
        running_outlier += outlier
        running_acc3d_relax += acc3d_relax
        running_acc3d_strict += acc3d_strict

    #
    mean_epe = running_epe / (it + 1)
    mean_outlier = running_outlier / (it + 1)
    mean_acc3d_relax = running_acc3d_relax / (it + 1)
    mean_acc3d_strict = running_acc3d_strict / (it + 1)

    # print(
    #     "EPE;{0:e};Outlier;{1:e};ACC3DR;{2:e};ACC3DS;{3:e};Size;{4:d}".format(
    #         mean_epe,
    #         mean_outlier,
    #         mean_acc3d_relax,
    #         mean_acc3d_strict,
    #         len(testloader),
    #     )
    # )
    print(
        "EPE; %.5f;ACC3DS; %.5f;ACC3DR; %.5f;Outlier; %.5f;Size;%d"%(
            mean_epe,
            mean_acc3d_strict,
            mean_acc3d_relax,
            mean_outlier,
            len(testloader),
        )
    )
    # print("Total time consuming is %.4f." % (scene_flow.total_time / 2003.0))
    return mean_epe, mean_outlier, mean_acc3d_relax, mean_acc3d_strict


def my_main(dataset_name, max_points, path2ckpt, test=False):
    """
    Entry point of the script.

    Parameters
    ----------
    dataset_name : str
        Dataset on which to evaluate. Either HPLFlowNet_kitti or HPLFlowNet_FT3D
        or flownet3d_kitti or flownet3d_FT3D.
    max_points : int
        Number of points in point clouds.
    path2ckpt : str
        Path to saved model.
    test : bool, optional
        Whether to use test set of validation. Has only an effect for FT3D.
        The default is False.

    Raises
    ------
    ValueError
        Unknown dataset.

    """

    # Path to current file
    pathroot = os.path.dirname(__file__)
    random_data = True

    # Select dataset
    if dataset_name.split("_")[0].lower() == "HPLFlowNet".lower():

        # HPLFlowNet version of the datasets
        path2data = os.path.join(pathroot, "..", "data", "HPLFlowNet")

        # KITTI
        if dataset_name.split("_")[1].lower() == "kitti".lower():
            mode = "test"
            # path2data = os.path.join(path2data, "KITTI_processed_occ_final")
            
            from flot.datasets.kitti_hplflownet import Kitti

            dataset = Kitti(root_dir=path2data, nb_points=max_points)

        # FlyingThing3D
        elif dataset_name.split("_")[1].lower() == "ft3d".lower():
            # path2data = os.path.join(path2data, "FlyingThings3D_subset_processed_35m")
            
            path2data = '/public_dataset_nas/flownet3d/FlyingThings3D_subset_processed_35m'
            from flot.datasets.flyingthings3d_hplflownet import FT3D

            mode = "test" if test else "val"
            assert mode == "val" or mode == "test", "Problem with mode " + mode
            dataset = FT3D(root_dir=path2data, nb_points=max_points, mode=mode)
        
        else:
            raise ValueError("Unknown dataset " + dataset_name)

    elif dataset_name.split("_")[0].lower() == "flownet3d".lower():

        # FlowNet3D version of the datasets
        path2data = "/home2/wangsj/Dataset/"

        # KITTI
        if dataset_name.split("_")[1].lower() == "kitti".lower():
            mode = "test"
            path2data = os.path.join(path2data, "kitti_rm_ground")
            from flot.datasets.kitti_flownet3d import Kitti

            dataset = Kitti(root_dir=path2data, nb_points=max_points)

        # FlyingThing3D
        elif dataset_name.split("_")[1].lower() == "ft3d".lower():
            # path2data = os.path.join(path2data, "data_processed_maxcut_35_20k_2k_8192")
            path2data = '/public_dataset_nas/flownet3d/data_processed_maxcut_35_20k_2k_8192/data_processed_maxcut_35_20k_2k_8192'
            from flot.datasets.flyingthings3d_flownet3d import FT3D

            mode = "test" if test else "val"
            assert mode == "val" or mode == "test", "Problem with mode " + mode
            dataset = FT3D(root_dir=path2data, nb_points=max_points, mode=mode)

        else:
            raise ValueError("Unknown dataset" + dataset_name)
    elif dataset_name.lower() == 'Carla3D'.lower():
        # npoints = 40000
        from flot.datasets.carla import CARLA3D
        # path2data = '/public/public_dataset_nas/public_workspace/ActiveSceneFlow/record2022_0315_1209/rm_egoV/SF/active5/'
        path2data = '/dataset/public_dataset_nas/public_workspace/ActiveSceneFlow/record2022_0315_1209_0316/rm_egoV/SF/'
        # path2data = '/dataset/public_dataset_nas/public_workspace/ActiveSceneFlow/record2022_0315_1209/rm_egoV_fg/SF/val/'
        # path2data = '/dataset/public_dataset_nas/public_workspace/ActiveSceneFlow/record2022_0315_1209_active/1_5m_20000/'
        path2data = '/dataset/public_dataset_nas/public_workspace/ActiveSceneFlow/record2022_0315_1209_passive/val/'
        # path2data = '/dataset/public_dataset_nas/public_workspace/ActiveSceneFlow/record2022_0315_1209_passive/SF/'
        # path2data = '/home2/public_workspace/carla_scene_flow' #TITS-Dataset
        # path2data = '/public/public_dataset_nas/public_workspace/ActiveSceneFlow/Passive_1209_1838/SF/'
        # path2data = '/public/public_dataset_nas/public_workspace/ActiveSceneFlow/record2022_0314_0305_1614/rm_egoV/SF/'
        # path2data = '/public/public_dataset_nas/public_workspace/ActiveSceneFlow/Active_1531_1519_1451/rm_egoV/SF/'
        lr_lambda = lambda epoch: 1.0 if epoch < 340 else 0.1
        random_data = True
        use_fg_inds = True
        if random_data:
            dataset = CARLA3D(root_dir=path2data, nb_points=max_points, mode="test", use_fg_inds=use_fg_inds)
        else:
            dataset = CARLA3D(root_dir=path2data+'/val/', nb_points=max_points, mode="test", use_fg_inds=use_fg_inds)
    else:
        raise ValueError("Unknown dataset " + dataset_name)


    # Dataloader
    testloader = DataLoader(
        dataset,
        batch_size=10,
        pin_memory=True,
        shuffle=True,
        num_workers=2,
        collate_fn=Batch,
        drop_last=False, 
    )

    # Load FLOT model
    scene_flow = FLOT(nb_iter=3)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # scene_flow = scene_flow.to(device, non_blocking=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_ids = [0]
    # device_ids = [0]
    
    # if len(device_ids) > 1:
    #     scene_flow = nn.DataParallel(scene_flow, device_ids=device_ids)
    # else:
    #     scene_flow = scene_flow.to(device, non_blocking=True)
    if len(device_ids) == 1:
        scene_flow = scene_flow.to(device, non_blocking=True)
        scene_flow = nn.DataParallel(scene_flow, device_ids=device_ids)
    else:
        scene_flow = nn.DataParallel(scene_flow)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    file = torch.load(path2ckpt)
    scene_flow.module.nb_iter = 3 #file["nb_iter"]
    # net_dict = scene_flow.state_dict()
    # for k in file["model"]:
    #     # print(k)
    #     if 'module' in k:
    #         name = k[7:]  # remove `module.`
    #     else:
    #         name = 'module.' + k[:]  # add `module.`
    #     # print(name)
    #     # file["model"][name] = v
    #     net_dict[name] = file["model"][k]
    # scene_flow.load_state_dict(net_dict, strict=True)
    scene_flow.load_state_dict(file["model"], strict=True)
    
    # net_dict = scene_flow.state_dict()
    # pretrained_dict = file["model"]
  
    scene_flow = scene_flow.eval()

    # Evaluation
    epsilon = 0.03 + torch.exp(scene_flow.module.epsilon).item()
    gamma = torch.exp(scene_flow.module.gamma).item()
    power = gamma / (gamma + epsilon)
    print("Epsilon;{0:e};Power;{1:e}".format(epsilon, power))
    eval_model(scene_flow, testloader)


if __name__ == "__main__":

    # Args
    parser = argparse.ArgumentParser(description="Test FLOT.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="flownet3d_kitti",
        help="Dataset. Either HPLFlowNet_kitti or "
        + "HPLFlowNet_FT3D or flownet3d_kitti or flownet3d_FT3D.",
    )
    parser.add_argument(
        "--test", action="store_true", default=False, help="Test or validation datasets"
    )
    parser.add_argument(
        "--nb_points",
        type=int,
        # default=2048,
        default=10000,
        help="Maximum number of points in point cloud.",
    )
    parser.add_argument(
        "--path2ckpt",
        type=str,
        # default="../experiments/logs_carla3d/22_12_17-01_14_53_464803__Iter_3__Pts_2048/model.tar",
        default="../pretrained_models/model_calar_iter3_200_10000",
        help="Path to saved checkpoint.",
    )
    args = parser.parse_args()

    # Launch training
    my_main(args.dataset, args.nb_points, args.path2ckpt, args.test)
