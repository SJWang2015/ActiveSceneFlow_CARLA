import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from datetime import datetime
# from flot.datasets.generic2 import Batch
from flot.datasets.carla_dataset_ITS import CARLA3D,Batch
from flot.models.scene_flow import FLOT
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
global blue
blue = lambda x: '\033[1;32m' + x + '\033[0m'
global red
red = lambda x: '\033[1;35m' + x + '\033[0m'

def compute_epe(est_flow, batch):
    """
    Compute EPE during training.

    Parameters
    ----------
    est_flow : torch.Tensor
        Estimated flow.
    batch : flot.datasets.generic.Batch
        Contains ground truth flow and mask.

    Returns
    -------
    epe : torch.Tensor
        Mean EPE for current batch.

    """

    # mask = batch["ground_truth"][0][..., 0]
    true_flow = batch["ground_truth"][1]
    error = est_flow - true_flow
    # error = error[mask > 0]
    epe_per_point = torch.sqrt(torch.sum(torch.pow(error, 2.0), -1))
    epe = epe_per_point.mean()

    return epe


def compute_loss(est_flow, batch):
    """
    Compute training loss.

    Parameters
    ----------
    est_flow : torch.Tensor
        Estimated flow.
    batch : flot.datasets.generic.Batch
        Contains ground truth flow and mask.

    Returns
    -------
    loss : torch.Tensor
        Training loss for current batch.

    """

    # mask = batch["ground_truth"][0][..., 0]
    true_flow = batch["ground_truth"][1]
    error = est_flow - true_flow
    # error = error[mask > 0]
    loss = torch.mean(torch.abs(error))

    return loss


def train(scene_flow, trainloader, delta, optimizer, scheduler, path2log, nb_epochs):
    """
    Train scene flow model.

    Parameters
    ----------
    scene_flow : flot.models.FLOT
        FLOT model
    trainloader : flots.datasets.generic.SceneFlowDataset
        Dataset loader.
    delta : int
        Frequency of logs in number of iterations.
    optimizer : torch.optim.Optimizer
        Optimiser.
    scheduler :
        Scheduler.
    path2log : str
        Where to save logs / model.
    nb_epochs : int
        Number of epochs.

    """

    # Log directory
    if not os.path.exists(path2log):
        os.makedirs(path2log)
    writer = SummaryWriter(path2log)

    # Reload state
    total_it = 0
    epoch_start = 0

    # # # Train
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device_ids = [0,1]
    # scene_flow = scene_flow.to(device, non_blocking=True)
    # if len(device_ids) > 1:
    #     scene_flow = nn.DataParallel(scene_flow, device_ids=device_ids)
    # else:
    #     scene_flow = nn.DataParallel(scene_flow)
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    
    for epoch in range(epoch_start, nb_epochs):

        # Init.
        running_epe = 0
        running_loss = 0

        # Train for 1 epoch
        start = time.time()
        scene_flow = scene_flow.train()
        for it, batch in enumerate(tqdm(trainloader)):
            # Send data to GPU
            batch = batch.to(device, non_blocking=True)

            # ego_flow = batch['ground_truth'][0]
            # flow = batch['ground_truth'][1]
            # bg_mask = torch.norm(flow - ego_flow, dim=-1).cpu().numpy()
            # bg_flag = np.ones([flow.shape[0], flow.shape[1], 3])
            # for i in range(flow.shape[0]):
            #     bg_flag[i, bg_mask[i,:] > 1e-3, :] = 0
            # bg_flag = torch.tensor(bg_flag).cuda()
            # pred_pc1 = batch["sequence"][0] + ego_flow
            # pcs = [pred_pc1, batch["sequence"][1]]
            # Gradient step
            optimizer.zero_grad()
            est_flow = scene_flow(batch["sequence"])
            # est_flow = est_flow * (1-bg_flag) + ego_flow * bg_flag 
            loss = compute_loss(est_flow, batch)
            loss.backward()
            optimizer.step()

            # Loss evolution
            running_loss += loss.item()
            running_epe += compute_epe(est_flow, batch).item()

            # Logs
            if it % delta == delta - 1:
                # Print / save logs
                writer.add_scalar("Loss/epe", running_epe / delta, total_it)
                writer.add_scalar("Loss/loss", running_loss / delta, total_it)
                print(
                    "Epoch {0:d} - It. {1:d}: loss = {2:e}".format(
                        epoch, total_it, running_loss / delta
                    )
                )
                print(time.time() - start, "seconds")
                # Re-init.
                running_epe = 0
                running_loss = 0
                start = time.time()

            total_it += 1
            # pbar.set_postfix({blue('Loss'): '{0:1.5f}'.format(running_loss * 1.0 / total_it)})  # 输入一个字典，显示实验指标
            # pbar.update(1)

        # Scheduler
        scheduler.step()

        # Save model after each epoch
        state = {
            # "nb_iter": scene_flow.nb_iter,
            "model": scene_flow.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(state, os.path.join(path2log, "model.tar"))

    #
    print("Finished Training")

    return None


def my_main(dataset_name, nb_iter, batch_size, max_points, nb_epochs):
    """
    Entry point of the script.

    Parameters
    ----------
    dataset_name : str
        Version of FlyingThing3D used for training: 'HPLFlowNet' / 'flownet3d'.
    nb_iter : int
        Number of unrolled iteration of Sinkhorn algorithm in FLOT.
    batch_size : int
        Batch size.
    max_points : int
        Number of points in point clouds.
    nb_epochs : int
        Number of epochs.

    Raises
    ------
    ValueError
        If dataset_name is an unknow dataset.

    """

    # Path to current file
    pathroot = os.path.dirname(__file__)

    # Path to dataset
    if dataset_name.lower() == "HPLFlowNet".lower():
        path2data = os.path.join(
            pathroot, "..", "data", "HPLFlowNet", "FlyingThings3D_subset_processed_35m"
        )
        from flot.datasets.flyingthings3d_hplflownet import FT3D

        lr_lambda = lambda epoch: 1.0 if epoch < 50 else 0.1
    elif dataset_name.lower() == "flownet3d".lower():
        path2data = os.path.join(
            pathroot, "..", "data", "flownet3d", "data_processed_maxcut_35_20k_2k_8192"
        )
        from flot.datasets.flyingthings3d_flownet3d import FT3D

        lr_lambda = lambda epoch: 1.0 if epoch < 340 else 0.1
    elif dataset_name.lower() == 'Carla3D'.lower():
        # npoints = 40000
        from flot.datasets.carla_dataset_ITS import CARLA3D
        # path2data = '/public/public_dataset_nas/public_workspace/ActiveSceneFlow/record2022_0315_1838/rm_egoV/SF/'
        path2data = '/dataset/public_dataset_nas/carla_scene_flow2/'
        lr_lambda = lambda epoch: 1.0 if epoch < 150 else 0.1
    else:
        raise ValueError("Invalid dataset name: " + dataset_name)

    # Training dataset
    if dataset_name.lower() == 'Carla3D'.lower():
        dataset = CARLA3D(root_dir=path2data+'/train/', nb_points=max_points, mode="train",use_hybrid_sample=True)
        # test_dataset = CARLA3D(root_dir=path2data+'/val/', nb_points=max_points, mode="test")
    else:
        dataset = FT3D(root_dir=path2data, nb_points=max_points, mode="train")
    trainloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=9,
        collate_fn=Batch,
        drop_last=True,
    )

    # testloader = DataLoader(
    #     test_dataset,
    #     batch_size=1,
    #     pin_memory=True,
    #     shuffle=True,
    #     num_workers=6,
    #     collate_fn=Batch,
    #     drop_last=False,
    # )

    # Model
    scene_flow = FLOT(nb_iter=nb_iter)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # scene_flow = scene_flow.to(device, non_blocking=True)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_ids = [0,1,2]
    scene_flow = scene_flow.to(device, non_blocking=True)
    if len(device_ids) == 1:
        scene_flow = nn.DataParallel(scene_flow, device_ids=device_ids)
    else:
        scene_flow = nn.DataParallel(scene_flow)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    # file = torch.load("/home2/wangsj/FLOT/flot/pretrained_models/model_carla_1209.tar")
    scene_flow.module.nb_iter = 3 #file["nb_iter"]
    # scene_flow.load_state_dict(file["model"])

    # Optimizer
    optimizer = torch.optim.Adam(scene_flow.parameters(), lr=1e-3)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Log directory
    now = datetime.now().strftime("%y_%m_%d-%H_%M_%S_%f")
    now += "__Iter_" + str(scene_flow.module.nb_iter)
    now += "__Pts_" + str(max_points)
    path2log = os.path.join(pathroot, "..", "experiments", "logs_" + dataset_name, now)

    # Train
    print("Training started. Logs in " + path2log)
    train(scene_flow, trainloader, 200, optimizer, scheduler, path2log, nb_epochs)

    return None


if __name__ == "__main__":

    # Args
    parser = argparse.ArgumentParser(description="Train FLOT.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="HPLFlowNet",
        help="Training dataset. Either HPLFlowNet or " + "flownet3d.",
    )
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size.")
    parser.add_argument("--nb_epochs", type=int, default=400, help="Number of epochs.")
    parser.add_argument(
        "--nb_points",
        type=int,
        default=8192,
        help="Maximum number of points in point cloud.",
    )
    parser.add_argument(
        "--nb_iter",
        type=int,
        default=3,
        help="Number of unrolled iterations of the Sinkhorn " + "algorithm.",
    )
    args = parser.parse_args()

    # Launch training
    my_main(args.dataset, args.nb_iter, args.batch_size, args.nb_points, args.nb_epochs)
