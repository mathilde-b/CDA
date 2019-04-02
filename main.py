#!/usr/bin/env python3.6
import argparse
import warnings
from pathlib import Path
from operator import add, itemgetter
from shutil import copytree, rmtree
from typing import Any, Callable, List, Tuple
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from functools import reduce

from networks import weights_init, FCDiscriminator
from dataloader import get_loaders
from utils import map_, dice_coef, save_images, tqdm_
from utils import probs2one_hot, probs2class, resize, haussdorf, save_dict_to_file


from losses import d_loss_calc
from dice3d import dice3d
import datetime


def setup(args, n_class, dtype) -> Tuple[Any, Any, Any, List[Callable], List[float],List[Callable], List[float], Callable]:
    print(">>> Setting up")
    cpu: bool = args.cpu or not torch.cuda.is_available()
    device = torch.device("cpu") if cpu else torch.device("cuda")

    if args.model_weights:
        if cpu:
            net = torch.load(args.model_weights, map_location='cpu')
        else:
            net = torch.load(args.model_weights)
    else:
        net_class = getattr(__import__('networks'), args.network)
        net = net_class(1, n_class, dtype=dtype).type(dtype).to(device)
        net.apply(weights_init)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.l_rate, betas=(0.9, 0.999))

    print(args.losses)
    losses = eval(args.losses)
    loss_fns: List[Callable] = []
    for loss_name, loss_params, _, _, fn, _ in losses:
        loss_class = getattr(__import__('losses'), loss_name)
        loss_fns.append(loss_class(**loss_params, dtype=dtype, fn=fn))

    loss_weights = map_(itemgetter(5), losses)

    print(args.losses_source)
    losses_source = eval(args.losses_source)
    loss_fns_source: List[Callable] = []
    for loss_name, loss_params, _, _, fn, _ in losses_source:
        loss_class = getattr(__import__('losses'), loss_name)
        loss_fns_source.append(loss_class(**loss_params, dtype=dtype, fn=fn))

    loss_weights_source = map_(itemgetter(5), losses_source)

    if args.scheduler:
        scheduler = getattr(__import__('scheduler'), args.scheduler)(**eval(args.scheduler_params))
    else:
        scheduler = ''

    return net, optimizer, device, loss_fns, loss_weights, loss_fns_source, loss_weights_source, scheduler


def for_back_step_comb(optimizer, mode, source_image, target_image, gt_source, labels,
                       net, loss_fns,loss_weights, loss_fns_source, loss_weights_source, new_w, device, bounds,
                        model_D, optimizer_D, lambda_adv_target):
    source_label = 0
    target_label = 1

    # Reset gradients
    if optimizer:
        optimizer.zero_grad()

    optimizer_D.zero_grad()

    # don't accumulate grads in D
    for param in model_D.parameters():
        param.requires_grad = False

    # Forward
    with torch.set_grad_enabled(mode == "train"):
        #Forward
        pred_logits_source: Tensor = net(source_image)
        probs_source: Tensor = F.softmax(pred_logits_source, dim=1)

        pred_logits_target: Tensor = net(target_image)
        probs_target: Tensor = F.softmax(pred_logits_target, dim=1)
        predicted_mask_target: Tensor = probs2one_hot(probs_target)

        if new_w > 0:
            probs_source = resize(probs_source, new_w)
            probs_target = resize(probs_target, new_w)
            if labels[0].shape[3]!= new_w:
                labels = [resize(label, new_w) for label in labels ]
            gt_source = resize(gt_source, new_w)

    assert len(bounds) == len(loss_fns) == len(loss_weights)

    loss_adv_target = torch.zeros(1, requires_grad=True).to(device)

    loss_vec = []

    # Losses on source
    ziped = zip(loss_fns_source, [gt_source], loss_weights_source)
    losses = [w * loss_fn(probs_source, label, torch.randn(1)) for loss_fn, label, w in ziped]
    loss_vec.extend([loss.item() for loss in losses])
    loss_source = reduce(add, losses)

    # add adversarial loss of target im if not a pair of negative images
    if lambda_adv_target > 0 and max(predicted_mask_target[0,1,...].sum(),predicted_mask_target[0,1,...].sum()).item()>0:
        D_out = model_D(probs_target)
        loss_adv_target = d_loss_calc(D_out, source_label).to(device=device)*lambda_adv_target
        loss_vec.append(loss_adv_target.item())
    else:
        loss_vec.append(0)

    # Constraint loss on target images (and eventually also cross entropy with FGT)
    ziped = zip(loss_fns, labels, loss_weights, bounds)
    losses = [w * loss_fn(probs_target, label, bound) for loss_fn, label, w, bound in ziped]
    loss_vec.extend([loss.item() for loss in losses])
    loss_target = reduce(add, losses)

    loss = loss_source + loss_adv_target + loss_target

    # Backward
    if optimizer:
        loss.backward()
        optimizer.step()

    # train D
    if lambda_adv_target > 0 and max(predicted_mask_target[0,1,...].sum(),predicted_mask_target[0,1,...].sum()).item()>0:
        # bring back requires_grad
        for param in model_D.parameters():
            param.requires_grad = True

        # train with source
        probs_source = probs_source.detach()
        D_out = model_D(probs_source)
        loss_D_s = d_loss_calc(D_out, source_label).to(device=device)/2
        if optimizer:
            loss_D_s.backward()

        # train with target
        probs_target = probs_target.detach()
        D_out_t = model_D(probs_target)
        loss_D_t = d_loss_calc(D_out_t, target_label).to(device=device)/2
        if optimizer:
            loss_D_t.backward()
            optimizer_D.step()
        loss_vec.append(loss_D_s.item()+loss_D_t.item())
    else:
        loss_vec.append(0)

    return probs_source, probs_target, loss, loss_vec[0], loss_vec[1], loss_vec[2], loss_vec[3], loss_vec[4]


def compute_metrics(pred_probs, gt, labels):

    predicted_mask: Tensor = probs2one_hot(pred_probs)
    b, c, _,_ = predicted_mask.shape

    dices = dice_coef(predicted_mask.detach(), gt.detach()).cpu().numpy()
    baseline_dices = dice_coef(labels.detach(), gt.detach()).cpu().numpy()
    haussdorf_res = haussdorf(predicted_mask.detach(), gt.detach(), dtype= pred_probs.dtype).cpu().numpy()

    assert haussdorf_res.shape == (b, c)
    posim = torch.einsum("bcwh->b", [gt[:, 1:, :, :]]).detach() > 0
    posim = posim.cpu().numpy()

    return dices, baseline_dices, posim, haussdorf_res


def do_save_images(pred_probs, savedir, filenames, mode, epc):
    if savedir:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            predicted_class: Tensor = probs2class(pred_probs)
            save_images(predicted_class, filenames, savedir, mode, epc, True)


def do_epoch(args, mode: str, net: Any, device: Any, loader: DataLoader, epc: int,
             loss_fns: List[Callable], loss_weights: List[float],loss_fns_source: List[Callable],
             loss_weights_source: List[float], new_w:int, num_steps:int, C: int, savedir: str = "",
             optimizer: Any = None, target_loader: Any = None, lambda_adv_target:float =0.001) -> Tuple[List,List,List,List]:

    assert mode in ["train", "val"]
    L: int = len(loss_fns)

    if mode == "train":
        net.train()
        desc = f">> Training   ({epc})"
    elif mode == "val":
        net.eval()
        desc = f">> Validation ({epc})"

    total_iteration, total_images = len(loader), len(loader.dataset)

    # losses metrics
    loss_seg_log = np.zeros(total_images)
    loss_cons_log = np.zeros(total_images)
    loss_inf_log = np.zeros(total_images)
    loss_adv_log = np.zeros(total_images)
    loss_D_log = np.zeros(total_images)

    # source metrics
    dices_log_s = np.zeros((total_images, C))
    posim_log_s = np.zeros(total_images)
    haussdorf_log_s = np.zeros((total_images, C))

    # target metrics
    dices_log_t = np.zeros((total_images, C))
    dices_baseline_log_t = np.zeros((total_images, C))
    posim_log_t = np.zeros(total_images)
    haussdorf_log_t = np.zeros((total_images, C))

    cudnn.benchmark = True
    model_D = FCDiscriminator(num_classes=C)
    model_D.train()
    model_D.to(device)
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=args.l_rate_D, betas=(0.9, 0.99))
    tq_iter = tqdm_(enumerate(zip(loader, target_loader)), total=total_iteration, desc=desc)
    done: int = 0
    dice_3d_s = 0
    dice_3d_sd_s = 0
    dice_3d_t = 0
    dice_3d_sd_t = 0
    baseline_target_vec = [0,0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j, (source_data, target_data) in tq_iter:
            source_data[1:] = [e.to(device) for e in source_data[1:]]  # Move all tensors to device
            filenames_source, source_image, source_gt = source_data
            target_data[1:] = [e.to(device) for e in target_data[1:]]  # Move all tensors to device
            filenames_target, target_image, target_gt = target_data[:3]
            labels = target_data[3:3+L]
            bounds = target_data[3+L:]
            assert len(labels) == len(bounds)
            B = len(target_image)
            #print("source: %s , target: %s" % (filenames_source, filenames_target))

            source_probs, target_probs, loss, loss_seg, loss_adv, loss_inf, loss_cons, loss_D = for_back_step_comb(optimizer, mode, source_image, target_image, source_gt, labels,
                                                                            net, loss_fns, loss_weights,loss_fns_source, loss_weights_source, new_w, device, bounds,
                                                                             model_D, optimizer_D, lambda_adv_target)

            #compute metrics for current batch

            if new_w > 0:
                source_gt = resize(source_gt, new_w)
                target_gt = resize(target_gt, new_w)
                labels[0] = resize(labels[0], new_w)

            dices_s, _, posim_s, haussdorf_s = compute_metrics(source_probs, source_gt, source_gt)
            dices_t, dices_baseline_t, posim_t, haussdorf_t = compute_metrics(target_probs, target_gt, labels[0])

            do_save_images(target_probs, savedir, filenames_target, mode, epc)
            do_save_images(source_probs, savedir, filenames_source, "_".join(("source", mode)), epc)

            # keep metrics in ndarrays
            sm_slice = slice(done, done + B)
            loss_seg_log[sm_slice] = loss_seg
            loss_cons_log[sm_slice] = loss_cons
            loss_adv_log[sm_slice] = loss_adv
            loss_inf_log[sm_slice] = loss_inf
            loss_D_log[sm_slice] = loss_D

            dices_log_s[sm_slice, ...] = dices_s
            haussdorf_log_s[sm_slice] = haussdorf_s
            posim_log_s[sm_slice] = posim_s

            dices_log_t[sm_slice, ...] = dices_t
            dices_baseline_log_t[sm_slice, ...] = dices_baseline_t
            haussdorf_log_t[sm_slice] = haussdorf_t
            posim_log_t[sm_slice] = posim_t

            done +=B

    # calculate mean of metrics on all images
    loss_seg_log = loss_seg_log.mean()
    loss_adv_log = loss_adv_log.mean()
    loss_cons_log = loss_cons_log.mean()
    loss_inf_log = loss_inf_log.mean()
    loss_D_log = loss_inf_log.mean()

    # first select positive and negative images
    dice_posim_log_s = np.compress(posim_log_s,[dices_log_s[:,1]]).mean()
    dice_negim_log_s = np.compress(1-posim_log_s, [dices_log_s[:,1]]).mean()

    dice_posim_log_t = np.compress(posim_log_t, [dices_log_t[:,1]]).mean()
    dice_negim_log_t = np.compress(1-posim_log_t, [dices_log_t[:,1]]).mean()

    # mean on the source images
    dices_log_s = dices_log_s[:, -1].mean()
    haussdorf_log_s = haussdorf_log_s[:, -1].mean()

    # mean on the target images
    dices_log_t = dices_log_t[:, -1].mean()
    haussdorf_log_t = haussdorf_log_t[:, -1].mean()

    # dice3D gives back the 3d dice mean on images
    if not args.debug:
        dice_3d_s, dice_3d_sd_s = dice3d(args.workdir,   f"iter{epc:03d}", "source_"+mode, "Subj_\\d+_", args.dataset+mode+'/GT')
        dice_3d_t, dice_3d_sd_t = dice3d(args.workdir,   f"iter{epc:03d}", mode, "Subj_\\d+_", args.target_dataset+mode+'/GT')
        if epc == 0:
            dice_3d_baseline, dice_3d_sd_baseline = dice3d(args.target_dataset, mode, 'Wat_on_Inn_n', "Subj_\\d+_",args.target_dataset+mode+'/GT')
            baseline_target_vec = [dice_3d_baseline, dice_3d_sd_baseline]

    stat_dict = {"dice 3D source": dice_3d_s,
                 "dice 3D target": dice_3d_t}
    nice_dict = {k: f"{v:.4f}" for (k, v) in stat_dict.items()}

    print(f"{desc} " + ', '.join(f"{k}={v}" for (k, v) in nice_dict.items()))

    # Keep metrics in vectors
    losses_vec = [loss_seg_log, loss_adv_log,loss_inf_log, loss_cons_log , loss_D_log]
    source_vec = [dices_log_s, dice_posim_log_s, dice_negim_log_s, dice_3d_s, dice_3d_sd_s, haussdorf_log_s]
    target_vec = [dices_log_t, dice_posim_log_t, dice_negim_log_t, dice_3d_t, dice_3d_sd_t, haussdorf_log_t]

    return losses_vec, source_vec, target_vec, baseline_target_vec


def run(args: argparse.Namespace) -> None:

    # save args to dict
    d = vars(args)
    d['time'] = str(datetime.datetime.now())
    save_dict_to_file(d,args.workdir)

    n_class: int = args.n_class
    lr: float = args.l_rate
    dtype = eval(args.dtype)

    # Proper params
    savedir: str = args.workdir
    n_epoch: int = args.n_epoch

    net, optimizer, device, loss_fns, loss_weights, loss_fns_source, loss_weights_source, scheduler = setup(args, n_class, dtype)
    print(f'> Loss weights cons: {loss_weights}, Loss weights source:{loss_weights_source}, Loss weights adv: {args.lambda_adv_target}')
    shuffle = False
    if args.mix:
        shuffle = True
    loader, loader_val = get_loaders(args, args.dataset,args.folders,
                                           args.batch_size, n_class,
                                           args.debug, args.in_memory, dtype, False)

    target_loader, target_loader_val = get_loaders(args, args.target_dataset,args.target_folders,
                                           args.batch_size, n_class,
                                           args.debug, args.in_memory, dtype, shuffle)

    n_tra: int = len(loader.dataset)  # Number of images in dataset
    l_tra: int = len(loader)  # Number of iteration per epoch: different if batch_size > 1
    n_val: int = len(loader_val.dataset)
    l_val: int = len(loader_val)

    num_steps = n_epoch * len(loader)

    best_dice_pos: Tensor = np.zeros(1)
    best_dice: Tensor = np.zeros(1)
    best_3d_dice: Tensor = np.zeros(1)

    print(">>> Starting the training")
    for i in range(n_epoch):
        # Do training and validation loops

        tra_losses_vec, tra_source_vec, tra_target_vec, tra_baseline_target_vec = do_epoch(args, "train", net, device, loader, i, loss_fns, loss_weights, loss_fns_source, loss_weights_source, args.resize,
                                                                            num_steps, n_class, savedir=savedir, optimizer=optimizer, target_loader=target_loader,  lambda_adv_target = args.lambda_adv_target)

        with torch.no_grad():
            val_losses_vec, val_source_vec, val_target_vec, val_baseline_target_vec = do_epoch(args, "val", net, device, loader_val, i, loss_fns, loss_weights,loss_fns_source, loss_weights_source, args.resize,
                                                                            num_steps, n_class, savedir=savedir, target_loader=target_loader_val, lambda_adv_target=args.lambda_adv_target )

        if i == 0:
            keep_tra_baseline_target_vec = tra_baseline_target_vec
            keep_val_baseline_target_vec = val_baseline_target_vec

        df_s_tmp = pd.DataFrame({"tra_dice": tra_source_vec[0],
                                 "tra_dice_pos": tra_source_vec[1],
                                 "tra_dice_neg": tra_source_vec[2],
                                 "tra_dice_3d": tra_source_vec[3],
                                 "tra_dice_3d_sd": tra_source_vec[4],
                                 "tra_haussdorf": tra_source_vec[5],
                                 "tra_loss_seg": tra_losses_vec[0],
                                 "tra_loss_adv": tra_losses_vec[1],
                                 "tra_loss_inf": tra_losses_vec[2],
                                 "tra_loss_cons": tra_losses_vec[3],
                                 "tra_loss_D": tra_losses_vec[4],
                                 "val_dice": val_source_vec[0],
                                 "val_dice_pos": val_source_vec[1],
                                 "val_dice_neg": val_source_vec[2],
                                 "val_dice_3d": val_source_vec[3],
                                 "val_dice_3d_sd": val_source_vec[4],
                                 "val_haussdorf": val_source_vec[5],
                                 "val_loss_seg": val_losses_vec[0]}, index=[i])

        df_t_tmp = pd.DataFrame({
                                "tra_dice": tra_target_vec[0],
                                "tra_dice_pos": tra_target_vec[1],
                                "tra_dice_neg": tra_target_vec[2],
                                "tra_dice_3d": tra_target_vec[3],
                                "tra_dice_3d_sd": tra_target_vec[4],
                                "tra_haussdorf": tra_target_vec[5],
                                "tra_dice_3d_baseline": keep_tra_baseline_target_vec[0],
                                "tra_dice_3d_sd_baseline": keep_tra_baseline_target_vec[1],
                                "val_dice": val_target_vec[0],
                                "val_dice_pos": val_target_vec[1],
                                "val_dice_neg": val_target_vec[2],
                                "val_dice_3d": val_target_vec[3],
                                "val_dice_3d_sd": val_target_vec[4],
                                "val_haussdorf": val_target_vec[5],
                                "val_dice_3d_baseline": keep_val_baseline_target_vec[0],
                                "val_dice_3d_sd_baseline": keep_val_baseline_target_vec[1]}, index=[i])

        if i == 0:
            df_s = df_s_tmp
            df_t = df_t_tmp
        else:
            df_s = df_s.append(df_s_tmp)
            df_t = df_t.append(df_t_tmp)

        df_s.to_csv(Path(savedir, args.csv), float_format="%.4f", index_label="epoch")
        df_t.to_csv(Path(savedir, "_".join(("target", args.csv))), float_format="%.4f", index_label="epoch")

        # Save model if better
        current_val_target_3d_dice = val_target_vec[3]

        if current_val_target_3d_dice > best_3d_dice:
            best_epoch = i
            best_3d_dice = current_val_target_3d_dice
            with open(Path(savedir, "best_epoch_3d.txt"), 'w') as f:
                f.write(str(i))
            best_folder_3d = Path(savedir, "best_epoch_3d")
            if best_folder_3d.exists():
                rmtree(best_folder_3d)
            copytree(Path(savedir, f"iter{i:03d}"), Path(best_folder_3d))
            torch.save(net, Path(savedir, "best_3d.pkl"))

        # remove images from iteration
        rmtree(Path(savedir, f"iter{i:03d}"))

        if args.scheduler:
            optimizer, loss_fns, loss_weights = scheduler(i, optimizer, loss_fns, loss_weights)
            if (i % (best_epoch + 20) == 0) and i > 0 :
                for param_group in optimizer.param_groups:
                    lr *= 0.5
                    param_group['lr'] = lr
                    print(f'> New learning Rate: {lr}')


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--target_dataset', type=str, default='', required=True)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--losses", type=str, required=True,
                        help="List of (loss_name, loss_params, bounds_name, bounds_params, fn, weight)")
    parser.add_argument("--losses_source", type=str, required=True,
                        help="List of (loss_name, loss_params, bounds_name, bounds_params, fn, weight)")
    parser.add_argument("--folders", type=str, required=True,
                        help="List of (subfolder, transform, is_hot)")
    parser.add_argument("--target_folders", type=str, required=True,
                        help="List of (subfolder, transform, is_hot)")
    parser.add_argument("--network", type=str, required=True, help="The network to use")
    parser.add_argument("--grp_regex", type=str, required=True)
    parser.add_argument("--n_class", type=int, required=True)
    parser.add_argument("--csv", type=str, default='metrics.csv')
    parser.add_argument("--model_weights", type=str, default='',
                        help="Eventually load a pretrained model")
    parser.add_argument("--resize", type=int, default=0,
                        help='resize image width')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100,
                        help='# of the epochs')
    parser.add_argument('--l_rate', nargs='?', type=float, default=5e-4,
                        help='Learning Rate Segmentator')
    parser.add_argument("--l_rate_D", type=float, default=1e-4,
                        help='Learning Rate Discriminator')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--dtype", type=str, default="torch.float32")
    parser.add_argument("--scheduler", type=str, default="DummyScheduler")
    parser.add_argument("--scheduler_params", type=str, default="{}")
    parser.add_argument("--lambda_adv_target", type=float, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--in_memory", action='store_true')
    args = parser.parse_args()
    print(args)

    return args


if __name__ == '__main__':

    run(get_args())




