import argparse
from typing import Any, Dict, List, Tuple
from pathlib import Path
from functools import partial
from operator import itemgetter

from PIL import ImageOps
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import einsum, Tensor
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

import datetime
from networks import resnext101,weights_init
from dataloader import SliceDataset, Concat
from utils import class2one_hot, tqdm_, map_, id_, str2bool,save_dict_to_file
import pandas as pd

def main(args: argparse.Namespace) -> None:
    print("\n>>> Setting up")
    d = vars(args)
    d['time'] = str(datetime.datetime.now())
    #save_dict_to_file(d,args.workdir)
    cpu: bool = args.cpu or not torch.cuda.is_available()
    device = torch.device("cpu") if cpu else torch.device("cuda")

    cudnn.benchmark = True

    if args.weights:
        print(f">> Loading weights from {args.weights}")
        net = torch.load(args.weights)
    elif args.pretrained:
        print(">> Starting from pre-trained network")
        net = models.resnet101(pretrained=True)
        print("> Recreating its last FC layer")
        in_, out_ = net.fc.in_features, net.fc.out_features
        print(f"> Going from shape {(in_, out_)} to {(8192, args.n_class)}")
        net.fc = nn.Linear(8192, args.n_class)  # Change only the last layer
    else:
        #print(">> Using a brand new netwerk")
        #net = resnext101(baseWidth=args.base_width, cardinality=args.cardinality, n_class=args.n_class)
        net_class = getattr(__import__('networks'), 'Enet')
        net = net_class(1, args.n_class)
        net.apply(weights_init)
    net.to(device)

    lr: float = args.lr
    criterion = torch.nn.MSELoss(reduction="sum")
    if not args.adam:
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    # Dataloaderz and shitz
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    png_transform = transforms.Compose([
        lambda img: img.convert('L'),
        ImageOps.equalize if args.equalize else id_,
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: nd / 255,  # max <= 1
        lambda nd: torch.tensor(nd, dtype=torch.float32),
        normalize if args.pretrained else id_
    ])
    gt_transform = transforms.Compose([
        lambda img: img.convert('L'),
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: torch.tensor(nd, dtype=torch.int64),
        partial(class2one_hot, C=args.n_class),
        itemgetter(0)
    ])

    gen_dataset = partial(SliceDataset,
                          transforms=[png_transform, gt_transform],
                          are_hots=[False, True],
                          debug=args.debug,
                          C=args.n_class,
                          in_memory=args.in_memory,
                          bounds_generators=[])

    data_loader = partial(DataLoader,
                          num_workers=4,
                          pin_memory=True)

    if not args.GT:
        gt="GT"
    else:
        gt= args.GT
    if not args.val_GT:
        val_gt="GT"
    else:
        val_gt= args.val_GT
    train_filenames: List[str] = map_(lambda p: str(p.name), Path(args.data_root, args.train_subfolder, args.modality).glob("*"))
    train_folders: List[Path] = [Path(args.data_root, args.train_subfolder, f) for f in [args.modality, gt]]

    val_filenames: List[str] = map_(lambda p: str(p.name), Path(args.data_root, args.val_subfolder, args.val_modality).glob("*"))
    val_folders: List[Path] = [Path(args.data_root, args.val_subfolder, f) for f in [args.val_modality, val_gt]]


    train_set: Dataset = gen_dataset(train_filenames, train_folders, augment=args.augment)
    train_set =  Concat([train_set, train_set])
    val_set: Dataset = gen_dataset(val_filenames, val_folders)
    val_set = Concat([val_set, val_set, val_set, val_set, val_set])

    train_loader: DataLoader = data_loader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader: DataLoader = data_loader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print()

    best_perf: float = -1
    best_epc: int = -1

    metrics: Dict[str, Tensor] = {"tra_loss": torch.zeros((args.epc, len(train_loader)),
                                                          dtype=torch.float32, device=device),
                                  "tra_diff": torch.zeros((args.epc, len(train_set), args.n_class),
                                                          dtype=torch.float32, device=device),
                                  "tra_gt_size": torch.zeros((args.epc, len(train_set), args.n_class),
                                                          dtype=torch.float32, device=device),
                                  "tra_pred_size": torch.zeros((args.epc, len(train_set), args.n_class),
                                                          dtype=torch.float32, device=device),
                                  "val_loss": torch.zeros((args.epc, len(val_loader)),
                                                          dtype=torch.float32, device=device),
                                  "val_diff": torch.zeros((args.epc, len(val_set), args.n_class),
                                                          dtype=torch.float32, device=device),
                                  "val_gt_size": torch.zeros((args.epc, len(val_set), args.n_class),
                                                          dtype=torch.float32, device=device),
                                  "val_pred_size": torch.zeros((args.epc, len(val_set), args.n_class),
                                                          dtype=torch.float32, device=device)}
    for i in range(args.epc):
        sizes: Tensor
        predicted_sizes: Tensor
        loss: Tensor

        if not args.no_training:
            net, train_metrics,train_ids = do_epc(i, "train", net, train_loader, device, criterion, args, optimizer)
            for k in train_metrics:
                metrics["tra_" + k][i] = train_metrics[k][...]

        with torch.no_grad():
            net, val_metrics,val_ids = do_epc(i, "val", net, val_loader, device, criterion, args)
            for k in val_metrics:
                metrics["val_" + k][i] = val_metrics[k][...]

        diff = metrics["val_diff"][i,..., args.idc]
        gt_size = metrics["val_gt_size"][i,..., args.idc]

        savepath = Path(args.save_dest)
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        savepath.mkdir(parents=True, exist_ok=True)
        if i==0:
            save_dict_to_file(d,savepath)
        #epc_perf = float(metrics["val_diff"][i, metrics["val_gt_size"][i,...,args.idc]!=0, args.idc].mean())
        #epc_perf: float = float(metrics["val_diff"][i, ..., args.idc].mean())
        epc_perf = float(diff[gt_size!=0].mean())
        if epc_perf < best_perf or i == 0:
            best_perf = epc_perf
            best_epc = i
            d = pd.DataFrame(0, index=np.arange(len(val_set)), columns=["val_ids","val_diff",
                                                                      "val_gt_size","val_pred_size"])
            d['val_ids'] = val_ids
            d['val_diff'] = metrics["val_diff"].cpu().tolist()[i]
            d['val_gt_size'] = metrics["val_gt_size"].cpu().tolist()[i]
            d['val_pred_size'] = metrics["val_pred_size"].cpu().tolist()[i]
            d.to_csv(Path(args.save_dest, args.val_subfolder+str(i)+'reg_metrics.csv'), float_format="%.4f")

            print(f"> Best results at epoch {best_epc}: diff: {best_perf:12.2f}")
            print(f"> Saving network weights to {args.save_dest}")
            Path(args.save_dest,'pred_size.pkl').parent.mkdir(parents=True, exist_ok=True)
            torch.save(net, str(Path(args.save_dest,'pred_size.pkl')))

        if not i% 10:
            torch.save(net, str(Path(args.save_dest,'pred_size'+str(i)+'.pkl')))

        if i in [args.epc // 2, 3 * args.epc // 4]:
            for param_group in optimizer.param_groups:
                lr *= 0.5
                param_group['lr'] = lr
                print(f'> New learning Rate: {lr}')


def do_epc(epc: int, mode: str, net: Any, loader: DataLoader, device, criterion, args,
           optimizer: Any = None) -> Tuple[Any, Dict[str, Tensor]]:
    assert mode in ["train", "val"]

    desc: str
    if mode == "train":
        net.train()
        desc = f">> Training   ({epc})"
    elif mode == "val":
        net.eval()
        desc = f">> Validation ({epc})"

    total_iteration: int = len(loader)  # U
    total_images: int = len(loader.dataset)  # D

    ids = ['id'] * total_images
    metrics = {"loss": torch.zeros((total_iteration), dtype=torch.float32, device=device),
               "diff": torch.zeros((total_images, args.n_class), dtype=torch.float32, device=device),
               "gt_size": torch.zeros((total_images, args.n_class), dtype=torch.float32, device=device),
               "pred_size": torch.zeros((total_images, args.n_class), dtype=torch.float32, device=device)}

    tq_iter = tqdm_(total=total_iteration, desc=desc)
    done_img: int = 0
    for j, data in enumerate(loader):
        data[1:] = [e.to(device) for e in data[1:]]  # Move all tensors to device
        # filenames, images, targets = data[:3]
        filenames, images, targets = data
        assert len(filenames) == len(images) == len(targets)
        B: int = len(images)

        sizes = einsum("bcwh->bc", targets).type(torch.float32)

        if optimizer:
            optimizer.zero_grad()

        if (args.pretrained or args.weights) and not args.onechan:
            b, c, w, h = images.shape
            assert c == 1
            viewed = images.view((b, w, h))
            new_img = torch.stack([viewed, viewed, viewed], dim=1)
            assert new_img.shape == (b, 3, w, h), new_img.shape
            images = new_img

        predicted_sizes = net(images)
        assert sizes.shape == predicted_sizes.shape

        loss = criterion(predicted_sizes[:, args.idc], sizes[:, args.idc])

        if optimizer:
            loss.backward()
            optimizer.step()

        ids[done_img:done_img + B] = filenames
        metrics["loss"][j] = loss.detach().item()
        metrics["diff"][done_img:done_img + B, ...] = torch.abs(predicted_sizes.detach() - sizes.detach())[...]
        metrics["gt_size"][done_img:done_img + B, ...] = torch.abs(sizes.detach())[...]
        metrics["pred_size"][done_img:done_img + B, ...] = torch.abs(predicted_sizes.detach())[...]
        
        diff = metrics["diff"][:done_img + B, args.idc]
        gt_size = metrics["gt_size"][:done_img + B, args.idc]

        stat_dict: Dict = {"loss": metrics["loss"][:j].mean(),
                           "diff": diff.mean(),
                           "diff_pos": diff[gt_size!=0].mean()}
        nice_dict: Dict = {k: f"{v:12.2f}" for (k, v) in stat_dict.items()}

        done_img += B
        tq_iter.set_postfix(nice_dict)
        tq_iter.update(1)
    tq_iter.close()
    print(f"{desc} " + ', '.join(f"{k}={v}" for (k, v) in nice_dict.items()))

    return net, metrics, ids


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyperparams')

    parser.add_argument("--n_class", type=int, required=True)
    parser.add_argument("--save_dest", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epc", type=int, required=True)
    parser.add_argument("--train_subfolder", type=str, required=True)
    parser.add_argument("--val_subfolder", type=str, required=True)
    parser.add_argument("--idc", type=int, nargs='+')

    parser.add_argument("--modality", type=str, required=True)
    parser.add_argument("--val_modality", type=str, required=True)
    parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
    parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
    parser.add_argument('--lr', '--learning-rate', default=0.0000005, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--augment', type=str2bool, default=False)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument("--equalize", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--onechan", action="store_true")
    parser.add_argument("--GT", type=str)
    parser.add_argument("--val_GT", type=str)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--adam", action="store_true")
    parser.add_argument("--no_training", action="store_true", help="Trick to rerun evaluation a trained network.")
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--in_memory", action='store_true')

    args = parser.parse_args()
    print("\n", args)

    return args


if __name__ == "__main__":
    main(get_args())
