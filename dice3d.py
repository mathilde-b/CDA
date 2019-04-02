#!/usr/env/bin python3.6

import re
from pathlib import Path
from typing import Any, Callable, BinaryIO, Dict, List, Match, Pattern, Tuple, Union
import torch
import numpy as np
from utils import id_, map_, class2one_hot, resize_im
from utils import simplex, sset, one_hot, dice_batch
from argparse import Namespace, ArgumentParser
import pandas as pd
import imageio


def run_dices(args: Namespace) -> None:

    for folder in args.folders:
        subfolders = args.subfolders
        all_dices=[0] * len(subfolders)
        for i, subfolder in enumerate(subfolders):
            print(subfolder)
            epc = int(subfolder.split('r')[1])
            dice_i = dice3d(args.base_folder, folder, subfolder, args.grp_regex, args.gt_folder)
            all_dices[epc] = dice_i

        df = pd.DataFrame({"3d_dice": all_dices})
        df.to_csv(Path(args.save_folder, 'dice_3d.csv'), float_format="%.4f", index_label="epoch")


def dice3d(base_folder, folder, subfoldername, grp_regex, gt_folder) -> List:
    if base_folder == '':
        work_folder = Path(folder, subfoldername)
    else:
        work_folder = Path(base_folder,folder, subfoldername)
    #print(work_folder)
    filenames = map_(lambda p: str(p.name), work_folder.glob("*.png"))
    grouping_regex: Pattern = re.compile(grp_regex)

    stems: List[str] = [Path(filename).stem for filename in filenames]  # avoid matching the extension
    matches: List[Match] = map_(grouping_regex.match, stems)
    patients: List[str] = [match.group(0) for match in matches]

    unique_patients: List[str] = list(set(patients))
    #print(unique_patients)
    batch_dice = []
    for i, patient in enumerate(unique_patients):
        patient_slices = [f for f in stems if f.startswith(patient)]
        w,h = [256,36]
        n = len(patient_slices)
        t_seg = np.ndarray(shape=(w, h, n))
        t_gt = np.ndarray(shape=(w, h, n))
        for slice in patient_slices:
            slice_nb = int(re.split(grp_regex, slice)[1])
            seg = imageio.imread(str(work_folder)+'/'+slice+'.png')
            gt = imageio.imread(str(gt_folder )+'/'+ slice+'.png')
            if seg.shape != (w, h):
                seg = resize_im(seg, 36)
            if gt.shape != (w, h):
                gt = resize_im(gt, 36)
            seg[seg == 255] = 1
            t_seg[:, :, slice_nb] = seg
            t_gt[:, :, slice_nb] = gt
        t_seg = torch.from_numpy(t_seg)
        t_gt = torch.from_numpy(t_gt)
        batch_dice.append(dice_batch(class2one_hot(t_seg,2), class2one_hot(t_gt,2))[-1].item())
    return np.mean(batch_dice)


def get_args() -> Namespace:
    parser = ArgumentParser(description='Hyperparams')
    parser.add_argument('--base_folder', type=str, required=True)
    parser.add_argument('--folders', type=str, default='')
    parser.add_argument('--gt_folder', type=str, default='')
    parser.add_argument('--save_folder', type=str, default='')
    parser.add_argument('--grp_regex', type=str, default='')
    parser.add_argument("--subfolders", type=str, required=True)
    parser.add_argument("--workdir", type=str, required=True)
    args = parser.parse_args()
    print(args)

    return args


if __name__ == "__main__":
    run_dices(get_args())





