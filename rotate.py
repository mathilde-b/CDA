#!/usr/env/bin python3.6

import argparse
import re
from pathlib import Path
from typing import List, Match, Pattern
import numpy as np
from utils import map_, resize_im
from argparse import Namespace
import os
import shutil
import imageio


def main(args: Namespace) -> None:

    if args.folders:
        folders = [args.folders]
    else:
        folders = next(os.walk(args.base_folder))[1]
    try:
        folders.remove("plots")
    except:
        pass
    for folder in folders:
        folder_pn = str(Path(args.base_folder,folder))
        print(folder_pn)
        subfolders = next(os.walk(folder_pn))[1]
        print(subfolders)
        if subfolders != []:
            read_paths = [Path(args.base_folder, folder, subfolder) for subfolder in subfolders]
            save_paths = [Path(args.save_folder, folder, subfolder) for subfolder in subfolders]
        else:
            read_paths = [Path(args.base_folder, folder)]
            save_paths = [Path(args.save_folder, folder)]
        if "best_epoch_3d" in subfolders:
            print('rotating the best epoch of a run')
            subfolders = next(os.walk(str(Path(args.base_folder, folder, "best_epoch_3d"))))[1]
            print(subfolders)
            read_paths = [Path(args.base_folder, folder, 'best_epoch_3d', subfolder) for subfolder in subfolders]
            save_paths = [Path(args.save_folder, folder, 'best_epoch_3d', subfolder) for subfolder in subfolders]
            copytree(str(Path(args.base_folder, folder, 'best_epoch_3d')), str(Path(args.save_folder, folder, 'best_epoch_3d')))

        else:
            print('rotating a data folder')
            #os.mkdir(str(Path(args.save_folder, folder)))
            #copytree(folder_pn, str(Path(args.save_folder, folder)))
        for r_path, s_path in zip(read_paths,save_paths):
            if not os.path.isdir(s_path):
                if not os.path.isdir(args.save_folder):
                    os.mkdir(args.save_folder)
                s_path.parent.mkdir(exist_ok=True)
                s_path.mkdir(exist_ok=True)
            if args.rot == 'rot':
                rotate(r_path, args.grp_regex, s_path)
            if args.rot == 'rot_back':
                rotate_back(r_path, args.grp_regex, s_path)


def rotate(r_path, grp_regex, s_path):
    filenames = [f for f in os.listdir(r_path) if f.endswith('.png')]
    # Might be needed in case of escape sequence problems
    # self.grp_regex = bytes(grp_regex, "utf-8").decode('unicode_escape')
    grouping_regex: Pattern = re.compile(grp_regex)

    stems: List[str] = [Path(filename).stem for filename in filenames]  # avoid matching the extension
    matches: List[Match] = map_(grouping_regex.match, stems)
    patients: List[str] = [match.group(0) for match in matches]

    unique_patients: List[str] = list(set(patients))

    for patient in unique_patients:
        patient_slices = [f for f in stems if f.startswith(patient)]
        w,h = [256,256]
        n = len(patient_slices)
        t = np.ndarray(shape=(w, h, n))
        for slice in patient_slices:
            slice_nb = int(re.split(grp_regex, slice)[1])
            t[:, :, slice_nb] = imageio.imread(r_path+'/'+slice+'.png')
        for i in range(0, h):
            im = np.pad(t[i,:,:], [(0, 0), (110, 110)], 'constant')
            imageio.imwrite(s_path+'/'+patient+str(i)+'.png', im)


def rotate_back(r_path, grp_regex, s_path):
    filenames = [f for f in os.listdir(r_path) if f.endswith('.png')]
    grouping_regex: Pattern = re.compile(grp_regex)

    stems: List[str] = [Path(filename).stem for filename in filenames]  # avoid matching the extension
    matches: List[Match] = map_(grouping_regex.match, stems)
    patients: List[str] = [match.group(0) for match in matches]

    unique_patients: List[str] = list(set(patients))
    print(unique_patients)
    for patient in unique_patients:
        patient_slices = [f for f in stems if f.startswith(patient)]
        w,h = [256,36]
        n = len(patient_slices)
        t = np.ndarray(shape=(w, h, n))
        for slice in patient_slices:
            slice_nb = int(re.split(grp_regex, slice)[1])
            im_or = imageio.imread(str(r_path)+'/'+slice+'.png')
            if im_or.shape !=(w,h):
                im_or = resize_im(im_or, 36)
            t[:, :, slice_nb] = im_or
        for i in range(0, h):
            im = t[:,i,:]
            imageio.imwrite(str(s_path)+'/'+patient+str(i)+'.png', im)


def copytree(src, dst, symlinks=False):
    for item in os.listdir(src):
        if not item.startswith(".") and not item.endswith(".csv"):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ig_f)
            else:
                shutil.copy2(s, d)


def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--base_folder', type=str, required=True)
    parser.add_argument('--folders', type=str, default=None)
    parser.add_argument('--save_folder', type=str, required=True)
    parser.add_argument('--rot', type=str, default='rot')
    parser.add_argument('--grp_regex', type=str, required=True)
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":

    main(get_args())

