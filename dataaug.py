#!/usr/bin/env python3.6

import warnings
from sys import argv
from typing import Dict, Iterable
from pathlib import Path
from functools import partial
import os
import numpy as np
from skimage.io import imread, imsave
from skimage.exposure import rescale_intensity
from utils import mmap_,read_nii_image
import nibabel as nib
from multiprocessing import Pool
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image

def aug(base_folder,folder_gt, folder_im):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

    a_pool = Pool()

    gts: Iterable[str] = map(str,Path(base_folder,folder_gt).glob("*.nii"))
    print(gts)
    result = a_pool.map(save_aug,gts)

def save_aug(gt):
	subj_id = gt.split(reg)[1] 
	gt_or = nib.load(gt)
	try:
	    im_or = nib.load(str(Path(base_folder,folder_im,reg+subj_id)))
	    print('read', str(Path(base_folder,folder_im,reg+subj_id)))
	    img = read_nii_image(str(Path(base_folder,folder_im,reg+subj_id))).squeeze(0)
	    gt = read_nii_image(str(Path(base_folder,folder_gt,reg+subj_id))).astype('uint8')
	    img_max = img.max()
	    img_min = img.min()
	    img = rescale_intensity(img, in_range='image', out_range=(0,255)).astype('uint8') 
	    print(img.shape, img.min(), img.max(), gt.shape)
	    #contrast=iaa.GammaContrast(gamma=2.0)
	    #contrast = iaa.HistogramEqualization()
	    img, gt = seq(image=img, segmentation_maps= np.expand_dims(gt, axis=3))
	    #img, gt = seq(image=img, segmentation_maps= gt)
	    img = np.expand_dims(img, axis=0)
	    gt = gt.squeeze(3) 
	    print(img.shape, gt.shape)
	    img = rescale_intensity(img, in_range='image', out_range=(img_min,img_max))
	    img = nib.Nifti1Image(img, im_or.affine, im_or.header)
	    gt = nib.Nifti1Image(gt, gt_or.affine, gt_or.header)
	    nib.save(img,os.path.join(base_folder,folder_im+'aug', reg+'711111'+subj_id.split(".nii")[0]+".nii"))
	    im_or.to_filename(os.path.join(base_folder,folder_im+'aug', reg+subj_id))
	    gt.to_filename(os.path.join(base_folder,folder_gt+'aug', reg+'711111'+subj_id))
	    gt_or.to_filename(os.path.join(base_folder,folder_gt+'aug', reg+subj_id))
	except:
	    print("An exception occurred",str(Path(base_folder,folder_im,reg+subj_id)))

def main():
    base_folder = argv[1] #data/all_transverse/train
    folder_gt = argv[2] #GT
    folder_im = argv[3] #Inn or Wat
    reg = argv[4] #Inn or Wat
    aug(base_folder,folder_gt,folder_im) 

base_folder = argv[1] #data/all_transverse/train
folder_gt = argv[2] #GT
folder_im = argv[3] #Inn or Wat
reg = argv[4] #Inn or Wat
seq = iaa.Sequential([
    #iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    #iaa.Fliplr(0.5), # horizontally flip 50% of the images
    #iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
    iaa.HistogramEqualization()
])

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    #iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    #iaa.Sometimes(
	#0.5,
	#iaa.GaussianBlur(sigma=(0, 0.5))
    #),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    #iaa.Affine(
	#scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
	#translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
	#rotate=(-25, 25),
        #shear=(-8, 8)
    #)
], random_order=True) # apply augmenters in random order



if __name__ == "__main__":
    main()
