#!/usr/bin/env python3.6

from random import random
from pathlib import Path
from multiprocessing.pool import Pool

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union

import torch
import numpy as np
from tqdm import tqdm
from torch import einsum
from torch import Tensor
from functools import partial, reduce
from skimage.io import imsave
from PIL import Image, ImageOps
from scipy.spatial.distance import directed_hausdorff
import torch.nn as nn
#import pydensecrf.densecrf as dcrf
#from pydensecrf.utils import unary_from_labels
#from pydensecrf.utils import unary_from_softmax
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from viewer import display_item

# functions redefinitions
tqdm_ = partial(tqdm, ncols=125,
                leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [' '{rate_fmt}{postfix}]')

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", Tensor, np.ndarray)


def compose(fns, init):
    return reduce(lambda acc, f: f(acc), fns, init)


def compose_acc(fns, init):
    return reduce(lambda acc, f: acc + [f(acc[-1])], fns, [init])


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return Pool().map(fn, iter)


def uc_(fn: Callable) -> Callable:
    return partial(uncurry, fn)


def uncurry(fn: Callable, args: List[Any]) -> Any:
    return fn(*args)


def id_(x):
    return x


# fns
def soft_size(a: Tensor) -> Tensor:
    return torch.einsum("bcwh->bc", [a])[..., None]


def batch_soft_size(a: Tensor) -> Tensor:
    return torch.einsum("bcwh->c", [a])[..., None]


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1, dtype=torch.float32) -> bool:
    _sum = t.sum(axis).type(dtype)
    _ones = torch.ones_like(_sum, dtype=_sum.dtype)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1, dtype=torch.float32) -> bool:
    return simplex(t, axis, dtype) and sset(t, [0, 1])


# # Metrics and shitz
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8, dtype=torch.float32) -> float:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(dtype)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(dtype)

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices


dice_coef = partial(meta_dice, "bcwh->bc")
dice_batch = partial(meta_dice, "bcwh->c")  # used for 3d dice


def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a & b


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a | b


# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


# Misc utils
def save_images(segs: Tensor, names: Iterable[str], root: str, mode: str, iter: int, remap: True) -> None:
    b, w, h = segs.shape  # Since we have the class numbers, we do not need a C axis

    for seg, name in zip(segs, names):
        seg = seg.cpu().numpy()
        if remap:
            #assert sset(seg, list(range(2)))
            seg[seg == 1] = 255
        save_path = Path(root, f"iter{iter:03d}", mode, name).with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        imsave(str(save_path), seg)

# Misc utils
def save_images_inf(segs: Tensor, names: Iterable[str], root: str, mode: str, remap: True) -> None:
    b, w, h = segs.shape  # Since we have the class numbers, we do not need a C axis

    for seg, name in zip(segs, names):
        seg = seg.cpu().numpy()
        if remap:
            #assert sset(seg, list(range(2)))
            seg[seg == 1] = 255
        save_path = Path(root, mode, name).with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        imsave(str(save_path), seg)


def augment(*arrs: Union[np.ndarray, Image.Image]) -> List[Image.Image]:
    imgs: List[Image.Image] = map_(Image.fromarray, arrs) if isinstance(arrs[0], np.ndarray) else list(arrs)

    if random() > 0.5:
        imgs = map_(ImageOps.flip, imgs)
    if random() > 0.5:
        imgs = map_(ImageOps.mirror, imgs)
    if random() > 0.5:
        angle = random() * 90 - 45
        imgs = map_(lambda e: e.rotate(angle), imgs)
    return imgs


def mask_resize(t, new_w):
    b, c, h, w = t.shape
    new_t = t
    if w != new_w:
        device = t.device()
        dtype = t.dtype()
        padd_lr = int((w - int(new_w)) / 2)
        m = torch.nn.ZeroPad2d((0, 0, padd_lr, padd_lr))
        mask_resize = torch.ones([new_w, h], dtype=dtype)
        mask_resize_fg = m(mask_resize)
        mask_resize_bg = 1 - mask_resize_fg
        new_t = torch.einsum('wh,bcwh->bcwh', [mask_resize, t]).to(device)
    return new_t

def resize(t, new_w):
    b, c, h, w = t.shape
    new_t = t
    if w != new_w:
        padd_lr = int((w - int(new_w)) / 2)
        new_t = t[:,: , :, padd_lr-1:padd_lr+new_w-1]
    return new_t


def resize_im(t, new_w):
    w, h = t.shape
    padd_lr = int((w - int(new_w)) / 2)
    new_t = t[:,padd_lr-1:padd_lr+new_w-1]
    return new_t


def haussdorf(preds: Tensor, target: Tensor, dtype=torch.float32) -> Tensor:
    assert preds.shape == target.shape
    assert one_hot(preds)
    assert one_hot(target)

    B, C, _, _ = preds.shape

    res = torch.zeros((B, C), dtype=dtype, device=preds.device)
    n_pred = preds.cpu().numpy()
    n_target = target.cpu().numpy()

    for b in range(B):
        for c in range(C):
            res[b, c] = numpy_haussdorf(n_pred[b, c], n_target[b, c])

    return res


def numpy_haussdorf(pred: np.ndarray, target: np.ndarray) -> float:
    assert len(pred.shape) == 2
    assert pred.shape == target.shape
    return max(directed_hausdorff(pred, target)[0], directed_hausdorff(target, pred)[0])


def interp(input):
    _, _, w, h = input.shape
    return nn.Upsample(input, size=(h, w), mode='bilinear')


def interp_target(input):
    _, _, w, h = input.shape
    return nn.Upsample(size=(h, w), mode='bilinear')


def plot_t(input):
    _, c, w, h = input.shape
    axis_to_plot = 1
    if c ==1:
        axis_to_plot = 0
    if input.requires_grad:
        im = input[0, axis_to_plot, :, :].detach().cpu().numpy()
    else:
        im = input[0, axis_to_plot, :, :].cpu().numpy()
    plt.close("all")
    plt.imshow(im, cmap='gray')
    plt.title('plotting on channel:'+ str(axis_to_plot))
    plt.colorbar()


def plot_all(gt_seg, s_seg, t_seg, disc_t):
    _, c, w, h = s_seg.shape
    axis_to_plot = 1
    if c ==1:
        axis_to_plot = 0
    s_seg = s_seg[0, axis_to_plot, :, :].detach().cpu().numpy()
    t_seg = t_seg[0, axis_to_plot, :, :].detach().cpu().numpy()
    gt_seg = gt_seg[0, axis_to_plot, :, :].cpu().numpy()
    disc_t = disc_t[0, 0, :, :].detach().cpu().numpy()
    plt.close("all")
    plt.subplot(141)
    plt.imshow(gt_seg, cmap='gray')
    plt.subplot(142)
    plt.imshow(s_seg, cmap='gray')
    plt.subplot(143)
    plt.imshow(t_seg, cmap='gray')
    plt.subplot(144)
    plt.imshow(disc_t, cmap='gray')
    plt.suptitle('gt, source seg, target seg, disc_t', fontsize=12)
    plt.colorbar()


def plot_as_viewer(gt_seg, s_seg, t_seg, s_im, t_im):
    _, c, w, h = s_seg.shape
    axis_to_plot = 1
    if c ==1:
        axis_to_plot = 0

    s_seg = s_seg[0, axis_to_plot, :, :].detach().cpu().numpy()
    t_seg = t_seg[0, axis_to_plot, :, :].detach().cpu().numpy()
    s_im = s_im[0, 0, :, :].detach().cpu().numpy()
    s_im = resize_im(s_im, s_seg.shape[1])
    t_im = t_im[0, 0, :, :].detach().cpu().numpy()
    t_im = resize_im(t_im, t_seg.shape[1])
    gt_seg = gt_seg[0, axis_to_plot, :, :].cpu().numpy()

    plt.close("all")
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 3)

    axe = fig.add_subplot(gs[0, 0])
    axe.imshow(gt_seg, cmap='gray')
    axe = fig.add_subplot(gs[0, 1])
    display_item(axe, s_im, s_seg, True)
    axe = fig.add_subplot(gs[0, 2])
    display_item(axe, t_im, t_seg, True)
    #fig.show()

    fig.suptitle('gt, source seg, target seg', fontsize=12)


def save_dict_to_file(dic, workdir):
    save_path = Path(workdir, 'params.txt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(save_path,'w')
    f.write(str(dic))
    f.close()


def load_dict_from_file(workdir):
    f = open(workdir+'/params.txt','r')
    data = f.read()
    f.close()
    return eval(data)


def eval_t(it_s_train, it_s_val, n_epoch):
    n_val = 768
    n_tra = 3328
    time_1epc = n_tra/it_s_train + n_val/it_s_val
    time_nepc_min = n_epoch*time_1epc/60
    time_nepc_hours = time_nepc_min/60
    print(f'> time left in hours: {round(time_nepc_hours, 2)}')
    print(f'> time left in min: {round(time_nepc_min, 2)}')


