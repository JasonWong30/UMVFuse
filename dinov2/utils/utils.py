import os
from os import path as osp
import numpy as np
import torch
import cv2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def img2tensor(imgs, bgr2rgb=True, float32=True):

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == "float64":
                img = img.astype("float32")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    if suffix is not None and (not isinstance(suffix, (str, tuple))):
        raise TypeError('"suffix" must be a string or tuple of strings')
    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith(".") and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)
                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            elif recursive:
                yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
            else:
                continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def mod_crop(img, scale):
    img = img.copy()
    if img.ndim in (2, 3):
        (h, w) = (img.shape[0], img.shape[1])
        (h_remainder, w_remainder) = (h % scale, w % scale)
        img = img[: h - h_remainder, : w - w_remainder, ...]
    else:
        raise ValueError(f"Wrong img ndim: {img.ndim}.")
    return img


def read_img_seq(
    path,
    require_mod_crop=False,
    scale=1,
    return_imgname=False,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
):
    if isinstance(path, list):
        img_paths = path
    else:
        img_paths = sorted(list(scandir(path, full_path=True)))
    imgs = []
    for v in img_paths:
        img = cv2.imread(v).astype(np.float32) / 255.0
        bgr_mean = np.array([mean[2], mean[1], mean[0]]).reshape(1, 1, 3)
        bgr_std = np.array([std[2], std[1], std[0]]).reshape(1, 1, 3)
        img = (img - bgr_mean) / bgr_std
        imgs.append(img)
    if require_mod_crop:
        imgs = [mod_crop(img, scale) for img in imgs]
    imgs = img2tensor(imgs, bgr2rgb=True, float32=True)
    imgs = torch.stack(imgs, dim=0)
    if return_imgname:
        imgnames = [osp.splitext(osp.basename(path))[0] for path in img_paths]
        return (imgs, imgnames)
    else:
        return imgs
