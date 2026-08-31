# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import random
from abc import ABCMeta, abstractmethod
import numpy as np
import torch
from os import path as osp
import cv2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data.sampler import Sampler
import math

logger = logging.getLogger("dinov2")


class EnlargedSampler(Sampler):

    def __init__(self, dataset, num_replicas, rank, ratio=1):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) * ratio / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=g).tolist()

        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dist_info():
    if torch.distributed.is_available():
        initialized = torch.distributed.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends.

    All backends need to implement two apis: ``get()`` and ``get_text()``.
    ``get()`` reads the file as a byte stream and ``get_text()`` reads the file
    as texts.
    """

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass

class HardDiskBackend(BaseStorageBackend):
    """Raw hard disks storage backend."""

    def get(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def get_text(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'r') as f:
            value_buf = f.read()
        return value_buf   

class FileClient(object):

    _backends = {
        'disk': HardDiskBackend,
    }

    def __init__(self, backend='disk', **kwargs):
        if backend not in self._backends:
            raise ValueError(f'Backend {backend} is not supported. Currently supported ones'
                             f' are {list(self._backends.keys())}')
        self.backend = backend
        self.client = self._backends[backend](**kwargs)

    def get(self, filepath, client_key='default'):
        # client_key is used only for lmdb, where different fileclients have
        # different lmdb environments.
        if self.backend == 'lmdb':
            return self.client.get(filepath, client_key)
        else:
            return self.client.get(filepath)

    def get_text(self, filepath):
        return self.client.get_text(filepath)

def paired_random_crop_multi(img_ir_gts, img_vi_gts, gt_patch_size, scale=1, gt_path=None):
    """Paired random crop for both IR and VI images. Support Numpy array and Tensor inputs.
    It crops lists of GT images with corresponding locations for both IR and VI.
    
    Args:
        img_ir_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT IR images.
        img_vi_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT VI images.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor. Default: 1 (not used but kept for compatibility).
        gt_path (str): Path to ground-truth. Default: None.
        
    Returns:
        tuple: (cropped GT IR images, cropped GT VI images)
    """

    # Convert inputs to lists if they are not already
    if not isinstance(img_ir_gts, list):
        img_ir_gts = [img_ir_gts]
    if not isinstance(img_vi_gts, list):
        img_vi_gts = [img_vi_gts]

    # Determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_ir_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_gt_ir, w_gt_ir = img_ir_gts[0].size()[-2:]
        h_gt_vi, w_gt_vi = img_vi_gts[0].size()[-2:]
    else:
        h_gt_ir, w_gt_ir = img_ir_gts[0].shape[0:2]
        h_gt_vi, w_gt_vi = img_vi_gts[0].shape[0:2]

    if h_gt_ir < gt_patch_size or w_gt_ir < gt_patch_size:
        raise ValueError(f'GT IR ({h_gt_ir}, {w_gt_ir}) is smaller than patch size '
                         f'({gt_patch_size}, {gt_patch_size}). '
                         f'Please remove {gt_path}.')
                         
    if h_gt_vi < gt_patch_size or w_gt_vi < gt_patch_size:
        raise ValueError(f'GT VI ({h_gt_vi}, {w_gt_vi}) is smaller than patch size '
                         f'({gt_patch_size}, {gt_patch_size}). '
                         f'Please remove {gt_path}.')

    # Randomly choose top and left coordinates for gt patch
    top = random.randint(0, h_gt_ir - gt_patch_size)
    left = random.randint(0, w_gt_ir - gt_patch_size)

    # Crop GT patches
    if input_type == 'Tensor':
        img_ir_gts = [v[:, :, top:top + gt_patch_size, left:left + gt_patch_size] for v in img_ir_gts]
        img_vi_gts = [v[:, :, top:top + gt_patch_size, left:left + gt_patch_size] for v in img_vi_gts]
    else:
        img_ir_gts = [v[top:top + gt_patch_size, left:left + gt_patch_size, ...] for v in img_ir_gts]
        img_vi_gts = [v[top:top + gt_patch_size, left:left + gt_patch_size, ...] for v in img_vi_gts]

    # Return results
    if len(img_ir_gts) == 1:
        img_ir_gts = img_ir_gts[0]
    if len(img_vi_gts) == 1:
        img_vi_gts = img_vi_gts[0]
        
    return img_ir_gts, img_vi_gts


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
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
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img

def read_img_seq(path, require_mod_crop=False, scale=1, return_imgname=False, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    """Read a sequence of images from a given folder path.

    Args:
        path (list[str] | str): List of image paths or image folder path.
        require_mod_crop (bool): Require mod crop for each image.
            Default: False.
        scale (int): Scale factor for mod_crop. Default: 1.
        return_imgname(bool): Whether return image names. Defalt False.

    Returns:
        Tensor: size (t, c, h, w), RGB, [0, 1].
        list[str]: Returned image name list.
    """
    if isinstance(path, list):
        img_paths = path
    else:
        img_paths = sorted(list(scandir(path, full_path=True)))

    imgs = []
    for v in img_paths:
        img = cv2.imread(v).astype(np.float32) / 255.
        # ImageNet statistics use RGB order, while OpenCV loads BGR images.
        # Convert the mean and standard deviation from RGB to BGR order.
        bgr_mean = np.array([mean[2], mean[1], mean[0]]).reshape(1, 1, 3)  # BGR order for OpenCV
        bgr_std = np.array([std[2], std[1], std[0]]).reshape(1, 1, 3)
        img = (img - bgr_mean) / bgr_std
        imgs.append(img)

    if require_mod_crop:
        imgs = [mod_crop(img, scale) for img in imgs]
    imgs = img2tensor(imgs, bgr2rgb=True, float32=True)

    imgs = torch.stack(imgs, dim=0)

    if return_imgname:
        imgnames = [osp.splitext(osp.basename(path))[0] for path in img_paths]
        return imgs, imgnames
    else:
        return imgs


def generate_frame_indices(crt_idx, max_frame_num, num_frames, padding='reflection'):
    """Generate an index list for reading `num_frames` frames from a sequence
    of images.

    Args:
        crt_idx (int): Current center index.
        max_frame_num (int): Max number of the sequence of images (from 1).
        num_frames (int): Reading num_frames frames.
        padding (str): Padding mode, one of
            'replicate' | 'reflection' | 'reflection_circle' | 'circle'
            Examples: current_idx = 0, num_frames = 5
            The generated frame indices under different padding mode:
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            reflection_circle: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        list[int]: A list of indices.
    """
    assert num_frames % 2 == 1, 'num_frames should be an odd number.'
    assert padding in ('replicate', 'reflection', 'reflection_circle', 'circle'), f'Wrong padding mode: {padding}.'

    max_frame_num = max_frame_num - 1  # start from 0
    num_pad = num_frames // 2

    indices = []
    for i in range(crt_idx - num_pad, crt_idx + num_pad + 1):
        if i < 0:
            if padding == 'replicate':
                pad_idx = 0
            elif padding == 'reflection':
                pad_idx = -i
            elif padding == 'reflection_circle':
                pad_idx = crt_idx + num_pad - i
            else:
                pad_idx = num_frames + i
        elif i > max_frame_num:
            if padding == 'replicate':
                pad_idx = max_frame_num
            elif padding == 'reflection':
                pad_idx = max_frame_num * 2 - i
            elif padding == 'reflection_circle':
                pad_idx = (crt_idx - num_pad) - (i - max_frame_num)
            else:
                pad_idx = i - num_frames
        else:
            pad_idx = i
        indices.append(pad_idx)
    return indices


def normalize_image_np(img, flag='color', float32=False, mean = IMAGENET_DEFAULT_MEAN, std = IMAGENET_DEFAULT_STD):
    
    img_np = np.frombuffer(img, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    bgr_mean = np.array([mean[2], mean[1], mean[0]]).reshape(1, 1, 3)
    bgr_std = np.array([std[2], std[1], std[0]]).reshape(1, 1, 3)
    img = (img - bgr_mean) / bgr_std
   
    return img

def get_dataloaders(args, opt):
    # Import here to avoid circular import
    from dinov2.utils.dataset import build_dataset, build_dataloader
    
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, args.world_size, args.local_rank, dataset_enlarge_ratio)
            train_loader = build_dataloader(train_set,dataset_opt, num_gpu=args.num_gpu, 
                dist=args.world_size, sampler=train_sampler, seed=args.seed)

        elif phase == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=args.num_gpu, dist=args.world_size, sampler=None, seed=args.seed
            )
           
    return train_loader, val_loader
