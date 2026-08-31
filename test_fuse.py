import argparse
import logging
import os
import warnings
from collections import defaultdict
from os import path as osp

import yaml
import torch
import torch.nn.functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from modeling.unic import build_fusionmodel_from_args
from utils import get_root_logger

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description='Video Fusion Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset_config_path', type=str, default='options/fusion.yml')
    parser.add_argument('--window_size', type=int, default=384)
    parser.add_argument('--stride', type=int, default=192)
    parser.add_argument('--output_dir', type=str, default='./test_fusedHDO')
    parser.add_argument('--save_images', action='store_true')
    parser.add_argument(
        '--fuse_vi_y_channel',
        action='store_true',
        help='Convert VI RGB to Yx3 before fusion, then restore fused RGB with VI CbCr.',
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(osp.join(args.output_dir, 'images_fuse'), exist_ok=True)
    return args


def denormalize_video_tensor(vid, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    mean = torch.tensor(mean, device=vid.device, dtype=vid.dtype).view(-1, 1, 1, 1)
    std = torch.tensor(std, device=vid.device, dtype=vid.dtype).view(-1, 1, 1, 1)
    if vid.ndim == 5:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    vid = vid * std + mean
    vid = vid.clamp(0, 1)
    return vid


def normalize_video_tensor(vid, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    mean = torch.tensor(mean, device=vid.device, dtype=vid.dtype).view(-1, 1, 1, 1)
    std = torch.tensor(std, device=vid.device, dtype=vid.dtype).view(-1, 1, 1, 1)
    if vid.ndim == 5:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return (vid - mean) / std


def rgb_video_to_ycbcr(vid):
    """Convert RGB video tensor [3, T, H, W] in [0, 1] to Y/Cb/Cr."""
    r, g, b = vid[0:1], vid[1:2], vid[2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.5 * b + 128 / 255.0
    cr = 0.5 * r - 0.419 * g - 0.081 * b + 128 / 255.0
    return y.clamp(0, 1), cb.clamp(0, 1), cr.clamp(0, 1)


def ycbcr_video_to_rgb(y, cb, cr):
    """Convert Y/Cb/Cr video tensors [1, T, H, W] in [0, 1] to RGB."""
    r = y + 1.402 * (cr - 128 / 255.0)
    g = y - 0.344136 * (cb - 128 / 255.0) - 0.714136 * (cr - 128 / 255.0)
    b = y + 1.772 * (cb - 128 / 255.0)
    return torch.cat([r, g, b], dim=0).clamp(0, 1)


def prepare_vi_y3_for_fusion(gt_vi_single):
    vi_rgb = denormalize_video_tensor(gt_vi_single)
    vi_y, vi_cb, vi_cr = rgb_video_to_ycbcr(vi_rgb)
    vi_y3 = vi_y.repeat(3, 1, 1, 1)
    vi_y3_norm = normalize_video_tensor(vi_y3)
    return vi_y3_norm, vi_cb, vi_cr


def restore_fused_rgb_with_vi_chroma(fused_img, vi_cb, vi_cr):
    fused_rgb_like = denormalize_video_tensor(fused_img.unsqueeze(0)).squeeze(0)
    fused_y, _, _ = rgb_video_to_ycbcr(fused_rgb_like)
    return ycbcr_video_to_rgb(fused_y, vi_cb, vi_cr)


def sliding_window_inference(model, vi_video, ir_video, window_size=224, stride=112):
    _, _, H, W = vi_video.shape
    device = vi_video.device

    if H == window_size and W == window_size:
        with torch.no_grad():
            vi_video_batch = vi_video.unsqueeze(0)
            ir_video_batch = ir_video.unsqueeze(0)
            reconstructed = model(ir_video_batch, vi_video_batch)
            reconstructed = reconstructed.squeeze(0)
        return reconstructed

    pad_h = (window_size - H % stride) % stride if H > window_size else window_size - H
    pad_w = (window_size - W % stride) % stride if W > window_size else window_size - W

    vi_video_padded = F.pad(vi_video, (0, pad_w, 0, pad_h), mode='reflect')
    ir_video_padded = F.pad(ir_video, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, H_pad, W_pad = vi_video_padded.shape

    reconstructed = torch.zeros_like(vi_video_padded)
    weight_map = torch.zeros(1, 1, H_pad, W_pad, device=device)
    hann_1d = torch.hann_window(window_size, device=device)
    hann_2d = hann_1d.view(-1, 1) * hann_1d.view(1, -1)
    weight_window = hann_2d.view(1, 1, window_size, window_size)

    y_positions = list(range(0, H_pad - window_size + 1, stride))
    x_positions = list(range(0, W_pad - window_size + 1, stride))
    if len(y_positions) == 0 or y_positions[-1] + window_size < H_pad:
        y_positions.append(H_pad - window_size)
    if len(x_positions) == 0 or x_positions[-1] + window_size < W_pad:
        x_positions.append(W_pad - window_size)

    with torch.no_grad():
        for y in y_positions:
            for x in x_positions:
                vi_window = vi_video_padded[:, :, y:y + window_size, x:x + window_size]
                ir_window = ir_video_padded[:, :, y:y + window_size, x:x + window_size]
                vi_window_batch = vi_window.unsqueeze(0)
                ir_window_batch = ir_window.unsqueeze(0)
                rec_window = model(ir_window_batch, vi_window_batch)
                rec_window = rec_window.squeeze(0)
                reconstructed[:, :, y:y + window_size, x:x + window_size] += rec_window * weight_window
                weight_map[:, :, y:y + window_size, x:x + window_size] += weight_window

    reconstructed = reconstructed / (weight_map + 1e-8)
    reconstructed = reconstructed[:, :, :H, :W]
    return reconstructed


def save_merged_frames(frame_buffer, output_dir):
    """Merge frames from all clips and save one image per filename.

    This avoids saving duplicate frames from overlapping clips.
    """
    for modality in frame_buffer:
        modality_dir = osp.join(output_dir, f'images_{modality}')

        for prefix in frame_buffer[modality]:
            clips = frame_buffer[modality][prefix]
            sorted_clips = sorted(clips.items(), key=lambda x: x[0])
            video_dir = osp.join(modality_dir, prefix)
            os.makedirs(video_dir, exist_ok=True)

            saved_filenames = set()

            for _, frames in sorted_clips:
                # Sort by temporal index.
                sorted_frames = sorted(frames, key=lambda x: x[0])

                for _, frame, filename in sorted_frames:
                    # Skip frames already saved from an overlapping clip.
                    if filename in saved_filenames:
                        continue
                    saved_filenames.add(filename)

                    save_path = osp.join(video_dir, filename)
                    save_image(frame, save_path)


def test_video_reconstruction(model, test_loader, args, logger):
    logger.info("Starting video reconstruction testing...")
    if args.fuse_vi_y_channel:
        logger.info("Color mode: VI RGB -> Yx3 for fusion, fused Y + original VI CbCr -> RGB for saving.")
    else:
        logger.info("Color mode: normal RGB fusion path.")
    frame_buffer = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            gt_ir = data['gt_ir'].permute(0, 2, 1, 3, 4).cuda()
            gt_vi = data['gt_vi'].permute(0, 2, 1, 3, 4).cuda()
            folder = data['folder'][0] if isinstance(data['folder'], list) else data['folder']

            B, C, T, H, W = gt_ir.shape
            assert B == 1
            
            gt_ir_single = gt_ir[0]
            gt_vi_single = gt_vi[0]

            logger.info(f"Processing video {idx+1}/{len(test_loader)}: {folder}, size: {H}x{W}")
            
            # Determine original video prefix and clip index (e.g., 'scene-003' -> ('scene', 3))
            if isinstance(folder, str) and '-' in folder:
                try:
                    video_prefix, clip_suffix = folder.rsplit('-', 1)
                    clip_index = int(clip_suffix)
                except ValueError:
                    video_prefix, clip_index = folder, 0
            else:
                video_prefix, clip_index = folder, 0

            dataset = test_loader.dataset
            if hasattr(dataset, 'imgs_ir_gt') and folder in dataset.imgs_ir_gt:
                ir_paths = dataset.imgs_ir_gt[folder]
                filenames = [osp.basename(p) for p in ir_paths]
            else:
                filenames = [f'{t+1:04d}.png' for t in range(T)]

            vi_for_fusion = gt_vi_single
            vi_cb, vi_cr = None, None
            if args.fuse_vi_y_channel:
                vi_for_fusion, vi_cb, vi_cr = prepare_vi_y3_for_fusion(gt_vi_single)

            fused_img = sliding_window_inference(
                model, vi_for_fusion, gt_ir_single,
                window_size=args.window_size, stride=args.stride
            )

            if args.fuse_vi_y_channel:
                fused_img_denorm = restore_fused_rgb_with_vi_chroma(fused_img, vi_cb, vi_cr)
            else:
                fused_img_denorm = denormalize_video_tensor(fused_img.unsqueeze(0)).squeeze(0)

            if args.save_images:
                for t in range(T):
                    frame = fused_img_denorm[:, t, :, :].cpu()
                    filename = filenames[t] if t < len(filenames) else f'{t+1:04d}.png'
                    frame_buffer['fuse'][video_prefix][clip_index].append((t, frame, filename))

    if args.save_images and len(frame_buffer) > 0:
        logger.info("Merging and saving frames...")
        save_merged_frames(frame_buffer, args.output_dir)
        logger.info("Frames saved successfully!")


def load_dataset_config(dataset_config_path):
    with open(dataset_config_path, 'r') as f:
        opt = yaml.safe_load(f)
    return opt


def load_model(args, logger):

    logger.info(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model = build_fusionmodel_from_args(args).cuda()
    # model = build_student2_from_args(args).cuda()
    state_dict = checkpoint['model']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    logger.info("Model loaded successfully.")

    return model

def main():
    args = get_args()
    log_file = osp.join(args.output_dir, 'test_log.txt')
    logger = get_root_logger(logger_name='test', log_level=logging.INFO, log_file=log_file)
    dataset_opt = load_dataset_config(args.dataset_config_path)
    test_dataset_opt = dataset_opt['datasets']['val']
    from dinov2.utils.dataset_base import VideoFusionTestDataset
    test_dataset = VideoFusionTestDataset(test_dataset_opt)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    model = load_model(args, logger)
    test_video_reconstruction(model, test_loader, args, logger)


if __name__ == "__main__":
    main()
