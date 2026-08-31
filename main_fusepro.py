import argparse
import logging
import os
import sys
import time
import warnings
from os import path as osp

import yaml
import torch
import torch.distributed as dist
import torch.optim as optim
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid

from dinov2.utils.dataset import CPUPrefetcher
from dinov2.utils.utils import get_dataloaders
from loss import Fusion_loss2
from modeling.unic import build_fusionmodel_from_args
from utils import get_root_logger, reduce_loss_dict

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description='Video Fusion Training')

    # === Model Loading ===
    parser.add_argument('--checkpoint', type=str,
                        default=None,
                        help='Optional path to a pretrained encoder checkpoint')
    # === Dataset Configuration ===
    parser.add_argument('--dataset_config_path', type=str,
                        default='options/fusion.yml',
                        help='Path to dataset configuration YAML file')
    # === Training Hyperparameters ===
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_decay_step', type=int, default=200,
                        help='Learning rate decay every N iterations')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.95,
                        help='Learning rate decay factor')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=1,
                        help='Starting epoch')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Gradient clipping max norm')
    # === Logging & Saving ===
    parser.add_argument('--output_dir', type=str, default='./experiments/fusion3',
                        help='Output directory for logs and checkpoints')
    parser.add_argument('--exp_name', type=str, default='VideoFusionV2',
                        help='Experiment name')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print frequency (iterations)')
    parser.add_argument('--save_checkpoint_freq', type=int, default=500,
                        help='Save checkpoint frequency (iterations)')

    # === Distributed Training ===
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(osp.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(osp.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(osp.join(args.output_dir, 'tb_logger'), exist_ok=True)

    return args



def load_dataset_config(dataset_config_path):
    with open(dataset_config_path, 'r') as f:
        opt = yaml.safe_load(f)
    if 'datasets' not in opt:
        raise ValueError(f"Configuration file {dataset_config_path} must contain 'datasets' key")

    logging.getLogger('fusion').info(
        f"Loaded dataset configuration from {dataset_config_path}"
    )
    return opt


def denormalize_video_tensor(
    vid: torch.Tensor,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
):
    """Denormalize video tensor from ImageNet normalization"""
    mean = torch.tensor(mean, device=vid.device, dtype=vid.dtype).view(-1, 1, 1, 1)
    std = torch.tensor(std, device=vid.device, dtype=vid.dtype).view(-1, 1, 1, 1)

    if vid.ndim == 5:  # [N, C, T, H, W]
        mean = mean.unsqueeze(0)  # [1, C, 1, 1, 1]
        std = std.unsqueeze(0)    # [1, C, 1, 1, 1]

    vid = vid * std + mean
    vid = vid.clamp(0, 1)

    return vid


def calculate_loss(fused_img, gt_ir, gt_vi, loss_fn):   
        
    loss_dict = loss_fn(fused_img, gt_ir, gt_vi)   
    total_loss = loss_dict['loss_fusion']
        
    return total_loss, loss_dict

def main():
    torch.autograd.set_detect_anomaly(True)
    args = get_args()

    # === Initialize Distributed Training ===
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.local_rank = local_rank
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    args.world_size = dist.get_world_size()
    args.num_gpu = args.world_size
    rank = dist.get_rank()

    # === Set Random Seed ===
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)

    # === Logger Setup ===
    log_file = osp.join(
        args.output_dir,
        'logs',
        f"train_{args.exp_name}_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.log"
    )
    logger = get_root_logger(logger_name='fusion', log_level=logging.INFO, log_file=log_file)
    logger.info(f"Arguments: {vars(args)}")

    # TensorBoard logger
    from torch.utils.tensorboard import SummaryWriter
    if rank == 0:
        tb_logger = SummaryWriter(log_dir=osp.join(args.output_dir, 'tb_logger'))
    else:
        tb_logger = None

    # === Build Model ===
    model = build_fusionmodel_from_args(args).cuda()

    # ===  encoder  ===
    for name, param in model.automodel.encoder.named_parameters():
        param.requires_grad = False

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # === Optimizer ===
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    start_epoch = args.start_epoch
    start_iter = 0

    # === Build Dataset from Config File ===
    dataset_opt = load_dataset_config(args.dataset_config_path)

    train_loader, _ = get_dataloaders(args, dataset_opt)

    # === Prefetcher ===
    prefetcher = CPUPrefetcher(train_loader)

    # === Training Loop ===
    logger.info(f'Start training from epoch: {start_epoch}, iter: {start_iter}')

    fusion_loss_fn = Fusion_loss2().cuda()      

    iter_time = time.time()
    current_iter = start_iter
    model.train()

    # model.module.encoder.eval()  # Keep encoder in eval mode
    for epoch in range(start_epoch, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            current_iter += 1

            # === Data Preparation ===
            gt_ir = train_data['gt_ir'].permute(0, 2, 1, 3, 4).cuda()  # B C T H W
            gt_vi = train_data['gt_vi'].permute(0, 2, 1, 3, 4).cuda()  # B C T H W            

            # === Forward Pass (Fusion Model) ===
            fused_img = model(gt_ir, gt_vi)

            # === Denormalize for Loss Calculation ===
            fused_img_denorm = denormalize_video_tensor(fused_img)
            gt_ir_denorm = denormalize_video_tensor(gt_ir)
            gt_vi_denorm = denormalize_video_tensor(gt_vi)

            # === Loss Calculation ===
            fusion_loss, loss_dict = calculate_loss(
                fused_img_denorm, gt_ir_denorm, gt_vi_denorm, fusion_loss_fn
            )           
            total_loss = fusion_loss

            # === Backward Pass ===
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)

            optimizer.step()

            # === Learning Rate Decay ===
            if current_iter % args.lr_decay_step == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * args.lr_decay_gamma

            # === Logging ===
            if current_iter % args.print_freq == 0:
                # Gather losses across GPUs
                log_dict = reduce_loss_dict(loss_dict)

                current_lr = optimizer.param_groups[0]['lr']
                iter_duration = time.time() - iter_time

                log_msg = (
                    f"Epoch: {epoch:04d} | Iter: {current_iter:06d} | "
                    f"LR: {current_lr:.6f} | "
                    f"Loss: {total_loss.item():.4f} | "
                    f"Time: {iter_duration:.3f}s"
                )
                logger.info(log_msg)

                # TensorBoard logging
                if rank == 0 and tb_logger is not None:
                    tb_logger.add_scalar('train/total_loss', total_loss.item(), current_iter)
                    tb_logger.add_scalar('train/lr', current_lr, current_iter)
                    for key, value in log_dict.items():
                        tb_logger.add_scalar(f'train/{key}', value, current_iter)

                    # Log images (middle frame)
                    fused_img_vis = rearrange(fused_img_denorm, 'b c t h w -> b t c h w')
                    gt_ir_vis = rearrange(gt_ir_denorm, 'b c t h w -> b t c h w')
                    gt_vi_vis = rearrange(gt_vi_denorm, 'b c t h w -> b t c h w')

                    b, n, _, _, _ = gt_ir_vis.size()
                    t = n // 2  # Middle frame

                    tb_img_samples = []
                    for sample_idx in [0, 1, -2, -1]:  # First, second, second-last, last
                        if abs(sample_idx) < b:
                            tb_img_samples.extend([
                                gt_ir_vis[sample_idx, t].detach().float().cpu(),
                                gt_vi_vis[sample_idx, t].detach().float().cpu(),
                                fused_img_vis[sample_idx, t].detach().float().cpu()
                            ])

                    if len(tb_img_samples) > 0:
                        tb_img = make_grid(tb_img_samples, nrow=3, padding=2)
                        tb_logger.add_image('train/images', tb_img, current_iter)

                    tb_logger.flush()

                # Check for NaN
                if torch.isnan(total_loss):
                    logger.error("Loss is NaN! Stopping training.")
                    sys.exit(1)

            # === Save Checkpoint ===
            if current_iter % args.save_checkpoint_freq == 0 and rank == 0:
                save_filename = f'VideoFusionNet_{current_iter:06d}.pth'
                save_path = osp.join(args.output_dir, 'models', save_filename)

                save_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'iter': current_iter,
                    'args': vars(args),
                }

                torch.save(save_dict, save_path)
                logger.info(f"Saved checkpoint to {save_path}")

            iter_time = time.time()
            # Synchronize
            dist.barrier()
            # Next batch
            train_data = prefetcher.next()

    # === Cleanup ===
    if rank == 0 and tb_logger is not None:
        tb_logger.close()

    dist.destroy_process_group()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
