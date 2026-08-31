import logging

import torch
import torch.distributed as dist


def reduce_loss_dict(loss_dict):
    """Average a loss dictionary across distributed workers."""
    with torch.no_grad():
        if dist.is_available() and dist.is_initialized():
            names = list(loss_dict)
            losses = torch.stack([loss_dict[name] for name in names])
            dist.all_reduce(losses, op=dist.ReduceOp.SUM)
            losses /= dist.get_world_size()
            loss_dict = dict(zip(names, losses))

        return {
            name: value.mean().item()
            for name, value in loss_dict.items()
        }


_initialized_loggers = set()


def get_root_logger(
    logger_name='fusion',
    log_level=logging.INFO,
    log_file=None,
):
    """Create a process-aware stream and optional file logger."""
    logger = logging.getLogger(logger_name)
    if logger_name in _initialized_loggers:
        return logger

    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    logger.setLevel(logging.ERROR if rank != 0 else log_level)

    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    _initialized_loggers.add(logger_name)
    return logger
