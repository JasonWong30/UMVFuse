import logging
import torch

initialized_logger = {}


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
    return (rank, world_size)


def get_root_logger(logger_name="basicsr", log_level=logging.INFO, log_file=None):
    logger = logging.getLogger(logger_name)
    if logger_name in initialized_logger:
        return logger
    format_str = "%(asctime)s %(levelname)s: %(message)s"
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    (rank, _) = get_dist_info()
    if rank != 0:
        logger.setLevel("ERROR")
    elif log_file is not None:
        logger.setLevel(log_level)
        file_handler = logging.FileHandler(log_file, "w")
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger
