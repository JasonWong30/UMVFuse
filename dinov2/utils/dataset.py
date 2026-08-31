import torch
from functools import partial
from .dataset_base import VideoFusionDataset, VideoFusionTestDataset
from dinov2.utils.utils import get_dist_info, worker_init_fn

class CPUPrefetcher():
    """CPU prefetcher.

    Args:
        loader: Dataloader.
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)

    def next(self):
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def reset(self):
        self.loader = iter(self.ori_loader)

def build_dataset(dataset_opt):
    """Build dataset from options.

    Args:
        dataset_opt (dict): Configuration for dataset. It must constain:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    if dataset_opt['type']=='VideoFusionDataset':
        dataset = VideoFusionDataset(dataset_opt)
    elif dataset_opt['type']=='VideoFusionTestDataset':
        dataset = VideoFusionTestDataset(dataset_opt)
    return dataset

def build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):

    phase = dataset_opt['phase'] 
    print(f"------------------------------------------------phase: {phase}")
    rank, dist = get_dist_info()
    if phase == 'train':
       
        if dist:  # distributed training
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:  # non-distributed training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True)
        if sampler is None:
            dataloader_args['shuffle'] = True
        dataloader_args['worker_init_fn'] = partial(
            worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
    elif phase in ['val', 'test']:  # validation
        dataloader_args = dict(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    else:
        raise ValueError(f'Wrong dataset phase: {phase}. ' "Supported ones are 'train', 'val' and 'test'.")

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)

    return torch.utils.data.DataLoader(**dataloader_args)


def create_test_dataloader(opt):
    # create train and val dataloaders
    test_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            test_set = build_dataset(dataset_opt)
            test_loader = build_dataloader(
                test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            print(f'Number of val images/folders in {dataset_opt["name"]}: ' f'{len(test_set)}')
        # else:
        #     raise ValueError(f'Dataset phase {phase} is not recognized.')

    return test_loader
