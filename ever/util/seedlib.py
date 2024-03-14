import random

import numpy as np
import torch

from ever.core.dist import get_rank


def worker_init_fn(worker_id, num_workers, rank, seed=2333):
    """Worker init func for dataloader.
    The seed of each worker equals to num_worker * rank + worker_id + user_seed
    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def seed_torch(seed=2333, deterministic=False, device_specific=False):
    if device_specific:
        rank = get_rank()
        seed = seed + rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


seed_everything = seed_torch
