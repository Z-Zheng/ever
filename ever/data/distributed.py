import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

__all__ = [
    'StepDistributedSampler',
    'DistributedNonOverlapSeqSampler',
    'StepDistributedRandomSubsetSampler',
    'DistributedInfiniteSampler',
    'as_ddp_inference_loader'
]


class StepDistributedSampler(DistributedSampler):
    def __init__(self, dataset, *, seed=0, drop_last=False, shuffle=True):
        super(StepDistributedSampler, self).__init__(
            dataset=dataset,
            num_replicas=None,
            rank=None,
            seed=seed,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        self.step = 0

    def set_step(self, step):
        self.step = step

    def __iter__(self):
        # deterministically shuffle based on step
        g = torch.Generator()
        g.manual_seed(self.seed + self.step)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class StepDistributedRandomSubsetSampler(StepDistributedSampler):
    def __init__(self, indices):
        super().__init__([], seed=0, drop_last=False, shuffle=True)

        self.indices = indices
        self.num_samples = int(math.ceil(len(self.indices) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on step
        g = torch.Generator()
        g.manual_seed(self.step)
        indices = [self.indices[i] for i in torch.randperm(len(self.indices), generator=g)]

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedNonOverlapSeqSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        super(DistributedNonOverlapSeqSampler, self).__init__(dataset, num_replicas, rank)

        self.num_samples = [len(self.dataset) // self.num_replicas] * self.num_replicas
        for i in range(len(self.dataset) % self.num_replicas):
            self.num_samples[i] += 1
        self.total_size = len(self.dataset)
        assert sum(self.num_samples) == self.total_size

    def __iter__(self):
        indices = torch.arange(len(self.dataset)).tolist()

        # subsample
        start = sum(self.num_samples[0:self.rank])
        end = sum(self.num_samples[0:self.rank + 1])

        indices = indices[start:end]
        assert len(indices) == self.num_samples[self.rank]

        return iter(indices)

    def __len__(self):
        return self.num_samples[self.rank]


class DistributedNonOverlapSubsetSeqSampler(DistributedSampler):
    def __init__(self, indices, num_replicas=None, rank=None):
        super().__init__([], num_replicas, rank)

        self.indices = indices
        self.num_samples = [len(self.indices) // self.num_replicas] * self.num_replicas
        for i in range(len(self.indices) % self.num_replicas):
            self.num_samples[i] += 1
        self.total_size = len(self.indices)
        assert sum(self.num_samples) == self.total_size

    def __iter__(self):
        # subsample
        start = sum(self.num_samples[0:self.rank])
        end = sum(self.num_samples[0:self.rank + 1])

        indices = self.indices[start:end]
        assert len(indices) == self.num_samples[self.rank]

        return iter(indices)

    def __len__(self):
        return self.num_samples[self.rank]


def as_ddp_inference_loader(dataloader):
    kwargs = dict(
        dataset=dataloader.dataset,
        batch_size=dataloader.batch_size,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        worker_init_fn=dataloader.worker_init_fn,
        collate_fn=dataloader.collate_fn,
        generator=dataloader.generator
    )
    if hasattr(dataloader.sampler, 'indices'):
        if not isinstance(dataloader.sampler, DistributedNonOverlapSubsetSeqSampler):
            dataloader = DataLoader(
                sampler=DistributedNonOverlapSubsetSeqSampler(dataloader.sampler.indices),
                **kwargs
            )
    else:
        if not isinstance(dataloader.sampler, DistributedNonOverlapSeqSampler):
            dataloader = DataLoader(
                sampler=DistributedNonOverlapSeqSampler(dataloader.dataset),
                **kwargs
            )

    return dataloader


class DistributedInfiniteSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1

        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size
        self.step = 0

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1

    def set_step(self, step):
        self.step = step

    def __len__(self):
        return math.ceil(len(self.dataset) / self.num_replicas)
