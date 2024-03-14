import math
from functools import reduce

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, Sampler

from ever.data.distributed import StepDistributedRandomSubsetSampler
from ever.data.distributed import DistributedNonOverlapSubsetSeqSampler
__all__ = [
    'CrossValSamplerGenerator',
    'make_CVSamplers'
]


class SubsetSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


class CrossValSamplerGenerator(object):
    def __init__(self,
                 dataset: Dataset,
                 distributed=True,
                 seed=2333):
        """

        Args:
            dataset: a instance of torch.utils.data.dataset.Dataset
            distributed: whether to use distributed random sampler
            seed: random seed for torch.randperm

        Example::
            >>> CV = CrossValSamplerGenerator(dataset, distributed=True, seed=2333)
            >>> sampler_pairs = CV.k_fold(5) # 5-fold CV
            >>> train_sampler, val_sampler = sampler_pairs[0] # 0-th as val, 1, 2, 3, 4 as train
        """
        self.num_samples = len(dataset)
        self.seed = seed
        self.distributed = distributed

    def k_fold(self, k=5):
        g = torch.Generator()
        g.manual_seed(self.seed)

        indices = torch.randperm(self.num_samples, generator=g).tolist()
        total_size = int(math.ceil(len(indices) / k) * k)
        offset = k - (total_size - self.num_samples)
        indices += indices[offset:(offset + total_size - len(indices))]

        assert len(indices) == total_size

        # subsample
        sampler_pairs = []
        k_fold_indices = [indices[i:total_size:k] for i in range(k)]
        for i in range(k):
            cp = k_fold_indices.copy()
            val_indices = cp.pop(i)
            train_indices = reduce(lambda a, b: a + b, cp)
            assert len(val_indices) + len(train_indices) == total_size

            if self.distributed:
                sampler_pairs.append((StepDistributedRandomSubsetSampler(train_indices),
                                      DistributedNonOverlapSubsetSeqSampler(val_indices)))
            else:
                sampler_pairs.append((SubsetRandomSampler(train_indices), SubsetSampler(val_indices)))

        return sampler_pairs


def make_CVSamplers(dataset, i=0, k=5, distributed=True, seed=2333):
    CV = CrossValSamplerGenerator(dataset, distributed=distributed, seed=seed)
    sampler_pairs = CV.k_fold(k)
    train_sampler, val_sampler = sampler_pairs[i]
    return train_sampler, val_sampler
