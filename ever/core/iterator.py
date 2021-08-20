import warnings

import torch
from torch.utils.data.distributed import DistributedSampler

from . import to

__all__ = ['get_iterator',
           'Iterator',
           'PrefetchedIterator']


def get_iterator(type_name):
    if type_name in ITERATOR_TYPE:
        return ITERATOR_TYPE[type_name]
    else:
        raise KeyError('{} is not support.'.format(type_name))


class Iterator(object):
    def __init__(self, data_loader):
        self._data_loader = data_loader
        self._iterator = iter(self._data_loader)
        self._current_epoch = 0

    def next(self, forward_times=1, call_backs=None, is_master=True):
        data_list = []
        for _ in range(forward_times):
            try:
                data = next(self._iterator)
            except StopIteration:
                self.reset()
                self._current_epoch += 1
                if is_master:
                    if call_backs is not None:
                        for f in call_backs:
                            if isinstance(f, tuple):
                                f, call_back_interval = f
                                if self._current_epoch % call_back_interval != 0:
                                    continue
                            f()
                data = next(self._iterator)

            data_list.append(data)
        return data_list

    def reset(self):
        self._iterator = iter(self._data_loader)

    def iter(self, forward_times=1):
        """ a droplast iterator
        
        Args:
            forward_times: int

        Returns:
        
        """
        data_list = []
        for data in self._iterator:
            data_list.append(data)
            if len(data_list) == forward_times:
                yield data_list
                data_list = []

    def set_seed_for_dist_sampler(self, seed):
        if not isinstance(self._data_loader.sampler, DistributedSampler):
            return

        if self._data_loader.batch_sampler is not None:
            if hasattr(self._data_loader.batch_sampler.sampler, 'set_step'):
                self._data_loader.batch_sampler.sampler.set_step(seed)
            elif hasattr(self._data_loader.batch_sampler.sampler, 'set_epoch'):
                self._data_loader.batch_sampler.sampler.set_epoch(seed)

        elif self._data_loader.sampler is not None:
            if hasattr(self._data_loader.sampler, 'set_step'):
                self._data_loader.sampler.set_step(seed)
            elif hasattr(self._data_loader.sampler, 'set_epoch'):
                self._data_loader.sampler.set_epoch(seed)
        else:
            warnings.warn('batch_sampler and sampler are not found in data_loader, therefore no shuffle here.')


class Prefetcher(object):
    def __init__(self, dataloader):
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._dataloader = dataloader
        self.loader = iter(dataloader)
        self.stream = torch.cuda.Stream()

        self.preload()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.data
        try:
            self.preload()
        except StopIteration:
            raise StopIteration
        return data

    def preload(self):
        try:
            self.data = next(self.loader)
        except StopIteration:
            self.data = None
            raise StopIteration

        with torch.cuda.stream(self.stream):
            self.data = to.to_device(self.data, self._device, non_blocking=True)

    def reset(self):
        self.loader = iter(self._dataloader)
        self.preload()


class PrefetchedIterator(Iterator):
    def __init__(self, data_loader):
        super(PrefetchedIterator, self).__init__(data_loader)

        self._prefetcher = Prefetcher(data_loader)

    def next(self, forward_times=1, call_backs=None, is_master=True):
        data_list = []
        for _ in range(forward_times):
            try:
                data = self._prefetcher.next()
            except StopIteration:
                self.reset()
                if is_master:
                    if call_backs is not None:
                        for f in call_backs:
                            f()
                data = self._prefetcher.next()

            data_list.append(data)
        return data_list

    def reset(self):
        self._prefetcher.reset()
        self._iterator = iter(self._data_loader)


ITERATOR_TYPE = dict(
    normal=Iterator,
    prefetched=PrefetchedIterator,
)
