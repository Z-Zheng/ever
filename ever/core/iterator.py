import warnings

from torch.utils.data.distributed import DistributedSampler

from ever.interface.callback import Callback
from ever.core.dist import synchronize

__all__ = [
    'get_iterator',
    'Iterator',
]


def get_iterator(type_name):
    if type_name in ITERATOR_TYPE:
        return ITERATOR_TYPE[type_name]
    else:
        raise KeyError('{} is not support.'.format(type_name))


def run_callbacks(call_backs, current_epoch, is_master):
    if call_backs is None:
        return

    for f in call_backs:
        assert isinstance(f, Callback), 'f should be a er.Callback object'

        if f.interval < 0:
            continue
        if (current_epoch - 1) % f.interval != 0 or current_epoch == 1:
            continue

        if f.only_master:
            if is_master:
                f.func()
            synchronize()
        else:
            f.func()
            synchronize()


class Iterator:
    def __init__(self, data_loader):
        self._data_loader = data_loader
        self._iterator = iter(self._data_loader)
        self._step = 0
        self._look_up = {}

    def epoch(self, forward_times):
        return forward_times * self._step // len(self._data_loader) + 1

    def _ft1_get_data(self):
        try:
            data = next(self._iterator)
        except StopIteration:
            self.reset()
            data = next(self._iterator)
        return data

    def _ft2_get_data(self, forward_times=1):
        data_list = [self._ft1_get_data() for _ in range(forward_times)]
        return data_list

    def next(self, forward_times=1, call_backs=None, is_master=True):
        self._step += 1
        if self.epoch(forward_times) not in self._look_up:
            run_callbacks(call_backs, self.epoch(forward_times), is_master)
            self._look_up[self.epoch(forward_times)] = True

        if forward_times == 1:
            return [self._ft1_get_data()]
        else:
            return self._ft2_get_data(forward_times)

    def reset(self):
        self._iterator = iter(self._data_loader)

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
            warnings.warn(
                'batch_sampler and sampler are not found in data_loader, therefore no shuffle here.')

    def __del__(self):
        del self._iterator


ITERATOR_TYPE = dict(
    normal=Iterator,
)
