import math

import numpy as np

from ever.core import registry
from ever.interface import LearningRateBase


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class WarmupMixin(object):
    def __init__(self, warmup_type, warmup_step, warmup_ratio):
        self.warmup_type = warmup_type
        self.warmup_step = warmup_step
        self.warmup_ratio = warmup_ratio

    def linear_warmup(self, cur_step, base_lr):
        k = (1 - cur_step / self.warmup_step) * (1 - self.warmup_ratio)
        warmup_lr = base_lr * (1 - k)
        return warmup_lr

    def exp_warmup(self, cur_step, base_lr):
        k = self.warmup_ratio ** (1 - cur_step / self.warmup_step)
        warmup_lr = base_lr * k
        return warmup_lr

    def constant_warmup(self, cur_step, base_lr):
        warmup_lr = base_lr * self.warmup_ratio
        return warmup_lr

    def get_warmup_lr(self, cur_step, base_lr):
        if not hasattr(self, f'{self.warmup_type}_warmup'):
            raise ValueError(f'unknonw warmup_type: {self.warmup_type}')

        return getattr(self, f'{self.warmup_type}_warmup')(cur_step, base_lr)


@registry.LR.register('multistep')
class MultiStepLearningRate(LearningRateBase, WarmupMixin):
    def __init__(self,
                 steps,
                 base_lr=0.1,
                 gamma=0.1,
                 warmup=None,
                 ):
        super(MultiStepLearningRate, self).__init__(base_lr=base_lr)
        self._steps = np.array(list(steps))
        self._gamma = gamma

        self.warmup = warmup
        if warmup is None:
            WarmupMixin.__init__(self, None, 0, None)
        else:
            WarmupMixin.__init__(self, warmup['type'], warmup['step'], warmup['ratio'])

        self._check()

    def _check(self):
        if self._steps.shape[0] > 1:
            assert np.all(np.diff(self._steps) > 0)
        assert self.warmup_step < self._steps[0]

    def step(self, global_step, optimizer):
        cur_step = global_step

        if self.warmup is not None:
            if global_step <= self.warmup_step:
                warmup_lr = self.get_warmup_lr(global_step, self.base_lr)
                set_lr(optimizer, warmup_lr)
                return

        lr = self._compute_lr(cur_step)

        set_lr(optimizer, lr)

    def _compute_lr(self, cur_step):
        return self._base_lr * (
                self._gamma ** int((cur_step > self._steps).sum(dtype=np.int32)))

    def _compute_warmup_lr(self, cur_step):
        lr = cur_step * (
                self._base_lr - self._warmup_init_lr) / self._warmup_step + self._warmup_init_lr
        return lr


@registry.LR.register('poly')
class PolyLearningRate(LearningRateBase, WarmupMixin):
    def __init__(self,
                 base_lr,
                 power,
                 max_iters,
                 warmup=None
                 ):
        super(PolyLearningRate, self).__init__(base_lr)
        self.power = power
        self.max_iters = max_iters

        self.warmup = warmup
        if warmup is None:
            WarmupMixin.__init__(self, None, 0, None)
        else:
            WarmupMixin.__init__(self, warmup['type'], warmup['step'], warmup['ratio'])

        assert self.warmup_step < self.max_iters

    def step(self, global_step, optimizer):
        factor = (1 - (global_step - self.warmup_step) / (
                self.max_iters - self.warmup_step)) ** self.power
        cur_lr = self.base_lr * factor

        if self.warmup is not None:
            if global_step <= self.warmup_step:
                warmup_lr = self.get_warmup_lr(global_step, self.base_lr)
                set_lr(optimizer, warmup_lr)
                return

        set_lr(optimizer, cur_lr)


@registry.LR.register('cosine')
class CosineAnnealingLearningRate(LearningRateBase):
    def __init__(self, base_lr, max_iters, eta_min):
        super(CosineAnnealingLearningRate, self).__init__(base_lr)
        self.eta_min = eta_min
        self.max_iters = max_iters

    def step(self, global_step, optimizer):
        cur_lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
                1 + math.cos(math.pi * global_step / self.max_iters))

        set_lr(optimizer, cur_lr)


@registry.LR.register('constant')
class ConstantLearningRate(LearningRateBase):
    def __init__(self, base_lr):
        super(ConstantLearningRate, self).__init__(base_lr)

    def step(self, global_step, optimizer):
        return self.base_lr


@registry.LR.register('search')
class SearchLearningRate(LearningRateBase):
    def __init__(self, init_lr, final_lr, max_iters):
        super(SearchLearningRate, self).__init__(init_lr)
        assert init_lr < final_lr
        assert max_iters > 0

        self.mult = (final_lr / init_lr) ** (1 / max_iters)

    def step(self, global_step, optimizer):
        mult = self.mult ** global_step
        set_lr(optimizer, mult * self.base_lr)
