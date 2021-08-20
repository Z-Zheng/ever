class LearningRateBase(object):
    def __init__(self, base_lr):
        self._base_lr = base_lr

    @property
    def base_lr(self):
        return self._base_lr

    def step(self, global_step, optimizer):
        raise NotImplementedError