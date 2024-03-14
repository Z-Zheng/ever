from ever.interface.configurable import ConfigurableMixin
from ever.data.distributed import StepDistributedSampler, DistributedInfiniteSampler

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data._utils.collate import default_collate


class ERDataLoader(DataLoader, ConfigurableMixin):
    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)

        DataLoader.__init__(self, **self.dataloader_params)

    @property
    def dataloader_params(self):
        return dict(dataset=list(),
                    batch_size=1,
                    shuffle=False,
                    sampler=None,
                    batch_sampler=None,
                    num_workers=0,
                    collate_fn=default_collate,
                    pin_memory=False,
                    drop_last=False,
                    timeout=0,
                    worker_init_fn=None)

    def set_default_config(self):
        return NotImplementedError


class ERDataset(Dataset, ConfigurableMixin):
    SUPPORT_SAMPLERS = {
        'StepDistributedSampler': StepDistributedSampler,
        'RandomSampler': RandomSampler,
        'SequentialSampler': SequentialSampler,
        'DistributedInfiniteSampler': DistributedInfiniteSampler
    }

    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)
        self.config.update(dict(
            batch_size=1,
            num_workers=0,
            prefetch_factor=2,
            persistent_workers=False,
            pin_memory=True,
            sampler_type='StepDistributedSampler',
        ))
        self.config.update(config)

    def set_default_config(self):
        return NotImplementedError

    def to_dataloader(self):
        sampler = self.SUPPORT_SAMPLERS[self.config.sampler_type](self)
        return DataLoader(
            dataset=self,
            sampler=sampler,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.persistent_workers,
            pin_memory=self.config.pin_memory,
        )
