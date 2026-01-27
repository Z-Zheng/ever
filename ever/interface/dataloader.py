from ever.core.dist import get_world_size
from ever.core.logger import info
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
            total_batch_size=-1,
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

    def to_dataloader(self, batch_size=None, num_workers=None, prefetch_factor=None, persistent_workers=None, pin_memory=None):
        sampler = self.SUPPORT_SAMPLERS[self.config.sampler_type](self)

        if self.config.total_batch_size > 0:
            num_processors = get_world_size()
            assert self.config.total_batch_size % num_processors == 0, \
                f'total_batch_size ({self.config.total_batch_size}) must be divisible by num_processors ({num_processors}).'

            self.config.batch_size = self.config.total_batch_size // num_processors
            info(f'using [`total_batch_size` = {self.config.total_batch_size}, `num_processors` = {num_processors}] instead of `batch_size`')

        batch_size = batch_size or self.config.batch_size
        num_workers = num_workers or self.config.num_workers
        prefetch_factor = prefetch_factor or self.config.prefetch_factor
        persistent_workers = persistent_workers or self.config.persistent_workers
        pin_memory = pin_memory or self.config.pin_memory

        return DataLoader(
            dataset=self,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
        )
