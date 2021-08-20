from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from ever.interface.configurable import ConfigurableMixin


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
