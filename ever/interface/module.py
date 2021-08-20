import re

import torch
import torch.nn as nn

from ever.core import checkpoint, logger
from ever.interface.configurable import ConfigurableMixin

_logger = logger.get_logger()


class ERModule(nn.Module, ConfigurableMixin):
    __Keys__ = ['GLOBAL', ]

    def __init__(self, config=None):
        super(ERModule, self).__init__()
        if config is None:
            config = dict()
        ConfigurableMixin.__init__(self, config)

        for key in ERModule.__Keys__:
            if key not in self.config:
                self.config[key] = dict()

    def forward(self, *input):
        raise NotImplementedError

    def set_default_config(self):
        raise NotImplementedError('A default config should be overridden.')

    def init_from_weight_file(self):
        if 'weight' not in self.config.GLOBAL:
            return
        if not isinstance(self.config.GLOBAL.weight, dict):
            return
        if 'path' not in self.config.GLOBAL.weight:
            return
        if self.config.GLOBAL.weight.path is None:
            return

        state_dict = torch.load(self.config.GLOBAL.weight.path, map_location=lambda storage, loc: storage)
        if checkpoint.is_checkpoint(state_dict):
            state_dict = state_dict[checkpoint.CheckPoint.MODEL]
        ret = {}
        if 'excepts' in self.config.GLOBAL.weight and self.config.GLOBAL.weight.excepts is not None:
            pattern = re.compile(self.config.GLOBAL.weight.excepts)
        else:
            pattern = None

        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k.replace('module.', '')
            if getattr(pattern, 'match', lambda _: False)(k):
                # ignore
                continue
            ret[k] = v

        self.load_state_dict(ret, strict=False)
        _logger.info('Load weights from: {}'.format(self.config.GLOBAL.weight.path))

    def log_info(self):
        return dict()
