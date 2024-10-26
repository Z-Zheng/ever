import re

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad

import ever as er
from ever.core import checkpoint
from ever.interface.configurable import ConfigurableMixin


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
        raise NotImplementedError('The default config should be overridden.')

    def init_from_weight_file(self):
        if 'weight' not in self.config.GLOBAL:
            return
        if not isinstance(self.config.GLOBAL.weight, dict):
            return
        if 'path' not in self.config.GLOBAL.weight:
            return
        if self.config.GLOBAL.weight.path is None:
            return

        state_dict = torch.load(self.config.GLOBAL.weight.path,
                                map_location=lambda storage, loc: storage)
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

        IncompatibleKeys = self.load_state_dict(ret, strict=False)
        er.info('Load weights from: {}'.format(self.config.GLOBAL.weight.path))
        er.info(f'missing_keys ({len(IncompatibleKeys.missing_keys)}): {IncompatibleKeys.missing_keys}')
        er.info(f'unexpected_keys ({len(IncompatibleKeys.unexpected_keys)}): {IncompatibleKeys.unexpected_keys}')

    def log_info(self):
        return dict()

    def custom_param_groups(self):
        return [{'params': self.parameters()}, ]

    def backward(self, loss_dict, amp, **kwargs):
        total_loss = sum([e for e in loss_dict.values()])
        if amp:
            kwargs['scaler'].scale(total_loss).backward()
        else:
            total_loss.backward()

    def apply_gradients(self, optimizer, amp, **kwargs):
        if amp:
            kwargs['scaler'].unscale_(optimizer)
            grad_info = self.clip_grad(optimizer)
            kwargs['scaler'].step(optimizer)
            kwargs['scaler'].update()
        else:
            grad_info = self.clip_grad(optimizer)
            optimizer.step()

        optimizer.zero_grad()
        return grad_info

    def clip_grad(self, optimizer):
        grad_info = dict()
        if 'grad_clip' in optimizer.er_config:
            grad_clip_config = optimizer.er_config.get(
                'grad_clip',
                dict(max_norm=35, norm_type=2)
            )
            total_norm = clip_grad.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.parameters()),
                **grad_clip_config
            )
            grad_info['grad_norm'] = total_norm
        return grad_info
