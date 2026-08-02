import re
import inspect
from functools import wraps

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad

import ever as er
from ever.core import checkpoint
from ever.interface.configurable import ConfigurableMixin


class ERModule(nn.Module, ConfigurableMixin):
    __Keys__ = ['GLOBAL', ]
    _is_auto_config_class = False

    def __init__(self, config=None):
        super(ERModule, self).__init__()
        if config is None:
            config = dict()
        ConfigurableMixin.__init__(self, config)

        for key in ERModule.__Keys__:
            if key not in self.config:
                self.config[key] = dict()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if '__init__' not in cls.__dict__:
            return
        original_init = cls.__init__

        try:
            source_code = inspect.getsource(original_init)

            if 'super().__init__' in source_code or \
                    'super(cls, self).__init__' in source_code or \
                    'ERModule.__init__' in source_code:
                return

        except (OSError, TypeError):
            return

        @wraps(original_init)
        def wrapper(self, *args, **kwargs):
            sig = inspect.signature(original_init)
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()

            config_dict = dict(bound.arguments)
            config_dict.pop('self', None)

            super(cls, self).__init__(config_dict)

            original_init(self, *args, **kwargs)

        cls.__init__ = wrapper
        cls._is_auto_config_class = True
        # no need to set default config if using new verision __init__
        cls.set_default_config = lambda self: None

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

        state_dict = torch.load(
            self.config.GLOBAL.weight.path,
            map_location=lambda storage, loc: storage,
            weights_only=False
        )
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
            if '_orig_mod.' in k:
                k = k.replace('_orig_mod.', '')

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

    def compile_model(self, compile_config=None):
        if compile_config is False or compile_config is None:
            return self
        if compile_config is True:
            compile_config = {}
        self.compile(**compile_config)
        return self

    def fsdp2_modules(self, fsdp2_config=None):
        """Return child modules to shard before the root module with FSDP2.

        Subclasses may override this for architecture-aware wrapping, e.g. one
        transformer block per FSDP group. The default keeps the policy generic:
        match explicit class names, then a parameter-count threshold, and fall
        back to trainable top-level children when neither selects anything.
        """
        fsdp2_config = fsdp2_config or {}
        wrap_classes = set(fsdp2_config.get('wrap_module_classes', []) or [])
        min_num_params = fsdp2_config.get('min_num_params', None)
        modules = []

        for _, module in self.named_modules():
            if module is self:
                continue
            if not any(p.requires_grad for p in module.parameters()):
                continue
            if wrap_classes and module.__class__.__name__ in wrap_classes:
                modules.append(module)
                continue
            if min_num_params is not None:
                num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                if num_params >= min_num_params:
                    modules.append(module)

        if modules:
            return modules

        return [
            module for module in self.children()
            if any(p.requires_grad for p in module.parameters())
        ]

    def backward(self, loss_dict, amp, scaler, **kwargs):
        total_loss = sum([e for e in loss_dict.values()])
        if amp and scaler is not None:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

    def apply_gradients(self, optimizer, amp, scaler, **kwargs):
        if amp and scaler is not None:
            scaler.unscale_(optimizer)
            grad_info = self.clip_grad(optimizer)
            scaler.step(optimizer)
            scaler.update()
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
