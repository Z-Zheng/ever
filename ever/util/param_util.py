from collections import OrderedDict
from functools import reduce

import numpy as np
import torch
import torch.nn as nn

from ever.core.logger import get_logger

logger = get_logger(__name__)


def trainable_parameters(module, _default_logger=logger):
    cnt = 0
    for p in module.parameters():
        if p.requires_grad:
            if len(p.shape) == 0:
                cnt += 1
            else:
                cnt += reduce(lambda x, y: x * y, list(p.shape))
    _default_logger.info('#trainable params: {}, {} M'.format(cnt, round(cnt / float(1e6), 3)))
    return cnt


def count_model_parameters(module, _default_logger=logger):
    cnt = 0
    for p in module.parameters():
        if len(p.shape) == 0:
            cnt += 1
        else:
            cnt += reduce(lambda x, y: x * y, list(p.shape))
    _default_logger.info('#params: {}, {} M'.format(cnt, round(cnt / float(1e6), 3)))

    return cnt


def freeze_params(module):
    for name, p in module.named_parameters():
        p.requires_grad = False
        # todo: show complete name
        # logger.info('[freeze params] {name}'.format(name=name))
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def freeze_modules(module, specific_class=None):
    for m in module.modules():
        if specific_class is not None:
            if not isinstance(m, specific_class):
                continue
        freeze_params(m)


def freeze_bn(module):
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            freeze_params(m)
            m.eval()


def count_model_flops(model, x, _default_logger=logger):
    try:
        from torchprofile import profile_macs
    except:
        print('please install torchprofile: pip install torchprofile')
        raise ModuleNotFoundError('torchprofile')
    model.eval()
    macs = profile_macs(model, x)
    _default_logger.info("# Mult-Adds: {}, {} B".format(macs, round(macs / 1e9, 2)))
    return macs


def count_model_params_flops(model, x=torch.ones(1, 3, 256, 256), _default_logger=logger):
    count_model_parameters(model, _default_logger)
    count_model_flops(model, x, _default_logger)


def copy_conv_parameters(src: nn.Conv2d, dst: nn.Conv2d):
    dst.weight.data = src.weight.data.clone().detach()
    if hasattr(dst, 'bias') and dst.bias is not None:
        dst.bias.data = src.bias.data.clone().detach()

    for name, v in src.__dict__.items():
        if name.startswith('_'):
            continue
        if name == 'kernel_size':
            assert dst.__dict__[name] == src.__dict__[name]

        dst.__dict__[name] = src.__dict__[name]


def copy_bn_parameters(src: nn.modules.batchnorm._BatchNorm, dst: nn.modules.batchnorm._BatchNorm):
    if dst.affine:
        dst.weight.data = src.weight.data.clone().detach()
        dst.bias.data = src.bias.data.clone().detach()
    dst.running_mean = src.running_mean
    dst.running_var = src.running_var
    dst.num_batches_tracked = src.num_batches_tracked
    for name, v in src.__dict__.items():
        if name.startswith('_'):
            continue
        dst.__dict__[name] = src.__dict__[name]


def copy_weight_bias(src: nn.Module, dst: nn.Module):
    if dst.weight is not None:
        dst.weight.data = src.weight.data.clone().detach()
    if dst.bias is not None:
        dst.bias.data = src.bias.data.clone().detach()
    for name, v in src.__dict__.items():
        if name.startswith('_'):
            continue
        dst.__dict__[name] = src.__dict__[name]
