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


def count_model_flops(model, x):
    """ count the macs of model
    This implementation is modified version of
    https://github.com/nmhkahn/torchsummaryX/blob/558b0ec4e5f8efdbdf4244cc7b5e10ce66095910/torchsummaryX/torchsummaryX.py#L5

    Args:
        model: nn.Module
        x: 4-D tensor as the input of model

    Returns:

    """

    def register_hook(module):
        def hook(module, inputs, outputs):
            cls_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            key = "{}_{}".format(module_idx, cls_name)

            info = OrderedDict()
            info["id"] = id(module)
            if isinstance(outputs, (list, tuple)):
                info["out"] = list(outputs[0].size())
            else:
                info["out"] = list(outputs.size())

            info["ksize"] = "-"
            info["inner"] = OrderedDict()
            info["params"], info["macs"] = int(0), int(0)
            for name, param in module.named_parameters():
                info["params"] += param.nelement()

                if name == "weight":
                    ksize = list(param.size())
                    # to make [in_shape, out_shape, ksize, ksize]
                    if len(ksize) > 1:
                        ksize[0], ksize[1] = ksize[1], ksize[0]
                    info["ksize"] = ksize

                    # ignore N, C when calculate Mult-Adds in ConvNd
                    if "Conv" in cls_name:
                        info["macs"] += int(param.nelement() * int(np.prod(info["out"][2:])))
                    else:
                        info["macs"] += param.nelement()

                # RNN modules have inner weights such as weight_ih_l0
                elif "weight" in name:
                    info["inner"][name] = list(param.size())
                    info["macs"] += param.nelement()

            # if the current module is already-used, mark as "(recursive)"
            # check if this module has params
            if list(module.named_parameters()):
                for v in summary.values():
                    if info["id"] == v["id"]:
                        info["params"] = "(recursive)"

            if info["params"] == 0:
                info["params"], info["macs"] = "-", "-"

            summary[key] = info

        # ignore Sequential and ModuleList
        if not module._modules:
            hooks.append(module.register_forward_hook(hook))

    hooks = []
    summary = OrderedDict()

    model.apply(register_hook)
    with torch.no_grad():
        model(x)

    for hook in hooks:
        hook.remove()

    total_params, total_macs = 0, 0
    for layer, info in summary.items():
        repr_macs = info["macs"]

        if isinstance(repr_macs, (int, float)):
            total_macs += repr_macs

    logger.info("# Mult-Adds: {0:,.2f} B".format(total_macs / 1000000000))
    return total_macs


def onnx_based_count_model_flops(model, x, _default_logger=logger):
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
    onnx_based_count_model_flops(model, x, _default_logger)


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
