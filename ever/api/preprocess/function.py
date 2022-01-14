import math

import torch
import torch.nn.functional as F

dim_error = ValueError('image dim should be 3 or 4.')


def th_mean_std_normalize(image, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
    """ this version faster than torchvision.transforms.functional.normalize


    Args:
        image: 3-D or 4-D array of shape [batch (optional), channel, height, width]
        mean:  a list or tuple or ndarray
        std: a list or tuple or ndarray

    Returns:

    """
    shape = [1] * image.dim()
    if image.dim() == 3:
        idx = 0
    elif image.dim() == 4:
        idx = 1
    else:
        raise dim_error
    shape[idx] = -1
    mean = torch.tensor(mean, requires_grad=False).reshape(*shape)
    std = torch.tensor(std, requires_grad=False).reshape(*shape)

    return image.sub(mean).div(std)


def th_divisible_pad(tensor, size_divisor: int, mode='constant', value=0):
    """

    Args:
        tensor: 3-D or 4-D tensor of shape [batch (optional), channel, height, width]
        size_divisor: int
        mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            Default: ``'constant'``
        value: fill value for ``'constant'`` padding. Default: ``0``

    Returns:

    """
    if tensor.dim() == 4:
        height, width = tensor.size(2), tensor.size(3)
        tail_pad = [0, 0, 0, 0]
    elif tensor.dim() == 3:
        height, width = tensor.size(1), tensor.size(2)
        tail_pad = [0, 0]
    elif tensor.dim() == 2:
        height, width = tensor.size(0), tensor.size(1)
        tail_pad = []
    else:
        raise dim_error
    nheight = math.ceil(height / size_divisor) * size_divisor
    nwidth = math.ceil(width / size_divisor) * size_divisor
    pad = [0, nwidth - width, 0, nheight - height] + tail_pad

    pad_tensor = F.pad(tensor, pad=pad, mode=mode, value=value)
    return pad_tensor


def th_pad_to_size(tensor, size, mode='constant', value=0):
    if tensor.dim() == 4:
        height, width = tensor.size(2), tensor.size(3)
        tail_pad = [0, 0, 0, 0]
    elif tensor.dim() == 3:
        height, width = tensor.size(1), tensor.size(2)
        tail_pad = [0, 0]
    elif tensor.dim() == 2:
        height, width = tensor.size(0), tensor.size(1)
        tail_pad = []
    else:
        raise dim_error
    ph = size[0] - height
    pw = size[1] - width
    assert ph >= 0 and pw >= 0
    pad_tensor = F.pad(tensor, pad=[0, pw, 0, ph] + tail_pad, mode=mode, value=value)
    return pad_tensor
