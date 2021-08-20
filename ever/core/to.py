import numpy as np
import torch


def to_tensor(blob):
    if isinstance(blob, np.ndarray):
        return torch.from_numpy(blob)
    if isinstance(blob, int) or isinstance(blob, float):
        return torch.Tensor(blob)

    if isinstance(blob, dict):
        ts = {}
        for k, v in blob.items():
            ts[k] = to_tensor(v)
        return ts

    if isinstance(blob, list):
        ts = list([to_tensor(e) for e in blob])
        return ts
    if isinstance(blob, tuple):
        # namedtuple
        if hasattr(blob, '_fields'):
            ts = {k: to_tensor(getattr(blob, k)) for k in blob._fields}
            ts = type(blob)(**ts)
        else:
            ts = tuple([to_tensor(e) for e in blob])
        return ts


def to_device(blob, device, *args, **kwargs):
    if hasattr(blob, 'to'):
        return blob.to(device, *args, **kwargs)
    if isinstance(blob, torch.Tensor):
        return blob.to(device, *args, **kwargs)

    if isinstance(blob, dict):
        ts = {}
        for k, v in blob.items():
            ts[k] = to_device(v, device)
        return ts

    if isinstance(blob, list):
        ts = list([to_device(e, device) for e in blob])
        return ts
    if isinstance(blob, tuple):
        # namedtuple
        if hasattr(blob, '_fields'):
            ts = {k: to_device(getattr(blob, k), device) for k in blob._fields}
            ts = type(blob)(**ts)
        else:
            ts = tuple([to_device(e, device) for e in blob])
        return ts
    return blob
