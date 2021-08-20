import torch

__all__ = ['auto_device',
           ]


def auto_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def cpu_device():
    return torch.device('cpu')


def gpu_device(id=None):
    if id:
        return torch.device(f'cuda:{id}')
    else:
        return torch.device('cuda')
