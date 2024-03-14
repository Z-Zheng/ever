import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PadCropWrapper(nn.Module):
    def __init__(self, module, size_divisor):
        super(PadCropWrapper, self).__init__()
        self.size_divisor = size_divisor
        self.module = module

    def forward(self, input: torch.Tensor):
        height, width = input.size(2), input.size(3)
        tail_pad = [0, 0, 0, 0]
        nheight = math.ceil(height / self.size_divisor) * self.size_divisor
        nwidth = math.ceil(width / self.size_divisor) * self.size_divisor
        pad = [0, nwidth - width, 0, nheight - height] + tail_pad

        pad_tensor = F.pad(input, pad=pad)

        pad_out = self.module(pad_tensor)
        return pad_out[:, :, :height, :width]
