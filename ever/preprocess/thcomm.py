import torch.nn as nn

import ever.preprocess.function as pF
from ever.core.to import to_tensor


class Pipeline(nn.Sequential):
    def __call__(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class FuncWrapper(nn.Module):
    def __init__(self, fn):
        super(FuncWrapper, self).__init__()
        self.fn = fn

    def __call__(self, *input):
        return self.fn(*input)


class ToTensor(nn.Module):
    def __call__(self, *input):
        return to_tensor(input)


class THChannelFirst(nn.Module):
    @staticmethod
    def _is_channel_first(tensor):
        return tensor.size(0) <= 8

    def __call__(self, image):
        if THChannelFirst._is_channel_first(image):
            return image
        return image.permute(2, 0, 1)


class THChannelFirst2(THChannelFirst):
    def __call__(self, image, other):
        return super(THChannelFirst2, self).__call__(image), other


class THMeanStdNormalize(nn.Module):
    def __init__(self, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
        super(THMeanStdNormalize, self).__init__()
        self._m = mean
        self._s = std

    def __call__(self, image):
        image = image.float()
        nimage = pF.th_mean_std_normalize(image, self._m, self._s)

        return nimage


class THMeanStdNormalize2(THMeanStdNormalize):
    def __init__(self, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
        super(THMeanStdNormalize2, self).__init__(mean, std)

    def __call__(self, image, other):
        nimage = super(THMeanStdNormalize2, self).__call__(image)
        return nimage, other


class THDivisiblePad(nn.Module):
    def __init__(self, size_divisor, mask_pad_value=255, pad_to_size=None):
        super(THDivisiblePad, self).__init__()
        self.size_divisor = size_divisor
        self.mask_pad_value = mask_pad_value
        self.pad_to_size = pad_to_size

    def __call__(self, image, mask=None):
        pimage = pF.th_divisible_pad(image, self.size_divisor)
        if self.pad_to_size is not None:
            pimage = pF.th_pad_to_size(image, self.pad_to_size)

        if mask is not None:
            pmask = pF.th_divisible_pad(mask, self.size_divisor, value=self.mask_pad_value)
            if self.pad_to_size:
                pmask = pF.th_pad_to_size(pmask, self.pad_to_size, value=self.mask_pad_value)
        else:
            pmask = mask

        return pimage, pmask
