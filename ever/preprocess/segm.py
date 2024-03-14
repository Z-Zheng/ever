import math
import random

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F


def transpose(img):
    if not F._is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    return img.transpose(Image.TRANSPOSE)


class ToTensor(object):
    def __init__(self, image_keep_255=False):
        self.image_keep_255 = image_keep_255

    def __call__(self, image, mask=None):
        if isinstance(image, np.ndarray) and image.dtype != np.uint8:
            if self.image_keep_255:
                return F.to_tensor(image)
            else:
                return F.to_tensor(image).div(255.)

        if self.image_keep_255:
            image = 255. * F.to_tensor(image)
        else:
            image = F.to_tensor(image)
        if mask is None:
            return image
        else:
            return image, torch.from_numpy(np.array(mask, copy=False))


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask):
        if random.random() < self.prob:
            image = F.hflip(image)
            mask = F.hflip(mask)
        return image, mask


class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask):
        if random.random() < self.prob:
            image = F.vflip(image)
            mask = F.vflip(mask)
        return image, mask


class RandomTranspose(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask):
        if random.random() < self.prob:
            image = transpose(image)
            mask = transpose(mask)
        return image, mask


class RandomRotate90K(object):
    def __init__(self, k=(0, 1, 2, 3)):
        self.k = k

    def __call__(self, image, mask):
        k = random.choice(self.k)
        if k == 0:
            return image, mask

        image = F.rotate(image, 90 * k, expand=True)
        mask = F.rotate(mask, 90 * k, expand=True)

        return image, mask


class RandomScale(object):
    def __init__(self, scales, size_divisor=32):
        self.scales = scales
        self.size_divisor = size_divisor

    def compute_size(self, image):
        h, w = image.height, image.width
        scale = random.choice(self.scales)
        nh = int(h * scale) // self.size_divisor * self.size_divisor
        nw = int(w * scale) // self.size_divisor * self.size_divisor
        return nw, nh

    def __call__(self, image, mask):
        new_size = self.compute_size(image)
        image = F.resize(image, new_size, Image.BILINEAR)
        mask = F.resize(mask, new_size, Image.NEAREST)
        return image, mask


class RandomCrop(object):
    def __init__(self, crop_size, mask_pad_value=255):
        self.crop_size = crop_size
        self.mask_pad_value = mask_pad_value

    def __call__(self, image, mask):
        ih, iw = image.height, image.width
        ch, cw = self.crop_size

        if ch > ih or cw > iw:
            ph = ch - ih
            pw = cw - iw
            image = F.pad(image, (0, 0, pw, ph), 0)
            mask = F.pad(mask, (0, 0, pw, ph), self.mask_pad_value)

        ih, iw = image.height, image.width

        ylim = ih - ch + 1
        xlim = iw - cw + 1

        ymin = random.randint(0, ylim)
        xmin = random.randint(0, xlim)

        image = F.crop(image, ymin, xmin, ch, cw)
        mask = F.crop(mask, ymin, xmin, ch, cw)
        return image, mask


class DivisiblePad(object):
    def __init__(self, size_divisor, mask_pad_value=255):
        self.size_divisor = size_divisor
        self.mask_pad_value = mask_pad_value

    def __call__(self, image, mask=None):
        ph = math.ceil(image.height / self.size_divisor) * self.size_divisor - image.height
        pw = math.ceil(image.width / self.size_divisor) * self.size_divisor - image.width

        if ph == 0 and pw == 0:
            if mask is None:
                return image
            return image, mask

        image = F.pad(image, (0, 0, pw, ph), 0)
        if mask is None:
            return image
        mask = F.pad(mask, (0, 0, pw, ph), self.mask_pad_value)
        return image, mask


class FixedPad(object):
    def __init__(self, target_size, mask_pad_value=255):
        self.target_size = target_size
        self.mask_pad_value = mask_pad_value

    def __call__(self, image, mask=None):
        th, tw = self.target_size
        h, w = image.height, image.width
        assert th >= h and tw >= w

        if th == h and tw == w:
            if mask is None:
                return image
            else:
                return image, mask

        ph = th - h
        pw = tw - w

        image = F.pad(image, (0, 0, pw, ph), 0)
        if mask is None:
            return image

        mask = F.pad(mask, (0, 0, pw, ph), self.mask_pad_value)
        return image, mask
