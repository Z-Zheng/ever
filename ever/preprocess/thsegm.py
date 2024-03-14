import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class THRandomRotate90k(nn.Module):
    def __init__(self, k=None):
        super(THRandomRotate90k, self).__init__()
        self.k = k

    def __call__(self, images, masks=None):
        """ Rotate 90 * k degree for image and mask

        Args:
            images: 3-D tensor of shape [height, width, channel]
            masks: 2-D tensor of shape [height, width]

        Returns:
            images_tensor
            masks_tensor
        """
        ret = list()
        k = int(np.random.choice([0, 1, 2, 3], 1)[0]) if self.k is None else self.k
        if k == 0:
            ret.append(images)
            if masks is not None:
                ret.append(masks)
            return tuple(ret) if len(ret) > 1 else ret[0]

        images_tensor = torch.rot90(images, k, [0, 1])
        ret.append(images_tensor)
        if masks is not None:
            masks_tensor = torch.rot90(masks, k, [0, 1])
            ret.append(masks_tensor)

        return tuple(ret) if len(ret) > 1 else ret[0]


class THRandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5):
        super(THRandomHorizontalFlip, self).__init__()
        self.p = p

    def __call__(self, images, masks=None):
        """

        Args:
            images: 3-D tensor of shape [height, width, channel]
            masks: 2-D tensor of shape [height, width]

        Returns:
            images_tensor
            masks_tensor
        """

        ret = list()
        if self.p < np.random.uniform():
            ret.append(images)
            if masks is not None:
                ret.append(masks)
            return tuple(ret) if len(ret) > 1 else ret[0]

        images_tensor = torch.flip(images, [1])
        ret.append(images_tensor)
        if masks is not None:
            masks_tensor = torch.flip(masks, [1])
            ret.append(masks_tensor)

        return tuple(ret) if len(ret) > 1 else ret[0]


class THRandomVerticalFlip(nn.Module):
    def __init__(self, p=0.5):
        super(THRandomVerticalFlip, self).__init__()
        self.p = p

    def __call__(self, images, masks=None):
        """

        Args:
            images: 3-D tensor of shape [height, width, channel]
            masks: 2-D tensor of shape [height, width]

        Returns:
            images_tensor
            masks_tensor
        """
        ret = list()

        if self.p < np.random.uniform():
            ret.append(images)
            if masks is not None:
                ret.append(masks)
            return tuple(ret) if len(ret) > 1 else ret[0]

        images_tensor = torch.flip(images, [0])
        ret.append(images_tensor)
        if masks is not None:
            masks_tensor = torch.flip(masks, [0])
            ret.append(masks_tensor)

        return tuple(ret) if len(ret) > 1 else ret[0]


class THRandomCrop(nn.Module):
    def __init__(self, crop_size=(512, 512)):
        super(THRandomCrop, self).__init__()
        self.crop_size = crop_size

    def __call__(self, images, masks=None):
        """

        Args:
            images: 3-D tensor of shape [height, width, channel]
            masks: 2-D tensor of shape [height, width]

        Returns:
            images_tensor
            masks_tensor
        """
        im_h, im_w, _ = images.shape
        c_h, c_w = self.crop_size

        pad_h = c_h - im_h
        pad_w = c_w - im_w
        if pad_h > 0 or pad_w > 0:
            images = F.pad(images, [0, 0, 0, max(pad_w, 0), 0, max(pad_h, 0)], mode='constant', value=0)
            masks = F.pad(masks, [0, max(pad_w, 0), 0, max(pad_h, 0)], mode='constant', value=0)
        im_h, im_w, _ = images.shape

        y_lim = im_h - c_h + 1
        x_lim = im_w - c_w + 1
        ymin = int(np.random.randint(0, y_lim, 1))
        xmin = int(np.random.randint(0, x_lim, 1))

        xmax = xmin + c_w
        ymax = ymin + c_h
        ret = list()
        images_tensor = images[ymin:ymax, xmin:xmax, :]
        ret.append(images_tensor)
        if masks is not None:
            masks_tensor = masks[ymin:ymax, xmin:xmax]
            ret.append(masks_tensor)

        return tuple(ret)


class THRandomScale(nn.Module):
    def __init__(self, scale_range=(0.5, 2.0), scale_step=0.25):
        super(THRandomScale, self).__init__()
        scale_factors = np.linspace(scale_range[0], scale_range[1],
                                    int((scale_range[1] - scale_range[0]) / scale_step) + 1)
        self.scale_factor = np.random.choice(scale_factors, size=1)[0]

    def __call__(self, images, masks=None):
        """

        Args:
            images: 3-D tensor of shape [height, width, channel]
            masks: 2-D tensor of shape [height, width]

        Returns:
            images_tensor
            masks_tensor
        """
        ret = list()
        _images = images.permute(2, 0, 1)[None, :, :, :]
        images_tensor = F.interpolate(_images, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        images_tensor = images_tensor[0].permute(1, 2, 0)
        ret.append(images_tensor)
        if masks is not None:
            masks_tensor = F.interpolate(masks[None, None, :, :], scale_factor=self.scale_factor, mode='nearest')[0][0]
            ret.append(masks_tensor)

        return tuple(ret)
