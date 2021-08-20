import random

import numpy as np
from albumentations import RandomScale, DualTransform
from albumentations.pytorch import ToTensorV2

__all__ = [
    'RandomDiscreteScale',
    'ToTensor',
    'ConstantPad'
]

class RandomDiscreteScale(RandomScale):
    def __init__(self, scales, always_apply=False, p=0.5):
        super(RandomDiscreteScale, self).__init__(0, always_apply=always_apply, p=p)
        self.scales = scales

    def get_params(self):
        return {"scale": random.choice(self.scales)}


class ToTensor(ToTensorV2):
    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask, 'masks': self.apply_to_masks}

    def apply_to_masks(self, masks, **params):
        return [self.apply_to_mask(m, **params) for m in masks]


class ConstantPad(DualTransform):
    def __init__(self,
                 min_height=1024,
                 min_width=1024,
                 value=None,
                 mask_value=None,
                 always_apply=False,
                 p=1.0, ):
        super(ConstantPad, self).__init__(always_apply, p)
        self.min_height = min_height
        self.min_width = min_width
        self.value = value
        self.mask_value = mask_value

    def update_params(self, params, **kwargs):
        params = super(ConstantPad, self).update_params(params, **kwargs)
        rows = params["rows"]
        cols = params["cols"]

        if rows < self.min_height:
            h_pad_top = 0
            h_pad_bottom = self.min_height - rows
        else:
            h_pad_top = 0
            h_pad_bottom = 0

        if cols < self.min_width:
            w_pad_left = 0
            w_pad_right = self.min_width - cols
        else:
            w_pad_left = 0
            w_pad_right = 0

        params.update(
            {"pad_top": h_pad_top, "pad_bottom": h_pad_bottom, "pad_left": w_pad_left, "pad_right": w_pad_right}
        )
        return params

    def apply(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant',
                      constant_values=self.value)

    def apply_to_mask(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                      constant_values=self.mask_value)

    def get_transform_init_args_names(self):
        return ("min_height", "min_width", "value", "mask_value")
