# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np

def percentile_minmax_normalize(image, p_min=2, p_max=98, to_uint8=False):
    minv = np.percentile(image, q=p_min, axis=(0, 1))
    maxv = np.percentile(image, q=p_max, axis=(0, 1))
    x = (image - minv) / (maxv - minv)
    x = x.clip(0, 1)

    if to_uint8:
        x *= 255
        x = x.astype(np.uint8)

    return x