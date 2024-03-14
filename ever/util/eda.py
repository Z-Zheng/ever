from PIL import Image
import tifffile
import numpy as np


def thumbnail(image_path, sample_ratio=0.1) -> Image.Image:
    img = Image.open(image_path)
    img.thumbnail((int(img.width * sample_ratio), int(img.height * sample_ratio)))

    return img


def render_multi_binary_mask(file_paths, indexes, palette):
    imgs = [tifffile.imread(fp) for fp in file_paths]
    ret = np.zeros_like(imgs[0]).astype(np.uint8, copy=False)
    for img, index in zip(imgs, indexes):
        ret = np.where(ret == 0, index * (img / 255).astype(np.uint8, copy=False), ret)

    rret = Image.fromarray(ret)
    rret.putpalette(palette)

    return rret
