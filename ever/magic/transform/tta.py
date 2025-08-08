import torch
import torch.nn as nn
import ever as er

__all__ = [
    'tta',
    'TestTimeAugmentation'
]


def tta(model, image, tta_config):
    trans = er.MultiTransform(
        *tta_config
    )
    images = trans.transform(image)
    with torch.no_grad():
        outs = [model(im) for im in images]

    outs = trans.inv_transform(outs)

    out = sum(outs) / len(outs)

    return out


class TestTimeAugmentation(nn.Module):
    def __init__(self, module, tta_config):
        super(TestTimeAugmentation, self).__init__()
        self.module = module
        self.trans = er.MultiTransform(
            *tta_config
        )

    @torch.no_grad()
    def forward(self, image):
        images = self.trans.transform(image)
        outs = [self.module(im) for im in images]

        outs = self.trans.inv_transform(outs)

        out = sum(outs) / len(outs)
        return out
