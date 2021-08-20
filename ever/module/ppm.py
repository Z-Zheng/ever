import torch
import torch.nn as nn

import ever as er
from .ops import ConvBlock, PoolBlock


class PyramidPoolModule(nn.Module):
    def __init__(self,
                 in_channels,
                 pool_channels,
                 out_channels,
                 bins,
                 bottleneck_conv='3x3',
                 dropout=0):
        assert out_channels % len(bins) == 0
        super(PyramidPoolModule, self).__init__()
        self.pools = nn.ModuleList([
            PoolBlock(size, in_channels, pool_channels) for size in bins
        ])
        if bottleneck_conv == '3x3':
            self.conv = ConvBlock(pool_channels * len(bins) + in_channels, out_channels, 3, 1, 1, bias=False)
        elif bottleneck_conv == '1x1':
            self.conv = ConvBlock(pool_channels * len(bins) + in_channels, out_channels, 1, bias=False)
        else:
            self.conv = nn.Identity()

        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = [x]
        out += [p(x) for p in self.pools]
        out = torch.cat(out, dim=1)
        out = self.conv(out)
        return self.dropout(out)


@er.registry.MODEL.register()
class PPMHead(er.ERModule):
    def __init__(self, config):
        super(PPMHead, self).__init__(config)
        self.head = nn.Sequential(
            PyramidPoolModule(**self.config.ppm),
            nn.Conv2d(self.config.ppm.out_channels, self.config.num_classes, 1),
            nn.UpsamplingBilinear2d(scale_factor=self.config.upsample_scale)
        )

    def forward(self, x):
        x = self.head(x)
        return x

    def set_default_config(self):
        self.config.update(dict(
            ppm=dict(
                in_channels=2048,
                pool_channels=512,
                out_channels=512,
                bins=(1, 2, 3, 6)
            ),
            num_classes=3,
            upsample_scale=8.0
        ))
