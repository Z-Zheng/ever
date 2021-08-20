import torch
import torch.nn as nn

import ever as er
from .ops import ConvBlock, PoolBlock


class AtrousSpatialPyramidPool(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, conv_block=ConvBlock):
        super(AtrousSpatialPyramidPool, self).__init__()
        modules = []
        modules.append(conv_block(in_channels, out_channels, 1, bias=False))

        for rate in atrous_rates:
            modules.append(conv_block(in_channels, out_channels, 3, 1, rate, rate, bias=False))

        modules.append(PoolBlock(1, in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            conv_block(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class ASPPHead(er.ERModule):
    def __init__(self, config):
        super(ASPPHead, self).__init__(config)
        self.head = nn.Sequential(
            AtrousSpatialPyramidPool(**self.config.aspp),
            nn.Conv2d(self.config.aspp.out_channels, self.config.num_classes, 1),
            nn.UpsamplingBilinear2d(scale_factor=self.config.upsample_scale)
        )

    def forward(self, x):
        x = self.head(x)
        return x

    def set_default_config(self):
        self.config.update(dict(
            aspp=dict(
                in_channels=2048,
                out_channels=256,
                atrous_rates=[6, 12, 18]
            ),
            num_classes=3,
            upsample_scale=8.0
        ))
