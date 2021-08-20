from ever.interface import ERModule
import torch.nn.functional as F
import torch
import torch.nn as nn
from ever import registry


class SimpleFusion(nn.Module):
    def __init__(self, in_channels):
        super(SimpleFusion, self).__init__()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )

    def forward(self, feat_list):
        x0 = feat_list[0]
        x0_h, x0_w = x0.size(2), x0.size(3)
        x1 = F.interpolate(feat_list[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(feat_list[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(feat_list[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x = torch.cat([x0, x1, x2, x3], dim=1)
        x = self.fuse_conv(x)
        return x


@registry.MODEL.register()
class HRNetHead(ERModule):
    def __init__(self, config):
        super(HRNetHead, self).__init__(config)
        self.head = nn.Sequential(
            SimpleFusion(**self.config.hrnet_decoder),
            nn.Conv2d(self.config.hrnet_decoder.in_channels, self.config.num_classes, 1),
            nn.UpsamplingBilinear2d(scale_factor=self.config.upsample_scale)
        )

    def forward(self, x):
        x = self.head(x)
        return x

    def set_default_config(self):
        self.config.update(dict(
            hrnet_decoder=dict(
                in_channels=480,
            ),
            num_classes=3,
            upsample_scale=4.0
        ))



