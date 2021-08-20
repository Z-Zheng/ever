import torch
import torch.nn as nn
from ever import registry
import ever as er
from ever import module as erm


class Deeplabv3pDecoder(nn.Module):
    """
    This module is a reimplemented version in the following paper.
    Chen L C, Zhu Y, Papandreou G, et al. Encoder-decoder with atrous
    separable convolution for semantic image segmentation[J],
    """

    def __init__(self,
                 os4_feature_channels=256,
                 os16_feature_channels=2048,
                 aspp_channels=256,
                 aspp_atrous=(6, 12, 18),
                 reduction_dim=48,
                 out_channels=256,
                 num_3x3_convs=2,
                 scale_factor=4.0,
                 ):
        super(Deeplabv3pDecoder, self).__init__()
        self.scale_factor = scale_factor
        # 3x3 conv is better
        self.os4_transform = erm.ConvBlock(os4_feature_channels, reduction_dim, 3, 1, 1, bias=False)

        # 3x3 conv is better
        self.os16_transform = nn.Sequential(
            erm.AtrousSpatialPyramidPool(os16_feature_channels, aspp_channels, aspp_atrous),
            erm.ConvBlock(aspp_channels, aspp_channels, 3, 1, 1, bias=False)
        )

        layers = [erm.SeparableConvBlock(aspp_channels + reduction_dim, out_channels, 3, 1, 1, bias=False)]
        for i in range(num_3x3_convs - 1):
            layers.append(erm.SeparableConvBlock(out_channels, out_channels, 3, 1, 1, bias=False))

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        self.stack_conv3x3 = nn.Sequential(*layers)

    def forward(self, feat_list):
        os4_feat, os16_feat = feat_list
        os4_feat = self.os4_transform(os4_feat)
        os16_feat = self.os16_transform(os16_feat)

        feat_upx = self.upsample(os16_feat)

        concat_feat = torch.cat([os4_feat, feat_upx], dim=1)

        out = self.stack_conv3x3(concat_feat)

        return out


@registry.MODEL.register()
class Deeplabv3pHead(er.ERModule):
    def __init__(self, config):
        super(Deeplabv3pHead, self).__init__(config)
        self.head = nn.Sequential(
            Deeplabv3pDecoder(**self.config.deeplabv3p_decoder),
            nn.Conv2d(self.config.deeplabv3p_decoder.out_channels, self.config.num_classes, 1),
            nn.UpsamplingBilinear2d(scale_factor=self.config.upsample_scale)
        )

    def forward(self, x):
        x = self.head(x)
        return x

    def set_default_config(self):
        self.config.update(dict(
            deeplabv3p_decoder=dict(
                os4_feature_channels=256,
                os16_feature_channels=2048,
                aspp_channels=256,
                aspp_atrous=(6, 12, 18),
                reduction_dim=48,
                out_channels=256,
                num_3x3_convs=2,
                scale_factor=4.0,
            ),
            num_classes=3,
            upsample_scale=4.0
        ))
