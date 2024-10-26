import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from .ops import SeparableConv2d, ConvBlock, Bf16compatible

__all__ = ['FPN',
           'LastLevelMaxPool',
           'LastLevelP6P7',
           'Fusion',
           'BiFPN',
           'AssymetricDecoder'
           ]


def init_conv(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=1)


def conv_with_kaiming_uniform(use_bn=False, use_relu=False):
    def make_conv(
            in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        return ConvBlock(in_channels, out_channels, kernel_size, stride,
                         padding=ConvBlock.same_padding(kernel_size, dilation),
                         dilation=dilation, bias=False, bn=use_bn, relu=use_relu,
                         init_fn=init_conv)

    return make_conv


default_conv_block = conv_with_kaiming_uniform(use_bn=False, use_relu=False)
conv_bn_block = conv_with_kaiming_uniform(use_bn=True, use_relu=False)
conv_bn_relu_block = conv_with_kaiming_uniform(use_bn=True, use_relu=True)


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(self,
                 in_channels_list,
                 out_channels,
                 conv_block=default_conv_block,
                 top_blocks=None
                 ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            conv_block: (nn.Module)
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            inner_block_module = conv_block(in_channels, out_channels, 1)
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = [getattr(self, self.layer_blocks[-1])(last_inner)]
        for feature, inner_block, layer_block in zip(
                x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue

            # make it compatible with bf16
            dtype = last_inner.dtype
            if dtype == torch.bfloat16:
                last_inner = last_inner.to(torch.float32)
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            if dtype == torch.bfloat16:
                inner_top_down = inner_top_down.to(dtype)  # cast back to bf16

            inner_lateral = getattr(self, inner_block)(feature)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))

        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        return tuple(results)


class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """

    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


class AssymetricDecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_feat_output_strides=(4, 8, 16, 32),
                 out_feat_output_stride=4,
                 norm_fn=nn.BatchNorm2d,
                 classifier_config=None):
        super(AssymetricDecoder, self).__init__()
        self.cls_cfg = classifier_config

        norm_fn_args = dict(num_features=out_channels)

        self.blocks = nn.ModuleList()
        for in_feat_os in in_feat_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - int(math.log2(int(out_feat_output_stride)))

            num_layers = num_upsample if num_upsample != 0 else 1

            self.blocks.append(nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2d(in_channels if idx == 0 else out_channels, out_channels, 3, 1, 1, bias=False),
                    norm_fn(**norm_fn_args) if norm_fn is not None else nn.Identity(),
                    nn.ReLU(True) if norm_fn == nn.BatchNorm2d else nn.GELU(),
                    Bf16compatible(nn.UpsamplingBilinear2d(scale_factor=2)) if num_upsample != 0 else nn.Identity(),
                )
                for idx in range(num_layers)]))

        if self.cls_cfg:
            scale_factor = classifier_config.get('scale_factor', 1)
            num_classes = classifier_config.get('num_classes', -1)
            kernel_size = classifier_config.get('kernel_size', 1)
            dropout_rate = classifier_config.get('dropout_rate', -1)
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
            self.classifier = nn.Sequential(
                nn.Conv2d(out_channels, num_classes, kernel_size, padding=(kernel_size - 1) // 2),
                Bf16compatible(nn.UpsamplingBilinear2d(scale_factor=scale_factor)) if scale_factor > 1 else nn.Identity()
            )

    def forward(self, feat_list: list):
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(feat_list[idx])
            inner_feat_list.append(decoder_feat)

        out_feat = sum(inner_feat_list) / len(inner_feat_list)
        if self.cls_cfg:
            out_feat = self.dropout(out_feat)
            out_feat = self.classifier(out_feat)
        return out_feat


class Fusion(nn.Module):
    eps = 0.0001

    def __init__(self, num_inputs, norm_method='fast_normalize'):
        super(Fusion, self).__init__()
        assert num_inputs > 1
        self.norm_method = norm_method
        self.weights = nn.Parameter(torch.Tensor(num_inputs))
        assert norm_method in ['softmax', 'fast_normalize']
        if 'softmax' == norm_method:
            self.norm_fn = lambda x: F.softmax(x, dim=0).view(num_inputs, 1, 1, 1, 1)
        elif 'fast_normalize' == norm_method:
            def _fast_normalize(x):
                x = F.relu(x)
                return x / (torch.sum(x, dim=0, keepdim=True) + Fusion.eps)

            self.norm_fn = lambda x: _fast_normalize(x).view(num_inputs, 1, 1, 1, 1)
        else:
            raise NotImplementedError(f'{norm_method} is not support for feature fusion.')
        self.reset_parameters()

    def forward(self, features: List[torch.Tensor]):
        return torch.sum(self.norm_fn(self.weights) * torch.stack(features, dim=0), dim=0)

    def reset_parameters(self):
        if self.norm_method == 'softmax':
            nn.init.zeros_(self.weights)
        elif self.norm_method == 'fast_normalize':
            nn.init.ones_(self.weights)


class FastNormalizedFusionConv3x3(nn.Sequential):
    def __init__(self, num_inputs, in_channels, out_channels):
        super(FastNormalizedFusionConv3x3, self).__init__(
            Fusion(num_inputs, 'fast_normalize'),
            SeparableConv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )


class NormalizedFusionConv3x3(nn.Sequential):
    def __init__(self, num_inputs, in_channels, out_channels):
        super(NormalizedFusionConv3x3, self).__init__(
            Fusion(num_inputs, 'softmax'),
            SeparableConv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )


class BiFPN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 feature_strides: List[int],
                 normalized_fusion: str = 'fast_normalize',
                 downsample_op='conv'  # conv | maxpool
                 ):
        super(BiFPN, self).__init__()
        cs = max(feature_strides)
        nf_op = FastNormalizedFusionConv3x3 if normalized_fusion == 'fast_normalize' else NormalizedFusionConv3x3
        self.feature_strides = feature_strides
        self.bin_fusion_modules = nn.ModuleList([
            nf_op(2, in_channels, in_channels) for _ in range(len(feature_strides) - 1)
        ])
        self.triple_fusion_modules = nn.ModuleList([
            nf_op(3, in_channels, in_channels) for _ in range(len(feature_strides) - 1)
        ])
        self.upsample_modules = nn.ModuleList([
            nn.UpsamplingNearest2d(scale_factor=2.)
            if cs / fs > 1 else nn.Sequential(nn.Conv2d(in_channels, in_channels, 1),
                                              nn.BatchNorm2d(in_channels),
                                              nn.ReLU(True)) for fs in feature_strides[::-1][1:]
        ])
        self.downsample_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 2, 1) if downsample_op == 'conv' else nn.MaxPool2d(3, 2, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True)
            )
            if cs / fs > 1 else nn.Sequential(nn.Conv2d(in_channels, in_channels, 1),
                                              nn.BatchNorm2d(in_channels),
                                              nn.ReLU(True)) for fs in feature_strides[:-1]
        ])

    def forward(self, features: List[torch.Tensor]):
        # top-down
        in_features = features.copy()
        inner_features = []
        for idx in range(len(self.feature_strides) - 1):
            x_top = features.pop()
            x_down = features.pop()
            upsample_op = self.upsample_modules[idx]

            inner_feature = self.bin_fusion_modules[idx]([x_down, upsample_op(x_top)])
            features.append(inner_feature)
            inner_features.append(inner_feature)

        inner_features.reverse()
        inner_features.append(in_features[-1])
        # bottom-up
        out_features = [inner_features[0]]
        for idx in range(len(self.feature_strides) - 1):
            x_bottom = inner_features.pop(0)
            x_up = inner_features.pop(0)

            downsample_op = self.downsample_modules[idx]

            out_feature = self.triple_fusion_modules[idx]([in_features[idx + 1], x_up, downsample_op(x_bottom)])

            inner_features.insert(0, out_feature)
            out_features.append(out_feature)

        return out_features
