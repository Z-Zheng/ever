import torch
import torch.nn as nn

from ..util import param_util

GlobalAvgPool2D = lambda: nn.AdaptiveAvgPool2d(1)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction):
        super(SEBlock, self).__init__()
        self.gap = GlobalAvgPool2D()
        self.seq = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        v = self.gap(x)
        score = self.seq(v.view(v.size(0), v.size(1)))
        y = x * score.view(score.size(0), score.size(1), 1, 1)
        return y


class SCSEModule(nn.Module):
    # according to https://arxiv.org/pdf/1808.08127.pdf concat is better
    def __init__(self, in_channels, reduction=16, mode='concat'):
        super(SCSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

        self.spatial_se = nn.Sequential(nn.Conv2d(in_channels, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())
        self.mode = mode

    def forward(self, x):
        module_input = x

        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        chn_se = self.sigmoid(x)
        chn_se = chn_se * module_input

        spa_se = self.spatial_se(module_input)
        spa_se = module_input * spa_se
        if self.mode == 'concat':
            return torch.cat([chn_se, spa_se], dim=1)
        elif self.mode == 'maxout':
            return torch.max(chn_se, spa_se)
        else:
            return chn_se + spa_se


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, ratio, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.downsample = downsample
        self.stride = stride
        self.se = SEBlock(planes * self.expansion, ratio)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ratio, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.se = SEBlock(planes * self.expansion, ratio)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def plugin_to_basicblock(module: nn.Module, ratio):
    classname = module.__class__.__name__
    module_output = module
    if classname.find('BasicBlock') != -1:
        module_output = BasicBlock(module.conv1.in_channels,
                                   module.conv1.out_channels,
                                   ratio=ratio,
                                   stride=module.stride,
                                   downsample=module.downsample)
        # conv1 bn1
        param_util.copy_conv_parameters(module.conv1, module_output.conv1)
        if isinstance(module.bn1, nn.modules.batchnorm._BatchNorm):
            param_util.copy_bn_parameters(module.bn1, module_output.bn1)
        elif isinstance(module.bn1, nn.GroupNorm):
            param_util.copy_weight_bias(module.bn1, module_output.bn1)
        # conv2 bn2
        param_util.copy_conv_parameters(module.conv2, module_output.conv2)
        if isinstance(module.bn2, nn.modules.batchnorm._BatchNorm):
            param_util.copy_bn_parameters(module.bn2, module_output.bn2)
        elif isinstance(module.bn2, nn.GroupNorm):
            param_util.copy_weight_bias(module.bn2, module_output.bn2)

        del module
        return module_output

    for name, sub_module in module.named_children():
        module_output.add_module(name, plugin_to_basicblock(sub_module, ratio))
    del module
    return module_output


def plugin_to_bottleneck(module: nn.Module, ratio):
    """

    Args:
        module: (nn.Module): containing module
        ratio: (float) reduction ratio

    Returns:
        The original module with the converted `context_block.Bottleneck` layer

    Example::

            >>> # r16 ct c3-c5
            >>> from ever.module import ResNetEncoder
            >>> m = ResNetEncoder({})
            >>> m.resnet.layer2 = plugin_to_bottleneck(m.resnet.layer2, 1 / 16.)
            >>> m.resnet.layer3 = plugin_to_bottleneck(m.resnet.layer3, 1 / 16.)
            >>> m.resnet.layer4 = plugin_to_bottleneck(m.resnet.layer4, 1 / 16.)
    """
    classname = module.__class__.__name__
    module_output = module
    if classname.find('Bottleneck') != -1:
        module_output = Bottleneck(module.conv1.in_channels,
                                   module.conv1.out_channels,
                                   ratio=ratio,
                                   stride=module.stride,
                                   downsample=module.downsample)
        # conv1 bn1
        param_util.copy_conv_parameters(module.conv1, module_output.conv1)
        if isinstance(module.bn1, nn.modules.batchnorm._BatchNorm):
            param_util.copy_bn_parameters(module.bn1, module_output.bn1)
        elif isinstance(module.bn1, nn.GroupNorm):
            param_util.copy_weight_bias(module.bn1, module_output.bn1)
        # conv2 bn2
        param_util.copy_conv_parameters(module.conv2, module_output.conv2)
        if isinstance(module.bn2, nn.modules.batchnorm._BatchNorm):
            param_util.copy_bn_parameters(module.bn2, module_output.bn2)
        elif isinstance(module.bn2, nn.GroupNorm):
            param_util.copy_weight_bias(module.bn2, module_output.bn2)
        # conv3 bn3
        param_util.copy_conv_parameters(module.conv3, module_output.conv3)
        if isinstance(module.bn3, nn.modules.batchnorm._BatchNorm):
            param_util.copy_bn_parameters(module.bn3, module_output.bn3)
        elif isinstance(module.bn3, nn.GroupNorm):
            param_util.copy_weight_bias(module.bn3, module_output.bn3)

        del module
        return module_output

    for name, sub_module in module.named_children():
        module_output.add_module(name, plugin_to_bottleneck(sub_module, ratio))
    del module
    return module_output
