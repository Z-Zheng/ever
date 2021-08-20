"""
Modified from https://github.com/xvjiarui/GCNet/blob/master/mmdet/ops/ct/context_block.py
"""
import torch
from torch import nn

from ..util import param_util


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


class ContextBlock2d(nn.Module):

    def __init__(self, in_channels, inner_dim, pool='att', fusions=('channel_add',)):
        """
        
        Args:
            in_channels: (int): Number of channels in the input image
            inner_dim: (int): Number of channels produced by the convolution
            pool: (str) pool type, `avg` or `att`
            fusions: list(str) names of funsion op, `channel_add` and `channel_mul`
        """
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.in_channels = in_channels
        self.inner_dim = inner_dim
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.inner_dim, kernel_size=1),
                nn.LayerNorm([self.inner_dim, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inner_dim, self.in_channels, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.inner_dim, kernel_size=1),
                nn.LayerNorm([self.inner_dim, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inner_dim, self.in_channels, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


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
        self.context_block = ContextBlock2d(planes * self.expansion, int(planes * self.expansion * ratio))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.context_block(out)
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
        self.context_block = ContextBlock2d(planes * self.expansion, int(planes * self.expansion * ratio))

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
        out = self.context_block(out)
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


def plugin_to_resnet(module: nn.Module, ratio):
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
            >>> m.resnet.layer2 = plugin_to_resnet(m.resnet.layer2, 1 / 16.)
            >>> m.resnet.layer3 = plugin_to_resnet(m.resnet.layer3, 1 / 16.)
            >>> m.resnet.layer4 = plugin_to_resnet(m.resnet.layer4, 1 / 16.)
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
        module_output.add_module(name, plugin_to_resnet(sub_module, ratio))
    del module
    return module_output


plugin_to_bottleneck = plugin_to_resnet
