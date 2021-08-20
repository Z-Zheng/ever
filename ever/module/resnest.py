from ..interface import ERModule
import torch.nn as nn

try:
    from resnest.torch import resnest50, resnest101, resnest200, resnest269

    MODEL = dict(
        resnest50=resnest50,
        resnest101=resnest101,
        resnest200=resnest200,
        resnest269=resnest269,
    )
except ImportError:
    pass


class ResNeStEncoder(ERModule):
    def __init__(self, config):
        super(ResNeStEncoder, self).__init__(config)
        assert self.config.output_stride in [8, 16, 32]
        dilation = 32 // self.config.output_stride
        self.resnet = MODEL[self.config.name](pretrained=self.config.pretrained,
                                              dilation=dilation,
                                              norm_layer=self.config.norm_layer)
        del self.resnet.fc

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5

    def reset_in_channels(self, in_channels):
        if in_channels == 3:
            return self

        if self.config.name == 'resnest50':
            stem_width = 32
        else:
            stem_width = 64

        self.resnet.conv1[0] = nn.Conv2d(in_channels,
                                         stem_width, kernel_size=3, stride=2, padding=1,
                                         bias=False)

    @property
    def layer1(self):
        return self.resnet.layer1

    @layer1.setter
    def layer1(self, value):
        del self.resnet.layer1
        self.resnet.layer1 = value

    @property
    def layer2(self):
        return self.resnet.layer2

    @layer2.setter
    def layer2(self, value):
        del self.resnet.layer2
        self.resnet.layer2 = value

    @property
    def layer3(self):
        return self.resnet.layer3

    @layer3.setter
    def layer3(self, value):
        del self.resnet.layer3
        self.resnet.layer3 = value

    @property
    def layer4(self):
        return self.resnet.layer4

    def set_default_config(self):
        self.config.update(dict(
            name='',
            pretrained=False,
            output_stride=32,
            norm_layer=nn.BatchNorm2d
        ))
