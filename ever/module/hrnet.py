import torch.nn as nn
from torch.utils import checkpoint as cp

from ..core import registry, logger
from ..interface import ERModule
from . import context_block
from . import se_block
from ._hrnet import hrnetv2_w18
from ._hrnet import hrnetv2_w32
from ._hrnet import hrnetv2_w40
from ._hrnet import hrnetv2_w48

_logger = logger.get_logger()
registry.MODEL.register('hrnetv2_w18', hrnetv2_w18)
registry.MODEL.register('hrnetv2_w32', hrnetv2_w32)
registry.MODEL.register('hrnetv2_w40', hrnetv2_w40)
registry.MODEL.register('hrnetv2_w48', hrnetv2_w48)
defalut_config = dict(
    hrnet_type='hrnetv2_w18',
    pretrained=False,
    weight_path=None,
    norm_eval=False,
    frozen_stages=-1,
    with_cp=False
)


@registry.MODEL.register('HRNetEncoder')
class HRNetEncoder(ERModule):
    def __init__(self, config=defalut_config):
        super(HRNetEncoder, self).__init__(config)
        self.hrnet = registry.MODEL[self.config.hrnet_type](pretrained=self.config.pretrained,
                                                            weight_path=self.config.weight_path,
                                                            norm_eval=self.config.norm_eval,
                                                            frozen_stages=self.config.frozen_stages)
        _logger.info('HRNetEncoder: pretrained = {}'.format(self.config.pretrained))

    def forward(self, x):
        if self.config.with_cp and not self.training:
            return cp.checkpoint(self.hrnet, x)
        return self.hrnet(x)

    def reset_in_channels(self, in_channels):
        if in_channels == 3:
            return
        self.hrnet.add_module('conv1',
                              nn.Conv2d(in_channels,
                                        64, kernel_size=3, stride=2, padding=1,
                                        bias=False))

    # stage 2
    @property
    def stage2(self):
        return self.hrnet.stage2

    @stage2.setter
    def stage2(self, value):
        del self.hrnet.stage2
        self.hrnet.stage2 = value

    # stage 3
    @property
    def stage3(self):
        return self.hrnet.stage3

    @stage3.setter
    def stage3(self, value):
        del self.hrnet.stage3
        self.hrnet.stage3 = value

    # stage 4
    @property
    def stage4(self):
        return self.hrnet.stage4

    @stage4.setter
    def stage4(self, value):
        del self.hrnet.stage4
        self.hrnet.stage4 = value

    def set_default_config(self):
        self.config.update(defalut_config)

    def output_channels(self):
        if self.config.hrnet_type == 'hrnetv2_w18':
            return 18, 36, 72, 144
        elif self.config.hrnet_type == 'hrnetv2_w32':
            return 32, 64, 128, 256
        elif self.config.hrnet_type == 'hrnetv2_w40':
            return 40, 80, 160, 320
        elif self.config.hrnet_type == 'hrnetv2_w48':
            return 48, 96, 192, 384
        else:
            raise NotImplementedError('{} is not implemented.'.format(self.config.hrnet_type))

    def with_context_block(self, ratio):
        _logger.info('With context block (ratio = {})'.format(ratio))
        assert ratio in [1 / 16., 1 / 8.]
        self.stage2 = context_block.plugin_to_basicblock(self.stage2, ratio)
        self.stage3 = context_block.plugin_to_basicblock(self.stage3, ratio)
        self.stage4 = context_block.plugin_to_basicblock(self.stage4, ratio)

    def with_squeeze_excitation(self, inv_ratio):
        _logger.info('With squeeze_excitation block (inv_ratio = {})'.format(inv_ratio))
        assert inv_ratio in [16, 8]
        self.stage2 = se_block.plugin_to_basicblock(self.stage2, inv_ratio)
        self.stage3 = se_block.plugin_to_basicblock(self.stage3, inv_ratio)
        self.stage4 = se_block.plugin_to_basicblock(self.stage4, inv_ratio)
