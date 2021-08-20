from ..core import registry, logger
from ..interface import ERModule
from ._densenet import densenet121
from ._densenet import densenet161
from ._densenet import densenet169
from ._densenet import densenet201

registry.MODEL.register('densenet121', densenet121)  # params=6.952 M,  32, (6, 12, 24, 16), 64
registry.MODEL.register('densenet161', densenet161)  # params=26.468 M, 48, (6, 12, 36, 24), 96
registry.MODEL.register('densenet201', densenet201)  # params=18.089 M, 32, (6, 12, 48, 32), 64
registry.MODEL.register('densenet169', densenet169)  # params=12.481 M, 32, (6, 12, 32, 32), 64

_logger = logger.get_logger()


@registry.MODEL.register('densenet_encoder')
class DenseNetEncoder(ERModule):
    def __init__(self, config):
        super(DenseNetEncoder, self).__init__(config)

        self.densenet = registry.MODEL[self.config.densenet_type](pretrained=self.config.pretrained,
                                                                  memory_efficient=self.config.memory_efficient)

        _logger.info('DenseNetEncoder: pretrained = {}'.format(self.config.pretrained))
        _logger.info('DenseNetEncoder: memory_efficient = {}'.format(self.config.memory_efficient))

    def layers(self, inds=(1, 2, 3, 4)):
        if not isinstance(inds, tuple) and not isinstance(inds, list):
            inds = (inds,)

        return [getattr(self.densenet.features, 'denseblock{}'.format(i)) for i in inds]

    def forward(self, inputs):
        feat_list = []
        self.densenet.features(inputs, feat_list)

        return feat_list

    def set_default_config(self):
        self.config.update(dict(
            densenet_type='densenet121',
            pretrained=True,
            memory_efficient=False
        ))

    def out_channels(self):
        if self.config.densenet_type == 'densenet121':
            return (256, 512, 1024, 1024)
        elif self.config.densenet_type == 'densenet161':
            return (384, 768, 2112, 2208)
        elif self.config.densenet_type == 'densenet201':
            return (256, 512, 1792, 1920)
        elif self.config.densenet_type == 'densenet169':
            return (256, 512, 1280, 1664)
        else:
            raise ValueError('do not support {}'.format(self.config.densenet_type))


if __name__ == '__main__':
    model = DenseNetEncoder(dict(densenet_type='densenet169', ))
    from ever.util import param_util
    import torch

    param_util.count_model_parameters(model)

    layers = model.layers()

    os = model(torch.ones(1, 3, 256, 256))
    for o in os:
        print(o.shape)
