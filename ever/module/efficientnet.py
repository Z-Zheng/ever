import math
import warnings

import ever as er


@er.registry.MODEL.register()
class EFNetEncoder(er.ERModule):
    def __init__(self, config):
        super(EFNetEncoder, self).__init__(config)
        try:
            from efficientnet_pytorch import EfficientNet
        except ImportError:
            warnings.warn('Please install efficientnet_pytorch via pip')

        class EfficientNetFeatureExtractor(EfficientNet):
            def extract_features(self, inputs):
                # Stem
                x = self._swish(self._bn0(self._conv_stem(inputs)))
                # Blocks
                feat_list = []
                for idx, block in enumerate(self._blocks):
                    prev_x = x
                    drop_connect_rate = self._global_params.drop_connect_rate
                    if drop_connect_rate:
                        drop_connect_rate *= float(idx) / len(self._blocks)
                    x = block(prev_x, drop_connect_rate=drop_connect_rate)
                    if tuple(block._depthwise_conv.stride) == (2, 2) or hasattr(block._depthwise_conv, 'endpoint'):
                        feat_list.append(prev_x)
                feat_list.append(x)
                return feat_list

        if self.config.pretrained:
            self.features = EfficientNetFeatureExtractor.from_pretrained(self.config.ef_name)
        else:
            self.features = EfficientNetFeatureExtractor.from_name(self.config.ef_name)
        self.tune_output_stride(self.config.output_stride)

    def forward(self, x):
        feat_list = self.features.extract_features(x)
        return feat_list

    def set_default_config(self):
        self.config.update(dict(
            ef_name='efficientnet-b0',
            pretrained=False,
            # 8, 16, 32
            output_stride=32,
        ))

    def tune_output_stride(self, output_stride):
        OS = output_stride
        assert OS in [8, 16, 32]
        stem_stride = self.features._conv_stem.stride[0]
        n = int(math.log2(OS) - math.log2(stem_stride))
        m = 0
        for block in self.features._blocks:
            if tuple(block._depthwise_conv.stride) == (2, 2):
                if n == 0:
                    m += 1
                else:
                    n -= 1
            if m > 0:
                nostride_dilate(block._depthwise_conv, 2 ** m)

    @property
    def out_channels(self):
        assert self.config.ef_name in ['efficientnet-b{}'.format(i) for i in range(8)]
        if 'efficientnet-b0' == self.config.ef_name:
            return 16, 24, 40, 112, 320
        if 'efficientnet-b1' == self.config.ef_name:
            return 16, 24, 40, 112, 320
        if 'efficientnet-b2' == self.config.ef_name:
            return 16, 24, 48, 120, 352
        if 'efficientnet-b3' == self.config.ef_name:
            return 24, 32, 48, 136, 384
        if 'efficientnet-b4' == self.config.ef_name:
            return 24, 32, 56, 160, 448
        if 'efficientnet-b5' == self.config.ef_name:
            return 24, 40, 64, 176, 512
        if 'efficientnet-b6' == self.config.ef_name:
            return 32, 40, 72, 200, 576
        if 'efficientnet-b7' == self.config.ef_name:
            return 32, 48, 80, 224, 640


def nostride_dilate(m, dilate):
    from efficientnet_pytorch.utils import Conv2dStaticSamePadding
    if isinstance(m, Conv2dStaticSamePadding):
        if tuple(m.stride) == (2, 2):
            m.stride = (1, 1)
            setattr(m, 'endpoint', True)
            if m.kernel_size == (3, 3):
                m.dilation = (dilate // 2, dilate // 2)
                m.static_padding.padding = (dilate // 2, dilate // 2, dilate // 2, dilate // 2)
            if m.kernel_size == (5, 5):
                m.dilation = (dilate // 2, dilate // 2)
                m.static_padding.padding = (dilate, dilate, dilate, dilate)
        else:
            if m.kernel_size == (3, 3):
                m.dilation = (dilate, dilate)
                m.static_padding.padding = (dilate, dilate, dilate, dilate)

            if m.kernel_size == (5, 5):
                m.dilation = (dilate, dilate)
                m.static_padding.padding = (2 * dilate, 2 * dilate, 2 * dilate, 2 * dilate)


if __name__ == '__main__':
    import torch

    model = EFNetEncoder(dict(ef_name='efficientnet-b0',
                              output_stride=32))
    print(model)
    model.eval()

    with torch.no_grad():
        out = model(torch.ones(1, 3, 512, 512))
        for o in out:
            print(o.shape)
