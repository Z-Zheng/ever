## EVER Preset Module

#### Backbone
- ResNets as Encoder (support os 8, 16, 32)
    - ResNet18
    - ResNet34
    - ResNet50
    - ResNet101
    - ResNeXt50_32x4d
    - ResNeXt101_32x4d
    - ResNeXt101_32x8d
    - [GC-ResNets](https://arxiv.org/abs/1904.11492)
    - Config of ResNet with GN can be found [here](https://github.com/Z-Zheng/SimpleCV/tree/master/config_demo/resnet_with_gn.py).
    - ResNeSt50    
    - ResNeSt101
    - ResNeSt200
    - ResNeSt269
- DenseNets as Encoder
    - DenseNet 121
    - DenseNet 161
    - DenseNet 201
    - DenseNet 169

- HRNets as Encoder
    - HRNetv2-w18
    - HRNetv2-w32
    - HRNetv2-w40

- EfficientNet as Encoder
    - B0
    - ...
    - B7

#### Module
- [AtrousSpatialPyramidPool](https://arxiv.org/abs/1802.02611)
- [SE Block](https://arxiv.org/pdf/1709.01507.pdf)
- [Context Block](https://arxiv.org/abs/1904.11492)
- [FPN](https://arxiv.org/abs/1612.03144)
- [PPM](https://arxiv.org/abs/1612.01105)
- [Foreground-Scene Relation]()