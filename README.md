## EVer - A Library for Earth Vision Researcher
[![Downloads](https://pepy.tech/badge/ever-beta)](https://pepy.tech/project/ever-beta)

EVer is a Pytorch-based Python library to simplify the training and inference of the deep learning model in the remote sensing domain.

> This is a **beta** version for research only.


## Features

- Common codebase for reproducible research
- Accelerating our Earth Vision research
- Single workflow of "data-module-configs"


## Installation

### stable version (0.4.1)
```bash
pip install ever-beta
```

### nightly version (master)
```bash
pip install --upgrade git+https://github.com/Z-Zheng/ever.git
```


## Getting Started
[Basic Usage](https://github.com/Z-Zheng/ever/tree/master/docs/USAGE.md)

## Projects using EVer

### Change Detection (Our Change Family)
- (AnyChange) Segment Any Change, arxiv 2024 [[`Paper`](https://arxiv.org/abs/2402.01188)]
- (Changen) Scalable Multi-Temporal Remote Sensing Change Data Generation via Simulating Stochastic Change Process, ICCV 2023 [[`Paper`](https://arxiv.org/pdf/2309.17031)], [[`Code`](https://github.com/Z-Zheng/Changen)]
- (ChangeMask) ChangeMask: Deep Multi-task Encoder-Transformer-Decoder Architecture for Semantic Change Detection, ISPRS P&RS 2022. [[`Paper`](https://www.sciencedirect.com/science/article/pii/S0924271621002835)]
- (ChangeStar) Change is Everywhere: Single-Temporal Supervised Object Change Detection
in Remote Sensing Imagery, ICCV 2021. [[`Paper`](https://arxiv.org/abs/2108.07002)], [[`Project`](https://zhuozheng.top/changestar/)], [[`Code`](https://github.com/Z-Zheng/ChangeStar)]
- (ChangeOS) Building damage assessment for rapid disaster response with a deep object-based semantic change detection framework: from natural disasters to man-made disasters, RSE 2021. [[`Paper`](https://www.sciencedirect.com/science/article/pii/S0034425721003564)], [[`Code`](https://github.com/Z-Zheng/ChangeOS)]

### Segmentation
- FarSeg++: Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery, IEEE TPAMI 2023. [[`Paper`](https://ieeexplore.ieee.org/document/10188509)], [[`Code`](https://github.com/Z-Zheng/FarSeg)]
- Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery, CVPR 2020. [[`Paper`](https://arxiv.org/pdf/2011.09766.pdf)], [[`Code`](https://github.com/Z-Zheng/FarSeg)]
- Deep multisensor learning for missing-modality all-weather mapping, ISPRS P&RS 2021. [[`Paper`](https://www.sciencedirect.com/science/article/abs/pii/S0924271620303476)]
- FactSeg: Foreground Activation Driven Small Object Semantic Segmentation in Large-Scale Remote Sensing Imagery, TGRS 2021. [[`Paper`](https://www.researchgate.net/publication/353357122_FactSeg_Foreground_Activation_Driven_Small_Object_Semantic_Segmentation_in_Large-Scale_Remote_Sensing_Imagery)], [[`Code`](https://github.com/Junjue-Wang/FactSeg)]
- LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation, NeurIPS 2021 Datasets and Benchmarks. [[`Paper`](https://arxiv.org/pdf/2110.08733.pdf)], [[`Code/Dataset`](https://github.com/Junjue-Wang/LoveDA)]

### Hyperspectral Image Classification
- FPGA: Fast Patch-Free Global Learning Framework for Fully End-to-End Hyperspectral Image Classification, TGRS 2020. [[`Paper`](https://ieeexplore.ieee.org/document/9007624)], [[`Code`](https://github.com/Z-Zheng/FreeNet)]


## License
EVer is released under the [Apache License 2.0](https://github.com/Z-Zheng/ever/blob/master/LICENSE).

Copyright (c) Zhuo Zheng. All rights reserved.
