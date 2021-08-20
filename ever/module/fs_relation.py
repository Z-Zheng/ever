import torch
import torch.nn as nn
import ever as er
from .fpn import FPN, AssymetricDecoder
import torch.nn.functional as F


class FSRelation(nn.Module):
    """
    F-S Relation module in CVPR 2020 paper "Foreground-Aware Relation Network for Geospatial Object Segmentation in
     High Spatial Resolution Remote Sensing Imagery"
    """

    def __init__(self,
                 scene_embedding_channels,
                 in_channels_list,
                 out_channels,
                 scale_aware_proj=False):
        super(FSRelation, self).__init__()
        self.scale_aware_proj = scale_aware_proj

        if scale_aware_proj:
            self.scene_encoder = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(scene_embedding_channels, out_channels, 1),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels, out_channels, 1),
                ) for _ in range(len(in_channels_list))]
            )
        else:
            # 2mlp
            self.scene_encoder = nn.Sequential(
                nn.Conv2d(scene_embedding_channels, out_channels, 1),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 1),
            )
        self.content_encoders = nn.ModuleList()
        self.feature_reencoders = nn.ModuleList()
        for c in in_channels_list:
            self.content_encoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )
            self.feature_reencoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )

        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature, features: list):
        # [N, C, H, W]
        content_feats = [c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)]
        if self.scale_aware_proj:
            scene_feats = [op(scene_feature) for op in self.scene_encoder]
            relations = [self.normalizer((sf * cf).sum(dim=1, keepdim=True)) for sf, cf in
                         zip(scene_feats, content_feats)]
        else:
            # [N, C, 1, 1]
            scene_feat = self.scene_encoder(scene_feature)
            relations = [self.normalizer((scene_feat * cf).sum(dim=1, keepdim=True)) for cf in content_feats]

        p_feats = [op(p_feat) for op, p_feat in zip(self.feature_reencoders, features)]

        refined_feats = [r * p for r, p in zip(relations, p_feats)]

        return refined_feats


class FSRelationV2(nn.Module):
    def __init__(self,
                 scene_embedding_channels,
                 in_channels_list,
                 out_channels,
                 scale_aware_proj=False,
                 ):
        super().__init__()
        self.scale_aware_proj = scale_aware_proj

        if scale_aware_proj:
            self.scene_encoder = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(scene_embedding_channels, out_channels, 1),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels, out_channels, 1),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(True),
                ) for _ in range(len(in_channels_list))]
            )
            self.project = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                    nn.Dropout2d(p=0.1)
                ) for _ in range(len(in_channels_list))]
            )
        else:
            # 2mlp
            self.scene_encoder = nn.Sequential(
                nn.Conv2d(scene_embedding_channels, out_channels, 1),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 1),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(True),
            )
            self.project = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Dropout2d(p=0.1)
            )

        self.content_encoders = nn.ModuleList()
        self.feature_reencoders = nn.ModuleList()
        for c in in_channels_list:
            self.content_encoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )
            self.feature_reencoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )

        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature, features: list):
        # [N, C, H, W]
        content_feats = [c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)]
        if self.scale_aware_proj:
            scene_feats = [op(scene_feature) for op in self.scene_encoder]
            relations = [self.normalizer((sf * cf).sum(dim=1, keepdim=True)) for sf, cf in
                         zip(scene_feats, content_feats)]
        else:
            # [N, C, 1, 1]
            scene_feat = self.scene_encoder(scene_feature)
            relations = [self.normalizer((scene_feat * cf).sum(dim=1, keepdim=True)) for cf in content_feats]

        p_feats = [op(p_feat) for op, p_feat in zip(self.feature_reencoders, features)]

        refined_feats = [torch.cat([r * p, o], dim=1) for r, p, o in zip(relations, p_feats, features)]

        if self.scale_aware_proj:
            ffeats = [op(x) for op, x in zip(self.project, refined_feats)]
        else:
            ffeats = [self.project(x) for x in refined_feats]

        return ffeats


@er.registry.MODEL.register()
class FarSegHead(er.ERModule):
    def __init__(self, config):
        super(FarSegHead, self).__init__(config)
        self.fpn = FPN(**self.config.fpn)
        self.fs_relation = FSRelation(**self.config.fs_relation)
        self.fpn_decoder = AssymetricDecoder(**self.config.fpn_decoder)

    def forward(self, feature_list):
        fpn_feature_list = self.fpn(feature_list)
        coarsest_feature = feature_list[-1]
        scene_embedding = F.adaptive_avg_pool2d(coarsest_feature, 1)

        refined_fpn_feature_list = self.fs_relation(scene_embedding, fpn_feature_list)

        return self.fpn_decoder(refined_fpn_feature_list)

    def set_default_config(self):
        self.config.update(dict(
            fpn=dict(
                in_channels_list=(256, 512, 1024, 2048),
                out_channels=256,
            ),
            fs_relation=dict(
                scene_embedding_channels=2048,
                in_channels_list=(256, 256, 256, 256),
                out_channels=256,
                scale_aware_proj=True
            ),
            fpn_decoder=dict(
                in_channels=256,
                out_channels=256,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                classifier_config=dict(
                    scale_factor=4.0,
                    num_classes=1,
                    kernel_size=1
                )
            )
        ))
