# Copyright (c) SenseTime. All Rights Reserved.

# add backbone efficientnet B0-B5
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.backbone.alexnet import alexnetlegacy, alexnet
from pysot.models.backbone.mobile_v2 import mobilenetv2
from pysot.models.backbone.resnet_atrous import resnet18, resnet34, resnet50
from pysot.models.backbone.efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5

BACKBONES = {
              'alexnetlegacy': alexnetlegacy,
              'mobilenetv2': mobilenetv2,
              'resnet18': resnet18,
              'resnet34': resnet34,
              'resnet50': resnet50,
              'alexnet': alexnet,
              'efficientnet-b0': efficientnet_b0,
              'efficientnet-b1': efficientnet_b1,
              'efficientnet-b2': efficientnet_b2,
              'efficientnet-b3': efficientnet_b3,
              'efficientnet-b4': efficientnet_b4,
              'efficientnet-b5': efficientnet_b5,
            }


def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
