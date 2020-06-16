#2@: 完全使用one-stage Object Detector FCOS 定位目标
from pysot.models.head.rpn import RPN, DepthwiseXCorr
import torch
from torch import nn
import math
import torch.nn.functional as F
import pysot.core.config as cfg

class FCOS(nn.Module):
    def __init__(self):
        super(FCOS, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class DepthwiseFCOS(FCOS):
    def __init__(self, in_channels=256, out_channels=256):
        super(DepthwiseFCOS, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, 1, Center_ness=True)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4)

    def forward(self, z_f, x_f):
        cls, center_ness= self.cls(z_f, x_f)
        loc = torch.exp(self.loc(z_f, x_f))

        return cls, center_ness, loc

class MultiFCOS(FCOS):
    def __init__(self, in_channels, weighted=False):
        super(MultiFCOS, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('fcos'+str(i+2),
                    DepthwiseFCOS(in_channels[i], in_channels[i]))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.cen_weight = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs):
        cls = []
        cen = []
        loc = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            fcos = getattr(self, 'fcos'+str(idx))
            c, q, l = fcos(z_f, x_f)
            cls.append(c)
            cen.append(q)
            loc.append(l)

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            cen_weight = F.softmax(self.cen_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(cen, cen_weight), weighted_avg(loc, loc_weight)
        else:
            return avg(cls), avg(cen), avg(loc)

class DepthwiseGVFCOS(FCOS):
    def __init__(self, in_channels=256, out_channels=256):
        super(DepthwiseFCOS, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, 1, Center_ness=True)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4, Glide_vertex=True)

    def forward(self, z_f, x_f):
        cls, center_ness= self.cls(z_f, x_f)
        loc, glide_vertex, overlap_rate = self.loc(z_f, x_f)
        return cls, center_ness, loc, glide_vertex, overlap_rate

class CARHead(torch.nn.Module):
    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(CARHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = 2
        cls_tower = []
        bbox_tower = []
        for i in range(4):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )
        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        cls_tower = self.cls_tower(x)
        logits = self.cls_logits(cls_tower)
        centerness = self.centerness(cls_tower)
        bbox_reg = torch.exp(self.bbox_pred(self.bbox_tower(x)))

        return logits, centerness, bbox_reg


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale