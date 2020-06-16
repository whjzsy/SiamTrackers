"""
This file contains specific functions for computing losses of FCOS
file
"""
import torch
from torch import nn
import numpy as np
from pysot.loss.iou_loss import IOULoss
from pysot.loss.loss import select_cross_entropy_loss
from pysot.loss.focal_loss import FocalLoss
from pysot.core.config import cfg
# 安装fcos作为第三方库
# 参考Detectron2中RetinaNet.py使用的Focal Loss
# from fvcore.nn.focal_loss import sigmoid_focal_loss_jit
INF = 100000000

class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """
    def __init__(self, cfg):
        # # L_cls 的损失计算使用focal loss
        # self.cls_loss_func = SigmoidFocalLoss(
        #     cfg.FCOS.FOCAL_LOSS_GAMMA,
        #     cfg.FCOS.FOCAL_LOSS_ALPHA
        # )
        # self.cls_loss_func = FocalLoss(
        #     1,
        #     cfg.FCOS.FOCAL_LOSS_GAMMA,
        #     cfg.FCOS.FOCAL_LOSS_ALPHA
        # )
        self.cls_loss_func = FocalLoss(
            cfg.FCOS.FOCAL_LOSS_ALPHA,
            cfg.FCOS.FOCAL_LOSS_GAMMA,
            reduction="mean"
        )
        self.iou_loss_type = cfg.FCOS.IOU_LOSS_TYPE
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance

        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="mean")

    def prepare_targets(self, points, labels, gt_bbox):

        labels, reg_targets, num = self.compute_targets_for_locations(
            points, labels, gt_bbox
        )
        return labels, reg_targets, num

    def compute_targets_for_locations(self, locations, labels, gt_bbox):
        # reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        bboxes = gt_bbox
        # area = (bboxes[:,2] - bboxes[:,0] + 1) * (bboxes[:,3] - bboxes[:,1] + 1)
        labels = labels.view(625,-1)

        # 计算location坐标和真实边框之间的距离
        l = xs[:, None] - bboxes[:, 0][None].float()
        t = ys[:, None] - bboxes[:, 1][None].float()
        r = bboxes[:, 2][None].float() - xs[:, None]
        b = bboxes[:, 3][None].float() - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
        # print(reg_targets_per_im.size()) batchsize * 64 * 4
        s1 = reg_targets_per_im[:, :, 0] > 0.6*((bboxes[:,2]-bboxes[:,0])/2).float()
        s2 = reg_targets_per_im[:, :, 2] > 0.6*((bboxes[:,2]-bboxes[:,0])/2).float()
        s3 = reg_targets_per_im[:, :, 1] > 0.6*((bboxes[:,3]-bboxes[:,1])/2).float()
        s4 = reg_targets_per_im[:, :, 3] > 0.6*((bboxes[:,3]-bboxes[:,1])/2).float()
        is_in_boxes = s1*s2*s3*s4
        is_in_boxes = is_in_boxes.cpu()
        pos = np.where(is_in_boxes == 1)
        num = pos[0].shape[0]
        labels[pos] = 1
        # datasets 中labels为 1的部分对应了reg_targets_per_im负数部分
        return labels.permute(1,0).contiguous(), reg_targets_per_im.permute(1,0,2).contiguous(),num

    def select(self, position, keep_num=16):
        num = position[0].shape[0]
        if num <= keep_num:
            return position, num
        slt = np.arange(num)
        np.random.shuffle(slt)
        slt = slt[:keep_num]
        return tuple(p[slt] for p in position), keep_num

    def compute_centerness_targets(self, reg_targets):
        # reg_targets 中存在负数, centerness有负数 --> sqrt(负数) --> nan
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, labels, reg_targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor]) batchsize * 1 * 25 * 25
            box_regression (list[Tensor]) batchsize * 4 * 25 * 25
            centerness (list[Tensor])
            labels (list(BoxList) Batchsize * 1 * 25 * 25
            targets (list[BoxList]) batchsize * 4
        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        # print(reg_targets) value 不存在问题
        # print(locations)
        label_cls, reg_targets, num = self.prepare_targets(locations, labels, reg_targets)
        # print(reg_targets.size()) reg_targets中存在负数
        # print(reg_targets)
        box_regression_flatten = (box_regression.permute(0, 2, 3, 1).contiguous().view(-1, 4))
        # print(label_cls.size())  batchsize * 625
        # print(reg_targets.size()) batchsize * 625 * 4
        labels_flatten = (label_cls.view(-1))
        reg_targets_flatten = (reg_targets.view(-1, 4))
        centerness_flatten = (centerness.view(-1))

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        # print(labels_flatten.type())
        # print(box_cls.size())
        # print(labels_flatten.size())
        # print(labels_flatten)
        # print(box_cls)
        if cfg.FCOS.TYPE == 'CARHead':
            cls_loss = select_cross_entropy_loss(box_cls, labels_flatten)
        else:
            cls_loss = self.cls_loss_func(
                box_cls,
                labels_flatten
            )
        # print(centerness_flatten.size())
        # print(centerness_flatten)
        # 网络模型生成的centerness-branch 值有正 有负
        # print(reg_targets_flatten) 存在负数?? --> nan
        # print(reg_targets_flatten)
        if pos_inds.numel() > 0:

            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            # print(centerness_targets)
            # print(centerness_targets.size())
            # print(centerness_targets) 人为生成的cen-branch 标注 存在nan --> loss = nan
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator