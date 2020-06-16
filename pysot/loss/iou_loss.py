# Copyright (c) yeyi. All Rights Reserved.
'''
GIoU |  Linear IoU  | CIOU | DIOU are added by following
'''
import math
import torch
import torch.nn as nn

def compute_iou(pred, target):
    '''
    :param pred: 基于FCOS回归网络预测的 l r t b 值
    :param target: 真实的标注
    :return: IOU = |B_p and B_t| / |B_p or B_t|
    '''
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_area = (target_left + target_right) * \
                  (target_top + target_bottom)
    pred_area = (pred_left + pred_right) * \
                (pred_top + pred_bottom)
    w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect

    ious = (area_intersect + 1.0) / (area_union + 1.0)

    return ious

def compute_giou(pred, target):
    '''
    :param pred: 基于FCOS回归网络预测的 l r t b 值
    :param target: 真实的标注
    :return: GIOU = |B_p and B_t| / |B_p or B_t| - |C - B_p and B_t|/|C|
    C: 覆盖B_p 和 B_t的最小 Bounding box
    '''
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_area = (target_left + target_right) * \
                  (target_top + target_bottom)
    pred_area = (pred_left + pred_right) * \
                (pred_top + pred_bottom)

    w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
    g_w_intersect = torch.max(pred_left, target_left) + torch.max(
        pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
    g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)

    ac_uion = g_w_intersect * g_h_intersect + 1e-7
    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect
    ious = (area_intersect + 1.0) / (area_union + 1.0)
    gious = ious - (ac_uion - area_union) / ac_uion

    return gious

def compute_diou(pred, target):
    '''
    :param pred: 基于FCOS回归网络预测的 l r t b 值
    :param target: 真实的标注
    :return: DIOU = |B_p and B_t| / |B_p or B_t| - R_DIOU
    R_DIOU =
    '''
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_area = (target_left + target_right) * \
                  (target_top + target_bottom)
    pred_area = (pred_left + pred_right) * \
                (pred_top + pred_bottom)
    w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect

    ious = (area_intersect + 1.0) / (area_union + 1.0)
    # compute penalty term : DIOU 通过计算 两个B-box中心点之间的距离作为惩罚项
    cen_distance_x = target_right - target_left - pred_right + pred_left
    cen_distance_y = target_bottom - target_top - pred_bottom + pred_top
    inter_diag = cen_distance_x ** 2 + cen_distance_y ** 2

    g_w_intersect = torch.max(pred_left, target_left) + torch.max(
        pred_right, target_right)
    g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
    outer_diag = g_h_intersect**2 + g_w_intersect**2

    r_dious = inter_diag / outer_diag
    dious = ious - r_dious
    dious = torch.clamp(dious, min=-1.0, max=1.0)

    return dious

def compute_ciou(pred, target):
    '''
        :param pred: 基于FCOS回归网络预测的 l r t b 值
        :param target: 真实的标注
        :return: DIOU = |B_p and B_t| / |B_p or B_t| - R_DIOU
        R_DIOU =
        '''
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_area = (target_left + target_right) * \
                  (target_top + target_bottom)
    pred_area = (pred_left + pred_right) * \
                (pred_top + pred_bottom)
    w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect

    ious = (area_intersect + 1.0) / (area_union + 1.0)
    # compute penalty term : DIOU 通过计算 两个B-box中心点之间的距离作为惩罚项
    cen_distance_x = target_right - target_left - pred_right + pred_left
    cen_distance_y = target_bottom - target_top - pred_bottom + pred_top
    inter_diag = cen_distance_x ** 2 + cen_distance_y ** 2

    g_w_intersect = torch.max(pred_left, target_left) + torch.max(
        pred_right, target_right)
    g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
    outer_diag = g_h_intersect ** 2 + g_w_intersect ** 2

    r_dious = inter_diag / outer_diag

    w2 = target_left + target_right
    h2 = target_top + target_bottom
    w1 = pred_left + pred_right
    h1 = pred_top + pred_bottom

    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
    alpha = v / (1 - ious + v)

    cious = ious - r_dious - (alpha * v)
    cious = torch.clamp(cious, min=-1.0, max=1.0)
    return cious


class IOULoss(nn.Module):
    def __init__(self, loss_type="iou"):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, weight=None):
        # l t r b
        if self.loss_type == 'log_iou':
            ious = compute_iou(pred, target)
            losses = -torch.log(ious)
        elif self.loss_type == 'iou':
            ious = compute_iou(pred, target)
            losses = 1 - ious
        elif self.loss_type == 'giou':
            gious = compute_giou(pred, target)
            losses = 1 - gious
        elif self.loss_type == 'diou':
            dious = compute_diou(pred, target)
            losses = 1- dious
        elif self.loss_type == 'ciou':
            cious = compute_ciou(pred, target)
            losses = 1 -cious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.sum()