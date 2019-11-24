import torch
import torch.nn as nn
import torch.nn.functional as F
from pysot.models.model_builder import ModelBuilder
from template_TSA.template_tsa import Template_TSA, Template_MTSA
from pysot.utils.model_load import load_pretrain
from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
import numpy as np
from pysot.utils.anchor import Anchors
import cv2

class Template_Enhance(nn.Module):

    def __init__(self):
        super(Template_Enhance, self).__init__()
        # 模板增强网络
        self.template_mtsa = Template_MTSA(cfg.TSA.IN_CHANNELS, arfa=20)
        # siamRPN++主干网络
        self.model = ModelBuilder()
        # 存储池， 用于当前最关键的其余帧数。
        # 存储池的容量
        self.max_count = cfg.TSA.TIME_STEP

    def initial_state(self, z):
        zf = self.model.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.model.neck(zf)
        # 保存序列的第一帧模板， 对应三个通道。
        # self.init_template ： 设立初始化的模板
        # self.enhance :判断下一帧跟踪模板是否更新。
        # self.templtae : 初始化用于跟踪的模板。
        self.init_template = zf
        self.template = zf
        self.memory = [[], [], []]
        self.best_score = []
        self.enhance = False

    def get_feature(self, x):
        xf = self.model.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.model.neck(xf)
        return xf

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor


    def get_template_feature(self, pos_1, pos_2, xf):
        # 直接从搜索区域提取正样例模板特征
        p1 = xf[0][:, :, pos_1:pos_1 + 7, pos_2:pos_2 + 7]
        p2 = xf[1][:, :, pos_1:pos_1 + 7, pos_2:pos_2 + 7]
        p3 = xf[2][:, :, pos_1:pos_1 + 7, pos_2:pos_2 + 7]
        pos_template = [p1, p2, p3]
        return pos_template

    def update_memory(self, cls, loc, xf):
        # when train , use it
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # 针对每个batchsize 图像的不同，最佳匹配对象的位置不同，要做分开处理。
        batchsize = cls.size(0)
        channel_num = len(xf)
        score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
                     cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        # print(score_size)

        # hanning = np.hanning(score_size)
        anchors = self.generate_anchor(score_size)

        best_score = []
        pos_template = [[], [], []]
        cls = cls.chunk(batchsize, 0)
        loc = loc.chunk(batchsize, 0)
        for channel in range(channel_num):
            xf[channel] = xf[channel].chunk(batchsize, 0)
        for i in range(batchsize):

            cls_idx = cls[i]
            loc_idx = loc[i]

            # print(cls_idx.shape)
            # xf是一个多通道的特征
            xf_idx = []
            for channel in range(channel_num):
                xf_idx_channel = xf[channel][i]
                xf_idx.append(xf_idx_channel)

            score_idx = self._convert_score(cls_idx)
            pred_bbox_idx = self._convert_bbox(loc_idx, anchors)

            ## scale penalty
            # s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
            #              (sz(self.size[0]*scale_z, self.size[1]*scale_z)))
            # # aspect ratio penalty
            # r_c = change((self.size[0] / self.size[1]) /
            #              (pred_bbox[2, :] / pred_bbox[3, :]))
            # penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
            # pscore = penalty * score
            #
            # # window penalty
            # pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            #     self.window * cfg.TRACK.WINDOW_INFLUENCE
            # best_idx = np.argmax(pscore)

            best_idx = np.argmax(score_idx)
            best_score.append(score_idx[best_idx])

            # processing mask
            pos = np.unravel_index(best_idx, (5, score_size, score_size))
            delta_x, delta_y = pos[2], pos[1]

            # 得到当前batchsize 匹配的模板, 模板是多个通道的特征
            #根据分值决定当前模板是否提取进入存储池, 并且分三个通道
            if best_score[i] > 0.85:
                pos_template_idx = self.get_template_feature(delta_x, delta_y, xf_idx)
                for channel in range(channel_num):
                    pos_template[channel].append(pos_template_idx[channel])
            else:
                pos_template_idx = torch.zeros(1, 256, 7, 7).cuda()
                best_score[i] = 0
                for channel in range(channel_num):
                    pos_template[channel].append(pos_template_idx)

        #batchsize重新拼接
        for channel in range(channel_num):
            pos_template[channel] = torch.cat(pos_template[channel], dim = 0)

        # 更新存储池
        if len(self.memory[0]) < self.max_count:
            self.memory[0].append(pos_template[0])
            self.memory[1].append(pos_template[1])
            self.memory[2].append(pos_template[2])

            self.best_score.append(best_score)
            self.enhance = True
        else:
            # 对应每个batch,找出confidence_value的最小值，和位置。
            # 确定每个batch的最小值和位置
            best_score_numpy = np.array(self.best_score)
            locate_batchsize = best_score_numpy.argmin(axis=0)
            # 针对每个batch处理， 每个通道都要处理
            for channel in range(channel_num):
                for idx in range(batchsize):
                    if(best_score[idx] > self.best_score[locate_batchsize[idx]][idx]):
                        # 对分值最低的batch_idx进行更新
                        self.memory[channel][locate_batchsize[idx]][idx, :, :, :] = pos_template[channel][idx, :, :, :]
                        self.enhance = True


    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def track(self, x):
        # 更新模板
        if self.enhance:
            # 使用template的记忆力池增强初始化模板的特征。
            new_template = self.template_mtsa(self.init_template, self.memory, self.template)
            self.template = new_template
            self.enhance = False

        xf = self.get_feature(x)
        # data的格式为图片序列数，一个一个图片序列训练。# 得到当前cls最大值的位置，提取出新的模板。
        # print(xf[0].shape)
        cls, loc = self.model.rpn_head(self.template, xf)
        # print(cls.shape)
        # print(loc.shape)
        return {
            'cls': cls,
            'loc': loc
        }


    def forward(self, data):

        # 模板只有序列的第一张图片。
        template = data['template'].cuda()
        searchs = data['searchs']
        label_clss = data['label_cls']
        label_locs = data['label_loc']
        label_loc_weights = data['label_loc_weight']

        # 导入现成的训练好的SiamRPN++
        # cfg.merge_from_file(config_path)
        # cfg.CUDA = torch.cuda.is_available()
        # device = torch.device('cuda' if cfg.CUDA else 'cpu')
        # if ~model_path:
        #     raise RuntimeError("The path of model is not exist!")
        # else:
        #     #导入训练好的modelbuild()
        #     model = load_pretrain(self.model_build, model_path).cuda().eval()
        #     model.eval().to(device)

        # 图片序列第一张输入，初始化模型相关参数。
        self.initial_state(template)

        # 开始对图片序列进行匹配
        outputs = {
                   'total_loss': 0. ,
                   'cls_loss': 0. ,
                   'loc_loss': 0. ,
                   'first_loss':0. ,
                   }
        for idx, (search, label_cls, label_loc, label_loc_weight) in enumerate(zip(searchs, label_clss, label_locs, label_loc_weights)):
            #数据导入cuda
            search = search.cuda()
            label_loc = label_loc.cuda()
            label_cls = label_cls.cuda()
            label_loc_weight = label_loc_weight.cuda()
            #更新模板
            if self.enhance:
                # 使用template的记忆力池增强初始化模板的特征。
                new_template = self.template_mtsa(self.init_template, self.memory, self.template)
                self.template = new_template

            xf = self.get_feature(search)
            #data的格式为图片序列数，一个一个图片序列训练。# 得到当前cls最大值的位置，提取出新的模板。
            # print(xf[0].shape)
            cls, loc = self.model.rpn_head(self.template, xf)
            # print(cls.shape)
            # print(loc.shape)
            #更新存储池
            self.update_memory(cls=cls, loc=loc, xf=xf)

            # get current loss
            cls = self.log_softmax(cls)
            # print(cls.shape)
            cls_loss = select_cross_entropy_loss(cls, label_cls)
            # print(label_loc.shape)
            loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

            outputs['total_loss'] += cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                cfg.TRAIN.LOC_WEIGHT * loc_loss
            if idx == 0:
                outputs['first_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                cfg.TRAIN.LOC_WEIGHT * loc_loss
            outputs['cls_loss'] += cls_loss
            outputs['loc_loss'] += loc_loss

        # 考虑到不同图片序列的长度不一样，需要取均值损失
        outputs['total_loss'] = outputs['total_loss'] / len(searchs)
        outputs['cls_loss'] += outputs['cls_loss'] / len(searchs)
        outputs['loc_loss'] += outputs['loc_loss'] / len(searchs)
        return outputs