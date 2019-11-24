# 处理VID数据集
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import sys
import os

import random
import cv2
import numpy as np
from torch.utils.data import Dataset

from pysot.utils.bbox import center2corner, Center
from pysot.datasets.anchor_target import AnchorTarget
from pysot.datasets.augmentation import Augmentation
from pysot.core.config import cfg

logger = logging.getLogger("global")
# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)

class VID_Dataset(object):
    def __init__(self, name, root, anno, start_idx):
        self.name = name
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.root = os.path.join(cur_path, '../../', root)
        self.anno = os.path.join(cur_path, '../../', anno)
        self.start_sequence = start_idx
        logger.info("loading " + name)
        with open(self.anno, 'r') as f:
            meta_data = json.load(f)
            meta_data = self._filter_zero(meta_data)

        for video in list(meta_data.keys()):
            for track in meta_data[video]:
                frames = meta_data[video][track]
                frames = list(map(int,
                              filter(lambda x: x.isdigit(), frames.keys())))
                frames.sort()
                meta_data[video][track]['frames'] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]

        for video in list(meta_data.keys()):
            if len(meta_data[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]
        self.labels = meta_data
        self.num = len(self.labels)
        self.videos = list(meta_data.keys())
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.{}.{}.jpg'
        self.pick = self.shuffle()

    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        logger.info("{} start-sequence {} select [{}/{}] path_format {}".format(
            self.name, self.start_sequence, self.num,
            self.num, self.path_format))

    def shuffle(self):
        #将数据集所有的视频序列打乱重排
        lists = list(range(self.start_sequence, self.start_sequence + self.num))
        pick = []
        while len(pick) < self.num:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num]


    def get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)
        image_path = os.path.join(self.root, video,
                                  self.path_format.format(frame, track, 'x'))
        image_anno = self.labels[video][track][frame]
        return image_path, image_anno


    def get_video_anno(self, video, track, frames, template_frame_index):
        # 返回一个视频序列的标注信息
        sequence_path = []
        sequence_anno = []

        # 随机抽取序列的 图片数目
        select_list = range(0, len(frames))
        if len(select_list) > cfg.TSA.SEQUENCE_NUM:
            random_array = range(template_frame_index + 1, template_frame_index + cfg.TSA.SEQUENCE_NUM)
        else:
            random_array = select_list
        for frame in random_array:
            search_frame = frames[frame]
            image_path, image_anno = self.get_image_anno(video, track, search_frame)
            sequence_path.append(image_path)
            sequence_anno.append(image_anno)

        return sequence_path, sequence_anno

    def get_positive_video(self, index):
        #一个跟踪序列，返回一个模板和搜索序列。
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info['frames']
        if len(frames) > cfg.TSA.SEQUENCE_NUM:
            template_frame_index = np.random.randint(0, len(frames)-cfg.TSA.SEQUENCE_NUM)
        else:
            template_frame_index = 0
        template_frame = frames[template_frame_index]

        return self.get_image_anno(video_name, track, template_frame), \
               self.get_video_anno(video_name, track, frames, template_frame_index)

class Input_Dataset(Dataset):
    def __init__(self, name):
        super(Input_Dataset, self).__init__()
        # size match
        desired_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) / \
                       cfg.ANCHOR.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
        if desired_size != cfg.TRAIN.OUTPUT_SIZE:
            raise Exception('size not match!')

        # create anchor target
        self.anchor_target = AnchorTarget()

        # create datasets
        self.dataset = []
        start_video = 0
        self.num = 0 # 图片序列数目

        # create train datasets
        if name == 'train':
            train_dataset = VID_Dataset(
                        name,
                        cfg.DATASET.VID.ROOT,
                        cfg.DATASET.VID.ANNO,
                        start_video
                    )
            self.dataset = train_dataset
            self.num += train_dataset.num
            train_dataset.log()

        # create val datasets
        if name == 'val':
            val_dataset = VID_Dataset(
                        name,
                        cfg.DATASET.VID.ROOT,
                        cfg.DATASET.VID.VALANNO,
                        start_video
            )
            self.dataset = val_dataset
            self.num += val_dataset.num
            val_dataset.log()

        # data augmentation
        self.template_aug = Augmentation(
                cfg.DATASET.TEMPLATE.SHIFT,
                cfg.DATASET.TEMPLATE.SCALE,
                cfg.DATASET.TEMPLATE.BLUR,
                cfg.DATASET.TEMPLATE.FLIP,
                cfg.DATASET.TEMPLATE.COLOR
            )
        self.search_aug = Augmentation(
                cfg.DATASET.SEARCH.SHIFT,
                cfg.DATASET.SEARCH.SCALE,
                cfg.DATASET.SEARCH.BLUR,
                cfg.DATASET.SEARCH.FLIP,
                cfg.DATASET.SEARCH.COLOR
            )


    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox


    def __len__(self):
        return self.num

    def __getitem__(self, index):
        # 直接随机获取VID数据集的一个图片序列，该序列只为一个目标的连续序列。当一个序列同时跟踪多个目标时，将一个序列拆分成多个序列。

        dataset = self.dataset
        index = dataset.pick[index]

        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        # neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random
        # neg值决定是否使用负样例对进行训练，在templtae_TSA阶段暂时不使用。
        neg = 0
        # 得到模板和一个跟踪序列
        template, sequence = dataset.get_positive_video(index)

        # 基于图片序列生成训练TSA的数据。
        searchs = []
        clss = []
        deltas = []
        delta_weights = []
        bboxs = []

        # get template image
        template_image = cv2.imread(template[0])

        # get template
        template_box = self._get_bbox(template_image, template[1])

        # template augmentation
        template, _ = self.template_aug(template_image,
                                        template_box,
                                        cfg.TRAIN.EXEMPLAR_SIZE,
                                        gray=gray)
        template = template.transpose((2, 0, 1)).astype(np.float32)
        # search_info
        for idx, search in enumerate(zip(sequence[0], sequence[1])):

            if idx < cfg.TSA.SEQUENCE_NUM:
                # get search image
                search_image = cv2.imread(search[0])

                # get bounding box
                search_box = self._get_bbox(search_image, search[1])

                # augmentation
                search, bbox = self.search_aug(search_image,
                                               search_box,
                                               cfg.TRAIN.SEARCH_SIZE,
                                               gray=gray)

                # get labels
                cls, delta, delta_weight, overlap = self.anchor_target(
                    bbox, cfg.TRAIN.OUTPUT_SIZE, neg)
                search = search.transpose((2, 0, 1)).astype(np.float32)

                searchs.append(search)
                clss.append(cls)
                deltas.append(delta)
                delta_weights.append(delta_weight)
                bboxs.append(np.array(bbox))


        return {
            'template': template,
            'searchs': searchs,
            'label_cls': clss,
            'label_loc': deltas,
            'label_loc_weight': delta_weights,
            'bbox': bboxs
        }
