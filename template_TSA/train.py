# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
from torch.optim import lr_scheduler

from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.distributed import dist_init, DistModule, reduce_gradients,\
        average_reduce, get_rank, get_world_size
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.average_meter import AverageMeter
from pysot.utils.misc import describe, commit
from template_TSA.template_enhance import Template_Enhance
from template_TSA.datasets.dataset import Input_Dataset
from pysot.core.config import cfg


logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siam_ta tracking')
parser.add_argument('--cfg', type=str, default='/home/ubuntu/Desktop/Object_Track/SiamTrackers/experiments/siamrpn_r50_l234_dwxcorr/config.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_data_loader():

    logger.info("build train&val datasets")
    # train_dataset
    train_dataset = Input_Dataset(name = 'train')
    val_dataset = Input_Dataset(name ='val')
    logger.info("build train&val datasets done")

    train_sampler = None
    val_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TSA.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True,
                              sampler=train_sampler)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.TSA.BATCH_SIZE_EVAL,
                            num_workers=cfg.TRAIN.NUM_WORKERS,
                            pin_memory=True,
                            sampler=val_sampler)
    return train_loader, val_loader


def build_opt_lr(model):

    trainable_params = []

    # TSA机制，需要参与训练
    trainable_params += [{'params': model.template_mtsa.parameters(),
                          'lr': cfg.TSA.LR}]

    #SiamRPN++使用训练好的模型，不参与模型的训练
    for param in model.model.parameters():
        param.requires_grad = False
    for m in model.model.modules():
        m.eval()

    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.model.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    # if cfg.ADJUST.ADJUST:
    #     trainable_params += [{'params': model.model.neck.parameters(),
    #                           'lr': cfg.TSA.LR}]
    #
    # trainable_params += [{'params': model.model.rpn_head.parameters(),
    #                       'lr': cfg.TSA.LR}]


    optimizer = torch.optim.Adam(trainable_params, betas=[0.0, 0.9])
    # optimizer = torch.optim.SGD(trainable_params,
    #                             momentum=cfg.TRAIN.MOMENTUM,
    #                             weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    # 学习率的变化方案
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.8)
    scheduler.step(cfg.TRAIN.START_EPOCH)

    return optimizer, scheduler

# 进行训练的类
class Estimator():

    def __init__(self, train_loader, eval_loader, model, optimizer, scheduler, tb_writer):

        self._train_loader = train_loader
        self._eval_loader = eval_loader
        self._model = model
        # self._model_eval = Template_Enhance()

        # 优化器和学习率迭代方案
        self._optimizer = optimizer

        self._scheduler = scheduler
        self._tb_writer = tb_writer

        # 保留
        self._max_patience = 10 * cfg.TSA.VALIDATE_STEP
        self._best_value = None
        self._best_step = None

    def train(self):

        rank = get_rank()

        def is_valid_number(x):
            return not (math.isnan(x) or math.isinf(x) or x > 1e4)

        # world_size = get_world_size()

        if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and \
                get_rank() == 0:
            os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

        logger.info("model\n{}".format(describe(self._model.module)))

        eval_result = False
        epoch = cfg.TRAIN.START_EPOCH + 1
        while not eval_result and epoch <= cfg.TSA.MAX_ITERATIONS:

            for idx, data in enumerate(self._train_loader):

                outputs = self._model(data)
                loss = outputs['total_loss']

                start_time = time.time()
                tb_idx = idx
                if is_valid_number(loss.data.item()):
                    self._optimizer.zero_grad()
                    loss.backward()
                    reduce_gradients(self._model)

                    if rank == 0 and cfg.TRAIN.LOG_GRADS:
                        log_grads(self._model.module, self._tb_writer, tb_idx)

                    # clip gradient
                    clip_grad_norm_(self._model.parameters(), cfg.TRAIN.GRAD_CLIP)
                    self._optimizer.step()

                print("Step： %d  first_loss: %f Total_loss: %f  Speed:  %.0f examples per second" %
                      (epoch, outputs['first_loss'], loss.data.item(), cfg.TSA.BATCH_SIZE * cfg.TSA.TIME_STEP / (time.time() - start_time)))

                if epoch % cfg.TSA.MODEL_SAVE_STEP == 0 or epoch == cfg.TSA.MAX_ITERATIONS or epoch % cfg.TSA.VALIDATE_STEP == 0:
                    if get_rank() == 0:
                        torch.save(
                            {'epoch': epoch,
                             'state_dict': self._model.module.state_dict(),
                             'optimizer': self._optimizer.state_dict()},
                            cfg.TRAIN.SNAPSHOT_DIR + '/checkpoint_e%d.pth' % (epoch))
                    print('Save to checkpoint at step %d' % (epoch))

                if epoch % cfg.TSA.VALIDATE_STEP == 0:
                    if self.evaluate(epoch, 'loss'):
                        eval_result = True
                        break
                if epoch > cfg.TSA.MAX_ITERATIONS:
                    break
                # 更新迭代的次数
                epoch += 1
                if epoch % cfg.TSA.LR_UPDATE == 0:
                    self._scheduler.step(epoch)
                    cur_lr = self._scheduler.get_lr()
                    logger.info('epoch: {} cur_lr: {}'.format(epoch, cur_lr))



    def evaluate(self, global_step, stop_metric='loss'):

        # cfg.CUDA = torch.cuda.is_available()
        # device = torch.device('cuda' if cfg.CUDA else 'cpu')

        # # 加载训练好的模型进行评估。
        # model_path = cfg.TRAIN.SNAPSHOT_DIR + '/checkpoint_e%d.pth' % (epoch)
        # if os.path.exists(model_path):
        #     checkpoint = torch.load(model_path)  # modelpath是你要加载训练好的模型文件地址
        #     self._model_eval.load_state_dict(checkpoint['state_dict'])
        #     self._model_eval.eval().cuda()
        #     print('加载 epoch {} 成功！'.format(epoch))
        # 模型转为eval模式
        self._model.eval()
        print('第 {} 次迭代，模型进入评估模式成功！'.format(global_step))
        # 开始评估模型
        epoch = cfg.TRAIN.START_EPOCH # 0
        total_loss = 0
        while epoch < cfg.TSA.MAX_ITERATIONS_EVAL:

            for idx, data in enumerate(self._eval_loader):
                with torch.no_grad():
                    outputs = self._model(data)
                loss = outputs['total_loss']
                get_loss = loss.item()
                total_loss += get_loss
                print('Example %d loss: %f' %(epoch, get_loss))

                if epoch > cfg.TSA.MAX_ITERATIONS_EVAL:
                    break
                epoch += 1

        avg_loss = total_loss / cfg.TSA.MAX_ITERATIONS_EVAL
        print('val_loss: %f' % (avg_loss))
        self._tb_writer.add_scalar('loss', avg_loss)

        if stop_metric == 'loss':
            value = avg_loss
        else:
            value = avg_loss

        if (self._best_value is None) or \
                (value < self._best_value):
            self._best_value = value
            self._best_step = global_step

        should_stop = (global_step - self._best_step >= self._max_patience)
        if should_stop:
            print('Stopping... Best step: {} with {} = {}.' \
                  .format(self._best_step, stop_metric, self._best_value))
        self._model.train()
        return should_stop

def log_grads(model, tb_writer, tb_index):
    def weights_grads(model):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights

    grad, weights = weights_grads(model)
    feature_norm, rpn_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if 'feature' in k:
            feature_norm += _norm ** 2
        else:
            rpn_norm += _norm ** 2

        tb_writer.add_scalar('grad_all/'+k.replace('.', '/'),
                             _norm, tb_index)
        tb_writer.add_scalar('weight_all/'+k.replace('.', '/'),
                             w_norm, tb_index)
        tb_writer.add_scalar('w-g/'+k.replace('.', '/'),
                             w_norm/(1e-20 + _norm), tb_index)
    tot_norm = feature_norm + rpn_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    rpn_norm = rpn_norm ** 0.5

    tb_writer.add_scalar('grad/tot', tot_norm, tb_index)
    tb_writer.add_scalar('grad/feature', feature_norm, tb_index)
    tb_writer.add_scalar('grad/rpn', rpn_norm, tb_index)


def main():
    rank, world_size = dist_init()
    logger.info("init done")

    # load cfg
    cfg.merge_from_file(args.cfg)
    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global',
                             os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                             logging.INFO)

        logger.info("Version Information: \n{}\n".format(commit()))
        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    # create model
    model = Template_Enhance().cuda().train()
    dist_model = DistModule(model)

    # load pretrained SiamRPN++ model
    if cfg.TSA.MODELBUILD_PATH:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        backbone_path = os.path.join(cur_path, '../', cfg.TSA.MODELBUILD_PATH)
        load_pretrain(model.model, backbone_path)

    # create tensorboard writer
    if rank == 0 and cfg.TRAIN.LOG_DIR:
        tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
    else:
        tb_writer = None

    # build datasets loader
    train_loader, val_loader = build_data_loader()

    # build optimizer and lr_scheduler
    optimizer, scheduler = build_opt_lr(dist_model.module)

    logger.info(scheduler)
    logger.info("model prepare done")

    # start estimation
    # train(train_loader, dist_model, optimizer, scheduler, tb_writer)
    estimation = Estimator(train_loader, val_loader, dist_model, optimizer, scheduler, tb_writer)
    # estimation.evaluate(5000, 'loss')
    estimation.train()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    seed_torch(args.seed)
    main()
