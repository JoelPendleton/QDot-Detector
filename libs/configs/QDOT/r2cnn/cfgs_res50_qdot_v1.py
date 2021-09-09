# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.faster_rcnn_r50_fpn import *
from libs.configs._base_.datasets.qdot_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2,3"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 0.001 * BATCH_SIZE * NUM_GPU
SAVE_WEIGHTS_INTE = 4000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset

# model
NMS_IOU_THRESHOLD = RPN_NMS_IOU_THRESHOLD # in faster_rcnn_r50_fpn
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

VERSION = 'FPN_Res50D_QDOT_v1_20210903'

"""
R2CNN

FLOPs: 1235505473;    Trainable params: 41692732

rotation eval:
Writing diamond VOC resutls file
cls : diamond|| Recall: 0.9573200992555831 || Precison: 0.8101637967240655|| AP: 0.8907657711228334
F1:0.9457199070973091 P:0.9445544554455445 R:0.9468982630272953
mAP is : 0.8907657711228334

"""
