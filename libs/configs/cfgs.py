# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.faster_rcnn_r50_fpn import *
from libs.configs._base_.datasets.qdot_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 100
GPU_GROUP = "0,1,2,3"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 0.001 * BATCH_SIZE * NUM_GPU
SAVE_WEIGHTS_INTE = 2* 10000
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

VERSION = 'FPN_Res50D_QDOT_v1_20210911'

"""
R2CNN

FLOPs: 1235327155;    Trainable params: 41686582

cls : diamond|| Recall: 0.9977286805698947 || Precison: 0.9673673673673674|| AP: 0.9067456141608767
F1:0.9946319637268228 P:0.9936121986400165 R:0.9956638447243444
mAP is : 0.9067456141608767




"""
