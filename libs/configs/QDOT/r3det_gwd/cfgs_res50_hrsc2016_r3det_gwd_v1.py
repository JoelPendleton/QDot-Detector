# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.qdot_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1  # r3det only support 1
GPU_GROUP = '0,1,2,3'
NUM_GPU = len(GPU_GROUP.strip().split(','))
SAVE_WEIGHTS_INTE = 5000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'QDOT'
IMG_SHORT_SIDE_LEN = 500
IMG_MAX_LENGTH = 500
CLASS_NUM = 1

# model
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
NUM_REFINE_STAGE = 1

# sample
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 2.0
USE_IOU_FACTOR = False

GWD_TAU = 2.0
GWD_FUNC = tf.sqrt

VERSION = 'RetinaNet_QDOT_R3Det_GWD_v1_20210906'

"""
r3det+gwd (only refine stage) + sqrt tau=2

FLOPs: 1230766570;    Trainable params: 37059716

cls : diamond|| Recall: 0.862531017369727 || Precison: 0.3183733284484338|| AP: 0.7905563595811357
F1:0.8437365270725933 P:0.9364406779661016 R:0.7677419354838709
mAP is : 0.7905563595811357

"""