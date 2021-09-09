# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.qdot_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2,3"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SAVE_WEIGHTS_INTE = 4000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
METHOD = 'R'
ANCHOR_RATIOS = [1, 1 / 3., 3.]

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = 1 # IoU-Smooth L1

VERSION = 'RetinaNet_QDOT_v4_20210903'

"""
RetinaNet-R + IoU-Smooth L1

FLOPs: 1043869710;    Trainable params: 32906106

cls : diamond|| Recall: 0.6645161290322581 || Precison: 0.5727117194183062|| AP: 0.6165835776751639
F1:0.7355996929185904 P:0.9370199692780338 R:0.6054590570719603
mAP is : 0.6165835776751639


"""

