# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

DATASET_NAME = 'QDOT'
CLASS_NUM = 1
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
PIXEL_MEAN_ = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]
IMG_SHORT_SIDE_LEN = 500
IMG_MAX_LENGTH = 500

# data augmentation
IMG_ROTATE = True
RGB2GRAY = True
VERTICAL_FLIP = True
HORIZONTAL_FLIP = False
IMAGE_PYRAMID = False
