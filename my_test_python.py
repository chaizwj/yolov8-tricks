# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from ultralytics import RTDETR, YOLO
from ultralytics.cfg import TASK2DATA
from ultralytics.data.build import load_inference_source
from ultralytics.utils import (ASSETS, DEFAULT_CFG, DEFAULT_CFG_PATH, LINUX, MACOS, ONLINE, ROOT, WEIGHTS_DIR, WINDOWS,
                               checks, is_dir_writeable)
from ultralytics.utils.downloads import download
from ultralytics.utils.torch_utils import TORCH_1_9

#  测试自己加了模块之后的 yolov8的  结构 yml文件

CFG = 'ultralytics/cfg/models/rt-detr/rtdetr-l.yaml'
SOURCE = ASSETS / 'bus.jpg'


#  这个方法是用来前向传播的，就是测试这个模型是否可行， 也就是测试各个结构里的tensor输入输出的 数据维度 是否正确

def test_model_forward():
    """Test the forward pass of the YOLO model."""

    model = YOLO(CFG)
    model(source=None, imgsz=32, augment=True)  # also test no source and augment