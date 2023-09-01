# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import torch
import os
import sys
# import pytest
# sys.path.append('/home/gaoyun/Desktop/faster_rcnn with contrastive learning/modules/')
# sys.path.append("../../")
# print(sys.path)
from modules.trackers import DeepSort


def test_deep_sort():
    DeepSort()


if __name__ == "__main__":
    pass