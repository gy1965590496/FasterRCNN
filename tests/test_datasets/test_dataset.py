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
from modules.datasets import CocoDataset
from modules.datasets import VOCDataSet
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import GroupSampler
from torch.utils.data.sampler import BatchSampler

def _create_ids_error_coco_json(json_name):
    image = {
        'id': 0,
        'width': 640,
        'height': 640,
        'file_name': 'fake_name.jpg',
    }

    annotation_1 = {
        'id': 1,
        'image_id': 0,
        'category_id': 0,
        'area': 400,
        'bbox': [50, 60, 20, 20],
        'iscrowd': 0,
    }

    annotation_2 = {
        'id': 1,
        'image_id': 0,
        'category_id': 0,
        'area': 900,
        'bbox': [100, 120, 30, 30],
        'iscrowd': 0,
    }

    categories = [{
        'id': 0,
        'name': 'car',
        'supercategory': 'car',
    }]

    fake_json = {
        'images': [image],
        'annotations': [annotation_1, annotation_2],
        'categories': categories
    }
    mmcv.dump(fake_json, json_name)


def test_coco_annotation_ids_unique():
    tmp_dir = tempfile.TemporaryDirectory()
    fake_json_file = osp.join(tmp_dir.name, 'fake_data.json')
    _create_ids_error_coco_json(fake_json_file)

    # test annotation ids not unique error
    with pytest.raises(AssertionError):
        CocoDataset(ann_file=fake_json_file, classes=('car', ), pipeline=[])

def test_voc_dataset():
    from transforms import Compose, ToTensor, RandomHorizontalFlip
    data_transform = {
        "train": Compose([ToTensor(),RandomHorizontalFlip(0.5)]),
        "val": Compose([ToTensor()])
    }
    train_dataset = VOCDataSet('/data', "2007", data_transform["train"], "train.txt")
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    # 统计所有图像高宽比例在bins区间中的位置索引
    group_ids = create_aspect_ratio_groups(train_dataset, k=3)
    # 每个batch图片从同一高宽比例区间中取
    train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, 8)
    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]) 
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_sampler=train_batch_sampler,
                                                pin_memory=True,
                                                num_workers=nw,
                                                collate_fn=train_dataset.collate_fn)
    for i, [imgaes, targets] in enumerate(train_data_loader):
        imgaes
        targets
        if i == 0:
            break


def test_coco_dataset():
    from transforms import Compose, ToTensor, RandomHorizontalFlip
    data_transform = {
        "train": Compose([ToTensor(), RandomHorizontalFlip(0.5)]),
        "val": Compose([ToTensor()])
    }
    train_dataset = CocoDataset(ann_file='/data/mmdet_coco/coco/annotations/instances_train2017.json',data_root='/data/mmdet_coco/coco/train2017/', transforms=data_transform["train"])
    batch_size = 8
    # 采样器的前置迭代器，自定义的
    train_sampler = GroupSampler(train_dataset, batch_size)   
    # pytroch内置采样器
    train_batch_sampler = BatchSampler(train_sampler, batch_size, drop_last=False) 
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]) 
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_sampler=train_batch_sampler,
                                                pin_memory=True,
                                                num_workers=nw,
                                                collate_fn=train_dataset.collate_fn)
    for i, [imgaes, targets] in enumerate(train_data_loader):
        imgaes
        targets
        if i == 0:
            break

if __name__ == "__main__":
    test_voc_dataset()