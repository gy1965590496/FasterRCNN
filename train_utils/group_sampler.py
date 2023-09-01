# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
from torch.utils.data import Sampler


class GroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        # flag标志由dataset在初始化时确定，详见customdataset
        # flag只有两个取值，根据ratio是否大于1，分为两组
        self.flag = dataset.flag.astype(np.int64)          

        self.group_sizes = np.bincount(self.flag)  # 对每组的数量进行计数，详见bincount的使用方法
        self.num_samples = 0 # 作为__len__的返回值

        # group_size不一定能确保被samples_per_gpu整除，因此需要向上取整
        # 比如分组0的数量是100个，分组1的数量是200个，samples_per_gpu为29
        # 那么num_samples = 116+203 = 319        
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(np.ceil(size / self.samples_per_gpu)
                                    ) * self.samples_per_gpu


    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0] # 获得同组的图片下标
            assert len(indice) == size
            np.random.shuffle(indice) # 打乱
            # 计算某一组中多出来的数量
            # 比如分组0的数量是100个，分组1的数量是200个，samples_per_gpu为29
            # num_extra = 116 - 100 = 16
            num_extra = int(np.ceil(size / self.samples_per_gpu) 
                            ) * self.samples_per_gpu - len(indice)
            
            # 从indice列表中随机取num_extra个数，并拼接在indice列表后面
            indice = np.concatenate(
                [indice, np.random.choice(indice, num_extra)])
            indices.append(indice)
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples