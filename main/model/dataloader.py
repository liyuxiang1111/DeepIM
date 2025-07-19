# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
import dgl
from dgl.dataloading import GraphDataLoader


class GINDataLoader():
    """
    GIN模型的数据加载器，支持多种数据集划分方式

    功能：
    1. 支持10折交叉验证划分和随机划分两种方式
    2. 创建训练集和验证集的数据加载器
    3. 支持分层抽样保持类别分布

    参数:
    dataset: 图数据集
    batch_size: 批次大小
    device: 计算设备（CPU/GPU）
    collate_fn: 数据整理函数（可选）
    seed: 随机种子
    shuffle: 是否打乱数据
    split_name: 数据集划分方式 ('fold10' 或 'rand')
    fold_idx: 交叉验证的折索引（0-9）
    split_ratio: 随机划分时的训练集比例

    """
    def __init__(self,
                 dataset,
                 batch_size,
                 device,
                 collate_fn=None,
                 seed=0,
                 shuffle=True,
                 split_name='fold10',
                 fold_idx=0,
                 split_ratio=0.7):

        self.shuffle = shuffle # 是否打乱数据
        self.seed = seed # 随机种子
        # 设置数据加载参数：如果是CUDA设备，启用内存锁定
        self.kwargs = {'pin_memory': True} if 'cuda' in device.type else {}

        # 提取数据集的标签列表
        labels = [l for _, l in dataset]

        # 根据指定的划分方式创建索引
        if split_name == 'fold10':
            train_idx, valid_idx = self._split_fold10(
                labels, fold_idx, seed, shuffle)
        elif split_name == 'rand':
            train_idx, valid_idx = self._split_rand(
                labels, split_ratio, seed, shuffle)
        else:
            raise NotImplementedError()

        # 创建训练集和验证集的采样器
        train_sampler = SubsetRandomSampler(train_idx) # 训练集随机采样器
        valid_sampler = SubsetRandomSampler(valid_idx) # 验证集随机采样器

        # 创建训练集数据加载器
        self.train_loader = GraphDataLoader(
            dataset,
            sampler=train_sampler, # 使用采样器
            batch_size=batch_size,
            collate_fn=collate_fn, # 数据整理函数
            **self.kwargs) # 设备相关参数
        self.valid_loader = GraphDataLoader(
            dataset,
            sampler=valid_sampler,
            batch_size=batch_size,
            collate_fn=collate_fn,
            **self.kwargs)

    def train_valid_loader(self):
        return self.train_loader, self.valid_loader

    def _split_fold10(self, labels, fold_idx=0, seed=0, shuffle=True):
        '''
        10折分层交叉验证划分

        参数:
        labels: 数据标签列表
        fold_idx: 当前使用的折索引(0-9)
        seed: 随机种子
        shuffle: 是否打乱数据

        返回:
        train_idx: 训练集索引
        valid_idx: 验证集索引
        '''
        assert 0 <= fold_idx and fold_idx < 10, print(
            "fold_idx must be from 0 to 9.")

        # 创建分层K折交叉验证分割器
        skf = StratifiedKFold(n_splits=10, shuffle=shuffle, random_state=seed)
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):    # split(x, y)
            idx_list.append(idx)

        # 获取指定折的训练集和验证集索引
        train_idx, valid_idx = idx_list[fold_idx]

        # 打印数据集大小信息
        print(
            "train_set : test_set = %d : %d",
            len(train_idx), len(valid_idx))

        return train_idx, valid_idx

    def _split_rand(self, labels, split_ratio=0.7, seed=0, shuffle=True):
        '''
        随机比例划分数据集

        参数:
        labels: 数据标签列表
        split_ratio: 训练集比例
        seed: 随机种子
        shuffle: 是否打乱数据

        返回:
        train_idx: 训练集索引
        valid_idx: 验证集索引
        '''
        num_entries = len(labels) # 数据集大小
        indices = list(range(num_entries)) # 创建索引列表
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)
        split = int(math.floor(split_ratio * num_entries))
        train_idx, valid_idx = indices[:split], indices[split:]

        print(
            "train_set : test_set = %d : %d",
            len(train_idx), len(valid_idx))

        return train_idx, valid_idx