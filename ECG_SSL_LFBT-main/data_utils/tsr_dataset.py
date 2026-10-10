# -*- coding: utf-8 -*-
"""W8 E3 (w8-tsr): TSR 官方四变体 pretext 数据集 (BobZwr/ReverseECG pretraining.py)。

官方 generate_reverse: 变体块 (original, spatial_reverse, temporal_reverse, ts_reverse)
配多标签 y=[0,0]/[0,1]/[1,0]/[1,1] (槽位=[时间反转?, 幅值反转?]), BCEWithLogitsLoss,
spatial_reverse=-(x-mu)+mu (mu=逐导联时间均值), temporal_reverse=x[:, ::-1]。

本仓适配(逐条入 runlog/W8/w8_hparams.csv):
  - 每条记录每次抽取均匀抽 1/4 变体(官方为 4 块拼接全量每 epoch; 步数对齐公共协议
    ~27.3k steps, 变体访问总量相应少于官方 4 倍);
  - 无 RRC-TO 等额外增强(官方无增强, pretext 即变换);
  - 偏差记录: 官方 snippet ts_reverse=spatial_reverse[::-1] 反的是记录轴(与其标签
    [1,1] 语义矛盾, 属笔误), 此处按标签意图实现双重反转=(幅值反转)[:, ::-1]。
返回 ([x_v, x_v], y2) 兼容 run_pt 批协议(双视图槽复用, flag=2 维目标)。
"""
import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def build_variant(x, k):
    """x:(8,2048) float32, k∈{0,1,2,3} -> (变体信号, y2=[时间反转?,幅值反转?])。

    y2 槽位与官方块顺序对齐: k=0 原图[0,0] / 1 幅值反转[0,1] / 2 时间反转[1,0]
    / 3 双重反转[1,1]。"""
    x = np.asarray(x, dtype=np.float32)
    if k == 0:
        y2 = (0.0, 0.0)
    elif k == 1:
        mu = x.mean(axis=-1, keepdims=True)
        x = -(x - mu) + mu
        y2 = (0.0, 1.0)
    elif k == 2:
        x = x[:, ::-1]
        y2 = (1.0, 0.0)
    else:
        mu = x.mean(axis=-1, keepdims=True)
        x = (-(x - mu) + mu)[:, ::-1]
        y2 = (1.0, 1.0)
    return np.ascontiguousarray(x), np.array(y2, dtype=np.float32)


class TSReverseDataset(Dataset):
    """NFH npy 目录 -> (views=[x_v,x_v], y2)。官方无增强, 原始信号直接进变体。"""

    def __init__(self, data_dir):
        self.files = sorted(glob.glob(str(Path(data_dir) / "**" / "*.npy"), recursive=True))
        if not self.files:
            raise FileNotFoundError(f"TSReverseDataset: {data_dir} 下无 npy")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(self.files[idx]).astype(np.float32, copy=False)
        k = int(np.random.randint(0, 4))
        x_v, y2 = build_variant(x, k)
        t = torch.from_numpy(x_v)
        return [t, t], torch.from_numpy(y2)
