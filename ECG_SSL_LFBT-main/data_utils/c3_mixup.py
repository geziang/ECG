# -*- coding: utf-8 -*-
"""c3_mixup.py — C3 准周期 MixUp 数据集 (主机B 自实现, HOSTS §四 P3)。

设计: 混合发生在增强前的原始信号上, 混合后同一信号独立过两次增强(与
MultiViewDataInjector 语义一致)。以概率 p 混合随机另一样本:
    x = (1-λ)·x_i + λ·roll(x_j, lag),  λ ~ U(0.1, 0.3)(小比例, 身份主导)
对齐版(align=1): lag = FFT 循环互相关 argmax(以 II 导联为相位基准),
把 x_j 的波峰对到 x_i 上(ECG 准周期; 普通 MixUp 峰位错叠≈噪声, 正是对照
要检验的差异)。对照版(align=0): 同代码路径仅 lag=0。
RNG 流与 B0 不同, 独立车道(同 N3/H3 先例, Δ 对本机锚点)。
"""
import random

import numpy as np

from data_utils.data_folder import ECGDatasetFolder


def _best_lag(a, b):
    """FFT 循环互相关: 返回 k 使 roll(b, k) 与 a 对齐(峰对峰)。a/b: (T,) 一维。"""
    n = a.shape[-1]
    corr = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n=n)
    return int(np.argmax(corr))


class MixUpDataset:
    """返回 ((v1, v2), mix_flag); v1/v2 为混合(或原样)信号的两视图。"""

    def __init__(self, root, transform1, transform2, prob=0.5, lam_lo=0.1,
                 lam_hi=0.3, align=True, align_lead=0):
        self.base = ECGDatasetFolder(root)   # 无 transform -> (np(8,T), y)
        self.t1, self.t2 = transform1, transform2
        self.prob = prob
        self.lam_lo, self.lam_hi = lam_lo, lam_hi
        self.align = align
        self.align_lead = align_lead

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]                # np.ndarray (8, T)
        flag = 0.0
        if self.prob > 0 and random.random() < self.prob:
            j = random.randrange(len(self.base))
            while j == idx:
                j = random.randrange(len(self.base))
            xj, _ = self.base[j]
            lam = random.uniform(self.lam_lo, self.lam_hi)
            if self.align:
                lag = _best_lag(x[self.align_lead], xj[self.align_lead])
                xj = np.roll(xj, lag, axis=-1)
            x = (1.0 - lam) * x + lam * xj
            flag = 1.0
        return (self.t1(x), self.t2(x)), flag
