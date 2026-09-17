# -*- coding: utf-8 -*-
"""sinc_conv.py — C1 Sinc 参数化带通前端 (主机B 自实现, HOSTS §四 P2)。

设计(Ravanelli SincNet 迁移到 1D ECG):
  - 第一层 Conv1d(1->C1, k=3) 替换为参数化带通滤波器组 SincConv1d(1->M, k=kernel);
  - 每个滤波器由可学习的 (f_low, bandwidth) 生成, 截止频率硬约束在 [0.5, 40] Hz;
  - 频带正则 sinc_band_penalty 把带宽压在 [bw_min, bw_max](防退化为全带≈普通卷积);
  - 滤波器 L2 归一化, 后接 BN+ReLU(与被替换层语义一致), 下层卷积输入改 M 通道。

数据有效采样率: prepare_data.py 把 10s@500Hz(5000 点)重采样为 2048 点,
故 fs_eff = 204.8 Hz(奈奎斯特 102.4 Hz), 0.5–40 Hz 约束完全可表示。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FS_EFF = 204.8  # 500Hz/5000点 -> 2048点 的有效采样率


class SincConv1d(nn.Module):
    """参数化 Sinc 带通卷积(单输入通道)。滤波器 = band-pass sinc × Hamming 窗。"""

    def __init__(self, out_channels, kernel_size=101, fs=FS_EFF,
                 f_min=0.5, f_max=40.0, bw_min=1.0, bw_max=15.0):
        super().__init__()
        assert kernel_size % 2 == 1, "Sinc 核长须为奇数"
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.fs = float(fs)
        self.f_min, self.f_max = float(f_min), float(f_max)
        self.bw_min, self.bw_max = float(bw_min), float(bw_max)
        # 初始化: f_low 均匀铺满可用区间, 带宽取中值
        f1_init = torch.linspace(f_min, f_max - bw_max, out_channels)
        self.low_hz = nn.Parameter(f1_init)
        self.band_hz = nn.Parameter(torch.full((out_channels,), (bw_min + bw_max) / 2))
        t = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32) / self.fs
        self.register_buffer("tgrid", t)
        # Hamming 窗(固定, 不可学习)
        w = torch.hamming_window(kernel_size, periodic=False)
        self.register_buffer("window", w)

    def bands(self):
        """有效 (f1, bw, f2), 全部硬约束在 [f_min, f_max] 内(单位 Hz)。"""
        f1 = torch.clamp(self.low_hz, self.f_min, self.f_max - self.bw_min)
        bw = torch.minimum(torch.clamp(self.band_hz, self.bw_min), self.f_max - f1)
        return f1, bw, f1 + bw

    def filters(self):
        f1, _, f2 = self.bands()
        t = self.tgrid[None, :]                        # (1, K)
        # 右半低通 sinc 族(与 SincNet 相同的 flip-shift 构造)
        def low_pass(fc):
            fc = fc[:, None]                           # (M, 1)
            return 2 * fc * torch.sinc(2 * fc * t) * self.fs
        bp = low_pass(f2) - low_pass(f1)               # (M, K)
        bp = bp * self.window[None, :]
        # L2 归一化(后接 BN, 幅度不敏感)
        bp = bp / (bp.norm(dim=1, keepdim=True) + 1e-8)
        return bp[:, None, :]                          # (M, 1, K)

    def forward(self, x):
        return F.conv1d(x, self.filters(), padding=self.kernel_size // 2)


def sinc_band_penalty(sinc):
    """C1 频带正则: 对原始参数(未 clamp 视图)惩罚越界带宽——
    带宽 >bw_max(趋向全带=普通卷积)或 <bw_min(退化成窄线谱)都受罚;
    f1 越出 [f_min, f_max-bw_min] 同罚。返回标量(Hz²量级)。"""
    f1 = sinc.low_hz
    bw = sinc.band_hz
    pen = (F.relu(bw - sinc.bw_max) + F.relu(sinc.bw_min - bw)).pow(2).mean()
    pen = pen + (F.relu(f1 - (sinc.f_max - sinc.bw_min)) + F.relu(sinc.f_min - f1)).pow(2).mean()
    return pen


def sinc_bands(sinc):
    """导出学到的截止频率(诊断/直方图用)。"""
    with torch.no_grad():
        f1, bw, f2 = sinc.bands()
        return dict(low=f1.tolist(), bw=bw.tolist(), high=f2.tolist())


def apply_sinc_frontend(backbone, M, kernel_size=101, fs=FS_EFF):
    """对 VGG16 实例做前端替换手术:
    model[0][0]: Conv1d(1->C1,k3)+BN+ReLU -> SincConv(1->M,k)+BN+ReLU
    model[0][1]: Conv1d(C1->C1,k3)       -> Conv1d(M->C1,k3)(其余结构不变)
    """
    from models.vgg_1d import conv_layer
    c1 = int(64 * backbone.alpha)
    backbone.model[0][0] = nn.Sequential(
        SincConv1d(M, kernel_size=kernel_size, fs=fs),
        nn.BatchNorm1d(M),
        nn.ReLU())
    backbone.model[0][1] = conv_layer(M, c1, 3, 1)
    return backbone
