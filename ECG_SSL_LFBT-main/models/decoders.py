# -*- coding: utf-8 -*-
"""decoders.py — D7 掩码重建支路的轻量解码器。

输入 backbone 池化前的特征图 (B, 64ch, 64len), 经 5 级转置卷积上采样回 (B, 1, 2048)。
参数量 ~0.16M/导联(与 VGG16α=0.125 的 ~0.24M 同量级)。
"""
import torch
import torch.nn as nn


class SmallDecoder(nn.Module):
    def __init__(self, in_ch=64, out_len=2048):
        super().__init__()
        chs = [in_ch, 48, 32, 16, 8, 1]
        layers = []
        length = 64  # backbone 池化前特征长度 (2048/32)
        for i in range(5):
            layers.append(nn.ConvTranspose1d(chs[i], chs[i + 1], kernel_size=4,
                                             stride=2, padding=1))
            if i < 4:
                layers.append(nn.BatchNorm1d(chs[i + 1]))
                layers.append(nn.ReLU(inplace=True))
            length *= 2
        assert length == out_len, f"decoder 输出长度 {length} != {out_len}"
        self.net = nn.Sequential(*layers)

    def forward(self, feat_map):
        # feat_map: (B, in_ch, 64) -> (B, 1, 2048)
        return self.net(feat_map)
