# -*- coding: utf-8 -*-
"""h3_dataset.py — H3 患者身份不变性(轻量去相关臂)的三元组数据集 (主机B 自实现)。

设计(HOSTS §四 P1, 与 n3 同机同协议配对):
  - (y1, y2) = B0 完全一致的标准双视图(MultiViewDataInjector 两次独立增强),
    BT 主损失与 B0 车道同路径;
  - y_pair   = 同患者另一条记录的第三视图(独立增强), 仅用于 H3 去相关惩罚中
    估计"患者共享方向" mu_p = (z(y1)+z(y_pair))/2; 无同患者伙伴或未触发
    概率时为零张量且 flag=0(损失端按 flag 过滤, 不前向);
  - 有伙伴时以 prob(默认 0.3, 与 n3_prob03 同值)触发, 保持两臂数据流对称。

PTB-XL 患者多为单记录(同患者共现约 27%), batch 内随机共现≈0, 故与 N3 一样
必须在数据集层构造患者对。RNG 消耗序列与 B0 不同, 为独立车道(同 N3 先例,
Δ 对本机 B0 锚点计算)。患者映射来自 data/manifest.json(仅 in_pretrain=True)。
"""
import random
from collections import defaultdict
from pathlib import Path

import torch

from data_utils.data_folder import ECGDatasetFolder
from data_utils.multi_view_data_injector import MultiViewDataInjector
from data_utils.n3_dataset import load_patient_map


class H3TripletDataset:
    def __init__(self, root, transform1, transform2, patient_map, prob=0.3):
        # 双视图流(B0 同款): MultiViewDataInjector([t, t2])
        self.pair_base = ECGDatasetFolder(
            root, transform=MultiViewDataInjector([transform1, transform2]))
        # 单视图流: 给同患者伙伴记录做独立增强
        self.solo_base = ECGDatasetFolder(root, transform=transform1)
        self.prob = prob
        by_patient = defaultdict(list)
        for idx, s in enumerate(self.pair_base.samples):
            pid = patient_map.get(Path(s[0]).stem)
            if pid is not None:
                by_patient[pid].append(idx)
        self.partner_of = {}
        for idx, s in enumerate(self.pair_base.samples):
            pid = patient_map.get(Path(s[0]).stem)
            self.partner_of[idx] = [j for j in by_patient.get(pid, []) if j != idx]

    def __len__(self):
        return len(self.pair_base)

    def __getitem__(self, idx):
        (y1, y2), _ = self.pair_base[idx]        # B0 双视图
        cands = self.partner_of.get(idx, [])
        if self.prob > 0 and cands and random.random() < self.prob:
            y_pair, _ = self.solo_base[random.choice(cands)]
            flag = 1.0
        else:
            y_pair = torch.zeros_like(y1)
            flag = 0.0
        return (y1, y2, y_pair), flag
