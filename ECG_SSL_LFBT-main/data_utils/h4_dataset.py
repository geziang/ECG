# -*- coding: utf-8 -*-
"""h4_dataset.py — H4 HRV 借口任务数据集 (主机B 自实现, HOSTS §四 P3)。

(y1, y2)=B0 同款双视图; 额外返回 (hrv_target(4,), valid) —— 由
runlog/prep_hrv.py 生成的 data/pt_hrv.npz 提供全库归一化后的
[mean_RR, SDNN, RMSSD, HR] 回归目标。RNG 流与 B0 不同, 独立车道。
"""
import numpy as np
import torch

from data_utils.data_folder import ECGDatasetFolder
from data_utils.multi_view_data_injector import MultiViewDataInjector


class HRVDataset:
    def __init__(self, root, transform1, transform2, hrv_npz):
        self.base = ECGDatasetFolder(
            root, transform=MultiViewDataInjector([transform1, transform2]))
        d = np.load(hrv_npz, allow_pickle=True)
        names, stats, valid = d['names'], d['stats'], d['valid']
        mu, std = d['mu'], d['std']
        self.target = {}
        for i, n in enumerate(names):
            t = (stats[i] - mu) / std               # 全库 z-score 归一化
            self.target[str(n)] = (torch.tensor(t, dtype=torch.float32),
                                   float(valid[i]))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        (v1, v2), _ = self.base[idx]
        from pathlib import Path
        name = Path(self.base.samples[idx][0]).stem
        tgt, valid = self.target.get(name, (torch.zeros(4), 0.0))
        return (v1, v2), (tgt, valid)
