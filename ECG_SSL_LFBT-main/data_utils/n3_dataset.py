# -*- coding: utf-8 -*-
"""n3_dataset.py — N3 患者级正对: 同患者不同记录以概率 p 替换第二视图的来源。

两视图各自独立调用增强(与 MultiViewDataInjector 的两次独立增强语义一致),
仅当触发交换时第二视图的信号源换成同患者另一条记录。p=0 时等价于原双视图管线
(RNG 消耗序列不同,故为独立车道,不作 B0 逐位对照)。
患者映射来自 data/manifest.json(records[].patient_id, 仅 in_pretrain=True)。
"""
import json
import random
from collections import defaultdict
from pathlib import Path

from data_utils.data_folder import ECGDatasetFolder


class N3PairsDataset:
    def __init__(self, root, transform, patient_map, prob=0.0):
        self.base = ECGDatasetFolder(root, transform=transform)
        self.prob = prob
        by_patient = defaultdict(list)
        for idx, s in enumerate(self.base.samples):
            pid = patient_map.get(Path(s[0]).stem)
            if pid is not None:
                by_patient[pid].append(idx)
        self.partner_of = {}
        for idx, s in enumerate(self.base.samples):
            pid = patient_map.get(Path(s[0]).stem)
            self.partner_of[idx] = [j for j in by_patient.get(pid, []) if j != idx]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        view1, _ = self.base[idx]          # 独立增强 #1
        cands = self.partner_of.get(idx, [])
        if self.prob > 0 and cands and random.random() < self.prob:
            view2, _ = self.base[random.choice(cands)]
        else:
            view2, _ = self.base[idx]      # 独立增强 #2
        return (view1, view2), 0


def load_patient_map(manifest_path="data/manifest.json"):
    m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return {f"sample_{r['ecg_id']:05d}": r["patient_id"]
            for r in m["records"] if r.get("in_pretrain")}
