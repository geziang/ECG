# -*- coding: utf-8 -*-
"""dv_dataset.py — W6 Stage4 双视角数据集(整段 + R峰中心心拍片段)。

(y1,y2)=C1 同款双视图; 额外返回第三视图 beat(B,8,W)=R峰中心窗逐导联平均
(窗长 W 由全局中位 RR×1.2 定, 固定值便于组批), 与 (rr3(B,3), valid(B,))
=[mean_RR, SDNN, RMSSD] 全库 z-score 回归目标(H4 口径的 3 统计量版,
fs=204.8 秒单位)。R 峰缓存 = prep_rpeaks.py 产物(npz{stem→索引})。
峰数 <3 的样本 valid=0 且 beat 用零填充(训练时整行剔除)。
"""
import numpy as np
import torch

from data_utils.data_folder import ECGDatasetFolder
from data_utils.multi_view_data_injector import MultiViewDataInjector

FS = 204.8
W_BEAT = 205  # round(2048/12 median 峰 × 1.2) ≈ 205, 全局固定


def beat_view(raw, peaks, w=W_BEAT):
    """raw:(8,T) 原始未裁剪信号; peaks: R峰索引数组 -> (8,w) 坰心窗平均或 None。"""
    if peaks is None or len(peaks) < 3:
        return None
    half = w // 2
    segs = []
    for p in peaks:
        lo, hi = int(p) - half, int(p) + half + 1
        pad_lo, pad_hi = max(0, -lo), max(0, hi - raw.shape[1])
        seg = raw[:, max(lo, 0):min(hi, raw.shape[1])]
        if pad_lo or pad_hi:
            seg = np.pad(seg, ((0, 0), (pad_lo, pad_hi)), mode="edge")
        segs.append(seg)
    return np.mean(segs, axis=0).astype(np.float32)


def rr_stats(peaks):
    """peaks->(mean_RR, SDNN, RMSSD) 秒; 峰数<3 返回 None。"""
    if peaks is None or len(peaks) < 3:
        return None
    rr = np.diff(np.asarray(peaks, dtype=np.float64)) / FS
    return (float(rr.mean()), float(rr.std()), float(np.sqrt(np.mean(np.diff(rr) ** 2))))


class DVDataset:
    def __init__(self, root, transform1, transform2, rpeak_npz):
        self.base = ECGDatasetFolder(
            root, transform=MultiViewDataInjector([transform1, transform2]))
        cache = np.load(str(rpeak_npz), allow_pickle=True)
        self.rp = {str(k): np.asarray(v) for k, v in cache.items()}
        # 全库 rr3 统计(z-score 归一化, H4 口径); 峰数<3 不参与统计
        stats = [rr_stats(v) for v in self.rp.values()]
        stats = [s for s in stats if s is not None]
        arr = np.asarray(stats)
        self.mu, self.sd = arr.mean(axis=0), arr.std(axis=0) + 1e-8
        self.n_valid = len(stats)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        (v1, v2), _ = self.base[idx]
        from pathlib import Path
        key = Path(self.base.samples[idx][0]).stem
        peaks = self.rp.get(key)
        raw = np.load(self.base.samples[idx][0]).astype(np.float32)
        bv = beat_view(raw, peaks)
        rs = rr_stats(peaks)
        if bv is None or rs is None:
            beat = torch.zeros(8, W_BEAT)
            rr3 = torch.zeros(3)
            valid = 0.0
        else:
            beat = torch.from_numpy(bv)
            rr3 = torch.tensor((np.asarray(rs) - self.mu) / self.sd, dtype=torch.float32)
            valid = 1.0
        return (v1, v2, beat), (rr3, valid)
