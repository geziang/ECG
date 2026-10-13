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



def rand_beat_view(raw, n_windows, rng, w=W_BEAT):
    """N1 反事实(JOINT N1): 与 beat_view 同窗长同窗数, 起点均匀随机, 无 R 峰对齐。

    rng: np.random.RandomState; 逐记录确定性由调用方用 crc32(stem) 播种。
    """
    T = raw.shape[1]
    if n_windows < 3 or T < w:
        return None
    starts = rng.randint(0, T - w + 1, size=int(n_windows))
    segs = [raw[:, s:s + w] for s in starts]
    return np.mean(segs, axis=0).astype(np.float32)


def build_rr_perm(valid_idx, seed=123):
    """N2 反事实(JOINT N2): valid_idx 内固定置换, 返回 idx->源 idx 映射。

    第 k 个有效样本拿第 perm[k] 个有效样本的 RR 统计; 自映射不排除
    (n≈3.5 万下占比可忽略, 属目标噪声一部分)。
    """
    if not valid_idx:
        return {}
    perm = np.random.RandomState(seed).permutation(len(valid_idx))
    return {valid_idx[k]: valid_idx[int(perm[k])] for k in range(len(valid_idx))}


class DVDataset:
    def __init__(self, root, transform1, transform2, rpeak_npz,
                 rand_window=False, rr_shuffle=False):
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
        self.rand_window = bool(rand_window)  # N1: 默认关=原行为
        if rr_shuffle:  # N2: idx->源idx 固定置换(顺序=base.samples, 种子123)
            from pathlib import Path as _P
            stats_by_idx = [rr_stats(self.rp.get(_P(self.base.samples[i][0]).stem))
                            for i in range(len(self.base))]
            self.rr_src = build_rr_perm(
                [i for i, s in enumerate(stats_by_idx) if s is not None])
        else:
            self.rr_src = None

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
        if self.rand_window and bv is not None:
            # N1: 窗数=真 R 峰数, 起点均匀随机, 逐记录确定性种子(crc32(stem))
            import zlib
            rng = np.random.RandomState(zlib.crc32(key.encode()) % (2 ** 31))
            bv = rand_beat_view(raw, len(peaks), rng)
        if bv is None or rs is None:
            beat = torch.zeros(8, W_BEAT)
            rr3 = torch.zeros(3)
            valid = 0.0
        else:
            beat = torch.from_numpy(bv)
            if self.rr_src is not None and self.rr_src.get(idx) is not None:
                # N2: 目标换成置换源记录的 RR 统计(掩码与心拍视图不动)
                rs_src = rr_stats(self.rp.get(
                    Path(self.base.samples[self.rr_src[idx]][0]).stem))
                if rs_src is not None:
                    rs = rs_src
            rr3 = torch.tensor((np.asarray(rs) - self.mu) / self.sd, dtype=torch.float32)
            valid = 1.0
        return (v1, v2, beat), (rr3, valid)
