# -*- coding: utf-8 -*-
"""thesis-inc NFH 无标签预训练数据集(确定性流版)。

每条记录产出:
  v1/v2        整段两视图 (8,2048): NSTDB噪声->RRC-TO, 流=(seed,epoch,rid,g1/g2)
  template     I1/I2 心拍模板 (8,205): 全部有效缓存峰 W205 edge-pad 后 mean/median
  w1/w2        I3/I5 局部窗两视图 (Kmax=8,8,205) 零填充 + wmask (Kmax,)
               K=min(峰数,8); I3 中心=等距选定峰, I5 中心=floor((j+0.5)*2048/K);
               每窗每视图独立噪声流=(seed,epoch,rid,loc{j}v{1,2})
  rr3/valid    RR 三统计(全库z-score)与 峰数>=3 有效性

无效记录(峰数<3): template/rr 置零, wmask 全 False, valid=0; 整段主目标仍训练。
R 峰仅来自增强前无标签 NFH 缓存, 不访问任何下游数据。
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from tools.thesis_inc.det_aug import (AUG_IMPL_VERSION, global_view, local_view,
                                      make_rngs, select_evenly, fixed_centers, cut_window)

FS = 204.8
W_BEAT = 205
W_LO = -102  # [p-102, p+103)
KMAX = 8


def rr_stats(peaks: np.ndarray):
    """peaks -> (mean_RR, SDNN, RMSSD) 秒; 峰数<3 -> None。"""
    if peaks is None or len(peaks) < 3:
        return None
    rr = np.diff(np.asarray(peaks, dtype=np.float64)) / FS
    return (float(rr.mean()), float(rr.std()), float(np.sqrt(np.mean(np.diff(rr) ** 2))))


class NFHPTDataset(Dataset):
    def __init__(self, samples_dir, rpeak_npz, seed: int,
                 noise_prob: float = 0.5, rrc_params=(0.5, 1.0, 0.0, 0.5),
                 offset_range: str = "first_half",
                 template_mode: str | None = None,   # 'mean'|'median'|None
                 local_mode: str | None = None,      # 'peaks'|'fixed'|None
                 need_rr: bool = False,
                 limit: int | None = None):
        from pathlib import Path
        self.root = Path(samples_dir)
        paths = sorted(self.root.glob("*.npy"))
        if limit is not None:
            paths = paths[:limit]
        self.samples = [str(p) for p in paths]
        cache = np.load(str(rpeak_npz), allow_pickle=True)
        self.rp = {str(k): np.asarray(v) for k, v in cache.items()}
        self.seed = int(seed)
        self.epoch = 0  # 由训练循环在每 epoch 前设置(经 DataLoader 重建传入 worker)
        self.noise_prob = float(noise_prob)
        self.offset_range = offset_range
        self.rrc_params = list(rrc_params)
        self.template_mode = template_mode
        self.local_mode = local_mode
        self.need_rr = need_rr
        # RR 归一化统计: 仅由本批 NFH 训练记录计算(有效记录全体; 主进程算好随 pickle 下发)
        stats = [rr_stats(v) for v in (self.rp[Path(p).stem] for p in self.samples)]
        stats = [s for s in stats if s is not None]
        arr = np.asarray(stats)
        self.rr_mu, self.rr_sd = arr.mean(axis=0), arr.std(axis=0) + 1e-8
        self.n_valid_rr = len(stats)
        self.aug_impl_version = AUG_IMPL_VERSION

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        from pathlib import Path
        path = self.samples[idx]
        rid = Path(path).stem
        raw = np.load(path).astype(np.float32, copy=False)
        peaks = self.rp.get(rid)
        n_peaks = 0 if peaks is None else int(len(peaks))
        valid = n_peaks >= 3

        py1, np1 = make_rngs(self.seed, self.epoch, rid, "g1")
        py2, np2 = make_rngs(self.seed, self.epoch, rid, "g2")
        v1 = global_view(raw, py1, np1, self.noise_prob, self.rrc_params)
        v2 = global_view(raw, py2, np2, self.noise_prob, self.rrc_params)

        out = {"v1": torch.from_numpy(np.ascontiguousarray(v1)),
               "v2": torch.from_numpy(np.ascontiguousarray(v2)),
               "valid": torch.tensor(1.0 if valid else 0.0)}

        if self.template_mode is not None:
            if valid:
                segs = np.stack([cut_window(raw, int(p), W_BEAT, W_LO) for p in peaks])
                agg = np.mean(segs, axis=0) if self.template_mode == "mean" \
                    else np.median(segs, axis=0)
                tpl = agg.astype(np.float32)
            else:
                tpl = np.zeros((raw.shape[0], W_BEAT), dtype=np.float32)
            out["template"] = torch.from_numpy(np.ascontiguousarray(tpl))

        if self.local_mode is not None:
            w1 = np.zeros((KMAX, raw.shape[0], W_BEAT), dtype=np.float32)
            w2 = np.zeros((KMAX, raw.shape[0], W_BEAT), dtype=np.float32)
            mask = np.zeros(KMAX, dtype=bool)
            if valid:
                K = min(n_peaks, KMAX)
                if self.local_mode == "peaks":
                    sel = select_evenly(peaks, KMAX)
                    centers = np.asarray(peaks)[sel]
                else:
                    centers = fixed_centers(K, raw.shape[1])
                for j in range(K):
                    win = cut_window(raw, int(centers[j]), W_BEAT, W_LO)
                    pj1, n1j = make_rngs(self.seed, self.epoch, rid, f"loc{j}v1")
                    pj2, n2j = make_rngs(self.seed, self.epoch, rid, f"loc{j}v2")
                    w1[j] = local_view(win, pj1, n1j, self.noise_prob)
                    w2[j] = local_view(win, pj2, n2j, self.noise_prob)
                    mask[j] = True
            out["w1"] = torch.from_numpy(w1)
            out["w2"] = torch.from_numpy(w2)
            out["wmask"] = torch.from_numpy(mask)

        if self.need_rr:
            rs = rr_stats(peaks)
            if rs is None:
                rr3 = np.zeros(3, dtype=np.float32)
            else:
                rr3 = ((np.asarray(rs) - self.rr_mu) / self.rr_sd).astype(np.float32)
            out["rr3"] = torch.from_numpy(rr3)

        return out
