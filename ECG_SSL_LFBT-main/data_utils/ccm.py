# -*- coding: utf-8 -*-
"""ccm.py — T3: 心动周期感知遮挡重建数据管线 (任务书 07 §四 / 06 §四)。

设计要点:
  - R 峰离线预检(见 runlog/prep_rpeaks.py, gqrs 检 V5, 失败回退 lead II)缓存为
    data/pt_rpeaks.npz: 每条记录一个 int 数组(原始 2048 坐标系)。
  - B3 CCM: 每个映射后 RR 周期内随机取连续 ~ratio(20%) 子段置零,
    窗口整体落在单个周期内部 => 永不横跨 R 峰; 有效周期不足时回退 B2 多段 mask。
  - B2 多段: 总 ~ratio(20%), 段长 5%~10% × 3~5 段, 随机位置。
  - 两视图独立(独立裁剪 + 独立 mask 位置); mask 跨导联同步(同一时间窗, 与 TO 一致)。
  - 裁剪坐标映射: rrc 为 crop[start:end] 后 resample 回原长,
    映射 r' = (r - start) * out_len / crop_len(FFT 重采样对带限信号近似线性)。
  - 返回 (masked_v1, masked_v2, orig_v1, orig_v2), (mask_v1, mask_v2):
    masked 为网络输入(遮挡后), orig 为重建目标(遮挡前原波形), mask 为 (1, T) 0/1。

诊断量(任务书 §4.3 必记):
  - dataset.stats: {'ccm_ok', 'ccm_fallback', 'views'} — 每 epoch 由训练循环打印;
  - R 峰检测成功率: runlog/W1/rpeaks_stats.json(prep 脚本产出);
  - mask 不跨 R 峰: tests/test_w1_t3.py 单测保证。
"""
import random
import numpy as np
import torch
from scipy.signal import resample

MIN_CYCLE_LEN = 30      # 映射后短于 30 点(≈0.15s)的周期视为无效
FALLBACK_COVER = 0.5    # 有效周期覆盖信号 < 50% 时整视图回退多段 mask


def crop_with_params(sig, crop_ratio_low, crop_ratio_up):
    """与 augmentations.rrc 完全同逻辑的裁剪, 但返回 (crop 后信号, start, crop_len)。"""
    sig_length = sig.shape[1]
    crop_length = int(random.uniform(crop_ratio_low, crop_ratio_up) * sig_length)
    if sig_length == crop_length:
        start_point = 0
    elif crop_length == 0:
        return sig, 0, sig_length
    else:
        start_point = np.random.randint(0, sig_length - crop_length)
    return sig[:, start_point:start_point + crop_length], start_point, crop_length


def map_rpeaks(rpeaks, start, crop_len, out_len, sig_len):
    """原始坐标 R 峰 -> 裁剪重采样后坐标。"""
    if rpeaks is None or len(rpeaks) == 0:
        return np.empty(0, dtype=np.int64)
    m = (np.asarray(rpeaks, dtype=np.float64) - start) * (out_len / max(crop_len, 1))
    m = np.rint(m).astype(np.int64)
    return m[(m > 0) & (m < out_len)]


def ccm_time_mask(out_len, rpeaks_mapped, ratio=0.2):
    """周期内遮挡: 每个有效周期 [R_k, R_{k+1}) 内取连续 ratio 长度随机子窗。返回 0/1 mask。

    窗口 ⊂ 周期 => 不跨 R 峰(构造保证)。有效周期总覆盖 < FALLBACK_COVER 时返回 None。
    """
    mask = np.zeros(out_len, dtype=np.float32)
    covered = 0
    for k in range(len(rpeaks_mapped) - 1):
        a, b = int(rpeaks_mapped[k]), int(rpeaks_mapped[k + 1])
        cyc = b - a
        if cyc < MIN_CYCLE_LEN:
            continue
        mlen = max(1, int(round(ratio * cyc)))
        mstart = a + np.random.randint(0, cyc - mlen + 1)
        mask[mstart:mstart + mlen] = 1.0
        covered += cyc
    if covered < FALLBACK_COVER * out_len:
        return None
    return mask


def multiseg_time_mask(out_len, ratio=0.2, seg_lo=0.05, seg_hi=0.10):
    """B2 多段随机 mask: 总 ~ratio, 段数自适应(总比/段长)。"""
    mask = np.zeros(out_len, dtype=np.float32)
    budget = int(round(ratio * out_len))
    guard = 0
    while budget > 0 and guard < 100:
        seg_len = max(8, int(round(random.uniform(seg_lo, seg_hi) * out_len)))
        seg_len = min(seg_len, budget)
        s = np.random.randint(0, out_len - seg_len + 1)
        before = mask[s:s + seg_len].sum()
        mask[s:s + seg_len] = 1.0
        budget -= int(seg_len - before)
        guard += 1
    return mask


class CCMDataset(torch.utils.data.Dataset):
    """B2/B3 遮挡重建数据集(包装 ECGDatasetFolder 的文件清单)。

    参数:
      root: data/pt_pretrain
      rpeak_npz: data/pt_rpeaks.npz(键=文件名 stem)
      style: 'ccm'(B3) 或 'multiseg'(B2)
      ratio: 目标遮挡比例(默认 0.2)
      crop: (lo, up) RRC 裁剪参数(默认 0.5,1.0 = 与 B0 同分布; TO 随机遮挡由本管线替代)
    返回:
      ((m1, m2, o1, o2), (mask1, mask2)) — 全部 float32 tensor。
    """

    def __init__(self, root, rpeak_npz, style='ccm', ratio=0.2,
                 crop=(0.5, 1.0), num_leads=8):
        from data_utils.data_folder import ECGDatasetFolder
        self.base = ECGDatasetFolder(root)
        self.style = style
        self.ratio = float(ratio)
        self.crop = tuple(crop)
        self.num_leads = num_leads
        self.rp = {}
        if style == 'ccm':  # multiseg 不需要 R 峰缓存
            cache = np.load(str(rpeak_npz), allow_pickle=True)
            self.rp = {str(k): np.asarray(v) for k, v in cache.items()}
        self.init_fallback = None
        self.stats = {'ccm_ok': 0, 'ccm_fallback': 0, 'views': 0}
        if style == 'ccm' and len(self.base.samples):
            # 诊断(任务书 §4.3): 抽样 256 条估计周期覆盖不足的回退率。
            # 训练期逐视图计数发生在 worker 进程, 不回传主进程 -> 只报此 init 估计。
            import random as _rnd
            from pathlib import PurePath as _PP
            for _ix in _rnd.sample(range(len(self.base.samples)),
                                   min(256, len(self.base.samples))):
                _path, _ = self.base.samples[_ix]
                self._one_view(np.load(_path), _PP(str(_path)).stem)
            self.init_fallback = round(self.stats['ccm_fallback']
                                       / max(self.stats['views'], 1), 4)
            self.stats = {'ccm_ok': 0, 'ccm_fallback': 0, 'views': 0}

    def __len__(self):
        return len(self.base.samples)

    def _one_view(self, raw, key):
        sig = np.array(raw, copy=True)
        out_len = sig.shape[1]
        sig, start, crop_len = crop_with_params(sig, self.crop[0], self.crop[1])
        sig = resample(sig, out_len, axis=1)
        mask = None
        if self.style == 'ccm':
            rp = map_rpeaks(self.rp.get(key), start, crop_len, out_len, sig.shape[1])
            mask = ccm_time_mask(out_len, rp, self.ratio)
            if mask is None:
                self.stats['ccm_fallback'] += 1
            else:
                self.stats['ccm_ok'] += 1
        if mask is None:  # multiseg 或 CCM 回退
            mask = multiseg_time_mask(out_len, self.ratio)
        orig = sig
        masked = sig * (1.0 - mask)[None, :]
        self.stats['views'] += 1
        return (torch.tensor(masked).float(), torch.tensor(orig).float(),
                torch.tensor(mask).float().unsqueeze(0))  # mask: (1, T)

    def __getitem__(self, idx):
        path, _ = self.base.samples[idx]
        raw = np.load(path)
        # DatasetFolder 存的是 str 路径, 无 .stem 属性 —— 统一用 PurePath 提取文件名主干
        from pathlib import PurePath
        key = PurePath(str(path)).stem
        m1, o1, k1 = self._one_view(raw, key)
        m2, o2, k2 = self._one_view(raw, key)   # 两视图独立(独立裁剪+独立mask)
        return (m1, m2, o1, o2), (k1, k2)
