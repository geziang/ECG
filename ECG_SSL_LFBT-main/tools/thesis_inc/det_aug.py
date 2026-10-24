# -*- coding: utf-8 -*-
"""thesis-inc 确定性增强流。

按任务书 §4.1: 由 (pt_seed, epoch, record_id, view_id) 固定局部/整段随机流,
辅助视图 RNG 与整段分开; 各配方同一 seed 的整段视图逐位一致。
变换逻辑与调用顺序复刻源代码 (data_utils/nstdb_aug.py NoiseInjector 先于
data_utils/augmentations.py rrc_to), 仅把全局 np.random/random 换成显式
RandomState/Random 实例; 抽样种类与顺序不变。实现版本 = det-aug-v1。
"""
import hashlib
import random

import numpy as np
from scipy.signal import resample

from data_utils.nstdb_aug import _load  # 复用噪声段加载与缓存(wfdb 360->204.8Hz)

AUG_IMPL_VERSION = "det-aug-v1"


def stream_seeds(material: str) -> tuple:
    """material -> (py_seed, np_seed); 两个独立流, blake2b 派生。"""
    b = hashlib.blake2b(material.encode("utf-8"), digest_size=16).digest()
    py = int.from_bytes(hashlib.blake2b(b"py|" + b, digest_size=8).digest(), "big") % (2 ** 31)
    np_ = int.from_bytes(hashlib.blake2b(b"np|" + b, digest_size=8).digest(), "big") % (2 ** 31)
    return py, np_


def make_rngs(pt_seed: int, epoch: int, record_id: str, view_tag: str):
    py_seed, np_seed = stream_seeds(f"{pt_seed}|{epoch}|{record_id}|{view_tag}")
    return random.Random(py_seed), np.random.RandomState(np_seed)


def draw_offset(rng: np.random.RandomState, Ln: int, L: int, offset_range: str) -> int:
    """复刻 nstdb_aug.draw_offset(first_half 语义), 显式 rng。"""
    if offset_range == "full":
        return int(rng.randint(0, Ln - L))
    if offset_range == "first_half":
        hi = Ln // 2 - L
        if hi < 1:
            return int(rng.randint(0, Ln - L))
        return int(rng.randint(0, hi))
    if offset_range == "last_half":
        lo, hi = Ln // 2, Ln - L
        if hi <= lo:
            return int(rng.randint(0, Ln - L))
        return int(rng.randint(lo, hi))
    raise ValueError(offset_range)


def inject_noise(x: np.ndarray, rng_py: random.Random, rng: np.random.RandomState,
                 prob: float = 0.5, snr_lo: float = 5.0, snr_hi: float = 20.0,
                 offset_range: str = "first_half") -> np.ndarray:
    """复刻 NoiseInjector.__call__ 的抽样顺序: 决定->类型->逐导联offset->SNR。"""
    if rng.random() >= prob:  # np.random.random() 对应位
        return x
    name = ("bw", "ma", "em")[int(rng.randint(0, 3))]
    noise = _load(name)
    Ln = noise.shape[1]
    L = x.shape[-1]
    if Ln <= L:
        return x
    n = np.empty_like(x)
    for j in range(x.shape[0]):
        o = draw_offset(rng, Ln, L, offset_range)
        n[j] = noise[j % 2, o:o + L]
    snr = rng.uniform(snr_lo, snr_hi)
    px = (x.astype(np.float64) ** 2).mean(axis=-1, keepdims=True)
    pn = (n.astype(np.float64) ** 2).mean(axis=-1, keepdims=True)
    n = (n * np.sqrt(px / (pn * 10 ** (snr / 10.0) + 1e-12))).astype(np.float32)
    return x + n


def _rrc(sig: np.ndarray, rng_py: random.Random, rng: np.random.RandomState,
         crop_ratio_low: float, crop_ratio_up: float) -> np.ndarray:
    """复刻 augmentations.rrc: py uniform 比例 -> np randint 起点 -> resample。"""
    sig_length = sig.shape[1]
    crop_length = int(rng_py.uniform(crop_ratio_low, crop_ratio_up) * sig_length)
    if sig_length == crop_length:
        start_point = 0
    elif crop_length == 0:
        return sig
    else:
        start_point = int(rng.randint(0, sig_length - crop_length))
    end_point = start_point + crop_length
    sig_crop = sig[:, start_point:end_point]
    return resample(sig_crop, sig_length, axis=1)


def _to(sig: np.ndarray, rng_py: random.Random, rng: np.random.RandomState,
        mask_ratio_low: float, mask_ratio_up: float) -> np.ndarray:
    """复刻 augmentations.to: py uniform 比例 -> np randint 起点 -> 置零。"""
    num_leads = sig.shape[0]
    sig_length = sig.shape[1]
    mask_length = int(rng_py.uniform(mask_ratio_low, mask_ratio_up) * sig_length)
    mask_start = int(rng.randint(0, sig_length - mask_length))
    if mask_length > 0:
        sig = np.array(sig, copy=True)
        sig[:, mask_start:(mask_start + mask_length)] = np.zeros([num_leads, mask_length])
    return sig


def global_view(raw: np.ndarray, rng_py: random.Random, rng: np.random.RandomState,
                noise_prob: float, rrc_params: list) -> np.ndarray:
    """整段视图: NSTDB 噪声 -> RRC-TO (源码 Compose 顺序)。"""
    x = inject_noise(raw, rng_py, rng, prob=noise_prob)
    x = _rrc(x, rng_py, rng, rrc_params[0], rrc_params[1])
    x = _to(x, rng_py, rng, rrc_params[2], rrc_params[3])
    return x.astype(np.float32)


def local_view(win: np.ndarray, rng_py: random.Random, rng: np.random.RandomState,
               noise_prob: float) -> np.ndarray:
    """局部视图: 仅噪声, 无时间裁剪/平移/翻转 (任务书 §5)。"""
    return inject_noise(win, rng_py, rng, prob=noise_prob).astype(np.float32)


def select_evenly(peaks: np.ndarray, k: int) -> np.ndarray:
    """超过 K 按有序峰索引等距确定性取 K: round(linspace(0, P-1, K))。
    P>=K+1 且 K<=8 时间距>1, 取整后必不重复(见上方推导)。"""
    if len(peaks) <= k:
        return np.arange(len(peaks))
    return np.round(np.linspace(0, len(peaks) - 1, k)).astype(int)


def fixed_centers(K: int, T: int = 2048) -> np.ndarray:
    """I5 固定窗中心: floor((j+0.5)*T/K)。"""
    return np.floor((np.arange(K) + 0.5) * T / K).astype(int)


def cut_window(raw: np.ndarray, center: int, w: int = 205, lo: int = -102) -> np.ndarray:
    """W205 窗 [c+lo, c+lo+w) edge padding (复刻 dv_dataset.beat_view 单窗)。"""
    a, b = int(center) + lo, int(center) + lo + w
    pad_lo, pad_hi = max(0, -a), max(0, b - raw.shape[1])
    seg = raw[:, max(a, 0):min(b, raw.shape[1])]
    if pad_lo or pad_hi:
        seg = np.pad(seg, ((0, 0), (pad_lo, pad_hi)), mode="edge")
    return seg.astype(np.float32)
