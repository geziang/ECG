# -*- coding: utf-8 -*-
"""W7 E5: NSTDB 真实噪声训练期增强包装(默认不启用; run_pt --nstdb-aug > 0 时生效)。

口径对齐 W6 run_robustness_nstdb: 噪声源 data/nstdb/{bw,ma,em}.dat (2ch@360Hz→204.8Hz);
通道映射 lead->ch[lead%2]; SNR 逐记录逐导联(_match_snr 同式)。概率 p 下对样本注入
随机噪声类型与随机 SNR(默认 5~20dB), 其余样本原样——clean 主导, 噪声为辅的混合增强。
RNG: 全局 np.random(seeded by run_pt set_seed), 与既有随机增强同规范(训练增强不做逐样本确定性)。
"""
import numpy as np

_CACHE = {}


def _load(name):
    if name not in _CACHE:
        import wfdb
        from scipy.signal import resample_poly
        from pathlib import Path
        import sys
        root = Path(__file__).resolve().parents[1]
        rec = wfdb.rdrecord(str(root / "data/nstdb" / name))
        x = rec.p_signal.T.astype(np.float32)
        _CACHE[name] = resample_poly(x, up=2048, down=3600, axis=1)
    return _CACHE[name]


class NoiseInjector:
    """np (8,2048) float32 -> np (8,2048) float32; 概率 p 注入 NSTDB 噪声。"""

    def __init__(self, prob=0.5, snr_lo=5.0, snr_hi=20.0):
        self.prob = float(prob)
        self.snr_lo = float(snr_lo)
        self.snr_hi = float(snr_hi)

    def __call__(self, x):
        if np.random.random() >= self.prob:
            return x
        name = ("bw", "ma", "em")[np.random.randint(0, 3)]
        noise = _load(name)
        Ln = noise.shape[1]
        L = x.shape[-1]
        if Ln <= L:
            return x
        n = np.empty_like(x)
        for j in range(x.shape[0]):
            o = np.random.randint(0, Ln - L)
            n[j] = noise[j % 2, o:o + L]
        snr = np.random.uniform(self.snr_lo, self.snr_hi)
        px = (x.astype(np.float64) ** 2).mean(axis=-1, keepdims=True)
        pn = (n.astype(np.float64) ** 2).mean(axis=-1, keepdims=True)
        n = (n * np.sqrt(px / (pn * 10 ** (snr / 10.0) + 1e-12))).astype(np.float32)
        return x + n
