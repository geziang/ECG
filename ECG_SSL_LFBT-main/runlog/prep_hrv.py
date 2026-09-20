# -*- coding: utf-8 -*-
"""prep_hrv.py — H4 借口任务离线预计算: 每条预训练样本的 HRV 统计目标。

对 data/pt_pretrain/samples/*.npy 的 II 导联跑 wfdb xqrs R 峰检测
(有效采样率 204.8Hz = 500Hz/5000点 重采样至2048点), 计算 4 个统计量:
    [mean_RR_s, SDNN_s, RMSSD_s, HR_bpm]
R 峰 < 5 个或检测异常 -> invalid。输出 data/pt_hrv.npz:
    names(list[str]) stats(N,4) valid(N,) 以及全库归一化 mu/std(4,)
用法: python runlog/prep_hrv.py [--workers 16]
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

FS = 204.8  # 500Hz 信号重采样 5000->2048 的有效采样率
RR_CLIP = (0.3, 2.0)  # 生理范围外的 RR 视为检测噪声剔除


def hrv_of_npy(path_str):
    from wfdb.processing import gqrs_detect
    p = Path(path_str)
    try:
        x = np.load(p)[0].astype(np.float64)          # II 导联(导联序 0)
        x = (x - x.mean()) / (x.std() + 1e-8)
        r = np.asarray(gqrs_detect(x, fs=FS), dtype=np.int64)
        r = r[(r > 0) & (r < len(x))]
        rr = np.diff(r) / FS
        rr = rr[(rr >= RR_CLIP[0]) & (rr <= RR_CLIP[1])]
        if len(rr) < 4:
            return p.stem, None
        mean_rr = float(rr.mean())
        sdnn = float(rr.std())
        rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) >= 5 else float('nan')
        hr = 60.0 / mean_rr
        if not all(np.isfinite([mean_rr, sdnn, rmssd, hr])):
            return p.stem, None
        return p.stem, [mean_rr, sdnn, rmssd, hr]
    except Exception:
        return p.stem, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples-dir', default='data/pt_pretrain/samples')
    ap.add_argument('--out', default='data/pt_hrv.npz')
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()

    paths = sorted(str(p) for p in Path(a.samples_dir).glob('*.npy'))
    print(f'样本数 {len(paths)}, workers={a.workers}', flush=True)
    with Pool(a.workers) as pool:
        results = pool.map(hrv_of_npy, paths, chunksize=32)

    names, stats, valid = [], [], []
    bad = 0
    for name, s in results:
        names.append(name)
        if s is None:
            stats.append([0.0] * 4)
            valid.append(0.0)
            bad += 1
        else:
            stats.append(s)
            valid.append(1.0)
    stats = np.asarray(stats, dtype=np.float64)
    valid = np.asarray(valid, dtype=np.float32)
    v = stats[valid > 0.5]
    mu, std = v.mean(0), v.std(0) + 1e-8
    np.savez(a.out, names=np.array(names), stats=stats.astype(np.float32),
             valid=valid, mu=mu.astype(np.float32), std=std.astype(np.float32))
    print(f'有效 {int(valid.sum())}/{len(names)} (剔除 {bad}); '
          f'mu={mu.tolist()} std={std.tolist()} -> {a.out}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
