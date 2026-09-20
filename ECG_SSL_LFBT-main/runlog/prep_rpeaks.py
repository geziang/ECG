# -*- coding: utf-8 -*-
"""prep_rpeaks.py — T3 前置: 离线 R 峰检测缓存 (任务书 07 §4.2: R 峰优先 V5 离线检测并缓存)。

对 data/pt_pretrain 全部 .npy: wfdb.processing.gqrs_detect 检 V5(索引 6),
峰数 < 4(不足两个周期)时回退 lead II(索引 0), 仍不足则标空(训练时回退多段 mask)。
产出:
  data/pt_rpeaks.npz          — 键=文件名 stem, 值=R 峰索引(int64, 原始 2048 坐标)
  runlog/W1/rpeaks_stats.json — 检测成功率/中位峰数/回退统计(任务书 §4.3 必记诊断量)

用法(主机B DL env):
  python runlog/prep_rpeaks.py --data-dir data/pt_pretrain --out data/pt_rpeaks.npz
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def detect_one(sig, fs=204.8):
    """sig: (8, T) z-scored。返回 (rpeaks, lead_used)。"""
    import wfdb.processing as wp
    for lead_idx in (6, 0):  # V5 优先, II 回退
        try:
            pk = wp.gqrs_detect(sig[lead_idx].astype(np.float64), fs=fs)
        except Exception:
            pk = np.empty(0, dtype=np.int64)
        if len(pk) >= 4:
            return pk.astype(np.int64), lead_idx
    return np.empty(0, dtype=np.int64), -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, default=Path('data/pt_pretrain'))
    ap.add_argument('--out', type=Path, default=Path('data/pt_rpeaks.npz'))
    ap.add_argument('--fs', type=float, default=204.8,
                    help='有效采样率(2048 点 / 10 s)')
    args = ap.parse_args()

    files = sorted(args.data_dir.rglob('*.npy'))
    print(f'[prep_rpeaks] {len(files)} files, fs={args.fs}')
    cache = {}
    n_v5 = n_ii = n_fail = 0
    t0 = time.time()
    for i, f in enumerate(files):
        sig = np.load(f)
        pk, lead = detect_one(sig, fs=args.fs)
        cache[f.stem] = pk
        if lead == 6:
            n_v5 += 1
        elif lead == 0:
            n_ii += 1
        else:
            n_fail += 1
        if (i + 1) % 2000 == 0:
            print(f'  {i + 1}/{len(files)} ({time.time() - t0:.0f}s)', flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(args.out), **cache)

    counts = [len(v) for v in cache.values() if len(v) > 0]
    stats = dict(
        total=len(files), v5_detected=n_v5, ii_fallback=n_ii, failed=n_fail,
        success_rate=round((n_v5 + n_ii) / max(len(files), 1), 4),
        median_peaks=int(np.median(counts)) if counts else 0,
        min_peaks=int(np.min(counts)) if counts else 0,
        max_peaks=int(np.max(counts)) if counts else 0,
        wall_s=round(time.time() - t0, 1),
    )
    out_json = Path('runlog/W1/rpeaks_stats.json')
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(stats, indent=1), encoding='utf-8')
    print('[prep_rpeaks] stats:', json.dumps(stats))


if __name__ == '__main__':
    main()
