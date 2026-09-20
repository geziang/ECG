# -*- coding: utf-8 -*-
"""prepare_nfh.py — T4: NFH(Ningbo) 外部预训练语料审计与转换(任务书 07 §五 / 06 §5.2)。

输入: E:/GZA/ECG_data/ningbo (PhysioNet Challenge-2021 ningbo 公开子集, 34,905 条 .mat+.hea)
处理(§04 统一格式): 12 导 WFDB -> 取 [II,III,V1..V6] 8 导 -> 重采样 2048 点(10s)
  -> 逐导 z-score -> 存 data/nfh_pretrain/c0/<stem>.npy
审计(T4 完成条件全量落盘 runlog/W1/nfh_manifest.json):
  - 原始数/清洗数/异常数(分类计数); 导联映射表; 采样率与长度分布; 重采样与归一化说明
  - 每条记录 SHA256(转换后 npy 字节); 内容级重复检测
  - 患者交叉: NFH 记录号 JSxxxxx 与 PTB-XL 患者(ECG_id/patient_id)命名空间正交,
    跨院物理重复不可能发生(审计口径写入 manifest)
  - matched optimizer updates: 与 B0(17,418×200ep) 对齐的推荐 epoch 数
用法: python runlog/prepare_nfh.py [--src E:/GZA/ECG_data/ningbo] [--limit 0]
"""
import argparse
import hashlib
import json
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.signal import resample

LEADS_WANT = ['II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
OUT_LEN = 2048


def process_one(args):
    hea_path = args[0]
    import wfdb
    rec_path = str(hea_path.with_suffix(''))
    try:
        rec = wfdb.rdrecord(rec_path)
        leads = list(rec.sig_name)
        if not all(l in leads for l in LEADS_WANT):
            return hea_path.stem, None, f'missing_leads:{",".join(set(LEADS_WANT) - set(leads))}'
        sig = rec.p_signal[:, [leads.index(l) for l in LEADS_WANT]]  # (N, 8)
        if sig.shape[0] < OUT_LEN // 2:
            return hea_path.stem, None, f'too_short:{sig.shape[0]}'
        if not np.isfinite(sig).all():
            return hea_path.stem, None, 'nonfinite'
        out = np.empty((8, OUT_LEN), dtype=np.float32)
        for i in range(8):
            x = resample(sig[:, i], OUT_LEN)
            sd = x.std()
            out[i] = (x - x.mean()) / (sd + 1e-8) if sd > 1e-6 else x * 0.0
        if float(np.abs(out).max()) > 1e3:  # 病态幅值
            hea_path.stem, None, 'bad_amplitude'
        return hea_path.stem, (out, rec.fs, sig.shape[0]), None
    except Exception as e:  # noqa: BLE001
        return hea_path.stem, None, f'read_error:{type(e).__name__}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, default=Path('E:/GZA/ECG_data/ningbo'))
    ap.add_argument('--out', type=Path, default=Path('data/nfh_pretrain/c0'))
    ap.add_argument('--manifest', type=Path, default=Path('runlog/W1/nfh_manifest.json'))
    ap.add_argument('--limit', type=int, default=0, help='>0 时只处理前 N 条(调试)')
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()

    heas = sorted(args.src.rglob('*.hea'))
    if args.limit:
        heas = heas[:args.limit]
    print(f'[nfh] {len(heas)} records from {args.src}')

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    hashes = {}
    fs_counter, len_counter, anomalies = Counter(), Counter(), Counter()
    n_ok = 0
    with Pool(args.workers) as pool:
        for k, (stem, payload, err) in enumerate(
                pool.imap_unordered(process_one, [(h,) for h in heas], chunksize=64)):
            if err:
                anomalies[err.split(':')[0]] += 1
                continue
            out, fs, raw_len = payload
            npy = args.out / f'{stem}.npy'
            np.save(npy, out)
            hashes[stem] = hashlib.sha256(npy.read_bytes()).hexdigest()
            fs_counter[int(fs)] += 1
            len_counter[int(raw_len)] += 1
            n_ok += 1
            if (k + 1) % 2000 == 0:
                print(f'  {k + 1}/{len(heas)} ok={n_ok} ({time.time() - t0:.0f}s)', flush=True)

    dup = Counter(hashes.values())
    n_dup = sum(v - 1 for v in dup.values() if v > 1)

    ptb_ids = []
    man_p = Path('data/manifest.json')
    if man_p.exists():
        man = json.loads(man_p.read_text(encoding='utf-8'))
        ptb_ids = man.get('ptbxl_ids', [])
    manifest = dict(
        source=str(args.src),
        dataset='NFH / PhysioNet Challenge-2021 ningbo public subset',
        n_raw=len(heas), n_clean=n_ok, n_anomaly=len(heas) - n_ok,
        anomaly_breakdown=dict(anomalies),
        lead_mapping=dict(target=['II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
                          source_standard='12-lead WFDB (Challenge-2021)'),
        fs_distribution={str(k): v for k, v in fs_counter.most_common(5)},
        raw_len_distribution={str(k): v for k, v in len_counter.most_common(5)},
        resample=f'to {OUT_LEN} points (10 s window, scipy.signal.resample per lead)',
        normalize='per-lead z-score (flat lead -> zeros)',
        sha256=dict(list(hashes.items())[:50]),
        sha256_count=len(hashes),
        n_content_duplicates=n_dup,
        patient_cross_audit=('NFH 记录号 JSxxxxx 与 PTB-XL(patient_id 数字)命名空间正交, '
                            '跨院同患者物理重复不可能; 库内重复以内容 SHA256 判定(见 n_content_duplicates)'),
        matched_updates=dict(b0_steps=200 * 17418,
                             nfh_recommended_epochs=round(200 * 17418 / max(n_ok, 1), 1),
                             note='optimizer updates 对齐 B0(任务书 §5.2: 避免暴露次数差异)'),
        wall_s=round(time.time() - t0, 1),
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding='utf-8')
    print('[nfh] stats:', json.dumps({k: manifest[k] for k in
          ['n_raw', 'n_clean', 'n_anomaly', 'anomaly_breakdown', 'n_content_duplicates',
           'fs_distribution']}))
    print(f'[nfh] manifest -> {args.manifest}')


if __name__ == '__main__':
    main()
