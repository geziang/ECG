# -*- coding: utf-8 -*-
"""W3 B-3: 主机B NFH 97 条 non-finite 剔除明细重扫与对账。

背景: B 机 W1 制品 runlog/W1/nfh_manifest.json(ECG-main 工作树, 未推送) 记录
n_raw=34905 / n_clean=34808 / anomaly_breakdown={"nonfinite": 97}, 但当时只落计数未落名单;
A 机同库复扫零非有限(runlog/W2/nfh_manifest_reconciliation.md)。本脚本用与 B 当时
prepare_nfh.py 完全相同的读取与检测路径(wfdb.rdrecord -> 8导 p_signal -> np.isfinite,
重采样之前), 对同一原始数据全量重扫, 确定性复得 97 条名单并逐条诊断分歧根源。

输出(runlog/W3/):
  nfh_exclusion_reconciliation.csv   97 条: 记录名/非有限导联/NaN与Inf计数/原始长度/重采样后是否有限
  nfh_manifest_hostB_w3.json         B 口径 W3 版 manifest(含 excluded_records 全名单+复扫元数据)
红线: 只读原始数据; 不估算; 计数与 W1 manifest 对不上则 FAIL 退出, 交付物不落盘。
"""
import csv
import json
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(r"E:\GZA\ECG_data\ningbo")
W1_MANIFEST = Path(r"E:\GZA\ECG-main\ECG_SSL_LFBT-main\runlog\W1\nfh_manifest.json")
OUT_CSV = ROOT / "runlog" / "W3" / "nfh_exclusion_reconciliation.csv"
OUT_MANIFEST = ROOT / "runlog" / "W3" / "nfh_manifest_hostB_w3.json"
LEADS_WANT = ["II", "III", "V1", "V2", "V3", "V4", "V5", "V6"]
OUT_LEN = 2048


def scan_one(hea_path: str):
    """与 B 机 prepare_nfh.py process_one 相同的读取/导联/检测路径; 仅诊断更细。"""
    import wfdb
    from scipy.signal import resample
    hea = Path(hea_path)
    rec_path = str(hea.with_suffix(""))
    try:
        rec = wfdb.rdrecord(rec_path)
        leads = list(rec.sig_name)
        if not all(l in leads for l in LEADS_WANT):
            return {"stem": hea.stem, "stage": "missing_leads",
                    "detail": ",".join(sorted(set(LEADS_WANT) - set(leads)))}
        sig = rec.p_signal[:, [leads.index(l) for l in LEADS_WANT]]  # (N, 8) 原始, 未重采样
        if sig.shape[0] < OUT_LEN // 2:
            return {"stem": hea.stem, "stage": "too_short", "detail": str(sig.shape[0])}
        finite_mask = np.isfinite(sig)
        if finite_mask.all():
            return {"stem": hea.stem, "stage": "ok", "detail": ""}
        bad_leads = [LEADS_WANT[j] for j in range(8) if not finite_mask[:, j].all()]
        n_nan = int(np.isnan(sig).sum())
        n_inf = int(np.isinf(sig).sum())
        # 分歧根源诊断: 重采样(与训练制品同路径)后是否仍非有限
        still_nonfinite_after_resample = False
        for j in range(8):
            if finite_mask[:, j].all():
                continue
            y = resample(sig[:, j], OUT_LEN)
            if not np.isfinite(y).all():
                still_nonfinite_after_resample = True
        return {"stem": hea.stem, "stage": "nonfinite",
                "detail": f"leads={','.join(bad_leads)};nan={n_nan};inf={n_inf};"
                          f"raw_len={sig.shape[0]};fs={int(rec.fs)};"
                          f"nonfinite_after_resample={int(still_nonfinite_after_resample)}"}
    except Exception as e:  # noqa: BLE001
        return {"stem": hea.stem, "stage": f"read_error:{type(e).__name__}", "detail": ""}


def main():
    w1 = json.loads(W1_MANIFEST.read_text(encoding="utf-8"))
    expect_n = w1["anomaly_breakdown"].get("nonfinite")
    heas = sorted(str(p) for p in SRC.rglob("*.hea"))
    print(f"[scan] {len(heas)} .hea under {SRC}; W1 manifest expects nonfinite={expect_n}")

    t0 = time.time()
    rows = []
    counts = Counter()
    with Pool(12) as pool:
        for i, r in enumerate(pool.imap_unordered(scan_one, heas, chunksize=64)):
            counts[r["stage"]] += 1
            if r["stage"] == "nonfinite":
                rows.append(r)
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(heas)} ({time.time() - t0:.0f}s) "
                      f"nonfinite so far={counts['nonfinite']}", flush=True)
    rows.sort(key=lambda r: r["stem"])

    print(f"[scan] done in {time.time() - t0:.0f}s: {dict(counts)}")
    n_repro = counts["nonfinite"]
    ok = (n_repro == expect_n == 97 and counts["ok"] == w1["n_clean"]
          and len(heas) == w1["n_raw"])
    print(f"[check] reproduced nonfinite={n_repro} vs W1 manifest {expect_n}; "
          f"ok={counts['ok']} vs n_clean={w1['n_clean']}; raw={len(heas)} vs {w1['n_raw']}")

    if not ok:
        raise SystemExit("FAIL: 重扫计数与 W1 manifest 不一致, 交付物不落盘(停止线)")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["record", "stage", "nonfinite_leads", "n_nan", "n_inf",
                    "raw_len", "fs_hz", "nonfinite_after_resample",
                    "detection_path", "source_data", "source_manifest"])
        for r in rows:
            d = dict(kv.split("=", 1) for kv in r["detail"].split(";") if "=" in kv)
            w.writerow([r["stem"], "nonfinite(W1 B口径: wfdb p_signal 8导, 重采样前)",
                        d.get("leads", ""), d.get("nan", ""), d.get("inf", ""),
                        d.get("raw_len", ""), d.get("fs", ""),
                        d.get("nonfinite_after_resample", ""),
                        "prepare_nfh.py L42 同路径重扫", str(SRC), "runlog/W1/nfh_manifest.json(hostB W1)"])

    w3_manifest = {
        "task": "B-3 NFH exclusion reconciliation (host B)",
        "date": "2026-09-22",
        "supersedes_count_only": "W1 hostB manifest 只记计数; 本文件附全名单(确定性重扫复得)",
        "w1_manifest_summary": {k: w1[k] for k in
                                ("n_raw", "n_clean", "n_anomaly", "anomaly_breakdown",
                                 "n_content_duplicates")},
        "rescan": {"n_heas": len(heas), "counts": dict(counts),
                   "wall_s": round(time.time() - t0, 1),
                   "detection": "wfdb.rdrecord -> p_signal 8导(II,III,V1..V6) -> np.isfinite"
                                " (重采样之前; 与 prepare_nfh.py L42 逐字一致)",
                   "reproduces_w1_counts": True},
        "excluded_records": [r["stem"] for r in rows],
        "per_record_detail": [f'{r["stem"]}: {r["detail"]}' for r in rows],
        "note_for_reconciliation": (
            "A 口径(prepare_nfh A 版/数据在 F 盘)同库报 0 非有限; 本名单为 B 侧副本"
            "(E 盘)同一 WFDB 读取路径下的确定性结果。逐条含 raw NaN/Inf 计数与"
            "重采样后状态, 供 A 侧逐条比对原始文件字节, 判定传输损坏 vs 读取差异。"),
    }
    OUT_MANIFEST.write_text(json.dumps(w3_manifest, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    print(f"[out] {OUT_CSV}")
    print(f"[out] {OUT_MANIFEST}")
    print("[B-3] RECONCILIATION OK: 97/97 reproduced deterministically")


if __name__ == "__main__":
    main()
