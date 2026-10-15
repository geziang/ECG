# -*- coding: utf-8 -*-
"""run_w8_stats.py — W8 批次遗留统计检验(W8 任务书"统计"条, CPU-only)。

对照: 各线 vs C1(主); E2 simph 另加 vs S1(仅 cpsc——S1 逐记录工件只在 W5A
cpsc 重做集, ptbxl/chapman 无存档不可配对, 报告侧以冻结均值差描述)。
方法: 复用 JOINT 统计件(run_joint_stats)的 bootstrap 机器(10k/多种子同步
重采样/RNG 20261013)与 W4 DeLong; c1 锚点 = W5A cpsc 逐记录 + JOINT 批次
已重放的 predictions_anchor/m0_{ptbxl,chapman}(四位对拍过冻结值)。
产物: runlog/W8/paired_stats.csv; 完成标记行 "W8 统计完成"。
"""
import csv
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "rjs", ROOT / "runlog/JOINT_20261013/run_joint_stats.py")
J = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(J)  # 复用 paired_bootstrap/_m_auprc/_m_auroc/W4/load_preds
W4 = J.W4

OUT = ROOT / "runlog/W8/paired_stats.csv"
SEEDS = [0, 2, 4]
NC = {"ptbxl": 5, "cpsc": 9, "chapman": 4}


def log(msg):
    print(f"[w8stats {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def pred_dir(name, ds, seed):
    if name in ("rand", "simph", "tsr", "trcaff", "noiseiso"):
        return ROOT / f"runlog/W8/predictions/{name}_{ds}_lp_seed{seed}"
    if name == "c1":
        if ds == "cpsc":
            return ROOT / f"runlog/W5/predictions/c1_cpsc_seed{seed}"
        return ROOT / f"runlog/JOINT_20261013/predictions_anchor/m0_{ds}_seed{seed}"
    if name == "s1" and ds == "cpsc":
        return ROOT / f"runlog/W5/predictions/simclr_cpsc_seed{seed}"
    return None


def load(name, ds):
    out = {}
    for s in SEEDS:
        d = pred_dir(name, ds, s)
        if d is None or not (d / "y_prob.npy").exists():
            return None
        y, _, p = W4.load_pred_dir(d)
        out[s] = (y, p.astype(np.float64))
    return out


def record(row):
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "dataset", "pair", "metric", "delta_mean_pt",
                        "ci_lo_pt", "ci_hi_pt", "p_emp", "per_seed_pt"])
        w.writerow(row)


def main():
    pairs_by_domain = {
        "ptbxl": [("rand", "c1"), ("simph", "c1"), ("tsr", "c1"),
                  ("trcaff", "c1"), ("noiseiso", "c1")],
        "cpsc": [("rand", "c1"), ("simph", "c1"), ("simph", "s1"), ("tsr", "c1"),
                 ("trcaff", "c1"), ("noiseiso", "c1")],
        "chapman": [("rand", "c1"), ("simph", "c1"), ("tsr", "c1"),
                    ("noiseiso", "c1")],
    }
    for ds, pairs in pairs_by_domain.items():
        for a, b in pairs:
            pa, pb = load(a, ds), load(b, ds)
            if pa is None or pb is None:
                log(f"  [skip] {ds} {a}:{b}: 预测缺")
                continue
            for metric_name, mf in (("auprc", J._m_auprc), ("auroc", J._m_auroc)):
                r = J.paired_bootstrap(pa, pb, mf, n_boot=J.N_BOOT)
                row = [time.strftime("%Y-%m-%d %H:%M"), ds, f"{a}:{b}", metric_name,
                       round(r["delta_mean"] * 100, 3), round(r["ci_lo"] * 100, 3),
                       round(r["ci_hi"] * 100, 3), round(r["p_emp"], 4)]
                import json
                row.append(json.dumps({str(k): round(v * 100, 3)
                                       for k, v in r["per_seed"].items()}))
                record(row)
                log(f"  {ds} {a}:{b} [{metric_name}] Δ={row[4]:+.3f}pt "
                    f"CI[{row[5]:+.3f},{row[6]:+.3f}] p={row[7]}")
            if ds == "cpsc":
                for s in SEEDS:
                    ya, pa_s = pa[s]
                    _, pb_s = pb[s]
                    dl = [W4.delong_paired_class(ya, pa_s, pb_s, c)
                          for c in range(pa_s.shape[1])]
                    ps = [d[4] for d in dl if np.isfinite(d[4])]
                    if ps:
                        record([time.strftime("%Y-%m-%d %H:%M"), ds, f"{a}:{b}",
                                "delong_auroc_p_min", None, None, None,
                                round(float(np.min(ps)), 4), str(s)])
    log("W8 统计完成")


if __name__ == "__main__":
    main()
