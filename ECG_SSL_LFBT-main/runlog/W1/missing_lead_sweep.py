"""missing_lead_sweep.py — 任务书 §5.3 的 37 条件缺导曲线 (2026-09-21 晨)。

条件 = 完整(1) + 单导缺失(8) + 双导缺失 C(8,2)=28, 共 37; × {b0_anchor, c1_nfh_b0} 两权重,
在 PTB-XL LP 上评估(导联置零口径, 非电极物理错位)。幂等: 已有 csv 行自动跳过, 可反复重启。
输出: runlog/W1/missing_lead_curves.csv (ckpt,cond,auroc,auprc)
"""
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"F:\新实验\ECG_SSL_LFBT-main")
PY = r"C:\Users\admin\.conda\envs\DL\python.exe"
CKPTS = {
    "b0_anchor": ROOT / "checkpoint/ptxl_gamma08/encoder_group.pth",
    "c1_nfh_b0": ROOT / "checkpoint/M/c1_nfh_b0_seed0/encoder_group.pth",
}
CONDS = [("full", "")] + [(f"miss{i}", str(i)) for i in range(8)] + \
        [(f"miss{i}{j}", f"{i},{j}") for i, j in itertools.combinations(range(8), 2)]
OUT = ROOT / "runlog/W1/missing_lead_curves.csv"


def done_rows():
    if not OUT.exists():
        return set()
    with open(OUT, newline="", encoding="utf-8") as f:
        return {(r["ckpt"], r["cond"]) for r in csv.DictReader(f)}


def main():
    done = done_rows()
    if not OUT.exists():
        with open(OUT, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["ckpt", "cond", "auroc", "auprc"])
    total = len(CKPTS) * len(CONDS)
    for n, ((ck, ckpt), (cname, leads)) in enumerate(
            itertools.product(CKPTS.items(), CONDS), 1):
        if (ck, cname) in done:
            print(f"[{n}/{total}] {ck} {cname} 已有, 跳过", flush=True)
            continue
        feat = ROOT / f"feat/W1_ml_{ck}_{cname}"
        cmd = [PY, str(ROOT / "run_lp.py"), "--data-dir", str(ROOT / "data/ptbxl"),
               "--num-classes", "5", "--checkpoint", str(ckpt),
               "--feat-dir", str(feat), "--seed", "0", "--workers", "6"]
        if leads:
            cmd += ["--zero-leads", leads]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        try:
            m = json.load(open(feat / "metrics.json"))
            with open(OUT, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([ck, cname, f"{m['auroc']:.4f}", f"{m['auprc']:.4f}"])
            print(f"[{n}/{total}] {ck} {cname}: auprc={m['auprc']:.4f}", flush=True)
        except Exception:
            print(f"[{n}/{total}] {ck} {cname} FAILED rc={r.returncode}: "
                  f"{(r.stdout or r.stderr or '')[-200:]}", flush=True)


if __name__ == "__main__":
    main()
    print("SWEEP_DONE")
