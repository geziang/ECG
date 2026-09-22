"""missing_lead_c2_sweep.py — W2 C2 三种子 37 条件缺导曲线 (08任务书 §5 缺导关键条件, 2026-09-22)。

条件 = 完整(1)+单导缺失(8)+双导缺失 C(8,2)=28, 共 37 × C2 seeds{0,2,4};
PTB-XL LP 导联置零口径(非电极物理错位), C2 权重必须 --trc 1。
幂等: runlog/W2/missing_lead_c2.csv 已有 (seed,cond) 行自动跳过, 可反复重启。
对照: W1 b0/c1 单种子曲线在 runlog/W1/missing_lead_curves.csv(口径一致可比)。
"""
import csv
import itertools
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = os.environ.get("DL_PY", r"C:\Users\admin\.conda\envs\DL\python.exe")
OUT = ROOT / "runlog/W2/missing_lead_c2.csv"
LOG = ROOT / "runlog/W2/missing_lead_c2.log"
SEEDS = (0, 2, 4)
CONDS = [("full", "")] + [(f"miss{i}", str(i)) for i in range(8)] + \
        [(f"miss{i}{j}", f"{i},{j}") for i, j in itertools.combinations(range(8), 2)]


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def done_rows():
    if not OUT.exists():
        return set()
    with open(OUT, newline="", encoding="utf-8") as f:
        return {(int(r["seed"]), r["cond"]) for r in csv.DictReader(f)}


def main():
    done = done_rows()
    if not OUT.exists():
        with open(OUT, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["seed", "cond", "auroc", "auprc"])
    total = len(SEEDS) * len(CONDS)
    n = 0
    for s in SEEDS:
        ckpt = ROOT / f"checkpoint/confirm/c2_seed{s}/encoder_group.pth"
        if not ckpt.exists():
            log(f"s{s}: checkpoint 缺失 {ckpt}, 跳过该种子")
            continue
        for cname, leads in CONDS:
            n += 1
            if (s, cname) in done:
                continue
            feat = ROOT / f"results/confirm/ml_c2_s{s}_{cname}"
            cmd = [PY, "-u", "run_lp.py", "--data-dir", "data/ptbxl",
                   "--num-classes", "5", "--checkpoint", str(ckpt),
                   "--feat-dir", str(feat), "--seed", str(s), "--workers", "6",
                   "--trc", "1"]
            if leads:
                cmd += ["--zero-leads", leads]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
            try:
                m = json.loads((feat / "metrics.json").read_text())
                with open(OUT, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([s, cname, f"{m['auroc']:.4f}", f"{m['auprc']:.4f}"])
                log(f"[{n}/{total}] s{s} {cname}: auprc={m['auprc']:.4f}")
            except Exception:
                log(f"[{n}/{total}] s{s} {cname} FAILED rc={r.returncode}: "
                    f"{(r.stdout or r.stderr or '')[-200:]}")
    log("C2_ML_SWEEP_DONE")


if __name__ == "__main__":
    main()
