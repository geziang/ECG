# -*- coding: utf-8 -*-
"""W7 E4a: LP-FT 机制实验(CPSC)——b0/c2 × seeds{0,2,4}, FT10 档, 头初始化自 W5 LP 线性头。

LP-FT(Kumar et al., ICLR 2022): 先线性探查收敛→以该头初始化全参微调。对照读数不重跑:
  LP = runlog/W5/lp_results.csv(cpsc, b0/c2, seeds 0/2/4); full-FT10 = runlog/W6/ft10grid_results_L1.csv。
协议: run_ft.py 全同 W6 Stage3(epochs100/bs128/lr1e-4/fraction0.1), 仅加 --head-init(默认关=基线逐位一致)。
产物: runlog/W7/e4a_lpft.csv + e4a_summary.md(三方式排序表)。
性质: 描述性机制实验(无晋级门), test 单发, protocol_id=w7-e4a。
用法: python runlog/W7/run_e4a_lpft.py [--dry-run]
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.pathguard import open_out  # noqa: E402

PY = "C:/Users/admin/.conda/envs/DL/python.exe"
OUTD = ROOT / "runlog/W7"
LOGD = OUTD / "logs"
OUT = OUTD / "e4a_lpft.csv"
KINDS = ["b0", "c2"]
SEEDS = [0, 2, 4]
TRC = {"b0": "0", "c1": "0", "c2": "1"}


def ck_of(kind, seed):
    if (kind, seed) == ("b0", 0):
        return ROOT / "checkpoint/ptxl_gamma08"
    return ROOT / f"checkpoint/confirm/{kind}_seed{seed}"


def log(msg):
    print(f"[e4a {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def git_sha():
    import subprocess as sp
    return sp.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                  capture_output=True, text=True).stdout.strip()


def sha256_of(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_done(kind, seed):
    if not OUT.exists():
        return False
    with open(OUT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["ckpt"], r["seed"]) == (kind, str(seed)):
                return True
    return False


def record(kind, seed, auroc, auprc, cksha):
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "ckpt", "seed", "eval", "downstream", "fraction", "auroc", "auprc",
                        "git_sha", "checkpoint_sha256", "protocol_id"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), kind, seed, "lpft10", "cpsc", "0.1",
                    f"{auroc:.4f}", f"{auprc:.4f}", git_sha(), cksha, "w7-e4a"])
    log(f"  + e4a {kind} s{seed}: {auroc:.4f}/{auprc:.4f}")


def run_one(kind, seed, dry=False):
    tag = f"e4a_{kind}_cpsc_lpft_seed{seed}"
    if row_done(kind, seed):
        log(f"{tag}: 已有账, 跳过")
        return True
    ckdir = ck_of(kind, seed)
    ckpt = ckdir / "encoder_group.pth"
    head = ROOT / f"runlog/W5/feat/{kind}_cpsc_seed{seed}/classifier_best_ckpt.pth"
    if not ckpt.exists():
        log(f"{tag}: checkpoint 缺失 {ckpt}")
        return False
    if not head.exists():
        log(f"{tag}: LP 头缺失 {head}")
        return False
    mdir = OUTD / "ft" / tag
    if dry:
        log(f"{tag}: DRY ckpt={ckpt} head={head}")
        return True
    with open_out(LOGD, f"{tag}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_ft.py", "--data-dir", "data/cpsc", "--num-classes", "9",
             "--fraction", "0.1", "--checkpoint", str(ckpt),
             "--model-dir", str(mdir), "--workers", "4", "--epochs", "100",
             "--batch-size", "128", "--learning-rate", "0.0001",
             "--seed", str(seed), "--trc", TRC[kind],
             "--head-init", str(head),
             "--extended-metrics", "1",
             "--save-predictions", str(OUTD / "predictions" / tag),
             "--protocol-id", "w7-e4a"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((mdir / "metrics.json").read_text(encoding="utf-8"))
        record(kind, seed, m["auroc"], m["auprc"], sha256_of(ckpt))
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e})")
        return False


def summary():
    import numpy as np
    ours = {}
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ours[(r["ckpt"], r["seed"])] = (float(r["auroc"]), float(r["auprc"]))
    lp_ref, ft_ref = {}, {}
    lp_p = ROOT / "runlog/W5/lp_results.csv"
    ft_p = ROOT / "runlog/W6/ft10grid_results_L1.csv"
    if lp_p.exists():
        with open(lp_p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("downstream") == "cpsc" and r.get("ckpt") in ("b0", "c2"):
                    lp_ref[(r["ckpt"], r["seed"])] = (float(r["auroc"]), float(r["auprc"]))
    if ft_p.exists():
        with open(ft_p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("downstream") == "cpsc" and r.get("ckpt") in ("b0", "c2") and r.get("eval") == "ft10":
                    ft_ref[(r["ckpt"], r["seed"])] = (float(r["auroc"]), float(r["auprc"]))
    lines = ["# E4a LP-FT 机制实验摘要(CPSC, AUROC/AUPRC, 逐种子)", "",
             "| model | seed | LP | full-FT10 | LP-FT10 | LPFT−FT (auroc) |", "|---|---|---|---|---|---|"]
    for kind in KINDS:
        for s in SEEDS:
            k = (kind, str(s))
            if k in ours and k in ft_ref and k in lp_ref:
                lines.append(f"| {kind} | {s} | {lp_ref[k][0]:.4f}/{lp_ref[k][1]:.4f} | "
                             f"{ft_ref[k][0]:.4f}/{ft_ref[k][1]:.4f} | {ours[k][0]:.4f}/{ours[k][1]:.4f} | "
                             f"{ours[k][0]-ft_ref[k][0]:+.4f} |")
    got = [k for k in ours if k in ft_ref]
    if got:
        lines += ["", "## 种子均值", "", "| model | n | mean LPFT auroc | mean FT auroc | Δ |", "|---|---|---|---|---|"]
        for kind in KINDS:
            ks = [k for k in got if k[0] == kind]
            if not ks:
                continue
            mo = np.mean([ours[k][0] for k in ks])
            mf = np.mean([ft_ref[k][0] for k in ks])
            lines.append(f"| {kind} | {len(ks)} | {mo:.4f} | {mf:.4f} | {mo-mf:+.4f} |")
        lines += ["", "- 判读: LP 段 b0>c2(血缘优势)与 FT 段 c2>b0(反转)是否在 LP-FT 下保持——"
                  "若 LP-FT 保住 LP 头且反转消失→反转源于微调早期对 LP 解的形变(Kumar 2022 机制); 若反转保持→c2 特征本身更耐全参微调。"]
    (OUTD / "e4a_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("summary written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if not a.summary_only:
        for kind in KINDS:
            for seed in SEEDS:
                run_one(kind, seed, dry=a.dry_run)
    summary()
