# -*- coding: utf-8 -*-
"""W4 A-4: SimCLR(S1) 全量矩阵 = NFH 100ep PT × seeds{0,2,4} + 四行下游×3 seeds。

双车道用法: python run_simclr_matrix.py laneA   (seed0 全链 -> seed4 全链)
            python run_simclr_matrix.py laneB   (seed2 全链)
幂等: baseline_results.csv 已有行自动跳过; PT 以 encoder_group.pth+config.json 判完成。
下游优先级(§3.1): CPSC LP -> PTB FT10 -> PTB LP -> Chapman LP。
所有下游从第一次起带预测落盘(任务书 A-8 §3)。
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
OUT = ROOT / "runlog/W4/baseline_results.csv"
LOGD = ROOT / "runlog/W4/logs"
HP = "simclr-vs-c1(见runlog/W4/baseline_hparams.csv)"
SEEDS = [0, 2, 4]
LANES = {"laneA": [0, 4], "laneB": [2]}


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def git_sha():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def row_done(ckpt, seed, eval_t, ds):
    if not OUT.exists():
        return False
    for r in csv.DictReader(open(OUT, encoding="utf-8")):
        if (r["ckpt"], r["seed"], r["eval"], r["downstream"]) == (ckpt, str(seed), eval_t, ds):
            return True
    return False


def record(ckpt, seed, eval_t, ds, auroc, auprc, cksha):
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "ckpt", "seed", "eval", "downstream", "auroc", "auprc",
                        "git_sha", "checkpoint_sha256", "method_family", "hparams_ref"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), ckpt, seed, eval_t, ds,
                    f"{auroc:.4f}", f"{auprc:.4f}", git_sha(), cksha,
                    "contrastive_ssl", HP])
    log(f"  + {ckpt} s{seed} {eval_t}/{ds}: {auroc:.4f}/{auprc:.4f}")


def run_pt(seed):
    ckdir = ROOT / f"checkpoint/confirm/simclr_seed{seed}"
    tag = f"PT simclr s{seed}"
    if (ckdir / "encoder_group.pth").exists() and (ckdir / "config.json").exists():
        log(f"{tag}: 已完成, 跳过")
        return True
    ckdir.mkdir(parents=True, exist_ok=True)
    with open(LOGD / f"simclr_pt_seed{seed}.log", "w", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_pt.py", "--data-dir", "data/pt_pretrain_nfh",
             "--epochs", "100", "--batch-size", "128", "--workers", "4",
             "--seed", str(seed), "--trc", "0", "--resume",
             "--loss-mode", "simclr", "--checkpoint-dir", str(ckdir)],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    ok = rc == 0 and (ckdir / "encoder_group.pth").exists()
    log(f"{tag}: exit={rc} {'OK' if ok else 'FAIL(见日志)'}")
    return ok


def run_lp(ds, nc, seed):
    ckdir = ROOT / f"checkpoint/confirm/simclr_seed{seed}"
    tag = f"simclr_{ds}_seed{seed}"
    feat = ROOT / f"runlog/W4/feat/{tag}"
    with open(LOGD / f"{tag}.log", "w", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_lp.py", "--data-dir", f"data/{ds}", "--num-classes", str(nc),
             "--checkpoint", str(ckdir / "encoder_group.pth"), "--feat-dir", str(feat),
             "--seed", str(seed), "--workers", "6", "--trc", "0",
             "--extended-metrics", "1",
             "--save-predictions", f"runlog/W4/predictions/{tag}",
             "--protocol-id", "w4-simclr-baseline"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    m = json.loads((feat / "metrics.json").read_text(encoding="utf-8"))
    record("simclr", seed, "lp", ds, m["auroc"], m["auprc"], ck_sha(ckdir))
    return rc == 0


def run_ft10(seed):
    ckdir = ROOT / f"checkpoint/confirm/simclr_seed{seed}"
    tag = f"simclr_ft10_ptbxl_seed{seed}"
    mdir = ROOT / f"runlog/W4/ft/{tag}"
    with open(LOGD / f"{tag}.log", "w", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_ft.py", "--data-dir", "data/ptbxl", "--num-classes", "5",
             "--fraction", "0.1", "--checkpoint", str(ckdir / "encoder_group.pth"),
             "--model-dir", str(mdir), "--workers", "4", "--epochs", "100",
             "--batch-size", "128", "--learning-rate", "0.0001",
             "--seed", str(seed), "--trc", "0",
             "--extended-metrics", "1",
             "--save-predictions", f"runlog/W4/predictions/{tag}",
             "--protocol-id", "w4-simclr-baseline"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    m = json.loads((mdir / "metrics.json").read_text(encoding="utf-8"))
    record("simclr", seed, "ft10", "ptbxl", m["auroc"], m["auprc"], ck_sha(ckdir))
    return rc == 0


def ck_sha(ckdir):
    import hashlib
    return hashlib.sha256((ckdir / "encoder_group.pth").read_bytes()).hexdigest()


def downstream(seed):
    jobs = [("cpsc", 9, "lp"), ("ptbxl", 5, "ft10"), ("ptbxl", 5, "lp"), ("chapman", 4, "lp")]
    for ds, nc, ev in jobs:
        if row_done("simclr", seed, ev, ds):
            log(f"  simclr s{seed} {ev}/{ds}: 已有账, 跳过")
            continue
        log(f"  simclr s{seed} {ev}/{ds}: 启动")
        ok = run_ft10(seed) if ev == "ft10" else run_lp(ds, nc, seed)
        if not ok:
            log(f"  simclr s{seed} {ev}/{ds}: FAIL, 停止本 seed 链")
            return False
    return True


if __name__ == "__main__":
    lane = sys.argv[1] if len(sys.argv) > 1 else "laneA"
    assert lane in LANES, "用法: run_simclr_matrix.py [laneA|laneB]"
    for seed in LANES[lane]:
        if not run_pt(seed):
            sys.exit(1)
        if not downstream(seed):
            sys.exit(1)
    log(f"{lane} ALL DONE")
