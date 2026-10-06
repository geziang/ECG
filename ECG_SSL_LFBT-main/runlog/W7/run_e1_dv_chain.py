# -*- coding: utf-8 -*-
"""W7 E1: 双视角灰区加种子链(seed 2/4), 协议与 W6 Stage4 逐位同款 (PT + 三域 LP + PTB FT10), 协议 w6-dv。

用法: python runlog/W6/run_dv_chain.py [seed]   # seed 默认 0; 过门加跑 2/4; 权重定档 0.01/0.1
其余协议 = C1(无 TRC, NFH 100ep, bt, RRC-TO 默认), 加 --dualview-weight/--rr-weight;
权重 dualview=0.01/rr=0.1(Q2c smoke 定档); 预授权门同 Stage2 规格(Δ≥+0.3pt 加种子, <+0.1pt 关线)。

checkpoint: checkpoint/w6_dv/seed{S} (不入 git); 账本 runlog/W6/dv_results.csv;
预测落盘三参数(评估侧) protocol_id=w6-dv。失败跳行不停车(车道纪律)。
"""
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.pathguard import open_out

PY = sys.executable
LOGD = ROOT / "runlog/W7/logs"
LAM = sys.argv[1] if len(sys.argv) > 1 else ""
SEED = sys.argv[2] if len(sys.argv) > 2 else "0"
CK = ROOT / f"checkpoint/w7_dv/seed{SEED}"
DOMAINS = [("ptbxl", 5), ("cpsc", 9), ("chapman", 4)]
FAMILY = "bt_ssl+dv"


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def git_sha():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def sha256_of(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


OUT = ROOT / "runlog/W7/e1_dv_results.csv"


def row_done(ev, ds):
    if not OUT.exists():
        return False
    with open(OUT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["ckpt"], r["seed"], r["eval"], r["downstream"]) == ("dv", SEED, ev, ds):
                return True
    return False


def record(ev, ds, auroc, auprc, cksha):
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "ckpt", "seed", "dv_weights", "eval", "downstream", "auroc", "auprc",
                        "git_sha", "checkpoint_sha256", "protocol_id", "hparams_ref"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), "dv", SEED, "0.01/0.1", ev, ds,
                    f"{auroc:.4f}", f"{auprc:.4f}", git_sha(), cksha,
                    "w7-e1", "W7 E1: W6 Stage4 同款协议(dualview 0.01 + rr 0.1)灰区加种子, 见 run_pt.py --dualview-weight"])
    log(f"  + dv s{SEED} {ev}/{ds}: {auroc:.4f}/{auprc:.4f}")


def run_pt():
    tag = f"dv PT s{SEED}"
    if (CK / "encoder_group.pth").exists() and (CK / "config.json").exists():
        log(f"{tag}: 已完成, 跳过")
        return True
    CK.mkdir(parents=True, exist_ok=True)
    with open_out(LOGD, f"dv_pt_seed{SEED}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_pt.py", "--data-dir", "data/pt_pretrain_nfh",
             "--epochs", "100", "--batch-size", "128", "--workers", "4",
             "--seed", str(SEED), "--trc", "0",
             "--dualview-weight", "0.01", "--rr-weight", "0.1",
             "--rpeak-npz", "data/pt_rpeaks_nfh.npz",
             "--resume", "--checkpoint-dir", str(CK)],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    ok = rc == 0 and (CK / "encoder_group.pth").exists()
    log(f"{tag}: exit={rc} {'OK' if ok else 'FAIL(见日志)'}")
    return ok


def run_lp(ds, nc):
    tag = f"dv_{ds}_lp_seed{SEED}"
    if row_done("lp", ds):
        log(f"{tag}: 已有账, 跳过")
        return True
    ckpt = CK / "encoder_group.pth"
    if not ckpt.exists():
        return False
    feat = ROOT / f"runlog/W6/feat/dv_{ds}_seed{SEED}"
    with open_out(LOGD, f"{tag}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_lp.py", "--data-dir", f"data/{ds}", "--num-classes", str(nc),
             "--checkpoint", str(ckpt), "--feat-dir", str(feat),
             "--seed", str(SEED), "--workers", "6", "--trc", "0",
             "--extended-metrics", "1",
             "--save-predictions", f"runlog/W7/predictions/{tag}",
             "--protocol-id", "w6-dv"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((feat / "metrics.json").read_text(encoding="utf-8"))
        record("lp", ds, m["auroc"], m["auprc"], sha256_of(ckpt))
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e})")
        return False


def run_ft10():
    tag = f"dv_ptbxl_ft10_seed{SEED}"
    if row_done("ft10", "ptbxl"):
        log(f"{tag}: 已有账, 跳过")
        return True
    ckpt = CK / "encoder_group.pth"
    if not ckpt.exists():
        return False
    mdir = ROOT / f"runlog/W6/ft/{tag}"
    with open_out(LOGD, f"{tag}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_ft.py", "--data-dir", "data/ptbxl", "--num-classes", "5",
             "--fraction", "0.1", "--checkpoint", str(ckpt),
             "--model-dir", str(mdir), "--workers", "4", "--epochs", "100",
             "--batch-size", "128", "--learning-rate", "0.0001",
             "--seed", str(SEED), "--trc", "0",
             "--extended-metrics", "1",
             "--save-predictions", f"runlog/W7/predictions/{tag}",
             "--protocol-id", "w6-dv"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((mdir / "metrics.json").read_text(encoding="utf-8"))
        record("ft10", "ptbxl", m["auroc"], m["auprc"], sha256_of(ckpt))
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e})")
        return False


def main():
    if SEED not in ("0", "2", "4"):
        sys.exit("用法: python runlog/W6/run_dv_chain.py [0|2|4]")
    LOGD.mkdir(parents=True, exist_ok=True)
    log(f"W6 Stage4 双视角 seed{SEED} 全链启动(dv=0.01/rr=0.1), 解释器={PY}")
    if not run_pt():
        log("DV PT 失败, 链终止(见日志)")
        return
    for ds, nc in DOMAINS:
        run_lp(ds, nc)
    run_ft10()
    log(f"DV seed{SEED} 链结束")


if __name__ == "__main__":
    main()
