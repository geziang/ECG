# -*- coding: utf-8 -*-
"""W7 E2c: TRC×B0 域内链——B0 协议(PTB-XL 域内预训练 200ep)+TRC 开, 检验"TRC 增益是否限于外部语料"。

协议: 与 B0 逐位同款(data/pt_pretrain, 200ep, bs128, seed0, bt, gamma0.8 默认), 唯一差异 --trc 1;
更新数 ≈27.2k 与 B0 对齐。下游: ptbxl/cpsc LP(--trc 1 与预训练一致)。
参照: B0 = W2 ptbxl LP s0 (AUPRC 0.7177) / W5 cpsc LP s0 (AUROC 0.9506)。
判读(队列 Q6d): 任一域 Δ≥+0.5pt 判正(TRC 不限于外部语料); 均平/负=强化"TRC×域移"核心论点, 如实入账。
账本: runlog/W7/e2c_results.csv; checkpoint: checkpoint/w7_e2c/seed0(不入 git)。
用法: python runlog/W7/run_e2c_chain.py
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
from utils.pathguard import open_out  # noqa: E402

PY = "C:/Users/admin/.conda/envs/DL/python.exe"
OUTD = ROOT / "runlog/W7"
LOGD = OUTD / "logs"
OUT = OUTD / "e2c_results.csv"
CK = ROOT / "checkpoint/w7_e2c/seed0"
DOMAINS = [("ptbxl", 5), ("cpsc", 9)]


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


def row_done(ev, ds):
    if not OUT.exists():
        return False
    with open(OUT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["ckpt"], r["eval"], r["downstream"]) == ("b0trc", ev, ds):
                return True
    return False


def record(ev, ds, auroc, auprc, cksha):
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "ckpt", "seed", "eval", "downstream", "auroc", "auprc",
                        "git_sha", "checkpoint_sha256", "protocol_id", "hparams_ref"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), "b0trc", "0", ev, ds,
                    f"{auroc:.4f}", f"{auprc:.4f}", git_sha(), cksha,
                    "w7-e2c", "B0 协议+trc1: data/pt_pretrain 200ep bs128, 见 run_pt.py --trc"])
    log(f"  + e2c {ev}/{ds}: {auroc:.4f}/{auprc:.4f}")


def run_pt():
    if (CK / "encoder_group.pth").exists() and (CK / "config.json").exists():
        log("b0trc PT: 已完成, 跳过")
        return True
    CK.mkdir(parents=True, exist_ok=True)
    with open_out(LOGD, "e2c_pt_seed0.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_pt.py", "--data-dir", "data/pt_pretrain",
             "--epochs", "200", "--batch-size", "128", "--workers", "4",
             "--seed", "0", "--trc", "1",
             "--resume", "--checkpoint-dir", str(CK)],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    ok = rc == 0 and (CK / "encoder_group.pth").exists()
    log(f"b0trc PT: exit={rc} {'OK' if ok else 'FAIL'}")
    return ok


def run_lp(ds, nc):
    tag = f"e2c_b0trc_{ds}_lp_seed0"
    if row_done("lp", ds):
        log(f"{tag}: 已有账, 跳过")
        return True
    ckpt = CK / "encoder_group.pth"
    feat = OUTD / "feat" / tag
    with open_out(LOGD, f"{tag}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_lp.py", "--data-dir", f"data/{ds}", "--num-classes", str(nc),
             "--checkpoint", str(ckpt), "--feat-dir", str(feat),
             "--seed", "0", "--workers", "6", "--trc", "1",
             "--extended-metrics", "1",
             "--save-predictions", str(OUTD / "predictions" / tag),
             "--protocol-id", "w7-e2c"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((feat / "metrics.json").read_text(encoding="utf-8"))
        record("lp", ds, m["auroc"], m["auprc"], sha256_of(ckpt))
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e})")
        return False


if __name__ == "__main__":
    log("E2c TRC×B0 域内链启动")
    if run_pt():
        for ds, nc in DOMAINS:
            run_lp(ds, nc)
    log("E2C_CHAIN_END")
