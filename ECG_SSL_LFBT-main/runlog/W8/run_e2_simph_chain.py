# -*- coding: utf-8 -*-
"""W8 E2 w8-simph: SimCLR-Physio 预训练+评测链 = W4 S1 全套 + 唯一差异 --nstdb-aug 0.5。

用法(conda DL 解释器, 仓库根为 cwd, 分离进程启动):
    python runlog/W8/run_e2_simph_chain.py

链: 对 seeds{0,2,4} 依次 PT(NFH 100ep bs128 loss-mode simclr nstdb-aug 0.5 trc0,
    ≈27.3k steps 与 C1/C2/S1 对齐) -> LP 三域(ptbxl5/cpsc9/chapman4) -> FT10(ptbxl+cpsc)。
    主链 3seeds 完成后补 3 跑 S1 cpsc FT10(W4 冻结 simclr checkpoint 纯补格评测,
    无噪声, 供 E2 vs S1 在 cpsc FT10 配对; 行 ckpt=simclr-w4frozen 标注)。
零新仓库代码(两既有开关组合, 单测 tests/test_w8_e2.py 4/4); 判定无晋级门如实入表。
产物: runlog/W8/w8_simph_results.csv(W7 schema) + predictions/ + ft/。
幂等: csv 已有行且预测产物在 -> 跳过; PT 有 encoder_group.pth+config.json -> 跳过。
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
OUTD = ROOT / "runlog/W8"
OUT = OUTD / "w8_simph_results.csv"
LOGD = OUTD / "logs"
PRED = "runlog/W8/predictions"
SEEDS = [0, 2, 4]
CKD = ROOT / "checkpoint/w8_simph"
S1CK = ROOT / "checkpoint/confirm"
DOMAINS = [("ptbxl", 5), ("cpsc", 9), ("chapman", 4)]
FT_DOMS = [("ptbxl", 5), ("cpsc", 9)]
HP = ("W8 E2: S1 全套+唯一差异 --nstdb-aug 0.5(NSTDB bw/ma/em 两视图独立 p=0.5 "
      "SNR5-20dB 逐记录逐导联功率匹配); 其余 C1 等价, 见 runlog/W8/w8_hparams.csv "
      "与 runlog/W8/run_e2_simph_chain.py")


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


def row_done(ckpt, seed, evalkind, ds):
    if not OUT.exists():
        return False
    with open(OUT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["ckpt"], r["seed"], r["eval"], r["downstream"]) == (ckpt, str(seed), evalkind, ds):
                return True
    return False


def record(ckpt, seed, evalkind, ds, auroc, auprc, cksha, hp=HP):
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "ckpt", "seed", "eval", "downstream", "auroc", "auprc",
                        "git_sha", "checkpoint_sha256", "protocol_id", "hparams_ref"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), ckpt, seed, evalkind, ds,
                    f"{auroc:.4f}", f"{auprc:.4f}", git_sha(), cksha, "w8-simph", hp])
    log(f"  + {ckpt} s{seed} {evalkind}/{ds}: {auroc:.4f}/{auprc:.4f}")


def run_pt(seed):
    ck = CKD / f"simph_seed{seed}"
    if (ck / "encoder_group.pth").exists() and (ck / "config.json").exists():
        log(f"simph s{seed} PT: 已完成, 跳过")
        return True
    ck.mkdir(parents=True, exist_ok=True)
    with open_out(LOGD, f"simph_pt_seed{seed}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_pt.py", "--data-dir", "data/pt_pretrain_nfh",
             "--epochs", "100", "--batch-size", "128", "--workers", "4",
             "--seed", str(seed), "--trc", "0", "--loss-mode", "simclr",
             "--nstdb-aug", "0.5", "--resume",
             "--checkpoint-dir", str(ck)],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    ok = rc == 0 and (ck / "encoder_group.pth").exists()
    log(f"simph s{seed} PT: exit={rc} {'OK' if ok else 'FAIL(见 runlog/W8/logs)'}")
    return ok


def run_lp(ck, seed, ds, nc, ckpt_name):
    tag = f"{ckpt_name}_{ds}_lp_seed{seed}"
    if row_done(ckpt_name, seed, "lp", ds) and (ROOT / PRED / tag / "y_prob.npy").exists():
        log(f"{tag}: 已有账+预测, 跳过")
        return True
    feat = OUTD / "feat" / tag
    with open_out(LOGD, f"{tag}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_lp.py", "--data-dir", f"data/{ds}", "--num-classes", str(nc),
             "--checkpoint", str(ck / "encoder_group.pth"), "--feat-dir", str(feat),
             "--seed", str(seed), "--workers", "6", "--trc", "0",
             "--extended-metrics", "1",
             "--save-predictions", f"{PRED}/{tag}",
             "--protocol-id", "w8-simph"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((feat / "metrics.json").read_text(encoding="utf-8"))
        record(ckpt_name, seed, "lp", ds, m["auroc"], m["auprc"], sha256_of(ck / "encoder_group.pth"))
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e}; 见 runlog/W8/logs/{tag}.log 尾部)")
        return False


def run_ft(ck, seed, ds, nc, ckpt_name):
    tag = f"{ckpt_name}_{ds}_ft10_seed{seed}"
    if row_done(ckpt_name, seed, "ft10", ds) and (ROOT / PRED / tag / "y_prob.npy").exists():
        log(f"{tag}: 已有账+预测, 跳过")
        return True
    mdir = OUTD / "ft" / tag
    with open_out(LOGD, f"{tag}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_ft.py", "--data-dir", f"data/{ds}", "--num-classes", str(nc),
             "--fraction", "0.1",
             "--checkpoint", str(ck / "encoder_group.pth"),
             "--model-dir", str(mdir), "--workers", "4", "--epochs", "100",
             "--batch-size", "128", "--learning-rate", "0.0001",
             "--seed", str(seed), "--trc", "0",
             "--extended-metrics", "1",
             "--save-predictions", f"{PRED}/{tag}",
             "--protocol-id", "w8-simph"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((mdir / "metrics.json").read_text(encoding="utf-8"))
        record(ckpt_name, seed, "ft10", ds, m["auroc"], m["auprc"], sha256_of(ck / "encoder_group.pth"))
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e}; 见 runlog/W8/logs/{tag}.log 尾部)")
        return False


def main():
    LOGD.mkdir(parents=True, exist_ok=True)
    (OUTD / "feat").mkdir(parents=True, exist_ok=True)
    (OUTD / "predictions").mkdir(parents=True, exist_ok=True)
    (OUTD / "ft").mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        ck = CKD / f"simph_seed{seed}"
        log(f"=== simph seed{seed}: PT 启动 ===")
        if not run_pt(seed):
            sys.exit(1)
        for ds, nc in DOMAINS:
            log(f"simph s{seed} lp/{ds}: 启动")
            if not run_lp(ck, seed, ds, nc, "simph"):
                sys.exit(1)
        for ds, nc in FT_DOMS:
            log(f"simph s{seed} ft10/{ds}: 启动")
            if not run_ft(ck, seed, ds, nc, "simph"):
                sys.exit(1)
    # 主链收官后补格: S1 cpsc FT10(W4 冻结 checkpoint 纯评测, 无噪声, 供配对)
    for seed in SEEDS:
        ck = S1CK / f"simclr_seed{seed}"
        if not (ck / "encoder_group.pth").exists():
            log(f"S1 s{seed} checkpoint 缺失, 补格跳过")
            continue
        log(f"S1 s{seed} ft10/cpsc 补格: 启动")
        run_ft(ck, seed, "cpsc", 9, "simclr-w4frozen")
    log("E2 全链完成")


if __name__ == "__main__":
    main()
