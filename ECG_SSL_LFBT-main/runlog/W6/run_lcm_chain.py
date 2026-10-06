# -*- coding: utf-8 -*-
"""W6 Stage 2: LCM seed0 全链驱动 (PT + 三域 LP + PTB FT10), 协议 w6-lcm。

用法: python runlog/W6/run_lcm_chain.py <lambda> [seed]   # seed 默认 0; 过门加跑 2/4
其余协议 = C1(无 TRC, NFH 100ep, bt, RRC-TO 默认), 仅加 --lcm-weight λ;
任务书预授权: seed0 CPSC LP Δ≥+0.3pt 才加 seeds{2,4}, 判负即关线。

checkpoint: checkpoint/w6_lcm/seed0 (不入 git); 账本 runlog/W6/lcm_results.csv;
预测落盘三参数(评估侧) protocol_id=w6-lcm。失败跳行不停车(车道纪律)。
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
LOGD = ROOT / "runlog/W6/logs"
LAM = sys.argv[1] if len(sys.argv) > 1 else ""
SEED = sys.argv[2] if len(sys.argv) > 2 else "0"
CK = ROOT / f"checkpoint/w6_lcm/seed{SEED}"
DOMAINS = [("ptbxl", 5), ("cpsc", 9), ("chapman", 4)]
FAMILY = "bt_ssl+lcm"


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


OUT = ROOT / "runlog/W6/lcm_results.csv"


def row_done(ev, ds):
    if not OUT.exists():
        return False
    with open(OUT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["ckpt"], r["seed"], r["eval"], r["downstream"]) == ("lcm", SEED, ev, ds):
                return True
    return False


def record(ev, ds, auroc, auprc, cksha):
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "ckpt", "seed", "lcm_weight", "eval", "downstream", "auroc", "auprc",
                        "git_sha", "checkpoint_sha256", "protocol_id", "hparams_ref"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), "lcm", SEED, LAM, ev, ds,
                    f"{auroc:.4f}", f"{auprc:.4f}", git_sha(), cksha,
                    "w6-lcm", f"W6 Stage2: C1 协议 + lcm-weight {LAM}, 见 run_pt.py --lcm-weight"])
    log(f"  + lcm(λ={LAM}) {ev}/{ds}: {auroc:.4f}/{auprc:.4f}")


def run_pt():
    tag = f"lcm PT s{SEED} (λ={LAM})"
    if (CK / "encoder_group.pth").exists() and (CK / "config.json").exists():
        log(f"{tag}: 已完成, 跳过")
        return True
    CK.mkdir(parents=True, exist_ok=True)
    with open_out(LOGD, f"lcm_pt_seed{SEED}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_pt.py", "--data-dir", "data/pt_pretrain_nfh",
             "--epochs", "100", "--batch-size", "128", "--workers", "4",
             "--seed", str(SEED), "--trc", "0", "--lcm-weight", LAM,
             "--resume", "--checkpoint-dir", str(CK)],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    ok = rc == 0 and (CK / "encoder_group.pth").exists()
    log(f"{tag}: exit={rc} {'OK' if ok else 'FAIL(见日志)'}")
    return ok


def run_lp(ds, nc):
    tag = f"lcm_{ds}_lp_seed{SEED}"
    if row_done("lp", ds):
        log(f"{tag}: 已有账, 跳过")
        return True
    ckpt = CK / "encoder_group.pth"
    if not ckpt.exists():
        return False
    feat = ROOT / f"runlog/W6/feat/lcm_{ds}_seed0"
    with open_out(LOGD, f"{tag}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_lp.py", "--data-dir", f"data/{ds}", "--num-classes", str(nc),
             "--checkpoint", str(ckpt), "--feat-dir", str(feat),
             "--seed", str(SEED), "--workers", "6", "--trc", "0",
             "--extended-metrics", "1",
             "--save-predictions", f"runlog/W6/predictions/{tag}",
             "--protocol-id", "w6-lcm"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((feat / "metrics.json").read_text(encoding="utf-8"))
        record("lp", ds, m["auroc"], m["auprc"], sha256_of(ckpt))
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e})")
        return False


def run_ft10():
    tag = f"lcm_ptbxl_ft10_seed{SEED}"
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
             "--save-predictions", f"runlog/W6/predictions/{tag}",
             "--protocol-id", "w6-lcm"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((mdir / "metrics.json").read_text(encoding="utf-8"))
        record("ft10", "ptbxl", m["auroc"], m["auprc"], sha256_of(ckpt))
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e})")
        return False


def main():
    if LAM not in ("0.01", "0.05") or SEED not in ("0", "2", "4"):
        sys.exit("用法: python runlog/W6/run_lcm_chain.py {0.01|0.05} [0|2|4]")
    LOGD.mkdir(parents=True, exist_ok=True)
    log(f"W6 Stage2 LCM seed{SEED} 全链启动, λ={LAM}, 解释器={PY}")
    if not run_pt():
        log("PT 失败, 链终止(事件已可从日志追溯)")
        return
    for ds, nc in DOMAINS:
        run_lp(ds, nc)
    run_ft10()
    log(f"LCM seed{SEED} 链结束")


if __name__ == "__main__":
    main()
