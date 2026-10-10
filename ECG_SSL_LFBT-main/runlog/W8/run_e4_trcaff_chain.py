# -*- coding: utf-8 -*-
"""W8 E4 w8-trcaff: TRC 参数匹配仿射对照链 (VGG16 trc=2 = C1+逐通道仿射, 去 ⊙n 门控)。

用法(conda DL 解释器, 仓库根为 cwd, 分离进程启动):
    python runlog/W8/run_e4_trcaff_chain.py

链: seeds{0,2,4} 依次 PT(NFH 100ep bs128 bt loss trc2, 与 C1 唯一差异=仿射模块)
    -> LP {cpsc, ptbxl}。三臂对照 = C1(trc0)/affine(trc2)/C2(trc1), 参照读数引
    W5A/W2 冻结账本不重跑。单测 tests/test_w8_e4.py 7/7(零初始化逐位一致/梯度/公式)。
产物: runlog/W8/w8_trcaff_results.csv(W7 schema) + predictions/。
幂等: csv 已有行且预测在 -> 跳过; PT 有 encoder+config -> 跳过。
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
OUT = OUTD / "w8_trcaff_results.csv"
LOGD = OUTD / "logs"
PRED = "runlog/W8/predictions"
SEEDS = [0, 2, 4]
CKD = ROOT / "checkpoint/w8_trcaff"
DOMAINS = [("cpsc", 9), ("ptbxl", 5)]
HP = ("W8 E4: C1 协议 + --trc 2(逐通道仿射 F~=F+gamma*F+beta 零初始化, 与 TRC 同位置"
      "同参数量, 去 n 门控; models/vgg_1d.AffineChannel1D); 见 runlog/W8/w8_hparams.csv "
      "与 runlog/W8/run_e4_trcaff_chain.py")


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


def row_done(seed, evalkind, ds):
    if not OUT.exists():
        return False
    with open(OUT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["ckpt"], r["seed"], r["eval"], r["downstream"]) == ("trcaff", str(seed), evalkind, ds):
                return True
    return False


def record(seed, evalkind, ds, auroc, auprc, cksha):
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "ckpt", "seed", "eval", "downstream", "auroc", "auprc",
                        "git_sha", "checkpoint_sha256", "protocol_id", "hparams_ref"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), "trcaff", seed, evalkind, ds,
                    f"{auroc:.4f}", f"{auprc:.4f}", git_sha(), cksha, "w8-trcaff", HP])
    log(f"  + trcaff s{seed} {evalkind}/{ds}: {auroc:.4f}/{auprc:.4f}")


def run_pt(seed):
    ck = CKD / f"trcaff_seed{seed}"
    if (ck / "encoder_group.pth").exists() and (ck / "config.json").exists():
        log(f"trcaff s{seed} PT: 已完成, 跳过")
        return True
    ck.mkdir(parents=True, exist_ok=True)
    with open_out(LOGD, f"trcaff_pt_seed{seed}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_pt.py", "--data-dir", "data/pt_pretrain_nfh",
             "--epochs", "100", "--batch-size", "128", "--workers", "4",
             "--seed", str(seed), "--trc", "2", "--resume",
             "--checkpoint-dir", str(ck)],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    ok = rc == 0 and (ck / "encoder_group.pth").exists()
    log(f"trcaff s{seed} PT: exit={rc} {'OK' if ok else 'FAIL(见 runlog/W8/logs)'}")
    return ok


def run_lp(ck, seed, ds, nc):
    tag = f"trcaff_{ds}_lp_seed{seed}"
    if row_done(seed, "lp", ds) and (ROOT / PRED / tag / "y_prob.npy").exists():
        log(f"{tag}: 已有账+预测, 跳过")
        return True
    feat = OUTD / "feat" / tag
    with open_out(LOGD, f"{tag}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_lp.py", "--data-dir", f"data/{ds}", "--num-classes", str(nc),
             "--checkpoint", str(ck / "encoder_group.pth"), "--feat-dir", str(feat),
             "--seed", str(seed), "--workers", "6", "--trc", "2",
             "--extended-metrics", "1",
             "--save-predictions", f"{PRED}/{tag}",
             "--protocol-id", "w8-trcaff"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((feat / "metrics.json").read_text(encoding="utf-8"))
        record(seed, "lp", ds, m["auroc"], m["auprc"], sha256_of(ck / "encoder_group.pth"))
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e}; 见 runlog/W8/logs/{tag}.log 尾部)")
        return False


def main():
    LOGD.mkdir(parents=True, exist_ok=True)
    (OUTD / "feat").mkdir(parents=True, exist_ok=True)
    (OUTD / "predictions").mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        ck = CKD / f"trcaff_seed{seed}"
        log(f"=== trcaff seed{seed}: PT 启动 ===")
        if not run_pt(seed):
            sys.exit(1)
        for ds, nc in DOMAINS:
            log(f"trcaff s{seed} lp/{ds}: 启动")
            if not run_lp(ck, seed, ds, nc):
                sys.exit(1)
    log("E4 全链完成")


if __name__ == "__main__":
    main()
