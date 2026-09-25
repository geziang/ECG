# -*- coding: utf-8 -*-
"""W5 A-2: CPSC LP 重评估 = {b0,c1,c2,simclr,clocs} × seeds{0,2,4} 共 15 跑。

背景: A-1 审计判"残留"(test=2004 并集, 见 runlog/W5/cpsc_audit.md), 数据已重建为
干净 1385(见 runlog/W5/cpsc_manifest.json)。新分区=新 test, 旧 CPSC 数字不复现是正常的。

协议: 与 W2/W4 LP 同构(run_lp.py 默认超参 100ep/bs128/lr1e-3), c2 --trc 1 其余 --trc 0;
checkpoint 已过 SHA256 反查 15/15(见 runlog/W5/ckpt_sha_check.json)。
幂等: lp_results.csv 已有行且预测产物在 -> 跳过。
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = r"C:/Users/admin/.conda/envs/DL/python.exe"
OUT = ROOT / "runlog/W5/lp_results.csv"
LOGD = ROOT / "runlog/W5/logs"
PRED = "runlog/W5/predictions"

CK_OF = {("b0", 0): "checkpoint/ptxl_gamma08"}
for s in (2, 4):
    CK_OF[("b0", s)] = f"checkpoint/confirm/b0_seed{s}"
for k in ("c1", "c2", "simclr", "clocs"):
    for s in (0, 2, 4):
        CK_OF[(k, s)] = f"checkpoint/confirm/{k}_seed{s}"
TRC = {"c2": 1, "c1": 0, "b0": 0, "simclr": 0, "clocs": 0}
FAMILY = {"b0": "bt_ssl(baseline)", "c1": "bt_ssl", "c2": "bt_ssl+trc",
          "simclr": "contrastive_ssl", "clocs": "clocs_ssl(multi-positive)"}
HP = "W5重评估(见runlog/W5/cpsc_audit.md; LP协议与W2/W4同构)"


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def git_sha():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def ck_sha(rel):
    import hashlib
    return hashlib.sha256((ROOT / rel / "encoder_group.pth").read_bytes()).hexdigest()


def row_done(kind, seed):
    if not OUT.exists():
        return False
    for r in csv.DictReader(open(OUT, encoding="utf-8")):
        if (r["ckpt"], r["seed"], r["eval"], r["downstream"]) == (kind, str(seed), "lp", "cpsc"):
            return True
    return False


def record(kind, seed, auroc, auprc, sha):
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "ckpt", "seed", "eval", "downstream", "auroc", "auprc",
                        "git_sha", "checkpoint_sha256", "method_family", "hparams_ref"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), kind, seed, "lp", "cpsc",
                    f"{auroc:.4f}", f"{auprc:.4f}", git_sha(), sha, FAMILY[kind], HP])
    log(f"  + {kind} s{seed} lp/cpsc: {auroc:.4f}/{auprc:.4f}")


def run_one(kind, seed):
    tag = f"{kind}_cpsc_seed{seed}"
    ck = CK_OF[(kind, seed)]
    feat = ROOT / f"runlog/W5/feat/{tag}"
    if row_done(kind, seed) and (ROOT / PRED / tag / "y_prob.npy").exists():
        log(f"{tag}: 已有账+预测, 跳过")
        return True
    with open(LOGD / f"{tag}.log", "w", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_lp.py", "--data-dir", "data/cpsc", "--num-classes", "9",
             "--checkpoint", f"{ck}/encoder_group.pth", "--feat-dir", str(feat),
             "--seed", str(seed), "--workers", "6", "--trc", str(TRC[kind]),
             "--extended-metrics", "1",
             "--save-predictions", f"{PRED}/{tag}",
             "--protocol-id", "w5-cpscredo"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    if rc != 0:
        log(f"{tag}: FAIL exit={rc} (见 runlog/W5/logs/{tag}.log)")
        return False
    m = json.loads((feat / "metrics.json").read_text(encoding="utf-8"))
    record(kind, seed, m["auroc"], m["auprc"], ck_sha(ck))
    return True


def main():
    LOGD.mkdir(parents=True, exist_ok=True)
    order = [("b0", 0), ("b0", 2), ("b0", 4),
             ("c1", 0), ("c1", 2), ("c1", 4),
             ("c2", 0), ("c2", 2), ("c2", 4),
             ("simclr", 0), ("simclr", 2), ("simclr", 4),
             ("clocs", 0), ("clocs", 2), ("clocs", 4)]
    for kind, seed in order:
        log(f"{kind} s{seed}: 启动")
        if not run_one(kind, seed):
            sys.exit(1)
    log("全部 15 跑完成")


if __name__ == "__main__":
    main()
