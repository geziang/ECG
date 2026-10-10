# -*- coding: utf-8 -*-
"""W8 E5(w8-dvsplit) + E7(w8-b0trc-chap): 殿后双链。

E5 Beat-RR 组件拆分(机制消融, 单种子, 任务书 2026-10-10 增补):
  - 仅双视角: L_BT + 0.01·L_dv (--dualview-weight 0.01, rr 0), seed0, NFH 100ep;
  - 仅 RR:    L_BT + 0.1·L_rr  (--rr-weight 0.1, dv 0),    seed0, NFH 100ep;
  - 评测 = CPSC LP(+PTB-XL 观测); 与 W7 E1 联合 3-seed 并列报告, 描述性, 主结果
    仍为联合 3-seed, 不据此单项归因(任务书判定列)。
E7 B0-TRC→Chapman LP 补格:
  - W7 E2c 冻结 b0trc checkpoints(checkpoint/w7_e2c/seed{0,2,4}, trc=1), seeds{0,2,4}
    Chapman LP, 补全 2 语料(ptbxl/cpsc 已有)x3 目标域矩阵; 如实入表 5 注。
产物: runlog/W8/w8_dvsplit_results.csv + w8_b0trc_results.csv(W7 schema)。
幂等: csv 已有行且预测在 -> 跳过。
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
LOGD = OUTD / "logs"
PRED = "runlog/W8/predictions"
E5_OUT = OUTD / "w8_dvsplit_results.csv"
E7_OUT = OUTD / "w8_b0trc_results.csv"
E7CK = ROOT / "checkpoint/w7_e2c"
E5_DOMS = [("cpsc", 9), ("ptbxl", 5)]


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


def row_done(out, ckpt, seed, evalkind, ds):
    if not out.exists():
        return False
    with open(out, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["ckpt"], r["seed"], r["eval"], r["downstream"]) == (ckpt, str(seed), evalkind, ds):
                return True
    return False


def record(out, ckpt, seed, evalkind, ds, auroc, auprc, cksha, hp):
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "ckpt", "seed", "eval", "downstream", "auroc", "auprc",
                        "git_sha", "checkpoint_sha256", "protocol_id", "hparams_ref"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), ckpt, seed, evalkind, ds,
                    f"{auroc:.4f}", f"{auprc:.4f}", git_sha(), cksha,
                    "w8-dvsplit" if out is E5_OUT else "w8-b0trc", hp])
    log(f"  + {ckpt} s{seed} {evalkind}/{ds}: {auroc:.4f}/{auprc:.4f}")


def run_lp(out, ck, seed, ds, nc, ckpt_name, hp, protocol):
    tag = f"{ckpt_name}_{ds}_lp_seed{seed}"
    if row_done(out, ckpt_name, seed, "lp", ds) and (ROOT / PRED / tag / "y_prob.npy").exists():
        log(f"{tag}: 已有账+预测, 跳过")
        return True
    feat = OUTD / "feat" / tag
    with open_out(LOGD, f"{tag}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_lp.py", "--data-dir", f"data/{ds}", "--num-classes", str(nc),
             "--checkpoint", str(ck / "encoder_group.pth"), "--feat-dir", str(feat),
             "--seed", str(seed), "--workers", "6",
             "--trc", "1" if "b0trc" in ckpt_name else "0",
             "--extended-metrics", "1",
             "--save-predictions", f"{PRED}/{tag}",
             "--protocol-id", protocol],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((feat / "metrics.json").read_text(encoding="utf-8"))
        record(out, ckpt_name, seed, "lp", ds, m["auroc"], m["auprc"],
               sha256_of(ck / "encoder_group.pth"), hp)
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e})")
        return False


def e5_pt(name, flag, weight):
    ck = ROOT / f"checkpoint/w8_dvsplit/{name}_seed0"
    if (ck / "encoder_group.pth").exists() and (ck / "config.json").exists():
        log(f"{name} PT: 已完成, 跳过")
        return ck
    ck.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-u", "run_pt.py", "--data-dir", "data/pt_pretrain_nfh",
           "--epochs", "100", "--batch-size", "128", "--workers", "4",
           "--seed", "0", "--trc", "0", flag, weight, "--resume",
           "--checkpoint-dir", str(ck)]
    if flag == "--dualview-weight":
        cmd += ["--rr-weight", "0"]
    with open_out(LOGD, f"{name}_pt_seed0.log", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    ok = rc == 0 and (ck / "encoder_group.pth").exists()
    log(f"{name} PT({flag}={weight}): exit={rc} {'OK' if ok else 'FAIL'}")
    return ck if ok else None


def main():
    LOGD.mkdir(parents=True, exist_ok=True)
    (OUTD / "feat").mkdir(parents=True, exist_ok=True)
    (OUTD / "predictions").mkdir(parents=True, exist_ok=True)
    # E5: 两臂 seed0
    for name, flag, weight, hp in (
        ("dvo", "--dualview-weight", "0.01",
         "W8 E5: 仅双视角 L_BT+0.01*L_dv(rr=0), seed0, NFH 100ep; 与 W7 E1 联合 3-seed 并列描述性"),
        ("rro", "--rr-weight", "0.1",
         "W8 E5: 仅 RR L_BT+0.1*L_rr(dv=0), seed0, NFH 100ep; 与 W7 E1 联合 3-seed 并列描述性")):
        log(f"=== E5 {name}: PT 启动 ===")
        ck = e5_pt(name, flag, weight)
        if ck is None:
            log(f"{name} PT 失败, 停链")
            sys.exit(1)
        for ds, nc in E5_DOMS:
            log(f"{name} lp/{ds}: 启动")
            if not run_lp(E5_OUT, ck, 0, ds, nc, name, hp, "w8-dvsplit"):
                sys.exit(1)
    # E7: b0trc chapman LP 3 seeds(W7 E2c 冻结件纯评测)
    for seed in (0, 2, 4):
        ck = E7CK / f"seed{seed}"
        if not (ck / "encoder_group.pth").exists():
            log(f"b0trc seed{seed} checkpoint 缺失, 跳过")
            continue
        log(f"E7 b0trc s{seed} lp/chapman: 启动")
        if not run_lp(E7_OUT, ck, seed, "chapman", 4, "b0trc",
                      "W8 E7: W7 E2c 冻结 b0trc(ptbxl 语料+trc1) -> Chapman LP 补格, 见 runlog/W7/e2c_results.csv",
                      "w8-b0trc"):
            sys.exit(1)
    log("E5+E7 链完成")


if __name__ == "__main__":
    main()
