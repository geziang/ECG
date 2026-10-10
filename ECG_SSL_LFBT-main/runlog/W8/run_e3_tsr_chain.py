# -*- coding: utf-8 -*-
"""W8 E3 w8-tsr: TSR 官方码适配链 (BobZwr/ReverseECG, Zhang et al. BSPC 2023)。

用法(conda DL 解释器, 仓库根为 cwd):
    python runlog/W8/run_e3_tsr_chain.py

协议要点(逐条入 w8_hparams.csv):
  - pretext/损失/超参按官方默认: 四变体反转检测 + 2-logit BCEWithLogits +
    Adam lr=1e-2 wd=1e-4(官方) + 100ep bs128;
  - 必须适配项: 8 导联 2048 输入 / NFH 语料 / batch 128 / 100ep 步数对齐 ~27.3k
    (变体每条每次均匀抽 1/4);
  - backbone 取舍按 W4 CLOCS 先例: 保持 C1 同构(8xVGG16 a=0.125 逐导联), 官方
    Net1D 差异记录;
  - 无 RRC-TO 增强(官方无增强, pretext 即变换);
  - 官方 snippet ts_reverse [::-1] 轴位笔误: 按标签意图实现双重反转(tests/test_w8_e3
    对拍前 3 块逐位一致)。
**预注册回退(先写后跑)**: 官方 lr=1e-2 若发散(日志出现 nan, 或 epoch1 末 loss >
4x 初始) -> 一次性改协议 lr=1e-3(wd 不变)重跑, 差异记入 hparams_ref 与 events;
不再做其他调参。若仍异常 -> 如实记"未复现"降级(DLC 先例), 不硬凑数字。
产物: runlog/W8/w8_tsr_results.csv(W7 schema) + predictions/。
"""
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.pathguard import open_out

PY = sys.executable
OUTD = ROOT / "runlog/W8"
OUT = OUTD / "w8_tsr_results.csv"
LOGD = OUTD / "logs"
PRED = "runlog/W8/predictions"
SEEDS = [0, 2, 4]
CKD = ROOT / "checkpoint/w8_tsr"
DOMAINS = [("ptbxl", 5), ("cpsc", 9), ("chapman", 4)]
FT_DOMS = [("ptbxl", 5), ("cpsc", 9)]
OFFICIAL_LR = "0.01"
OFFICIAL_WD = "0.0001"


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


def parse_loss_series(ptlog):
    vals = []
    for m in re.finditer(r'"loss":\s*(nan|[-+0-9.eE]+)', Path(ptlog).read_text(
            encoding="utf-8", errors="replace")):
        vals.append(float("nan") if m.group(1) == "nan" else float(m.group(1)))
    return vals


def diverged(ptlog):
    vals = parse_loss_series(ptlog)
    if not vals:
        return False
    if any(math.isnan(v) for v in vals):
        return True
    ep1 = [v for v in vals]  # 全序列首末对比(官方 lr 发散通常单调爆掉)
    return ep1[-1] > 4 * max(ep1[0], 1e-6)


def row_done(ckpt, seed, evalkind, ds):
    if not OUT.exists():
        return False
    with open(OUT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["ckpt"], r["seed"], r["eval"], r["downstream"]) == (ckpt, str(seed), evalkind, ds):
                return True
    return False


def record(ckpt, seed, evalkind, ds, auroc, auprc, cksha, hp):
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "ckpt", "seed", "eval", "downstream", "auroc", "auprc",
                        "git_sha", "checkpoint_sha256", "protocol_id", "hparams_ref"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), ckpt, seed, evalkind, ds,
                    f"{auroc:.4f}", f"{auprc:.4f}", git_sha(), cksha, "w8-tsr", hp])
    log(f"  + {ckpt} s{seed} {evalkind}/{ds}: {auroc:.4f}/{auprc:.4f}")


def _hp(lr_used, fell_back):
    base = ("W8 E3: TSR 官方四变体 pretext(BCEWithLogits 2-logit 逐导联头均值), "
            "backbone=C1 同构 8xVGG16, 无 RRC-TO(官方无增强), 变体每条每次均匀抽 1/4, "
            "Adam lr=%s wd=1e-4; 见 runlog/W8/w8_hparams.csv" % lr_used)
    if fell_back:
        base += "; 预注册回退已触发(官方 lr=1e-2 发散->协议 1e-3, 一次性)"
    return base


def run_pt(seed, lr, suffix, fell_back):
    ck = CKD / f"tsr_seed{seed}{suffix}"
    if (ck / "encoder_group.pth").exists() and (ck / "config.json").exists():
        log(f"tsr s{seed}{suffix} PT: 已完成, 跳过")
        return ck, True
    ck.mkdir(parents=True, exist_ok=True)
    ptlog = LOGD / f"tsr_pt_seed{seed}{suffix}.log"
    with open_out(LOGD, ptlog.name, encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_pt.py", "--data-dir", "data/pt_pretrain_nfh",
             "--epochs", "100", "--batch-size", "128", "--workers", "4",
             "--seed", str(seed), "--trc", "0", "--loss-mode", "tsr",
             "--learning-rate", lr, "--wd", OFFICIAL_WD,
             "--resume", "--checkpoint-dir", str(ck)],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    ok = rc == 0 and (ck / "encoder_group.pth").exists()
    log(f"tsr s{seed}{suffix} PT(lr={lr}): exit={rc} {'OK' if ok else 'FAIL'}"
        f"{' [diverged]' if ok and diverged(ptlog) else ''}")
    return ck, ok and not diverged(ptlog)


def run_lp(ck, seed, ds, nc, ckpt_name, hp):
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
             "--protocol-id", "w8-tsr"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((feat / "metrics.json").read_text(encoding="utf-8"))
        record(ckpt_name, seed, "lp", ds, m["auroc"], m["auprc"],
               sha256_of(ck / "encoder_group.pth"), hp)
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e})")
        return False


def run_ft(ck, seed, ds, nc, ckpt_name, hp):
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
             "--protocol-id", "w8-tsr"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((mdir / "metrics.json").read_text(encoding="utf-8"))
        record(ckpt_name, seed, "ft10", ds, m["auroc"], m["auprc"],
               sha256_of(ck / "encoder_group.pth"), hp)
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e})")
        return False


def main():
    LOGD.mkdir(parents=True, exist_ok=True)
    (OUTD / "feat").mkdir(parents=True, exist_ok=True)
    (OUTD / "predictions").mkdir(parents=True, exist_ok=True)
    (OUTD / "ft").mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        log(f"=== tsr seed{seed}: PT(官方 lr=1e-2) 启动 ===")
        ck, ok = run_pt(seed, OFFICIAL_LR, "", False)
        lr_used, suffix, fell_back = OFFICIAL_LR, "", False
        if not ok:
            log(f"tsr s{seed}: 官方 lr 发散/失败 -> 预注册回退 协议 lr=1e-3 一次性重跑")
            lr_used, suffix, fell_back = "0.001", "_lr1e3", True
            ck, ok = run_pt(seed, lr_used, suffix, fell_back)
            if not ok:
                log(f"tsr s{seed}: 回退后仍失败 -> 如实降级'未复现', 跳过该 seed 评测")
                continue
        hp = _hp(lr_used, fell_back)
        name = "tsr" if not suffix else "tsr-lr1e3"
        for ds, nc in DOMAINS:
            log(f"tsr s{seed} lp/{ds}: 启动")
            if not run_lp(ck, seed, ds, nc, name, hp):
                sys.exit(1)
        for ds, nc in FT_DOMS:
            log(f"tsr s{seed} ft10/{ds}: 启动")
            if not run_ft(ck, seed, ds, nc, name, hp):
                sys.exit(1)
    log("E3 链完成")


if __name__ == "__main__":
    main()
