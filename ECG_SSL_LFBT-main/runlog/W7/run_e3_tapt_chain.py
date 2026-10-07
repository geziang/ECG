# -*- coding: utf-8 -*-
"""W7 E3: 目标域 SSL 续训(TAPT)——C1 三种子在 CPSC 去标签语料上小步长续 BT, 攻血缘优势。

协议(队列 Q7 预注册): C1 confirm/seed{0,2,4} 的 train_state.pth 拷入 checkpoint/w7_e3/c1_cpsc_s{S}
→ run_pt --data-dir data/cpsc/train(9 类子目录=伪标签, SSL 不用) --epochs 150(=100 原始+50 续)
   --resume --cont-lr 0.0001(优化器恢复后覆盖步长) --trc 0 --batch-size 128 --seed S(更新数≈1.9k)
→ cpsc(9类)+ptbxl(5类) LP(trc=0), 账本 runlog/W7/e3_tapt_results.csv, protocol_id=w7-e3。
门: cpsc LP 3-seed mean Δ≥+0.5pt vs C1(confirm_results.csv 同种子) 且 ptbxl 掉幅≤0.3pt;
   过门→加 chapman 同构; 判负→与混合语料 −3.07pt 合并写"语料先验注入方式边界"。
注: 顺序续训≠已判负的混合语料重跑(新假设新线)。
用法: python runlog/W7/run_e3_tapt_chain.py
"""
import csv
import hashlib
import json
import shutil
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
OUT = OUTD / "e3_tapt_results.csv"
SEEDS = [0, 2, 4]
DOMAINS = [("cpsc", 9), ("ptbxl", 5)]
C1_REF = ROOT / "runlog/W2/confirm_results.csv"


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


def row_done(seed, ev, ds):
    if not OUT.exists():
        return False
    with open(OUT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["seed"], r["eval"], r["downstream"]) == (str(seed), ev, ds):
                return True
    return False


def record(seed, ev, ds, auroc, auprc, cksha):
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "ckpt", "seed", "eval", "downstream", "auroc", "auprc",
                        "git_sha", "checkpoint_sha256", "protocol_id", "hparams_ref"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), "c1tapt", seed, ev, ds,
                    f"{auroc:.4f}", f"{auprc:.4f}", git_sha(), cksha, "w7-e3",
                    "C1 + BT续训@cpsc_train 50ep lr1e-4, 见 run_pt.py --cont-lr"])
    log(f"  + e3 s{seed} {ev}/{ds}: {auroc:.4f}/{auprc:.4f}")


def run_pt(seed):
    src = ROOT / f"checkpoint/confirm/c1_seed{seed}/train_state.pth"
    ck = ROOT / f"checkpoint/w7_e3/c1_cpsc_s{seed}"
    if (ck / "encoder_group.pth").exists() and (ck / "config.json").exists():
        log(f"e3 PT s{seed}: 已完成, 跳过")
        return True
    if not src.exists():
        log(f"e3 PT s{seed}: 源 train_state 缺失 {src}")
        return False
    ck.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, ck / "train_state.pth")
    with open_out(LOGD, f"e3_tapt_pt_s{seed}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_pt.py", "--data-dir", "data/cpsc/train",
             "--epochs", "150", "--batch-size", "128", "--workers", "4",
             "--seed", str(seed), "--trc", "0",
             "--resume", "--cont-lr", "0.0001",
             "--checkpoint-dir", str(ck)],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    ok = rc == 0 and (ck / "encoder_group.pth").exists()
    log(f"e3 PT s{seed}: exit={rc} {'OK' if ok else 'FAIL'}")
    return ok


def run_lp(seed, ds, nc):
    if row_done(seed, "lp", ds):
        log(f"e3 s{seed} lp/{ds}: 已有账, 跳过")
        return True
    ckpt = ROOT / f"checkpoint/w7_e3/c1_cpsc_s{seed}/encoder_group.pth"
    tag = f"e3_c1tapt_{ds}_lp_seed{seed}"
    feat = OUTD / "feat" / tag
    with open_out(LOGD, f"{tag}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_lp.py", "--data-dir", f"data/{ds}", "--num-classes", str(nc),
             "--checkpoint", str(ckpt), "--feat-dir", str(feat),
             "--seed", str(seed), "--workers", "6", "--trc", "0",
             "--extended-metrics", "1",
             "--save-predictions", str(OUTD / "predictions" / tag),
             "--protocol-id", "w7-e3"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((feat / "metrics.json").read_text(encoding="utf-8"))
        record(seed, "lp", ds, m["auroc"], m["auprc"], sha256_of(ckpt))
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e})")
        return False


def gate_summary():
    ours, ref = {}, {}
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ours[(r["seed"], r["downstream"])] = float(r["auroc"])
    if C1_REF.exists():
        with open(C1_REF, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["ckpt"] == "c1" and r["eval"] == "lp":
                    ref[(r["seed"], r["downstream"])] = float(r["auroc"])
    lines = ["# E3 TAPT 续训门判定(队列 Q7 预注册)", "",
             "| seed | cpsc: C1→TAPT (Δ) | ptbxl: C1→TAPT (Δ) |", "|---|---|---|"]
    d_cp, d_pt = [], []
    for s in map(str, SEEDS):
        a = (ours.get((s, "cpsc")), ours.get((s, "ptbxl")))
        rc_, rp_ = ref.get((s, "cpsc")), ref.get((s, "ptbxl"))
        if None not in a and rc_ is not None and rp_ is not None:
            d_cp.append(a[0] - rc_)
            d_pt.append(a[1] - rp_)
            lines.append(f"| {s} | {rc_:.4f}→{a[0]:.4f} ({(a[0]-rc_)*100:+.2f}pt) | "
                         f"{rp_:.4f}→{a[1]:.4f} ({(a[1]-rp_)*100:+.2f}pt) |")
    if d_cp:
        import numpy as np
        m_cp, m_pt = np.mean(d_cp) * 100, np.mean(d_pt) * 100
        same = all(d > 0 for d in d_cp)
        verdict = ("✅过门(cpsc≥+0.5 且 ptbxl 无伤≤0.3)→加 chapman 同构" if same and m_cp >= 0.5 and m_pt >= -0.3
                   else "❌判负→与混合语料−3.07合并写'语料先验注入方式边界'一节")
        lines += ["", f"- cpsc mean Δ = {m_cp:+.2f}pt ({len(d_cp)}/3 种子, 同向 {'是' if same else '否'})",
                  f"- ptbxl mean Δ = {m_pt:+.2f}pt (无伤容差 −0.3pt)", f"- 判定: {verdict}"]
    (OUTD / "e3_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("gate summary written")


if __name__ == "__main__":
    log("E3 TAPT 链启动(C1×3种子 × cpsc续训50ep + 双域LP)")
    for seed in SEEDS:
        if run_pt(seed):
            for ds, nc in DOMAINS:
                run_lp(seed, ds, nc)
    gate_summary()
    log("E3_CHAIN_END")
