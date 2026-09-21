"""confirm_matrix.py — W2 确认矩阵驱动器 (任务书 08 §3.1/§3.3, 2026-09-21)。

C1(--trc 0)/C2(--trc 1) × seeds{0,2,4} NFH 100ep + B0-local seeds{2,4} PTB 200ep
→ 下游按优先级: CPSC LP → PTB FT10 → PTB LP → Chapman LP(血缘域最后)。
特性: 幂等(checkpoint/config 完成即跳)、PT 带 --resume 断点续训、两槽并发、
     全元数据账本 runlog/W2/confirm_results.csv(从各 checkpoint config.json 读 git_sha/sha256)。
路径: 全部相对仓库根(Path(__file__) 推导); PY 可用环境变量 DL_PY 覆盖。
"""
import csv
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = os.environ.get("DL_PY", r"C:\Users\admin\.conda\envs\DL\python.exe")
OUT = ROOT / "runlog/W2/confirm_results.csv"
LOGF = ROOT / "runlog/W2/confirm_matrix.log"
SEEDS = (0, 2, 4)


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGF, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def pt_done(ckdir):
    return (ckdir / "encoder_group.pth").exists() and (ckdir / "config.json").exists()


def ckpt_meta(ckdir):
    try:
        cfg = json.loads((ckdir / "config.json").read_text(encoding="utf-8"))
        return cfg.get("git_sha", "?"), cfg.get("checkpoint_sha256", "?")
    except Exception:
        return "?", "?"


def row_done(ckpt, seed, eval_t, ds):
    if not OUT.exists():
        return False
    for r in csv.DictReader(open(OUT, encoding="utf-8")):
        if (r["ckpt"], r["seed"], r["eval"], r["downstream"]) == (ckpt, seed, eval_t, ds):
            return True
    return False


def record(ckpt, seed, eval_t, ds, auroc, auprc, ckdir):
    sha, cksum = ckpt_meta(ckdir)
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "ckpt", "seed", "eval", "downstream", "auroc", "auprc",
                        "git_sha", "checkpoint_sha256"])
        w.writerow([time.strftime("%m-%d %H:%M"), ckpt, seed, eval_t, ds,
                    f"{auroc:.4f}", f"{auprc:.4f}", sha[:10], cksum[:12]])
    log(f"  ✔ {ckpt} s{seed} {eval_t}/{ds}: {auroc:.4f}/{auprc:.4f}")


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))


def run_pt(kind, seed):
    ckdir = ROOT / f"checkpoint/confirm/{kind}_seed{seed}"
    tag = f"PT {kind} s{seed}"
    if pt_done(ckdir):
        log(f"{tag}: 已完成, 跳过")
        return True
    ckdir.mkdir(parents=True, exist_ok=True)
    data = "data/pt_pretrain_nfh" if kind in ("c1", "c2") else "data/pt_pretrain"
    epochs = "100" if kind in ("c1", "c2") else "200"
    trc = "1" if kind == "c2" else "0"
    log(f"{tag}: 启动 ({data} {epochs}ep trc={trc})")
    r = sh([PY, "-u", "run_pt.py", "--data-dir", data, "--epochs", epochs,
            "--batch-size", "128", "--workers", "4", "--seed", str(seed),
            "--trc", trc, "--resume", "--checkpoint-dir", str(ckdir)])
    ok = r.returncode == 0 and pt_done(ckdir)
    log(f"{tag}: exit={r.returncode} {'OK' if ok else 'FAIL: ' + (r.stdout or r.stderr or '')[-300:]}")
    return ok


def run_lp(kind, seed, ds, nc, trc):
    ckdir = ROOT / f"checkpoint/confirm/{kind}_seed{seed}"
    tag = f"LP {kind} s{seed} {ds}"
    if row_done(kind, str(seed), "lp", ds):
        log(f"{tag}: 已有账, 跳过")
        return True
    feat = ROOT / f"results/confirm/{kind}_{ds}_seed{seed}"
    r = sh([PY, "-u", "run_lp.py", "--data-dir", f"data/{ds}", "--num-classes", str(nc),
            "--checkpoint", str(ckdir / "encoder_group.pth"), "--feat-dir", str(feat),
            "--seed", str(seed), "--workers", "6", "--trc", trc])
    try:
        m = json.loads((feat / "metrics.json").read_text())
        record(kind, seed, "lp", ds, m["auroc"], m["auprc"], ckdir)
        return True
    except Exception:
        log(f"{tag}: FAIL rc={r.returncode} " + (r.stdout or r.stderr or "")[-300:])
        return False


def run_ft10(kind, seed, trc):
    ckdir = ROOT / f"checkpoint/confirm/{kind}_seed{seed}"
    tag = f"FT10 {kind} s{seed}"
    if row_done(kind, str(seed), "ft10", "ptbxl"):
        log(f"{tag}: 已有账, 跳过")
        return True
    mdir = ROOT / f"ft_models/confirm/{kind}_seed{seed}"
    r = sh([PY, "-u", "run_ft.py", "--data-dir", "data/ptbxl", "--num-classes", "5",
            "--fraction", "0.1", "--checkpoint", str(ckdir / "encoder_group.pth"),
            "--model-dir", str(mdir), "--workers", "4", "--epochs", "100",
            "--batch-size", "128", "--learning-rate", "0.0001", "--seed", str(seed),
            "--trc", trc])
    try:
        out = r.stdout
        au = float(out.split("AUROC:")[1].split()[0])
        ap = float(out.split("AUPRC:")[1].split()[0])
        record(kind, seed, "ft10", "ptbxl", au, ap, ckdir)
        return True
    except Exception:
        log(f"{tag}: FAIL rc={r.returncode} " + (r.stdout or r.stderr or "")[-300:])
        return False


def main():
    import sys
    scope = sys.argv[1] if len(sys.argv) > 1 else "all"
    assert scope in ("all", "c2", "c1", "b0"), "用法: confirm_matrix.py [all|c2|c1|b0]"
    ROOT.joinpath("runlog/W2").mkdir(exist_ok=True)
    ROOT.joinpath("results/confirm").mkdir(parents=True, exist_ok=True)
    # ---- Phase 1: 预训练 ----
    plans = {
        "c2": [("c2", s) for s in SEEDS],
        "c1": [("c1", s) for s in SEEDS],
        "b0": [("b0", s) for s in (2, 4)],
    }
    order = [t for k in ("c2", "c1", "b0") for t in plans[k]] if scope == "all" else plans[scope]
    for kind, seed in order:
        run_pt(kind, seed)
    # ---- Phase 2: 下游(§3.1 优先级: CPSC LP → PTB FT10 → PTB LP → Chapman) ----
    trc_of = {"c2": "1", "c1": "0", "b0": "0"}
    kinds = [scope] if scope != "all" else ["c2", "c1", "b0"]
    for s in SEEDS:
        for k in kinds:
            if k == "b0" and s not in (2, 4):
                continue
            run_lp(k, s, "cpsc", 9, trc_of[k]) if k != "b0" else None
    for s in SEEDS:
        for k in kinds:
            if k == "b0" and s not in (2, 4):
                continue
            run_ft10(k, s, trc_of[k])
    for s in SEEDS:
        for k in kinds:
            if k == "b0" and s not in (2, 4):
                continue
            run_lp(k, s, "ptbxl", 5, trc_of[k])
    for s in SEEDS:
        for k in kinds:
            if k == "b0":
                continue
            run_lp(k, s, "chapman", 4, trc_of[k])
    log(f"CONFIRM_MATRIX_DONE scope={scope}")


if __name__ == "__main__":
    main()
