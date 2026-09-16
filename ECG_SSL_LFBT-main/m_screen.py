# -*- coding: utf-8 -*-
"""m_screen.py — M 矩阵自动筛选管线 (v1)。

用法:
  python m_screen.py --config runlog/M/matrix_batch1.yaml           # 批量粗探(1 seed)
  python m_screen.py --config ... --confirm d1l_05_02               # 对指定配置补 seed 2,4
  python m_screen.py --config ... --plan                            # 只打印计划不执行

配置格式 (yaml):
  baseline: {name: B0, auprc_seed0: 0.7177}
  configs:
    - {name: d1l_05_02, args: "--d1l 0.5,0.2", seeds: [0]}

输出:
  runlog/M/matrix_results.csv  (滚动追加: name,seed,args,auroc,auprc,delta,fast,ts)
  checkpoint/M/<name>_seed<k>/  feat/M_<name>_seed<k>/
判号: delta = auprc - baseline.auprc_seed0 (粗探); --confirm 后按同 seed 配对算 delta。
"""
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import yaml

PY = sys.executable
ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoint" / "M"
FEAT = ROOT / "feat"
LOGD = ROOT / "runlog" / "M"
RESD = LOGD / "matrix_results.csv"
ANCHOR = {"B0": 0.7177, "B0_seed2": 0.7258}  # 冻结协议锚点(2026-09-16)


def sh(cmd):
    print("[m_screen] $", " ".join(map(str, cmd)), flush=True)
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True)


def parse_lp(log_path):
    auroc = auprc = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("AUROC"):
            auroc = float(line.split("=")[1])
        elif line.startswith("AUPRC"):
            auprc = float(line.split("=")[1])
    return auroc, auprc


def append_result(row):
    new = not RESD.exists()
    RESD.parent.mkdir(parents=True, exist_ok=True)
    with RESD.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ts", "name", "seed", "args", "auroc",
                                          "auprc", "delta", "fast"])
        if new:
            w.writeheader()
        w.writerow(row)


def done_names():
    if not RESD.exists():
        return set()
    with RESD.open(encoding="utf-8") as f:
        return {(r["name"], r["seed"]) for r in csv.DictReader(f)}


def run_one(cfg, seed, fast, data_dir, baseline_auprc):
    name = cfg["name"]
    tag = f"{name}_seed{seed}"
    ck = CKPT / tag
    lp_log = LOGD / f"lp_{tag}.log"
    pt_log = LOGD / f"pt_{tag}.log"

    if lp_log.exists() and "AUPRC" in lp_log.read_text(errors="replace"):
        print(f"[m_screen] 跳过(已完成): {tag}")
    else:
        cmd = [PY, "run_pt.py", "--data-dir", data_dir, "--epochs", 200,
               "--batch-size", 128, "--workers", 6, "--seed", seed,
               "--checkpoint-dir", ck]
        if fast:
            cmd.append("--fast-backbone")
        cmd += cfg["args"].split()
        r = sh(cmd)
        pt_log.write_text(r.stdout + r.stderr, encoding="utf-8")
        if "SHA256" not in r.stdout:
            print(f"[m_screen] 预训练失败 {tag}:", r.stderr[-400:], flush=True)
            return None

    r = sh([PY, "run_lp.py", "--data-dir", "data/ptbxl",
            "--checkpoint", ck / "encoder_group.pth", "--num-classes", 5,
            "--feat-dir", FEAT / f"M_{tag}", "--seed", 0, "--workers", 6])
    lp_log.write_text(r.stdout + r.stderr, encoding="utf-8")
    auroc, auprc = parse_lp(lp_log)
    if auprc is None:
        print(f"[m_screen] LP 失败 {tag}:", r.stderr[-300:], flush=True)
        return None
    delta = round(auprc - baseline_auprc, 4)
    append_result({"ts": time.strftime("%m-%d %H:%M"), "name": name, "seed": seed,
                   "args": cfg["args"], "auroc": round(auroc, 4),
                   "auprc": round(auprc, 4), "delta": delta, "fast": int(fast)})
    print(f"[m_screen] ✔ {tag}: AUPRC={auprc:.4f} Δ={delta:+.4f}", flush=True)
    return auprc


def main():
    ap = argparse.ArgumentParser(description="M 矩阵自动筛选")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--fast", action="store_true", help="预训练用 fast-backbone 提速路径")
    ap.add_argument("--data-dir", default="data/pt_pretrain")
    ap.add_argument("--confirm", nargs="*", help="对指定配置补 seed 2,4(默认全部 mean>0 的)")
    ap.add_argument("--plan", action="store_true", help="只打印计划")
    args = ap.parse_args()

    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base_auprc = spec.get("baseline", {}).get("auprc_seed0", ANCHOR["B0"])
    configs = spec["configs"]
    done = done_names()

    plan = []
    for cfg in configs:
        seeds = cfg.get("seeds", [0])
        if args.confirm is not None:
            seeds = [2, 4]
            if args.confirm:  # 指定了名字
                seeds = seeds if cfg["name"] in args.confirm else []
        for s in seeds:
            if (cfg["name"], str(s)) in done:
                continue
            plan.append((cfg, s))

    print(f"[m_screen] 计划 {len(plan)} 个 run (fast={args.fast}, 基线 Δ 锚点 {base_auprc})")
    for cfg, s in plan:
        print(f"  - {cfg['name']} seed{s}: {cfg['args']}")
    if args.plan:
        return

    for cfg, s in plan:
        run_one(cfg, s, args.fast, args.data_dir, base_auprc)

    # 批末小结
    if RESD.exists():
        with RESD.open(encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f)]
        print("\n[m_screen] 当前矩阵:")
        for r in rows[-len(plan):]:
            print(f"  {r['name']:<18} s{r['seed']} Δ={r['delta']} {'✅' if float(r['delta'])>0 else '❌'}")


if __name__ == "__main__":
    main()
