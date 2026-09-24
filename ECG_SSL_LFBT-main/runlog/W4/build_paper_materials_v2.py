# -*- coding: utf-8 -*-
"""W4 B-4b: 把外部基线行并入复算管线, 生成 runlog/W4/paper_materials_v2/。

输入(全部只读):
  runlog/W4/baseline_results.csv       A 侧 31 行账本(A-4~A-7, git_sha=2468a14)
  runlog/W3/paper_materials/main_table_recomputed.csv   W3 主表(b0/c1/c2)
  runlog/W4/stats/paired_stats.csv     B-4a 逐记录配对统计(存在才并入)

输出:
  paper_materials_v2/baseline_table_v2.csv        基线行聚合(mean±sd, 3-seed)
  paper_materials_v2/main_table_merged_v2.csv     W3 主表+基线行合并总表
  paper_materials_v2/paired_stats_v2.csv          统计表(加 source 列; 若存在)
  paper_materials_v2/paper_materials_v2_manifest.json

全部数字附 source_file / source_git_sha(任务书 B-4 要求)。
用法: 在 ECG_SSL_LFBT-main/ 下 python runlog/W4/build_paper_materials_v2.py
"""
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.pathguard import open_out  # noqa: E402

W4 = ROOT / "runlog" / "W4"
W3PM = ROOT / "runlog" / "W3" / "paper_materials"
OUT = W4 / "paper_materials_v2"

A_SHA = "2468a14"  # A 侧账本记录的 git_sha(baseline_results.csv 全部行一致)
B_SHA = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True).stdout.strip()

AGG_SRC = "runlog/W4/baseline_results.csv"


def read_rows(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fmt(x):
    return f"{x:.4f}"


def build_baseline_table():
    """(ckpt, eval, downstream) 聚合 -> mean/sd/per_seed/mean_pm_sd + 溯源列。"""
    rows = read_rows(W4 / "baseline_results.csv")
    groups = {}
    for r in rows:
        key = (r["ckpt"], r["eval"], r["downstream"])
        groups.setdefault(key, []).append(r)
    out = []
    for (ckpt, ev, ds), rs in sorted(groups.items()):
        aurocs = sorted(rs, key=lambda r: int(r["seed"]))
        vals = [float(r["auroc"]) for r in aurocs]
        seeds = [r["seed"] for r in aurocs]
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out.append({
            "eval": ev, "downstream": ds, "ckpt": ckpt,
            "method_family": aurocs[0]["method_family"],
            "hparams_ref": aurocs[0]["hparams_ref"],
            "n_seeds": len(vals),
            "mean_auroc": fmt(mean), "sd_auroc": fmt(sd),
            "mean_auprc": fmt(statistics.mean(float(r["auprc"]) for r in aurocs)),
            "per_seed_auroc": "/".join(fmt(v) for v in vals),
            "seeds": "/".join(seeds),
            "mean_pm_sd": f"{mean:.4f}±{sd:.4f}",
            "checkpoint_sha256": aurocs[0]["checkpoint_sha256"][:16] + "...",
            "source_file": AGG_SRC, "source_git_sha": A_SHA,
        })
    return out


def build_merged_main(baseline_rows):
    """W3 main_table_recomputed(b0/c1/c2) + W4 基线行 -> 合并主表(统一 schema)。"""
    w3 = read_rows(W3PM / "main_table_recomputed.csv")
    out = []
    for r in w3:
        out.append({
            "eval": r["eval"], "downstream": r["downstream"], "ckpt": r["ckpt"],
            "origin": "W2/W3(冻结)",
            "n_seeds": r["n_seeds"], "mean": r["mean"], "sd": r["sd"],
            "per_seed": r["per_seed"], "mean_pm_sd": r["mean_pm_sd"],
            "source_file": r["source_file"], "source_git_sha": r["source_git_sha"],
        })
    for b in baseline_rows:
        out.append({
            "eval": b["eval"], "downstream": b["downstream"], "ckpt": b["ckpt"],
            "origin": f"W4外部基线({b['method_family']})",
            "n_seeds": b["n_seeds"], "mean": b["mean_auroc"], "sd": b["sd_auroc"],
            "per_seed": b["per_seed_auroc"], "mean_pm_sd": b["mean_pm_sd"],
            "source_file": b["source_file"], "source_git_sha": b["source_git_sha"],
        })
    order = {"eval": 0, "downstream": 1}
    out.sort(key=lambda r: (r["eval"], r["downstream"], r["ckpt"]))
    return out


def build_paired_stats_v2():
    src = W4 / "stats" / "paired_stats.csv"
    if not src.exists():
        return None
    rows = read_rows(src)
    for r in rows:
        r["source_file"] = "runlog/W4/predictions/(A-8 重放+A-4/A-5/A-6 下游)"
        r["source_git_sha"] = B_SHA
        r["level"] = "record(非 patient; 见 stats_methods.md §4)"
    return rows


def write_csv(name, rows):
    if not rows:
        return None
    cols = list(rows[0].keys())
    with open_out(OUT, name, mode="w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] {name}: {len(rows)} 行")
    return name


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = build_baseline_table()
    merged = build_merged_main(baseline)
    paired = build_paired_stats_v2()
    written = [x for x in [
        write_csv("baseline_table_v2.csv", baseline),
        write_csv("main_table_merged_v2.csv", merged),
        write_csv("paired_stats_v2.csv", paired),
    ] if x]
    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M"),
        "built_by": "B-4b (hostB DESKTOP-0PBLCND)",
        "inputs": {
            "baseline_rows": len(baseline),
            "baseline_source_git_sha": A_SHA,
            "stats_present": paired is not None,
            "stats_source_git_sha": B_SHA if paired else None,
        },
        "outputs": written,
        "note": "全部数字附 source_file/source_git_sha; record-level 统计不得声称 patient-level(任务书 B-4)",
    }
    with open_out(OUT, "paper_materials_v2_manifest.json", mode="w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"[done] paper_materials_v2 -> {len(written)} 文件")


if __name__ == "__main__":
    main()
