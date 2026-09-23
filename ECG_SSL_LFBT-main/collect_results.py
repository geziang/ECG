"""collect_results.py — 汇总所有下游实验的 metrics.json 为 results.csv

用法: python collect_results.py [--results-dir results] [--out results.csv]
"""
import argparse
import csv
import json
import os

from utils.pathguard import open_out_file

EXPERIMENT_ORDER = [
    "ptbxl_lp", "ptbxl_lp_tfs",
    "ptbxl_ft_1.0", "ptbxl_ft_0.1", "ptbxl_ft_tfs", "ptbxl_ft_tfs_0.1",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    rows = []
    for root, dirs, files in os.walk(args.results_dir):
        for f in files:
            if f != "metrics.json":
                continue
            path = os.path.join(root, f)
            with open(path, encoding="utf-8") as fh:
                m = json.load(fh)
            exp = os.path.basename(root)
            rows.append(dict(
                experiment=exp,
                auroc=round(m.get("auroc", float("nan")), 4),
                auprc=round(m.get("auprc", float("nan")), 4),
                checkpoint=m.get("checkpoint", ""),
                fraction=m.get("fraction", ""),
                num_classes=m.get("num_classes", ""),
                note="",
            ))

    # 按预定义顺序排序, 其余按名称
    rank = {e: i for i, e in enumerate(EXPERIMENT_ORDER)}
    rows.sort(key=lambda r: (rank.get(r["experiment"], 99), r["experiment"]))

    with open_out_file(args.out, newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["experiment", "auroc", "auprc", "checkpoint", "fraction", "num_classes", "note"])
        w.writeheader()
        w.writerows(rows)

    print(f"已写入 {args.out}: {len(rows)} 个实验")
    for r in rows:
        print(f"  {r['experiment']:<16} AUROC={r['auroc']:<8} AUPRC={r['auprc']}")


if __name__ == "__main__":
    main()
