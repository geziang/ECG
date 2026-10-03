# -*- coding: utf-8 -*-
"""W5B B-3(W6 Stage0 执行): CPSC 修正分区论文材料 v3。

输入(全部只读):
  runlog/W4/paper_materials_v2/  W4 v2 材料(PTB/Chapman 行原样沿用; W4 文件只读不改)
  runlog/W3/paper_materials/     W3 配对差与符号表(非 cpsc 行原样沿用)
  runlog/W5/lp_results.csv       W5A 重评估账本(15 行, git_sha=32ec160)
  runlog/W5/stats/paired_stats.csv  W6 Stage0 B-1 官方配对统计(cpsc 四行)

输出 runlog/W5/paper_materials_v3/:
  main_table_merged_v3.csv   非 cpsc 行=v2 原样; cpsc LP 5 方法行全部换 W5 数
  paired_stats_v3.csv        ptbxl/chapman 行=v2 原样; cpsc 四行=W5 新数
  paired_delta_v3.csv        非 cpsc 行=W3 原样(+metric=auroc); cpsc 行=B-2 新算
                            (auroc+auprc 双口径, perm_p/t-CI 与 W3 管线同式)
  seed_sign_v3.csv           W3 行原样 + cpsc 4 行(AUPRC 口径, B-2)
  README.md / paper_materials_v3_manifest.json

红线: W2/W3/W4 冻结产物只读; PTB/Chapman 统计行不重算; 全部 record-level 措辞。
用法: 在 ECG_SSL_LFBT-main/ 下 python runlog/W5/build_paper_materials_v3.py
"""
import csv
import json
import statistics
import subprocess
import sys
import time
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.pathguard import open_out  # noqa: E402

W5 = ROOT / "runlog" / "W5"
V2 = ROOT / "runlog" / "W4" / "paper_materials_v2"
W3PM = ROOT / "runlog" / "W3" / "paper_materials"
OUT = W5 / "paper_materials_v3"

W5_SHA = "32ec160"  # runlog/W5/lp_results.csv 各行记录的 git_sha(W5A 产出时点)
HEAD = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
T975_DF2 = 4.302651  # t_{0.975, df=2}
SEEDS = [0, 2, 4]


def read_rows(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def perm_p(deltas):
    """精确符号翻转置换, 单侧(与 W3 build_paper_materials.py 逐字同式)。"""
    obs = sum(deltas)
    hit = tot = 0
    for mask in range(1 << len(deltas)):
        s = sum(d if (mask >> i) & 1 else -d for i, d in enumerate(deltas))
        tot += 1
        hit += (s >= obs - 1e-12)
    return hit / tot


def load_w5_lp():
    """-> {(ckpt, seed): (auroc, auprc)}"""
    out = {}
    for r in read_rows(W5 / "lp_results.csv"):
        out[(r["ckpt"], int(r["seed"]))] = (float(r["auroc"]), float(r["auprc"]))
    return out


def build_main_v3():
    """v2 主表: 非 cpsc 行原样; cpsc LP 行全部换 W5 数(b0 的两条历史行一并替换单行)。"""
    rows = read_rows(V2 / "main_table_merged_v2.csv")
    kept = [r for r in rows if r["downstream"] != "cpsc"]
    lp = load_w5_lp()
    for ckpt in ["b0", "c1", "c2", "simclr", "clocs"]:
        vals = [lp[(ckpt, s)][0] for s in SEEDS]
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals)
        kept.append({
            "eval": "lp", "downstream": "cpsc", "ckpt": ckpt,
            "origin": "W5A修正分区重评估(污染审计后重建test=1385)",
            "n_seeds": "3",
            "mean": f"{mean:.4f}", "sd": f"{sd:.4f}",
            "per_seed": "/".join(f"{v:.4f}" for v in vals),
            "mean_pm_sd": f"{mean:.4f}±{sd:.4f}",
            "source_file": "runlog/W5/lp_results.csv", "source_git_sha": W5_SHA,
        })
    kept.sort(key=lambda r: (r["eval"], r["downstream"], r["ckpt"]))
    return kept


def build_paired_stats_v3():
    """v2 统计表: 非 cpsc 行原样; cpsc 四行 = W5/stats 新数(补 source 列与 level 注记)。"""
    v2rows = read_rows(V2 / "paired_stats_v2.csv")
    kept = [r for r in v2rows if r["dataset"] != "cpsc"]
    for r in read_rows(W5 / "stats" / "paired_stats.csv"):
        r["source_file"] = "runlog/W5/predictions/(W5A A-2 重评估, protocol_id=w5-cpscredo)"
        r["source_git_sha"] = HEAD
        r["level"] = "record(非 patient; 见 stats_methods_w5.md)"
        kept.append(r)
    return kept


def _delta_rows(lp, pair, hi, lo, metric_i, mname):
    ds = [lp[(hi, s)][metric_i] - lp[(lo, s)][metric_i] for s in SEEDS]
    d_pt = [d * 100 for d in ds]
    mean = statistics.mean(d_pt)
    sd = statistics.stdev(d_pt)
    half = T975_DF2 * sd / sqrt(len(SEEDS))
    sign = "".join("+" if d > 0 else ("-" if d < 0 else "0") for d in d_pt)
    return {
        "eval": "lp", "downstream": "cpsc", "pair": f"{hi.upper()}-{lo.upper()}",
        **{f"d_s{s}": f"{d_pt[i]:+.2f}" for i, s in enumerate(SEEDS)},
        "mean_pt": f"{mean:+.2f}", "sign": sign,
        "all_same_direction": "yes" if sign in ("+++", "---") else "no",
        "perm_p_1sided_exact": f"{perm_p(d_pt):.3f}",
        "ci95_seed_level_t": f"[{mean - half:+.2f},{mean + half:+.2f}]",
        "n_seeds": "3",
        "source_file": "runlog/W5/lp_results.csv", "source_git_sha": W5_SHA,
        "metric": mname,
    }, d_pt, sign, mean


def build_paired_delta_v3():
    """W3 配对差表: 非 cpsc 行原样(+metric=auroc 标注); cpsc 行双口径新算(B-2)。"""
    w3rows = read_rows(W3PM / "paired_delta_recomputed.csv")
    kept = []
    for r in w3rows:
        if r["downstream"] == "cpsc":
            continue  # 污染分区历史行不入 v3(README 注明不可比; W3 冻结文件原样保留)
        r["metric"] = "auroc"
        kept.append(r)
    lp = load_w5_lp()
    for hi, lo in [("c2", "c1"), ("c2", "b0"), ("c1", "simclr"), ("c1", "clocs")]:
        for mi, mname in [(0, "auroc"), (1, "auprc")]:
            row, _, _, _ = _delta_rows(lp, None, hi, lo, mi, mname)
            kept.append(row)
    return kept


def build_seed_sign_v3():
    """W3 符号表(非 cpsc 行)原样 + cpsc 4 行(AUPRC 口径 = B-2 任务书指定交付)。"""
    rows = [r for r in read_rows(W3PM / "seed_sign_table.csv")
            if "/cpsc/" not in r["comparison"]]  # 污染分区历史行不入 v3(README 注明)
    lp = load_w5_lp()
    for hi, lo in [("c2", "c1"), ("c2", "b0"), ("c1", "simclr"), ("c1", "clocs")]:
        _, d_pt, sign, mean = _delta_rows(lp, None, hi, lo, 1, "auprc")
        same = sign in ("+++", "---")
        rows.append({
            "comparison": f"lp/cpsc/{hi.upper()}-{lo.upper()}(auprc)",
            "per_seed_pt": "/".join(f"{d:+.2f}" for d in d_pt),
            "sign": sign, "mean_pt": f"{mean:+.2f}",
            "verdict": "3/3同向" if same else "非一致",
            "source_git_sha": W5_SHA,
        })
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
    main3 = build_main_v3()
    ps3 = build_paired_stats_v3()
    pd3 = build_paired_delta_v3()
    ss3 = build_seed_sign_v3()
    written = [x for x in [
        write_csv("main_table_merged_v3.csv", main3),
        write_csv("paired_stats_v3.csv", ps3),
        write_csv("paired_delta_v3.csv", pd3),
        write_csv("seed_sign_v3.csv", ss3),
    ] if x]
    readme = """# paper_materials_v3 · CPSC 修正分区合并材料

**CPSC 行为修正分区数据（W5A 审计重建 test=1385，见 `runlog/W5/cpsc_audit.md`），
与 W2/W4 时代的污染分区行（test=2004 新旧并集）不可比。**
W2/W3/W4 冻结文件原样保留作污染分区历史记录，本目录是其"CPSC 行换新"版本：

- `main_table_merged_v3.csv`: PTB/Chapman 行 = v2 原样（源列不动）；CPSC LP 5 方法行全部换
  W5 数（b0 在 v2 的两条历史行——W2/W3 冻结 n=1 行与 W4 外部基线 n=2 行——同为污染分区产物，
  一并替换为 W5 n=3 单行）。
- `paired_stats_v3.csv`: PTB/Chapman 行 = v2 原样；CPSC 四行 = `runlog/W5/stats/paired_stats.csv`
  （RNG 20260923, 10k 次, record-level 配对 bootstrap）。
- `paired_delta_v3.csv` / `seed_sign_v3.csv`: 非 cpsc 行 = W3 原样（补 metric 列标注 auroc）；
  CPSC 行 = W5 账本种子级新算，**auroc 与 auprc 双口径**（B-2 指定交付为 AUPRC 口径：
  C2−C1 = +0.70/+0.59/+1.20 mean +0.83, 3/3⊕, t-CI [+0.02,+1.64]）。
  W3 表中被替换剔除的唯一 cpsc 行（lp/cpsc/C2-C1 auroc +2.30, 污染分区）仍留在 W3 冻结文件中。

统计口径: 全部 record-level 描述性证据, 禁写"统计显著/patient-level"(红线)。
逐记录预测输入: `runlog/W5/predictions/`(15 套, protocol_id=w5-cpscredo)。
"""
    with open_out(OUT, "README.md", mode="w", encoding="utf-8") as f:
        f.write(readme)
    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M"),
        "built_by": "W6 Stage0 B-3 (hostA Win4090)",
        "inputs": {
            "v2_dir": "runlog/W4/paper_materials_v2 (read-only)",
            "w3_dir": "runlog/W3/paper_materials (read-only)",
            "w5_ledger": "runlog/W5/lp_results.csv (git_sha 32ec160)",
            "w5_stats": "runlog/W5/stats/paired_stats.csv",
        },
        "outputs": written,
        "w5_ledger_git_sha": W5_SHA,
        "build_git_sha": HEAD,
        "note": "CPSC 行=修正分区(1385), 与 W2/W4 污染分区行不可比; record-level",
    }
    with open_out(OUT, "paper_materials_v3_manifest.json", mode="w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"[done] paper_materials_v3 -> {len(written)} 文件 + README + manifest")


if __name__ == "__main__":
    main()
