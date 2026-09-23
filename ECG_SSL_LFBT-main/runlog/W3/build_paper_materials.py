# -*- coding: utf-8 -*-
"""B-1 统计与图表复算代码(主机B) —— 任务书《下一阶段双机任务书-2026-09-22》B-1。

读取 W2 原始 CSV 与冻结清单(只读), 生成论文材料的独立复算版本, 全部写入
runlog/W3/paper_materials/。每个输出表附带 source_file / source_git_sha 溯源列。

输出:
  main_table_recomputed.csv      主表(均值±SD, per-seed)
  paired_delta_recomputed.csv    配对 delta(精确置换 p, seed-level t-CI, 符号)
  seed_sign_table.csv            seed 符号表(紧凑版)
  missing_lead_summary.csv       缺导单导/双导汇总(vs full 的逐条件掉幅)
  missing_lead_heatmap_data.csv  双缺导 8x8 热图数据(长表: 3seed均值与逐seed)
  failed_directions_table.csv    失败方向表(复制自 B-0 索引并加溯源列)
  paper_materials_manifest.json  输入文件 blob 溯源 + 方法说明
  README.md                      方法与统计限制说明

统计口径(与 W2 冻结表一致, B-0 已核对):
  - 均值/SD: seed 样本统计, SD 为 ddof=1(n=1 时 0);
  - 置换 p: 3 seed 符号翻转精确单侧检验, 最小可达 p=1/8=0.125;
  - CI95: mean ± t(0.975,df=2)=4.302655 × sd/√3 —— seed-level 区间;
  - 明确禁止: 3 seed ≠ patient-level bootstrap(无逐记录预测概率)。

用法: python runlog/W3/build_paper_materials.py   (在 ECG_SSL_LFBT-main/ 下运行)
"""
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W2 = ROOT / "runlog" / "W2"
W1 = ROOT / "runlog" / "W1"
OUT = ROOT / "runlog" / "W3" / "paper_materials"
OUT.mkdir(parents=True, exist_ok=True)
T2 = 4.302655  # t(0.975, df=2)


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))



def _open_out(p, *args, **kw):
    """输出文件守卫: 在校验后的路径上打开文件(安全扫描整改)。"""
    q = Path(p).expanduser().resolve()
    if q.is_dir():
        raise ValueError(f"输出路径是已存在目录: {q}")
    return open(q, *args, **kw)


def write_csv(path, cols, rows):
    with _open_out(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def blob_sha(path):
    r = subprocess.run(["git", "-C", str(ROOT), "hash-object", str(ROOT / path)],
                       capture_output=True, text=True)
    return r.stdout.strip()[:12] if r.returncode == 0 else "NA"


HEAD = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                      capture_output=True, text=True).stdout.strip()

SRC_LEDGER = "runlog/W2/confirm_results.csv"
SRC_W1 = "runlog/W1/c3_evals.csv"
SRC_ML = "runlog/W2/missing_lead_c2.csv"
SRC_MANIFEST = "runlog/W2/freeze_manifest_v2.json"

ledger = read_csv(ROOT / SRC_LEDGER)
w1 = read_csv(ROOT / SRC_W1)
manifest = json.loads((ROOT / SRC_MANIFEST).read_text(encoding="utf-8"))

# ---------------------------------------------------------------- 数据装配
# seed0 参考值(W1 单种子): b0 全部 + c1/c2 的 s0(与账本逐位一致, B-0 已验证)
w1_ref = {(r["ckpt"], r["eval"], r["downstream"]): float(r["auprc"]) for r in w1}


def seeds_of(ev, ds, ck):
    got = {}
    for r in ledger:
        if r["eval"] == ev and r["downstream"] == ds and r["ckpt"] == ck:
            got[int(r["seed"])] = float(r["auprc"])
    w1_key = {"b0": "b0_anchor", "c1": "c1_nfh_b0", "c2": "c2_nfh_trc"}[ck]
    if (w1_key, ev, ds) in w1_ref:
        got.setdefault(0, w1_ref[(w1_key, ev, ds)])
    return got


# ---------------------------------------------------------------- 1. 主表复算
main_rows = []
for ev, ds, ck in [("lp", "ptbxl", "b0"), ("lp", "ptbxl", "c1"), ("lp", "ptbxl", "c2"),
                   ("ft10", "ptbxl", "b0"), ("ft10", "ptbxl", "c1"), ("ft10", "ptbxl", "c2"),
                   ("lp", "cpsc", "b0"), ("lp", "cpsc", "c1"), ("lp", "cpsc", "c2"),
                   ("lp", "chapman", "b0"), ("lp", "chapman", "c1"), ("lp", "chapman", "c2")]:
    sd_map = seeds_of(ev, ds, ck)
    vals = [sd_map[s] for s in sorted(sd_map)]
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    main_rows.append({
        "eval": ev, "downstream": ds, "ckpt": ck, "n_seeds": len(vals),
        "mean": f"{mean:.4f}", "sd": f"{sd:.4f}",
        "per_seed": "/".join(f"{sd_map[s]:.4f}" for s in sorted(sd_map)),
        "mean_pm_sd": f"{mean:.4f}±{sd:.4f}",
        "seed0_source": "W1 c3_evals(单种子参考)" if 0 not in
                        [int(r["seed"]) for r in ledger if r["eval"] == ev
                         and r["downstream"] == ds and r["ckpt"] == ck] else "W2 ledger",
        "source_file": f"{SRC_LEDGER};{SRC_W1}", "source_git_sha": HEAD[:7],
    })
write_csv(OUT / "main_table_recomputed.csv",
          list(main_rows[0].keys()), main_rows)

# ---------------------------------------------------------------- 2. 配对 delta + 3. 符号表
def perm_p(deltas):
    obs = sum(deltas)
    hit = tot = 0
    for mask in range(1 << len(deltas)):
        s = sum(d if (mask >> i) & 1 else -d for i, d in enumerate(deltas))
        tot += 1
        hit += (s >= obs - 1e-12)
    return hit / tot


pd_rows, sign_rows = [], []
for ev, ds, pair in [("lp", "cpsc", "C2-C1"), ("lp", "chapman", "C2-C1"),
                     ("lp", "ptbxl", "C2-C1"), ("ft10", "ptbxl", "C2-C1"),
                     ("lp", "ptbxl", "C1-B0"), ("ft10", "ptbxl", "C1-B0")]:
    a, b = pair.split("-")
    ma, mb = seeds_of(ev, ds, a.lower()), seeds_of(ev, ds, b.lower())
    seeds = sorted(set(ma) & set(mb))
    if not seeds:
        continue
    d_pt = {s: (ma[s] - mb[s]) * 100 for s in seeds}
    vals = [d_pt[s] for s in seeds]
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    ci_lo, ci_hi = mean - T2 * sd / math.sqrt(len(vals)), mean + T2 * sd / math.sqrt(len(vals))
    sign = "".join("+" if v > 1e-9 else ("-" if v < -1e-9 else "0") for v in vals)
    p = perm_p(vals)
    row = {
        "eval": ev, "downstream": ds, "pair": pair,
        **{f"d_s{s}": f"{d_pt[s]:+.2f}" for s in seeds},
        "mean_pt": f"{mean:+.2f}", "sign": sign,
        "all_same_direction": "yes" if sign in ("+++", "---") else "no",
        "perm_p_1sided_exact": f"{p:.3f}",
        "ci95_seed_level_t": f"[{ci_lo:+.2f},{ci_hi:+.2f}]",
        "n_seeds": len(seeds),
        "source_file": f"{SRC_LEDGER};{SRC_W1}", "source_git_sha": HEAD[:7],
    }
    pd_rows.append(row)
    sign_rows.append({
        "comparison": f"{ev}/{ds}/{pair}", "per_seed_pt":
            "/".join(f"{d_pt[s]:+.2f}" for s in seeds),
        "sign": sign, "mean_pt": f"{mean:+.2f}",
        "verdict": "3/3同向" if sign in ("+++", "---") else "非一致",
        "source_git_sha": HEAD[:7],
    })
write_csv(OUT / "paired_delta_recomputed.csv", list(pd_rows[0].keys()), pd_rows)
write_csv(OUT / "seed_sign_table.csv", list(sign_rows[0].keys()), sign_rows)

# ---------------------------------------------------------------- 4. 缺导汇总 + 5. 热图数据
ml = read_csv(ROOT / SRC_ML)
by_seed = {}
for r in ml:
    by_seed.setdefault(int(r["seed"]), {})[r["cond"]] = float(r["auprc"])

single_drops, double_drops = [], []
for s, conds in by_seed.items():
    full = conds["full"]
    for i in range(8):
        single_drops.append({"seed": s, "lead": i, "drop_pt": (conds[f"miss{i}"] - full) * 100})
    for i in range(8):
        for j in range(i + 1, 8):
            double_drops.append({"seed": s, "lead_i": i, "lead_j": j,
                                 "drop_pt": (conds[f"miss{i}{j}"] - full) * 100})

s_mean = sum(d["drop_pt"] for d in single_drops) / len(single_drops)
d_mean = sum(d["drop_pt"] for d in double_drops) / len(double_drops)
ml_rows = [{
    "scope": "single_lead_miss(mean over 8 leads x 3 seeds)", "n_conditions": len(single_drops),
    "mean_drop_pt": f"{s_mean:+.2f}",
    "min": f"{min(d['drop_pt'] for d in single_drops):+.2f}",
    "max": f"{max(d['drop_pt'] for d in single_drops):+.2f}",
    "source_file": SRC_ML, "source_git_sha": HEAD[:7],
}, {
    "scope": "double_lead_miss(mean over 28 pairs x 3 seeds)", "n_conditions": len(double_drops),
    "mean_drop_pt": f"{d_mean:+.2f}",
    "min": f"{min(d['drop_pt'] for d in double_drops):+.2f}",
    "max": f"{max(d['drop_pt'] for d in double_drops):+.2f}",
    "source_file": SRC_ML, "source_git_sha": HEAD[:7],
}]
for i in range(8):  # 逐导联单缺敏感度(3seed均值)
    drops = [d["drop_pt"] for d in single_drops if d["lead"] == i]
    ml_rows.append({
        "scope": f"single_miss_lead{i}", "n_conditions": len(drops),
        "mean_drop_pt": f"{sum(drops) / len(drops):+.2f}",
        "min": f"{min(drops):+.2f}", "max": f"{max(drops):+.2f}",
        "source_file": SRC_ML, "source_git_sha": HEAD[:7],
    })
write_csv(OUT / "missing_lead_summary.csv", list(ml_rows[0].keys()), ml_rows)

hm_rows = []
for i in range(8):
    for j in range(8):
        if i == j:
            continue
        key = f"miss{min(i, j)}{max(i, j)}"
        vals = {s: by_seed[s][key] for s in sorted(by_seed)}
        drops = {s: (by_seed[s][key] - by_seed[s]["full"]) * 100 for s in sorted(by_seed)}
        mean_a = sum(vals.values()) / len(vals)
        mean_d = sum(drops.values()) / len(drops)
        hm_rows.append({
            "lead_i": i, "lead_j": j, "cond": key,
            "auprc_mean": f"{mean_a:.4f}",
            "drop_vs_full_pt_mean": f"{mean_d:+.2f}",
            "auprc_per_seed": "/".join(f"{vals[s]:.4f}" for s in sorted(vals)),
            "drop_per_seed_pt": "/".join(f"{drops[s]:+.2f}" for s in sorted(drops)),
            "source_file": SRC_ML, "source_git_sha": HEAD[:7],
        })
write_csv(OUT / "missing_lead_heatmap_data.csv", list(hm_rows[0].keys()), hm_rows)

# ---------------------------------------------------------------- 6. 失败方向表(加溯源)
fd_src = ROOT / "runlog" / "W3" / "failed_directions_index.csv"
fd = read_csv(fd_src)
for r in fd:
    r["source_file"] = r["source_file"] + "(经 runlog/W3/failed_directions_index.csv)"
write_csv(OUT / "failed_directions_table.csv", list(fd[0].keys()), fd)

# ---------------------------------------------------------------- 7. manifest + README
inputs = {
    "runlog/W2/confirm_results.csv": "W2 确认矩阵账本(28行)",
    "runlog/W1/c3_evals.csv": "W1 单种子参考读数",
    "runlog/W2/missing_lead_c2.csv": "缺导 3seed×37条件",
    "runlog/W2/freeze_manifest_v2.json": "冻结清单(checkpoint SHA/参数)",
    "runlog/W3/failed_directions_index.csv": "B-0 失败方向索引",
}
provenance = {k: {"blob_sha12": blob_sha(k), "role": v} for k, v in inputs.items()}
provenance["head_git_sha"] = HEAD

manifest_out = {
    "task": "B-1 paper materials recompute",
    "host": "B(DESKTOP-0PBLCND)",
    "generated": HEAD,
    "inputs": provenance,
    "methods": {
        "mean_sd": "seed-level; sd ddof=1, n=1 时 sd=0",
        "perm_p": "精确符号翻转置换检验, 单侧, n=3 -> 最小 p=1/8=0.125",
        "ci95": "seed-level t 区间 mean±t(0.975,df=2)*sd/sqrt(3); 非记录级区间",
        "missing_lead": "drop = cond_auprc - full_auprc (pt), 按 seed 内配对",
    },
    "outputs": sorted(p.name for p in OUT.iterdir() if p.suffix in (".csv", ".md", ".json")
                      and p.name != "paper_materials_manifest.json"),
    "statistics_limitation": (
        "W2 无逐记录预测概率, 本目录全部统计为 seed-level; "
        "禁止表述为 patient-level bootstrap 或 '统计显著'"),
}
(OUT / "paper_materials_manifest.json").write_text(
    json.dumps(manifest_out, ensure_ascii=False, indent=1), encoding="utf-8")

readme = f"""# W3 论文材料复算目录(B-1, 主机B)

生成 git HEAD: `{HEAD}`(输入文件 blob 溯源见 paper_materials_manifest.json)。

| 文件 | 内容 | 输入 |
|---|---|---|
| main_table_recomputed.csv | 4下游×3模型 主表(均值±SD/per-seed/seed0来源标注) | W2账本+W1参考 |
| paired_delta_recomputed.csv | C2−C1/C1−B0 配对(逐seed pt/符号/精确置换p/seed级t-CI) | 同上 |
| seed_sign_table.csv | 符号表紧凑版(3/3同向判定) | 同上 |
| missing_lead_summary.csv | 单导/双缺导掉幅汇总+逐导联敏感度 | missing_lead_c2.csv |
| missing_lead_heatmap_data.csv | 双缺导 8×8 热图长表(3seed均值+逐seed) | missing_lead_c2.csv |
| failed_directions_table.csv | 失败方向表(54行, 溯源列) | B-0 索引 |

## 方法
- SD 为 seed 样本标准差(ddof=1); n=1 的单元(如 b0 外部域单种子)SD=0 并在 seed0_source 列标注来源。
- 置换 p:3 seed 符号翻转精确单侧检验, **最小可达 p=0.125**。
- CI95:seed-level t 区间 mean±4.3027·sd/√3;CPSC LP C2−C1 = +2.30 [+0.26,+4.34]。

## 统计限制(不可违反)
W2 未保存逐记录预测概率 → 本目录全部为 **seed-level 描述统计**;
不得写成 patient-level bootstrap;3 seed 置换 p 下限 0.125,
不得使用"统计显著"表述;跨机绝对 AUPRC 不可比(双机各自对本机锚点)。
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")

# ---------------------------------------------------------------- 校验输出 vs 冻结表(一致性自检)
frozen_mt = read_csv(ROOT / "runlog/W2/stats/main_table.csv")
frozen_pd = read_csv(ROOT / "runlog/W2/stats/paired_delta.csv")
mt_by_key = {(r["eval"], r["downstream"], r["ckpt"]): r for r in frozen_mt}
pd_by_key = {(r["eval"], r["downstream"], r["pair"]): r for r in frozen_pd}
mism = []
for r in main_rows:
    fr = mt_by_key.get((r["eval"], r["downstream"], r["ckpt"]))
    if fr is None or not (r["mean"] == f'{float(fr["mean"]):.4f}'
                          and r["per_seed"] == fr["per_seed"]):
        mism.append(f'main {r["eval"]}/{r["downstream"]}/{r["ckpt"]}')
for r in pd_rows:
    fr = pd_by_key.get((r["eval"], r["downstream"], r["pair"]))
    if fr is None or not (r["sign"] == fr["sign"] and r["mean_pt"] == fr["mean_pt"]):
        mism.append(f'paired {r["eval"]}/{r["downstream"]}/{r["pair"]}')
print("self-check vs frozen stats:", "ALL MATCH" if not mism else f"MISMATCH {mism}")
print(f"outputs in {OUT}: " + ", ".join(sorted(p.name for p in OUT.iterdir())))
print(f"headline: cpsc_lp={next(r for r in pd_rows if r['downstream']=='cpsc')['mean_pt']}"
      f" sign={next(r for r in pd_rows if r['downstream']=='cpsc')['sign']}; "
      f"ft10={next(r for r in pd_rows if r['eval']=='ft10')['mean_pt']}"
      f" sign={next(r for r in pd_rows if r['eval']=='ft10')['sign']}")
print(f"missing-lead: single {s_mean:+.2f}pt / double {d_mean:+.2f}pt (vs full)")
