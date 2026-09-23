# -*- coding: utf-8 -*-
"""B-0 交付物3: failed_directions_index.csv 生成器(主机B)。

数据源(全部只读):
  - 失败实验归档-2026-09-22.md §二/§三 探针表(markdown 解析)
  - runlog/M/matrix_results.csv / matrix_results_hostB.csv (host 归属判定 + W1-B 新增行)
  - 失败实验归档 §四 历史实验族(族级关闭行, 手写枚举与原文对应)

输出: runlog/W3/failed_directions_index.csv
列: family,probe,host,seeds,auprc,delta_pt_vs_local_b0,closure_reason,source_file,source_git_sha
"""
import csv
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT.parent / "失败实验归档-2026-09-22.md"
OUT = ROOT / "runlog" / "W3" / "failed_directions_index.csv"

def _open_out(p, *args, **kw):
    """输出文件守卫: 在校验后的路径上打开文件(安全扫描整改)。"""
    q = Path(p).expanduser().resolve()
    if q.is_dir():
        raise ValueError(f"输出路径是已存在目录: {q}")
    return open(q, *args, **kw)


head_sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()[:7]

rows = []

# ---- 双机账本加载(host 归属判定 + W1-B 行) ----
def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

mA = load(ROOT / "runlog/M/matrix_results.csv")
mB = load(ROOT / "runlog/M/matrix_results_hostB.csv")
names_A = {r["name"] for r in mA}
names_B = {r["name"] for r in mB}

def host_of(name):
    inA, inB = name in names_A, name in names_B
    if inA and not inB:
        return "A"
    if inB and not inA:
        return "B"
    if inA and inB:
        return "A+B"
    return "?"  # 归档表里的别名, 手工映射

ALIAS = {"a1_d1l_fix": "A"}  # 归档表写法对应 A 机 W1 行(0.6471/-0.0706 为 A 账本口径)

# ---- 解析归档 markdown §二/§三 表 ----
text = ARCHIVE.read_text(encoding="utf-8")
table_pat = re.compile(r"^\|([^|]+)\|([^|]+)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]+)\|$")
for line in text.splitlines():
    m = table_pat.match(line.strip())
    if not m:
        continue
    fam, probe, seed, auprc, delta, reason = (c.strip() for c in m.groups())
    if fam in ("实验族", "---", "实验族---") or probe.startswith("---"):
        continue
    if fam == "探针":  # §三 表头
        continue
    if not re.match(r"^-?\d", delta) and not delta.startswith("+"):
        continue  # 非数据行
    if probe.startswith("`") and probe.endswith("`"):
        probe = probe[1:-1]
    if fam == "实验族":  # §三 表头行防御
        continue
    host = ALIAS.get(probe, host_of(probe))
    src = ("runlog/M/matrix_results.csv" if host in ("A", "A+B")
           else "runlog/M/matrix_results_hostB.csv")
    # §三 表: 主机B 6 探针(delta 相对 B0=0.7158)
    if seed == "":  # §三 无 seed 列(表结构与 §二 不同, 由下面单独处理)
        continue
    rows.append({
        "family": fam, "probe": probe, "host": host, "seeds": f"s{seed}",
        "auprc": auprc, "delta_pt_vs_local_b0": f"{float(delta)*100:+.2f}",
        "closure_reason": reason,
        "source_file": src, "source_git_sha": head_sha,
    })

# §三 表结构: | 实验族 | 探针 | AUPRC | Δ vs 主机 B0=0.7158 | 结论 |
pat3 = re.compile(r"^\|\s*([^|]+?)\s*\|\s*`?([^|`]+?)`?\s*\|\s*([\d.]+)\s*\|\s*(-?[\d.]+)\s*\|\s*([^|]+?)\s*\|$")
for line in text.splitlines():
    m = pat3.match(line.strip())
    if not m:
        continue
    fam, probe, auprc, delta, reason = m.groups()
    if fam in ("实验族",) or probe.startswith("探针") or probe.startswith("---"):
        continue
    if float(auprc) < 0.5:
        continue
    host = host_of(probe)
    rows.append({
        "family": fam, "probe": probe, "host": host, "seeds": "s0",
        "auprc": auprc, "delta_pt_vs_local_b0": f"{float(delta)*100:+.2f}",
        "closure_reason": reason,
        "source_file": "runlog/M/matrix_results_hostB.csv", "source_git_sha": head_sha,
    })

# ---- W1 主机B 7 跑(账本行, 归档文档未收录) + b1 弃跑注记 ----
W1B_REASONS = {
    "a1_d1lfix_05_02": "判负: τ对角修正版破坏性强于旧bug, 结构化跨导目标线关闭",
    "a1_d1lfix_neg": "判负: 打乱先验(-0.0689)≈真实(-0.0758)→伤害源于机制本身非先验对错",
    "b3_ccm_rec01": "判负: 周期内遮挡劣于多段遮挡, 重建支路无增益(η=0.1)",
    "b2_multiseg_rec01": "噪声级(-0.0017)≈中性, 多段遮挡无增益",
    "a4_rand_p101": "判负: ACL加性随机分区三分区符号一致微负",
    "a4_rand_p102": "判负: ACL加性随机分区三分区符号一致微负",
    "a4_rand_p103": "判负: ACL加性随机分区三分区符号一致微负",
}
FAMILY = {"a1": "W1-结构先验(D1L-fix)", "b2": "W1-重建线", "b3": "W1-重建线(CCM)",
          "a4": "W1-ACL随机分区"}
for r in mB:
    name = r["name"]
    if name in W1B_REASONS:
        rows.append({
            "family": FAMILY[name.split("_")[0] if not name.startswith("a4") else "a4"],
            "probe": name, "host": "B", "seeds": "s0",
            "auprc": f'{float(r["auprc"]):.4f}',
            "delta_pt_vs_local_b0": f'{float(r["delta"])*100:+.2f}',
            "closure_reason": W1B_REASONS[name],
            "source_file": "runlog/M/matrix_results_hostB.csv", "source_git_sha": head_sha,
        })
rows.append({
    "family": "W1-重建线(旧D7)", "probe": "b1_d7_arch", "host": "B", "seeds": "-",
    "auprc": "NA", "delta_pt_vs_local_b0": "NA",
    "closure_reason": "弃跑(工程决策): 旧路径未提速~23h, 归档审计价值低; 非结果判负",
    "source_file": "runlog/HOSTS.md(W1终报)", "source_git_sha": head_sha,
})

# ---- §四 历史实验族(族级关闭) ----
FAMILIES = [
    ("E002", "Attention/ECA/均值/融合变体", "无稳定超过B0; AR-B3仅作历史对照基座"),
    ("E003", "Mixer/JEPA/自动复合", "Mixer明显退化, JEPA部分缓解, 自动复合验证过拟合; 堆模块路线关闭"),
    ("E005", "跨导联掩码蒸馏", "clean -2.3pt, 双导缺失-0.9pt, 同时损伤通用与鲁棒, 关闭"),
    ("E006", "生理-空间先验P1/S1", "AR-B3上+0.40pt弱正, B0无增益, 未过粗筛门槛; 保留为边界分析"),
    ("E007", "LGA(×TRC)", "LGA单开FT100 -0.89pt且FT10负; LGA关闭, TRC单独进入确认(W2已确认)"),
    ("N1", "多库混合预训练", "五库混合 -3.07pt, 域冲突>数据量收益; 无域适配不再混库"),
    ("D9", "VICReg/白化目标重写", "多形态均负(-1.3~-10.7pt), 不再替换Barlow Twins目标"),
]
for fam, probe, reason in FAMILIES:
    rows.append({
        "family": f"{fam}-历史族", "probe": probe, "host": "A(历史)", "seeds": "mixed",
        "auprc": "NA", "delta_pt_vs_local_b0": "见族描述",
        "closure_reason": reason,
        "source_file": "失败实验归档-2026-09-22.md§四", "source_git_sha": head_sha,
    })

# ---- 写出 ----
cols = ["family", "probe", "host", "seeds", "auprc", "delta_pt_vs_local_b0",
        "closure_reason", "source_file", "source_git_sha"]
# 去重(归档§二与§三同表头解析可能重叠)
seen, uniq = set(), []
for r in rows:
    k = (r["family"], r["probe"], r["seeds"], r["auprc"])
    if k in seen:
        continue
    seen.add(k)
    uniq.append(r)
with _open_out(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(uniq)
print(f"written {OUT} rows={len(uniq)}")
for r in uniq:
    print(f'  {r["host"]:>3} {r["probe"]:<22} {r["delta_pt_vs_local_b0"]:>7} {r["closure_reason"][:40]}')
