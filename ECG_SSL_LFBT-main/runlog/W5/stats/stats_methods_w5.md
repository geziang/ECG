# stats_methods_w5 · CPSC 修正分区官方统计方法与自查（W6 Stage 0 / 原W5B）

执行: 主机A Win4090, 2026-10-03 14:09–14:14 (W6 批次 CPU 车道)。任务书: W6 Stage 0(原 W5B B-1~B-4)。

## 1. 输入分区

- 逐记录预测: `runlog/W5/predictions/{b0,c1,c2,simclr,clocs}_cpsc_seed{0,2,4}/`（15 套,
  y_true/y_pred/y_prob.npy, **n=1,385**, 9 类, protocol_id=w5-cpscredo, W5A A-2 产出,
  账本 git_sha=32ec160）。
- 分区来源: W5A 污染审计后的重建分区（test=1385, train=4809, val=683, dup=0, 与 W1 manifest
  逐类一致; 审计全记录见 `runlog/W5/cpsc_audit.md`）。

## 2. 方法与命令

- 脚本: `runlog/W4/stats/paired_stats.py`（W4 B-4 审计版, **一行未改**; tests/test_paired_stats.py
  21 项单测在库）。
- 统计: record-level 配对 bootstrap on macro-AUROC, 10,000 次, 3 seeds 同步重采样,
  RNG seed=20260923; DeLong per-class 作参考列。
- 实际命令(含一处传参勘误, 事件记录见 `runlog/W6/events.md`):

```
python runlog/W4/stats/paired_stats.py --pred-root runlog/W5/predictions \
  --out-dir runlog/W5/stats --datasets cpsc --seeds 0 2 4 \
  --pairs c1:c2 c1:simclr c1:clocs c2:b0
```

- 勘误说明: 脚本 DEFAULT_PAIRS 用占位名 s1/s2, 而预测目录实名 simclr_/clocs_, 按任务书原样
  命令(不带 --pairs)会 skip 其中两对; W4 官方产物 `runlog/W4/stats/paired_stats.csv` 的
  method_b 即为 simclr/clocs, 证明 W4 当次同为显式 --pairs 传参。故显式传参才是"默认四对"
  的正确执行方式, 代码零改动。

## 3. 三项自查门(全过)

**门① schema 与记录数**: 列集与 `runlog/W4/stats/paired_stats.csv` 完全一致; 4/4 行
n_records=1385, n_classes=9, level=record, n_boot=10000。 **PASS**

**门② 两侧 macro-AUROC 对账**: 4 行×2 侧共 8 个值与 `runlog/W5/lp_results.csv` 种子均值
四位小数逐一一致(c1 0.9463 / c2 0.9474 / simclr 0.9150 / clocs 0.9418 / b0 0.9506)。 **PASS**

**门③ 与办公机临时复算交叉核对**(临时数 RNG 20260925, 10k 次; 容差 delta≤0.05pt /
CI 端点≤0.1pt / p 差≤0.02 或同侧):

| 对 | 本次 delta(pt) | 临时 delta(pt) | Δ差 | 本次 95%CI(pt) | 临时 95%CI(pt) | 本次 p | 临时 p |
|---|---:|---:|---:|---|---|---:|---:|
| C1 vs C2 | −0.103 | −0.104 | 0.001 | [−0.258, +0.048] | [−0.253, +0.047] | 0.187 | 0.19 |
| C1 vs SimCLR | +3.132 | +3.132 | 0.000 | [+2.360, +3.934] | [+2.389, +3.949] | <0.0001 | <0.0001 |
| C1 vs CLOCS | +0.451 | +0.451 | 0.000 | [+0.148, +0.758] | [+0.157, +0.763] | 0.0046 | 0.0038 |
| C2 vs B0 | −0.326 | −0.327 | 0.001 | [−0.708, +0.026] | [−0.705, +0.024] | 0.0682 | 0.0696 |

**PASS**(delta 差 ≤0.002pt、CI 端点差 ≤0.03pt、p 差 ≤0.0014 且全同侧——全部远小于容差)。
预授权停机条件(方向翻转/显著性翻转/delta 差>0.1pt)均未触发, 不需上推差异记录。

## 4. B-2 种子级配对差(独立复算, 与任务书参照数逐位一致)

C2−C1 (AUPRC pt): **+0.70 / +0.59 / +1.20, mean +0.83, 3/3⊕, 置换 p=0.125,
seed-level t-CI [+0.02, +1.64]**(确认 CI 下界贴边含零外, 属描述性证据, 不写"统计显著")。
C2−B0 (AUPRC pt): −1.16 / −0.66 / +0.35, mean −0.49, 1/3⊕ 非同向。
顺带: C1−SimCLR +8.61/+10.20/+9.66 (3/3⊕); C1−CLOCS +1.77/+1.15/+0.82 (3/3⊕)。
auroc 口径与全部明细见 `runlog/W5/paper_materials_v3/paired_delta_v3.csv`。

## 5. 与 W4 各行的关系

- `runlog/W4/stats/paired_stats.csv` 与 v2 材料中的 **cpsc 各行为污染分区历史记录**
  (test=2004 新旧并集), 原样保留不再引用; 本目录为其修正替换行。
- PTB / Chapman 统计行**未重算**(预测文件未变), v3 中原样沿用。

## 6. 红线遵守

W2/W3/W4 冻结产物只读未动; 全部 record-level 措辞; 无新实验; 评估代码零改动。
