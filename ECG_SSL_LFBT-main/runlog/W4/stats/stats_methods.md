# W4 B-4 配对统计方法说明（stats_methods.md）

> 主机B（DESKTOP-0PBLCND），先行于 A-8 predictions 落盘；A-8 落盘后重跑
> `python runlog/W4/stats/paired_stats.py` 即出正式 `paired_stats.csv`。
> 单测：`python tests/test_paired_stats.py`（21 项全绿，合成数据边界验证）。

## 1. 输入

`runlog/W4/predictions/{method}_{dataset}_seed{N}/` 下 `metrics_ext.save_eval_artifacts`
落盘的 `y_true.npy / y_pred.npy / y_prob.npy`（顺序=test loader 顺序）。
加载时强制校验 `y_pred == argmax(y_prob)` 与配对两侧 `y_true` 逐位一致，不符即拒绝
（防目录错配/改动）。

## 2. 检验方法

**主检验：记录级配对 bootstrap on macro-AUROC**

- 单元 = 单条 test 记录；每次 bootstrap 有放回重采样记录索引，**同一索引同时作用于
  方法 A 与 B（配对）**；
- 多 seed 合并：同一 bootstrap 索引同步作用于全部 seeds（0/2/4），
  `delta_boot = mean_over_seeds(AUROC_A,s - AUROC_B,s)`；
- 双侧 p = `2·min(P(delta_boot≤0), P(delta_boot≥0))`；同时报 delta 的 95% 百分位 CI；
- 默认 10000 次，RNG seed=20260923 固定可复现；
- macro-AUROC = 各类 one-vs-rest AUROC（秩基、平均秩处理 ties）的宏平均。

**辅助检验：DeLong 配对检验（per-class，seed0）**

- 每类 one-vs-rest 上计算两模型的 DeLong 结构成分（V10/V01），
  配对协方差由同一批记录的成分叉积得到，z 检验双侧 p；
- 报 per-class p 的 min 与 median；**macro 层面不做解析合并**（类间方差独立性
  无保证），macro 层显著性一律以 bootstrap 主检验为准。

## 3. 覆盖对比对（任务书 B-4 指定）

C1 vs C2、C1 vs S1(SimCLR)、C1 vs S2(CLOCS)、C2 vs B0，各下游（ptbxl / cpsc / chapman）。

## 4. 局限与措辞边界（硬约束）

- **全部为 record-level 统计**：同一患者的多条记录视为相关样本，未做 patient-level
  聚类校正（如 patient-level cluster bootstrap）；论文**不得声称 patient-level 显著性**。
- 多 seed 合并的 bootstrap 把 seed 差异平均进 delta，不度量 seed 间方差的全部结构；
  单 seed 敏感性可另跑 `--seeds 0` 等对照。
- p 值仅为描述性证据；主结论仍以 W2/W3 预注册协议（3/3 seed 同向门）为准。

## 5. 输出 schema（paired_stats.csv）

`dataset, method_a, method_b, seeds, n_records, auroc_macro_a, auroc_macro_b,
delta_macro, boot_p, boot_ci95_low, boot_ci95_high, delong_p_min(seed0),
delong_p_median(seed0), n_boot, n_classes, level, note`

其中 `level=record` 恒定，防止下游表格误标。
