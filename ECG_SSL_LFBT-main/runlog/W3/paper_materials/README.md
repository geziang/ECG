# W3 论文材料复算目录(B-1, 主机B)

生成 git HEAD: `edd9aecf73521d9a9e2214f2cb8afd72c39ca898`(输入文件 blob 溯源见 paper_materials_manifest.json)。

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
