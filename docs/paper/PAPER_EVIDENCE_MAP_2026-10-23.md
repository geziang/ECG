# V7 表图与证据映射

更新：2026-10-23。与[小论文收官总账](PAPER_CLOSEOUT_2026-10-23.md)配套。当前正文为[根目录 V7](../../../主论文中文成稿版V7-投稿修订与图文一致性-2026-10-23.md)，当前图源为[V7 图集](../../../outputs/主论文图集_V7_投稿修订_2026-10-23/)。以下是查证入口，本轮没有逐格重算全部论文数字。

## 1. 协议与预测

| 范围 | 协议/版本入口 | 原始证据 |
|---|---|---|
| 早期基线 | [W2 冻结清单](../../ECG_SSL_LFBT-main/runlog/W2/freeze_manifest_v2.json)、[W4 参数](../../ECG_SSL_LFBT-main/runlog/W4/baseline_hparams.csv) | [W4 结果](../../ECG_SSL_LFBT-main/runlog/W4/baseline_results.csv)；其中旧 CPSC 行必须转用 [W5 结果](../../ECG_SSL_LFBT-main/runlog/W5/lp_results.csv) |
| JOINT 联合框架及 A1/A2 更正 | [协议](../../ECG_SSL_LFBT-main/runlog/JOINT_20261013/protocol.md)、[任务书](../../ECG_SSL_LFBT-main/runlog/JOINT_20261013/任务书.md) | [results.csv](../../ECG_SSL_LFBT-main/runlog/JOINT_20261013/results.csv)、[权重清单](../../ECG_SSL_LFBT-main/runlog/JOINT_20261013/checkpoints_manifest.json) |
| ABC 全因子与补充比较 | [冻结协议](../../ECG_SSL_LFBT-main/runlog/ABC_20261015/protocol_frozen.md) | [results.csv](../../ECG_SSL_LFBT-main/runlog/ABC_20261015/results.csv)、[权重清单](../../ECG_SSL_LFBT-main/runlog/ABC_20261015/checkpoints_manifest.json)、[数据哈希](../../ECG_SSL_LFBT-main/runlog/ABC_20261015/data_sha.json) |
| W9 FT10 补齐 | [任务书](../../ECG_SSL_LFBT-main/runlog/W9/W9_任务书_正式投稿FT10补齐_2026-10-19.md)、[任务矩阵](../../ECG_SSL_LFBT-main/runlog/W9/task_matrix.csv) | [results.csv](../../ECG_SSL_LFBT-main/runlog/W9/results.csv)、[数据清单](../../ECG_SSL_LFBT-main/runlog/W9/data_manifest.json)、[权重清单](../../ECG_SSL_LFBT-main/runlog/W9/checkpoints_manifest.json) |
| 逐记录预测与配对顺序 | [archive_manifest.json](../../ECG_SSL_LFBT-main/runlog/preds_archive/archive_manifest.json)、[预测归档目录](../../ECG_SSL_LFBT-main/runlog/preds_archive/) | 按对应批次 ZIP、sample_order、模型/种子/任务读取；不跨批次凭近似模型名配对；压缩归档存在不表示原始工作目录一定仍在 |

## 2. 按 V7 当前位置查证

| V7 位置 | 内容 | 查证入口与边界 |
|---|---|---|
| 表 1、§3.1、补表 S1 | 样本数、划分和类别映射 | [NFH 清单](../../ECG_SSL_LFBT-main/runlog/W1/nfh_manifest.json)、[CPSC 修正清单](../../ECG_SSL_LFBT-main/runlog/W5/cpsc_manifest.json)、[Chapman 清单](../../ECG_SSL_LFBT-main/runlog/W1/chapman_manifest.json)、[W9 数据清单](../../ECG_SSL_LFBT-main/runlog/W9/data_manifest.json)；预处理见[项目地图](../../PROJECT_MAP.md) |
| §3.2–3.7、图 1、补图 S1、补表 S6 | 架构、损失、超参、模型身份与评估流程 | [训练入口](../../ECG_SSL_LFBT-main/run_pt.py)、[TRC 实现](../../ECG_SSL_LFBT-main/models/vgg_1d.py)、[心拍/RR 实现](../../ECG_SSL_LFBT-main/data_utils/dv_dataset.py)、[噪声实现](../../ECG_SSL_LFBT-main/data_utils/nstdb_aug.py)，结合对应冻结协议；图示以 V7 图集为准；缓存生成环境仍待核验 |
| 表 2、图 2 左侧 LP | 三域 LP、外部基线与联合配置 | [ABC 三域 LP 汇总](../../ECG_SSL_LFBT-main/runlog/ABC_20261015/table_lp_3domains.csv)、[JOINT 三域 LP](../../ECG_SSL_LFBT-main/runlog/JOINT_20261013/table1_lp_3domains.csv)、[W5 CPSC](../../ECG_SSL_LFBT-main/runlog/W5/lp_results.csv)、[W4 基线](../../ECG_SSL_LFBT-main/runlog/W4/baseline_results.csv)；Random/SimCLR-Physio/TSR 分别查 [rand](../../ECG_SSL_LFBT-main/runlog/W8/w8_rand_results.csv)、[simph](../../ECG_SSL_LFBT-main/runlog/W8/w8_simph_results.csv)、[tsr](../../ECG_SSL_LFBT-main/runlog/W8/w8_tsr_results.csv)；按配置/任务筛行，不混入 S1 补格 |
| 表 3、图 2 左侧 FT10 | 三域 10% 标签微调 | [W9 完整汇总](../../ECG_SSL_LFBT-main/runlog/W9/table4_complete.md)、[W9 新增结果](../../ECG_SSL_LFBT-main/runlog/W9/results.csv)、[ABC FT10 汇总](../../ECG_SSL_LFBT-main/runlog/ABC_20261015/table_ft10.csv)及对应历史来源；W9 文件名含 table4，当前 V7 是表 3 |
| 表 4、图 3 | TRC、Affine 与两种语料匹配比较 | [W5 LP](../../ECG_SSL_LFBT-main/runlog/W5/lp_results.csv)、[W7 E2c](../../ECG_SSL_LFBT-main/runlog/W7/e2c_results.csv)、[W8 Affine](../../ECG_SSL_LFBT-main/runlog/W8/w8_trcaff_results.csv)、[W8 Chapman LP](../../ECG_SSL_LFBT-main/runlog/W8/w8_b0trc_results.csv)、[W6 五种子汇总](../../ECG_SSL_LFBT-main/runlog/W6/seed5_summary.csv)；严格匹配范围按[10-20 裁决](../../ECG_SSL_LFBT-main/runlog/W9/裁决备忘_TRC匹配与效率表_2026-10-20.md) |
| 表 5A、图 3 | 通道均值/RMS 偏移与 TRC 参数统计 | [W7 通道统计](../../ECG_SSL_LFBT-main/runlog/W7/e2d_channel_stats.csv)、[参数统计](../../ECG_SSL_LFBT-main/runlog/W7/e2d_gamma_beta.csv)、[说明](../../ECG_SSL_LFBT-main/runlog/W7/e2d_summary.md)；训练后关联证据，不升级为因果结论 |
| 表 5B | 预训练 TRC 与下游 TRC 适配 | [W7 下游适配](../../ECG_SSL_LFBT-main/runlog/W7/e2a_trcft.csv)、[E2c](../../ECG_SSL_LFBT-main/runlog/W7/e2c_results.csv)、[W5 LP](../../ECG_SSL_LFBT-main/runlog/W5/lp_results.csv) |
| 表 6A/6B、图 2 右侧 | Beat-RR 消融与 CPSC 核心配对比较 | [ABC paired_stats.csv](../../ECG_SSL_LFBT-main/runlog/ABC_20261015/paired_stats.csv)、[JOINT 消融汇总](../../ECG_SSL_LFBT-main/runlog/JOINT_20261013/table2_ablation_cpsc.csv)、[JOINT 配对统计](../../ECG_SSL_LFBT-main/runlog/JOINT_20261013/paired_stats.csv)；核心族 Holm 校正；A1/A2 采用 JOINT 正确缓存重跑，不能取 W8 旧拆分表 |
| 表 7A/7B、图 4 | 时间隔离噪声评估与配对收益 | [W8 隔离结果](../../ECG_SSL_LFBT-main/runlog/W8/w8_noiseiso_results.csv)、[JOINT 配对噪声表](../../ECG_SSL_LFBT-main/runlog/JOINT_20261013/table3_noise_paired.csv)、[ABC 27 格汇总](../../ECG_SSL_LFBT-main/runlog/ABC_20261015/table_noise27.csv)、[ABC 配对统计](../../ECG_SSL_LFBT-main/runlog/ABC_20261015/paired_stats.csv)；训练前半/测试后半与未隔离历史结果分开 |
| §4.1 第二预训练语料 | M4b 三域 LP 与 CPSC FT10 | [ABC results.csv](../../ECG_SSL_LFBT-main/runlog/ABC_20261015/results.csv)、[三域 LP](../../ECG_SSL_LFBT-main/runlog/ABC_20261015/table_lp_3domains.csv)、[配对统计](../../ECG_SSL_LFBT-main/runlog/ABC_20261015/paired_stats.csv)；未扩成已完成三域 FT10 |
| §4.1 标签比例与 §4.5、补表 S3 | FT20/40、LP-FT、五种子与患者重分层 | [W6 标签比例结果](../../ECG_SSL_LFBT-main/runlog/W6/ft2040_results.csv)、[五种子汇总](../../ECG_SSL_LFBT-main/runlog/W6/seed5_summary.csv)、[W7 LP-FT](../../ECG_SSL_LFBT-main/runlog/W7/e4a_lpft.csv)、[患者重分层](../../ECG_SSL_LFBT-main/runlog/W7/e4c_patient_lp.csv)；重分层敏感性不代替主协议，也不升级为所有主结果 patient-level 推断 |
| 补表 S2A | 完整三因素交互 | [ABC factorial_stats.csv](../../ECG_SSL_LFBT-main/runlog/ABC_20261015/factorial_stats.csv)、[冻结协议](../../ECG_SSL_LFBT-main/runlog/ABC_20261015/protocol_frozen.md)；CI 跨零时保留不确定性 |
| 补表 S2B/S2C | Beat/RR 拆分、随机窗口、RR 置换与节律分层 | [ABC 配对统计](../../ECG_SSL_LFBT-main/runlog/ABC_20261015/paired_stats.csv)、[W7 分层汇总](../../ECG_SSL_LFBT-main/runlog/W7/e4d_bucket_summary.csv)、[分层差值](../../ECG_SSL_LFBT-main/runlog/W7/e4d_delta_summary.csv)；以三种子及相应校正结果为准 |
| 补表 S4 | 原始多诊断真值下重评冻结 softmax 分数 | [multihot 真值 CSV](../../ECG_SSL_LFBT-main/runlog/preds_archive/cpsc_multilabel/cpsc_multilabel.csv)、[真值摘要](../../ECG_SSL_LFBT-main/runlog/preds_archive/cpsc_multilabel/summary.json)、[导出脚本](../../ECG_SSL_LFBT-main/runlog/preds_archive/export_cpsc_multilabel.py)、[逐记录预测清单](../../ECG_SSL_LFBT-main/runlog/preds_archive/archive_manifest.json)；当前 S4 固定复算脚本与结果表见总账 A04 待归档，不能用 ABC 的 one-hot `multilib_results.csv` 替代，也不等同原生多标签训练 |
| 补表 S5A | 历史预训练墙钟/显存 | [JOINT 权重与运行清单](../../ECG_SSL_LFBT-main/runlog/JOINT_20261013/checkpoints_manifest.json)、[ABC 清单](../../ECG_SSL_LFBT-main/runlog/ABC_20261015/checkpoints_manifest.json)、[10-20 裁决](../../ECG_SSL_LFBT-main/runlog/W9/裁决备忘_TRC匹配与效率表_2026-10-20.md)；缺项 n/a，不补训；不同批次只作描述性报告 |
| 补表 S5B | 参数量、MACs、延迟、吞吐 | [ABC efficiency.csv](../../ECG_SSL_LFBT-main/runlog/ABC_20261015/efficiency.csv)、[测量脚本](../../ECG_SSL_LFBT-main/runlog/ABC_20261015/run_abc_bench.py)；沿原统计范围，不把未计入运算解释为全流程成本 |

## 3. 图源与旧入口的关系

- 当前正文引用 5 张图：图 1–4 和补图 S1，均在 V7 图集；本轮仅检查引用存在性，不重绘、不改数。
- [V7 修订说明](../../../outputs/paper_revision_v7_20261023/修订说明.md)记录图文调整和数字舍入说明。V6 Word 未同步 V7，不能作为当前表号和图文源。
- [旧逐表索引](../../../实验报告与结果索引.md)主体截至 2026-10-10，只用于追溯；其“W8 未跑、图未画、旧表号”等内容已经过时。
- 本映射确认入口，不证明权重已独立备份、S4 复算材料已齐全或所有历史待核验项均已闭合；未完成项回到收官总账 §4–5。
