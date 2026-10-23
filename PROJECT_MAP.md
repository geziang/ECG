# ECG 项目地图

更新：2026-10-23。以下路径相对本文件所在的 Git 根 `ECG/`。本文件负责定位；真实数值仍须核对协议与原始结果。

## 1. 项目与当前状态

研究主题：基于生理先验的多导联 ECG 自监督表征学习。基础为 LFBT；当前小论文围绕 TRC、Beat-RR 与真实噪声增强联合框架。按本轮用户说明，小论文作为已完成的研究基础，后续聚焦大论文选线。大论文开题题目为“基于先验知识引导的心电表征预训练及其在多导联心律分析中的应用”。

**小论文任务状态先读[收官总账](docs/paper/PAPER_CLOSEOUT_2026-10-23.md)，按当前表号查[V7 表图与证据映射](docs/paper/PAPER_EVIDENCE_MAP_2026-10-23.md)。** 历史任务书保留原位，已完成/判负/替代/待归档分别记录；独立代码与权重冻结包尚未完成。本轮仅整理本地文档并已提交推送（2026-10-23 整理批）；没有新增主机 A 可认领任务，未做远端核验。

当前小论文在 Git 仓库外的同级工作区根：

- [V7 Markdown（当前正文）](../主论文中文成稿版V7-投稿修订与图文一致性-2026-10-23.md)
- [V7 可编辑 PPT 与导出图片](../outputs/主论文图集_V7_投稿修订_2026-10-23/)
- [V7 修订说明](../outputs/paper_revision_v7_20261023/修订说明.md)
- [V6 Word（历史稿，未同步 V7）](../主论文中文成稿版V6-联合框架与正向收益-2026-10-23.docx)
- [工作区总目录](../WORKSPACE_INDEX.md)

这些外部链接适用于当前工作区布局；仅克隆本仓库不会获得上述文件。[paper/](paper/) 内的稿件不能代替当前 V7。本轮仅修订文稿、图示与导航，保留冻结结果及训练代码。

截至本次只读检查，Git 根约有 1,807 个已跟踪文件，其中 Markdown 64 个、Python 146 个（不计本次新增导航与计划文档）。采用人工维护的任务地图，不在启动时加载完整文件清单。

## 2. 目录职责

| 路径 | 用途 |
|---|---|
| [ECG_SSL_LFBT-main/](ECG_SSL_LFBT-main/) | 主代码目录与真实实验账本 |
| [报告归档/](报告归档/) | 开题报告、历史设计、完成与未完成报告 |
| [paper/](paper/) | 历史成稿与负结果支撑稿 |
| [docs/paper/](docs/paper/PAPER_CLOSEOUT_2026-10-23.md) | 小论文收官总账、V7 表图证据映射与本轮整理核验 |
| [docs/experiments/](docs/experiments/THESIS_P0_20261023/TASKBOOK.md) | 首批主机 A 实验设计草案及协议/22 作业矩阵/来源锁；未下发、未实现新入口、未启动训练 |
| [docs/research/](docs/research/) | 当前后续研究计划与阅读清单 |
| [../参考代码/](../参考代码/论文与代码总览.md) | Git 仓库外的九篇论文/源码资料；PDF、源码、权重及复现状态分别记录；尚未运行复现 |
| [runlog/HANDOFF.md](runlog/HANDOFF.md) | 2026-09-18 旧快照，不是当前总账本 |

## 3. 代码地图

以下文件都在 `ECG_SSL_LFBT-main/` 下。旧 README 是上游 LFBT 的运行说明，不能覆盖后续实验协议。

| 任务 | 主要入口 | 作用 |
|---|---|---|
| 自监督预训练 | [run_pt.py](ECG_SSL_LFBT-main/run_pt.py) | LFBT 训练及 TRC / 生理辅助 / 噪声等开关 |
| 线性探查 | [run_lp.py](ECG_SSL_LFBT-main/run_lp.py) | 冻结编码器训练分类头；当前主线 CE 单标签 |
| 少标签微调 | [run_ft.py](ECG_SSL_LFBT-main/run_ft.py) | 全模型微调；当前主线 CE 单标签 |
| 主干与 TRC | [models/vgg_1d.py](ECG_SSL_LFBT-main/models/vgg_1d.py) | 一维 VGG、GRN1D / TRC、Affine 对照 |
| 导联并行结构 | [models/parallel_vgg.py](ECG_SSL_LFBT-main/models/parallel_vgg.py) | 多导联编码器结构 |
| 心拍与 RR 数据 | [data_utils/dv_dataset.py](ECG_SSL_LFBT-main/data_utils/dv_dataset.py) | R 峰窗平均为心拍视图；RR 三统计量；随机窗/置换反事实 |
| 真实噪声 | [data_utils/nstdb_aug.py](ECG_SSL_LFBT-main/data_utils/nstdb_aug.py) | NSTDB 噪声增强 |
| 鲁棒性评价 | [run_robustness.py](ECG_SSL_LFBT-main/run_robustness.py)、[run_robustness_nstdb.py](ECG_SSL_LFBT-main/run_robustness_nstdb.py) | 输入退化与 NSTDB 压力测试 |
| 数据预处理 | [preprocess_ptbxl.py](ECG_SSL_LFBT-main/preprocess_ptbxl.py)、[prepare_cpsc_chapman.py](ECG_SSL_LFBT-main/prepare_cpsc_chapman.py)、[prepare_nfh.py](ECG_SSL_LFBT-main/prepare_nfh.py) | 对应数据源的准备和映射 |
| 数据划分/分类加载 | [data_utils/cls_datasets.py](ECG_SSL_LFBT-main/data_utils/cls_datasets.py)、[data_utils/data_folder.py](ECG_SSL_LFBT-main/data_utils/data_folder.py) | 样本读取与分类任务 |
| 多标签与扩展指标 | [utils/multilabel.py](ECG_SSL_LFBT-main/utils/multilabel.py)、[metrics_ext.py](ECG_SSL_LFBT-main/metrics_ext.py) | 评价基础；存在这些工具不等于原生多标签训练已完成 |
| 测试 | [tests/](ECG_SSL_LFBT-main/tests/) | 按被修改模块选择相关测试 |

当前 V7 基于 8 个独立导联与长度 2048 的输入，不能因数据原始包含 12 导联而写成已完成所有 12 导联结构验证。原生多标签训练须另建协议，不能只把既有 softmax 分数换一种真值重算。

## 4. 论文证据入口

所有以下入口位于真实账本 `ECG_SSL_LFBT-main/runlog/`。引用时记录语料、方法、评估任务、标签口径、种子、划分和结果版本。

| 要核对的内容 | 入口 |
|---|---|
| 小论文历史状态/当前材料缺口 | [收官总账](docs/paper/PAPER_CLOSEOUT_2026-10-23.md)、[V7 表图证据映射](docs/paper/PAPER_EVIDENCE_MAP_2026-10-23.md)；不等同主机认领队列 |
| 当前任务与主机状态快照 | [HOSTS.md](ECG_SSL_LFBT-main/runlog/HOSTS.md)；先读顶部当前有效区，下文全部旧队列/活跃措辞均为历史语境；当前无本轮新下发任务 |
| 联合框架最初完整链路 | [JOINT 报告](ECG_SSL_LFBT-main/runlog/JOINT_20261013/JOINT_最终报告.md)、[protocol.md](ECG_SSL_LFBT-main/runlog/JOINT_20261013/protocol.md)、[results.csv](ECG_SSL_LFBT-main/runlog/JOINT_20261013/results.csv)、[paired_stats.csv](ECG_SSL_LFBT-main/runlog/JOINT_20261013/paired_stats.csv) |
| 完整因子、拆分与反事实 | [ABC 冻结协议](ECG_SSL_LFBT-main/runlog/ABC_20261015/protocol_frozen.md)、[ABC 报告](ECG_SSL_LFBT-main/runlog/ABC_20261015/ABC最终报告.md)、[paired_stats.csv](ECG_SSL_LFBT-main/runlog/ABC_20261015/paired_stats.csv)、[factorial_stats.csv](ECG_SSL_LFBT-main/runlog/ABC_20261015/factorial_stats.csv) |
| LP 与噪声汇总 | [table_lp_3domains.csv](ECG_SSL_LFBT-main/runlog/ABC_20261015/table_lp_3domains.csv)、[table_noise27.csv](ECG_SSL_LFBT-main/runlog/ABC_20261015/table_noise27.csv)；结合 JOINT 来源及具体比较对 |
| 最新正式 FT10 补齐 | [W9 最终报告](ECG_SSL_LFBT-main/runlog/W9/W9_最终报告.md)、[table4_complete.md](ECG_SSL_LFBT-main/runlog/W9/table4_complete.md)、[results.csv](ECG_SSL_LFBT-main/runlog/W9/results.csv) |
| TRC / 效率补跑裁决 | [2026-10-20 裁决备忘](ECG_SSL_LFBT-main/runlog/W9/裁决备忘_TRC匹配与效率表_2026-10-20.md) |
| W8 新增对照 | [W8 目录](ECG_SSL_LFBT-main/runlog/W8/)：Random/SimCLR-Physio/TSR、Affine、Chapman LP 和噪声隔离已有结果；`w8_dvsplit_results.csv` 为错误 R 峰缓存的历史记录，已由 JOINT 更正结果替代；W8 单独最终报告本轮未定位 |
| CPSC 原始多诊断真值 | [cpsc_multilabel/](ECG_SSL_LFBT-main/runlog/preds_archive/cpsc_multilabel/)、[导出脚本](ECG_SSL_LFBT-main/runlog/preds_archive/export_cpsc_multilabel.py)；9 类顺序须与预测对齐 |
| 逐记录预测归档 | [preds_archive/](ECG_SSL_LFBT-main/runlog/preds_archive/)、[archive_manifest.json](ECG_SSL_LFBT-main/runlog/preds_archive/archive_manifest.json)；按对应压缩包和 sample_order 取数 |
| 早期数值总账本 | [2026-10-08 数据总报告](报告归档/已完成报告/数据总报告-论文换数版-2026-10-08.md)；只作该日期前入口，不覆盖 W8 / JOINT / ABC / W9 |

ABC 报告有自动装配的条件性判读模板。尤其是 R 峰“必要性”、RR 独立贡献和协同显著性，应查看实际统计与 V7 的最终限定措辞，不能把模板中的“若……则……”当成结论。

## 5. 阶段账本速查

| 阶段 | 内容 |
|---|---|
| W0–W3 | 数据审计、早期矩阵、C1/C2 链路、鲁棒性与失败方向归档 |
| W4–W5 | 外部基线、CPSC 审计与重评、统计与 LP 材料 |
| W6 | FT20/40、LCM 判负、双视角早期结果、种子扩展 |
| W7 | TRC 跨语料验证、机制、LP-FT、TAPT 与质量门控边界、噪声增强 |
| W8 | Random init / SimCLR-Physio / TSR、Affine、拆分与噪声独立性对照 |
| JOINT_20261013 | TRC + Beat-RR + 噪声联合框架 |
| ABC_20261015 | 全因子补齐、目标拆分、反事实、统计与效率汇总 |
| W9 | 投稿用 FT10 矩阵补齐、校验与最终裁决 |

旧 B0/C1/C2 与当前 M0–M9 的身份、预训练语料和批次要分别核对；不能仅看名称近似就配对计算增量。

## 6. 新研究前先查的负结果

| 方向 | 当前证据与限制 |
|---|---|
| 原始导联相关矩阵软匹配 LCM | [W6 报告](ECG_SSL_LFBT-main/runlog/W6/W6_最终报告.md)：CPSC LP seed0 增量 −0.12 个百分点，按当时门槛关闭；不是未做 |
| D1L / ACL / LGA、旧重建/掩码等 | [失败方向索引](ECG_SSL_LFBT-main/runlog/W3/failed_directions_index.csv)、[失败实验归档](报告归档/已完成报告/失败实验归档-2026-09-22.md) |
| 频带功率比事后质量门控 | [W7 报告](ECG_SSL_LFBT-main/runlog/W7/W7_最终报告.md)、[val 扫描](ECG_SSL_LFBT-main/runlog/W7/e4b_val_sweep.csv)：干净性能约束未满足，验证阶段关闭 |
| 简单目标域续训 TAPT | [e3_summary.md](ECG_SSL_LFBT-main/runlog/W7/e3_summary.md)：跨域小增益伴域内代价 |
| 当前 Beat-RR 的边界 | V7 §4.3 / §5.2：M4−M6 约 −0.051 个百分点；拆分和反事实校正后不显著。不能写成 RR 无用或 R 峰对齐必要性已证实 |

新路线图可提出不同机制的候选，但不自动翻案或启动旧线。需要先明确新假设与旧负结果的实质差别。

## 7. 后续工作入口与维护

- [整体实验设计](docs/research/EXPERIMENT_DESIGN_2026-10-23.md)、[主机 A 首批任务书草案](docs/experiments/THESIS_P0_20261023/TASKBOOK.md)：拟按预检→多标签 LP→匹配表示诊断→HeartLang 单任务验证执行；9+12+1 个主作业，训练/验证范围与建议预算写入协议。当前仅本地设计，未下发或启动。
- [小论文收官总账](docs/paper/PAPER_CLOSEOUT_2026-10-23.md)：历史任务接续、关闭/替代裁决、独立冻结材料缺口和本轮整理核验。`报告归档/未完成报告/` 内的 W5A/W5B/W6 原任务书保留原路径，不能按目录名启动旧任务。
- [论文与代码总览](../参考代码/论文与代码总览.md)、[PDF 下载记录](../参考代码/论文下载清单.json)、[本地参考代码总目录](../参考代码/README.md)、[复现与移植顺序](../参考代码/实验推进建议.md)：公开全文与固定源码分别记账；原始快照与主实验分开；先最小运行/行为核对，再按问题移植，不同时完整重训所有论文。
- [近期论文代码与发表平台核查](docs/research/CODE_AVAILABILITY_AUDIT_2026-10-23.md)：实际源码入口、公开权重、缺失部分、固定 commit 及复现顺序；尚未运行复现。
- [选线评估与近期文献补充](docs/research/THESIS_DIRECTION_ASSESSMENT_2026-10-23.md)：结合 V7 与原始配对统计，比较可靠心拍表示、可变导联、测试时适配和类特异证据定位；均为候选建议。
- [研究路线图](docs/research/RESEARCH_ROADMAP_2026-10-23.md)：推荐方向、最小矩阵、停止条件和建议节奏。
- [阅读清单](docs/research/READING_LIST_2026-10-23.md)：论文原始链接、阅读重点、与本项目的区别。
- [开题报告](报告归档/广东工业大学研究生学位论文开题报告.md)：研究内容和学位论文定位。

本次只新增导航和研究计划，未训练、未改模型、未重算结果。目录新增时按职责维护一行入口；不为每个临时日志新增启动必读项。新增实验另用协议 ID 与目录，保留失败记录；大数据、权重、缓存和已有 `.gitignore` 规则不因导航整理而改变。

## 8. 工作区与 Git 减负

[2026-10-23 减负审计与清单](docs/maintenance/CLEANUP_AUDIT_2026-10-23.md)：本地 Git 已打包并保留不可达对象，新增开发缓存忽略规则；没有提交、推送、重写历史或删除已跟踪文件。已将 1,541 个未跟踪渲染/缓存文件（约 137.38 MiB）移入 Windows 回收站，未清空回收站。V6/V7、图源、冻结账本、逐记录预测、历史失败记录及全部原有未提交改动均保留；清理后核验通过。清单、回收日志与核验结果位于 `docs/maintenance/cleanup_recycle_2026-10-23/`，无需将其加入新会话必读内容。
