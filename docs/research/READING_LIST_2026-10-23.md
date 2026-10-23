# 后续研究阅读清单

检索与核查日期：2026-10-23。返回 [研究路线图](RESEARCH_ROADMAP_2026-10-23.md) / [项目地图](../../PROJECT_MAP.md)。

本清单按“能解决本项目哪个问题”排序。已核查论文原始页面、题录和摘要，HeartLang 与 tokenizer 对照研究另查看了 HTML 正文；不等于已逐篇完成全文精读、代码审计或复现，也不是穷尽性新颖性检索。下面的“可借鉴”属于结合本项目的研究判断。

本轮新增入口与阅读状态见第 6 节；选线建议见 [大论文选线评估](THESIS_DIRECTION_ASSESSMENT_2026-10-23.md)。后续源码静态核查及权重/发表平台比较见 [代码可用性核查](CODE_AVAILABILITY_AUDIT_2026-10-23.md)，优先 HeartLang、TolerantECG 和 tokenizer 对照代码；CODE-II 的公开下载尚未核实。

本地入口：[论文全文与代码总览](../../../参考代码/论文与代码总览.md)、[PDF 下载及校验记录](../../../参考代码/论文下载清单.json)、[九篇参考代码归档](../../../参考代码/README.md)、[复现与移植顺序](../../../参考代码/实验推进建议.md)。公开论文原文件保存于对应目录 `paper/`，实际下载结果以清单为准；下载、全文精读、代码运行与数值复现分别记录，不将文件下载完成视为已完成精读或复现。

当前文献插件未返回可用工作空间，本次使用公开原始论文、正式会议页面及作者论文页面检索；未创建外部文献库或启动付费/长时深度检索。

## 1. 第一批：直接决定新方向

### 1）HeartLang — Reading Your Heart: Learning ECG Words and Sentences via Pre-training ECG Language Model

- 作者：Jiarui Jin 等。ICLR 2025；[原始论文](https://arxiv.org/abs/2502.10707)、[HTML 正文](https://arxiv.org/html/2502.10707v1)。
- 内容：利用 QRS-Tokenizer 将心拍组织成 ECG“词/句”，学习形态与节律表示。
- 为什么先读：拟议“逐心拍序列”与它直接相邻，必须先弄清已有边界。不能将心拍 token、RR 或 Transformer 拼接就称为首次提出。
- 精读重点：token 定义、空间与时间位置编码、预训练目标、短记录处理、QRS 失败处理、少标签与少导联实验、预训练语料与下游重叠。
- 读后产物：一张“当前平均心拍—HeartLang—拟议可靠性机制”的差异表；列出能排除简单复现解释的两个对照。

### 2）On the role of the tokenizer in ECG transformer models

- 作者：Jiawei Li 等。2026-09-14 的 arXiv 预印本；[原始论文](https://arxiv.org/abs/2609.15433)、[HTML 正文](https://arxiv.org/html/2609.15433v1)。截至本次核查，不将其标为已确认期刊/会议论文。
- 内容：在其 CPSC2018 设定中控制主要主干容量，比较八类 tokenizer；median-beat 与 HeartLang 表示均表现较强。
- 对本项目的价值：提醒我们保留简单模板作为强对照。不能把“平均会削弱偶发异常”直接等同于“逐拍一定更好”。
- 精读重点：标签与划分、token 数量、输入维度、参数/计算匹配、归一化、每次训练预算，以及结论依赖单数据集的限制。
- 读后产物：平均模板、中位模板、无序心拍集合、有序心拍序列四种诊断的公平比较表。

### 3）Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL

- 作者：Nils Strodthoff、Patrick Wagner、Tobias Schaeffter、Wojciech Samek。IEEE JBHI 2021，25(5):1519–1528；[DOI](https://doi.org/10.1109/JBHI.2020.3022989)、[作者预印本](https://arxiv.org/abs/2004.13701)。
- 内容：提出 PTB-XL 的结构化任务基准，并讨论不确定性与可解释性。
- 对本项目的价值：先把诊断/形态/节律任务和标签层级定义清楚，再评价新方法。当前旧五类单标签任务与新原生多标签任务不能混为一谈。
- 精读重点：任务标签集合、样本排除、患者划分、验证与阈值、类别支持数、指标定义、隐藏亚组差异。
- 读后产物：新多标签数据协议和评估表模板。原始论文不是本地当前数据管线正确性的证明，仍要检查实际 manifest。

## 2. 第二批：近期方法基线与对照设计

### 4）ST-MEM — Guiding Masked Representation Learning to Capture Spatio-Temporal Relationship of Electrocardiogram

- 作者：Yeongyeon Na、Minje Park、Yunwon Tae、Sunghoon Joo。**ICLR 2024**；[原始论文](https://arxiv.org/abs/2402.09450)。
- 内容：面向 12 导联 ECG 的时空 masked modeling，并评价不同导联组合的适用性。
- 阅读目的：理解导联身份、时空 token 和任务设置，作为可变导联路线的已有方法参照。
- 本项目限制：已有旧掩码/重建负结果，不因读到 ST-MEM 就自动重启这些训练轴。先比较结构、目标、预算和失败机制。
- 读后产物：与 LFBT 的信息保留方式对照；是否值得引入公开 checkpoint 的可行性结论。

### 5）ECG-JEPA — Learning General Representation of 12-Lead Electrocardiogram with a Joint-Embedding Predictive Architecture

- 作者：Sehun Kim；[原始预印本](https://arxiv.org/abs/2410.08559)，2024 年首次提交，本次页面所示最新修订为 2026-04-10。
- 内容：在潜在空间预测被遮蔽表示，使用 Cross-Pattern Attention 建模多导联模式。
- 阅读目的：理解“预测表征”与“重建波形”分别保留什么，帮助解释我们旧波形重建负结果的适用边界。
- 注意：搜索可见其期刊卷期标为 2027，本清单按当前已公开预印本讨论，不把未来卷期当成当前发表日期。同名 ECG-JEPA 还有其他作者工作，引用时以作者、题名及 arXiv ID 区分。
- 读后产物：一页目标函数与反事实设计分析；不直接变成另一轮 JEPA/掩码扫描。

### 6）ECG-FM: an open electrocardiogram foundation model

- 作者：Kaden McKeen 等。JAMIA Open 2025，8(5):ooaf122；[DOI](https://doi.org/10.1093/jamiaopen/ooaf122)、[开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12530324/)。
- 内容：开放权重的 ECG 基础模型，评价多标签及低标签任务。
- 阅读目的：寻找可以直接用冻结编码器或标准微调评价的现代参照，而不是在本项目中重训百万级基础模型。
- 精读重点：输入采样与导联、预训练数据清单、公开权重版本、标签体系、adapter/分类头、语料重叠风险。
- 比较边界：开放权重模型的预训练规模与 LFBT 不同，单列为迁移参照；不能混进“同语料同预算”的模块归因表。
- 名称辨析：该 ECG-FM 与其他作者发表于 Information Fusion 的“ECGFM”不是同一模型，按 DOI 区分。

## 3. 第三批：备选导联路线与更大范围背景

### 7）K-MERL — Knowledge-enhanced Multimodal ECG Representation Learning with Arbitrary-Lead Inputs

- 作者：Che Liu 等。Findings of EMNLP 2025；[正式论文页](https://aclanthology.org/2025.findings-emnlp.385/)、[DOI](https://doi.org/10.18653/v1/2025.findings-emnlp.385)。
- 内容：从临床报告提取结构化知识，结合 lead-aware 编码和动态导联 masking 处理任意导联输入。
- 阅读目的：核对可变导联机制的已有边界，学习导联身份建模与训练目标如何配合。
- 本项目限制：尚未核验配对临床自由文本报告资源。不能把类别名称或大模型生成文字视为拥有同等多模态监督。
- 读后产物：可复用的纯信号结构与依赖报告数据的机制分别列出；判断 P2 是否还有清晰研究空间。

### 8）ECG-LFM — A self-supervised electrocardiogram foundation model for empowering cardiovascular disease prediction and genetic factor discovery

- 作者：Siying Lin 等。Nature Communications 2026，17:5771，页面标注 2026-04-27；[正式论文](https://www.nature.com/articles/s41467-026-72436-2)、[DOI](https://doi.org/10.1038/s41467-026-72436-2)。
- 内容：大规模多导联预训练结合对比与 masked modeling，并开展多任务及遗传关联分析。
- 阅读目的：参考近期对多数据源、泛化范围和下游任务的论证方式，更新大论文综述。
- 实施定位：其数据规模与任务资源不适合作为当前必须复现的训练计划；是否有适用权重需另核验。本次主要核查正式页面检索内容，未完成全文精读。

## 4. 已有论文怎样用

LFBT 仍是方法来源与当前实验对照的基础，应保留，但阅读重点转为核对输入契约、增强、损失和代码实现。SimCLR、Barlow Twins、CLOCS 与已有 TRC 机制来源，在相关章节补读即可，不应占据本轮新方向阅读的全部时间。

## 5. 每篇统一记录六项

1. 它要解决什么具体问题，证据是否支持？
2. 输入、导联、采样、记录长度与标签契约是什么？
3. 预训练语料与下游有没有重叠，划分单位是什么？
4. 真正起作用的是表示、目标、架构还是数据规模？有哪些匹配对照？
5. 与本项目当前设计和失败记录有什么实质差别？
6. 最小可复用实验是什么；什么结果会让我们停止这条线？

建议第一轮顺序：**HeartLang → tokenizer 对照预印本 → PTB-XL 基准 → ECG-FM → ST-MEM / ECG-JEPA → K-MERL → ECG-LFM**。先形成差异表和实验协议，再决定是否增加更多论文或复现任务。

## 6. 本轮新增：2026 年近邻方法、鲁棒性与数据资源

以下是对前五节的补充。优先顺序更新为 **HeartLang → Beat-SSL / ShapeEFM → tokenizer 对照 → PTB-XL 基准**；选择导联、适配或解释方向后再进入对应分组。先读原文方法和局限，再判断复现范围。

| 论文 | 日期与状态 | 对本项目的直接作用 |
|---|---|---|
| [Beat-SSL: Capturing Local ECG Morphology Through Heartbeat-Level Contrastive Learning with Soft Targets](https://doi.org/10.1109/ISBI61048.2026.11515777) | ISBI 2026；[作者机构记录](https://eprints.gla.ac.uk/376785/)、[预印本](https://arxiv.org/abs/2601.16147) | 心拍/节律双层对比及软目标已有先例；借鉴多标签与分割的双任务评价 |
| [ShapeEFM: A Shape-Based Pretrained Foundation Model on ECG Morphology Comprehension](https://papers.miccai.org/miccai-2026/0954-Paper2861.html) | MICCAI 2026，会议页列 September 2026 | 用形态趋势变化组织 token；与逐拍表示一起限定新颖性范围。正式页有作者代码入口，本轮未审计 |
| [Test-Time Adaptation for ECG Classification via SQI-Gated Self-training and Beat-Rhythm Consistency](https://papers.miccai.org/miccai-2026/1040-Paper2304.html) | MICCAI 2026；[2026-08-24 预印本](https://arxiv.org/abs/2608.23347) | 已有质量门控、伪标签和双层一致性；借鉴离线/持续/独立在线协议，进一步研究每类证据和错误累积 |
| [TolerantECG: A Foundation Model for Imperfect Electrocardiogram](https://doi.org/10.1145/3746027.3755287) | ACM MM 2025；[原文](https://arxiv.org/abs/2507.09887) | 噪声与任意缺导已被联合处理，随机缺导本身不能作为新点；其文本可由 SCP 诊断生成，需区分标签监督来源 |
| [IECG: Weakly-Supervised Interpretability-Enhancing Foundation Model for Electrocardiogram](https://arxiv.org/abs/2610.28083) | 2026-10-20 公开预印本；页面有作者关于 WACV 2027 接收的备注，本清单按已公开预印本讨论 | 规则伪标签监督注意力；局限明确提出多诊断下尚不能按类别定位，适合作为储备问题。标签辅助步骤不能称纯无监督 |
| [Domain Knowledge is Power: Leveraging Physiological Priors for Self-Supervised Representation Learning in Electrocardiography](https://doi.org/10.1109/TBME.2026.3656904) | IEEE TBME，2026-01-22 在线；[作者预印本](https://arxiv.org/abs/2509.08116) | PhysioCLR 将生理相似性用于对比关系，可借鉴如何定义生理目标，而不是只叠加 RR 回归头 |
| [CoRe-ECG: Advancing Self-Supervised Representation Learning for 12-Lead ECG via Contrastive and Reconstructive Synergy](https://arxiv.org/abs/2604.11359) | 2026-04-13 预印本 | 结合对比/重建、频率增强和时空 masking；参考增强破坏与导联捷径的诊断，不自动重启旧重建轴 |
| [CODE-II: a large-scale dataset for artificial intelligence in ECG analysis](https://www.nature.com/articles/s41746-026-02704-4) | npj Digital Medicine，2026-05-26 | CODE-II-open 为新的外部数据候选。全库受限；CODE-II-test 标签不公开，不能把它当本地可计分的完整公开测试集 |
| [Foundation models for electrocardiogram interpretation: clinical implications](https://doi.org/10.1093/eurheartj/ehaf1119) | European Heart Journal，2026-01-22 在线 | DeepECG 的多标签、多机构及少标签评价可借鉴；大规模基础模型定位为迁移参照，不要求本项目从头复现 |

核查深度：文献阅读阶段对 Beat-SSL、BeatRhythm-TTA、TolerantECG 和 IECG 定向查看了 HTML 方法/局限；ShapeEFM 已读正式页摘要，PDF 后续读取超时；其他新增项以正式题录、摘要及页面为主。后续已补充 [源码静态核查](CODE_AVAILABILITY_AUDIT_2026-10-23.md)，尚未执行复现。CODE-II 作者仓库当前只有 README，PhysioNet 链接仍写 coming soon，下载可用性不能视为已确认。
