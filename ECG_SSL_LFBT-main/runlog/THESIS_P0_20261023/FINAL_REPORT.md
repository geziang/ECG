# THESIS_P0_20261023 最终报告（P1）

协议 `thesis-p0-20261023-v1`。主机 A=Win4090（RTX 4090 24G，conda DL：Python 3.11.9 / torch 2.5.1+cu121 / wfdb 4.1.2）。认领 2026-10-23 22:57（`3c3c35c`），收官 2026-10-24 00:3x。**22/22 主作业全部完成（A2×9 + A3×12 + A4×1，唯一键无重无缺），零 partial，测试集信号/标签访问计数=0。** 按任务书 §9 停在 P1 报告，未启动任何新预训练。

## 预算对账（上限/实用）

| 预算 | 上限 | 实用 |
|---|---|---|
| GPU 进程墙钟 | 12 h | **0.24 h**（34 段 GPU 记录，A4 占 535.6s） |
| CPU 计算 | 6 h | 0.06 h（峰检测 54.9s + 分词/导出） |
| 工程排障与适配 | 8 h | ≈1.6 h（含 A4 三次第三方环境启动排障，各≈5 min） |
| 主训练作业 | 22 | 22 |

账本：`resource_usage.csv`（37 行）；逐作业 `jobs/<job_id>/result.json`（含权重/数据/代码 SHA、耗时、峰值显存、停止原因、复算最大差）。

## A0 预检（全部通过）

机器/环境实测过；九 checkpoint（M0=confirm/c1、M4=joint_m4_noise、M6=joint_m6）SHA256 与来源锁逐个一致并备份核验（`D:\thesis_p0_20261023_backup\checkpoints\`）；参考源码 HeartLang@a08afd8 / ecg_tokenizer@ce976ce / shapeefm@9bb1b6e 克隆 + 9/9 锁定文件哈希过（本机 autocrlf 修复后）；HeartLang 主 checkpoint（HF rev b8118c1，535MB）实测 SHA256==LFS oid；代码快照 `code_snapshot_3c3c35c.zip`。local_assets 6/7：`export_cpsc_multilabel.py` 哈希不符系办公机 f106744 等价重构（锁哈希的是重构前工作文件），其产物 `cpsc_multilabel.csv` 逐字节与锁一致，标签链完好（定因见 events.md）。

## A1 数据契约（通过）

4809/683/1385 计数、类序（NSR,AF,IAVB,LBBB,RBBB,PAC,PVC,STD,STE）、多诊断 train 343/val 36、canonical 样本序、跨 split 零重叠、dir-主类 0 不符全部通过；train/val 导出隔离副本（SHA 锁 + 逐记录源 SHA）；test 仅 CSV 元数据审计（数量/身份对齐 cpsc_sample_order [test] 段），其 ECG 内容从未读取。**记录两项实测**：①归一化=整记录全局标量 z-score（mean 0.0000/std 1.0000 实测；协议文"逐导联"表述与实测不符，以实测为准，未二次归一化）；②train 字节级重复内容 131 组（记录级观察；CPSC 无患者 ID，全部去重/重叠结论均为记录级）。

## A2 冻结编码器多标签 LP（9/9，验证集 macro-AUPRC）

| 模型 | seed0 | seed2 | seed4 | 均值 | 均值 AUROC |
|---|---|---|---|---|---|
| M0 | 0.7131 | 0.7171 | 0.7228 | **0.7177** | 0.9296 |
| M4 | 0.7264 | 0.7276 | 0.7266 | **0.7269** | 0.9299 |
| M6 | 0.7316 | 0.7355 | 0.7283 | **0.7318** | 0.9322 |

三种子内排序一致（M6>M4>M0），多标签口径下与既有单标签结论方向相容。本表只描述 M0/M4/M6 的原生多标签表征，不作新方法胜负声明。seed0 工程门过（有限性/范围/复算≤1e-8/冻结不变）。

## A3 心拍表示诊断（12/12）——工程选线门**未通过（有效阴性）**

三种子均值：D0 matched_fixed **0.5973** > D1 mean_template 0.5925 > D3 beat_set 0.5865 > D2 median_template 0.5619。

选线门（协议 §engineering_selection_gate，参照=D0/D1/D2 中最强=D0）：
- D3−D0 均值 = **−0.0108**（要求 ≥ +0.005）✗
- 正向种子数 = **0/3**（要求 3/3）✗
- 逐类 AP 均值降幅（阳性≥20 的类，9 类全部入选）：IAVB −0.0285、PVC −0.0285、STE −0.0298 三处超 0.02 ✗

三条件全败 → **在当前记录级多标签诊断设置下，逐拍集合（D3）未提供超过非峰对齐对照（D0）的信息价值；建议不进入逐拍后续研究预算**（此为研究预算筛选结论，非统计/临床声明；门未在结果后放宽）。

峰扰动敏感性（仅验证推理，seed=20261023）：±4 样点（≈±20ms）|ΔAUPRC|≤0.002；±8（≈±40ms）Δ∈[−0.004,+0.002]——峰定位扰动不敏感，与上述阴性一致。峰缓存 5492/5492 成功（V5 5485 / II 回退 7 / 失败 0；无记录触发无峰回退，回退路径以合成用例验证）；小网络参数量 12713 逐位符合；64 条过拟合检查 0.684→0.050。

## A4 HeartLang 公开权重验证（1/1）

HeartLang_finetune_base（HF checkpoint-200.pth，SHA 核验），trainable=linear（6,921 参数），backbone 140 项权重全载、0 缺失（预训练专属 lm_head 因 VQ 词表尺寸 8192 vs 构造默认 1024 排除，不在 LP 前向路径，已记录）、训练前后 backbone 哈希不变。**验证集 macro-AUPRC 0.4464 / AUROC 0.8121 / BCE 0.2616，best_epoch 92（100ep 跑满），GPU 535.6s。**

相对作者流程的已记录偏差：①backbone 全程 eval（作者整模型 train()，任务书 §6 要求）；②选模=min 验证 BCE+patience15（作者按 AUC）；③split/multihot 用本项目冻结版（作者 icbeb csv 为 4950/551/1376）；④标签列序=本项目类序（作者 csv 序仅作映射）；⑤作者 utils.load_state_dict 语义等价复刻（其文件顶部 tensorboardX 依赖本环境无）。**本表是公开模型在明确本地适配设置下的验证，不构成对其论文数字的复现声明，也不与 A2 表做绝对数字跨表排名**（输入契约不同：作者管线取前 5s→QRS token@100Hz，信息量显著低于本批 10s 全记录输入）。

## 限制

1. 全部指标为验证集、记录级；无患者 ID，无患者级去重（train 131 组字节级重复内容保留并记录）。
2. A3 无逐拍病理真值：即使 D3 优于模板也不能定位具体异常心拍——本批 D3 未过门，此限制不触发。
3. A4 的五项偏差使该行身份=本地适配验证；HeartLang 原生 12 导联/全时长潜能未测（超出本批范围）。
4. gqrs/xqrs 输出非人工峰真值；峰扰动仅确定性平移，未测真实漏检/错检。
5. 历史 CPSC test 在先前批次中被使用过（historical_cpsc_test_is_not_unseen=true），本批未新增任何 test 访问。

## 交付物清单

`runlog/THESIS_P0_20261023/`：protocol_resolved.json（首训前冻结于 7d60621）/ source_and_checkpoint_manifest.json / data_manifest.json / preflight.json / refs_verification.json / dry-run 等价物=各阶段 events.md 运行记录 / results.csv（22 行，固定 schema，summarize 重建）/ summary_local_lp.csv / summary_representation.csv / summary_heartlang.csv / events.md / resource_usage.csv / resume_manifest.json / jobs/<22>/（参数、best checkpoint 引用、val 预测 record_id+y_true+y_prob、日志）/ peak_sensitivity/（18 组配对扰动输出+summary）/ FINAL_REPORT.md（本文）。

Git 外存储（`D:\thesis_p0_20261023_backup\`，含 SHA 记录）：九 checkpoint 备份、代码快照、三参考仓库固定 commit 克隆、HeartLang 权重、A4 完整模型（`heartlang_lp_s0_model_best.pth`，sha256 `88d11a7d82985b5aec23392ee71490746b50417d17962ccd46b7fae92c6addef`）、train/val 隔离数据与特征缓存。A2 头 checkpoint（~18KB×9）与全部逐记录预测（28K×22 npz）已随 git 交接。

## 验收自检（任务书 §9）

唯一键 22/22 无重无缺 ✓；每行配置+预测齐全、形状/顺序（canonical record ID 序）正确 ✓；独立复算 AP/AUROC/BCE 与结果差 ≤1e-8（逐作业断言+记录 max diff）✓；冻结参数/来源检查全过（backbone 哈希、权重 SHA、数据 SHA、代码 SHA）✓；测试访问计数 0 ✓；预算可对账 ✓；无单一论文环境支线缺失（A4 完成但有五项已记录偏差）✓。

## 下一阶段（P2 决策输入，非自动执行）

1. 逐拍线：选线门三条件全败——**建议关闭**，除非 P2 协议明确引入新信息源（如逐拍真值或更长上下文）后重立。
2. M6 在多标签口径下的优势与 W9 单标签结论同向——多标签原生基准表可直接用于论文附录。
3. HeartLang 本地适配行可作为"公开模型+明确设置"的对照参照；如需其原生设置数字，须另立协议（全 12 导联/全时长/官方 split），本批不做。
4. 本批未动用预算余量（GPU 0.24/12h）；不据此自动扩批。
