# W4 B-5 基线实现审计底稿（baseline_impl_audit.md）

> 主机B（DESKTOP-0PBLCND），2026-09-23 先行版。官方论文/仓库参照已逐条核对（见 §2/§3）；
> **A-4/A-5 实现核对栏（§5）留空待主机A实现落地后回填**——按任务书 B-5：
> 发现实现偏差 → 立即停止该基线入表并上报。
> 论文核对来源：SimCLR = arXiv:2002.05709v3（ICML 2020）+ 官方 repo README；
> CLOCS = arXiv:2005.13249v3（IEEE TBME 2021）。

## 1. 公平性条款基线（任务书 §1.4，适用于 S1/S2/S3 全部）

- 一律"发表默认超参或 C1 等价配置"，禁止任何超参搜索；
- 与 C1 的全部差异逐条写入 `runlog/W4/baseline_hparams.csv`；
- backbone / projector / 增强 / batch / 优化器 = C1 等价（`--trc 0`，NFH 100ep，batch 128）。

## 2. S1 = SimCLR 官方参照表

| 项 | 官方值与出处 | 任务书 A-4 条款 | 一致性 |
|---|---|---|---|
| 损失 | NT-Xent：`ℓ_{i,j} = -log[ exp(sim(z_i,z_j)/τ) / Σ_{k=1..2N, k≠i} exp(sim(z_i,z_k)/τ) ]`，sim=ℓ2 归一化后余弦；对 (i,j) 与 (j,i) 双向求均值（论文 Eq.1 + Algorithm 1） | `--loss-mode simclr` NT-Xent | 待 A-4 实现 |
| 温度 τ | 论文正文默认口径两层：官方 repo **CIFAR-10 预训练 = 0.5**（batch 512）；**ImageNet 主结果 = 0.1**（batch 4096；论文 Table 5 消融中 0.1 优于 0.5） | **0.5** | 口径说明：任务书 0.5 与官方 CIFAR（中等 batch 单机）口径一致；本仓 batch 128 单卡场景选 0.5 合理，hparams_ref 需记"CIFAR 口径"（若审稿人问及 ImageNet 口径差异，以同预算条款回应） |
| 正对 | 同实例两增强视图（单正对）；负样本 = batch 内其余 2(N−1) 个增强实例，**不显式采样** | 两视图 RRC-TimeOut `0.5,1.0,0.0,0.5`（与 C1 相同） | 增强按公平条款用 C1 的 |
| projector | 论文正文 MLP（单隐层 ReLU）；官方 repo `--num_proj_layers=3`、2048 维 | 2048-2048-2048 | 一致（repo 三层口径） |
| 优化器/batch | 论文 LARS + batch 256–8192（大规模口径） | 与 C1 全同（`--trc 0`，batch 128） | 公平条款覆盖（差异入 csv） |
| 单测要求 | — | ①默认 `--loss-mode bt` 与现 HEAD 逐位一致；②NT-Xent 数值对手工小例 | **审计关键点①**：分母必须排除 k=i（自身）；与 CLOCS（§3）分母含自身相反，两实现不得共享分母逻辑而不加开关 |

## 3. S2 = CLOCS 官方参照表

| 项 | 官方值与出处 | 任务书 A-5 条款 | 一致性 |
|---|---|---|---|
| 损失 | 多正对 NT-Xent 变体（论文 Eq.1–4）：Eq.1 `s = cos(h_A^i, h_B^k)/τ`；Eq.2 diag 项 `−E_i log[ e^{s(i,i)} / Σ_j e^{s(i,j)} ]`（**分母含全部 j，包括正对自身**）；Eq.3 off-diag 项为同患者跨实例对（同患者 mini-batch 重现时）；Eq.4 = A→B、B→A 双向 + diag/off-diag 四项组合 | 多正对 NT-Xent 变体，正对定义按官方默认 | 待 A-5 实现 |
| 温度 τ | **0.1**（论文 §5.5，"as per Chen et al. 2020"） | （未显式指定 → 按官方 0.1） | 待 A-5 实现；**注意与 S1 的 0.5 不同**，入 baseline_hparams.csv |
| temporal 正对 | CMSC：同记录 V 个不重叠相邻时间段互为正对 | temporal 正对=同记录不同时间窗 | 一致 |
| spatial 正对 | CMLC：同时刻不同导联（子集）互为正对 | spatial 正对=同窗不同导联子集 | 一致 |
| 增强 | 论文 §5.4：spectrogram 域 SA_t/SA_f 掩码 + 时域顺序扰动（多种组合消融） | 增强与 C1 相同（公平条款） | **差异点**：增强不按 CLOCS 官方而按 C1——由公平条款裁定，逐条记录 |
| 网络 | 官方 = 3×Conv1D(BN+ReLU+MaxPool2+Dropout0.1) + 2×FC（论文 Appendix 表）；**非 VGG16** | "结构优先同构 C1；结构冲突时以官方默认为准并逐条记录" | ⚠️ **条款张力**：若 backbone 换官方 3conv+2FC 则违反 §5"同 backbone 家族"措辞边界。审计建议：损失/正对/τ 按官方，backbone 保持 C1 同构（VGG16-1D 逐导联），差异入 baseline_hparams.csv——**请下发方在 A-5 开工前确认此取舍** |
| 优化器 | Adam，lr=1e-4，batch 256（PhysioNet2020/Chapman/2017；Cardiology=16）（论文 Appendix 表 6） | 与 C1 全同 | 公平条款覆盖 |
| 官方仓库 | 论文给出 **github.com/danikiyasseh/CLOCS** | 任务书写 `tomato1min/clocs` | 待 A 机核对两地址可达性/一致性（疑似同人两账号），以论文地址为权威出处 |
| 单测要求 | — | 适配+单测（任务书未细列） | **审计关键点②**：①多正对分母包含正对（与 SimCLR 相反）；②off-diag 项在本仓数据管线（患者 ID 分桶）是否出现需实现方说明；③τ=0.1 |

## 4. S3 = 监督直训参照（无外部论文参照，条款即参照）

任务书 A-6：`run_ft.py` 省略 `--checkpoint`（随机初始化路径，代码已内建）、
`--fraction 1.0`、`--epochs 100 --batch-size 128 --learning-rate 0.0001`、PTB 五类、
seeds {0,2,4}；若 `--fraction 1.0` 不受校验支持 → 改全量 train split 等价实现并记录差异。
审计点：命令行与条款逐条比对 + "省略 --checkpoint 时确实走随机初始化"的代码路径确认。

## 5. A-4/A-5/A-6 实现核对栏（2026-09-24 主机B 回填终态）

核对基准：实现入库 f5c23c9（S1/S2/S3+A-8a）/ c4251f3（S3 幂等补丁）；账本 git_sha=2468a14。

| 基线 | 实现提交 SHA | 单测结果 | 逐条核对结论 | 偏差与处置 |
|---|---|---|---|---|
| S1 SimCLR | f5c23c9（run_pt.py `--loss-mode simclr`） | `tests/test_simclr_loss.py` 4/4 ✅ + `verify_bt_head_bitwise.py` 默认 bt 与 HEAD 逐位一致 ✅ | **全部一致**：①NT-Xent=论文 Eq.1/Algorithm 1（L2 归一化→cos/τ→对角掩码排除自身→交叉熵，双向）✓（审计关键点①通过）；②τ=0.5（`--simclr-temp` 默认，hparams_ref 已记 CIFAR 口径出处）✓；③正对=同记录同导联两视图、负对=批内 2N−2 ✓；④backbone/projector(2048×3)/增强(RRC-TO 0.5,1.0,0.0,0.5)/优化器/batch128 全部与 C1 同构（§A-4"唯一差异=目标函数"达成）；⑤BT 跨导联 loss_t 置 0（NT-Xent 结构性不适用，已记录） | 无偏差入表 |
| S2 CLOCS | f5c23c9（run_pt.py `--loss-mode clocs` + run_clocs_matrix.py） | A 侧单测 3/3（台账）✅ | **全部一致**：①官方口径=danikiyasseh/CLOCS `obtain_contrastive_loss`（与本底稿 §3 权威出处一致；任务书原写 tomato1min/clocs 的差异已由 A 机按论文地址采用）✓；②τ=0.1 ✓；③diag 双向项 `−mean(log(diag/整行和))` × 2，**分母含正对自身**（审计关键点②通过，与 S1 相反且不共享分母逻辑）✓；④temporal 正对=同导联两时间窗（CMSC 语义）、spatial 正对=同窗跨导联 i<j（CMLC 语义逐导联化）✓；⑤off-diag（同患者跨实例）：本仓 NFH 无 pid → 官方"无 pid"路径 loss_terms=2（底稿遗留审计点已由实现方 docstring 说明）✓；⑥损失归一化 Σ/(2×n_pairs) 与官方 loss_terms×ncombinations 等价 ✓；⑦**backbone 裁定=保持 C1 同构**（官方 3×Conv1D+2FC 差异逐条记录于 baseline_hparams.csv）——与本底稿建议一致，公平条款优先 | 表示层级差异（官方编码器末层 vs 本仓 projector 输出）已入 baseline_hparams.csv；增强按公平条款用 C1 的 RRC（非官方 spectrogram 掩码）已记录 |
| S3 监督直训 | f5c23c9 + c4251f3（run_s3_supervised.py） | 幂等+账本回读 ✅（s3×3 行入账） | **一致**：TFS 随机初始化（省略 --checkpoint 路径）、PTB 五类全量（fraction 1.0 等价）、seeds {0,2,4}、从第一次起带预测落盘 ✓ | 无偏差入表 |

**B-5 结论：三个基线实现与官方论文/仓库口径及任务书条款逐条一致，未发现实现偏差，全部放行入表。**
（对照材料：`runlog/W4/baseline_hparams.csv` 差异逐条 12+12 项完整；`predreplay_gate.csv` W2 对象 20/20 逐位。）

## 6. 阻塞说明（已解除）

A-4~A-8 曾被安全加固回归阻塞（详见 HOSTS.md 09-23 08:26/08:28 回写），已于当天由
主机A 修复（2468a14 safe-globals 白名单）并经 B 机复验解除；批次于 09-23 18:02
31/31 行收口。本节为过程留痕。
