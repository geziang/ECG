# 正向收益总清单（v2 门规口径，2026-10-07 16:4x 编制）

> **依据**：门规修订 v2（gate_revision_2026-10-07.md，用户指令"正向收益全报告，小正向也算正向"）。
> **数据红线**：cpsc 只引 W5 修正分区（1385 test）及以后；W2/W4 旧 cpsc 行=污染数据禁引（清单中仅作历史脚注）。ptbxl/chapman 的 W2/W4 行有效。
> **分级**：A=确认级正向（3/3 同向且 mean≥+0.10，或 record-level 检验过）；B=单种子过线正读数；C=带代价标注的单域正向；D=机制/解释级正果（不涨点但立论）；E=边缘读数（如实列示不入正）。
> 全部数字源自己入库账本，逐条可溯源（源文件标注在行内）。

## A. 确认级正向（论文正表可直接用）

**A1. LFBT 全面优于外部基线（跨域防御性正向，全部 3 种子）**
| 对比 | 域 | 效应 | 检验 | 源 |
|---|---|---|---|---|
| C1 vs SimCLR | cpsc LP | AUROC **+3.13pt** (0.9463/0.9150), AUPRC +9.49pt 3/3 | boot p=0.0; DeLong p=0.0 | W5/paired_stats.csv |
| C1 vs CLOCS | cpsc LP | AUROC **+0.45pt**, AUPRC +1.25pt 3/3 | boot p=0.0046; DeLong min p=0.00018 | 同上 |
| C1 vs CLOCS | ptbxl LP | AUROC **+1.59pt** | boot p=0.0; DeLong p=0.0 | W5/paired_stats_v3.csv |
| C1 vs SimCLR | chapman LP | AUROC **+0.88pt** | boot p=0.0; DeLong p=0.0 | 同上 |
| C1 vs SimCLR | ptbxl LP | AUROC +0.57pt | boot p=0.247（非显著，DeLong seed0 min p=0.007）——如实标注 | 同上 |
| 排序 C2>CLOCS>SimCLR | 三下游 | 3/3 同向, 无任一外部基线反超 C2 | — | W4/summary_w4_baselines.md |

**A2. TRC 第三域增益（论文第3章主线）**
| 对比 | 设置 | 效应 | 一致性/检验 | 源 |
|---|---|---|---|---|
| C2−C1 | cpsc LP AUPRC | **+0.83pt** (+0.70/+0.59/+1.20) | 3/3 同向 | W5/seed_sign_v3.csv |
| C2−B0 | chapman LP AUROC | **+0.31pt** (0.9963 vs 0.9932) | 3/3 同向; boot p=0.0; DeLong p=0.0 | W4 终表+W5/paired_stats_v3 |
| C2−B0 | chapman FT10 AUROC | +0.14pt | 3/3 同向（与 LP 同向） | W6/Stage3 |
| C2−C1 | ptbxl FT10 AUPRC | +0.65pt (+0.19/+1.38/+0.39) | 3/3 同向 | W2/paired_delta + W6 复核 |
| （注）cpsc C2−C1 AUROC | cpsc LP | +0.11pt (p=0.187) | 2正1平，非显著 → 归 E 档 | W5/paired_stats.csv |

注：events 中"TRC +0.47 过检"为值班轮综合读数（上表 cpsc 两指标均值），论文引用请用分指标原始数。

**A3. E1 整段-心拍双视角+RR头（心拍线最终定格，v2 升格）**
- cpsc LP Δ = +0.27/+0.26/+0.17，3/3 同向，**mean +0.2333pt ≥ +0.10 → 确认级正向**（v1 措辞"温和正"升格）。
- 证据链闭环：CCM 遮挡重建 −0.91 / 多段 CCM −0.17 / 双视角+RR头 三种子 +0.2333。
- 源：W7/events（05:23 值班轮）+ e1_dv_results.csv。

**A4. NFH 预训练抗退化（W3 A-2 结论，W6 真实噪声复核保持）**
- NFH 系（C1/C2）在扰动下比 B0 **少退化 ~1.2–1.35pt**；W6 Stage6 真实噪声 276 格 + clean 门 23/23 逐位通过，结论保持。
- 附带：TRC 鲁棒性中性（Δ0.14pt）——"TRC 不伤鲁棒性"作防御性读数。
- 源：W3/robustness_*.csv + W6/robustness_real_*.csv + W6 终报 Stage6。

**A5. B0 域内血缘优势（基座选择正向）**
- ptbxl LP：B0−C2 = +2.77pt（record-level boot p=0.0）；B0−C1 = +5.86pt（3/3）；cpsc LP：B0 0.9506 全场第一（W5 重评估）。
- 论文落点=语料-域匹配时同源基座占优，与 A2 合成"何时用谁"的完整故事。
- 源：W5/paired_stats_v3.csv + W2/paired_delta.csv + W5/lp_results.csv。

## B. 单种子过线正读数（须标"单种子"）

**B1. E2c TRC×B0（域移是活性成分）**：b0trc vs B0（seed0），cpsc LP **+0.30pt AUROC / +0.60pt AUPRC**（过预注册 +0.5pt 线，主指标 AUPRC）；ptbxl −0.43pt AUPRC 微损如实并列。TRC 第三域增益不依赖外部语料，获得第二种语料设置独立支持。源：W7/events 12:23。

## C. 带代价标注的单域正向

**C1. E3 TAPT cpsc 单域**：Δ=+0.11/+0.10/+0.09（3/3 同向，mean +0.10pt 达 v2 确认线）——但 ptbxl −0.49pt 破无伤容差，**方法级判负维持**；正读数入积累表必须附代价标注。源：W7/events 15:24。

## D. 机制/解释级正果（不涨点，立论用）

| # | 读数 | 一致性 | 源 |
|---|---|---|---|
| D1 | **E4a LP-FT 消除 CPSC 微调反转**（LP-FT 3/3 保 b0>c2；naive-FT 反转=形变产物非特征优劣）；LPFT−FT 增益 b0 +3.06pt / c2 +2.05pt | 3/3 | W7/e4a_summary.md |
| D2 | **E2d TRC 能量尺度重标定**：通道 RMS 跨域偏移三域一致缩小 14–19%（ratio 0.81–0.86），norm-level domain adaptation 机制表述 | 三域一致 | W7/e2d_summary.md |
| D3 | **E4d 反转节律依赖**：c2−b0 反转集中在 meanRR q2 +1.73pt/高 SDNN 桶；TRC 把 b0−c1 慢心率缺口减半 | 逐桶 336 行 | W7/e4d_summary.md |
| D4 | **E4c 患者级敏感性**：K=3 重划下 c2>c1 三次一致（+0.6~+1.5pt）；b0≫c2≥c1 排序稳健 | 3/3 重划 | W7/e4c_summary.md |
| D5 | 标签效率曲线单调 FT10<FT20<FT40（全方法）；C2-FT10 与全量监督差距 −1.8pt 但省 10 倍标签 | 全方法 | W6 Stage1 + W4 终表 |
| D6 | CPSC 多标签同向：b0 0.8586>c2 0.8560>c1 0.8322（one-hot 真值限制如实标注） | — | W6 Stage7 |
| D7 | 5-seed 稳定性：cpsc b0 0.9506±0.0005 / c2 0.9473±0.0030 / c1 0.9462±0.0012 | — | W6/seed5_summary.csv |

## E. 边缘读数（如实列示，不入正表）

- cpsc C2−C1 AUROC +0.11pt（p=0.187，非显著；s0 打平）——AUPRC 口径已在 A2。
- ptbxl C2−C1 LP +0.21pt（符号 −+/+ 非一致）。
- W2 时代 cpsc C2−C1 +2.30pt（AUPRC）——**污染数据，永久禁引**，仅历史脚注。
- E4b 质量门控 / E2a 纯适配器 / LCM / D1L / TAPT(方法级) / CCM 心拍线——判负关线，无正向可报（负果入失败方向叙事）。

## 第三域"廉价增益"排序（v2 口径汇总）

TRC（+0.83 AUPRC cpsc 3/3；+0.31 chapman 3/3 p=0；E2c 单种子 +0.60）＞ 双视角+RR头（+0.233 cpsc，确认级）＞ TAPT（+0.10 cpsc，破无伤）＞ 0。
**结构先验在预训练期联合优化是唯一划算的第三域增益来源**——论文核心论点，三条独立证据（A2/B1/E2a 负果反证）。

## 论文正向叙事骨架（一章一线索）

1. 第2章 基座：A5（B0 域内）+ A4（NFH 抗退化）——两个预训练语料各有所长。
2. 第3章 TRC：A2 + B1 + D2 + E2a 判负反证——增益定位"跨域转移场景"，机制=能量尺度重标定。
3. 心拍线（研究内容2）：A3 + D3——方法提出+系统检验+确认级正效应与边界。
4. 应用章：D1（LP-FT 推荐协议）+ D5（标签效率）+ D6/D7（稳定性）。
5. 防御工事：A1（外部基线全不反超）+ A4 附带（TRC 不伤鲁棒性）+ E 档如实披露。

（待入：E5 噪声注入判定 ~18:45 出账后按门自动归档 A/C/E 之一。）
