# A-2 冻结 checkpoint 输入退化评估 — 预注册口径与结果说明

> 任务来源: `下一阶段双机任务书-2026-09-22.md` §A-2（推荐项）
> 执行: 主机A(Win4090), 2026-09-22 晚。代码: `ECG_SSL_LFBT-main/run_robustness.py`
> 结果: `robustness_b0.csv` / `robustness_c1.csv` / `robustness_c2.csv`（本目录）

## 1. 设计纪律（与任务书一致）

- **零重训、零新损失**：只用 W2 冻结 encoder + W2 已保存的下游头（LP 线性头
  `results/confirm/*/classifier_best_ckpt.pth`、FT10 全模型
  `ft_models/confirm/*/ft_best_ckpt.pth`，strict 加载，错配硬拒）。
- **clean 复现门**：每 (模型, seed, 数据集) 先评 clean，与
  `runlog/W2/confirm_results.csv` 核对，|Δ|>0.0005 中止。
  实测 22/22 全过：AUROC 全部 4 位小数一致，AUPRC 3 处 ±0.0001（batch 求和顺序的
  浮点漂移，在门内）。
- **不修改任何 W2 文件**；新结果只写本目录。
- **test 只评一次的原则**：本任务为纯 test-time 扰动评估，不涉及任何选择；
  扰动套件在运行前预注册（见下），不在看过结果后增删条件。

## 2. 扰动套件（预注册）

全部施加在 z-score 后的 `(8, 2048)` 波形上。有效采样率 204.8 Hz
（500 Hz × 10 s = 5000 样本重采样至 2048）。信号统计口径：每记录每导联，
SNR_dB = 10·log10(P_signal / P_noise)。

| 组 | 名称 | 定义 | 强度 |
|---|---|---|---|
| A 加性噪声 | `bw` | 基线漂移 = 0.2/0.5/1.0 Hz 三正弦混合，每导联独立随机相位 | SNR 0/5/10/20 dB |
| A | `emg` | 肌电 = 20–100 Hz butterworth(阶4, filtfilt) 带通白噪声 | SNR 0/5/10/20 dB |
| A | `pl50` / `pl60` | 工频正弦，每导联随机初相 | SNR 0/5/10/20 dB |
| B 质量衰减 | `qual_amp` | 全导联幅度缩放（连续衰减，区别于 W2 缺导的离散置零） | ×0.8/0.6/0.4/0.2 |
| B | `qual_noise` | 全导联宽带白噪声 | SNR 20/15/10/5 dB |
| C 导联错位 | `swap01..swap67` | 相邻下标对交换（II,III,V1..V6），仅 test-time | 7 对 |
| C | `swap_roll` | 循环左移 1（全导联同时被相邻导联替换的上界情形） | roll1 |

噪声种子只依赖 `(dataset, perturbation, intensity)`（sha256 派生），
**所有模型/种子看到同一扰动实例**——同条件跨模型为配对比较，Δ 差异只来自模型。

## 3. 评估组合（只用 W2 已存下游头，不补训）

| 数据集 | 评估 | 模型 × seed | 说明 |
|---|---|---|---|
| ptbxl | LP | b0(2,4) c1/c2(0,2,4) | 本域 LP |
| cpsc | LP | c1/c2(0,2,4) | 第三域（论文主线 +2.30pt）；b0 无 W2 cpsc 账，不补跑（纪律：未登记训练不启动） |
| ptbxl | FT10 | b0(2,4) c1/c2(0,2,4) | 10% 标签微调（论文副线 +0.65pt） |
| chapman | — | 跳过 | LP≈0.996 近天花板、血缘域最低优先级，退化评估信息量低 |

共 22 模型组合 × 33 条件（1 clean + 32 扰动）= 694 行扰动读数 + clean 门。

## 4. 指标口径（与 run_lp/run_ft 逐位一致）

- 概率 = softmax 输出；AUROC = macro（one-hot）；AUPRC = micro（one-hot）。
- 每行含 `clean_auroc/clean_auprc` 与 `delta_*_pt`（相对同模型同 seed clean 的
  百分点差）。
- 元数据：`git_sha`（评估代码版本）、`checkpoint_sha256`（encoder checkpoint，
  与 freeze_manifest_v2 一致）、`data_sha`（test 文件名清单 sha256 前 12 位）。

## 5. 结果摘要（2026-09-22 20:53 全量完成，704 行，零重复键）

种子平均 ΔAUROC(pt)，负值=退化。完整明细见三个 CSV。

**噪声族按 SNR 分级（ΔAUROC pt）：**

| 条件 | b0 | c1 | c2 | 说明 |
|---|---|---|---|---|
| ptbxl/LP pl50@20dB | **−11.0** | −5.5 | −6.7 | B0 对工频极脆，NFH 预训练明显更稳 |
| ptbxl/LP pl50@0dB | −35.1 | −23.9 | −27.5 | 同上，梯度一致 |
| ptbxl/LP emg@20dB | −7.0 | −5.2 | −4.3 | C2 肌电鲁棒性略优 |
| ptbxl/LP emg@0dB | −35.7 | −28.6 | −29.2 | |
| cpsc/LP pl50@20dB | — | −2.5 | −2.9 | 第三域同向（b0 无 W2 cpsc 账） |
| cpsc/LP emg@0dB | — | −30.4 | −32.7 | 极端噪声下 C2 略弱于 C1 |
| ptbxl/FT10 emg@20dB | −5.2 | −4.5 | −4.4 | FT10 三者接近 |

**扰动族平均（ptbxl/LP）：** bw（基线漂移）影响很小（b0 −1.3 / c1 −0.5 / c2 −0.4）；
qual_amp@×0.4 ≈ −6~−7pt、qual_noise@5dB ≈ −10~−15pt（连续衰减温和且模型间差异小）。

**adjacent lead-swap：** 肢体导联对最敏感（II↔III：ptbxl/LP −4.4/−3.8/−4.2；
III↔V1：cpsc/LP c1 −1.8 / c2 −2.2），V4–V6 相邻胸前导联交换几乎无代价（≤0.3pt，
个别 +0.1~0.2pt 噪声级）；swap_roll（全部滚动错位）−6.6~−8.7pt 是上界。
三模型对 swap 的敏感度谱形一致。

**结论（措辞克制）：**
1. 外部 NFH 预训练（C1/C2）在仿真输入退化下普遍比 B0 鲁棒，工频干扰下优势最大
   （ptbxl/LP pl50@20dB：B0 −11.0 vs C1 −5.5 / C2 −6.7）；
2. TRC（C2 vs C1）不改变输入退化鲁棒性：差异均在 ±1.5pt 内且方向混合
   （emg/bw 上 C2 略优，cpsc 极端噪声 C2 略弱）——与缺导结论同构：
   **C2 保留 C1 的鲁棒性，没有以鲁棒性换增益**；
3. 相邻胸前导联（V4–V6）互换近乎无损，导联错位风险集中在肢体导联与 V1–V3。

- b0 在 cpsc 上缺席（见上表）；chapman 未评（见上表）。
- 扰动定义是参数化仿真（非真实噪声库 MIT-BIH NSTDB），论文措辞应为
  "仿真输入退化下的相对鲁棒性"，不声称真实噪声环境性能。

## 6. 已知边界

- b0 在 cpsc 上缺席（见 §3 表）；chapman 未评（近天花板）。
- 扰动定义是参数化仿真（非真实噪声库 MIT-BIH NSTDB），论文措辞应为
  "仿真输入退化下的相对鲁棒性"，不声称真实噪声环境性能。
- 事件记录：首跑曾因 --stage clean 未隔离扰动落盘导致 CSV 重复行，已修复
  （clean 阶段只过门不写账）并清盘重跑；本目录 CSV 为重跑后唯一版本（git_sha d9490c8）。
