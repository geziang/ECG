# TRC–Beat-RR–Noise 联合验证方案

## 1. 实验目的

本轮实验不重新搜索方向，而是回答三个明确问题：

1. TRC 与 Beat-RR 是否具有互补性，联合模型能否稳定超过两个单独分支？
2. Beat 与 RR 两个组成部分的互补方向能否在 3 个随机种子中复现？
3. 在联合模型上加入 NSTDB 真实噪声视图后，能否提高噪声鲁棒性，同时保持无噪声测试性能？

论文当前以 TRC 为主线。只有联合实验满足预先定义的判定标准，才将 LFBT+TRC+Beat-RR 升级为后续“大论文”的统一核心模型；在此之前，不修改现有独立分支结论。

## 2. 固定协议

除模型开关外，以下设置全部锁定，不因中途结果调整：

- 预训练语料：NFH，`data/pt_pretrain_nfh`。
- 预训练轮数：100 epoch。
- 优化器：Adam；学习率 0.001；batch size 128。
- 主损失：LFBT，`loss-mode=bt`，`lambda=0.0051`，`gamma` 沿用现有 LFBT 设置。
- 主随机种子：`{0,2,4}`。
- TRC：`--trc 1`；关闭时为 `--trc 0`。
- 心拍形态约束：`--dualview-weight 0.01`。
- RR 回归：`--rr-weight 0.1`。
- R 峰缓存：`data/pt_rpeaks_nfh.npz`。
- 噪声增强：`--nstdb-aug 0.5 --nstdb-offset first_half`。
- 噪声测试：只使用 NSTDB 后半段，保持现有 W8 E6 的时间隔离口径。
- 数据划分、单标签映射、预处理、类别数和验证集选模规则均与现有实验完全一致。
- 主要下游任务：CPSC 线性评估。
- 主要指标：CPSC macro-AUPRC；macro-AUROC 为共同报告的次要指标。
- 次要任务：PTB-XL、Chapman 线性评估；PTB-XL FT10。

禁止用测试集选择损失权重、训练轮数或模型组合。若发生代码错误，只允许修复实现，不允许依据结果改动协议。

## 3. 模型矩阵

### 3.1. 已有锚点

| 编号 | 模型 | TRC | Beat | RR | Noise | 状态 |
|---|---|---:|---:|---:|---:|---|
| M0 | LFBT | 0 | 0 | 0 | 0 | 已完成 3 种子 |
| M1 | LFBT+TRC | 1 | 0 | 0 | 0 | 已完成 3 种子 |
| M2 | LFBT+Beat-RR | 0 | 0.01 | 0.1 | 0 | 已完成 3 种子 |
| M5 | LFBT+Noise | 0 | 0 | 0 | 0.5 | 已完成 3 种子 |

### 3.2. 本轮必须新增

| 编号 | 模型 | TRC | Beat | RR | Noise | 新增训练量 |
|---|---|---:|---:|---:|---:|---:|
| M3 | LFBT+TRC+Beat-RR | 1 | 0.01 | 0.1 | 0 | 3 个种子 |
| M4 | LFBT+TRC+Beat-RR+Noise | 1 | 0.01 | 0.1 | 0.5 | 3 个种子 |
| A1 | LFBT+Beat only | 0 | 0.01 | 0 | 0 | 补 s2、s4 |
| A2 | LFBT+RR only | 0 | 0 | 0.1 | 0 | 补 s2、s4 |

最低新增预训练量为 10 次：M3 三次、M4 三次、A1 两次、A2 两次。

### 3.3. 算力允许时新增

| 编号 | 模型 | 目的 | 新增训练量 |
|---|---|---|---:|
| M6 | LFBT+TRC+Noise | 判断噪声增益是否依赖 Beat-RR | 3 个种子 |
| N1 | TRC+Beat-RR，随机心拍窗口 | 反事实验证 R 峰对齐是否必要 | 先 s0，必要时补 3 种子 |
| N2 | TRC+Beat-RR，打乱 RR 标签 | 反事实验证节律目标是否提供有效信息 | 先 s0，必要时补 3 种子 |

M6 是推荐项，但不应阻塞 M3/M4。N1/N2 用于机制加强，不能替代正式组件消融。

## 4. 联合模型实现口径

### 4.1. 联合目标

M3/M4 使用：

$$
\mathcal L_{joint}=\mathcal L_{BT}^{r+t}+0.01\mathcal L_{dv}+0.1\mathcal L_{rr}.
$$

TRC 只改变每个导联编码器在全局平均池化前的特征变换，不增加额外损失。Noise 只改变两个整段自监督视图的构造，不增加额外损失。

### 4.2. R 峰与噪声的处理顺序

1. 在无噪声原始记录上离线检测并缓存 R 峰。
2. 从无噪声记录生成 R 峰对齐平均心拍和 RR 统计目标。
3. 对两个整段自监督视图分别执行 RRC-TO。
4. M4 再对两个整段视图独立注入 NSTDB 前半段噪声。
5. 整段视图、平均心拍视图均通过同一组带 TRC 的逐导联编码器。
6. RR 目标的 z-score 统计量只能由预训练训练集计算。

不得在加噪后的信号上重新检测 R 峰，否则噪声会同时改变辅助标签和有效样本掩码，破坏实验解释。

### 4.3. 现有代码开关

仓库：`ECG/ECG_SSL_LFBT-main`。

M3 单个随机种子的预训练命令模板：

```bash
python -u run_pt.py \
  --data-dir data/pt_pretrain_nfh \
  --epochs 100 --batch-size 128 --workers 4 \
  --seed SEED --trc 1 \
  --dualview-weight 0.01 --rr-weight 0.1 \
  --rpeak-npz data/pt_rpeaks_nfh.npz \
  --resume --checkpoint-dir checkpoint/joint_m3/seedSEED
```

M4 单个随机种子的预训练命令模板：

```bash
python -u run_pt.py \
  --data-dir data/pt_pretrain_nfh \
  --epochs 100 --batch-size 128 --workers 4 \
  --seed SEED --trc 1 \
  --dualview-weight 0.01 --rr-weight 0.1 \
  --rpeak-npz data/pt_rpeaks_nfh.npz \
  --nstdb-aug 0.5 --nstdb-offset first_half \
  --resume --checkpoint-dir checkpoint/joint_m4_noise/seedSEED
```

将 `SEED` 依次替换为 0、2、4。A1/A2 采用相同命令，只关闭 TRC，并分别将另一项辅助损失权重设为 0。

正式批量运行前必须进行 1 个 epoch 的 smoke test，确认以下组合可以同时工作：`trc=1`、`dualview_weight=0.01`、`rr_weight=0.1`、`nstdb_aug=0.5`。现有参数开关虽然已分别实现，但联合路径仍需测试数据批次结构、有效样本掩码、断点续训及 checkpoint 加载。

## 5. 下游评估矩阵

### 5.1. 所有新增 checkpoint

每个随机种子均运行：

- CPSC LP，9 类；保存逐记录预测。
- PTB-XL LP，5 类；保存逐记录预测。
- Chapman LP，4 类；保存逐记录预测。
- PTB-XL FT10；保存逐记录预测。

LP 必须与预训练 checkpoint 使用相同的 `--trc` 设置。M3/M4 下游加载时使用 `--trc 1`。

### 5.2. M4 噪声评估

在与 LFBT、LFBT+Noise 相同的测试噪声实现上运行：

- 数据集：PTB-XL、CPSC、Chapman。
- 噪声类型：bw、ma、em。
- SNR：0、5、10 dB。
- 共 27 个条件。
- 噪声片段：NSTDB 后半段。
- 同时评价无噪声测试集。

必须使用“同随机种子、同记录、同噪声片段”的配对差值，不能将不同噪声实现的绝对均值直接相减。

## 6. 统计分析

### 6.1. 主要比较

1. M3−M1：Beat-RR 是否在 TRC 上继续增加收益。
2. M3−M2：TRC 是否在 Beat-RR 上继续增加收益。
3. M3−M0：联合模型相对 LFBT 的总增益。
4. M4−M3：Noise 对联合模型的无噪声性能影响。
5. M4−M3：Noise 对 27 个噪声测试条件的增益。

### 6.2. 交互效应

以 CPSC macro-AUPRC 为主，计算：

$$
I_{TRC\times BeatRR}=(M3-M1)-(M2-M0).
$$

- $I>0$：观察到超加性互补。
- $I\approx0$：两者近似相加或彼此独立。
- $I<0$：存在收益重叠或优化冲突。

交互效应必须给出逐随机种子值和置信区间，不只报告均值。

### 6.3. 检验方法

- 逐记录配对 bootstrap 10,000 次，多随机种子同步重采样。
- AUROC 同时报告 DeLong 检验。
- 报告均值、标准差、95% CI、逐随机种子方向和绝对效应量。
- CPSC 同时报告单标签主分析与多标签敏感性分析。
- 类别级结果至少报告 AF、PAC、PVC、STE、STD，并将“节律类”和“形态类”汇总作为探索性分析；若做多重比较，应进行校正。

## 7. 预先定义的论文决策规则

### 7.1. 升级为统一核心模型

满足以下条件时，将 M3 写为大论文核心模型：

- M3 的 CPSC macro-AUPRC 高于 M1 和 M2 的最佳单分支结果；
- 3 个随机种子中至少 2 个同向，且总体配对置信区间不跨 0；
- PTB-XL、Chapman 无系统性明显退化；
- 联合模型的训练稳定性和有效心拍样本比例正常。

若 M3 仅在 AUROC 或单个随机种子上改善，不写“协同增益”，只写为探索性联合结果。

### 7.2. 保持 TRC 单主线

出现以下任一情况时，仍以 TRC 为唯一核心方法：

- M3 不超过 M1/M2 中的最佳单分支；
- 联合增益方向不稳定；
- 联合训练导致 PTB-XL 或 Chapman 明显退化；
- Beat/RR 的 3 种子组件拆分不能复现互补方向。

此时 Beat-RR 保留为独立探索，Noise 保留为鲁棒性扩展，不强行组成统一模型。

### 7.3. 噪声版本成立条件

M4 只有在以下条件同时满足时，才写成联合模型的鲁棒版本：

- 27 个噪声条件的平均 AUROC 相对 M3 为正，且3个随机种子方向一致；
- 无噪声测试性能的平均变化接近持平；
- 不隐瞒任何数据集或随机种子上的下降。

原有 −0.30 个百分点门槛仅作为描述性安全线，不替代置信区间和效应量。

## 8. 运行前检查

- 检查 NFH、CPSC、PTB-XL、Chapman 数据清单哈希未变化。
- 检查 CPSC 使用修正后的患者级分区，test=1385。
- 检查 `data/pt_rpeaks_nfh.npz` 与 NFH 样本 ID 一一对应。
- 统计 R 峰少于 3 个的记录比例，并按数据集保存。
- 检查 RR z-score 的均值和标准差只由预训练训练集计算。
- 检查两个噪声视图的随机采样彼此独立。
- 检查训练噪声 offset 为 `first_half`，测试为 `last_half`。
- 检查 checkpoint 保存 TRC、RR head、优化器、epoch 和 RNG 状态。
- 检查 resume 后损失和学习率连续。
- 检查联合模式下 LP/FT 能正确加载带 TRC 的编码器权重。

## 9. 必须保存的产物

每个配置、随机种子至少保存：

- `config.json`：完整超参数和数据路径。
- `encoder_group.pth`、训练状态及 checkpoint SHA256。
- `git_sha`、Python/PyTorch/CUDA 版本、GPU 型号。
- 每 epoch 的总损失、BT损失、Beat损失、RR损失。
- 训练时间、峰值显存和推理时间。
- 下游 `metrics.json`。
- 逐记录预测：sample_id、真实标签、预测概率、随机种子、checkpoint hash。
- 27 个噪声条件的逐记录预测和噪声片段标识。
- 有效 R 峰样本比例、RR 目标分布及异常记录清单。

建议统一目录：

```text
runlog/JOINT_20261013/
  protocol.md
  results.csv
  paired_stats.csv
  interaction_stats.csv
  checkpoints_manifest.json
  predictions/
  logs/
```

## 10. 回填论文所需最终表格

服务器完成后至少返回以下三张表：

1. `M0/M1/M2/M3` × 三个目标数据集的 LP AUROC/AUPRC，3 种子均值与标准差。
2. `M0/A1/A2/M2/M1/M3` 的 CPSC 组件消融及逐随机种子差值。
3. `M3/M4` 在无噪声测试和 27 个噪声条件下的配对差值。

同时返回 M3 相对 M1、M2、M0 的逐记录统计，以及 $I_{TRC\times BeatRR}$ 的逐随机种子结果。论文是否升级为统一模型，只依据本方案预先定义的比较和判定规则决定。
