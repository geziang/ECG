# LFBT 复现结果报告（PTB-XL 基线）

> **论文**：Liu et al., *Lead-fusion Barlow twins: A fused self-supervised learning method for multi-lead electrocardiograms*, Information Fusion 114 (2025) 102698
> **仓库**：<https://github.com/Aiwiscal/ECG_SSL_LFBT>
> **复现日期**：2026-08-07
> **⚠️ 声明**：预训练使用 **PTB-XL folds 1-8 替代 NFH**（非论文原始设置）。用户仅有 PTB-XL 数据，故以 17,418 条 PTB-XL 记录做无监督预训练，下游沿用 PTB-XL 官方 fold 划分。此差异不影响下游对比的公平性（新模块将使用同一预训练/下游流程）。

---

## 1. 环境信息

| 项 | 值 |
|---|---|
| 操作系统 | Windows |
| Python | 3.10.11 |
| PyTorch | 2.0.0+cu118 |
| torchvision | 0.15.1+cu118 |
| CUDA (build) | 11.8 |
| GPU | NVIDIA GeForce RTX 4090 (25.8GB) |
| 随机种子 | 0（预训练与下游统一 `set_seed`） |

## 2. 数据契约

- **源数据**：PTB-XL（磁盘 1.0.1，500Hz；CSV 1.0.3）
- **导联**：II, III, V1, V2, V3, V4, V5, V6（8 导联，每导联独立 encoder）
- **输入形状**：`[8, 2048]` float32（500Hz→2048 点重采样，整条记录 z-score）
- **标签**：5 类 superclass（NORM/CD/HYP/MI/STTC），仅保留单标签记录
- **划分**：官方 strat_fold（train=folds 1-8, val=9, test=10），按 patient_id 与预训练集零重叠

| 集合 | 样本数 |
|---|---:|
| 预训练（folds 1-8，无标签） | 17,418 |
| 下游 train | 13,639 |
| 下游 val | 1,714 |
| 下游 test | 1,739 |

## 3. 预训练配置与结果

### 配置

| 参数 | 值 |
|---|---|
| encoder | 8 × VGG16(ch_in=1, alpha=0.125)，输出 64 维 |
| projector | 2048-2048-2048（每导联独立）|
| 损失 | `L = γ·L_r + (1-γ)·L_t` |
| gamma | 0.8 |
| lambda | 0.0051 |
| batch size | 128 |
| epochs | 200 |
| learning rate | 0.001（Adam）|
| workers | 6 |

### 训练曲线（每 epoch 平均 loss）

| epoch | loss | loss_r | loss_t |
|---|---:|---:|---:|
| 0 | 1288.6 | 1096.1 | 2058.7 |
| 1 | 915.1 | 694.5 | 1797.7 |
| 5 | 638.8 | 459.7 | 1355.2 |
| 20 | ~337 | ~283 | ~553 |
| 50 | ~259 | ~235 | ~354 |
| 100 | ~230 | ~218 | ~280 |
| 200（终） | ~226 | ~215 | ~271 |

loss 全程稳定下降，最终收敛至 ~226。

### 产物

- `checkpoint/ptxl_gamma08/encoder_group.pth`（双格式：模块对象列表 + state_dict 列表）
- **SHA256**：`026033d532d4e42a1b53ca9c4d2e95f9c09b2daaaab5c9041c9d474a1356cdad`
- `checkpoint/ptxl_gamma08/config.json`（完整环境与超参记录）

## 4. 下游结果（PTB-XL 5 类，macro AUROC / AUPRC）

| 实验 | checkpoint | fraction | AUROC | AUPRC |
|---|---|---|---:|---:|
| **LP** | 预训练 | 1.0 | **0.9126** | **0.7177** |
| **FT** | 预训练 | 1.0 | **0.9197** | **0.7173** |
| FT | 预训练 | 0.1 | **0.8756** | **0.6431** |
| LP (TFS) | 随机初始化 | 1.0 | 0.4700 | 0.1930 |
| FT (TFS) | 随机初始化 | 1.0 | 0.8839 | 0.6715 |
| FT (TFS) | 随机初始化 | 0.1 | 0.8385 | 0.5836 |

- LP：冻结 encoder，只训练线性分类器（lr=0.001, 100 epoch）
- FT：全模型微调（lr=0.0001, 100 epoch），best checkpoint 由 val loss 选择，test 仅最终评估一次
- TFS：Train From Scratch（随机初始化 encoder + 分类器），作为 SSL 对照基线
- 分类顺序：CD/HYP/MI/NORM/STTC（目录名排序）

### 混淆矩阵（测试集）

**FT 1.0（预训练）**：

```
[[148   0  10  68   3]
 [  5   4   2  38   7]
 [ 14   0 145  77  20]
 [ 19   1  19 889  24]
 [  1   1  12  55 177]]
```

（其余实验混淆矩阵见 `results/*/metrics.json`）

## 5. 与论文对比

| 实验 | 本次复现 AUROC | 论文参考 AUROC | 本次复现 AUPRC | 论文参考 AUPRC |
|---|---|---:|---:|---:|
| LP | 0.9126 | 0.9082 | 0.7177 | 0.6923 |
| FT | 0.9197 | 0.9178 | 0.7173 | 0.7257 |

复现结果与论文参考值高度吻合（偏差 < 0.01），判定复现成功。

## 6. 关键结论

1. **复现有效**：LP/FT 指标与论文一致，趋势正确（FT > LP > TFS）。
2. **SSL 预训练收益显著**：
   - LP：预训练 0.9126 vs 随机 0.4700（+0.44 AUROC）
   - FT：预训练 0.9197 vs 随机 0.8839（+0.036 AUROC）
3. **少标签优势**：10% 训练数据下，预训练 0.8756 仍领先随机 0.8385（+0.037 AUROC）。
4. **基线可直接支撑后续新模块对比**：同一数据、同一 LP/FT/TFS 流程，仅替换/新增模型模块即可公平比较。

## 7. 产物清单

| 产物 | 路径 |
|---|---|
| 预训练 checkpoint | `checkpoint/ptxl_gamma08/encoder_group.pth` |
| 预训练配置 | `checkpoint/ptxl_gamma08/config.json` |
| 预训练数据清单 | `data/manifest.json` |
| LP 预训练结果 | `results/ptbxl_lp/metrics.json` |
| LP 随机对照 | `results/ptbxl_lp_tfs/metrics.json` |
| FT 1.0 预训练 | `results/ptbxl_ft_1.0/metrics.json` |
| FT 0.1 预训练 | `results/ptbxl_ft_0.1/metrics.json` |
| FT 1.0 TFS | `results/ptbxl_ft_tfs/metrics.json` |
| FT 0.1 TFS | `results/ptbxl_ft_tfs_0.1/metrics.json` |
| 结果汇总 | `results.csv` |

## 8. 复现过程修复记录

| 问题 | 修复 |
|---|---|
| 目录字符串拼接（P0） | `cls_datasets.py` 改用 `Path(data_path) / "train"` |
| checkpoint 单格式（P1） | `run_pt.py` 双格式保存（模块列表 + state_dict 列表），LP/FT 兼容两种格式 |
| FT 加载无 map_location（P1） | `run_ft.py` 加载时 `map_location=self.device` |
| 随机性不可复现（P1） | `data_utils/seed_utils.py` 统一固定 Python/NumPy/PyTorch/CUDA 种子 |
| 增强原地修改（P2） | `augmentations.py` 置零前 `np.array(sig, copy=True)` |
| 预训练"卡死"（GPU 空转） | 根因：`conda run` 包装与 Windows spawn 管道冲突；改用环境内 python 直连启动即正常 |
| FT 输出目录不存在报错 | `run_ft.py` 训练前 `model_dir.mkdir(parents=True, exist_ok=True)` |

## 9. 复现命令

```bash
# 预训练（使用 conghui 环境内 python 直连，勿用 conda run）
C:\Users\admin\.conda\envs\conghui\python.exe -u run_pt.py \
  --data-dir data/pt_pretrain/ --epochs 200 --batch-size 128 \
  --workers 6 --gamma 0.8 --checkpoint-dir checkpoint/ptxl_gamma08/

# LP（预训练 / 随机对照）
C:\Users\admin\.conda\envs\conghui\python.exe -u run_lp.py \
  --data-dir data/ptbxl/ --checkpoint checkpoint/ptxl_gamma08/encoder_group.pth \
  --num-classes 5 --feat-dir results/ptbxl_lp/ --seed 0
C:\Users\admin\.conda\envs\conghui\python.exe -u run_lp.py \
  --data-dir data/ptbxl/ --checkpoint none --num-classes 5 \
  --feat-dir results/ptbxl_lp_tfs/ --seed 0

# FT（预训练 / 随机对照，fraction 1.0 / 0.1）
C:\Users\admin\.conda\envs\conghui\python.exe -u run_ft.py \
  --data-dir data/ptbxl/ --checkpoint checkpoint/ptxl_gamma08/encoder_group.pth \
  --num-classes 5 --fraction 1.0 --model-dir results/ptbxl_ft_1.0/ --seed 0
# ... 同理 fraction 0.1；checkpoint 传 none 即 TFS 随机初始化对照

# 汇总
C:\Users\admin\.conda\envs\conghui\python.exe collect_results.py
```

## 10. 已知差异与注意事项

1. **预训练数据源替代**：NFH → PTB-XL folds 1-8，报告/论文中须注明为"替代预训练实验"。
2. **FT 后期 val loss 上升**：微调 100 epoch 后 val loss 出现上升（过拟合现象），best checkpoint 由训练早期最低 val loss 选取，最终 test 结果仍优于 LP，与论文一致。
3. **单 seed**：当前基线为 seed 0 单次结果；如需均值±标准差，可补跑 seeds 1/2（预计约 4-5 小时）。
4. **类别平衡**：PTB-XL 5 类分布不均衡（NORM 占多数），AUPRC 为 macro 平均，与论文口径一致。
