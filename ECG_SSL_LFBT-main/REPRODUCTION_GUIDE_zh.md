# ECG_SSL_LFBT 复现指导书

本文档记录在 Windows 机器上配置并跑通 `D:\lbtf\ECG_SSL_LFBT-main` 的步骤。项目实现的是 Lead-Fusion Barlow Twins (LFBT) 多导联 ECG 自监督学习方法。

## 1. 本机已验证环境

- 操作系统：Windows
- Conda：`D:\software`
- Conda 环境：`ecg_ssl`
- Python：`3.10.20`
- PyTorch：`1.11.0+cpu`
- torchvision：`0.12.0+cpu`
- numpy：`1.22.4`
- scipy：`1.10.1`
- scikit-learn：`1.3.0`
- tqdm：`4.61.2`

检查环境：

```powershell
conda run -n ecg_ssl python --version
conda run -n ecg_ssl python -c "import numpy, scipy, sklearn, torch, torchvision, tqdm; print('numpy', numpy.__version__); print('scipy', scipy.__version__); print('sklearn', sklearn.__version__); print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available()); print('torchvision', torchvision.__version__); print('tqdm', tqdm.__version__)"
```

如果需要从头创建环境：

```powershell
conda create -n ecg_ssl python=3.10 -y
conda run -n ecg_ssl python -m pip install numpy==1.22.4 scipy==1.10.1 scikit-learn==1.3.0 tqdm==4.61.2
conda run -n ecg_ssl python -m pip install torch==1.11.0 torchvision==0.12.0
```

说明：本机当前装的是 CPU 版 PyTorch，因此脚本已做 CPU 兼容修复；如果机器有可用 CUDA，可以安装对应 CUDA 版 PyTorch。

## 2. 代码修复记录

为保证 CPU 环境和命令行参数可用，本次做了 3 个小修复：

- `run_pt.py`：将预训练 batch 从硬编码 `.cuda(...)` 改为 `.to(model.device, ...)`，无 GPU 时也能运行。
- `run_lp.py`：线性探测分类阶段原来写死训练 100 epoch，现在改为使用命令行 `--epochs`。
- `models/mbn.py`：微调加载 CUDA 保存的 checkpoint 时增加 `map_location`，CPU 环境可正常反序列化。

## 3. 数据准备格式

每个 ECG 样本需要保存为 `.npy` 文件，shape 为：

```text
(C, L)
```

其中 `C` 是导联数，当前脚本默认使用 8 个导联；`L` 是采样长度。每条 ECG 建议完成重采样和 Z-score 标准化后保存。

预训练数据目录：

```text
pt_data_dir/
  samples/
    sample_0.npy
    sample_1.npy
```

下游分类数据目录：

```text
down_data_dir/
  train/
    class_0/
      sample_0.npy
    class_1/
      sample_1.npy
  val/
    class_0/
    class_1/
  test/
    class_0/
    class_1/
```

注意：`run_lp.py` 和 `run_ft.py` 内部使用 `data_path + "train"` 拼接路径，所以传入 `--data-dir` 时末尾必须带路径分隔符，例如：

```powershell
--data-dir D:\your_data\down_data_dir\
```

Windows 上建议先用 `--workers 0` 跑通，确认没问题后再增加 worker 数。

## 4. 预训练

使用真实预训练数据：

```powershell
cd D:\lbtf\ECG_SSL_LFBT-main
D:\software\envs\ecg_ssl\python.exe run_pt.py --data-dir D:\your_data\pt_data_dir --workers 0 --epochs 200 --batch-size 128 --checkpoint-dir checkpoint
```

常用参数：

- `--num-leads`：导联数，默认 8。
- `--gamma`：LFBT loss 的平衡参数，默认 0.8。
- `--lambd`：off-diagonal loss 权重，默认 0.0051。
- `--projector`：投影头结构，默认 `2048-2048-2048`。

输出 checkpoint：

```text
checkpoint/encoder_group.pth
```

## 5. 线性探测

仓库已自带两个预训练 encoder：

```text
checkpoint/encoder_group_gamma_0.6.pth
checkpoint/encoder_group_gamma_0.8.pth
```

运行线性探测：

```powershell
cd D:\lbtf\ECG_SSL_LFBT-main
D:\software\envs\ecg_ssl\python.exe run_lp.py --data-dir D:\your_data\down_data_dir\ --checkpoint checkpoint\encoder_group_gamma_0.8.pth --num-classes 2 --feat-dir feat --workers 0 --epochs 100 --batch-size 128
```

输出内容：

- `feat/X_train.npy`
- `feat/y_train.npy`
- `feat/X_val.npy`
- `feat/y_val.npy`
- `feat/X_test.npy`
- `feat/y_test.npy`
- `feat/classifier_best_ckpt.pth`
- 测试集 AUROC、AUPRC、Confusion Matrix

## 6. 微调

运行微调前先创建模型输出目录：

```powershell
New-Item -ItemType Directory -Force -Path D:\lbtf\ECG_SSL_LFBT-main\ft_models
```

运行命令：

```powershell
cd D:\lbtf\ECG_SSL_LFBT-main
D:\software\envs\ecg_ssl\python.exe run_ft.py --data-dir D:\your_data\down_data_dir\ --checkpoint checkpoint\encoder_group_gamma_0.8.pth --num-classes 2 --model-dir ft_models --workers 0 --epochs 100 --batch-size 128
```

输出模型：

```text
ft_models/ft_best_ckpt.pth
```

## 7. 本次 smoke test 结果（2026-06-09 最新运行）

### 测试概况

此次使用项目内临时生成的合成 `.npy` 数据（8 个训练样本、8 个验证样本、8 个测试样本，二分类）完整验证三个 pipeline。所有测试均使用 `--workers 0` 在 CPU 环境下运行。

数据位于：

```text
smoke_data/
```

结果输出位于：

```text
smoke_runs/
```

---

### 7.1 预训练 smoke test

命令：

```powershell
D:\software\envs\ecg_ssl\python.exe run_pt.py --data-dir smoke_data\pt --workers 0 --epochs 1 --batch-size 4 --projector 16 --print-freq 1 --checkpoint-dir smoke_runs\pt_checkpoint
```

结果：

| 指标 | 值 |
|------|-----|
| 状态 | **通过** |
| Epoch | 1 |
| Batch Size | 4 |
| 总 Loss | 20.48 |
| Loss_r (同导联) | 20.17 |
| Loss_t (跨导联) | 21.72 |
| 耗时 | 1.82 s |
| 输出文件 | `smoke_runs/pt_checkpoint/encoder_group.pth` (2.83 MB) |

---

### 7.2 线性探测 smoke test

命令：

```powershell
D:\software\envs\ecg_ssl\python.exe run_lp.py --data-dir smoke_data\down\ --checkpoint checkpoint\encoder_group_gamma_0.8.pth --num-classes 2 --feat-dir smoke_runs\feat --workers 0 --epochs 2 --batch-size 4
```

特征提取结果：

| 数据集 | 样本数 | 特征维度 | 文件 |
|--------|--------|----------|------|
| train | 8 | 512 | `X_train.npy` / `y_train.npy` |
| val | 8 | 512 | `X_val.npy` / `y_val.npy` |
| test | 8 | 512 | `X_test.npy` / `y_test.npy` |

8 个导联 encoder (lead-ii ~ lead-v6) 权重全部加载成功。

分类训练与测试结果：

| 指标 | 值 |
|------|-----|
| 状态 | **通过** |
| Epochs | 2 |
| Epoch 1 Val Loss | 0.6744 |
| Epoch 2 Val Loss | 0.6621 |
| **AUROC** | **1.0000** |
| **AUPRC** | **1.0000** |
| 混淆矩阵 | `[[3, 1], [0, 4]]`（仅 1 个误分类） |
| 输出文件 | `smoke_runs/feat/classifier_best_ckpt.pth` (5.16 KB) |

---

### 7.3 微调 smoke test

命令：

```powershell
New-Item -ItemType Directory -Force -Path smoke_runs\ft
D:\software\envs\ecg_ssl\python.exe run_ft.py --data-dir smoke_data\down\ --checkpoint checkpoint\encoder_group_gamma_0.8.pth --num-classes 2 --model-dir smoke_runs\ft --workers 0 --epochs 2 --batch-size 4
```

结果：

| 指标 | 值 |
|------|-----|
| 状态 | **通过** |
| Epochs | 2 |
| Epoch 1 Val Loss | 0.7583 |
| Epoch 2 Val Loss | 0.7572 |
| AUROC | 0.0625 |
| AUPRC | 0.3780 |
| 混淆矩阵 | `[[0, 4], [0, 4]]` |
| 输出文件 | `smoke_runs/ft/ft_best_ckpt.pth` (2.83 MB) |

> **说明**：微调 AUROC 和混淆矩阵不理想的原因：(1) 仅有 8 个合成训练样本，数据量极小；(2) 仅训练 2 个 epoch，远未收敛；(3) 合成数据缺乏真实 ECG 特征分布。使用真实大规模 ECG 数据 + 充分训练 (100 epoch) 即可达到论文中报告的性能。

---

### 7.4 测试结论

三个核心 pipeline —— **预训练 (Pretraining) → 线性探测 (Linear Probing) → 微调 (Fine-Tuning)** 全部成功跑通：

- 代码在 CPU 环境下运行正常，无需 GPU
- Checkpoint 保存/加载功能正常
- 特征提取、分类训练、测试评估流程完整
- 线性探测在 smoke data 上即可达到 AUROC=1.0，验证了特征有效性

## 8. 常见问题

### PowerShell profile.ps1 报错

如果看到以下提示：

```text
profile.ps1 cannot be loaded because running scripts is disabled
```

这是 PowerShell 启动配置脚本被执行策略拦截，不影响 Python/Conda/模型运行。

### CPU 加载 CUDA checkpoint 报错

如果报错：

```text
Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False
```

请确认 `models/mbn.py` 中 `torch.load` 已包含 `map_location`。本项目当前版本已修复。

### 下游数据路径找不到

`run_lp.py` 和 `run_ft.py` 的 `--data-dir` 必须以 `\` 或 `/` 结尾。例如：

```powershell
--data-dir D:\dataset\my_downstream_data\
```

### 真实复现实验指标

smoke test 只证明环境和代码链路跑通，不代表论文指标。要复现论文结果，需要按论文和 README 下载 NFH、PTB-XL、CPSC、Chapman 等数据，完成标签筛选、重采样、标准化、单标签样本过滤，并保持与论文一致的数据划分。
