# LFBT 论文复现指导（AI 执行版）

> 目标：复现 Liu et al., *Lead-fusion Barlow twins: A fused self-supervised learning method for multi-lead electrocardiograms*, Information Fusion 114 (2025) 102698。
>
> 代码仓库：`https://github.com/Aiwiscal/ECG_SSL_LFBT`
>
> 本文件是一份可以直接交给 AI 编程助手执行的复现规格。AI 应先检查本文件中的验收条件，再修改代码或运行实验。

## 1. 先给结论：应该选什么数据集

### 最推荐的复现组合

| 阶段 | 数据集 | 用途 | 是否需要标签 |
|---|---|---|---|
| SSL 预训练 | **NFH（Ningbo First Hospital）** | 论文主实验的无监督预训练 | 不使用标签 |
| 下游任务 1 | **PTB-XL** | 5 类心电异常分类 | 使用标签 |
| 下游任务 2 | **CPSC-2018** | 9 类心电异常分类 | 使用标签 |
| 下游任务 3 | **Chapman ECG** | 4 类节律分类 | 使用标签 |

这四个公开数据集是论文的可复现实验组合。SNPH（上海第九人民医院黄浦分院）只有作者可用，不能作为普通复现的前置条件。

### 按工作量选择

1. **严格复现论文**：NFH 预训练，然后分别在 PTB-XL、CPSC-2018、Chapman 上进行 LP 和 FT。
2. **最小可行复现**：NFH 预训练 + PTB-XL 一个下游任务，同时做 LP、FT、随机初始化 TFS 三组对照。
3. **拿不到 NFH 时**：可以用 PTB-XL 的训练部分作为无标签预训练集，再用其余数据做下游任务，但这不是论文原始设置；必须按患者/官方划分隔离预训练和测试，避免数据泄漏，并在报告中明确标注为“替代预训练实验”。

### 数据下载地址

- NFH：<https://physionet.org/content/challenge-2021/1.0.3/training/ningbo/#files-panel>
- PTB-XL：<https://physionet.org/content/ptb-xl/1.0.3/>
- CPSC-2018：<http://2018.icbeb.org/Challenge.html>
- Chapman ECG：<https://figshare.com/collections/ChapmanECG/4560497/2>

## 2. 论文方法的可执行定义

### 输入张量

- 原始标准 12 导联顺序：`I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`。
- LFBT 实际只保留 8 个导联：`II, III, V1, V2, V3, V4, V5, V6`。
- 每条记录统一为 10 秒、500 Hz，即 5000 点。
- 用 `scipy.signal.resample(signal, 2048, axis=1)` 重采样为 `[8, 2048]`。
- 对每条多导联记录做 z-score：`x = (x - mean(x)) / (std(x) + 1e-5)`。实现时固定统计维度并记录是否是“整条记录统一统计”还是“逐导联统计”；论文文字倾向于整条多导联记录，必须在实验日志中写清楚。
- CPSC 记录：超过 10 秒切成 10 秒片段；不足 10 秒在尾部补零；然后重采样和归一化。
- 只保留单标签记录。多标签记录按官方元数据过滤，不要把多标签强行映射成一个类别。

### 模型

- 每个导联使用一个独立的 VGG16-style 1D CNN：`VGG16(ch_in=1, alpha=0.125)`。
- 每个导联的 encoder 输出维度为 64；8 个导联在下游拼接后得到 512 维。
- 每个导联一个 3 层 MLP projector：`2048 -> 2048 -> 2048`，中间层使用 BN 和 ReLU。
- 下游使用 Multi-Branch Concatenation (MBC)：8 个 encoder 输出拼接，再接线性分类器。

### 两个视图与损失

对同一批 `[B, 8, 2048]` ECG 生成两个增强视图：

- Random Resize Crop：裁剪比例 `[0.5, 1.0]`，再重采样回原长度。
- Time Out：遮挡比例 `[0.0, 0.5]`，被遮挡区间置零。
- 代码中的 `RandomResizeCropTimeOut` 已实现这两个操作。

每个导联分别编码、投影、BN 后计算交叉相关矩阵：

```text
C_ij = BN(z1_i)^T @ BN(z2_j) / B
L_ij = sum((diag(C_ij) - 1)^2) + lambda * sum(offdiag(C_ij)^2)
L_r = mean_i L_ii
L_t = mean_{i != j} L_ij
L = gamma * L_r + (1 - gamma) * L_t
```

默认参数：`lambda=0.0051`；PTB-XL/CPSC 使用 `gamma=0.8`；Chapman 使用 `gamma=0.6`。如果只做一个任务，先用 `gamma=0.8`，然后再做 `gamma ∈ {0.6, 0.8, 1.0}` 消融。

## 3. 数据转换规范

官方仓库只读取 `.npy`，没有提供 WFDB/原始数据到 `.npy` 的转换脚本。因此必须先写一个独立的 `prepare_data.py`，并满足以下契约。

### 预训练目录

```text
data/pt_nfh/
└── samples/
    ├── sample_000000.npy
    ├── sample_000001.npy
    └── ...
```

每个文件必须满足：

```python
arr = np.load(path)
assert arr.shape == (8, 2048)
assert arr.dtype in (np.float32, np.float64)
assert np.isfinite(arr).all()
```

### 下游目录

代码要求 `train/`、`val/`、`test/` 位于数据目录下，并使用 ImageFolder 风格的类别子目录：

```text
data/ptbxl/
├── train/
│   ├── NORM/*.npy
│   ├── CD/*.npy
│   ├── HYP/*.npy
│   ├── MI/*.npy
│   └── STTC/*.npy
├── val/
└── test/
```

对应类别数：PTB-XL=`5`，CPSC=`9`，Chapman=`4`。

论文表 2 的目标规模如下；实际文件数量不同通常意味着筛选、切片或标签映射不一致，应先停下来排查：

| 数据集 | 类别 | train | val | test |
|---|---:|---:|---:|---:|
| PTB-XL | 5 | 12978 | 1642 | 1652 |
| CPSC | 9 | 8958 | 1303 | 2598 |
| Chapman | 4 | 7651 | 851 | 2126 |

### 标签映射

- PTB-XL：读取 `ptbxl_database.csv` 和 `scp_statements.csv`，按论文的 5 个 superclass 映射为 `NORM/CD/HYP/MI/STTC`；只保留单标签样本。
- CPSC-2018：使用官方标签文件映射为 `NORM/AF/I-AVB/LBBB/RBBB/PAC/PVC/STD/STE`；CPSC 长记录先切片，切片继承原记录标签。
- Chapman：按官方诊断标签映射为 `AF/GSVT/SB/SR`。
- 类别目录名必须稳定，且 train/val/test 使用同一套类别到整数标签的排序。不要让 Windows 文件夹排序或临时目录名改变标签编号。

### 数据划分

- PTB-XL 使用官方推荐的 stratified fold 划分（论文参考 PTB-XL 官方划分）。
- 其他数据集按论文/官方划分复现；如果重新随机划分，固定随机种子并在结果表中标注“非原始划分”。
- 任何预训练数据都不得包含下游 test 患者的记录。理想做法是按患者 ID 隔离，而不是只按文件名隔离。

## 4. 运行顺序

以下命令在仓库根目录执行。Windows 下建议使用 PowerShell 或 Conda 环境；路径末尾必须带 `/` 或 `\\`，因为当前代码使用字符串拼接。

### 环境

论文环境是 PyTorch 1.11 + Python 3.8 左右；仓库 `requirements.txt` 固定了：

```text
numpy==1.22.4
scikit_learn==1.3.0
scipy==1.10.1
torch==1.11.0
torchvision==0.12.0
tqdm==4.61.2
```

如果使用较新的 PyTorch，先做一次 shape/权重加载测试，再开始长时间训练。必须记录：Python、PyTorch、CUDA、GPU 型号、commit hash、随机种子。

### 预训练

```bash
python run_pt.py \
  --data-dir data/pt_nfh/ \
  --num-leads 8 \
  --epochs 200 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --gamma 0.8 \
  --lambd 0.0051 \
  --projector 2048-2048-2048 \
  --checkpoint-dir checkpoint/nfh_gamma_08/
```

输出应包含每个 epoch 的 `loss`、`loss_r`、`loss_t`，并最终生成 `encoder_group.pth`。

### 线性探测（LP）

```bash
python run_lp.py \
  --data-dir data/ptbxl/ \
  --checkpoint checkpoint/nfh_gamma_08/encoder_group.pth \
  --num-classes 5 \
  --feat-dir results/ptbxl_lp/
```

LP 冻结 encoder，只训练线性分类器；应报告 test AUROC、AUPRC 和混淆矩阵。

### 微调（FT）

```bash
python run_ft.py \
  --data-dir data/ptbxl/ \
  --checkpoint checkpoint/nfh_gamma_08/encoder_group.pth \
  --num-classes 5 \
  --fraction 1.0 \
  --model-dir results/ptbxl_ft/
```

论文还比较 `fraction=0.1, 0.2, 0.5, 1.0`，以及随机初始化的 TFS 对照。至少应完成 `fraction=0.1` 和 `1.0`。

## 5. 代码审计与必须修正项

以下问题来自官方仓库当前实现，不修正可能导致“命令能启动但结果不对”或 CPU 直接报错。

### P0：CPU 路径错误

`run_pt.py` 的训练循环无条件执行 `y1.cuda(gpu)` 和 `y2.cuda(gpu)`。应改为：

```python
y1 = y1.to(model.device, non_blocking=True)
y2 = y2.to(model.device, non_blocking=True)
```

或者在进入训练前明确要求 CUDA，并在文档中拒绝 CPU 运行。推荐前者。

### P0：数据目录字符串拼接

`data_utils/cls_datasets.py` 使用 `data_path + "train"`。因此必须传入带末尾分隔符的路径；更稳妥的代码修复是使用 `pathlib.Path(data_path) / "train"`。

### P1：预训练 checkpoint 保存格式

`run_pt.py` 保存的是由多个 `VGG16` 模块组成的对象列表，而不是纯 `state_dict`。这依赖 PyTorch pickle，跨版本加载可能失败。建议同时保存每个 encoder 的 `state_dict`，并在 LP/FT 中兼容两种格式。

### P1：FT 的 checkpoint 加载

`run_ft.py` 的 `torch.load(checkpoint)` 没有 `map_location`。在 CPU 或不同 GPU 环境加载时应增加 `map_location=device`。

### P1：随机性与子集采样

`random_split`、增强和 DataLoader worker 都没有统一随机种子。复现实验至少固定 Python、NumPy、PyTorch、CUDA、DataLoader worker 的 seed，并重复 3 次报告均值和标准差。

### P1：LP 结果可比性

LP 代码默认 `feat_dim=512`，这隐含了 8 个导联且每个 encoder 输出 64 维。若改变 `num_leads` 或 backbone 宽度，必须同步修改分类器输入维度。

### P2：增强的原地修改

`RandomResizeCropTimeOut` 会原地修改输入数组。确认数据集每次返回的是独立数组，避免缓存数组被下一次增强污染。

## 6. AI 执行任务清单

把本文件交给 AI 后，要求它严格按以下顺序工作：

1. 扫描仓库并报告 `run_pt.py`、`run_lp.py`、`run_ft.py`、`data_utils/`、`models/` 的实际路径；不要假设目录名。
2. 写 `prepare_data.py`，支持 NFH/PTB-XL/CPSC/Chapman，输出统一的 `[8,2048]` `.npy`，并生成 `manifest.json`（原始文件、患者 ID、标签、split、shape、采样率、预处理版本）。
3. 先运行数据质量检查：文件数、shape、NaN/Inf、每导联均值/标准差、类别分布、患者泄漏。
4. 修复本文件第 5 节列出的 P0/P1 问题，并为修复添加最小单元测试。
5. 用 2 个 epoch、batch size 4 做 smoke test：确认两个增强视图 shape 为 `[B,8,2048]`，loss 可反向传播，checkpoint 可保存和重新加载。
6. 再执行正式 NFH 预训练；保存配置、日志、git commit、GPU、耗时和 checkpoint SHA256。
7. 在 PTB-XL 上完成 LP、FT、TFS；确认 test 集只在最终 checkpoint 选择后评估一次。
8. 扩展到 CPSC 和 Chapman；每个数据集独立报告 AUROC/AUPRC，不要把不同数据集样本混在一个 test 集中。
9. 做最小消融：`gamma={0.6,0.8,1.0}`、`batch_size={32,128,256}`、LP 与 FT、预训练与随机初始化。
10. 输出 `results.csv`、训练曲线、混淆矩阵、环境信息和失败记录。

## 7. 复现验收标准

### 数据层

- [ ] NFH 预训练样本全部为 `[8,2048]`。
- [ ] PTB-XL/CPSC/Chapman 的 train/val/test 目录完整，类别数分别为 5/9/4。
- [ ] 没有 NaN/Inf，且没有患者跨 split 泄漏。
- [ ] 多标签记录处理策略与日志一致。

### 模型层

- [ ] 8 个独立 encoder 都收到对应导联输入，而不是先把 8 个导联混成单个通道。
- [ ] `loss_r` 使用 `i == j`，`loss_t` 使用 `i != j`，两者归一化正确。
- [ ] projector 维度为 2048，`lambda=0.0051`，默认 `gamma=0.8`。
- [ ] 下游 MBC 拼接后特征维度为 512。

### 实验层

- [ ] LP 只更新线性分类器；FT 更新 encoder 和分类器。
- [ ] 最优 checkpoint 由 validation loss 选择，test 只做最终评估。
- [ ] 报告 AUROC、AUPRC、混淆矩阵，并注明 macro/average 方式。
- [ ] 有 TFS 随机初始化对照和至少一次 10% 标签实验。

## 8. 论文结果的合理参考范围

不要把论文数值当作必须逐位相等的测试答案；数据版本、随机种子、标签筛选和 PyTorch 版本都会造成偏差。论文 NFH 预训练的主要参考值是：

| 下游数据集 | LP AUROC | LP AUPRC | FT AUROC | FT AUPRC |
|---|---:|---:|---:|---:|
| PTB-XL | 0.9082 | 0.6923 | 0.9178 | 0.7257 |
| CPSC | 0.9360 | 0.7258 | 0.9536 | 0.7892 |
| Chapman | 0.9934 | 0.9801 | 0.9970 | 0.9910 |

更重要的复现判据是趋势：LFBT 应优于随机初始化 LP/TFS，且在 10% 下游训练数据时仍保持明显优势；`gamma=1.0`（去掉 inter-lead loss）通常应比 `gamma=0.8` 差。

## 9. 结果记录模板

```yaml
paper: "Lead-fusion Barlow twins"
repo: "Aiwiscal/ECG_SSL_LFBT"
commit: "<git commit>"
pretrain_dataset: "NFH"
downstream_dataset: "PTB-XL"
input_shape: [8, 2048]
lead_order: [II, III, V1, V2, V3, V4, V5, V6]
sampling_rate_hz: 500
duration_seconds: 10
gamma: 0.8
lambd: 0.0051
projector: "2048-2048-2048"
batch_size: 128
pretrain_epochs: 200
downstream_epochs: 100
lp_learning_rate: 0.001
ft_learning_rate: 0.0001
seed: 0
metrics:
  auroc: null
  auprc: null
artifacts:
  checkpoint: null
  results_csv: null
  log: null
notes: null
```

## 10. 不要误判的事项

- 论文标题中的“多导联”不等于代码把 12 导联直接作为 12 通道输入；该实现是 8 个导联、8 个独立 encoder。
- NFH 用于预训练，标签不参与损失；PTB-XL/CPSC/Chapman 用于下游分类。
- SNPH 是论文的私有鲁棒性实验，不是必须下载的数据集。
- 只有跑通预训练 loss 下降，不能证明复现成功；必须完成下游 LP/FT 和随机初始化对照。
- 如果只能选择一个公开下游数据集，优先 PTB-XL；如果只能选择一个公开数据集做“方法演示”，可以用 PTB-XL 替代 NFH 预训练，但必须声明与论文不完全一致。

