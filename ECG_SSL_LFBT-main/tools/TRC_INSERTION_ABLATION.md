# TRC 插入位置与训练开销旁路探针

基线：冻结提交 `2c31a2b`（W2 论文材料包）。本工具只新增旁路代码，不改变
`run_pt.py`、`run_lp.py` 或 `run_ft.py` 的主线逻辑，也不会自动启动预训练。

## 目的与位置定义

- `off`：B0，无 TRC。
- `pre_gap`：当前 C2 的位置，`block5 + maxpool -> GRN1D -> GAP`。
- `post_gap`：位置消融，`block5 + maxpool -> GAP -> GRN1D`。

两种开态都使用已有 `models.vgg_1d.GRN1D`、零初始化 `gamma/beta` 和相同的
`2*C` 参数化。`alpha=0.125` 时每导联 128 个参数，8 导联共 1024 个参数，
与 W2 冻结材料中的 C2 参数核对一致。`post_gap` 只用于结构/开销 smoke，不能
把从 B0 checkpoint 继承卷积权重的结果当成已训练的 post-GAP 效能结论。

## 文件与运行

从 `ECG_SSL_LFBT-main` 目录执行：

```powershell
# 只做获准的 seed-0 短 smoke（CPU 或 A 机 CUDA 均可）
python tools/trc_insertion_ablation.py --smoke --device auto `
  --output-csv runlog/W2/trc_insertion_smoke.csv

# A 机使用一个冻结 encoder checkpoint 测量加载后的开销；不训练新权重
python tools/trc_insertion_ablation.py --mode pre_gap --smoke --device cuda `
  --checkpoint checkpoint/confirm/c2_seed0/encoder_group.pth `
  --output-csv runlog/W2/trc_pre_gap_smoke.csv

# 若要检查 post-GAP 的结构开销，可从 B0/C1 卷积权重加载；这仍不是 post-GAP 训练
python tools/trc_insertion_ablation.py --mode post_gap --smoke --device cuda `
  --checkpoint checkpoint/confirm/c1_seed0/encoder_group.pth `
  --output-csv runlog/W2/trc_post_gap_smoke.csv
```

单测（不启动长训练）：

```powershell
python tests/test_trc_insertion_ablation.py
```

`--measure-steps` 超过 20 时必须显式加 `--allow-long`；本任务不授权该选项。

## CSV 输出契约

每个模式一行，固定字段如下：

`timestamp_utc, mode, seed, device, cuda_name, torch_version, num_leads, alpha,
batch_size, signal_length, warmup_steps, measure_steps, params_total, params_trc,
trc_share_pct, forward_ms, train_step_ms, throughput_samples_per_s,
peak_memory_mb, output_checksum, checkpoint_path, checkpoint_sha256,
checkpoint_config_sha256, checkpoint_load_status, checkpoint_arch_hint, git_sha,
status, error`。

其中：

- `params_total/params_trc/trc_share_pct` 用于核对 1024/0.161% 的 W2 事实；
- `forward_ms` 是前向均值，`train_step_ms` 是一次合成标量损失的前向+反向+SGD
  均值，`throughput_samples_per_s` 按 batch 样本计；
- `peak_memory_mb=-1` 表示 CPU，CUDA 时为 `max_memory_allocated`；
- `checkpoint_sha256` 和旁边 `config.json` 的 SHA 防止错用 checkpoint；
- `checkpoint_load_status=loaded_with_trc_position_mismatch` 只表示允许的
  GRN 键差异，必须在报告里注明是“继承卷积权重的结构 smoke”；
- `git_sha` 是实际运行时的仓库 HEAD，不得手填。

## 预注册停止条件

1. 任何非 TRC 卷积/BN state key 缺失或多余：立即停止并修复 checkpoint/架构，
   不允许用 `strict=False` 掩盖。
2. smoke 中出现 NaN/Inf、输出 shape 不为 `[B, 8, 64]`、或同一 seed 重跑不一致：
   停止，不进入任何长训练。
3. 若 post-GAP 仅用于开销测量而未独立预训练，不报告下游 AUPRC/鲁棒性提升。
4. 训练开销消融只允许 seed 0、合成输入、最多 20 个测量步；不得借此替代 W2
   C1/C2 三 seed 统计，也不得重复 W2 主结果。
5. 只有在导师明确授权后，才可另行设计 post-GAP 5-epoch smoke；若 smoke 相对
   pre-GAP 的 `train_step_ms` 或显存增幅超过 10%，先归档为高开销候选，不做 full。

## 预期 A 机验收

- `python tests/test_trc_insertion_ablation.py` 返回 PASS（无 PyTorch 环境可明确
  输出 SKIP）；
- `--smoke` 生成三行 CSV，三行 `status=ok`；
- `off` 的 `params_trc=0`，`pre_gap/post_gap` 的 8 导联 `params_trc=1024`；
- 产物和运行命令写入 `runlog/W2/`，连同 checkpoint SHA、代码 SHA 一并归档。

