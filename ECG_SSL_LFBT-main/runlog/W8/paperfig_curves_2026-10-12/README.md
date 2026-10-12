# 论文配图母版第 6/7 页曲线数据包（2026-10-12）

对应《论文配图母版_14图_W8_2026-10-12_v3.pptx》第 6 页（预训练损失曲线）与第 7 页（下游训练曲线）。
本包由主机 A（F:\新实验）整备，全部数据可溯源到 runlog 原始日志或周批次台账。

## 文件清单

| 文件 | 内容 |
|---|---|
| `fig6_pretrain_curves.csv` | 第 6 页：四分支预训练逐 step 损失（长表） |
| `fig7_downstream_curves.csv` | 第 7 页：C2→CPSC 三阶段逐 epoch 验证损失 |
| `fig7_final_metrics.csv` | 第 7 页终点指标（AUROC/AUPRC，取自台账，非曲线） |
| `raw_logs/` | 16 份原始日志副本（仓库 .gitignore 忽略 *.log，故随包入库） |
| `build_curves.py` | 解析脚本，可复现上表（`python build_curves.py`） |

## 第 6 页：预训练损失曲线

长表字段：`branch, seed, source, note, epoch, step, loss, loss_r, loss_t, loss_dv, loss_rr`
（`loss_r`=导联内 BT，`loss_t`=导联间 BT，`loss_dv`/`loss_rr`=Beat-RR 支路；print_freq=100，每种子约 273 行，epoch 0–99 完整）

| branch | seed | 来源 | 最终 loss | 口径注记 |
|---|---|---|---|---|
| beat_rr | 0/2/4 | W6/logs/dv_pt_seed0.log、W7/logs/dv_pt_seed{2,4}.log | ~225–229 | 官方链原生日志，五字段全 |
| noise_iso | 0/2/4 | W8/logs/noiseiso_pt_seed{0,2,4}.log | ~259–266 | 官方链原生日志；**无 loss_dv/loss_rr 属设计**（该分支不含 dv/rr 头），图例勿画空系列 |
| c1_lfbt | 0 | W1/pt_c1_nfh_b0_seed0.log | 248.3 | **同协议替身，非 confirm 重跑本身**（见脚注 A） |
| c2_trc | 0 | W8/logs/c2pt_fig_seed0.log | 246.0 | **2026-10-12 补录**（见脚注 B） |

**脚注 A（C1 替身口径）**：W2 confirm 矩阵的六次预训练（c1/c2 × seeds 0/2/4，2026-09-21~22 各约 2h10m 全部成功）逐 epoch stdout 被 `runlog/W2/confirm_matrix.py` 第 76 行 `subprocess.run(..., capture_output=True)` 捕获后丢弃，全机无存档；`checkpoint/confirm/*/train_state.pth` 仅含权重/优化器/RNG，无法反推曲线。W1 期（2026-09-20）存在一次与 confirm 完全同配方的 C1 运行（`data/pt_pretrain_nfh`、100ep、bs128、lr0.001、gamma0.8、lambd0.0051、seed0、trc=0），checkpoint 落于 `checkpoint/M/c1_nfh_b0_seed0`（sha256=237f0607…，已录 W2/freeze_manifest_v2.json "c1_nfh_seed0(W1)"），其日志即本包 `W1_pt_c1_nfh_b0_seed0.log`（整文件 UTF-16LE BOM，解析器已处理）。**图注建议标"同协议同种子复跑"**。仅 seed0，无 2/4。

**脚注 B（C2 补录口径）**：C2（LFBT+TRC）全机无任何逐 epoch 损失记录（W1 期 c2_chain.log 亦只有 6 行状态行，PT stdout 同样被丢弃）。2026-10-12 按 confirm_matrix.py 第 90–92 行命令**逐字对齐**补跑一次（仅 `--checkpoint-dir` 改为全新路径 `checkpoint/w8_paperfig/c2_pt_seed0`，**未触碰 `checkpoint/confirm/c2_seed0`**）：

```
C:/Users/admin/.conda/envs/DL/python.exe -u run_pt.py --data-dir data/pt_pretrain_nfh \
  --epochs 100 --batch-size 128 --workers 4 --seed 0 --trc 1 --resume \
  --checkpoint-dir checkpoint/w8_paperfig/c2_pt_seed0
```

补录代码 = 仓库 HEAD `481ffab`；confirm 原跑代码 = 冻结版 `fb08bc5`（W2/freeze_manifest_v2.json）。两者 diff 仅为 W6–W8 新增的默认关闭分支（loss_mode=bt、wd=0.0、nstdb_aug=0 等均"默认=历史行为"），bt+trc 主路径未变。下游全部结论仍基于 confirm checkpoint，本曲线仅作图。

补录实况：2026-10-12 11:33–12:52（1h22m，GPU 独占，47s/epoch），exit=0，epoch 0–99 完整（273 行），最终 total loss 246.0（同配方 C1 替身终值 248.3，量级一致）。产物 checkpoint：`checkpoint/w8_paperfig/c2_pt_seed0/encoder_group.pth`，sha256=`607c69f5e4500d297e7771624e2a17f9812d1cd45133852a688b09b3c4d8eda7`（与 confirm/c2_seed0 的 `af242a89…` 不同属预期——两次独立同配方运行；下游未使用本 checkpoint）。

## 第 7 页：下游训练曲线（C2→CPSC）

长表字段：`phase, seed, source, epoch, val_loss`（各 100 epoch × 100 点完整）

| phase | seed | 来源 | 说明 |
|---|---|---|---|
| lp | 0/2/4 | W5/logs/c2_cpsc_seed{0,2,4}.log | LP，日志行格式 `Val_loss: x.` |
| ft10 | 0/2/4 | W6/logs/c2_cpsc_ft10_seed{0,2,4}.log | 10% 微调，`Val loss: x` |
| lpft | 0/2/4 | W7/logs/e4a_c2_cpsc_lpft_seed{0,2,4}.log | LP-FT（E4a，头用 LP 最优初始化） |

**口径限制**：run_lp.py/run_ft.py 逐 epoch 只打印验证损失，无逐 epoch 训练损失、无逐 epoch 验证 AUROC——现有日志只能画"验证损失曲线"。终点 AUROC/AUPRC 见 `fig7_final_metrics.csv`（W5 lp_results / W6 ft10grid_results_L1 / W7 e4a_lpft 三台账，checkpoint=c2、downstream=cpsc），如需在曲线终点标注可直接取用。若确需"训练损失+验证 AUROC"须加日志重跑，不建议为此返工。

## 复现

```
cd runlog/W8/paperfig_curves_2026-10-12
python build_curves.py     # 重新解析全部来源日志并覆盖生成三个 CSV
```
