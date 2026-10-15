# ABC_20261015 冻结协议（protocol_frozen）

- 冻结依据：`ABC联合框架补充实验方案.md`（ABC-SUPP-20261015，2026-10-15 制定并经 git 拉取确认）
- 本文件 = 方案的执行冻结版：锚点 checkpoint 映射、执行口径、阶段顺序、账本 schema。
- 冻结后只允许追加勘误（见 events.md），不允许静默改协议。
- 种子集合 `{0,2,4}`；bootstrap 10,000 次；统计 RNG `20261015`。

## 1. 模型矩阵与锚点 checkpoint 映射

| 矩阵编号 | 配置(A,B,C) | checkpoint 目录 | seeds | 来源批次 |
|---|---|---|---|---|
| M0 | 0,0,0 | `checkpoint/confirm/c1_seed{0,2,4}` | 0,2,4 | W2/W5A |
| M1 | 1,0,0 | `checkpoint/confirm/c2_seed{0,2,4}` | 0,2,4 | W2/W5A |
| M2 | 0,1,0 | `checkpoint/w6_dv/seed0` + `checkpoint/w7_dv/seed{2,4}` | 0,2,4 | W6/W7 |
| M3 | 1,1,0 | `checkpoint/joint_m3/seed{0,2,4}` | 0,2,4 | JOINT_20261013 |
| M4 | 1,1,1 | `checkpoint/joint_m4_noise/seed{0,2,4}` | 0,2,4 | JOINT_20261013 |
| M5 | 0,0,1 | `checkpoint/w8_noiseiso/seed{0,2,4}` | 0,2,4 | W8 |
| M6 | 1,0,1 | `checkpoint/joint_m6/seed{0,2,4}` | 0,2,4 | JOINT_20261013 |
| M7 | 0,1,1 | `checkpoint/abc_supp/m7_bc/seed{0,2,4}` | 0,2,4 | **本批次新增** |
| M8 | 1,Beat,1 | `checkpoint/abc_supp/m8_a_beat_c/seed{0,2,4}` | 0,2,4 | **本批次新增** |
| M9 | 1,RR,1 | `checkpoint/abc_supp/m9_a_rr_c/seed{0,2,4}` | 0,2,4 | **本批次新增** |
| N1 | M3+随机心拍窗口 | `checkpoint/joint_n1/seed{0,2,4}` | s0=JOINT, s2/s4 新增 | JOINT + 本批次 |
| N2 | M3+打乱RR标签 | `checkpoint/joint_n2/seed{0,2,4}` | s0=JOINT, s2/s4 新增 | JOINT + 本批次 |
| M4b | M4 于第二语料 | `checkpoint/abc_supp/m4_ptbxl/seed{0,2,4}` | 0,2,4 | **本批次新增(P1-2)** |

开关字面量：A=`--trc`；B=`--dualview-weight 0.01 --rr-weight 0.1 --rpeak-npz ...`；C=`--nstdb-aug 0.5 --nstdb-offset first_half`。
M8 = trc1 + dv0.01 + rr0 + C；M9 = trc1 + dv0 + rr0.1 + C；M7 = trc0 + dv0.01 + rr0.1 + C。
M8/M9 的 B 内部拆分口径与 JOINT A1/A2 一致（单开一项、显式传 R 峰缓存）。

语料：主语料 NFH=`data/pt_pretrain_nfh`(34,905 样本) + `data/pt_rpeaks_nfh.npz`；
第二语料(P1-2)=`data/pt_pretrain`(17,418 样本) + `data/pt_rpeaks.npz`（**禁止复用 NFH 缓存**，smoke 门强制样本 ID 一一对应断言）。

## 2. 预训练（沿用 JOINT protocol.md §2/§4 逐字）

NFH 100 epoch，Adam lr=0.001，batch 128，loss-mode=bt，lambda=0.0051，workers 4，`--resume`。
除被指定开关外全部默认值。R 峰离线缓存自无噪声原始信号；两个整段视图独立采噪；训练噪声=NSTDB 前半段，测试噪声=后半段（27 格 last_half，`_rng_for` 确定性实现，与 JOINT/W8 E6 同口径可直接配对）。

## 3. 下游评估口径（全部沿用既有冻结模板）

- LP：`run_lp.py --workers 6 --trc {与预训练一致} --extended-metrics 1 --save-predictions ... --protocol-id abc-20261015`，三域 ptbxl(5)/cpsc(9,W5A 修正分区, test=1385)/chapman(4)。
- FT10：W2 FT10 冻结模板（=W6 ft10grid 逐字）：`run_ft.py --fraction 0.1 --epochs 100 --batch-size 128 --learning-rate 0.0001 --workers 4 --trc {一致} --extended-metrics 1 --save-predictions ...`。10% 子集由 `get_data_loaders` 内 `torch.Generator().manual_seed(0)` 的 `random_split` 确定——同数据集同子集，与 W6 ft10grid 基线(M0=c1/M1=c2)自动一致，不重新抽样。
- 噪声 27 格：test-time last_half 扰动，逐记录 y_prob/y_true/noise_offsets 落盘；M4b 只跑 CPSC 9 格（bw/ma/em × 0/5/10 dB）。

## 4. 阶段顺序（§16 执行版）

```
audit → smoke(m7/m8/m9/m4b) → ft10fill(P0-1, 18格) → m7 → m8 → m9 →
bench(P0-4) → stats0(P0统计快照) → n1n2(P1-1, s2/s4) → m4b(P1-2) →
multilib(P1-3) → stats(全量) → report(ABC最终报告.md)
```

全程串行（F 盘 USB 桥接 IO 纪律：PT 独占串行）。幂等：PT 看 encoder+config；LP/FT 看账行+预测文件；噪声格看账行。任一阶段失败 exit 1，值班 cron 报警并由 launch_abc.ps1 幂等重启续跑。

### smoke 门（§9.1 执行版，每新配置）

1 epoch 试跑（`checkpoint/abc_smoke/`）→ 检查：各启用损失分项有限且出现（loss_dv iff dv>0；loss_rr iff rr>0），关闭分项不出现；config.json 的 A/B/C 开关、R 峰缓存、noise offset 与定义一致；`--epochs 2 --resume` 续训连续（`[resume] 恢复自 epoch 1` 且新行 epoch≥2 无 NaN）；编码器以匹配 `--trc` 结构加载（missing 仅 fc.*）；M4b 额外断言 `pt_pretrain` 样本 ID 与 `pt_rpeaks.npz` 键一一对应且非 NFH 缓存；M7 额外跑一次完整 CPSC LP 冒烟（链路端到端）。smoke 只修实现，不看测试集指标，不改科学协议。

## 5. 账本 schema（方案 §14）

`results.csv`：`ts,model,seed,pretrain_source,task,dataset,label_fraction,noise_type,snr,auroc,auprc,git_sha,checkpoint_sha,split_sha,protocol_id,hparams_ref`（末列为追加说明列）。
`split_sha` = 数据集 train/val/test 三目录排序文件清单的 SHA256（audit 阶段算一次缓存）；test 顺序样本清单随预测落盘一次（样本 ID 可追溯）。
`positive_gain_ledger.csv` / `factorial_stats.csv` / `efficiency.csv` 由统计与基准运行器产出，字段按方案 §11/§10.3/§14。

## 6. 统计口径（方案 §11）

- 逐记录配对 bootstrap 10,000，多种子同步重采样，RNG=20261015；AUPRC 主 / AUROC 次（DeLong 辅助，仅 CPSC）。
- 核心比较族（CPSC LP AUPRC 的 M4−{M0,M1,M2,M3,M5,M6,M7}）Holm 校正；探索族（B 拆分、次要域、FT10、噪声、跨语料）BH-FDR。
- 因子交互：方案 §10.3 全部 7 式 + 3 条件边际，CPSC LP，逐种子 + 同步 bootstrap CI。
- 正增益账本：delta>0 一律入账并赋 E1/E2/E3/E4 等级（方案 §3.2），措辞按 §12 决策表。
- 锚点预测目录映射与 JOINT run_joint_stats.py 一致（m0/m1=c1/c2、m2=dv、m5=noiseiso、n1n2-s0=JOINT）。

## 7. P0-4 效率基准口径

M0–M9 十模型统一：参数量(可训练/总)、单记录 MACs（hook 计数 Conv1d+Linear，环境无 thop/ptflops，口径注明）、PT 墙钟/峰值显存（manifest 有则报，锚点历史缺失则如实标 n/a）、FT10 墙钟、batch=1 推理延迟（预热 50 + 5 轮×200 次，cuda 同步，中位数+IQR）、batch=64 吞吐、推理峰值显存；记录 GPU/CUDA/PyTorch/输入长度环境字段。

## 8. P1-3 多标签口径

沿用 W6 run_multilabel_lane 冻结口径：data/cpsc 修正分区（同一患者级划分）、one-hot 主类真值（A 机未留存原始多标签引用，限制如实入账，与 W6 同映射）、冻结特征 + BCE 线性头 + val 选 checkpoint + val 选 per-class 阈值 test 一次。模型 = M0/M1/M2/M3/M4/M6/M7/M8/M9（方案 P1-3 清单）。主分析仍为单标签口径，本项为敏感性分析。

## 9. 已知缺口（如实记录，不擅自扩任务）

- M2/M5 的 CPSC/Chapman FT10 在历史批次从未运行（ft10grid 仅 b0/c1/c2），且不在方案 P0-1 的 18 任务内——论文表格该两行如需填满须另行预注册追加（本链支持 `--stages ft10ext` 一键补，默认不跑）。
- W8 最终报告尚未撰写（run_w8_stats.py 已于 10-15 完成，48 行落盘）——独立于本批次，另行处理。

## 10. 产物目录

`runlog/ABC_20261015/`：方案、本协议、events.md、patrol_log.md、results.csv、positive_gain_ledger.csv、paired_stats.csv、factorial_stats.csv、efficiency.csv、checkpoints_manifest.json、ABC最终报告.md、logs/、predictions/、feat/、ft/、multilib/。
