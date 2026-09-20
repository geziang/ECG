[20:43:40] 夜间队列启动
[22:35:46] N1 预训练完成
[22:35:47] == N1 LP 评估 ==
AUROC =  0.905015050727976
AUPRC =  0.6869711284563428
[22:37:12] == S1 粗筛: d9vicreg seed 0 开始 ==
[01:26:55] d9vicreg seed0 预训练完成
[01:26:56] d9vicreg seed0 LP 开始
AUROC =  0.9080459660628184
AUPRC =  0.6981199929490769
[01:28:31] == S1 粗筛: d9vicreg_bn seed 0 开始 ==
[04:19:12] d9vicreg_bn seed0 预训练完成
[04:19:12] d9vicreg_bn seed0 LP 开始
AUROC =  0.8886706807944627
AUPRC =  0.6533561994332703
[04:20:49] == S1 粗筛: d9vicreg_ln seed 0 开始 ==
[07:09:55] d9vicreg_ln seed0 预训练完成
[07:09:55] d9vicreg_ln seed0 LP 开始
AUROC =  0.9016437911241155
AUPRC =  0.6797653238274567
[07:11:21] == S1 粗筛: d9vicreg seed 2 开始 ==
[09:46] 等待超时(13h)放弃接力
[10:01:03] d9vicreg seed2 预训练完成
[10:01:03] d9vicreg seed2 LP 开始
AUROC =  0.9005837216616774
AUPRC =  0.6829720242622008
[10:02:26] 夜间队列全部结束
[10:03] == 接力: B0 seed2 复跑开始 ==
AUROC =  0.9157079113708878
AUPRC =  0.7257903894547031
[12:09] 接力队列结束(预计 ~08:30)
[12:10] == D9 确认: d9r1 seed --loss-mode vicreg --vicreg-sim 1 --vicreg-var 10 --vicreg-cov 1 开始 ==
[12:10] d9r1 seed--loss-mode vicreg --vicreg-sim 1 --vicreg-var 10 --vicreg-cov 1 预训练失败
[12:10] == D9 确认: d9lite seed --bt-var-hinge 10 开始 ==
[12:10] d9lite seed--bt-var-hinge 10 预训练失败
[12:10] 下午队列结束(D9 最终判定材料齐)
[12:11] 修正: 下午队列参数顺序 bug 已修复,重新启动(12:10 的秒败为脚本 bug 非训练问题)
[12:11] == D9 确认: d9r1 seed 0 开始 ==
[14:28] d9r1 seed0 预训练完成
AUROC =  0.863028840890142
AUPRC =  0.6105628673196649
[14:29] == D9 确认: d9lite seed 0 开始 ==
[16:25] == M 第一批粗探启动 (D1L / D1L-NEG / N4) ==
[m_screen] 计划 3 个 run (fast=False, 基线 Δ 锚点 0.7177)
  - d1l_05_02 seed0: --d1l 0.5,0.2
  - d1l_neg seed0: --d1l 0.5,0.2 --d1l-shuffle
  - n4_ema999 seed0: --ema-decay 0.999
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 --workers 6 --seed 0 --checkpoint-dir F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\d1l_05_02_seed0 --d1l 0.5,0.2
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_lp.py --data-dir data/ptbxl --checkpoint F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\d1l_05_02_seed0\encoder_group.pth --num-classes 5 --feat-dir F:\新实验\ECG_SSL_LFBT-main\feat\M_d1l_05_02_seed0 --seed 0 --workers 6
[m_screen] ✔ d1l_05_02_seed0: AUPRC=0.7117 Δ=-0.0060
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 --workers 6 --seed 0 --checkpoint-dir F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\d1l_neg_seed0 --d1l 0.5,0.2 --d1l-shuffle
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_lp.py --data-dir data/ptbxl --checkpoint F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\d1l_neg_seed0\encoder_group.pth --num-classes 5 --feat-dir F:\新实验\ECG_SSL_LFBT-main\feat\M_d1l_neg_seed0 --seed 0 --workers 6
[m_screen] ✔ d1l_neg_seed0: AUPRC=0.7081 Δ=-0.0096
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 --workers 6 --seed 0 --checkpoint-dir F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\n4_ema999_seed0 --ema-decay 0.999
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_lp.py --data-dir data/ptbxl --checkpoint F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\n4_ema999_seed0\encoder_group.pth --num-classes 5 --feat-dir F:\新实验\ECG_SSL_LFBT-main\feat\M_n4_ema999_seed0 --seed 0 --workers 6
[m_screen] ✔ n4_ema999_seed0: AUPRC=0.6909 Δ=-0.0268

[m_screen] 当前矩阵:
  d1l_05_02          s0 Δ=-0.006 ❌
  d1l_neg            s0 Δ=-0.0096 ❌
  n4_ema999          s0 Δ=-0.0268 ❌
[22:06] M 第一批结束
[22:08] == M 第二批启动: B0 超参邻域重调(gamma 0.7/0.9, lambd 0.003/0.01) ==
[m_screen] 计划 5 个 run (fast=False, 基线 Δ 锚点 0.7177)
  - aug_crop_strong seed0: --aug-params 0.4,1.0,0.0,0.5
  - aug_crop_weak seed0: --aug-params 0.7,1.0,0.0,0.5
  - aug_mask_weak seed0: --aug-params 0.5,1.0,0.0,0.3
  - proj_dim1024 seed0: --projector 1024-1024-1024
  - proj_dim4096 seed0: --projector 4096-4096-4096
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 --workers 6 --seed 0 --checkpoint-dir F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\aug_crop_strong_seed0 --aug-params 0.4,1.0,0.0,0.5
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_lp.py --data-dir data/ptbxl --checkpoint F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\aug_crop_strong_seed0\encoder_group.pth --num-classes 5 --feat-dir F:\新实验\ECG_SSL_LFBT-main\feat\M_aug_crop_strong_seed0 --seed 0 --workers 6
[m_screen] ✔ aug_crop_strong_seed0: AUPRC=0.7107 Δ=-0.0070
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 --workers 6 --seed 0 --checkpoint-dir F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\aug_crop_weak_seed0 --aug-params 0.7,1.0,0.0,0.5
ata/ptbxl --checkpoint F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\b0_gamma07_seed0\encoder_group.pth --num-classes 5 --feat-dir F:\新实验\ECG_SSL_LFBT-main\feat\M_b0_gamma07_seed0 --seed 0 --workers 6
[m_screen] ✔ b0_gamma07_seed0: AUPRC=0.7136 Δ=-0.0041
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 --workers 6 --seed 0 --checkpoint-dir F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\b0_gamma09_seed0 --gamma 0.9
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_lp.py --data-dir data/ptbxl --checkpoint F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\b0_gamma09_seed0\encoder_group.pth --num-classes 5 --feat-dir F:\新实验\ECG_SSL_LFBT-main\feat\M_b0_gamma09_seed0 --seed 0 --workers 6
[m_screen] ✔ b0_gamma09_seed0: AUPRC=0.7111 Δ=-0.0066
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 --workers 6 --seed 0 --checkpoint-dir F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\b0_lambd003_seed0 --lambd 0.003
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_lp.py --data-dir data/ptbxl --checkpoint F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\b0_lambd003_seed0\encoder_group.pth --num-classes 5 --feat-dir F:\新实验\ECG_SSL_LFBT-main\feat\M_b0_lambd003_seed0 --seed 0 --workers 6
[m_screen] ✔ b0_lambd003_seed0: AUPRC=0.7086 Δ=-0.0091
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 --workers 6 --seed 0 --checkpoint-dir F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\b0_lambd01_seed0 --lambd 0.01
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_lp.py --data-dir data/ptbxl --checkpoint F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\b0_lambd01_seed0\encoder_group.pth --num-classes 5 --feat-dir F:\新实验\ECG_SSL_LFBT-main\feat\M_b0_lambd01_seed0 --seed 0 --workers 6
[m_screen] ✔ b0_lambd01_seed0: AUPRC=0.7109 Δ=-0.0068

[m_screen] 当前矩阵:
  b0_gamma07         s0 Δ=-0.0041 ❌
  b0_gamma09         s0 Δ=-0.0066 ❌
  b0_lambd003        s0 Δ=-0.0091 ❌
  b0_lambd01         s0 Δ=-0.0068 ❌
[06:30] M 第二批结束
[06:31] == M 第三批启动: 增强扫描+投影维度 ==
[m_screen] 计划 4 个 run (fast=False, 基线 Δ 锚点 0.7177)
  - aug_crop_weak seed0: --aug-params 0.7,1.0,0.0,0.5
  - aug_mask_weak seed0: --aug-params 0.5,1.0,0.0,0.3
  - proj_dim1024 seed0: --projector 1024-1024-1024
  - proj_dim4096 seed0: --projector 4096-4096-4096
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 --workers 6 --seed 0 --checkpoint-dir F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\aug_crop_weak_seed0 --aug-params 0.7,1.0,0.0,0.5
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_lp.py --data-dir data/ptbxl --checkpoint F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\aug_crop_weak_seed0\encoder_group.pth --num-classes 5 --feat-dir F:\新实验\ECG_SSL_LFBT-main\feat\M_aug_crop_weak_seed0 --seed 0 --workers 6
[m_screen] ✔ aug_crop_weak_seed0: AUPRC=0.7204 Δ=+0.0027
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 --workers 6 --seed 0 --checkpoint-dir F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\aug_mask_weak_seed0 --aug-params 0.5,1.0,0.0,0.3
[09:34] == v2 链条启动: 裁剪趋势优先(aug_crop_weak +0.27 首个正探针) ==
[m_screen] 计划 5 个 run (fast=False, 基线 Δ 锚点 0.7177)
  - aug_crop_080 seed0: --aug-params 0.8,1.0,0.0,0.5
  - aug_crop_090 seed0: --aug-params 0.9,1.0,0.0,0.5
  - aug_mask_weak seed0: --aug-params 0.5,1.0,0.0,0.3
  - proj_dim1024 seed0: --projector 1024-1024-1024
  - proj_dim4096 seed0: --projector 4096-4096-4096
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 --workers 6 --seed 0 --checkpoint-dir F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\aug_crop_080_seed0 --aug-params 0.8,1.0,0.0,0.5
[10:15] == 车道A启动: batch3c(裁剪边界+mask+proj) ==
[m_screen] 计划 5 个 run (fast=False, 基线 Δ 锚点 0.7177)
  - aug_crop_080 seed0: --aug-params 0.8,1.0,0.0,0.5
  - aug_crop_090 seed0: --aug-params 0.9,1.0,0.0,0.5
  - aug_mask_weak seed0: --aug-params 0.5,1.0,0.0,0.3
  - proj_dim1024 seed0: --projector 1024-1024-1024
  - proj_dim4096 seed0: --projector 4096-4096-4096
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 --workers 6 --seed 0 --checkpoint-dir F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\aug_crop_080_seed0 --aug-params 0.8,1.0,0.0,0.5
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_lp.py --data-dir data/ptbxl --checkpoint F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\aug_crop_080_seed0\encoder_group.pth --num-classes 5 --feat-dir F:\新实验\ECG_SSL_LFBT-main\feat\M_aug_crop_080_seed0 --seed 0 --workers 6
[m_screen] ✔ aug_crop_080_seed0: AUPRC=0.7165 Δ=-0.0012
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 --workers 6 --seed [m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_lp.py --data-dir data/ptbxl --checkpoint F:\新实验\[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_lp.py --data-dir data/ptbxl --checkpoint F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\aug_crop_090_seed0\encoder_group.pth --num-classes 5 --feat-dir F:\新实验\ECG_SSL_LFBT-main\feat\M_aug_crop_090_seed0 --seed 0 --workers 6
[m_screen] ✔ aug_crop_090_seed0: AUPRC=0.7081 Δ=-0.0096
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 --workers 6 --seed 0 --[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_lp.py --data-dir data/ptbxl --checkpoint F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\aug_crop_weak_seed4\encoder_group.pth --num-classes 5 --feat-dir F:\新实验\ECG_SSL_LFBT-main\feat\M_aug_crop_weak_seed4 --seed 0 --workers 6
[m_screen] ✔ aug_crop_weak_seed4: AUPRC=0.7123 Δ=-0.0054

[m_screen] 当前矩阵:
  aug_crop_090       s0 Δ=-0.0096 ❌
  aug_crop_weak      s4 Δ=-0.0054 ❌
[16:00] 车道B: 确认轮结束
[16:00] == 车道B接力: crop_085 探针 ==
[m_screen] 计划 1 个 run (fast=False, 基线 Δ 锚点 0.7177)
  - aug_crop_085 seed0: --aug-params 0.85,1.0,0.0,0.5
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 --workers 6 --seed 0 --checkpoint-dir F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\aug_crop_085_seed0 --aug-params 0.85,1.0,0.0,0.5
[16:19] [laneC] 流水线启动: 队列 10 项, 显存闸门 heavy=9800/fast=5000MiB
[16:19] [laneC] 启动任务 b0fast_seed0(空闲显存 2756MiB)
[16:19] [laneC] 等待显存: b0fast_seed0 需要 5000MiB, 当前空闲 2756MiB (每60s重查)
[16:20] [laneC] 流水线启动: 队列 10 项, 显存闸门 heavy=9800/fast=5000MiB
[16:20] [laneC] 启动任务 b0fast_seed0(空闲显存 2761MiB)
[16:20] [laneC] 等待显存: b0fast_seed0 需要 9800MiB, 当前空闲 2761MiB (每60s重查)
[16:22] [laneD] 流水线启动: 队列 10 项, 显存闸门 heavy=9800/fast=5000MiB
[16:22] [laneD] 启动任务 speed_perturb_seed0(空闲显存 2747MiB)
[16:22] [laneD] 等待显存: speed_perturb_seed0 需要 9800MiB, 当前空闲 2747MiB (每60s重查)
[16:22] [laneD] 流水线启动: 队列 10 项, 显存闸门 heavy=9800/fast=5000MiB
[16:22] [laneD] 启动任务 asym_view_seed0(空闲显存 2747MiB)
[16:22] [laneD] 等待显存: asym_view_seed0 需要 9800MiB, 当前空闲 2747MiB (每60s重查)
[16:23] [laneC] 流水线启动: 队列 10 项, 显存闸门 heavy=9800/fast=5000MiB
[16:23] [laneC] 启动任务 b0fast_seed0(空闲显存 2724MiB)
[16:23] [laneC] 等待显存: b0fast_seed0 需要 7800MiB, 当前空闲 2724MiB (每60s重查)
[16:24] [laneD] 流水线启动: 队列 10 项, 显存闸门 heavy=9800/fast=5000MiB
[16:24] [laneD] 启动任务 speed_perturb_seed0(空闲显存 2751MiB)
[16:24] [laneD] 等待显存: speed_perturb_seed0 需要 7800MiB, 当前空闲 2751MiB (每60s重查)
[16:30] [laneC] 等待显存: b0fast_seed0 需要 9800MiB, 当前空闲 1254MiB (每60s重查)
[16:32] [laneD] 等待显存: speed_perturb_seed0 需要 9800MiB, 当前空闲 1313MiB (每60s重查)
[16:32] [laneD] 等待显存: asym_view_seed0 需要 9800MiB, 当前空闲 1386MiB (每60s重查)
[16:34] [laneD] 等待显存: speed_perturb_seed0 需要 7800MiB, 当前空闲 1387MiB (每60s重查)
[16:43] [laneC] 流水线启动: 队列 10 项, 准入=活跃预训练<3 + 空闲显存≥7800MiB + run_pt 自闸门6500 兜底
[16:43] [laneC] 启动任务 b0fast_seed0(空闲显存 10172MiB, 活跃预训练 2 个)
[18:56] 车道B: crop_085 结束
[18:56] FT10: b0_anchor
[18:56] FT10: crop_weak_s0
[18:56] 车道B: FT10 评估结束
[18:56] == 车道B: batch6(新模块) ==
[m_screen] 计划 4 个 run (fast=False, 基线 Δ 锚点 0.7177)
  - d7_rec01 seed0: --d7-weight 0.1
  - d7_rec003 seed0: --d7-weight 0.03
  - n3_prob03 seed0: --n3-prob 0.3
  - common_view01 seed0: --common-weight 0.1
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 --workers 6 --seed 0 --checkpoint-dir F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\d7_rec01_seed0 --d7-weight 0.1
[19:42] == 车道A2 重建启动 ==
[m_screen] 计划 2 个 run (fast=False, 基线 Δ 锚点 0.7177)
  - proj_dim1024 seed0: --projector 1024-1024-1024
  - proj_dim4096 seed0: --projector 4096-4096-4096
[m_screen] $ C:\Users\admin\.conda\envs\DL\python.exe run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 --workers 6 --seed 0 --checkpoint-dir F:\新实验\ECG_SSL_LFBT-main\checkpoint\M\proj_dim1024_seed0 --projector 1024-1024-1024
[19:44] == 车道B4 重建启动: FT10 重跑 ==
[19:44] FT10: b0_anchor
[19:47] 会话2(本聊天)发现 19:42 暂停指令,已停止误启的恢复车道 A2/B4;当前全机实验进程清零,遵守 ⏸ 状态;恢复走 pipeline_runner.py --lane C
[20:21] [laneC] 流水线启动: 队列 12 项, 准入=活跃预训练<3 + 空闲显存≥7800MiB + run_pt 自闸门6500 兜底
[20:21] [laneC] 启动任务 arb3_base_seed0(空闲显存 19774MiB, 活跃预训练 0 个)
[20:22] [laneD] 流水线启动: 队列 12 项, 准入=活跃预训练<3 + 空闲显存≥7800MiB + run_pt 自闸门6500 兜底
[20:22] [laneD] 启动任务 psfull_arb3_seed0(空闲显存 11038MiB, 活跃预训练 0 个)
[20:25] [laneD] 流水线启动: 队列 12 项, 准入=活跃预训练<3 + 空闲显存≥7800MiB + run_pt 自闸门6500 兜底
[20:25] [laneD] 启动任务 d7_rec01_seed0(空闲显存 11140MiB, 活跃预训练 1 个)
[20:36] [laneE] 流水线启动: 队列 12 项, 准入=活跃预训练<3 + 空闲显存≥7800MiB + run_pt 自闸门6500 兜底
[20:36] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[20:37] [laneE] 流水线启动: 队列 23 项, 准入=活跃预训练<3 + 空闲显存≥7800MiB + run_pt 自闸门6500 兜底
[20:37] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[21:58] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[20:47] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[20:57] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[21:07] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[21:18] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[21:28] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[21:38] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[21:48] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[22:09] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[22:19] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[22:29] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[22:39] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[22:49] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[22:59] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[23:10] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[23:20] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[23:30] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[23:40] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[23:50] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[00:00] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[00:04] [laneC] 下游 LP 解析失败 arb3_base_seed0(值可能已产出, 人工核对)
[00:04] [laneC] arb3_base_seed0 失败, 第 1 次重排队尾
[00:05] [laneC] 启动任务 psfull_arb3_seed0(空闲显存 8058MiB, 活跃预训练 1 个)
[00:11] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[00:21] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[00:31] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[00:41] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[00:51] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[01:01] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[01:12] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[01:22] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[01:32] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[01:42] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[01:52] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[02:03] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[02:13] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[02:23] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[02:33] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[02:44] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[02:54] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[03:04] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[03:14] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[03:24] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[03:35] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[03:45] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[03:55] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[04:05] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[04:15] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[04:25] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[04:36] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[04:46] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[04:56] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[05:06] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[05:16] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[05:26] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[05:37] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[05:47] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[05:57] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[06:07] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[06:17] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[06:27] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[06:38] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[06:48] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[06:54] [laneC] 预训练失败 psfull_arb3_seed0(详见 pt 日志)
[06:54] [laneC] psfull_arb3_seed0 失败, 第 1 次重排队尾
[06:55] [laneC] 启动任务 d7_rec003_seed0(空闲显存 13327MiB, 活跃预训练 1 个)
[06:58] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[07:08] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[07:18] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[07:28] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[07:39] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[07:49] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[07:59] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[08:09] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[08:19] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[08:29] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[08:40] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[08:50] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[09:00] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[09:10] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[09:20] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[09:29] [laneC] ✔ d7_rec003_seed0: AUPRC=0.7158 Δ=-0.0019
[09:29] [laneC] 启动任务 common_view01_seed0(空闲显存 14540MiB, 活跃预训练 1 个)
[09:30] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[09:41] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[09:51] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[10:01] [laneE] 全部剩余任务被他方占用或已认领, 10min 后重查
[10:30] [laneF] 流水线启动: 队列 23 项, 准入=run_pt 并发槽位 + 空闲显存闸门 + run_pt 自闸门6500 兜底
[10:30] [laneF] 全部剩余任务被他方占用或已认领, 10min 后重查
[10:35] [laneG] 流水线启动: 队列 23 项, 准入=run_pt 并发槽位 + 空闲显存闸门 + run_pt 自闸门6500 兜底
[10:35] [laneG] 全部剩余任务被他方占用或已认领, 10min 后重查
[10:41] [laneF] 全部剩余任务被他方占用或已认领, 10min 后重查
[10:45] [laneG] 启动任务 d7_rec01_seed0(空闲显存 14024MiB, 活跃预训练 1 个)
[10:51] [laneF] 全部剩余任务被他方占用或已认领, 10min 后重查
[11:01] [laneF] 全部剩余任务被他方占用或已认领, 10min 后重查
[11:05] [laneF] 流水线启动: 队列 23 项, 准入=run_pt 并发槽位 + 空闲显存闸门 + run_pt 自闸门6500 兜底
[11:05] [laneF] 启动任务 proj_dim1024_seed0(空闲显存 5155MiB, 活跃预训练 2 个)
[11:05] [laneF] 等待显存: proj_dim1024_seed0 需要 7800MiB, 当前空闲 5159MiB (每60s重查)
[11:15] [laneF] 等待显存: proj_dim1024_seed0 需要 7800MiB, 当前空闲 5170MiB (每60s重查)
[11:25] [laneF] 等待显存: proj_dim1024_seed0 需要 7800MiB, 当前空闲 5466MiB (每60s重查)
[11:30] 巡检: 判决线关闭——psfull_arb3 LP 0.7075 vs arb3_base 0.7180, 配对Δ=−1.04pt(AUPRC)❌, E006先验A机不复现, 不补seed; 车道健康(common~130ep/d7重跑/proj1024候闸); 详见 hourly_report.md
[11:35] [laneF] 等待显存: proj_dim1024_seed0 需要 7800MiB, 当前空闲 1843MiB (每60s重查)
[11:45] [laneF] 等待显存: proj_dim1024_seed0 需要 7800MiB, 当前空闲 1827MiB (每60s重查)
[11:55] [laneF] 等待显存: proj_dim1024_seed0 需要 7800MiB, 当前空闲 1888MiB (每60s重查)
[11:59] [laneC] ✔ common_view01_seed0: AUPRC=0.7028 Δ=-0.0149
[11:59] [laneC] 启动任务 proj_dim4096_seed0(空闲显存 5849MiB, 活跃预训练 2 个)
[11:59] [laneC] 等待显存: proj_dim4096_seed0 需要 7800MiB, 当前空闲 5849MiB (每60s重查)
[12:00] 巡检: common_view01 ❌Δ−1.49pt(C2随之不立项); 判决线维持关闭; 双预训练满载(d7_rec01 ep27/proj1024起步,proj4096候闸), 三车道健康, 无事故; 详见 hourly_report.md
[12:09] [laneC] 等待显存: proj_dim4096_seed0 需要 7800MiB, 当前空闲 5842MiB (每60s重查)
[12:19] [laneC] 等待显存: proj_dim4096_seed0 需要 7800MiB, 当前空闲 5842MiB (每60s重查)
[12:29] [laneC] 等待显存: proj_dim4096_seed0 需要 7800MiB, 当前空闲 5842MiB (每60s重查)
[12:39] [laneC] 等待显存: proj_dim4096_seed0 需要 7800MiB, 当前空闲 5842MiB (每60s重查)
[12:49] [laneC] 等待显存: proj_dim4096_seed0 需要 7800MiB, 当前空闲 5842MiB (每60s重查)
[12:59] [laneC] 等待显存: proj_dim4096_seed0 需要 7800MiB, 当前空闲 5841MiB (每60s重查)
[13:09] [laneC] 等待显存: proj_dim4096_seed0 需要 7800MiB, 当前空闲 5841MiB (每60s重查)
[13:19] [laneC] 等待显存: proj_dim4096_seed0 需要 7800MiB, 当前空闲 5842MiB (每60s重查)
[13:29] [laneC] 等待显存: proj_dim4096_seed0 需要 7800MiB, 当前空闲 5712MiB (每60s重查)
[13:39] [laneC] 等待显存: proj_dim4096_seed0 需要 7800MiB, 当前空闲 5864MiB (每60s重查)
[13:49] [laneC] 等待显存: proj_dim4096_seed0 需要 7800MiB, 当前空闲 5864MiB (每60s重查)
[13:59] [laneC] 等待显存: proj_dim4096_seed0 需要 7800MiB, 当前空闲 5859MiB (每60s重查)
[14:09] [laneC] 等待显存: proj_dim4096_seed0 需要 7800MiB, 当前空闲 5857MiB (每60s重查)
[14:19] [laneC] 等待显存: proj_dim4096_seed0 需要 7800MiB, 当前空闲 5815MiB (每60s重查)
[14:28] [laneG] ✔ d7_rec01_seed0: AUPRC=0.7064 Δ=-0.0113
[14:28] [laneG] 启动任务 h2_blur3_seed0(空闲显存 2577MiB, 活跃预训练 2 个)
[14:28] [laneG] 等待显存: h2_blur3_seed0 需要 7800MiB, 当前空闲 2577MiB (每60s重查)
[14:38] [laneG] 等待显存: h2_blur3_seed0 需要 7800MiB, 当前空闲 2322MiB (每60s重查)
[14:48] [laneG] 等待显存: h2_blur3_seed0 需要 7800MiB, 当前空闲 2322MiB (每60s重查)
[14:50] [laneF] ✔ proj_dim1024_seed0: AUPRC=0.7092 Δ=-0.0085
[14:50] [laneF] 启动任务 h2_blur3_seed2(空闲显存 7369MiB, 活跃预训练 1 个)
[14:50] [laneF] 等待显存: h2_blur3_seed2 需要 7800MiB, 当前空闲 7369MiB (每60s重查)
[14:58] [laneG] 等待显存: h2_blur3_seed0 需要 7800MiB, 当前空闲 7369MiB (每60s重查)
[15:00] [laneF] 等待显存: h2_blur3_seed2 需要 7800MiB, 当前空闲 7370MiB (每60s重查)
[15:08] [laneG] 等待显存: h2_blur3_seed0 需要 7800MiB, 当前空闲 7369MiB (每60s重查)
[15:10] [laneF] 等待显存: h2_blur3_seed2 需要 7800MiB, 当前空闲 7364MiB (每60s重查)
[15:18] [laneG] 等待显存: h2_blur3_seed0 需要 7800MiB, 当前空闲 7361MiB (每60s重查)
[15:20] [laneF] 等待显存: h2_blur3_seed2 需要 7800MiB, 当前空闲 7194MiB (每60s重查)
[15:28] [laneG] 等待显存: h2_blur3_seed0 需要 7800MiB, 当前空闲 7335MiB (每60s重查)
[15:30] [laneF] 等待显存: h2_blur3_seed2 需要 7800MiB, 当前空闲 7323MiB (每60s重查)
[15:38] [laneG] 等待显存: h2_blur3_seed0 需要 7800MiB, 当前空闲 7340MiB (每60s重查)
[15:40] [laneF] 等待显存: h2_blur3_seed2 需要 7800MiB, 当前空闲 7324MiB (每60s重查)
[15:48] [laneG] 等待显存: h2_blur3_seed0 需要 7800MiB, 当前空闲 7320MiB (每60s重查)
[15:50] [laneF] 等待显存: h2_blur3_seed2 需要 7800MiB, 当前空闲 7318MiB (每60s重查)
[15:58] [laneG] 等待显存: h2_blur3_seed0 需要 7800MiB, 当前空闲 7313MiB (每60s重查)
[16:00] [laneF] 等待显存: h2_blur3_seed2 需要 7800MiB, 当前空闲 7334MiB (每60s重查)
[16:02] [laneC] 预训练失败 proj_dim4096_seed0(详见 pt 日志)
[16:02] [laneC] proj_dim4096_seed0 失败, 第 1 次重排队尾
[16:03] [laneC] 启动任务 speed_perturb_seed0(空闲显存 12311MiB, 活跃预训练 2 个)
[16:04] [laneC] 预训练失败 speed_perturb_seed0(详见 pt 日志)
[16:04] [laneC] speed_perturb_seed0 失败, 第 1 次重排队尾
[16:10] 巡检更正: proj4096系挂死(CPU冻结/98%util系僵尸空转)已击杀; h2_blur3 s0+s2入场; laneC抢入的speed_perturb让位清除; 双探针17.7/24.5健康
[16:05] [laneC] 启动任务 asym_view_seed0(空闲显存 5276MiB, 活跃预训练 2 个)
[16:05] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 5276MiB (每60s重查)
[16:15] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 5280MiB (每60s重查)
[16:25] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 5276MiB (每60s重查)
[16:35] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 5151MiB (每60s重查)
[16:45] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 5170MiB (每60s重查)
[16:55] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 5101MiB (每60s重查)
[17:00] 巡检: 无新判定; h2_blur3 s0/s2 ep27/200(131s/ep, ETA~23:20)双探针健康(CPU验证); asym_view候闸; 无事故
[17:05] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 5099MiB (每60s重查)
[17:15] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 5096MiB (每60s重查)
[17:25] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 5103MiB (每60s重查)
[17:35] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 5108MiB (每60s重查)
[17:45] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 5136MiB (每60s重查)
[17:55] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 5136MiB (每60s重查)
[18:00] 巡检: 无新判定; h2 s0/s2 ep91/200提速(56s/ep, ETA提前~19:40), CPU验证双活; 无事故
[18:05] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 5162MiB (每60s重查)
[18:15] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 5153MiB (每60s重查)
[18:25] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 5162MiB (每60s重查)
[18:35] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 5155MiB (每60s重查)
[18:45] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 5159MiB (每60s重查)
[18:55] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 5130MiB (每60s重查)
[18:59] [laneG] ✔ h2_blur3_seed0: AUPRC=0.7085 Δ=-0.0092
[18:59] [laneG] 启动任务 proj_dim4096_seed0(空闲显存 12939MiB, 活跃预训练 1 个)
[19:00] [laneF] ✔ h2_blur3_seed2: AUPRC=0.7217 Δ=+0.0040
[19:00] [laneF] 启动任务 h2_blur3_seed4(空闲显存 1185MiB, 活跃预训练 2 个)
[19:00] [laneF] 等待显存: h2_blur3_seed4 需要 7800MiB, 当前空闲 1189MiB (每60s重查)
[19:01] [laneC] 预训练失败 asym_view_seed0(详见 pt 日志)
[19:01] [laneC] asym_view_seed0 失败, 第 1 次重排队尾
[19:02] [laneC] 启动任务 lead_swap02_seed0(空闲显存 7536MiB, 活跃预训练 1 个)
[19:02] [laneC] 等待显存: lead_swap02_seed0 需要 7800MiB, 当前空闲 7536MiB (每60s重查)
[19:02] [laneG2] 流水线启动: 队列 23 项, 准入=run_pt 并发槽位 + 空闲显存闸门 + run_pt 自闸门6500 兜底
[19:02] [laneG2] 启动任务 t3_pow3_seed0(空闲显存 19918MiB, 活跃预训练 0 个)
[19:05] 巡检: h2分裂数据(s0❌−0.92/s2✅+0.40,A-P2首正信号!), seed4决胜票已开跑(~22:00); proj4096让位挪独占窗口; t3_s0并行
[19:12] [laneC] 等待显存: lead_swap02_seed0 需要 7800MiB, 当前空闲 5726MiB (每60s重查)
[19:22] [laneC] 等待显存: lead_swap02_seed0 需要 7800MiB, 当前空闲 5726MiB (每60s重查)
[19:32] [laneC] 等待显存: lead_swap02_seed0 需要 7800MiB, 当前空闲 5731MiB (每60s重查)
[19:42] [laneC] 等待显存: lead_swap02_seed0 需要 7800MiB, 当前空闲 5731MiB (每60s重查)
[19:52] [laneC] 等待显存: lead_swap02_seed0 需要 7800MiB, 当前空闲 5726MiB (每60s重查)
[20:00] 巡检: 无新判定; h2_s4决胜票+t3_s0双跑(ep27/200, ETA~23:20); CPU双活; 无事故
[20:02] [laneC] 等待显存: lead_swap02_seed0 需要 7800MiB, 当前空闲 5728MiB (每60s重查)
[20:12] [laneC] 等待显存: lead_swap02_seed0 需要 7800MiB, 当前空闲 5700MiB (每60s重查)
[20:22] [laneC] 等待显存: lead_swap02_seed0 需要 7800MiB, 当前空闲 5686MiB (每60s重查)
[20:32] [laneC] 等待显存: lead_swap02_seed0 需要 7800MiB, 当前空闲 5700MiB (每60s重查)
[20:42] [laneC] 等待显存: lead_swap02_seed0 需要 7800MiB, 当前空闲 5682MiB (每60s重查)
[20:52] [laneC] 等待显存: lead_swap02_seed0 需要 7800MiB, 当前空闲 5727MiB (每60s重查)
[21:02] [laneC] 等待显存: lead_swap02_seed0 需要 7800MiB, 当前空闲 5724MiB (每60s重查)
[21:12] [laneC] 等待显存: lead_swap02_seed0 需要 7800MiB, 当前空闲 5718MiB (每60s重查)
[21:22] [laneC] 等待显存: lead_swap02_seed0 需要 7800MiB, 当前空闲 5722MiB (每60s重查)
[21:32] [laneC] 等待显存: lead_swap02_seed0 需要 7800MiB, 当前空闲 5448MiB (每60s重查)
[21:42] [laneC] 等待显存: lead_swap02_seed0 需要 7800MiB, 当前空闲 5452MiB (每60s重查)
[21:48] [laneG2] LP 失败 t3_pow3_seed0
[21:48] [laneG2] t3_pow3_seed0 失败, 第 1 次重试插队首
[21:49] [laneF] ✔ h2_blur3_seed4: AUPRC=0.7149 Δ=-0.0028
[21:49] [laneF] 启动任务 t3_pow3_seed2(空闲显存 12926MiB, 活跃预训练 1 个)
[21:49] [laneG2] 启动任务 t3_pow3_seed0(空闲显存 6226MiB, 活跃预训练 2 个)
[21:49] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6226MiB (每60s重查)
[21:59] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6128MiB (每60s重查)
[22:09] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6121MiB (每60s重查)
[22:19] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6121MiB (每60s重查)
[22:29] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6129MiB (每60s重查)
[22:39] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6120MiB (每60s重查)
[22:49] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6134MiB (每60s重查)
[22:59] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6128MiB (每60s重查)
[23:09] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6106MiB (每60s重查)
[23:19] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6097MiB (每60s重查)
[23:29] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6106MiB (每60s重查)
[23:39] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6104MiB (每60s重查)
[23:49] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6104MiB (每60s重查)
[23:59] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6107MiB (每60s重查)
[00:09] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6106MiB (每60s重查)
[00:19] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6106MiB (每60s重查)
[00:29] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6092MiB (每60s重查)
[00:39] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6104MiB (每60s重查)
[00:46] [laneC] ✔ lead_swap02_seed0: AUPRC=0.7109 Δ=-0.0068
[00:46] [laneC] 启动任务 cautious_adam_seed0(空闲显存 6038MiB, 活跃预训练 2 个)
[00:46] [laneC] 等待显存: cautious_adam_seed0 需要 7800MiB, 当前空闲 6038MiB (每60s重查)
[00:53] [laneF] LP 失败 t3_pow3_seed2
[00:53] [laneF] t3_pow3_seed2 失败, 第 1 次重试插队首
[00:54] [laneF] 启动任务 t3_pow3_seed2(空闲显存 6106MiB, 活跃预训练 2 个)
[00:54] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6106MiB (每60s重查)
[01:04] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6105MiB (每60s重查)
[01:14] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6170MiB (每60s重查)
[01:24] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6169MiB (每60s重查)
[01:34] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6169MiB (每60s重查)
[01:44] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6160MiB (每60s重查)
[01:54] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6169MiB (每60s重查)
[02:04] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6169MiB (每60s重查)
[02:14] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6169MiB (每60s重查)
[02:24] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6169MiB (每60s重查)
[02:34] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6168MiB (每60s重查)
[02:44] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6164MiB (每60s重查)
[02:54] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6164MiB (每60s重查)
[03:04] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6164MiB (每60s重查)
[03:14] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6164MiB (每60s重查)
[03:24] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6164MiB (每60s重查)
[03:34] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6164MiB (每60s重查)
[03:41] [laneG2] LP 失败 t3_pow3_seed0
[03:41] [laneG2] t3_pow3_seed0 失败, 第 2 次重试插队首
[03:42] [laneG2] 启动任务 t3_pow3_seed0(空闲显存 6162MiB, 活跃预训练 2 个)
[03:42] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6162MiB (每60s重查)
[03:52] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6166MiB (每60s重查)
[04:02] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6170MiB (每60s重查)
[04:12] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6164MiB (每60s重查)
[04:22] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6170MiB (每60s重查)
[04:32] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6171MiB (每60s重查)
[04:42] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6170MiB (每60s重查)
[04:52] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6170MiB (每60s重查)
[05:02] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6169MiB (每60s重查)
[05:12] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6170MiB (每60s重查)
[05:22] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6164MiB (每60s重查)
[05:32] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6164MiB (每60s重查)
[05:42] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6166MiB (每60s重查)
[05:52] [laneG2] 等待显存: t3_pow3_seed0 需要 7800MiB, 当前空闲 6156MiB (每60s重查)
[06:03] [laneC] ✔ cautious_adam_seed0: AUPRC=0.6316 Δ=-0.0861
[06:03] [laneC] 启动任务 b0fast_seed0(空闲显存 6092MiB, 活跃预训练 2 个)
[06:03] [laneC] 等待显存: b0fast_seed0 需要 7800MiB, 当前空闲 6094MiB (每60s重查)
[06:13] [laneC] 等待显存: b0fast_seed0 需要 7800MiB, 当前空闲 6094MiB (每60s重查)
[06:23] [laneC] 等待显存: b0fast_seed0 需要 7800MiB, 当前空闲 6102MiB (每60s重查)
[06:33] [laneC] 等待显存: b0fast_seed0 需要 7800MiB, 当前空闲 6102MiB (每60s重查)
[06:36] [laneC] 预训练失败 b0fast_seed0(详见 pt 日志)
[06:36] [laneC] b0fast_seed0 失败, 第 1 次重排队尾
[06:37] [laneC] 启动任务 speed_perturb_seed0(空闲显存 12429MiB, 活跃预训练 1 个)
[06:37] [laneF] LP 失败 t3_pow3_seed2
[06:37] [laneF] t3_pow3_seed2 失败, 第 2 次重试插队首
[06:38] [laneF] 启动任务 t3_pow3_seed2(空闲显存 6164MiB, 活跃预训练 2 个)
[06:38] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6164MiB (每60s重查)
[06:48] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6157MiB (每60s重查)
[06:58] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6164MiB (每60s重查)
[07:08] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6159MiB (每60s重查)
[07:18] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6159MiB (每60s重查)
[07:28] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6164MiB (每60s重查)
[07:38] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6162MiB (每60s重查)
[07:48] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6170MiB (每60s重查)
[07:58] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6162MiB (每60s重查)
[08:08] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6154MiB (每60s重查)
[08:18] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6156MiB (每60s重查)
[08:28] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6156MiB (每60s重查)
[08:38] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6159MiB (每60s重查)
[08:48] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6164MiB (每60s重查)
[08:58] [laneF] 等待显存: t3_pow3_seed2 需要 7800MiB, 当前空闲 6165MiB (每60s重查)
[09:05] [laneG2] LP 失败 t3_pow3_seed0
[09:05] [laneG2] t3_pow3_seed0 连续 3 次失败, 弃置待人工排查
[09:05] [laneG2] 启动任务 t3_pow3_seed4(空闲显存 6164MiB, 活跃预训练 2 个)
[09:05] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 6164MiB (每60s重查)
[09:15] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 6164MiB (每60s重查)
[09:25] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 6144MiB (每60s重查)
[09:35] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 6145MiB (每60s重查)
[09:37] [laneC] ✔ speed_perturb_seed0: AUPRC=0.7108 Δ=-0.0069
[09:37] [laneC] 启动任务 asym_view_seed0(空闲显存 6084MiB, 活跃预训练 2 个)
[09:37] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 6084MiB (每60s重查)
[09:47] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 6076MiB (每60s重查)
[09:57] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 6076MiB (每60s重查)
[10:07] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 6075MiB (每60s重查)
[10:17] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 6084MiB (每60s重查)
[10:27] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 6084MiB (每60s重查)
[10:37] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 6076MiB (每60s重查)
[10:47] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 6076MiB (每60s重查)
[10:57] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 6084MiB (每60s重查)
[11:07] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 6083MiB (每60s重查)
[11:17] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 6076MiB (每60s重查)
[11:27] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 6075MiB (每60s重查)
[11:37] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 6076MiB (每60s重查)
[11:47] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 6075MiB (每60s重查)
[11:57] [laneC] 等待显存: asym_view_seed0 需要 7800MiB, 当前空闲 6075MiB (每60s重查)
[12:02] [laneF] LP 失败 t3_pow3_seed2
[12:02] [laneF] t3_pow3_seed2 连续 3 次失败, 弃置待人工排查
[12:02] [laneF] 启动任务 proj_dim4096_seed0(空闲显存 6136MiB, 活跃预训练 2 个)
[12:02] [laneF] 等待显存: proj_dim4096_seed0 需要 7800MiB, 当前空闲 6136MiB (每60s重查)
[12:12] [laneF] 等待显存: proj_dim4096_seed0 需要 7800MiB, 当前空闲 6144MiB (每60s重查)
[12:22] [laneF] 等待显存: proj_dim4096_seed0 需要 7800MiB, 当前空闲 6144MiB (每60s重查)
[12:32] [laneF] 等待显存: proj_dim4096_seed0 需要 7800MiB, 当前空闲 6145MiB (每60s重查)
[12:34] [laneG2] LP 失败 t3_pow3_seed4
[12:34] [laneG2] t3_pow3_seed4 失败, 第 1 次重试插队首
[12:35] [laneG2] 启动任务 t3_pow3_seed4(空闲显存 646MiB, 活跃预训练 2 个)
[12:35] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 646MiB (每60s重查)
[12:45] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 651MiB (每60s重查)
[12:55] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 651MiB (每60s重查)
[13:05] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 643MiB (每60s重查)
[13:15] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 643MiB (每60s重查)
[13:25] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 638MiB (每60s重查)
[13:35] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 643MiB (每60s重查)
[13:45] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 637MiB (每60s重查)
[13:55] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 643MiB (每60s重查)
[14:05] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 643MiB (每60s重查)
[14:15] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 643MiB (每60s重查)
[14:25] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 669MiB (每60s重查)
[14:35] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 656MiB (每60s重查)
[14:45] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 670MiB (每60s重查)
[14:55] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 598MiB (每60s重查)
[15:03] [laneC] ✔ asym_view_seed0: AUPRC=0.7053 Δ=-0.0124
[15:03] [laneC] 启动任务 b0fast_seed0(空闲显存 7333MiB, 活跃预训练 1 个)
[15:03] [laneC] 等待显存: b0fast_seed0 需要 7800MiB, 当前空闲 7333MiB (每60s重查)
[15:05] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 7326MiB (每60s重查)
[15:13] [laneC] 等待显存: b0fast_seed0 需要 7800MiB, 当前空闲 7325MiB (每60s重查)
[15:15] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 7325MiB (每60s重查)
[15:23] [laneC] 等待显存: b0fast_seed0 需要 7800MiB, 当前空闲 7325MiB (每60s重查)
[15:25] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 7317MiB (每60s重查)
[15:33] [laneC] 等待显存: b0fast_seed0 需要 7800MiB, 当前空闲 7331MiB (每60s重查)
[15:35] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 7317MiB (每60s重查)
[15:43] [laneC] 等待显存: b0fast_seed0 需要 7800MiB, 当前空闲 7325MiB (每60s重查)
[15:45] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 7317MiB (每60s重查)
[15:53] [laneC] 等待显存: b0fast_seed0 需要 7800MiB, 当前空闲 7331MiB (每60s重查)
[15:55] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 7317MiB (每60s重查)
[16:03] [laneC] 等待显存: b0fast_seed0 需要 7800MiB, 当前空闲 7317MiB (每60s重查)
[16:05] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 7317MiB (每60s重查)
[16:13] [laneC] 等待显存: b0fast_seed0 需要 7800MiB, 当前空闲 7326MiB (每60s重查)
[16:15] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 7317MiB (每60s重查)
[16:23] [laneC] 等待显存: b0fast_seed0 需要 7800MiB, 当前空闲 7326MiB (每60s重查)
[16:25] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 7317MiB (每60s重查)
[16:33] [laneC] 等待显存: b0fast_seed0 需要 7800MiB, 当前空闲 7317MiB (每60s重查)
[16:35] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 7317MiB (每60s重查)
[16:43] [laneC] 等待显存: b0fast_seed0 需要 7800MiB, 当前空闲 7325MiB (每60s重查)
[16:45] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 7317MiB (每60s重查)
[16:53] [laneC] 等待显存: b0fast_seed0 需要 7800MiB, 当前空闲 7325MiB (每60s重查)
[16:55] [laneG2] 等待显存: t3_pow3_seed4 需要 7800MiB, 当前空闲 7317MiB (每60s重查)
[17:00] [laneC] 预训练失败 b0fast_seed0(详见 pt 日志)
[17:00] [laneC] b0fast_seed0 失败, 第 2 次重排队尾
[17:01] [laneC] 启动任务 b0fast_seed0(空闲显存 12547MiB, 活跃预训练 1 个)
[17:01] [laneF] ✔ proj_dim4096_seed0: AUPRC=0.7105 Δ=-0.0072
[17:01] [laneF] 启动任务 t3_pow3_seed0(空闲显存 13035MiB, 活跃预训练 2 个)
[17:01] [laneC] 预训练失败 b0fast_seed0(详见 pt 日志)
[17:01] [laneC] b0fast_seed0 连续 3 次失败, 弃置待人工排查
[17:01] [laneC] 流水线退出: 队列清空/全部完成
[19:49] [laneG2] LP 失败 t3_pow3_seed4
[19:49] [laneG2] t3_pow3_seed4 失败, 第 2 次重试插队首
[19:50] [laneF] LP 失败 t3_pow3_seed0
[19:50] [laneF] t3_pow3_seed0 失败, 第 1 次重试插队首
[19:50] [laneG2] 启动任务 t3_pow3_seed4(空闲显存 19727MiB, 活跃预训练 0 个)
[19:51] [laneF] 启动任务 t3_pow3_seed0(空闲显存 13035MiB, 活跃预训练 1 个)
[22:44] [laneG2] LP 失败 t3_pow3_seed4
[22:44] [laneG2] t3_pow3_seed4 连续 3 次失败, 弃置待人工排查
[22:44] [laneG2] 启动任务 t3_pow3_seed2(空闲显存 19320MiB, 活跃预训练 0 个)
[22:45] [laneF] LP 失败 t3_pow3_seed0
[22:45] [laneF] t3_pow3_seed0 失败, 第 2 次重试插队首
[22:46] [laneF] 启动任务 t3_pow3_seed0(空闲显存 13003MiB, 活跃预训练 1 个)
[01:20] [laneG2] LP 失败 t3_pow3_seed2
[01:20] [laneG2] t3_pow3_seed2 失败, 第 1 次重试插队首
[01:21] [laneG2] 启动任务 t3_pow3_seed2(空闲显存 19313MiB, 活跃预训练 0 个)
[01:22] [laneF] LP 失败 t3_pow3_seed0
[01:22] [laneF] t3_pow3_seed0 连续 3 次失败, 弃置待人工排查
[01:22] [laneF] 启动任务 t3_pow3_seed4(空闲显存 13093MiB, 活跃预训练 1 个)
[04:09] [laneG2] LP 失败 t3_pow3_seed2
[04:09] [laneG2] t3_pow3_seed2 失败, 第 2 次重试插队首
[04:10] [laneF] LP 失败 t3_pow3_seed4
[04:10] [laneF] t3_pow3_seed4 失败, 第 1 次重试插队首
[04:10] [laneG2] 启动任务 t3_pow3_seed2(空闲显存 19779MiB, 活跃预训练 0 个)
[04:11] [laneF] 启动任务 t3_pow3_seed4(空闲显存 13093MiB, 活跃预训练 1 个)
[07:09] [laneG2] LP 失败 t3_pow3_seed2
[07:09] [laneG2] t3_pow3_seed2 连续 3 次失败, 弃置待人工排查
[07:09] [laneG2] 启动任务 h1_white8_seed0(空闲显存 19308MiB, 活跃预训练 0 个)
[07:10] [laneF] LP 失败 t3_pow3_seed4
[07:10] [laneF] t3_pow3_seed4 失败, 第 2 次重试插队首
[07:11] [laneF] 启动任务 t3_pow3_seed4(空闲显存 13016MiB, 活跃预训练 1 个)
[09:52] [laneF] LP 失败 t3_pow3_seed4
[09:52] [laneF] t3_pow3_seed4 连续 3 次失败, 弃置待人工排查
[09:52] [laneF] 启动任务 h1_white8_seed2(空闲显存 13025MiB, 活跃预训练 1 个)
[09:55] [laneG2] ✔ h1_white8_seed0: AUPRC=0.7021 Δ=-0.0156
[09:55] [laneG2] 启动任务 h1_white8_seed4(空闲显存 13025MiB, 活跃预训练 1 个)
[12:51] [laneF] ✔ h1_white8_seed2: AUPRC=0.7074 Δ=-0.0103
[12:51] [laneF] 启动任务 h1_ln_neg8_seed0(空闲显存 12998MiB, 活跃预训练 1 个)
[12:53] [laneG2] ✔ h1_white8_seed4: AUPRC=0.7026 Δ=-0.0151
[12:53] [laneG2] 启动任务 h1_pos_neg8_seed0(空闲显存 13031MiB, 活跃预训练 1 个)
[15:28] [laneF] ✔ h1_ln_neg8_seed0: AUPRC=0.6990 Δ=-0.0187
[15:28] [laneF] 启动任务 b0fast_seed0(空闲显存 13061MiB, 活跃预训练 1 个)
[15:28] [laneF] 预训练失败 b0fast_seed0(详见 pt 日志)
[15:28] [laneF] b0fast_seed0 失败, 第 1 次重试插队首
[15:29] [laneF] 启动任务 b0fast_seed0(空闲显存 13061MiB, 活跃预训练 1 个)
[15:29] [laneF] 预训练失败 b0fast_seed0(详见 pt 日志)
[15:29] [laneF] b0fast_seed0 失败, 第 2 次重试插队首
[15:30] [laneF] 启动任务 b0fast_seed0(空闲显存 13062MiB, 活跃预训练 1 个)
[15:31] [laneF] 预训练失败 b0fast_seed0(详见 pt 日志)
[15:31] [laneF] b0fast_seed0 连续 3 次失败, 弃置待人工排查
[15:31] [laneF] 全部剩余任务被他方占用或已认领, 10min 后重查
[15:41] [laneF] 全部剩余任务被他方占用或已认领, 10min 后重查
[15:44] [laneG2] ✔ h1_pos_neg8_seed0: AUPRC=0.6951 Δ=-0.0226
[15:45] [laneG2] 启动任务 b0fast_seed0(空闲显存 19855MiB, 活跃预训练 0 个)
[15:45] [laneG2] 预训练失败 b0fast_seed0(详见 pt 日志)
[15:45] [laneG2] b0fast_seed0 失败, 第 1 次重试插队首
[15:46] [laneG2] 启动任务 b0fast_seed0(空闲显存 19855MiB, 活跃预训练 0 个)
[15:46] [laneG2] 预训练失败 b0fast_seed0(详见 pt 日志)
[15:46] [laneG2] b0fast_seed0 失败, 第 2 次重试插队首
[15:47] [laneG2] 启动任务 b0fast_seed0(空闲显存 19847MiB, 活跃预训练 0 个)
[15:47] [laneG2] 预训练失败 b0fast_seed0(详见 pt 日志)
[15:47] [laneG2] b0fast_seed0 连续 3 次失败, 弃置待人工排查
[15:47] [laneG2] 流水线退出: 队列清空/全部完成
[15:51] [laneF] 流水线退出: 队列清空/全部完成
[19:30] [laneF] 流水线启动: 队列 23 项, 准入=run_pt 并发槽位 + 空闲显存闸门 + run_pt 自闸门6500 兜底
[19:30] [laneF] 启动任务 b0fast_seed0(空闲显存 19858MiB, 活跃预训练 0 个)
[21:55] [laneF] ✔ b0fast_seed0: AUPRC=0.7055 Δ=-0.0122
[21:55] [laneF] 流水线退出: 队列清空/全部完成
[21:57] [laneF] 流水线启动: 队列 26 项, 准入=run_pt 并发槽位 + 空闲显存闸门 + run_pt 自闸门6500 兜底
[21:57] [laneF] 启动任务 a1_d1l_fix_seed0(空闲显存 13253MiB, 活跃预训练 1 个)
[21:57] [laneG2] 流水线启动: 队列 26 项, 准入=run_pt 并发槽位 + 空闲显存闸门 + run_pt 自闸门6500 兜底
[21:57] [laneG2] 启动任务 a2_acl_intra_seed0(空闲显存 12909MiB, 活跃预训练 2 个)
[22:36] [laneC] 流水线启动: 队列 26 项, 准入=run_pt 并发槽位 + 空闲显存闸门 + run_pt 自闸门6500 兜底
[22:36] [laneC] 启动任务 a3_acl_inter_seed0(空闲显存 7938MiB, 活跃预训练 2 个)
[22:40] [laneG2] 流水线启动: 队列 26 项, 准入=run_pt 并发槽位 + 空闲显存闸门 + run_pt 自闸门6500 兜底
[22:40] [laneG2] 启动任务 a2_acl_intra_seed0(空闲显存 13154MiB, 活跃预训练 1 个)
[22:40] [laneC] 流水线启动: 队列 26 项, 准入=run_pt 并发槽位 + 空闲显存闸门 + run_pt 自闸门6500 兜底
[22:40] [laneC] 启动任务 a3_acl_inter_seed0(空闲显存 12551MiB, 活跃预训练 2 个)
[00:02] [laneF] 预训练失败 a1_d1l_fix_seed0(详见 pt 日志)
[00:02] [laneF] a1_d1l_fix_seed0 失败, 第 1 次重试插队首
[00:02] [laneC] 预训练失败 a3_acl_inter_seed0(详见 pt 日志)
[00:02] [laneC] a3_acl_inter_seed0 失败, 第 1 次重试插队首
[00:02] [laneG2] 预训练失败 a2_acl_intra_seed0(详见 pt 日志)
[00:02] [laneG2] a2_acl_intra_seed0 失败, 第 1 次重试插队首
[02:44] [laneF] 流水线启动: 队列 26 项, 准入=run_pt 并发槽位 + 空闲显存闸门 + run_pt 自闸门6500 兜底
[02:44] [laneF] 启动任务 a1_d1l_fix_seed0(空闲显存 23467MiB, 活跃预训练 0 个)
[02:44] [laneG2] 流水线启动: 队列 26 项, 准入=run_pt 并发槽位 + 空闲显存闸门 + run_pt 自闸门6500 兜底
[02:44] [laneG2] 启动任务 a2_acl_intra_seed0(空闲显存 23482MiB, 活跃预训练 1 个)
[02:44] [laneC] 流水线启动: 队列 26 项, 准入=run_pt 并发槽位 + 空闲显存闸门 + run_pt 自闸门6500 兜底
[02:44] [laneC] 启动任务 a3_acl_inter_seed0(空闲显存 23493MiB, 活跃预训练 2 个)
[04:35] [laneC] ✔ a3_acl_inter_seed0: AUPRC=0.6994 Δ=-0.0183
[04:35] [laneG2] ✔ a2_acl_intra_seed0: AUPRC=0.6266 Δ=-0.0911
[04:35] [laneC] 全部剩余任务被他方占用或已认领, 10min 后重查
[04:35] [laneG2] 全部剩余任务被他方占用或已认领, 10min 后重查
