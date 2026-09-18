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
