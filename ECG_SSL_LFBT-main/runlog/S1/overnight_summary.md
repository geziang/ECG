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
