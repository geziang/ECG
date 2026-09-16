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
