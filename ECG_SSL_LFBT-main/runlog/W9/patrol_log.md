# W9 值班巡检日志(patrol_log)

## 2026-10-19 22:00 值班#1(首轮)
- 判活: **链已停**(run.pid 空, chain 20:51 退出, GPU 1345MiB 桌面占用)。
- 进度: 0/30 ok; clocs_cpsc_s0(20:00-20:41)/s2(20:10-20:51)两次 FT10 本身正常跑完(rc=0 产物完整), 但车道停止于同一条校验断言。
- 定因取证: 落盘 y_true/y_prob 独立复算与 metrics.json 逐位一致(0.000e+00); 断言 |metrics_ext.macro_ap−metrics.auprc|<1e-6 属**校验器假阳性**——macro_ap 为无ties处理手写 AP, 与 sklearn 固有差 ~1.05e-5(两任务同量级), 非实验失败。
- 动作: ①runner/finalize 校验改为"预测独立复算==metrics.json @1e-6"(任务书§七-5 本意)+macro_ap 1e-3 粗检; ②加 try_revalidate 复验路径(§五-1 工件齐全且一致可跳过); ③commit b671e68 push; ④22:02:51 launch_w9.ps1 重启(pid 49568)。
- 重启核验: L0 已 REVALIDATED clocs_cpsc_s0(auroc 0.9112449671/auprc 0.6684586504, 免重训)并 START clocs_cpsc_s4; L1 错峰 600s 后将复验 s2。
- 实验协议零改动(超参/矩阵/种子/数据未动); events.md 已同步留痕。
- ETA 修正: 28 新跑 × ~35min 双车道 ≈ 次日 06:00-07:00 收链。

## 2026-10-20 00:00 值班#2
- 判活: pid 49568 活(tasklist 直查), GPU 13%/6193MiB 双车道满载, chain.log 正常推进。
- 进度: **7/30 ok**(clocs-cpsc×3 + simclr-chapman×3 + simph-chapman-s0); L0 跑 simph_chapman_s4, L1 跑 simph_chapman_s2(23:30 起)。单次 ~37-39min 与预估一致。
- 动作: 无需重启; 推送新账行(7 ok)。ETA 维持 10-20 晨 ~07:00。
