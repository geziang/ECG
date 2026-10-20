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

## 2026-10-20 02:00 值班#3
- 判活: pid 49568 活, GPU 6201MiB 双车道在载(瞬时 util 0% 属 epoch 间隙, 账行 2h+6 与 wall ~2320s 稳定互证)。
- 进度: **13/30 ok**——P0: clocs-cpsc×3/simclr-chap×3/simph-chap×3/clocs-chap×3/tsr-chap-s0 已入账; L0=tsr_chapman_s4, L1=tsr_chapman_s2 在跑, 完成后 P0=15/15 收官, 进 P1。
- 动作: 无需重启; 推送新账行(13 ok)。ETA ~07:20-07:40。

## 2026-10-20 04:00 值班#4
- 判活: pid 49568 活, GPU 6164MiB 双车道在载, 账行 2h+6 稳定。
- 进度: **19/30 ok**——P0=15/15 收官(tsr-chap s2/s4 于 02:32 前完成); P1: m2-cpsc×3+m2-chap-s0 入账, L0=m2_chapman_s4, L1=m2_chapman_s2 在跑。余 m5×3域=9 次 + 在跑 2 次。
- 动作: 无需重启; 推送新账行(19 ok)。ETA ~07:30。

## 2026-10-20 06:00 值班#5
- 判活: pid 49568 活, GPU 6164MiB 双车道在载, 账行 2h+7。
- 进度: **26/30 ok**——m2-chap×3+m5-cpsc×3+m5-chap-s0/s2 入账; 在跑: L0=m5_chapman_s4, L1=m5_ptbxl_s0。收尾 4 次预计 08:10-08:20 全清。
- 动作: 无需重启; 推送新账行(26 ok)。30/30 后下轮值班跑 finalize 七门验收。

## 2026-10-20 08:00 值班#6(收官轮)
- 判活: 链已于 07:13:40 干净退出(run.pid 清空, GPU 回落桌面占用), **30/30 ok**。
- 动作: 执行 finalize 七门验收 → **ALL GATES PASS**(门1 30行唯一键/门2 10格恰{0,2,4}/门3 30套产物/门4-6 预测逐位复算一致+元数据与preflight一致/门7 18格全3seed); 产出 summary.csv(36行)+table4_complete.md(18格全满)+W9_最终报告.md(含验收结论); HOSTS.md 主机A状态改 ✅ W9 完成。
- 本批收官: 总 wall 19.39h(双车道有效~10h), 全程两次门禁拦截(clocs-s0 SHA 笔误/校验假阳性)均为主机A侧问题并闭环留痕, 实验协议零改动。待办公机§八论文回填。**W9 值班使命完成, 值班 cron 可删除。**

## 2026-10-20 10:00 值班#7(收官后核验轮)
- 状态: 批次已收官(08:0x, 85e9dbc)——30/30 ok、报告含 ALL GATES PASS 结论、HOSTS ✅、tracked files 干净, 状态与收官时一致, 无新账行。
- 动作: 无(规则5条件不成立: 报告非草稿且含门结论, 不重跑 finalize; 无重启/无 commit)。
- **本 cron 使命已完成, 建议用户侧删除 automation-fdf0c2c0; 后续每2h触发将仅做本类空核验。**

## 2026-10-20 12:00 值班#8(收官后核验轮)
- 状态: 与收官时一致(30/30、报告含 ALL GATES PASS、链已退出、git 与 origin/main 同步), 零动作。
- **cron automation-fdf0c2c0 使命已完成, 再次提醒用户侧删除; 未删除前每 2h 触发均为本类空核验。**

## 2026-10-20 14:00 值班#9(收官后核验轮)
- 状态与收官时一致(30/30+ALL GATES PASS+链退出+git同步), 零动作。cron 待用户侧删除。
