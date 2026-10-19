
## 2026-10-19 19:56 preflight FAIL: clocs s0 SHA c1f139d98059812aeee6453c5c7ec9cb2e99d57f338178833cd6e0cd159574d5 != 冻结 c1f139d98059812aeee6453c5c7ec9cb2e99d57f338178833cd4e0cd159574d5

## 2026-10-19 20:0x preflight 首轮 FAIL 定因=主机A自身誊写笔误(非任务书/非文件篡改)
- 现象: clocs s0 SHA 盘上 `...cd6e0cd...` vs runner 冻结表 `...cd4e0cd...`(单字符差),其余 17 项+三域 split SHA 全过。
- 取证: W4/baseline_results.csv(09-23 入档)clocs s0 四次在案(lp-cpsc/ft10-ptbxl/lp-ptbxl/lp-chapman)均=`...cd6e0cd...`;任务书§4.3 原文(git diff 证明未被改动)亦=`...cd6e0cd...`;clocs s2/s4 两侧一致。
- 结论: 笔误在 runner 的 SHA_FROZEN 常量誊写(6→4),任务书与 checkpoint 文件均正确。
- 处置: 修正 runner 常量;任务书保持原样零改动;曾误加的勘误注记已撤回。preflight 门按设计拦住了不一致,闭环。

## 2026-10-19 19:58 preflight PASS (git 3cfb5e7, GPU ok, split SHA 3/3, ckpt SHA 18/18, config 语义 18/18, 矩阵 30 次)

## 2026-10-19 19:59 smoke PASS (10/10 组合, 未触碰 test)

## 2026-10-19 20:00 run 启动: 30 任务/2 车道/错峰600s (任务书27 + 用户裁决扩充m5-ptbxl×3)

## 2026-10-19 20:41 clocs_cpsc_s0 产物校验失败(ValueError: metrics_ext.macro_ap 与 metrics.auprc 不一致(>1e-6)), 车道0停止

## 2026-10-19 20:51 clocs_cpsc_s2 产物校验失败(ValueError: metrics_ext.macro_ap 与 metrics.auprc 不一致(>1e-6)), 车道1停止

## 2026-10-19 20:51 run 结束: 0/30 ok, 停止原因: clocs_cpsc_s2 validation: metrics_ext.macro_ap 与 metrics.auprc 不一致(>1e-6)

## 2026-10-19 22:0x 值班#1: 0/30 停链定因=校验器假阳性(非实验失败), 已修复+复验路径恢复
- 现象: clocs_cpsc_s0/s2 两次 FT10 均正常跑完(rc=0, 产物完整), 但车道在产物校验处停止——断言 |metrics_ext.macro_ap − metrics.auprc|<1e-6 失败。
- 取证(22:00 值班轮): 从落盘 y_true/y_prob 独立复算 auroc/auprc 与 metrics.json 差 0.000e+00(逐位一致); macro_ap 为无ties处理手写 AP, 与 sklearn 固有差 ~1.05e-5。断言把两种不同口径公式做 1e-6 等价比较, 必然假阳性, 30 任务将全部误停。
- 修复: ①runner/finalize 校验改为任务书§七-5 本意=预测独立复算==metrics.json @1e-6(更强), macro_ap 只做 1e-3 粗检; ②runner 加 try_revalidate 复验路径(§五-1 工件齐全且一致可跳过), s0/s2 免重训直接复验入账。
- 动作: 修复 commit 后 launch_w9.ps1 幂等重启, 预计复验 2 + 新跑 28。实验协议零改动(超参/矩阵/种子未动)。

## 2026-10-19 22:02 run 启动: 30 任务/2 车道/错峰600s (任务书27 + 用户裁决扩充m5-ptbxl×3)
