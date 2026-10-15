# ABC_20261015 events

## 2026-10-15 18:5x 批次开仓（主机A）

- 方案 `ABC联合框架补充实验方案.md` 经 git 拉取（5f544f1）确认，按用户指令"设计实验流水线一次性搞完"冻结执行 P0+P1 连续队列（P2 不排程，方案 §13）。
- 冻结协议 = 本目录 protocol_frozen.md：锚点映射 M0=c1/M1=c2/M2=dv/M5=noiseiso（与 JOINT run_joint_stats 一致），新增 M7/M8/M9/M4b/N1N2-s2s4。
- 代码变更 1：metrics_ext._assert_w3_dir 允许清单追加 runlog/ABC_20261015/（沿每批次扩目录先例，W2 冻结只读不动，其余行为零变化）。影响面：仅新增落盘目录白名单。
- 已知缺口记录：M2/M5 的 CPSC/Chapman FT10 历史从未运行且不在方案 18 任务内（见 protocol_frozen §9）；W8 最终报告待另行撰写（run_w8_stats 已完成 48 行）。
- 主链 run_abc_chain.py 阶段序 = protocol_frozen §4；串行 IO 纪律；幂等可重启。

## 2026-10-15 16:58 audit 通过: 锚点齐(含 M2 跨批次), R峰缓存哈希 {'nfh': '8ed1e5402d09cffa', 'ptbxl': '7f23838ab25d64b3'}, pt_pretrain(17418)==pt_rpeaks.npz 严格一一对应, split_sha 就绪

## 2026-10-15 16:58 ABC 链阶段完成: audit

## 2026-10-15 16:58 audit 通过: 锚点齐(含 M2 跨批次), R峰缓存哈希 {'nfh': '8ed1e5402d09cffa', 'ptbxl': '7f23838ab25d64b3'}, pt_pretrain(17418)==pt_rpeaks.npz 严格一一对应, split_sha 就绪

## 2026-10-15 16:58 ABC 链阶段完成: audit

## 2026-10-15 16:58 audit 通过: 锚点齐(含 M2 跨批次), R峰缓存哈希 {'nfh': '8ed1e5402d09cffa', 'ptbxl': '7f23838ab25d64b3'}, pt_pretrain(17418)==pt_rpeaks.npz 严格一一对应, split_sha 就绪

## 2026-10-15 16:58 ABC 链阶段完成: audit

## 2026-10-15 16:59 audit 通过: 锚点齐(含 M2 跨批次), R峰缓存哈希 {'nfh': '8ed1e5402d09cffa', 'ptbxl': '7f23838ab25d64b3'}, pt_pretrain(17418)==pt_rpeaks.npz 严格一一对应, split_sha 就绪

## 2026-10-15 16:59 ABC 链阶段完成: audit

## 2026-10-15 17:00 audit 通过: 锚点齐(含 M2 跨批次), R峰缓存哈希 {'nfh': '8ed1e5402d09cffa', 'ptbxl': '7f23838ab25d64b3'}, pt_pretrain(17418)==pt_rpeaks.npz 严格一一对应, split_sha 就绪

## 2026-10-15 17:00 ABC 链阶段完成: audit

## 2026-10-15 17:12 smoke m7 通过: 项=['loss', 'loss_dv', 'loss_r', 'loss_rr', 'loss_t']; config/语料/R峰缓存核验一致

## 2026-10-15 17:15 smoke m8 通过: 项=['loss', 'loss_dv', 'loss_r', 'loss_t']; config/语料/R峰缓存核验一致

## 2026-10-15 17:18 smoke m9 通过: 项=['loss', 'loss_r', 'loss_rr', 'loss_t']; config/语料/R峰缓存核验一致

## 2026-10-15 17:24 smoke m4b 通过: 项=['loss', 'loss_dv', 'loss_r', 'loss_rr', 'loss_t']; config/语料/R峰缓存核验一致

## 2026-10-15 17:25 smoke m7 端到端 CPSC LP 通过(结构加载+预测链路)

## 2026-10-15 17:25 ABC 链阶段完成: smoke

## 2026-10-15 17:26 audit 通过: 锚点齐(含 M2 跨批次), R峰缓存哈希 {'nfh': '8ed1e5402d09cffa', 'ptbxl': '7f23838ab25d64b3'}, pt_pretrain(17418)==pt_rpeaks.npz 严格一一对应, split_sha 就绪

## 2026-10-15 17:27 流水线正式起跑（主机A）

- smoke 门四配置全过（五项检查 + m7 端到端 CPSC LP），§9.1 满足；期间修复一处 smoke 断言 bug（run_pt 的 train_state 为 0 基 epoch 索引，resume 标记应为"恢复自 epoch 0"——纯验证代码修正，不影响科学协议）。
- launch_abc.ps1 拉起全链（pid 21992，默认全阶段 audit→…→report）；audit 复验通过；ft10fill(P0-1) 开跑。
- 值班 cron automation-570dadd0 建立（每 2h：拉活+健康+进度+收官检测推送）。
- 预算: ~60h GPU 串行（18 FT10≈9h → M7/M8/M9 各≈11h → bench/stats0 → N1N2≈9h → M4b≈5h → multilib → stats+report）。
