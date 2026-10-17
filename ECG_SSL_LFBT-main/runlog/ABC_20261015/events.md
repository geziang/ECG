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

## 2026-10-16 03:12 P0-1 ft10fill 完成: M3/M4/M6 × cpsc/chapman × s024 = 18 格补齐

## 2026-10-16 05:36 m7 seed0 PT 完成: 144min, 峰值显存 8340MiB

## 2026-10-16 09:48 m7 seed2 PT 完成: 144min, 峰值显存 8773MiB

## 2026-10-16 13:14 m7 seed4 PT 完成: 99min, 峰值显存 8831MiB

## 2026-10-16 15:03 m7 seeds[0, 2, 4] 全下游完成(LP3域/FT3域/27格)

## 2026-10-16 16:0x 值班#12 环境记录: GPU 共用开始

- 15:0x-16:00 间出现共驻进程 pid13588(conda ASPP_AMGCN), 与 m8 s0 PT 并行; 总显存 17.1/24.5GB 余 7.5GB 无 OOM; PT 节奏 1.0-1.45→2.48min/ep(约×2)。批次继续串行不受影响, 全批 ETA 视共驻时长顺延 +10~15h。未做任何处置(共驻为用户侧安排, 值班只记录)。

## 2026-10-16 18:10 m8 seed0 PT 完成: 187min, 峰值显存 17214MiB

## 2026-10-16 22:29 m8 seed2 PT 完成: 149min, 峰值显存 9261MiB

## 2026-10-17 02:48 m8 seed4 PT 完成: 150min, 峰值显存 9034MiB

## 2026-10-17 04:37 m8 seeds[0, 2, 4] 全下游完成(LP3域/FT3域/27格)

## 2026-10-17 06:32 m9 seed0 PT 完成: 115min, 峰值显存 8355MiB

## 2026-10-17 10:15 m9 seed2 PT 完成: 115min, 峰值显存 8355MiB

## 2026-10-17 13:58 m9 seed4 PT 完成: 114min, 峰值显存 8352MiB

## 2026-10-17 15:47 m9 seeds[0, 2, 4] 全下游完成(LP3域/FT3域/27格)

## 2026-10-17 15:47 runlog/ABC_20261015/run_abc_bench.py 失败 exit=1(见 logs/run_abc_bench.log)

## 2026-10-17 16:00 audit 通过: 锚点齐(含 M2 跨批次), R峰缓存哈希 {'nfh': '8ed1e5402d09cffa', 'ptbxl': '7f23838ab25d64b3'}, pt_pretrain(17418)==pt_rpeaks.npz 严格一一对应, split_sha 就绪

## 2026-10-17 16:00 P0-1 ft10fill 完成: M3/M4/M6 × cpsc/chapman × s024 = 18 格补齐

## 2026-10-17 16:00 m7 seeds[0, 2, 4] 全下游完成(LP3域/FT3域/27格)

## 2026-10-17 16:00 m8 seeds[0, 2, 4] 全下游完成(LP3域/FT3域/27格)

## 2026-10-17 16:00 m9 seeds[0, 2, 4] 全下游完成(LP3域/FT3域/27格)

## 2026-10-17 16:00 runlog/ABC_20261015/run_abc_bench.py 失败 exit=1(见 logs/run_abc_bench.log)

## 2026-10-17 16:0x 值班告警-硬失败定因与修复（实现bug, 方案§9授权）

- 现象: bench 阶段连续两次 exit=1(15:47 老链 21992 / 16:00 新链 48172 同位置), 链终止; results.csv 316 行=m9 全部收口后死在 P0-4。
- 根因: run_abc_bench.py 缺 sys.path.insert(仓库根), `from models.mbn import MultiBranchNet` 抛 ModuleNotFoundError——主会话编写疏漏, 纯运行器实现 bug, 科学协议/数据/账本零影响(该阶段只读 checkpoint 产 efficiency.csv)。
- 处置(§9 实现错误可修复): 补 sys.path.insert(修复前后 SHA 见本条 commit); 无需重跑任何已完成任务(链幂等, bench 自身无状态); 修复后重启链继续 bench→stats0→n1n2。
- 附注: 16:00 新链幂等重过各阶段时重复追加了完成事件条目(ft10fill/m7/m8/m9), 属重复记账噪音非新数据。

## 2026-10-17 16:01 audit 通过: 锚点齐(含 M2 跨批次), R峰缓存哈希 {'nfh': '8ed1e5402d09cffa', 'ptbxl': '7f23838ab25d64b3'}, pt_pretrain(17418)==pt_rpeaks.npz 严格一一对应, split_sha 就绪

## 2026-10-17 16:01 P0-1 ft10fill 完成: M3/M4/M6 × cpsc/chapman × s024 = 18 格补齐

## 2026-10-17 16:01 m7 seeds[0, 2, 4] 全下游完成(LP3域/FT3域/27格)

## 2026-10-17 16:01 m8 seeds[0, 2, 4] 全下游完成(LP3域/FT3域/27格)

## 2026-10-17 16:01 m9 seeds[0, 2, 4] 全下游完成(LP3域/FT3域/27格)

## 2026-10-17 16:01 runlog/ABC_20261015/run_abc_bench.py 失败 exit=1(见 logs/run_abc_bench.log)

## 2026-10-17 16:18 audit 通过: 锚点齐(含 M2 跨批次), R峰缓存哈希 {'nfh': '8ed1e5402d09cffa', 'ptbxl': '7f23838ab25d64b3'}, pt_pretrain(17418)==pt_rpeaks.npz 严格一一对应, split_sha 就绪

## 2026-10-17 16:18 P0-1 ft10fill 完成: M3/M4/M6 × cpsc/chapman × s024 = 18 格补齐

## 2026-10-17 16:18 m7 seeds[0, 2, 4] 全下游完成(LP3域/FT3域/27格)

## 2026-10-17 16:18 m8 seeds[0, 2, 4] 全下游完成(LP3域/FT3域/27格)

## 2026-10-17 16:18 m9 seeds[0, 2, 4] 全下游完成(LP3域/FT3域/27格)

## 2026-10-17 16:21 P0-4 效率基准完成: efficiency.csv

## 2026-10-17 17:5x 值班#25 尾声优化(实现层, 方案§9)

- 发现: 原链尾部设计 = stats(全量~11h) → report(又全量重跑~11h 才生成报告), 重复浪费。
- 修复: ① stats/report-only 两模式加幂等完成标记(logs/stats0_done/stats_done), 链重启不再重跑统计; ② report 阶段改 --report-only(只从既有 CSV 装配报告, 分钟级)。影响面: 纯运行器效率, 统计口径/数值零变化。
- 执行注记: 在跑链 40104 仍是旧代码——待 stats0 结束、链进入 n1n2 之后由值班轮择 PT 间隙重启一次(PT 自带 --resume, 损失≤1 epoch), 使新代码在 stats 阶段生效; 全批预计省 ~11h。
