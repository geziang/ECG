# ABC_20261015 patrol_log

值班记录(每轮追加): 只记事实与处置, 判断需回查日志佐证。防双投递: 先查末轮时间戳, <45min 只做校对不重巡。

## 2026-10-15 17:0x 开仓轮(主会话)

- 链未启动(smoke 阶段验证中); audit 已过(锚点齐/R峰缓存一一对应/split_sha 三域就绪)。
- 值班 cron 建立(2h 周期), 职责: launch_abc.ps1 幂等拉活 + 链健康检查 + 完成检测(收官时跑 stats --report + git push)。

## 2026-10-15 18:00 值班#1(automation-570dadd0)

- 链活着(pid=21992, CIM 确认); 阶段=ft10fill, 进度 1/18(m3 s0 cpsc 0.9239/0.7057, 30min/个符合预算), 当前作业 m3_cpsc_ft10_seed2 训练中(日志 18:00:58 新鲜)。
- chain.out.log 无 Traceback/FAIL, err 空; GPU 3.0GB/低util=FT 期正常形态(小批间数据加载为主)。results.csv 2 行(含表头)。
- 处置: 无需干预; 未到收官(无完成标记), 不推送。

## 2026-10-15 20:00 值班#2(automation-570dadd0)

- 链活着(pid=21992); 阶段=ft10fill, 进度 4/18: m3/cpsc 三种子齐(s0 0.7057/s2 0.7021/s4 0.7041 AUPRC, 种子间一致), m3/chapman s0 0.9914/0.9748 已落账, 当前训 m3_chapman_ft10_seed2(日志 20:00:37 新鲜)。
- 无 Traceback/FAIL, err 空; GPU 3.4GB/采样瞬时低util=FT 正常。results.csv 5 行(含表头)。
- 处置: 无需干预; 未收官不推送。节奏 30min/个, 18 个预计 10-16 03:3x 完成后进 m7。

## 2026-10-15 22:00 值班#3(automation-570dadd0)

- 链活着(pid=21992); 阶段=ft10fill, 进度 8/18: m3 全齐(cpsc 0.7021-0.7057 / chapman 0.9746-0.9750, 种子间极稳), m4/cpsc s0 0.6994 + s2 0.6972 已落账, 当前训 m4_cpsc_ft10_seed4(日志 22:00:31 新鲜)。
- 无 Traceback/FAIL, err 空; GPU 3.4GB/低util=FT 正常。results.csv 9 行(含表头)。
- 初步观察(值班只记不判): m4/cpsc 两种子 AUPRC(0.6994/0.6972)略低于 m3(0.7021-0.7057), 与 JOINT 结论"M4 clean 持平、增益在噪声端"方向一致, 待统计阶段配对定论。
- 处置: 无需干预; 未收官不推送。

## 2026-10-16 00:00 值班#4(automation-570dadd0)

- 链活着(pid=21992); 阶段=ft10fill, 进度 12/18: m3/m4 各六格全齐。m4/cpsc AUPRC 0.6994/0.6972/0.7071(vs m3 0.7021-0.7057, 配对见统计), m4/chapman 0.9658/0.9730/0.9728(vs m3 0.9746-0.9750)。当前训 m6_cpsc_ft10_seed0(日志 00:00:46 新鲜)。
- 无 Traceback/FAIL, err 空; GPU 3.4GB/低util=FT 正常。results.csv 13 行(含表头)。
- 值班观察(只记不判): m4/m3 chapman 三种子配对差 = -0.0090/-0.0020/-0.0022, 全负但幅度小; cpsc 差 = -0.0063/-0.0049/+0.0030。M4 相对 M3 在 clean FT10 端无优势、方向与 JOINT 持平结论一致, 噪声端增益待 m7-m9 噪声评测后统计定论。
- 处置: 无需干预; 未收官不推送。m6 六格预计 10-16 03:1x 排空后进 m7 PT。

## 2026-10-16 02:00 值班#5(automation-570dadd0)

- 链活着(pid=21992); 阶段=ft10fill, 进度 15/18: m6/cpsc 三种子齐(0.7008/0.7073/0.6985, 与 m3/m4 同档), 当前训 m6_chapman_ft10_seed0(日志 02:00:36 新鲜)。
- 无 Traceback/FAIL, err 空; GPU 3.4GB/低util=FT 正常。results.csv 16 行(含表头)。
- 处置: 无需干预; 未收官不推送。剩 m6/chapman×3, 预计 10-16 04:0x 排空后 ft10fill 段完成进 m7 s0 PT(约 100min/种子)。

## 2026-10-16 04:00 值班#6(automation-570dadd0) — P0-1 完成里程碑

- 链活着(pid=21992); **ft10fill 排空: 18/18 格全落账**(results.csv 19 行含表头, events 03:12 已记)。m6/chapman 收尾三格 0.9663/0.9741/0.9766。
- 03:12 链自动进入 m7 阶段, s0 PT 启动参数核验正确(trc0+dv0.01+rr0.1+nstdb0.5/first_half+rpeak nfh); 04:00 已训至 epoch 33/100(loss_dv/loss_rr 分项齐现且有限, ~1.75min/ep ≈ 95min/种子符合预算)。
- 无 Traceback/FAIL, err 空; GPU 51%/8.3GB = PT 正常形态。chain.out.log mtime 03:12(48min, PT 期单作业 ~100min 无主链行, 正常; PT 子日志 04:00:47 新鲜)。
- 处置: 无需干预; 未收官不推送(按值班规程仅收官轮推送)。m7 s0 预计 ~04:5x 完 PT → LP×3+FT10×3+27格, m7 全三种子预计 10-16 ~14:3x。

## 2026-10-16 06:02 值班#7(automation-570dadd0)

- 链活着(pid=21992); m7 s0 全下游推进中: PT 完成(exit=0, 144min, vram 8.3GB; 实测 1.45min/ep, 慢于 JOINT m4 的 ~0.96min/ep, 同机同负载疑 USB 桥接盘 IO 波动, 只记不判), 三域 LP 已落账(ptbxl 0.6465 / cpsc 0.7605 / chapman 0.9866 AUPRC), 当前训 m7_ptbxl_ft10_seed0(日志 06:02:45 新鲜)。
- 值班观察(只记不判): m7(B+C 无 A) cpsc LP AUPRC 0.7605 处于矩阵较高档(M3≈0.745 档), 待配对统计定论。
- 无 Traceback/FAIL, err 空; GPU 3.4GB/低util=FT 正常。results.csv 22 行(含表头)。
- 预算修正: PT 若持续 144min/种子, m7-m9 三配置将多耗 ~7h, 全批 ETA 从 ~10-18 顺延至 ~10-19 凌晨。处置: 无需干预。

## 2026-10-16 08:00 值班#8(automation-570dadd0)

- 链活着(pid=21992); **m7 s0 全下游收口**(PT 144min + LP×3 + FT10×3: ptbxl 0.6346/cpsc 0.7000/chapman 0.9668 + 27格噪声); 07:24 m7 s2 PT 已接力(GPU 52%/8.7GB 正常)。
- 对账: results.csv 52 行 = 表头+18(ft10fill)+3(LP)+3(FT10)+27(噪声), 分毫不差; m7 三域噪声逐记录预测(y_prob/y_true/noise_offsets)抽验齐备。
- 值班观察(只记不判): 27 格噪声评测实测 <1min/种子(远快于方案预算 30min; 4090 上 8×alpha0.125 编码器 batch256 特征抽取为主, 无参照双跑), 后续预算按此修正——m7-m9 噪声端几乎不占时间。
- 无 Traceback/FAIL, err 空。处置: 无需干预; 未收官不推送。

## 2026-10-16 10:00 值班#9(automation-570dadd0)

- 链活着(pid=21992); m7 s2 PT 完成(144min/vram 8.8GB, 与 s0 同节奏=稳定工作负载), LP×3 落账(ptbxl 0.6493 / cpsc 0.7669 / chapman 0.9858; cpsc 两种子 0.7605/0.7669 同档高位), 当前 FT10 ptbxl s2 训练中。
- 无 Traceback/FAIL, err 空; GPU 3.8GB/低util=FT 正常。results.csv 55 行(含表头)。
- 处置: 无需干预; 未收官不推送。m7 s2 下游预计 ~11:4x 收口后进 s4, m7 全配置预计 ~14:1x, m8 接力。

## 2026-10-16 12:00 值班#10(automation-570dadd0)

- 链活着(pid=21992); m7 s2 全下游收口(11:35, FT10×3+27格全落账), m7 s4 PT 训练中(epoch 24/100, GPU 68%/8.8GB 健康; 本种子节奏 ~1.0min/ep 快于前两种子的 1.45, ETA ~13:1x 完 PT)。
- 对账: results.csv 85 行 = 52 + s2(LP3+FT3+噪声27), 分毫不差。
- 无 Traceback/FAIL, err 空。11:36 用户手动查显存一次(非值班轮): 总占用 8.8/24.5GB, 双 python=链+子作业(母子关系核验), ollama.exe 常驻小占用无压力; 已向用户报备共用额度口径(PT 期 ~15GB/下游期 ~21GB)。
- 处置: 无需干预; 未收官不推送。m7 预计 ~14:5x 收口交棒 m8。

## 2026-10-16 14:00 值班#11(automation-570dadd0)

- 链活着(pid=21992); m7 s4 PT 完成(99min/vram 8.8GB — 快于 s0/s2 的 144min, 证实前两种子为瞬态 IO 波动非负载漂移), LP×3+FT10(ptbxl 0.6403) 已落账, 当前 FT10 cpsc s4 训练中(GPU 5%/3.9GB 正常)。
- 值班观察(只记不判): m7 cpsc LP AUPRC 三种子 0.7605/0.7669/0.7675(极稳, mean≈0.765 高位), 配对定论留统计阶段。
- 对账: results.csv 89 行(含表头)吻合。无 Traceback/FAIL, err 空。
- 处置: 无需干预; 未收官不推送。m7 预计 ~14:5x 收口(剩 FT10×2+噪声) 交棒 m8 s0。

## 2026-10-16 16:00 值班#12(automation-570dadd0) — m7 收口 + 共用上卡

- 链活着(pid=21992); **m7 三种子全收口**(15:03, PT+LP×3+FT10×3+27格 ×3, results.csv 118 行=89+29 对账吻合), 15:04 自动交棒 m8, s0 PT 训练中(epoch 23/100)。
- **环境变化: 共用方上卡**——GPU 出现新进程 pid13588(conda env ASPP_AMGCN, 非本批次; 11:38 用户咨询过共用额度, 推测为用户安排)。当前 GPU 总占用 17.1/24.5GB = 批次 PT(~8.8) + ASPP_AMGCN(~8) + ollama 常驻小量; 余 7.5GB, PT 已稳态无 OOM 风险。
- 节奏影响(如实记录): m8 s0 实测 2.48min/ep(独占期 1.0-1.45), 算力争用使 PT 约慢一倍; 若共驻持续, m8/m9/n1n2/m4b 全部 PT 段 ETA 顺延约 +10~15h(科学结果零影响, 纯墙钟)。
- 无 Traceback/FAIL, err 空(GPU util 95% 为共驻双负载正常)。
- 处置: 只观察不干预(共驻为用户侧安排); 提醒后续轮次关注余量(<3GB 时需用户裁决优先级)。

## 2026-10-16 18:02 值班#13(automation-570dadd0)

- 链活着(pid=21992); m8 s0 PT epoch 91/100(ETA ~18:2x 完); **共用方已下卡**——GPU 总占用 17.1→9.0GB(ASPP_AMGCN 进程消失, 余 15.1GB), 节奏回升 2.48→1.78min/ep(共驻期与独占期混合均值)。
- 无 Traceback/FAIL, err 空。results.csv 118 行(m8 未新增=正常, PT 期不落账)。
- 净影响核算: 共驻约 1.5h(15:0x-17:xx), m8 s0 PT 损失约 ~30-40min 墙钟, 对全批 ETA 影响 <1h, 可忽略。
- 处置: 无需干预; 未收官不推送。m8 s0 下游(~1.5h)后 s2/s4 依序, m8 全配置预计 10-17 ~05:xx。

## 2026-10-16 20:00 值班#14(automation-570dadd0)

- 链活着(pid=21992); **m8 s0 全下游收口**(20:00: LP×3+FT10×3: ptbxl 0.6411/cpsc 0.7166/chapman + 27格噪声), 20:00:30 m8 s2 PT 接力(参数核验正确: trc1+dv0.01/rr0+nstdb; GPU 2.4GB=启动初始化瞬间)。
- 值班观察(只记不判): m8(A+Beat+C) cpsc FT10 s0 0.7166 高于 m7 同格 0.7000 与 m3/m4/m6(~0.70), B 内部拆分首格信号积极, 待配对。
- 对账: results.csv 151 行 = 118+33(m8 s0) 分毫不差。无 Traceback/FAIL, err 空。
- 处置: 无需干预; 未收官不推送。s2 PT 独占节奏预计 21:4x-22:2x 完, m8 全配置预计 10-17 03:xx-05:xx。

## 2026-10-16 22:00 值班#15(automation-570dadd0)

- 链活着(pid=21992); m8 s2 PT epoch 80/100(1.5min/ep 独占稳态节奏, 日志 22:00:24 新鲜, GPU 41%/9.0GB 健康), ETA ~22:30 完 PT → 下游(~1.5h) → s4 约 00:00 接力。
- 无 Traceback/FAIL, err 空。results.csv 151 行(PT 期无新增=正常)。
- 处置: 无需干预; 未收官不推送。

## 2026-10-17 00:00 值班#16(automation-570dadd0)

- 链活着(pid=21992); m8 s2 PT 完成(22:31), LP×3 落账(cpsc 0.7652 — 第二种子延续高位), FT10 已落 ptbxl 0.6342/cpsc 0.7041, 当前训 chapman s2(GPU 4.0GB FT 正常)。
- 值班观察(只记不判): m8 cpsc LP 两种子 0.7652/0.7652(s0)——与 m7 同档高位; m8 cpsc FT10 两种子 0.7166/0.7041。
- 对账: results.csv 156 行 = 151+5 吻合。无 Traceback/FAIL, err 空。
- 处置: 无需干预; 未收官不推送。s2 约 00:2x 收口后 s4 接力, m8 预计 10-17 04:xx-05:xx 收口交棒 m9。
  - [更正 00:0x] 上条"两种子 0.7652/0.7652"有笔误: 账本核实 m8 cpsc LP AUPRC = s0 0.7717 / s2 0.7652(均高位, s0 更高)。已回查 results.csv 原始行。

## 2026-10-17 02:00 值班#17(automation-570dadd0)

- 链活着(pid=21992); m8 s2 已收口(00:18, chapman FT10+27格噪声落账), m8 s4 PT epoch 68/100(1.5min/ep 稳态, GPU 59%/9.0GB 健康, 日志 02:00:25 新鲜)。
- 对账: results.csv 184 行 = 156+28 分毫不差。无 Traceback/FAIL, err 空。
- 处置: 无需干预; 未收官不推送。s4 PT ~02:5x 完 → 下游 → m8 全配置预计 ~04:2x 收口交棒 m9。

## 2026-10-17 04:00 值班#18(automation-570dadd0)

- 链活着(pid=21992); m8 s4 PT 完成(02:48, 150min/vram 9.0GB), LP×3 落账后 FT10 已落 ptbxl 0.6509, 当前训 cpsc s4(GPU 4.0GB 正常)。
- 值班观察(只记不判): m8(A+Beat+C) cpsc LP AUPRC 三种子 0.7717/0.7652/0.7708(mean≈0.769, sd 极小)——三种子高位极稳。
- 对账: results.csv 188 行 = 184+4 吻合。无 Traceback/FAIL, err 空。
- 处置: 无需干预; 未收官不推送。m8 预计 ~04:4x 收口交棒 m9。

## 2026-10-17 06:00 值班#19(automation-570dadd0) — m8 收口

- 链活着(pid=21992); **m8 三种子全收口**(04:37, B拆分之 A+Beat+C 侧齐), m9 阶段接力——s0 PT 已训至 epoch 72/100(1.15min/ep 快节奏, GPU 62%/8.4GB 健康, 日志 06:00:32 新鲜)。
- 对账: results.csv 217 行 = 188+29(m8 s4 剩余: FT10×2+噪声27) 分毫不差。
- 无 Traceback/FAIL, err 空。处置: 无需干预; 未收官不推送。m9 s0 PT ~06:3x 完, m9 全配置预计 10-17 17:xx-19:xx, 之后 bench+stats0 进 P1。

## 2026-10-17 08:00 值班#20(automation-570dadd0)

- 链活着(pid=21992); m9 s0 PT 完成(06:33), LP×3 落账(cpsc 0.7673), FT10 已落 ptbxl 0.6341/cpsc 0.7029, 当前训 chapman s0(GPU 4.0GB 正常)。
- 值班观察(只记不判): 新三配置 cpsc LP AUPRC 首种子对照——m7(B+C) 0.7605 / m8(A+Beat+C) 0.7717 / m9(A+RR+C) 0.7673, 三者同档高位, 均高于 m3 锚点档(~0.745); 正式配对与交互留统计阶段。
- 对账: results.csv 222 行 = 217+5 吻合。无 Traceback/FAIL, err 空。
- 处置: 无需干预; 未收官不推送。

## 2026-10-17 10:00 值班#21(automation-570dadd0)

- 链活着(pid=21992); m9 s0 已收口(08:21, chapman FT10+27格噪声落账), m9 s2 PT epoch 86/100(1.15min/ep 快节奏, GPU 53%/8.4GB 健康, 日志 10:00:30 新鲜)。
- 对账: results.csv 250 行 = 222+28 分毫不差。无 Traceback/FAIL, err 空。
- 处置: 无需干预; 未收官不推送。s2 PT ~10:1x 完 → 下游 → s4 ~11:5x 接力, m9 全配置预计 ~16:3x 收口, 之后 bench(P0-4)→stats0(P0统计快照)。

## 2026-10-17 12:00 值班#22(automation-570dadd0)

- 链活着(pid=21992); m9 s2 PT 完成(10:17), LP×3 落账(cpsc 0.7757 — 第二种子更高), FT10 已落 ptbxl 0.6388/cpsc 0.7012, 当前训 chapman s2(GPU 4.0GB 正常)。
- 值班观察(只记不判): m9(A+RR+C) cpsc LP 两种子 0.7673/0.7757, 与 m7/m8 同档高位带确认。
- 对账: results.csv 255 行 = 250+5 吻合。无 Traceback/FAIL, err 空。
- 处置: 无需干预; 未收官不推送。s2 ~12:1x 收口 → s4(全批 m 系最后一个 PT) → m9 预计 ~16:2x 收口 → bench+stats0。

## 2026-10-17 14:00 值班#23(automation-570dadd0)

- 链活着(pid=21992); m9 s2 已收口(12:04), m9 s4 PT 完成(13:58, 114min/vram 8.4GB — 全批 9 次 m 系新预训练至此全部完成), LP 推进中(ptbxl 0.6514 已落, GPU 2.1GB=LP 轻载正常)。
- 对账: results.csv 284 行 = 255+29 吻合。无 Traceback/FAIL, err 空。
- 处置: 无需干预; 未收官不推送。m9 s4 下游预计 ~15:3x 收口(快于此前 16:2x 估计) → bench(P0-4) → stats0(P0 统计快照) → n1n2(P1-1)。

## 2026-10-17 16:3x 值班#24(automation-570dadd0) — bench 双崩定因修复, P0 GPU 段全部完成

- 里程碑: **m9 三种子收口(15:47) = 9 次新预训练(M7/M8/M9×3)零失败**, results.csv 316 行; P0 的 GPU 侧全部完成。
- 事故与处置: bench 阶段连续两崩(15:47 链21992 / 16:00 链48172)——根因1=run_abc_bench.py 缺 sys.path.insert(models 导入必炸), 根因2=net.parameters() 应为 named_parameters(); 均为主会话编写疏漏的纯实现 bug(方案§9 授权修复, events 已记告警+影响面), 科学数据零影响。修复(3cca7cb)后手动跑通 bench(efficiency.csv 10 行: 全模型 ~630K 参数/145.1M MACs/b1 延迟 ~15ms/吞吐 ~4000r/s), 链重启(40104)通过 bench, 16:21 进入 stats0。
- stats0 进行中: 锚点重放四位吻合(m1/ptbxl 0.8860/0.8845/0.8876), 预计 1-3h(CPU bootstrap)。
- 教训入账: 新运行器此前只编译检查未实跑——bench 全新代码路径应在 smoke 阶段实跑一遍(已在本轮手动补课)。
- 处置: 无需再干预; 未收官不推送。

## 2026-10-17 18:01 值班#26(automation-570dadd0)

- 链活着(pid=40104); 阶段=stats0(P0统计快照), 已产出 29 项配对 bootstrap(最新 17:57:02, 日志新鲜活跃)。核心族已见: cpsc M4−M0 +1.308pt CI[+0.37,+2.25] p=0.0054 3/3(E1档), M4−M1 +0.346pt(AUROC 显著), M4−M2 +0.336pt(AUROC 显著), M4−M3 +0.275pt CI跨0(持平, 与JOINT一致)。
- 无 Traceback/FAIL, err 空; GPU 空闲=统计期正常(CPU bootstrap)。results.csv 316 行不变=正常。
- 处置: 无需干预; 未收官不推送。stats0 预计 10-18 凌晨 ~01:15 收尾, 收尾后按 events 重启规程执行优化重启(树杀+补 stats0_done 标记+拉新链, 训练损失≤1 epoch)。

## 2026-10-17 20:00 值班#27(automation-570dadd0)

- 链活着(pid=40104); 阶段=stats0, 67 项配对已产出(+38/2h ≈ 3.2min/项稳态), LP 三域族已跑完(n1/n2 对按设计跳过待 P1-1 后补), 当前进 FT10 对族(cpsc m4:m3 已出)。日志 19:56 新鲜。
- 无 Traceback/FAIL, err 空; GPU 空闲=统计期正常。results.csv 316 行不变。
- 处置: 无需干预; 未收官不推送。剩余 FT 对+噪声族+因子交互, stats0 预计 ~00:30-01:30 收尾后执行优化重启规程。
