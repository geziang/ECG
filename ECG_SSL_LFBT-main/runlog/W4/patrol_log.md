# W4 主机A 巡检日志(每 2h, automation-6bf636ac)

> 格式: 时间 | 车道状态 | 账本行数 | GPU | 动作

09-23 11:01 | **全巡(首次)** 全部健康,无需重启。
- A-6 S3: seed0 ✅ AUROC 0.8845 | seed2 ✅ 0.8797 | seed4 epoch 38/100(日志秒级新鲜)
- A-4 SimCLR: PT seed0 ✅ 100ep + seed2 ✅ 100ep;下游 cpsc LP 已入账(seed0 0.9380 / seed2 0.9371,均低于 C2 0.958x 约 2pt, 符合文献预期排序);ft10 双车道 epoch 75/100 进行中;seed4 排队(laneA seed0 下游完成后接续)
- A-8b: 未启动(主会话计划 A-6 完成后启动,非掉线)
- A-5 CLOCS: 未启动(主会话职责,巡检不动)
- 账本 8/27 行(b0×4+s3×2+simclr×2) | GPU 8.3G, util 瞬时 0%(epoch 交界采样, mtime 11:01:17-18 全新鲜)

09-23 11:15 | 增量: laneB(seed2)全链完成 exit 0(laneB ALL DONE 11:12); seed0 下游同窗完成 — SimCLR 8/12 行入账(cpsc 0.9380/0.9371, ft10 0.8502/0.8528, ptbxl LP 0.8791/0.8782, chapman 0.9880/0.9871); laneA 转 seed4 PT; A-8b 重放已启动(主会话计划内, laneB 空位)

09-23 11:25 | A-8b 事故与处置: 首启失败(脚本 KeyError: W1 账本无 seed 列)修复; 二启在 b0_ptbxl_seed0 触发 GATE_FAIL(d=+0.0010 vs W1 账本) — 定性: 任务书复现门对象=W2 账本, c2_ptbxl_seed0 探针对 W2 逐位一致(0.8860/0.6564)证明管线确定性完好, W1 行属 09-21 时代代码固有差, 已降为参考行(记录不拦截, 依据写入脚本 docstring)。三启运行中: b0_ptbxl_seed2/4 对 W2 PASS d=+0.0000 逐位, b0_cpsc_seed0 对 W1 REF d=+0.0000。

09-23 11:45 | A-6 ✅ 完结: s3 三行齐(0.8845/0.8797/0.8689, mean 0.8777; 监督全量 vs C2-FT10 0.861x 符合"监督参照≥C2"预期); A-5 启动: clocs 5ep smoke 过(loss 1.78→0.53 单调, temporal/spatial 分项合理), laneB(seed2)全链已启动。当前三车道: SimCLR seed4 PT + A-8b 重放 + CLOCS seed2 PT。CLOCS laneA 待 SimCLR laneA 空位接力。

09-23 12:0x | A-8b ✅ 收口: 复现门(W2账本对象)20/20 全绿逐位(d=+0.0000, c1/c2×18+b0 ptbxl s2/s4); W1 参考行 3 组(b0_cpsc/chapman_seed0 亦逐位, b0_ptbxl_seed0 d=+0.0010 已定性为 W1 时代代码固有差); SKIP+NOREF 4 组=A-7 新行。REPLAY DONE。B-4 统计输入(runlog/W4/predictions/ 全网格逐记录预测)已齐, B 机可开算。

09-23 13:00 | 轻量校对(上次记录12:0x<90min): 全部健康。SimCLR seed4 PT ✅100ep+下游推进中(cpsc LP 已入账 0.9379, ft10 epoch 58/100 mtime 新鲜); CLOCS seed2 PT epoch 63/100 活跃。账本 16/27 行(b0×4+s3×3+simclr×9)。GPU 9.0G/29%(epoch 交界瞬时)。无重启动作。

09-23 13:18 | A-4 ✅ 完结: SimCLR 12/12 行齐(3 seeds×4下游, 种子一致性±0.0005)。mean: cpsc 0.9377 / ft10 0.8500 / ptbxl lp 0.8791 / chapman 0.9876 — 全面低于 C2(−0.7~−2.1pt), 符合文献预期(SimCLR<ECG专用), 无基线反超, §1.5 条款未触发。CLOCS laneA(seed0→seed4)已接力空车道。当前两车道: clocs s2(ep63+) + clocs s0(启动)。

09-23 14:20 | CLOCS laneB ✅ seed2 全链 4 行入账: cpsc 0.9475 / ft10 0.8570 / ptbxl lp 0.8666 / chapman 0.9961 — 整体介于 SimCLR 与 C2 之间符合文献'中档代表'定位(chapman 0.9961 已贴近 C2 0.9963; ptbxl lp 0.8666 低于 SimCLR 属如实入表反序点, 不隐藏)。clocs 4/12 行, laneA seed0 PT ep54/100 继续中。账本 20 行。

09-23 15:00 | 轻量校对: 全健康。CLOCS laneA seed0 PT ep93/100(mtime 秒级), 即将转下游; clocs 4/12 行。账本 23/27(b0×4+clocs×4+s3×3+simclr×12全齐)。GPU 6.5G/54%。14:2x-15:00 推送断网 3 连败(30422bb 积压本地), 本轮重试。
