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
