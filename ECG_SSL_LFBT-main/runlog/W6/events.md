# W6 事件记录

## 2026-10-03 14:19 · L2 首发错峰不足, 主动停发重发（车道: L2）
- 现象: L1 于 14:13:58 起跑后, L2 于 14:19:09 误发(仅错峰 5min11s < 纪律下限 10min)——时间估算失误。
- 处置: 30 秒内发现, 杀掉 L2 全进程树(runner+run_ft+workers), 清理其部分产物(ft/predictions/logs, 均未到 test 阶段, 无账本污染); 幂等设计保证 14:24 后重发自动从 c2 ft20 seed0 干净重来。
- 附带发现: ①杀进程树不能按"启动时间>阈值"扫杀——run_ft 每 epoch 重spawn dataloader worker(父PID=run_ft), 扫杀会误伤在跑车道, 今后只按核验过命令行的精确 PID 杀; ②本次扫杀窗口碰巧命中 L1 换界 worker, L1 存活且进度正常(Epoch 10→19), 未受实质影响。

## 2026-10-03 14:0x · Stage 0 命令勘误（车道: CPU）
- 现象: 按任务书原样命令(默认 pairs)起跑后, `c1:s1`/`c1:s2` 两对被 skip——`paired_stats.py` 的 DEFAULT_PAIRS 用占位名 s1/s2, 而 W5/W4 预测目录实名 `simclr_/clocs_`。
- 依据: W4 官方产物 `runlog/W4/stats/paired_stats.csv` 中 method_b 即为 simclr/clocs, 证明 W4 当次为显式 `--pairs c1:c2 c1:simclr c1:clocs c2:b0` 传参。
- 处置: 停残跑(仅完成 c1:c2 一对), 以显式 --pairs 重启; **代码一行未改**, 四对齐全, 与任务书"默认四对"语义一致。
- 顺手清理: 09-25 遗留 4 个 python 陈尸进程(W5A dataloader worker, 0 CPU 不占卡)已按重启恢复手册清除。
