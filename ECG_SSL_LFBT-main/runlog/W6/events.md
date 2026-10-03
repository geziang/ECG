# W6 事件记录

## 2026-10-03 14:0x · Stage 0 命令勘误（车道: CPU）
- 现象: 按任务书原样命令(默认 pairs)起跑后, `c1:s1`/`c1:s2` 两对被 skip——`paired_stats.py` 的 DEFAULT_PAIRS 用占位名 s1/s2, 而 W5/W4 预测目录实名 `simclr_/clocs_`。
- 依据: W4 官方产物 `runlog/W4/stats/paired_stats.csv` 中 method_b 即为 simclr/clocs, 证明 W4 当次为显式 `--pairs c1:c2 c1:simclr c1:clocs c2:b0` 传参。
- 处置: 停残跑(仅完成 c1:c2 一对), 以显式 --pairs 重启; **代码一行未改**, 四对齐全, 与任务书"默认四对"语义一致。
- 顺手清理: 09-25 遗留 4 个 python 陈尸进程(W5A dataloader worker, 0 CPU 不占卡)已按重启恢复手册清除。
