# 每小时巡检报告 2026-09-18 11:10(首份)

## 一、GPU
18971 / 24564 MiB(77%),利用率 94%。正常(告警阈值 23500MiB)。

## 二、车道与进程
- 3 个 pipeline_runner 全部存活:laneC(pid 50312,跑 common_view01)、laneF(11:05 重启后认领 proj_dim1024,等显存闸门)、laneG(pid 20160,跑 d7_rec01);
- psfull 判决 LP(pid 27452,run_e006_downstream)在跑;
- 全部 python 进程与 DataLoader worker 数量正常,无孤儿。

## 三、在跑任务
| 任务 | 进度 | 佐证 | ETA |
|---|---|---|---|
| psfull_arb3 LP(判决) | epoch ~60/100,~24s/ep | 日志 65s 内 +4KB | **~11:25** |
| common_view01 PT | epoch ≥91/200 | pmon SM 20–72%(日志块缓冲延迟,非卡死) | ~12:00 |
| d7_rec01 PT(重跑) | 早期 epoch | pmon SM 41–72%(日志仅头部=缓冲) | ~15:00 |
| proj_dim1024 PT | 已认领,停 7800MiB 闸门(空闲 5159MiB) | laneF 日志 | ~12:00 随 common 释放自动启动 |

## 四、自上小时新完成
matrix_results.csv +1 行:`09-18 11:00, arb3_base, ar_pt, AUPRC 0.718`(与 lp_arb3_base_seed0/metrics.json 0.717958 一致,系前会话手动 LP 链收尾落行,巡检核验无误)。psfull 判决行尚未出现。

## 五、本小时采取的动作
纯巡逻,无写操作。(注:laneF 重启与 csv arb3 行均为主会话/前会话所为,本巡检仅核验。)

## 六、告警与待办
1. 【判读修正·重要】"PT 日志 mtime 冻结 >20min"为 stdout 重定向块缓冲假象(约 8KB/次刷盘);已用 `nvidia-smi pmon -s u` 确认两 PT 均在计算。**后续巡检判活以 pmon SM% 或日志字节数增长为准,mtime 单独不构成告警。**
2. 【待主会话】psfull 判决 LP ETA ~11:25:Δ = psfull − arb3_base(0.7180),≥ +0.003 → 队列加 seed2/4 并起新 runner 接走;≤0 → 记 ❌ 关线。
3. 【待办】laneC/laneG runner 内存仍持旧并发上限 2(10:50 修码前启动);二者各自空闲时可择机重启对齐 3,主会话已知。
4. 【git】main 与远端分叉(远端含主机B run_pt.py 改动,红线禁合并);本报告随 runlog 提交后停靠 hostA-sync 侧分支,待空闲窗口由主会话合并。
5. 无 OOM / Traceback / 同任务双失败;无 runner 死亡;watchdog_relay 按规定保持废弃未动。
