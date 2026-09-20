# 每小时巡检报告(主机A)— 2026-09-20 20:01

## 一、GPU

18,118 / 24,564 MiB(74%),利用率 92% —— 三任务共载(mimic ~4.3G + b0fast ~7.2G + C1 ~6.6G),低于 23.5G 告警线,健康。

## 二、车道与进程

- **laneF runner 存活**(pid 10172),压 b0fast_seed0(19:30 起);laneC/laneG2 已正常退出(旧 b0fast 秒败时代,bug 已修),laneG 日志陈旧属历史;无需要重启的死亡 runner;
- **b0fast run_pt 存活**(pid 57048,PID 直查 + GPU 双确认);
- **C1 NFH 链存活**(pid 14012,worktree 运行,claim c1_nfh_b0_seed0 在;runlog/W1/ 无状态文件 = PT 阶段进行中);
- mimic(他人进程,41136)未动。

## 三、在跑任务(名称+epoch+ETA)

| 任务 | 进度 | 判活依据 | ETA |
|---|---|---|---|
| b0fast_seed0(矩阵收官项) | PT 中,日志块缓冲仅头行(223B) | pid 57048 + GPU 92% | fast 路径速度无历史数据,不给数字,下时点用日志字节数增量推 |
| C1 NFH B0'(100ep matched) | PT 中 ~34min(19:57 起),日志 0B 同为缓冲 | pid 14012 + GPU | 粗估 3~4h → LP 自动接续 |

## 四、新完成(自上时点)

- matrix_results.csv 现 **37 数据行**:新增 t3_pow3 三行(19:27/19:28/19:30,s0 −0.40 / s2 −0.10 / s4 −0.27pt,**3/3 符号一致全负 → t3 关线**);LP 形状 bug 已由主会话修复(PowerPool1d),A-P2 三线(h2/t3/h1)全部关闭;
- C1 的 LP 未出(feat/M_c1_nfh_b0_seed0/metrics.json 不存在),无待记账项。

## 五、本小时采取的动作

纯巡检,无自愈写操作(runner 无死亡;无需重启)。

## 六、告警与待办

1. 【挂起·主会话】**b0fast 结束后待合并 w1-code-tasks 进 main 再重启车道**(A1/A2/A3 已在队列,合并前不可执行——main 的 --d1l 仍旧语义、--acl-* 不存在);巡检不代做;
2. 【观察】b0fast/C1 两个 PT 日志均为块缓冲假象(仅头行/0B),下时点用字节数增量+GPU 交叉判活,勿判卡死;
3. 【git】push 断网中(09-20 晚已知抖动),本地提交安全,待网络恢复由巡检自动补推。
