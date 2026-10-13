# JOINT_20261013 值班巡检日志

> 每轮一行:时间/链状态/最新完成段/异常与否。判活口径=PID(CIM)+GPU util 交叉,不单看日志 mtime。

| 10-13 19:26 | 链活(pid30880)+PT子进程(pid17892),GPU 62%/8.3G | m3 s0 PT 进行中 epoch 36/100(~63s/ep,与smoke定标一致,ETA ~20:33 完PT);损失健康 loss267/dv191/rr0.40 无nan | 无异常(首巡;results.csv/manifest 未生成=预期,首个PT未完) |
| 10-13 20:23 | 链活(chain+PT+watcher 3进程),GPU 82%/8.1G | m3 s0 PT 94/100ep(ETA ~20:30 收PT→LP段;守夜器守候首个安全窗口中) | 无异常(results.csv 未生成=PT未完预期;零FAIL) |
| 10-13 21:35 | 链恢复(pid27228 三启)跑通LP段, GPU 正常 | m3 s0: PT完成(幂等跳过)+ptbxl/cpsc LP 落账, chapman/FT10 在跑; 双崩根因(VRamPoller _stop 遮蔽+白名单)已修复 | 异常已处置: 链20:28/21:26两崩均定位修复, 零数据损失, m3 s0 首数 0.8858/0.6499+0.9486/0.7696 |
