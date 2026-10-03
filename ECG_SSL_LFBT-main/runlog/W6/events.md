# W6 事件记录

## 2026-10-03 15:2x · Stage 6 前置: NSTDB 下载完成（车道: CPU, 按排程 Stage 0 后接续）
- 直连 `/files/nstdb/1.0.0/*` 被 CDN 反爬拦(403, 换 UA 无效); 改走 `/content/nstdb/get-zip/1.0.0/` 整库 zip(70.9MB) 成功。
- 当前版本噪声记录名为 **bw / ma / em**(各 2 通道 650000 样本 @360Hz, 两段原始记录并为双通道), 非旧 physiobank 时代的 bw1/bw2/ma1/ma2/em1/em2——Stage 6 评估代码取噪声时按新命名。
- 产物: `data/nstdb/{bw,ma,em}.{dat,hea}` + `nstdb.txt` + `SHA256SUMS.txt`, 6/6 校验 OK; 原始 zip 留存 `data/nstdb_full.zip`(data/ 不入 git)。
- 预处理(重采样/映射到 12 导)留待 Stage 6 开工时与评估代码一并实现。

## 2026-10-03 14:19 · L2 首发错峰不足, 主动停发重发（车道: L2）
- 现象: L1 于 14:13:58 起跑后, L2 于 14:19:09 误发(仅错峰 5min11s < 纪律下限 10min)——时间估算失误。
- 处置: 30 秒内发现, 杀掉 L2 全进程树(runner+run_ft+workers), 清理其部分产物(ft/predictions/logs, 均未到 test 阶段, 无账本污染); 幂等设计保证 14:24 后重发自动从 c2 ft20 seed0 干净重来。
- 附带发现: ①杀进程树不能按"启动时间>阈值"扫杀——run_ft 每 epoch 重spawn dataloader worker(父PID=run_ft), 扫杀会误伤在跑车道, 今后只按核验过命令行的精确 PID 杀; ②本次扫杀窗口碰巧命中 L1 换界 worker, L1 存活且进度正常(Epoch 10→19), 未受实质影响。

## 2026-10-03 14:0x · Stage 0 命令勘误（车道: CPU）
- 现象: 按任务书原样命令(默认 pairs)起跑后, `c1:s1`/`c1:s2` 两对被 skip——`paired_stats.py` 的 DEFAULT_PAIRS 用占位名 s1/s2, 而 W5/W4 预测目录实名 `simclr_/clocs_`。
- 依据: W4 官方产物 `runlog/W4/stats/paired_stats.csv` 中 method_b 即为 simclr/clocs, 证明 W4 当次为显式 `--pairs c1:c2 c1:simclr c1:clocs c2:b0` 传参。
- 处置: 停残跑(仅完成 c1:c2 一对), 以显式 --pairs 重启; **代码一行未改**, 四对齐全, 与任务书"默认四对"语义一致。
- 顺手清理: 09-25 遗留 4 个 python 陈尸进程(W5A dataloader worker, 0 CPU 不占卡)已按重启恢复手册清除。

## 2026-10-03 19:2x · GitHub 定向阻断持续（运维）
- 用户 19:2x 通报"网络已恢复"(办公机侧); A 机实测: DNS 解析正常(20.205.243.166), physionet/baidu 均 200, 唯 github.com 443 连接超时 → 属 GitHub 定向阻断而非本机断网, 与 W4 时代小时级瞬态同款。
- 积压 commit 2 条(巡检#1/#2)+本条, 留本地待窗口自动补推; 车道运行不受影响(训练全部本地)。

## 2026-10-03 19:3x · GitHub 断流根因定位并修复（运维, 用户互动）
- 用户报"网络已恢复"但 A 机 git 仍连不上 → 排查发现: Windows 系统代理 127.0.0.1:7897 已开(用户恢复的是代理客户端), 浏览器等走代理故"感觉恢复"; git 无代理配置一直直连 github.com 被定向阻断超时; 与凭据/权限无关(16:23 前同凭据推送正常)。
- 修复: `git config --global http.https://github.com.proxy http://127.0.0.1:7897`(仅 github.com 走本机代理, 其余地址不受影响), 经代理实测 github 200/2.4s, 积压 3 commit 补推成功(1bec73a..bf44f24)。
- 留给后续巡检: 若再遇 push 失败且报错为"connection refused 127.0.0.1:7897"=代理客户端退出(非断网), 等代理恢复即自愈, 勿改回直连。
