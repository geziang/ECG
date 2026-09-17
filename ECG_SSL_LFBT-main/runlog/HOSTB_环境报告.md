# 主机B 环境登记报告(DESKTOP-0PBLCND / RTX 3080)

> 登记时间:2026-09-17 19:10 | 工程路径:`E:\GZA\ECG-main\ECG_SSL_LFBT-main`
> 本机已按 HOSTS.md §五-6 完成数据准备并启动入场 B0 锚点,**不认领 §二/§三 任何任务,等待主机A 按 §四 分配创新方法方向。**

## 一、硬件

| 项目 | 配置 | 备注 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3080(10,240 MiB) | 空闲约 9.3 GiB;桌面+远程工具常驻占约 0.9 GiB |
| 驱动 | 560.94 | 支持到 CUDA 12.6 |
| CPU | Intel i9-9900K(8 核 16 线程) | 单车道 workers 6 足够 |
| 内存 | 64 GB | 空闲约 47 GB |
| 磁盘 | E: 可用约 200 GB;C: 仅剩 18 GB(97% 满) | 数据/工程全在 E 盘;勿装 C 盘 |
| 系统 | Windows 10(10.0.17763) | 远程:ToDesk + 向日葵常驻 |

## 二、软件环境(conda env `DL`)

- Python 3.11.9
- torch 2.5.1+cu121 / torchvision 0.20.1+cu121 —— **CUDA 实测可用**(GPU 识别 RTX 3080,`torch.cuda.is_available()=True`)
- numpy 1.26.3 / scipy 1.13.1 / scikit-learn 1.5.0 / pandas 2.2.2 / tqdm 4.64.1 / wfdb 4.1.2

**与主机A 差异**:Py 3.11.9 vs 3.10.11;torch 2.5.1+cu121 vs 2.0.0+cu118。核心训练栈 API 兼容;B0 锚点即跨环境校验,本机一切 Δ 只对本机锚点(红线 §五-1)。

## 三、算力定位与流水线适配

- **单车道机器**:单预训练峰值 6.74 GiB,10 GiB 卡仅容 1 条预训练车道(无法像主机A 开 2~3 车道);预训练期间 LP 间隙约 2.5 GiB 可用。
- `pipeline_runner.py` 已按 10G 适配(仅本机副本):`wait_slot` 并发上限 3→**1**;两处显存闸门 7800→**7000 MiB**;其余防撞机制(claim/幂等/活进程扫描)未动。
- 单 run(预训练 200ep + LP)预计 **4.5~7 小时**(主机A 4090 实测 2~2.8h,3080 约慢一倍)。

## 四、数据状态(已就绪并校验)

- 原始:`E:\GZA\ptb-xl`(g1~g22 / HRxxxxx.mat,21,837 个 .mat + 官方 ptbxl_database.csv / scp_statements.csv)。
- **数据质量事故与修复**:发现 5 个 .mat 内容损坏(大小正常、头部乱码):g6/HR05097、HR05098、HR05099、HR05101 与 g7/HR06214;G 盘备份拷贝同样损坏(源头坏)。已用本机 F 盘官方 PhysioNet 原版(records500)重建,验证标准:.mat 数据与官方 wfdb 数字信号(d_signal)**逐位一致**,loadmat 复验 5/5 通过。
- 转换(prepare_data.py,冻结协议)结果,**与红线数字完全一致**:
  - pt_pretrain(folds 1–8)= **17,418**
  - ptbxl train/val/test = **13,639 / 1,714 / 1,739**
  - 五类:NORM 9,480 / MI 2,532 / STTC 2,428 / CD 2,115 / HYP 537(合计 17,092)
  - `data/manifest.json` 已生成。
- 说明:主机A 的 B0 checkpoint SHA256(026033d5…)无法在本机核对(checkpoint 不入 git);本机以"样本数+CSV 划分逻辑完全复现"作为数据一致性依据,本机 B0 锚点结果另落盘。

## 五、当前任务状态

| 任务 | 状态 | 启动 | 预计完成 |
|---|---|---|---|
| 入场 B0 锚点(§五-2 全默认协议) | 🏃 运行中(PID 23560,GPU 7.3 GiB) | 09-17 19:03 | 09-17 深夜~09-18 凌晨(4.5~7h) |

- 结果自动追加 `runlog/M/matrix_results_hostB.csv`;checkpoint 存 `checkpoint/ptxl_gamma08/`。
- 驱动脚本 `runlog/hostB_b0_anchor.sh`(幂等:预训练以日志含 SHA256 为准,LP 以解析出 AUPRC 为准)。

## 六、维护注意

1. 训练期间请勿在本机开启其他大显存程序;ToDesk/向日葵留一个即可,GeForce Experience 后台录制建议关闭。
2. `C:\Users\508\.condarc` 含 GBK 编码中文注释导致 conda 初始化报错(不影响训练,待修)。
3. **GitHub 连通性波动**(直连时通时断;本地代理 127.0.0.1:7897 可用但常关)。工程副本 E:\GZA\ECG-main 无 .git;推送走独立 git 副本 E:\GZA\ECG-push-tmp(clone 自远程),本报告与 HOSTS.md 登记已 push(commit f957df3)。
4. 本机对仓库的改动清单:`runlog/HOSTB_环境报告.md`(新增)、`runlog/HOSTS.md`(主机B 登记)、`pipeline_runner.py`(10G 闸门适配,若合并回主仓请注意主机A 的 24G 参数是 7800/3)、`runlog/hostB_b0_anchor.sh`(新增)。

---
*主机B(DESKTOP-0PBLCND)呈报,2026-09-17 19:10。*
