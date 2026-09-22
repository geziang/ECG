# 交接文档(HANDOFF)— 主机A实验状态快照

> 生成:2026-09-18 10:55(会话迁移交接)。新会话接手:先读本文件,再按需读 §五 索引的文件。
> 巡检自动化(每小时)持续运行中,正常时无需人工介入。

## 一、身份与环境(必读)

- **我们是主机A = `Win4090`**(多主机协作中的 A 机);工作区 **`F:\新实验`**,git 仓库 = github.com/geziang/ECG;
- 实验主目录:`F:\新实验\ECG_SSL_LFBT-main\`(所有命令在此执行);
- Python:`C:\Users\admin\.conda\envs\DL\python.exe`(torch 2.0.0+cu118);
- 其他文件夹:`F:\归档文件夹\LBTF创新`=只读资料库(E001证据链/代码包),`D:\edge下载文件`=报告文稿;
- 主机B = DESKTOP-0PBLCND(3080 10G,单车道),在跑它自己的 B0 锚点,互不干涉。

## 二、项目一句话与总原则

LFBT(多导联心电自监督)基线改进:**B0 锚点 LP AUPRC 0.7177 是一切 Δ 的分母**;判定纪律 = seed0 粗探 → Δ>0 才 3-seed(0,2,4)符号一致门;协议冻结(200ep/batch128/Adam1e-3/RRC-TO原值/PTB-XL单库)。**裁剪轴、掩码轴已永久关闭,任何任务不得重启**。台账 `runlog/HOSTS.md` 是多主机唯一任务认领依据,开工前必查。

## 三、当前系统状态(10:55 快照)

- **在跑**:`arb3_base` LP(手动恢复,epoch 90+/100)→ 之后 `psfull_arb3` LP(**判决读数,预计 ~11:05**,判读对 arb3 配对,Δ≥+0.003 触发 3-seed 确认);`common_view01` 预训练(epoch ~95/200,车道C,ETA ~12:00);
- **车道**:C(旧调度,跑 common_view01)、F、G(新调度,合并版代码)在等槽位;探针并发上限已升 **3**(mimic 已退出,显存富余);e006 类(psfull)独占运行;
- **每小时巡检自动化**:automation-150aa802(整点触发,自检/自愈/写 `runlog/M/hourly_report.md`/尝试推送);
- **过夜事故已恢复**:rebase 在训练运行中换文件 → d7_rec01 在 epoch28 崩溃挂死(已清理,**待重跑**,车道F会自动接);psfull/arb3 预训练其实都成功(checkpoint 在),LP 已修复路径补跑中;
- **关键教训(已入长期记忆)**:训练进程存活时,绝不在仓库内做 rebase/checkout/改被 import 的 .py(Windows DataLoader 每 epoch 从磁盘重 import);合并代码用 `git worktree` 或等车道空闲。

## 四、待办与队列

- 队列 `runlog/M/pipeline_queue.yaml`(23 项):d7_rec01 重跑 → proj_dim1024/4096 → **A-P2 五探针**(h2_blur3/t3_pow3/h1_white8 各预注册 seeds 0,2,4 + h1 两 NEG)→ 队尾跨域四件套 + b0fast;
- **h1 去留待裁决**:对 `feat/h5_b0_regen`(已验证的 B0 特征)跑 `python diagnose_rank.py --feat-dir feat/h5_b0_regen`,erank/d≥0.5(不塌缩)则把 h1 三条从队列删除;
- **H5 校准待跑**:`python calibrate_lp.py --feat-dir feat/h5_b0_regen --name b0`(新锚点冻结需人工确认后写 HOSTS §五-2,冻结前一切 Δ 仍用旧口径);
- **git 待推**:本地 2 个提交(并发3升级、巡检建规)因 GitHub 间歇不可达未推,`git -c http.lowSpeedLimit=0 -c http.lowSpeedTime=999 push -q origin main` 重试;
- 判定账:matrix_results.csv 16 行(1 正=crop_weak s0,已被 3-seed 门判噪声;其余全负);d7_rec003 ❌ −0.19pt(今晨);FT10×B0 锚点 ✅ 0.8756/0.6432。

## 五、文件索引

| 要看什么 | 文件 |
|---|---|
| 每小时巡检报告(最新状态首选) | `runlog/M/hourly_report.md` |
| 全部判定数字 | `runlog/M/matrix_results.csv` |
| 车道实时日志 | `runlog/M/lane{C,F,G}.out.log`、`runlog/S1/overnight_summary.md` |
| 任务台账(多主机) | `runlog/HOSTS.md`(§三=主机A任务,§四=主机B,§五=红线) |
| 队列定义 | `runlog/M/pipeline_queue.yaml` |
| 完整实验报告(总分总+公式) | `报告归档/历史文档/LFBT基线改进实验报告-2026-09-17-完善版.md` |
| 流水线代码 | `pipeline_runner.py`(闸门/准入/防饿死),入口 `--lane <字母>` |

## 六、接手后第一句话该做什么

1. `cat runlog/M/hourly_report.md`(或等下一个整点巡检)拿最新状态;
2. 若 psfull LP 已出:读 `runlog/M/lp_psfull_arb3_seed0/` 的 json,算 Δ=psfull−arb3,≥+0.003 则在队列加 psfull 的 seed2/4(补行进 csv 后由车道自动跑);
3. 跑 §四 的 h1 秩诊断复核与 H5 校准(两个都是分钟级 CPU/轻GPU任务);
4. 尝试 git push。
