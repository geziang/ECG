# JOINT_20261013 事件台账

> 批次=TRC–Beat-RR–Noise 联合验证(protocol.md);执行台账,只追加。判定/协议原文见 protocol.md,执行细则见 任务书.md。

## 2026-10-13 19:1x 批次开工前盘点与两项裁决(主机A)

- git 同步得 protocol.md(5072574);锚点映射完成(任务书 §一):M0=confirm/c1、M1=confirm/c2、M2=w6_dv/seed0+w7_dv/seed{2,4}(rpeak_nfh ✓×3)、M5=w8_noiseiso(时间隔离版,与 protocol §2 口径一致)。
- **preflight 全过**(preflight.md):rpeak_nfh↔NFH 一一对应 34,905;有效 R 峰 99.98%(7 条空);噪声两视图独立/偏移域/RR z-score 来源/CPSC 1385 均核验。
- **裁决一(A1/A2 s0 口径污染, 任务书 §二)**:W8 E5 dvo/rro s0 config 的 rpeak_npz=pt_rpeaks.npz(17,418 条=旧 PTB-XL 语料;E5 链模板漏传 --rpeak-npz)——NFH 前 17,418 条拿错语料 R 峰、后 17,487 条无键。A1/A2 改三种子全部重跑(显式 rpeak_nfh),最低新增 PT 10→12 次;w8_dvsplit_results.csv 冻结只读,本条即更正记录,W8 报告引用此条。
- **裁决二(排程)**:smoke→M3→M4→A1→A2→M6(含)串行;N1/N2 暂缓待 M3 落地。预算 29-31h(不含 M6)~38-40h(含)。

## 2026-10-13 18:2x-18:5x S0 smoke 全过 + 正式链启动

- **smoke 四项验收全过**(任务书 §五):①5ep 全开关(trc1+dv0.01+rr0.1+nstdb0.5/first_half)exit0,每 100 步 JSON 含 loss_dv/loss_rr 且有限(loss_dv 1128→246↓, loss_rr 1.20→0.60↓),[nstdb-aug] banner 出现;②续训 --epochs 6: `[resume] 恢复自 epoch 4, 从 5 续训`,损失连续(ep4 均值 557 → ep5 均值 520);③LP --trc 1 加载联合 checkpoint exit0 出 metrics(6ep 垃圾编码仅验机制);④config.json 全开关落盘(rpeak_npz=pt_rpeaks_nfh.npz)。
- checkpoint 内容核验:train_state 含 model/optimizer/epoch/torch_rng;rr_head 完整模块入 train_state;TRC 权重=encoder `model.5.gamma/beta`(trc=0 加载报 unexpected——LP 必须 --trc 1,链已对齐)。
- 实测定标:**59s/ep(全开关)** → 100ep≈98min/seed;预算更新:最小配置(12 PT)≈26h PT + 评测 ≈ 30h;含 M6 ≈ 33h。
- **18:48 正式链启动**(launch_joint.ps1 detached, pid 30880):m3 s0 PT 开跑,命令与 protocol §4.3 逐字一致;值班 cron automation-9fc9eceb 每小时 :23 看护。

## 2026-10-13 19:38 N1/N2 落地器 FAIL 已回滚: CalledProcessError(详见 logs/apply_n1n2.log), 链不受影响

## 2026-10-13 19:4x 用户指令扩展 + N1/N2/统计件就位 + 守夜器事故与修复(主机A)

- **用户 10-13 指令**:"无效实验重新做,需要补的实验做实验补充,最新实验需求一并做"——A1/A2 重跑与 M3/M4/M6 均已在链;据此把 protocol §3.3 的 N1/N2 反事实(先 s0)也入队执行。
- **N1/N2 入队**:run_joint_chain.py 增 n1(M3+随机心拍窗口)/n2(M3+打乱 RR 标签)配置(仅 s0,殿后于 m6);noise_eval 同步改造为逐记录预测+噪声偏移落盘(protocol §9,行已记但预测缺失时重算补存,收官可 --phase noise 补跑)。
- **守夜器事故(重要教训, 已修复)**: apply_n1n2.py 首发在 19:38 于 m3 PT 存活期间误判"安全窗口"并写入热文件——根因①判活探针走 CIM 全枚举, 撞上本机已知"CIM 随机漏行"病(09-20 b0fast 误报同源);根因②补丁文本一处缺右括号+锚点只取首行致 help 行重复, py_compile 失败。**链的 m3 PT(pid 17892)经查毫发无损**(1 秒坏文件窗口未撞上 DataLoader 重导入, epoch 50/100 正常), 原件已自动回滚逐位恢复。修复:①判活改双通道(链分段日志为主=链进程自写免疫 WMI 抽风, CIM 兜底, 任一在跑即视为在跑)+进入窗口后 20s 双重确认;②补丁先 compile() 内存预检再落盘;③锚点改两行完整语句。守夜器 19:44 重启(Start-Process 分离), 现守候首个安全窗口(预计 ~20:30 m3 s0 PT 结束后的 LP/FT 段), 落地成功后 touch .n1n2_ready, 值班 cron 据 phase-2 条件在 phase-1 完成后重启链接跑 n1/n2。
- **统计件就位并验证**: run_joint_stats.py(配对 bootstrap 10k 双指标[AUPRC 主/AUROC 次]+DeLong+交互效应 I 逐种子+protocol §10 三表+锚点预测重放[confirm 工件无 y_prob, X_test+classifier CPU 前向, 对拍冻结 auroc |Δ|<5e-3])。smoke 验证: m1:m0 cpsc AUPRC Δ+0.830pt CI[+0.295,+1.315] p=0.001 / AUROC +0.103pt p=0.206——与 W6 Stage0/W5 官方冻结数逐位吻合, 机器可直接用于收官。RNG=20261013, N_BOOT=10000, 全量约 2-3h CPU。

## 2026-10-13 21:26 N1/N2 落地器 FAIL 已回滚: RuntimeError(详见 logs/apply_n1n2.log), 链不受影响

## 2026-10-13 21:2x 值班#3: 链双崩事故定位修复 + 恢复 + m3 s0 首数落账

- **现象**: m3 s0 PT 20:28 正常收官(checkpoint 三件齐, SHA b46b382b)但链无下文; 21:23 值班发现链进程(pid30880)已死, stderr 残留 TypeError: 'Event' object is not callable(chain.out.log.err, 重启已被覆写, 本文即存档)。
- **根因一(VRamPoller 线程陷阱)**: 轮询线程把 threading.Event 存为 self._stop, 遮蔽 threading.Thread 内部私有方法 Thread._stop(); PT 完成时 stop()→join()→线程收尾内部 self._stop() 调用撞上 Event → TypeError → 链在 subprocess 返回后、写"PT: exit="行前崩死。修复: 属性改名 _halt(附注释固化教训), 回归自测 start/set/join 通过。PT 孤儿跑完无损; manifest m3/0 条目由本轮手工补录(wall≈5952s, vram 8353=巡检抽样[峰值器崩前未返回], sha 与 PT 日志打印值一致)。
- **根因二(预测白名单)**: 重启后 m3 s0 LP 训练成功但落盘被 metrics_ext._assert_w3_dir 拒——白名单只认 runlog/W3..W8, 不含 JOINT_20261013 → 链二次退出。修复: 白名单追加 /runlog/joint_20261013/(W2 硬拒与其余拒绝不变, 单测 4 例: JOINT 过/W2/M/tmp 拒)。
- **恢复**: 21:27:58 链三启(pid 27228, 配置含 n1/n2), m3 s0 PT 幂等跳过, LP 出数: ptbxl 0.8858/0.6499, cpsc 0.9486/0.7696(单种子, 不做判定; 对照家族: M2-s2 cpsc 0.9479/0.7709, M2-s4 0.9498/0.7658——同区间)。chapman 在跑→FT10→m3 s2。
- **守夜器状态**: 仍存活守候(pid 30240)。其链日志通道要求出现"PT: exit="行才认窗口——m3 s2 PT 完成后(~23:1x)自动解锁, 预计在 m3 s2 的 LP/FT 段落地 N1/N2 补丁; 对 phase-2(m6 后)时间表无影响。

## 2026-10-13 22:2x 值班#4: 守夜器第三次自愈(DL env 无 pytest) + m3 s0 全链收官

- **m3 s0 全链收官✅**: LP×3(ptbxl 0.8858/0.6499, cpsc 0.9486/0.7696, chapman 0.9962/0.9884)+FT10(0.8607/0.6361)4 行落账; m3 s2 PT 22:00:51 起跑(epoch 24/100 @22:24, 正常)。
- **守夜器事故#3(自愈, 零影响)**: 21:26:26 在真实安全窗口(链二崩与三启之间的空档, 双通道双确认均按设计工作)落地补丁——compile 预检通过、文件已写入, 但**测试步骤调用 pytest 而 DL env 未安装 pytest**("No module named pytest")→ 按设计回滚原件退出。核验: 热文件 git diff 为空逐位还原, 22:00 m3 s2 PT 用原始代码起跑。修复: 单测改自包含运行器(importlib+test_* 逐函数, 不依赖 pytest), 22:24:48 重启守候(被 m3 s2 PT 正确阻塞, 下一个窗口=m3 s2 的 LP/FT 段 ~23:4x)。
- **防御体系复盘**: 三次守夜器事故(19:38 CIM 误判/19:38 补丁语法/21:26 pytest 缺失)全部被"compile 预检+锚点断言+自动回滚"拦在数据损失之前, 在跑训练零受影响——机制按预期失效安全。
