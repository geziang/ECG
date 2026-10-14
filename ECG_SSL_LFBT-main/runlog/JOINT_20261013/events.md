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

## 2026-10-13 23:38 m3 seed2 PT 完成: 97min, 峰值显存 8156MiB

## 2026-10-13 23:39 N1/N2 落地器 FAIL 已回滚: RuntimeError(详见 logs/apply_n1n2.log), 链不受影响

## 2026-10-14 00:2x 值班#6: m3 s2 收官 + 守夜器事故#4(单测 ROOT 层级)修复

- **m3 s2 全链收官✅**: PT exit=0 wall=97min **vram_peak=8156MiB(_halt 修复后的峰值器首次正常返回)**; LP ptbxl 0.8914/0.6677, cpsc 0.9481/0.7752, chapman 0.9963/0.9884, FT10 0.8650/0.6353(8 行在账); m3 s4 PT 00:10:52 起跑。cpsc AUPRC 两种子对照: s0 0.7696 / s2 0.7752(M2: s2 0.7709)——s2 高于 M2 同种子 +0.43pt, 观察项不做判定。
- **守夜器事故#4(自愈, 零影响)**: 23:38 窗口正确进入+补丁 compile 过+文件已写, 但单测 0/5 全挂于 FileNotFoundError('F:/新实验/run_pt.py')——**tests/test_joint_n1n2.py 的 ROOT 误用 parents[2]**(tests/ 只在根下一层, 应 parents[1]; parents[2] 是 runlog/ 深度文件的约定), 全部测试读错路径必败→按设计回滚→热文件 diff 空逐位还原。修复一行+就地验证: 原始码上 5 测试全因"功能不存在"而 FAIL(AttributeError/KeyError/AssertionError)= 未打补丁的正确预期, 有判别力; 守夜器 00:24:06 重启, 下窗=m3 s4 的 LP/FT 段(~01:5x)。

## 2026-10-14 01:48 m3 seed4 PT 完成: 98min, 峰值显存 8149MiB

## 2026-10-14 01:49 N1/N2 落地器 FAIL 已回滚: RuntimeError(详见 logs/apply_n1n2.log), 链不受影响

## 2026-10-14 02:21 m3 seeds[0, 2, 4] phase=all 段完成

## 2026-10-14 02:3x 值班#8: M3 三种子齐 + 守夜器事故#5 定根因(补丁吞类行)修复

- **M3 三种子全链收官✅✅**: s4 PT exit=0/98min/8149MiB; LP ptbxl 0.8881/0.6497, cpsc 0.9491/0.7661, chapman 0.9960/0.9876, FT10 0.8569/0.6322。**M3 cpsc AUPRC 三种子: s0 0.7696 / s2 0.7752 / s4 0.7661**(M2 同位: s2 0.7709/s4 0.7658); m4 s0(首个噪声配置)02:21 起跑 GPU80%。判定留给统计阶段(3-seed 同向门+CI), 值班只记数。
- **守夜器事故#5 根因(比断言更深)**: 01:49 窗口 3 测试过/2 挂。真根因=**DV_INSERT_BEFORE_CLASS 替换文本末尾漏补回 "class DVDataset:" 行**——类体因缩进连续被解析成 build_rr_perm 内 return 后的死代码, compile 门抓不到(语法合法), 单测的签名测试真实检出("module has no attribute DVDataset")=防御层按设计工作。附带: 源码守卫断言串带冒号与补丁实际行不符(断言 bug)。修复: ①替换文本末尾补回类行; ②断言串对齐实际行; ③**新增全量内存预检**(exec 补丁后模块验 DVDataset 存在+签名默认+行为+守卫串, 5/5 语义全过)——补丁正确性不再依赖烧窗口试错。守夜器 02:3x 重启, 下窗=m4 s0 的 LP/FT 段(~04:00)。
- **复盘**: 五次守夜器事故=①CIM漏行②补丁语法③pytest缺失④测试ROOT层级⑤补丁吞类行+断言串——其中②⑤是补丁/测试真 bug, 均被 compile 门/单测拦下零损失; 语义级预检引入后, 落地成功率应有保障。

## 2026-10-14 03:59 m4 seed0 PT 完成: 98min, 峰值显存 8151MiB

## 2026-10-14 04:00 N1/N2 开关落地完成(单测过, 已提交): 链 phase-1(m3..m6)完成后由值班重启接跑 n1/n2(先 s0)

## 2026-10-14 04:0x 守夜器落地成功✅ + 值班#10: m4 s0 收官 LP 落账

- **守夜器第六次尝试(04:00:26-04:00:32)落地成功**: 04:00 窗口(链 m4 s0 PT 03:59 exit 后的 LP 段)进入→补丁写入→py_compile 过→单测 **RESULT OK(5/5)**→commit a2283f2→`.n1n2_ready` 标记。热文件现带 --dv-rand-window/--rr-shuffle(默认关==原行为); 下一次 run_pt 启动(m4 s2, ~05:05)将首次以补丁后代码运行, 默认关路径与原行为逐位一致(单测守卫)。phase-2 通道就绪: m6 完成后值班重启链自动接跑 n1/n2(s0)。
- **m4 s0 全链推进**: PT exit=0/98min/8151MiB; LP clean 三域落账——ptbxl 0.8936/0.6638, cpsc 0.9525/0.7737, chapman 0.9959/0.9872。**clean 不降反升(vs m3 s0: cpsc +0.39/+0.41pt, ptbxl +0.78/+1.39pt, chapman −0.03/−0.12pt)**——单种子观察项, §7.3 判定(27 格增益+clean 持平)待三种子+噪声评测。FT10 在跑(ep70/100), 其后 27 格噪声评测首战。

## 2026-10-14 05:24 m3 seeds[0, 2, 4] phase=all 段完成

## 2026-10-14 05:2x 值班#11: 链三崩(offs 未定义)修复 + m4 s0 全链收官(27 格噪声首战)

- **链三崩根因(值班定位)**: noise_eval 的 §9 预测落盘块引用 `offs`——该名只存在于 make_perturbed_lasthalf 局部, 04:32 第一格落账后 NameError 崩死(stderr 铁证)。修复: 扰动函数改返 (加噪信号, 偏移量), 噪声片段标识真实落盘。05:25 重启, 45 秒补完剩余 26 格(GPU 热缓存, 特征抽取快), m4 s2 PT 起跑。**m4 s0 cell1 首格崩溃前的指标行有效**(确定性重算逐位一致, 未重复记账)。
- **m4 s0 全链收官✅**: LP clean 三域(ptbxl 0.8936/0.6638, cpsc 0.9525/0.7737, chapman 0.9959/0.9872)+FT10(0.8637/0.6373)+27 格×2(m4+m3-lasteval)。
- **27 格配对总账(s0, Δ=m4−m3-lasteval 同种子同记录同噪声段)**: 总均值 **+2.49pt, 26/27 正**——ptbxl +3.98[+2.14,+8.96] 9/9 正; cpsc +2.12[+0.67,+4.52] 9/9 正; chapman +1.38[−0.01,+4.35] 8/9 正。对照 M5(noiseiso)三种子 +2.20/+2.64/+2.59: 联合模型上的噪声增益与纯噪声分支同量级。§7.3 三条件(27格正+3种子同向/clean 持平)在 s0 全部利好, 判定待三种子。

## 2026-10-14 07:03 m4 seed2 PT 完成: 99min, 峰值显存 8584MiB

## 2026-10-14 09:15 m4 seed4 PT 完成: 99min, 峰值显存 8582MiB

## 2026-10-14 09:48 m4 seeds[0, 2, 4] phase=all 段完成

## 2026-10-14 10:2x 值班#16: M4 三种子齐——§7.3 判定素材全收集✅

- **m4 s4 全链收官**: FT10 0.8566/0.6331(09:23-09:48)+27 格。**M4 噪声版三种子 27 格配对(Δ=m4−m3-lasteval)**: s0 **+2.49pt**(26/27 正) / s2 **+2.10pt**(27/27 正) / s4 **+1.91pt**(26/27 正)——三种子均值全正且逐格 79/81 正。
- **clean 无伤三种子成立**(cpsc AUPRC m4−m3): +0.41/+0.06/+0.35 全非负; chapman/ptbxl 同向(详见 results.csv)。
- **§7.3 判定读数(预注册门)**: ①27 格平均增益为正且三种子方向一致 ✅; ②无噪声测试平均变化持平或更好 ✅; ③无隐瞒下降(chapman s0 有 1 格 −0.01pt、s4 两格负均已在账, 如实保留)✅。M4 作为联合模型的鲁棒版本: 判定素材达标, 正式结论待统计阶段 CI+效应量。
- **a1 s0 PT 09:48:41 起跑**(trc0+dv0.01+rr0+rpeak_nfh 显式)——口径修正后的消融臂开跑, GPU 69%。

## 2026-10-14 11:24 a1 seed0 PT 完成: 96min, 峰值显存 8527MiB

## 2026-10-14 12:3x 主会话检查: A1 s0 收官(正确缓存首数) + 值班 cron 消失重建

- **a1 s0 全链收官✅**(exit=0/96min/8527MiB): LP ptbxl 0.8858/0.6572, **cpsc 0.9467/0.7640**, chapman 0.9961/0.9880, FT10 0.8592/0.6351。对照被作废的 W8 E5 污染值(dvo s0 0.9444/0.7544): **+0.23pt AUROC/+0.96pt AUPRC**——正确缓存下单臂数字显著不同, 口径修正重跑的必要性被数据证实。a1 s2 11:56 起跑。
- **值班 cron 中断与重建**: 原 automation-9fc9eceb 值守 17 轮(10-13 19:26 → 10-14 11:23)后随宿主会话重启消失(已知行为: automation 不随会话迁移, CronList 为空证实); 12:23 一轮漏值, 无害(链自驱健康)。12:3x 重建 automation-14582a0e(同款提示词, 13:23 起恢复)。**教训固化: 每次主会话苏醒先 CronList 核验, 空则立即重建。**
- 10-14 13:0x 用户指令: 值班 cron 改为**每 2 小时一轮**(automation-14582a0e, 23 */2 * * *); 45min 去重门保留(防同轮双投递, 与巡检间隔无关), >3h 无进展判死门不变。链当前: a1 s2 PT 在跑(11:56 起)。

## 2026-10-14 13:32 a1 seed2 PT 完成: 96min, 峰值显存 8616MiB

## 2026-10-14 15:40 a1 seed4 PT 完成: 95min, 峰值显存 8459MiB

## 2026-10-14 16:12 a1 seeds[0, 2, 4] phase=all 段完成

## 2026-10-14 17:26 a2 seed0 PT 完成: 73min, 峰值显存 7755MiB

## 2026-10-14 19:11 a2 seed2 PT 完成: 73min, 峰值显存 7734MiB

## 2026-10-14 20:56 a2 seed4 PT 完成: 73min, 峰值显存 7773MiB

## 2026-10-14 21:29 a2 seeds[0, 2, 4] phase=all 段完成

## 2026-10-14 22:2x 值班#22: A2 三种子齐(单臂水位定调) + M6 起跑(phase-1 最后一组)

- **a2 s4 全链收官**(73min/7773MiB): LP 0.8877/0.6512, cpsc 0.9465/0.7583, chapman 0.9962/0.9882, FT10 0.8542/0.6275。
- **A2(RR-only, 正确缓存)三种子定调**: cpsc AUPRC 0.7669/0.7583/0.7583 vs C1 同位 0.7589/0.7583/0.7627 → **+0.80/±0.00/−0.44pt, 均值约 +0.12pt**。值班叙述诚实修正: 16:23 值班与主会话答疑基于 s0 说的"RR 单臂≈TRC 量级增益"系种子噪声, 三种子定调为**单臂≈基线水平**。当前消融轮廓(cpsc AUPRC 均值): C1≈0.760 < A1≈0.762≈A2≈0.761 << M2≈0.768≈M1≈0.768 < M3≈0.770——单臂边缘、成对/三联增益明确, 精确归因(交互效应 I)留给统计阶段。
- **m6 s0 21:29:26 起跑**(trc1+nstdb, 非 DV 路径 ~75min/seed)——phase-1 最后一组三种子; 之后 phase-2 n1/n2(值班按 .n1n2_ready 条件自动接力)。

## 2026-10-14 22:42 m6 seed0 PT 完成: 73min, 峰值显存 7783MiB
- 10-15 00:28 值班#23: **m6 s0 全链收官**(LP+FT10+27格): 噪声配对 m6−m1-lasteval **+2.21pt/26正**——与 m4(+2.49)/M5(~+2.48)同量级, '噪声增益不依赖 Beat-RR'机制问题 s0 层面得答; m6 s2 23:15 起跑(GPU84%)零异常。phase-1 余 m6 s2/s4, 预计 ~03:4x 全链收官→phase-2 n1/n2。
