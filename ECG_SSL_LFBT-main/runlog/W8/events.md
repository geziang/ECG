# W8 事件账本

> 纪律: 判定/sanity 先写后跑; 冻结账本只读; 措辞 record-level。

## 10-10 18:1x W8 开工与 E1 预注册

- 挂号: HOSTS §一-E 认领 E1–E7(71eceda), 顺序 E1→E2→E4→E6→E3→E5→E7。
- 仓库代码变更(仅 1 处, E1 前置): `metrics_ext.py` 预测落盘白名单 +W8(W3–W7 不变, W2 硬拒不变); `tests/test_metrics_ext.py` +1 镜像用例, 22/22 绿。默认关闭行为不变(不传 --save-predictions 不触守卫)。
- E1 实现路径=零新代码: `run_lp.py --checkpoint none`(随机初始化冻结特征+线性头, W4 S3 同款 TFS 路径) + `run_ft.py` 省略 --checkpoint(随机初始化全参微调)。驱动脚本 `runlog/W8/run_e1_rand.py`。

**E1 预注册判定(先写后跑, 任务书 sanity 红线的操作化)**:
1. 无晋级门, 18 跑(LP 9 + FT10 9)如实入 `w8_rand_results.csv`。
2. sanity 红线(任务书): LP(rand) 三域均应显著低于 C1。操作化: 任一域 rand 3-seed mean AUROC 距该域 C1 3-seed mean < 2.0pt → 判"接近", 停下上报办公机, 该域数字冻结待裁决, 不擅自继续用于论文表述。C1 参照: ptbxl=W2/W4 账本, cpsc=W5A 新账本(0.9463), chapman=W4 账本。
3. FT10(rand)=从头训练参照, 无 sanity 门, 如实入表。
4. 统计(逐记录配对 bootstrap 10k + DeLong vs C1)由后续 stats 步骤统一算, E1 出数不等待统计。

## 10-10 18:3x E2 预注册与启动

- 实现=零新仓库代码: E2 = W4 S1 全套(`--loss-mode simclr` τ=0.5 默认) + 唯一差异 `--nstdb-aug 0.5`(W7 E5 注入器)。开关纪律单测 `tests/test_w8_e2.py` 4/4: prob=0 逐位恒等 / prob=1 注入生效 / 注入器无状态(两视图独立结构保证) / run_pt 源码守卫断言(nstdb 默认 0.0 + >0 守卫 + loss-mode 默认 bt + τ 默认 0.5)。"默认关==S1 逐位一致"的结构依据=守卫块默认不执行, S1 checkpoint 由同代码路径产出(W4), 无交互项。
- 链: seeds{0,2,4} × PT(NFH 100ep) → LP 三域 → FT10{ptbxl,cpsc}; 主链后补 S1 cpsc FT10 ×3(W4 冻结 simclr checkpoint 纯补格评测, 行 ckpt=simclr-w4frozen, 供 E2 vs S1 在 cpsc FT10 配对——S1 在 W4 只评过 ptbxl FT10)。
- 判定(任务书): 无晋级门如实入表; 配对口径=vs S1(隔离"ECG 增强"单因素, LP 三域+cpsc FT10)+vs C1; 与 C1/Noise(=W7 E5 c1na)/S1 构成 {BT,SimCLR}×{RRC-TO,+NSTDB} 2×2 四格收齐。
- 车道: GPU 车道2(与 E1 车道1 并行, 显存预算 PT~6.7G+LP~2.5G << 24G)。

## 10-10 18:4x E4 预注册与启动

- 实现: `models/vgg_1d.py` +AffineChannel1D(Y=F+γ⊙F+β, γ/β [1,C,1] 零初始化, 与 GRN1D 同位置[block5+maxpool 后 GAP 前]同参数量, 仅去 ⊙n 门控); VGG16 `--trc 2` 分支(trc∈{0,1,2} 断言)。入口零改动(run_pt/run_lp/mbn 均透传 trc)。**checkpoint 安全白名单 +AffineChannel1D**(utils/checkpoint.py——run_pt 双格式保存完整模块对象, 不注册则 weights_only=True 加载必炸, W4 安全门同款教训前置规避)。
- 单测 `tests/test_w8_e4.py` 8/8: trc=0 无校准键/trc=2 零初始化==trc=0 逐位/梯度流通/与 GRN 同γ输出不同(门控确实去掉)/GRN 公式回归守卫/仿射手动公式/非法 trc 拒绝/模块对象受限加载往返逐位。回归 test_w1 14/14 + test_safe_load 3/3 + test_repro 全绿。
- 链: seeds{0,2,4} × PT(NFH 100ep bt trc2) → LP {cpsc, ptbxl}; GPU 车道3(与 E1/E2 并行)。
- 判定(任务书双分支均可写, 无晋级门): 三臂 C1(trc0)/affine(trc2)/C2(trc1), 参照=W5A cpsc/W2 ptbxl 冻结账本同种子配对。读法: TRC>affine 且 affine≈C1 → "响应依赖校准是活性成分"; affine≈TRC → 如实降级 "预训练耦合校准有效"。主读数=cpsc LP AUROC+AUPRC 3-seed mean±SD, ptbxl 为域内观测; "≈"操作化=|Δmean|<0.3pt(种子噪声量级, 描述性非门)。

## 10-10 18:2x-18:4x E2 双发事故与 IO 车道裁决

- 18:22 E2 首发挂死(step0 后无进展, 4 worker 死 1); 18:29 重启后极慢(~0.25 steps/s vs 正常 ~2-5)。py-spy 定位: 主线程饿死在 DataLoader queue.get, 4 worker 全卡 np.load。
- **根因实测**: F: 为 USB 桥接盘(JMicron/USB), E1(LP 特征读写)+E2(PT 34905 npy 随机读)并发把随机 IO 打穿——裸读基准 400 文件/41.6s=9.6 files/s(0.6MB/s), 平均传输延迟 42ms 队列 8。W7 时代 PT 单车道快是因 NFH 2.3GB 进文件缓存后 USB 只承担写。
- **裁决(工程纪律, 非协议改动)**: 本机 PT 类重 IO 任务必须独占 IO 车道串行。E2 已杀干净(py-spy+ps1 双确认); 排程改串行接力=run_w8_relay.py(E1 收官标记或驱动进程退出 → E2 链 → E4 链, 幂等, detached 挂起 18:40)。判活检查走独立 ps1 文件(内联 powershell 引号被 bash 打穿导致首发接力误判, 已修复)。
- 预计新排程: E1(FT10 段 ~5h) → E2(~9.5h) → E4(~6h) 串行; E6/E3/E5/E7 其后。E4 实现已就绪(单测 8/8)等接力。

## 10-10 19:0x E6 预注册(实现完成, 入接力队尾)

- 实现: ①`data_utils/nstdb_aug.py` +`draw_offset(rng,Ln,L,offset_range)`(full=历史行为同式调用; first_half/last_half=时间片段隔离; 短记录回退全域), NoiseInjector 透传 offset_range 默认 full; ②`run_pt.py` +`--nstdb-offset`(默认 full, choices 三值, 仅 --nstdb-aug>0 时生效)。
- 单测 `tests/test_w8_e6.py` 8/8: full==legacy 逐调用一致/两半段区间+不交叠/非法拒绝/短记录回退/注入器默认 full/源码守卫。E2 单测同步更新 offset_range 属性断言(4/4), E4 8/8 保持。
- 链 `run_e6_noiseiso_chain.py`: seeds{0,2,4} × PT(--nstdb-aug 0.5 --nstdb-offset first_half) → LP 三域 clean → 27 格后半段评测。
- **参照口径(预注册)**: C1 冻结 checkpoint 用同一后半段偏移+同一噪声实现(同 _rng_for 种子)重评 27 格(行 ckpt=c1-lasteval), Δ=noiseiso−c1-lasteval 同种子同格配对; 不复用 W6 全域偏移旧 C1 数(避免混入评测口径变化)。
- 判定(任务书门): 隔离版 27 格均值方向为正(与 E5 原版 +2.17/+2.59/+2.59 同向)→"守住非片段记忆"; 方向为负/零→片段记忆嫌疑如实上报。幅度变化如实报告不设门; clean 三域无伤作观测项(容差同 E5 −0.30 参考描述性)。
- 排程: 接力器队列更新为 E2→E4→E6(串行独占 IO 车道)。

## 10-10 19:1x E1 LP 段 sanity 判定(FT10 段进行中)

- 9/9 行齐。rand 3-seed mean AUROC vs C1 参照(ptbxl 0.8848=W2 / cpsc 0.9463=W5A 修正分区 / chapman 0.9964=W2):
  ptbxl 0.4912(gap **39.36pt**) / cpsc 0.5796(gap **36.67pt**) / chapman 0.4372(gap **55.92pt**)。
- **sanity 红线判定: 三域全部大幅显著低于 C1(远超 2.0pt"接近"阈), 红线未触发**——线性可分性不来自数据本身, 论文结论不受限, E1 如实入表 3 地板行。rand chapman 种子间方差大(0.354~0.543)为随机特征常态, 如实保留。
- FT10 段(从头训练参照)18:41:36 起跑, 无 sanity 门。

## 10-10 19:3x E3 实现与预注册(待接力)

- 官方码已 clone 至 F:/新实验/ReverseECG_official(README+net1d.py+pretraining.py 三件, 无 requirements/下游)。pretext=四变体反转检测(generate_reverse)+BCEWithLogitsLoss(2-logit), 官方超参 Adam lr1e-2 wd1e-4 100ep bs128, backbone=Net1D。
- **官方 snippet 两处适配记录**: ①官方按 [N,length] 单导联 2D 处理, 本仓 8 导联推广=时间反转翻末轴/幅值反转逐导联均值中心反转(单测逐导联对拍前 3 块逐位一致); ②官方 ts_reverse=[::-1] 反的是记录轴(与其标签 [1,1] 语义矛盾, N=1 时恒等), 属笔误, 按标签意图实现双重反转。
- 实现: `data_utils/tsr_dataset.py`(build_variant+TSReverseDataset, 变体每次均匀抽 1/4, 无 RRC-TO——官方无增强, pretext 即变换); `run_pt.py`: loss-mode+tsr / --wd 默认 0(历史 Adam 行为不变) / tsr_heads(逐导联 64->2) / forward tsr 分支(BCE 8 导联平均, projector 不参与) / 训练循环 tsr 路由(flag=2 维目标)。单测 `tests/test_w8_e3.py` 6/6(含手工 BCE 对拍+梯度达 backbone), test_w1 14/14 回归绿。
- **预注册回退(先写后跑)**: 官方 lr=1e-2 发散(日志 nan 或 末 loss>4x 初始)-> 一次性改协议 lr=1e-3(wd 不变)重跑并记录; 再失败=如实降级"未复现"(DLC 先例)不硬凑。无晋级门, 出数填表 3。
- 排程: 接力队尾 E2→E4→E6→E3。

## 10-10 19:4x E5/E7 链就绪(殿后入队) + 全批次排程定格

- E5 w8-dvsplit: 两臂 seed0(仅双视角 dv=0.01/rr=0; 仅 RR rr=0.1/dv=0, run_pt 636/643 行已天然分臂)×NFH 100ep→CPSC LP(+ptbxl 观测); 与 W7 E1 联合 3-seed 并列描述性, 主结果仍为联合 3-seed 不据此单项归因(任务书判定)。零新代码(既有 W6 Stage4 开关单臂化)。
- E7 w8-b0trc-chap: W7 E2c 冻结 b0trc(checkpoint/w7_e2c/seed{0,2,4}, trc1)→Chapman LP 纯评测补格, 如实入表 5 注; 补全 2 语料×3 目标域矩阵。
- **接力队列定格: E1(FT10 段)→E2→E4→E6→E3→E5+E7**, run_w8_relay.py detached(PID 轮换以 restart_relay.ps1 为准), 全链幂等可断点续。值班 cron automation-66d31297 每小时:23 看护。
