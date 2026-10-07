# W7 自主工作队列（用户 24 小时委托 2026-10-06 起）

> **委托依据**：用户 10-06 指令（见 W7_任务书.md 头注）。**执行者**：主会话 + 每小时值班轮。
> **每轮只执行一个编号步骤**（可验证粒度），完成后勾选并 commit+push。卡壳→落 events.md 跳过不停车。
> 纪律同 W6：协议超参不动；W2–W6 冻结账本只读；test 单发；新代码默认关==基线逐位一致；判负也是交付物。
> 环境提醒：python 必须 `C:/Users/admin/.conda/envs/DL/python.exe`（repo venv 死链）；日志判活看字节增长/SM%。

## Q1 · E4d 心拍分层误差分析【主会话已实现 run_e4d_beat_stratified.py】
- [x] Q1a 值班轮：确认 R 峰三域 test 缓存生成完毕（data/pt_rpeaks_test_{ptbxl,cpsc,chapman}.npz，脚本幂等可续）；慢则等待，快则进入 Q1b。
- [x] Q1b 分析跑完后核验产物三件（e4d_rr_features.csv / e4d_bucket_summary.csv / e4d_summary.md），抽查 3 行与 W6 metrics_ext 对账（accuracy 全体值应一致），events 记录关键读数（HR/SDNN 分桶的 b0−c2 方向）。

## Q2 · E2d TRC 机理分析【主会话实现中】
- [x] Q2a 产物核验（e2d_channel_stats.csv / e2d_summary.md），重点：通道统计跨域距离与 CPSC 增益的相关系数方向；events 记录。

## Q3 · E4c 患者级敏感性附录（PTB-XL）【需实现，规格如下】
- [x] Q3a 实现 `runlog/W7/run_e4c_patient_lp.py`（主会话完成: manifest.json 含 patient_id, K=3 硬链接 splits 已建 70/10/20, LP 车道已启动 logs/e4c_lp_all2.log）：读 data/ptbxl 元数据 patient_id（侦察：ptbxl 原始 csv 在 data/ptbxl_* 或仓库何处，sample_XXXXX ↔ ecg_id 映射来自制备脚本——先 grep prepare 脚本确认）；按患者级分层重划 K=3 次（seed 101/102/103，train/val/test 比例对齐 70/10/20 患者比例），新 split 写 data/ptbxl_patient_k{i}/（软链或复制 .npy，磁盘不够则复制清单+DatasetFolder 子类按清单过滤）。
- [x] Q3b 跑 LP：9/9 格完成(23:56), 账本 e4c_patient_lp.csv（run_lp.py --data-dir 指向新 split 根；checkpoint 用 W2 冻结 SHA 校验版；protocol_id=w7-e4c-k{i}），9 跑 GPU 各~2min；账本 e4c_patient_lp.csv（含原 split AUROC 对照行）。
- [x] Q3c 汇总 e4c_summary.md 完成——排序 b0≫c2≥c1 三次重划全稳健; c2 三次一致小幅领先 c1(官方split两者打平); 绝对值 +1.7~+4.2pt 属重划测试池组成差异, 措辞=敏感性分析

## Q4 · E4b 质量门控推理【需实现，规格如下】
- [x] Q4a 实现(09:2x 轮): runlog/W7/run_e4b_quality_gate.py——常量零导联特征预计算+τ 档特征替换; val 扫描已跑完
- [x] Q4b **判负关线**(val 阶段 09:30): 三域无 τ 满足 clean≥−0.10pt 约束(cpsc τ=0.3 即 −1.20pt/chapman −0.20pt/ptbxl 亦负), 按'仍伤则关线'预注册分支执行; 质量分数(带内功率比)无法区分干净低幅与受扰导联。Q4c 不执行(test 零消耗)。
- [ ] Q4c NSTDB 扰动评估：{b0,c1,c2}×seed0×三域×3噪声×SNR{0,5,10} 子集（复用 run_robustness_nstdb.py 机制改推理侧），对照"固定 mask"同格读数；噪声条件 3-domain 平均 ≥+0.5pt 判正，e4b_quality_gate.csv + e4b_summary.md。

## Q5 · E4a LP-FT 机制实验（CPSC）【需实现，规格如下】
- [x] Q5a 实现（主会话完成: run_ft.py 加 --head-init 默认关=基线不变, metrics_ext.py 预测白名单扩 W7, 单测 21/21; 驱动 runlog/W7/run_e4a_lpft.py 就绪）（run_lp 同款至收敛）→ 以该头初始化+encoder 全解冻跑 run_ft 同款 FT10；对象 {b0,c2}×seeds{0,2,4}（c1 作参考可选）；protocol_id=w7-e4a；full-FT/LP 读数直接引用 W6 Stage3/W5 账本不重跑。
- [x] Q5b 6/6 完成(08:21)+终表: LP 0.9506/0.9473, naive-FT 0.9192/0.9226, LP-FT **0.9498/0.9431**(b0/c2)——LP-FT 3/3 保 b0>c2, **微调反转消失**; LPFT−FT 增益 b0+3.06pt>c2+2.05pt(naive-FT 对 b0 形变更大); 机制=反转系 naive-FT 形变产物, 对接 Kumar 2022。

## Q6 · 第二批预训练链（第一批 GPU 空档即排，Q1/Q2 不占 GPU 时可先发 E1）
- [x] Q6a 全清: seed2 链 01:46 收官(4/4, cpsc Δ+0.26 同向); seed4 链 02:24:09 启动(修复 save-predictions 路径 W6→W7, seed2 预测4目录已迁回 W7); 待 seed4 完成后 Q6b 门判定（日志 lane_E1_dv_s2.log），链结束且 dv_results.csv 出现 seed=2 四行后跑 seed 4；**注意该脚本位置在 W6 目录但账本追加 dv_results.csv 属 W6 文件——先把脚本复制为 runlog/W7/run_e1_dv_chain.py 并把 OUT 改为 runlog/W7/e1_dv_results.csv 后再跑**（W6 冻结账本只读红线）。
- [x] Q6b E1 门判定(10-07 04:32 链收官, 05:23 轮正式落笔)：s0 +0.27 / s2 +0.26 / s4 +0.17, **3/3 同向, mean=+0.2333pt ∈[+0.10,+0.30) → 温和正结论**(未达+0.30确认线)。心拍线最终定格: 证据链 = CCM −0.91 / 多段 −0.17 / 双视角+RR头 seed0 +0.27 → 三种子复核 3/3 同向 mean+0.2333。判定=温和正, 线就此收束(不再加种子)。
- [ ] Q6c E2a FT期TRC：实现 run_pt.py 侧 `--trc-ft-only` 或 run_ft.py 插零初始化 TRC（默认关逐位一致单测先行），冻结 encoder 只训 TRC+头；{c1,c2}×{ptbxl,cpsc}×FT10×seeds{0,2,4}，对照 LP 与 full-FT10（引用 W6 账本），门=vs LP 同向+≥+0.5pt，e2a_trcft.csv。
- [ ] Q6d E2c TRC×B0：`run_pt.py --data-dir data/pt_pretrain --epochs 200 --trc 1` seed0 单链（对齐 B0 更新数），PTB/CPSC LP；判读正负都入账。

## Q7 · 第三批（视门走，本轮只挂占位）
- [ ] E3 续训链 / E5 噪声注入 / E2b 变体：规格见任务书第三批；E1/E2a/E2c 门判定出来后由主会话或值班轮按决策树填具体步骤。

## 完成状态记录（执行轮次在此追加）
- 10-06 22:5x 主会话轮：W7 开工——任务书/队列/events 建立，E4d 实现中。
- 10-07 09:30 值班轮：E4b 实现+val 扫+判负关线(val 约束不满足, test 零消耗)。
- 10-07 08:23 值班轮：E4a 收线(6/6+终表, 反转消失判定落笔)。
- 10-07 05:23 值班轮：E1 全清(Q6b 温和正结论落笔)+E4a 启动。
- 10-07 02:24 值班轮：E1 接力完成(s2 收官+s4 启动); save-predictions 瑕疵修复。
- 10-07 00:23 值班轮：Q3 全清(E4c 9/9+summary: 排序稳健/重划下 c2>c1 一致/绝对差属测试池差异); E1 s2 PT 46/100 健康推进, seed4 接力与 E4a 启动留给后续轮。
- 10-06 23:5x 主会话轮：E4d 完成(63 目录对账全一致/反转节律依赖读数入 events)；E2d 完成(均值层否定+能量层缩偏支持)；E1 seed2 链 detached 运行中；值班 cron automation-499c4373 激活；E4c splits 建好+LP 车道运行中(e4c_lp_all2.log)；E4a 代码就绪待 GPU 空闲——**值班轮启动命令: nohup "C:/Users/admin/.conda/envs/DL/python.exe" -u runlog/W7/run_e4a_lpft.py > runlog/W7/logs/e4a_all.log 2>&1 &**(须 E1 链全部结束且 E4c 9 格齐后, 避免三重 GPU 争用; 跑完脚本自动出 e4a_summary.md)。
