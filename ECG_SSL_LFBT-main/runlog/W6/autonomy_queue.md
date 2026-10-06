# W6 自主工作队列（用户全权委托 2026-10-06 ~ 10-08）

> **委托依据**：用户 10-06 11:5x 指令——"全交给你了，完成了自己分析结果写代码，放大到巡检里/记忆里，后天来看最终结果"。
> **执行者**：每小时值班轮次（原巡检）。**每轮只执行一个编号步骤**（可验证粒度），完成后在本文件勾选并 commit+push。
> **总纪律不变**：协议超参不动；W2-W5 冻结只读；test 每格只评一次；新代码默认关==C1 逐位一致单测；分叉按任务书预授权自动走不等人；任何步骤卡壳→落 events.md 跳过不停车。判负也是完整交付物（负结果一等公民）。
> 最终交付：全部完成后写 `runlog/W6/W6_最终报告.md`（10-08 用户回看入口）。

## Q1 · LCM seed0 门判定 + 可能的 seeds{2,4}【当 lane_L2_lcm_runner.log 出现"LCM seed0 链结束"且 lcm_results.csv 有 seed=0 的 4 行后可执行】
- [x] Q1a 读 lcm_results.csv 的 (lcm, 0, lp, cpsc) auroc，算 Δ = auroc − 0.9455（C1 seed0 cpsc LP 参照，源 runlog/W5/lp_results.csv）。
- [x] Q1b 判定与动作（任务书 Stage2 预授权）：
  - Δ ≥ +0.0030 → 过门：detached 启动 `runlog\W6\run_lcm_chain.py 0.05 2`（日志 lane_L2_lcm_s2_runner.log），其"链结束"后再启动 seed 4（lane_L2_lcm_s4_runner.log）；events 记录门判定与 Δ。
  - Δ < +0.0010 → **判负关线**：events + HOSTS Stage2 行改 ✅判负(含 Δ 与四行读数)，入失败方向表叙事（与 D1L 的机制区别一并写：保持实测结构亦无增益 → 跨导结构先验线彻底关闭）。
  - 灰区（+0.0010~+0.0030）→ 不加跑，events 记"灰区裁决归办公机"，HOSTS 行改 ✅灰区。
- [ ] Q1c 若加了 seeds：三种子齐后算 cpsc LP 三种子 mean 与逐种子 Δ（对 C1 同种子 0.9455/0.9453/0.9481），按 3-seed 门（mean ≥+0.5pt 且 3/3 同向）判终，events+HOSTS 回填，**判负即关线不翻案**。
- [x] Q1d 无论正负，LCM 收口后 HOSTS"待开发"行去掉 Stage 2。

## Q2 · Stage 4 双视角一致性 + RR 头（开题关键问题(1)本体，LCM 收口后开工）【需要多轮】
设计规格（照此实现，机制上与判负 CCM 的区别=双视角表征一致性、不做重建）：
- [x] Q2a 侦察：R 峰缓存位置与格式（W1 gqrs 离线缓存，见 runlog/W1/ 与 prep_rpeaks.py；核对 17,418 条覆盖 data/pt_pretrain_nfh 对应语料——注意 C1 语料是 NFH 34,905/34,808，若缓存只覆盖 ptbxl 17,418 则双视图用 ptbxl 语料预训练并在 events 记录该差异与理由）。
- [x] Q2b 实现 `--dualview-weight`+`--rr-weight`（默认 0=逐位等于 C1，单测镜像 test_w6_lcm 模式）：心拍视图=每样本逐导联取 R 峰中心窗（窗长=中位 RR 的 1.2 倍，无 R 峰样本剔除该 batch 的该项），整段与心拍视图过**共享** per-lead backbone，BT 式一致性（同导联两视图 on-diag 相关逼近 1）；RR 头=lead-II 分支特征回归全库归一化 {mean RR, SDNN, RMSSD}（低权重 MSE，缺统计样本跳过）。单测：关态逐位一致 / 心拍窗提取对齐 / RR 目标归一。
- [x] Q2c smoke 5ep（权重取任务书"低权重"量级：dualview 0.05 / rr 0.1，若 loss 量级失衡可各降一档并记录，**不做网格**）→ 诊断 NaN/量级。
- [x] Q2d seed0 全链（仿 run_lcm_chain.py 写 run_dv_chain.py，protocol w6-dv，PT→三域LP→FT10），结束后按 Q1 同款门（C1 参照同上）自动判定与加种。
- [x] Q2e 收口：**无论正负心拍线就此关闭**（任务书 Stage4 第3条），events 写"研究内容(2) 证据链完整"结论行。

## Q3 · Stage 6 NSTDB 真实噪声鲁棒性（可与 Q2 并行插 L1 空档）【多轮】
- [x] Q3a 侦察 W3 的 run_robustness.py 口径（预注册：冻结 encoder+已存 LP 头、逐位复现 clean 门、SNR 定义、加噪方式），写 runlog/W6/run_robustness_nstdb.py：噪声源=data/nstdb/{bw,ma,em}.dat（wfdb 读取，360Hz 重采样到目标 fs，随机段循环），逐导联加性、per-lead SNR {0,5,10,20}dB。
- [x] Q3b clean 门：三域 LP clean 读数与 W3/W5 账本逐位一致后才开扰动（不一致→停，落事件）。
- [x] Q3c 跑全网格：{b0,c1,c2}×seeds{0,2,4}×三域×3噪声×4SNR（=324 评估，GPU ~8h，L1 夜跑），产物 runlog/W6/robustness_real_{b0,c1,c2}.csv（schema 同 W3）。
- [x] Q3d 汇总：与 W3 合成噪声结论对照（NFH 抗退化/TRC 中性是否保持），写 runlog/W6/stage6_summary.md（禁语合规）。

## Q4 · Stage 7 CPSC 多标签（轻，Q3 后任意空档）
- [ ] Q4a 侦察 utils/multilabel.py；写 run_multilabel_lane.py：冻结 encoder × {b0,c1,c2} × seeds{0,2,4} BCE LP 头（协议 w6-multilabel），指标=连续概率 macro-AUROC/AUPRC + val 阈值 per-class F1/Sens/Spec。
- [ ] Q4b 跑 9 训练+评估（GPU ~2h），产物 runlog/W6/multilabel_results.csv；与单标签主表**分列**呈现。
- [ ] Q4c HOSTS 回填 ✅。

## Q5 · 收口算术与总报告
- [ ] Q5a Stage1 配对差复算：ft2040_results.csv 里 C2−C1 与 C1−B0 每档（ft20/ft40）逐种子差+mean（AUROC 与 AUPRC），写 runlog/W6/stage1_paired_delta.md。
- [ ] Q5b Stage3 配对差复算：ft10grid 同款（cpsc/chapman × C2−C1/C2−B0/C1−B0），写 runlog/W6/stage3_paired_delta.md。
- [ ] Q5c 5-seed 汇总：三方法×5种子×{lp三域,ft10} mean±SD 与符号一致性 → runlog/W6/seed5_summary.csv（办公机换主表用）。
- [ ] Q5d **W6_最终报告.md**：各阶段结果/门判定走了哪条预授权分支/禁语合规自查/遗留事项（office 侧待办）。HOSTS §一-D 总回填。

## 完成状态记录（执行轮次在此追加）
- 10-06 12:xx 值班轮：队列建立。
- 10-06 12:26 值班轮：Q3a 完成——run_robustness_nstdb.py 编译+smoke(96样本)全通; 组合表 ptbxl8/cpsc9(W5A头)/chapman6; clean 参照分域(W2/W5)。
- 10-06 13:23 值班轮：Q1 完成——Δ=−0.12pt 判负关线(未触发加种子, Q1c 不适用), events+HOSTS 已回填。
- 10-06 14:28 值班轮：Q2a 完成——rpeaks 缓存重建 17,418/17,418(240s), run_pt 管线复用确认, 语料决策输入已入 events(倾向 NFH 补检测)。
- 10-06 15:28 值班轮：Q2b(双视角实现+单测13/13+回归全绿)/Q3b(clean门23/23)/Q3c(299评估全跑完) 完成; NFH rpeaks缓存就绪99.98%。
- 10-06 16:33 值班轮：Q2c 完成——smoke 零NaN, 权重定档 dualview=0.01(0.05档占1.9%失衡降一档)/rr=0.1。
- 10-06 18:25 值班轮：Q3d 完成——stage6_summary.md 入库(NFH抗退化/TRC中性两条W3结论在真实噪声下保持; 噪声类型敏感性域间差异如实记录)。
- 10-06 19:35 值班轮：Q2d/Q2e 完成——DV seed0 cpsc LP 0.9482(Δ=+0.27pt)落灰区不加种子, 心拍线关闭, 研究内容(2)证据链完整(负+灰区)。Q2 全清。
