# W7 自主工作队列（用户 24 小时委托 2026-10-06 起）

> **委托依据**：用户 10-06 指令（见 W7_任务书.md 头注）。**执行者**：主会话 + 每小时值班轮。
> **每轮只执行一个编号步骤**（可验证粒度），完成后勾选并 commit+push。卡壳→落 events.md 跳过不停车。
> 纪律同 W6：协议超参不动；W2–W6 冻结账本只读；test 单发；新代码默认关==基线逐位一致；判负也是交付物。
> 环境提醒：python 必须 `C:/Users/admin/.conda/envs/DL/python.exe`（repo venv 死链）；日志判活看字节增长/SM%。

## Q1 · E4d 心拍分层误差分析【主会话已实现 run_e4d_beat_stratified.py】
- [ ] Q1a 值班轮：确认 R 峰三域 test 缓存生成完毕（data/pt_rpeaks_test_{ptbxl,cpsc,chapman}.npz，脚本幂等可续）；慢则等待，快则进入 Q1b。
- [ ] Q1b 分析跑完后核验产物三件（e4d_rr_features.csv / e4d_bucket_summary.csv / e4d_summary.md），抽查 3 行与 W6 metrics_ext 对账（accuracy 全体值应一致），events 记录关键读数（HR/SDNN 分桶的 b0−c2 方向）。

## Q2 · E2d TRC 机理分析【主会话实现中】
- [ ] Q2a 产物核验（e2d_channel_stats.csv / e2d_summary.md），重点：通道统计跨域距离与 CPSC 增益的相关系数方向；events 记录。

## Q3 · E4c 患者级敏感性附录（PTB-XL）【需实现，规格如下】
- [ ] Q3a 实现 `runlog/W7/run_e4c_patient_lp.py`：读 data/ptbxl 元数据 patient_id（侦察：ptbxl 原始 csv 在 data/ptbxl_* 或仓库何处，sample_XXXXX ↔ ecg_id 映射来自制备脚本——先 grep prepare 脚本确认）；按患者级分层重划 K=3 次（seed 101/102/103，train/val/test 比例对齐 70/10/20 患者比例），新 split 写 data/ptbxl_patient_k{i}/（软链或复制 .npy，磁盘不够则复制清单+DatasetFolder 子类按清单过滤）。
- [ ] Q3b 跑 LP：{b0,c1,c2}×seed0×K=3 新 split LP（run_lp.py --data-dir 指向新 split 根；checkpoint 用 W2 冻结 SHA 校验版；protocol_id=w7-e4c-k{i}），9 跑 GPU 各~2min；账本 e4c_patient_lp.csv（含原 split AUROC 对照行）。
- [ ] Q3c 汇总 e4c_summary.md：患者级 vs record 级 AUROC 差（预期小幅波动），措辞"敏感性分析"不写成主表。

## Q4 · E4b 质量门控推理【需实现，规格如下】
- [ ] Q4a 实现 `runlog/W7/run_e4b_quality_gate.py`：质量分=每导联 5–40Hz 带内功率/(总功率+eps)（scipy butter 滤波实现，无新依赖）；阈值 τ 在各域 val 上扫 {0.3,0.4,0.5,0.6}（val 定档后 test 单发）；门控=低于 τ 的导联置零+valid_mask 摘除。
- [ ] Q4b clean 门：三域 clean 逐位与 W6/W5 账本一致（τ 取最保守 0.3 也必须不伤 clean——若伤，τ 上调至 val 无伤档，仍伤则判"质量门控在 clean 上有损"如实关线）。
- [ ] Q4c NSTDB 扰动评估：{b0,c1,c2}×seed0×三域×3噪声×SNR{0,5,10} 子集（复用 run_robustness_nstdb.py 机制改推理侧），对照"固定 mask"同格读数；噪声条件 3-domain 平均 ≥+0.5pt 判正，e4b_quality_gate.csv + e4b_summary.md。

## Q5 · E4a LP-FT 机制实验（CPSC）【需实现，规格如下】
- [ ] Q5a 实现 `runlog/W7/run_e4a_lpft.py`：LP-FT=先纯线性头收敛（run_lp 同款至收敛）→ 以该头初始化+encoder 全解冻跑 run_ft 同款 FT10；对象 {b0,c2}×seeds{0,2,4}（c1 作参考可选）；protocol_id=w7-e4a；full-FT/LP 读数直接引用 W6 Stage3/W5 账本不重跑。
- [ ] Q5b 6 跑完出 e4a_lpft.csv + e4a_summary.md：三适配方式排序表（LP / full-FT / LP-FT），b0 血缘优势与 c2 反转在 LP-FT 下是否保持/消失；机制讨论对接 LP-FT 文献（Kumar 2022）。

## Q6 · 第二批预训练链（第一批 GPU 空档即排，Q1/Q2 不占 GPU 时可先发 E1）
- [ ] Q6a E1 灰区加种子：detached 跑 `runlog/W6/run_dv_chain.py "" 2`（日志 lane_E1_dv_s2.log），链结束且 dv_results.csv 出现 seed=2 四行后跑 seed 4；**注意该脚本位置在 W6 目录但账本追加 dv_results.csv 属 W6 文件——先把脚本复制为 runlog/W7/run_e1_dv_chain.py 并把 OUT 改为 runlog/W7/e1_dv_results.csv 后再跑**（W6 冻结账本只读红线）。
- [ ] Q6b E1 门判定（任务书预注册）：3 seeds(0,2,4) cpsc LP 对 C1 同种子配对 Δ；≥+0.30pt 且 3/3 同向=过确认线；∈[+0.10,+0.30) 且 3/3 同向=温和正；任一反向=关线复核入账。events+HOSTS 回填，seed0 参照=W6 dv 0.9482 与 C1 seed0 0.9455。
- [ ] Q6c E2a FT期TRC：实现 run_pt.py 侧 `--trc-ft-only` 或 run_ft.py 插零初始化 TRC（默认关逐位一致单测先行），冻结 encoder 只训 TRC+头；{c1,c2}×{ptbxl,cpsc}×FT10×seeds{0,2,4}，对照 LP 与 full-FT10（引用 W6 账本），门=vs LP 同向+≥+0.5pt，e2a_trcft.csv。
- [ ] Q6d E2c TRC×B0：`run_pt.py --data-dir data/pt_pretrain --epochs 200 --trc 1` seed0 单链（对齐 B0 更新数），PTB/CPSC LP；判读正负都入账。

## Q7 · 第三批（视门走，本轮只挂占位）
- [ ] E3 续训链 / E5 噪声注入 / E2b 变体：规格见任务书第三批；E1/E2a/E2c 门判定出来后由主会话或值班轮按决策树填具体步骤。

## 完成状态记录（执行轮次在此追加）
- 10-06 22:5x 主会话轮：W7 开工——任务书/队列/events 建立，E4d 实现中。
