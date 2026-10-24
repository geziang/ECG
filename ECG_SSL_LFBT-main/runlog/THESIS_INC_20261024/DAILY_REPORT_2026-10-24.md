# thesis-inc-20261024 日报（2026-10-24）

任务：`thesis-inc-20261024`（协议 `thesis-inc-20261024-v1`）。Win4090 于 14:58 认领（42fe1c1），本日完成 A0 预检、工程实现、smoke、协议冻结并启动 R1。

## 今日完成

1. **A0 预检全过**（`preflight.json`，CPU 335s）：
   - 源锁 11 个 git blob 哈希逐一核对一致（base 9b135ee）；
   - NFH 34905 条记录全量逐记录内容 SHA（`nfh_record_manifest.csv`，manifest sha `nfh_content_manifest_sha256` 见 `data_manifest.json`），峰缓存键与样本完全匹配，34905 条中 34898 条峰数≥3 有效；
   - P0 隔离 LP 数据 train/val SHA 与来源锁一致（4809/683）；
   - 只读 M4/M6 六个 checkpoint 本体+D 盘备份双核验一致；
   - 环境：Python 3.11.9 / torch 2.5.1+cu121 / RTX 4090(610.74) / wfdb 在位。
2. **工程实现**（`tools/thesis_inc/`，共 9 个模块）：
   - `det_aug.py`：逐记录确定性增强流（det-aug-v1），整段两视图 NSTDB(0.5,first_half,5–20dB)→RRC-TO(0.5,1.0,0.0,0.5) 与源码同序同分布；局部视图仅噪声；RNG=(seed,epoch,record,view) 独立流。
   - `pt_dataset.py`/`pt_model.py`：I0–I5 六配方数据与模型（模板 mean/median 一致性复刻 `_dv_consistency`；I3/I5 局部加权 BT：共享 backbone+projector、w=1/(N_valid·K_b)、加权标准化 sqrt(var+1e-4)、C=Z1ᵀdiag(w)Z2、八导联平均；I4 RR 头复刻）。
   - `pretrain.py`：单作业 PT，per-epoch 确定性采样序（drop_last=True，272 批），完整断点状态（模型/优化器/四种 RNG/epoch/step/已耗资源/配置 SHA），epoch 边界与 epoch 内(step)双路恢复，每 batch 截止/预算检查。
   - `lp.py`：统一 LP（P0 协议逐条复刻：冻结 512 维特征、Linear(512,9)、BCE、min 元素加权 val BCE 选模、复算≤1e-8）。
   - `runner.py`：R1→R5 门调度（R2 组合/R3 权重档/R4 种子/R5 三种子平均门）、预算/截止/50 作业/单 GPU 进程守卫、manifest/decisions/备份回写。
3. **smoke 16/16 PASS**（`smoke_report.json`）：跨配置整段视图逐位一致、共享初始指纹一致、局部损失梯度非零接入、边界窗 edge-padding、I5 中心不越界、**断点恢复逐位一致（epoch 边界+epoch 内两路）**；LP 端到端冒烟通过（smoke 编码器 auprc 0.2049）。
4. **protocol_resolved.json 冻结**（实现版本 det-aug-v1 等；RRC/噪声/W205/K 选择/RR mu,sd 实测全解析）。
5. **R1 启动**：I0 已完成——**LP macro-AUPRC 0.7367 / AUROC 0.9320**（best_ep=100）；实测 41s/epoch（20ep≈14min），预算前景大幅改善。
6. **R-HL 旁线就绪**：`ref_heartlang.py`（作者 4950/551 split、作者列序、backbone train 模式 dropout+参数冻结、max val macro-AUROC 选模、100ep 固定）数据构建已在 CPU 并行进行。

## 事件与偏差（如实）

- **环境事故（已修复）**：首次 R1 启动误用 conda base python（无 CUDA），训练实际在 CPU 跑约 11 分钟后判定终止（GPU 显存低于空闲基线+worker 空闲+主进程 CPU 饱和三证据）；误记 gpu_s 1777s（含 smoke 段，smoke 实亦 CPU）整体重归类 cpu_s，jobs_started 复位，未完成 epoch 的作业目录清理。smoke 结论不受影响（确定性/恢复检查与环境无关）。此后统一使用 `C:/Users/admin/.conda/envs/DL/python.exe`。
- 早期两处工程缺陷（manifest 键、日志句柄提前关闭）+ smoke 期望值笔误 + cfg_sha 含 epochs 阻断合法续训，均已修复并留痕（提交 ddba233）。
- I1–I5 结果、R2+ 门解析见后续日报/最终报告。

## 资源（截至 16:50 左右）

GPU ~0.5h/36h（I0 PT+LP+启动检查）；CPU ~0.75h/12h（预检 335s+环境事故 1777s+R-HL 构建进行中）；工程 ~2.6h/12h；作业 2/50（I0 PT+LP）。

## 下一步

R1 六组今晚完成 → 按门自动推进 R2/R3/R4；R2 后穿插 R-HL GPU 训练（单进程约束下停窗口执行）；R5 视三种子门与截止；10-26 11:30 停 GPU、12:00 前交最终报告。

## 追更（19:4x，主线收官）

**R1 六组（seed0, 20ep, 统一 CPSC LP）全部非正**：

| 配方 | macro-AUPRC | Δ vs I0 | macro-AUROC |
|---|---|---|---|
| I0 主干（M6设计背景） | **0.7367** | — | 0.9320 |
| I2 中位模板 0.01 | 0.7363 | −0.0003 | 0.9305 |
| I1 均值模板 0.01 | 0.7314 | −0.0052 | 0.9300 |
| I5 固定窗局部BT 0.01 | 0.7293 | −0.0074 | 0.9289 |
| I4 RR回归 0.1 | 0.7268 | −0.0098 | 0.9288 |
| I3 逐拍局部BT 0.01 | 0.7235 | −0.0131 | 0.9226 |

**R2 唯一允许的单次组合 I2+RR（0.01+0.1）：0.7360**（vs I0 −0.0007；vs 母配方 I2 −0.0003，无协同）。按冻结门：R3（无正向不扫权重）、R4（无正向种子确认结束）、R5（无三种子正向）均不进入，决策已记录 decisions.jsonl，33 个未进门矩阵行标注 not_entered_by_gate（job_manifest）。**主线结论=本批增量辅助目标（模板一致性/逐拍局部BT/固定窗对照/RR 及其单次组合）在 20ep/seed0 口径下为有效阴性。**

I0 参考：P0 的 100ep M6 冻结权重 LP 为 0.7318；本批 20ep I0 为 0.7367（不同批次口径，只作量级参考不作跨批结论）。

REF 旁线：R-HL（作者协议：4950/551 作者 split、train 模式 dropout+冻结参数、max val macro-AUROC 选模、100ep 固定）训练中；R-TK 三行数据已就绪（作者 fold 复现：10 折 687–689；隔离缓存 train5500/val688；fold10 信号零读取），待 R-HL 后串行。资源：GPU ~3.1h/36h、REF 0h/4h、CPU ~0.75h/12h、工程 ~0.7h/12h、作业 15/50。
