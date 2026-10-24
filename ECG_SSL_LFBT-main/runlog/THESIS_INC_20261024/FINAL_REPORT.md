# thesis-inc-20261024 最终报告

任务 `thesis-inc-20261024`（协议 `thesis-inc-20261024-v1`，用户 2026-10-24 授权，截止 2026-10-26 12:00 北京时间）。主机 A = Win4090 于 2026-10-24 14:58 认领（42fe1c1），主线与旁线均提前收束，本报告为截止前最终版。

## 一、结论速览

1. **主线（有效阴性）**：NFH 无标签 SSL 六配方（I0–I5，20ep/seed0）在统一 CPSC train/val LP 下**全部非正**；规则允许的唯一单次组合 I2+RR 亦非正（vs I0 −0.0007，vs 母配方 −0.0003，无协同）。按冻结门 R3（权重档）/R4（种子确认）/R5（100ep 完整配对）**均不进入**。本批增量辅助目标（均值/中位模板一致性、逐拍局部 BT、时间均匀固定窗对照、RR 三统计及其单次组合）在 20ep/seed0 口径下为**有效阴性**。
2. **参考旁线（REF）**：R-HL（HeartLang 作者协议 LP）完成；R-TK 三行中 median_beat 与 heartlang 完成，patching 经两档实测（OOM/单epoch>120min）判定本卡不可行，按协议记 partial_infeasible_on_gpu（详见 §四）。
3. **合规**：test 信号/标签零访问（全批审计）；50 作业槽实际启动 18 个（14 主线 + 4 REF 中 3 complete + 1 partial）；预算远未触顶。

## 二、执行与工程（A0）

- **预检全过**（`preflight.json`）：源锁 11 git blob 哈希一致（base 9b135ee）；NFH 34905 条逐记录内容 SHA 清单（`nfh_record_manifest.csv`）；峰缓存键与样本集合完全匹配、34898 条峰数≥3；P0 隔离 LP 数据 train/val SHA 一致（4809/683）；只读 M4/M6 checkpoint 本体+D 盘备份双核验；环境 Python3.11.9/torch2.5.1+cu121/RTX4090。
- **新实现** `tools/thesis_inc/`（10 个模块，冻结于 protocol_resolved.json）：
  - 逐记录确定性增强流 det-aug-v1：整段两视图 NSTDB(0.5, first_half, 5–20dB)→RRC-TO(0.5,1.0,0.0,0.5) 与源码同序同分布；跨配方同 seed 整段视图逐位一致；局部视图 RNG=(seed,epoch,record,loc j) 独立。
  - 模型复刻 run_pt 慢路径（VGG16 α0.125+TRC1 逐导联 + 2048³ projector + 输出 BN），I1/I2 复刻 `_dv_consistency`，I3/I5 新局部加权 BT（w=1/(N_valid·K_b)，加权标准化 sqrt(var+1e-4)，C=Z1ᵀdiag(w)Z2，逐导联 on/off-diag 0.0051，八导联平均），I4 复刻 RR 头。
  - 断点：每 epoch 完整状态（模型/优化器/四 RNG/epoch/step/资源/配置 SHA）；epoch 边界与 epoch 内(step)双路恢复 **smoke 逐位一致验证（16/16 PASS）**。
  - 统一 LP：P0 协议逐条复刻（冻结 512 维、Linear(512,9)、BCE、min 元素加权 val BCE 选模、独立复算≤1e-8、逐记录预测落盘）。
  - runner：门调度/预算（36 GPU h、CPU 12h、工程 12h、REF 4h、50 作业、单 GPU 进程）/截止（10-26 10:00/11:00/11:30/12:00）/决策留痕/备份回写。
- **工程事件（如实）**：①首次启动误用 conda base python（无 CUDA）训练实际在 CPU 跑约 11 分钟，三证据判定后终止，误记 gpu 时间整体重归类 cpu；②梯度范数探针初版在 backward 后执行恒 null，移至 backward 前修复；③runner manifest 键/`_pos_rank` 签名两处小缺陷修复；④R-TK 三处启动缺陷（num_class 缺失/ids object-dtype/collate 未挂载）修复；⑤R-TK patching 32×8 OOM 后同进程降档重试因 CUDA 堆碎片 27 秒再 OOM（工程伪影），按每作业一次工程修复改新进程 16×16 限时跑。

## 三、主线结果（R1/R2，主指标=val macro-AUPRC）

| 配方 | macro-AUPRC | Δ vs I0 | macro-AUROC | best_ep |
|---|---|---|---|---|
| I0 主干（M6 设计背景=TRC+噪声） | **0.7367** | — | 0.9320 | 100 |
| I2 中位模板 0.01 | 0.7363 | −0.0003 | 0.9305 | 100 |
| I1 均值模板 0.01 | 0.7314 | −0.0052 | 0.9300 | 100 |
| I5 固定窗局部 BT 0.01 | 0.7293 | −0.0074 | 0.9289 | 100 |
| I4 RR 回归 0.1 | 0.7268 | −0.0098 | 0.9288 | 100 |
| I3 逐拍局部 BT 0.01 | 0.7235 | −0.0131 | 0.9226 | 100 |
| **R2: I2+RR（0.01+0.1）** | 0.7360 | **−0.0007** | 0.9281 | 100 |

- 门与处置：I1/I2/I3 全非正 → R2 规则给最佳形态项 I2+RR 一次组合机会（无第三槽）；结果非正且低于母配方 → 不宣称协同；R3 无正向不扫权重；R4 无正向种子确认按规则结束；R5 无三种子正向不启动。33 个矩阵行标 `not_entered_by_gate`（job_manifest.json），决策全留痕 decisions.jsonl。
- 解读边界：20ep/seed0 口径的阴性；I2 与 I0 差距 −0.0003 处于种子噪声量级（历史 ~1.3pt 以下），但按冻结门（>0 为正）判负，不做事后翻案。I0 参考（非跨批结论）：P0 的 100ep M6 冻结权重 LP 0.7318，本批 20ep I0 0.7367。
- 训练动力学证据：辅助损失项全部有限且下降（如 I1 aux 239→184），局部行 local_diag_mean 与梯度范数比在日志中留痕（探针修复后），smoke 证明局部目标梯度非零接入——阴性不是"未接入"伪影。

## 四、参考旁线（REF，子预算 4h 内）

| 行 | 结果 | 说明 |
|---|---|---|
| R-HL HeartLang 作者协议 | val macro-AUPRC 0.4218 / **AUROC 0.7947**（best_ep 81/100ep 固定） | 作者 4950/551 split 与列序；backbone train 模式 dropout+参数冻结（哈希核验）；max val macro-AUROC 选模；与 P0 A4（0.4464/0.8121，我们 split/eval 模式/BCE 选模）分表不并口径；不声称复现论文 test 数值 |
| R-TK median_beat | val macro-AUROC **0.8977**（best_ep 69，早停） | 作者 fold≤8/9 复现；16×16 统一档；300ep/patience15/seed2025/RAdam1e-3 |
| R-TK heartlang | val macro-AUROC **0.8971**（best_ep 84，早停） | 同上 |
| R-TK patching | **partial_infeasible_on_gpu** | 1500-token 注意力（12×125 token，显存 24G 饱和）：32×8 于 epoch1 中途 OOM（19 分钟未完成）；16×16 新进程工程修复尝试 >120 分钟仍未完成 epoch1（deadline 检查为 epoch 粒度，由人工按证据终止并补记 7260s REF 资源）；patience15 需 ≥16 epoch ≈ 32h+，远超 REF 4h 子预算。按协议不改输入/模型，如实记不可行 |

R-TK 数据身份（记录性偏差，均留痕 decisions.jsonl）：①ecg_id/age/sex/标签元数据取自 HeartLang 仓库公开 CSV（与本地 2020 镜像 .hea 交叉验证 294/300 全一致，6 处差异为 >89 岁审查伪影且同分层桶）；②作者 stratisfy_df（seed 0）原样运行复现 fold（10 折 687–689）；③信号取自本地 2020 WFDB 镜像（同一 CPSC 源重编码，未走作者 16bit wfdb 往返）；④**fold10 信号零读取**；fold10 标签/年龄/性别元数据仅作为作者 fold 复现的分层输入，未进入任何训练/选模/评价路径；⑤QRS/峰检测按信号内容缓存（输出与逐次检测逐位一致）。R-TK 解锁缘起：用户 2026-10-24 指示公开代码/数据集不构成限制；实测无需任何下载。

## 五、合规与资源

- **test 政策**：全程零 test 信号/标签访问；LP 入口 split 白名单守卫；R-HL 数据目录断言无 test_* 文件（test_loader_called=0）；R-TK fold10 信号零读取（caches_manifest 留痕）；results.csv 全文 0 次 "test" 引用；作者论文 test 数值复现未声明。
- **冻结材料**：P0/W0–W9/JOINT/ABC 全部只读（checkpoint SHA 复核一致）；无旧队列恢复；`protocol_resolved.json` 冻结后未改科学内容。
- **预算（终值）**：GPU ≈3.1h/36h；REF ≈2.9h/4h；CPU ≈0.8h/12h；工程 ≈1.9h/12h；作业 18/50 启动（14 主线 + 4 REF）；单 GPU 进程全程成立（REF 与主线串行、修复尝试均独立进程）。
- 截止：主线 10-24 19:31 收束、REF 10-24 深夜收束，均远早于 10-26 10:00/11:00/11:30/12:00 各闸口。

## 六、交付物索引

preflight.json / protocol_resolved.json / source_manifest.json / data_manifest.json（+nfh_record_manifest.csv 34905 行）/ job_manifest.json / results.csv / resource_usage.csv / events.md / decisions.jsonl / resume_manifest.json / summary_main_line.json / smoke_report.json / DAILY_REPORT_2026-10-24.md / DAILY_REPORT_2026-10-25.md / jobs/<id>/（config/train_log/val_predictions/result/DONE）；权重与恢复状态在 `checkpoint/thesis_inc/<id>/`（.gitignore 排除）并备份 `D:\thesis_inc_20261024_backup\`；REF 数据清单 ref_hl_data_manifest.json 与 ref_tokenizer_data/{folds,caches}_manifest.json。

## 七、给后续轮次的边界

- 本批阴性不否定"更强心拍/节律先验"方向本身：20ep 短训、单种子、0.01/0.1 旧权重量级是本批口径；若重启应先声明与 I2(−0.0003) 噪声量级的新鉴别设计。
- R-TK patching 行在本卡不可行的实测（32×8 OOM、epoch>20min）可作为后续算力规划的依据。
- HeartLang 两种协议口径（P0 A4 vs 本批 R-HL）已分表，后续引用须注明口径。
