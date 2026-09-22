# W2 冻结材料协议审计（B-0 交付物，主机B）

> 生成：2026-09-22，主机B（DESKTOP-0PBLCND），CPU-only，基线 git `f10e6fa`+B 认领提交。
> 数据核对脚本：`runlog/W3/freeze_audit_hostB.py`（ALL PASS，结果见 `freeze_audit_hostB.json`）。
> 性质：论文写作时可引用的协议事实与限制清单；每条附来源。W2 文件全程只读。

## 1. 协议事实（复算核对通过）

| # | 协议项 | 冻结口径 | 来源 |
|---|---|---|---|
| P1 | 导联配置 | 8 导联统一输入（`lead_mapping` 在册） | `runlog/W1/nfh_manifest.json`；A-0 freeze_audit manifests |
| P2 | 本域预训练 | PTB-XL folds 1–8（17,418 条），200ep / batch 128 / Adam 1e-3 / RRC-TO(0.5,1.0,0.0,0.5) | `runlog/HOSTS.md` §五-2；`runlog/W1/freeze_manifest.json` protocol |
| P3 | 外部预训练 | NFH 34,905 条，100ep **matched-updates**（≈27.3k 步，与本域 200ep 更新数对齐，非同 epoch 数） | 同上；HOSTS §三 W1 状态 |
| P4 | 下游评估 | LP=冻结+线性(1e-3)；FT10=10% 标签全量微调(1e-4)；各 100ep，val 选 checkpoint，**test 仅评一次** | HOSTS §五-2；任务书 §1-5 |
| P5 | 数据划分 | CPSC / Chapman 为 **record-level split**（.hea 无患者 ID 可用）；PTB-XL 沿用 folds 1–8 标准 split | `runlog/W1/cpsc_manifest.json`、`chapman_manifest.json`（csc-v1）；A-0 limitations |
| P6 | NFH 患者去重 | NFH 无患者 ID，以**记录内容 SHA256 查重**替代（3 对重复仅标记不剔除，6/34,905≈0.017%，保锚点可比性） | `runlog/W2/nfh_manifest_reconciliation.md` |
| P7 | 缺导评估语义 | 导联缺失 = **置零**（zeroing），不是导联错位/移位；adjacent lead-swap 属 W3 A-2 新增 test-time 扰动，两者不得混写 | 任务书 A-2（“明确区别于已有导联置零”）；`missing_lead_c2.csv` |
| P8 | TRC 参数占比 | TRC=GRN1D 零初始化旁路，128 参数/导联 × 8 = 1,024，占编码器 635,304 的 **0.161%**（复算 0.1612% ✓） | `freeze_manifest_v2.json` params；B 复算 |
| P9 | 主结果数字 | CPSC LP C2−C1 = **+2.30pt，3/3 同向（+++)，精确置换单侧 p=0.125，seed-level t-CI [+0.26,+4.34]**；PTB-XL FT10 = **+0.65pt，3/3 同向，CI [−0.93,+2.24]**；本域 PTB-XL LP +0.21（−+- 噪声）；血缘域 Chapman −0.03（≈0） | `stats/paired_delta.csv`；B 独立复算 6/6 行逐项一致 |
| P10 | 账本完整性 | 28 行、0 重复键、git_sha 统一 `fb08bc5`、ts 单调；checkpoint 12 位前缀与冻结清单全长 SHA 全匹配 | B 复算（`freeze_audit_hostB.json` ledger） |
| P11 | W1→W2 端到端确定性 | c2_seed0 与 W1 c2_nfh_trc_seed0 **字节哈希一致**；c1/c2 LP 读数 7/7 与 W1 c3_evals 逐位一致 | `freeze_manifest_v2.json` audit_notes；B 复算 w1_w2_bit_identical_reads |

## 2. 限制（论文必须携带的边界条件）

1. **无逐记录预测概率**：W2 仅保存聚合 metrics（metrics.json/CSV），**不得声称 patient-level bootstrap**；统计只能用 seed-level 描述统计 + 3 seed 精确符号翻转置换（p 下限 1/8=0.125）。CI 为 seed-level t(2) 区间，不是记录级区间。（B-2 已为此补逐记录保存接口，仅面向 W3 新结果。）
2. **n=3 seeds**：置换检验与 CI 均受 3 seed 自由度限制；`+2.30pt` 的 CI 含 0 距离很近的结论只能写“3/3 同向、均值 +2.30pt”，不能写“统计显著”。
3. **b0 外部域单种子**：B0 的 CPSC/Chapman 列为 W1 单种子参考（n_seeds=1，无 SD），主表已如实标注；CPSC/Chapman 的 3-seed 配对只在 C2−C1 之间。
4. **跨机绝对值不可比**：同配置绝对 AUPRC 跨 GPU 历史差 0.8~1.0pt（A 锚点 0.7177 vs B 锚点 0.7158）；双机各自对本机锚点算 Δ，跨机只比方向/符号。失败索引中 A/B 两列 Δ 不可直接拼表。
5. **NFH 清洗口径**：主结果采用 A 口径 `nfh-v1`（34,905 全保留，制备时+独立复扫双验证零非有限值）；B 机 97 条剔除仅作敏感性口径，**明细待 B-3 交付后方可解释差异来源**；两口径制品不得混用。
6. **域语义**：NFH(宁波)→CPSC(中国)→PTB-XL(德国) 的“第三域/血缘域”表述保持克制；不得写“跨医院泛化”或“缺导绝对性能超过 B0”（缺导结论仅“相对鲁棒性保留”：单导 −0.59 / 双导 −1.41pt，20/36 条件非系统性）。
7. **旧 `data/downstream` 体系作废**：存在患者跨 split 与 train/test 污染的历史结果一律不再引用。
8. **W2 材料只读**：`confirm_results.csv`、`stats/*.csv`、`freeze_manifest_v2.json`、`paper/*.png` 不得覆盖；新结果只写 `runlog/W3/`。

## 3. 审计覆盖与方法

- **对象**：W2 冻结集 10 文件 + W3/A-0 审计 + W1 佐证 8 文件 + 双机账本 2 文件，共 21 个；全部 SHA256 在册，且 `git hash-object` 与 HEAD blob 一致（工作区无未提交改动）。
- **方法**：纯 stdlib Python（csv/json/hashlib/statistics），无 GPU、无第三方依赖；脚本与结果同目录入库，可复跑。
- **checkpoint 本体**：11 个 checkpoint 在主机A 本地，B 侧无法字节级重算；B 的核对方式为 git 材料内一致性（前缀匹配 + W1 清单交叉 + A-0 审计 16 位前缀与 v2 全长 SHA 前缀一致 + A 机 11/11 重算结论引用）。此边界已写入 `freeze_audit_hostB.json` limitations。

## 4. B-0 验收对照（任务书 §3 B-0）

| 验收项 | 状态 |
|---|---|
| `runlog/W3/freeze_audit_hostB.json`：存在性/SHA256/CSV 行数与重复键/seed 符号 | ✅ ALL PASS（21 文件、28 行 0 重复、+++ 复算一致） |
| `runlog/W3/protocol_audit.md`：本文件 | ✅ |
| `runlog/W3/failed_directions_index.csv`：方向—代表配置—结果—关闭理由 | ✅ 54 行（含 host 归属与来源账本列） |
| 不修改 W2 文件 | ✅ 全程只读（git 工作区与 HEAD 一致性检查通过） |
| 无 GPU 环境可运行 | ✅ 纯 CPU stdlib |
