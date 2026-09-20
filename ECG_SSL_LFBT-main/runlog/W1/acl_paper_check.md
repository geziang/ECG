# ACL-ECG 论文核对记录(2026-09-20, 用户提供的 PDF: paper/4cbe1dc827181be0f7fc428470f88cde7916.pdf)

依据论文全文(pypdf 提取, 26 页)逐项核对 models/acl_region.py + run_pt.py 实现:

| 项 | 论文原文 | 实现 | 结论 |
|---|---|---|---|
| 四区分组 | Eq.(7): h=(II,III)/(V1,V2)/(V3,V4)/(V5,V6) concat, D=4 | REGIONS_ANATOMY 相同 | 一致 |
| 区域 projector | Eq.(8): 每区独立 g_wd | RegionProjectors(128-2048-2048-2048) | 结构一致, 维度沿 LFBT 2048 口径 |
| Eq.(10) intra | I[d=d'], 正样本=同区跨视图, 负样本=批内其余(2N 池, I[k!=j]) | nt_xent_pair 同构 | 一致 |
| Eq.(11) inter | I[d!=d'], 同记录异区域跨视图为正, 负样本=批内其余记录 | nt_xent_pair(z1_d, z2_d'), 池=(z1_d U z2_d'), 同记录其它区域不入池 | 一致(不推远区域) |
| 温度 τ | §4.2: "The temperature parameter is set to tau = 0.1" | 原默认 0.5 | **已修正默认 0.1** |
| γ | Eq.(12) + Fig.6 sweep: γ=0.5 最优 | eta1=eta2=0.5 | 一致 |
| batch | 64 | 128 | 保留本机冻结协议(红线 §五-2), 差异记录在案 |
| 归一 | L_inter 带 1/(D-1) 因子求和; L_intra D 项求和 | 两者均取平均 | 常数因子差异, 由 eta1/eta2 吸收, 记录在案 |
| 消融佐证 | Table 11: 仅 intra 0.7313 / 仅 inter 0.6509 / 双路 0.7421 (CPSC LP AUPRC) | A2/A3 消融设计同构 | 支持任务书 A2 vs A3 对照 |
| 预训练协议 | Adam 1e-3, 200ep, NFH 预训练->CPSC/PTB-XL/Chapman 下游 | Adam 1e-3, 200ep 一致; 语料按 C1 外部锚点推进 | 对齐中 |

修正动作: run_pt.py --acl-tau 默认 0.5 -> 0.1(09-20 晚, a4 任务启动前生效; 队列中 a1/b3 等不含 ACL 不受影响)。
