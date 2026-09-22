# 已完成报告

这里保存已经完成撰写、形成阶段性结论或完成归档的报告。完成报告不等于所有后续研究已经结束；若报告是阶段快照，文中会保留其对应日期和证据边界。

## 当前文件

- [08-论文收敛与下一阶段实验计划-2026-09-21.md](08-论文收敛与下一阶段实验计划-2026-09-21.md)：W2 收敛计划，排期 9/23–27 的实验已于 9/22 全部提前完成，文末附完成回填表；
- [下一阶段执行计划-2026-09-22.md](下一阶段执行计划-2026-09-22.md)：P0–P3 全部执行完毕（唯一未执行项=可选的 FT20/FT40），文末附执行回填；
- [下一阶段双机任务书-2026-09-22.md](下一阶段双机任务书-2026-09-22.md)：W3 双机任务全闭环（A-0/A-2、B-0~B-3 ✅，A-1 用户取消，A-3 可选后备），文末附收口记录；
- [09-论文材料冻结-2026-09-22.md](09-论文材料冻结-2026-09-22.md)：W2 主表、paired delta、TRC 参数量、缺导和失败方向材料已经冻结。
- [失败实验归档-2026-09-22.md](失败实验归档-2026-09-22.md)：已关闭/判负方向的统一台账。
- [W1执行报告-2026-09-21-主机A.md](W1执行报告-2026-09-21-主机A.md)：W1 主机 A 执行报告和证据边界。
- [项目总报告-2026-09-22.md](项目总报告-2026-09-22.md)：阶段性总览快照；它记录的是生成时点，不替代最新 W3 台账。
- [开题方向对照与未尝试清单-2026-09-22.md](开题方向对照与未尝试清单-2026-09-22.md)：开题方向→当前代码/结果/资源的映射快照；可验证项已全部闭合（W2 冻结、A-0/A-2、B-0~B-3），文末附 §六 完成回填表；"未尝试"行按 §四-P3 维持后续课题定位。

## 已完成的 W3 证据

以下机器可读结果保留在原运行目录，避免改动代码和审计引用：

- [A-0 冻结复核](../../ECG_SSL_LFBT-main/runlog/W3/freeze_audit.json)：11/11 checkpoint SHA 一致，账本无重复。
- [B-0 冻结材料审计](../../ECG_SSL_LFBT-main/runlog/W3/freeze_audit_hostB.json)：审计通过。
- [协议审计](../../ECG_SSL_LFBT-main/runlog/W3/protocol_audit.md)。
- [失败方向索引](../../ECG_SSL_LFBT-main/runlog/W3/failed_directions_index.csv)。
- [A-2 鲁棒性说明](../../ECG_SSL_LFBT-main/runlog/W3/robustness_README.md)及 `robustness_b0/c1/c2.csv`。
- [B-1 论文材料复算](../../ECG_SSL_LFBT-main/runlog/W3/paper_materials/paper_materials_manifest.json)：主表、paired delta、符号表和缺导汇总复算一致。
- [B-2 指标扩展](../../ECG_SSL_LFBT-main/runlog/W3/metrics_schema/SCHEMA.md)：单测 18/18，smoke 通过，默认路径保持兼容。
- [B-3 NFH 明细](../../ECG_SSL_LFBT-main/runlog/W3/nfh_exclusion_reconciliation.csv)：97/97 条剔除记录确定性复得。

最新主线提交：`e21ee51`；A-2、B-0、B-1、B-2、B-3 均已有 Git 交付记录。




