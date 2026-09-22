# 已完成报告

这里保存已经完成撰写、形成阶段性结论或完成归档的报告。完成报告不等于所有后续研究已经结束；若报告是阶段快照，文中会保留其对应日期和证据边界。

## 当前文件

- [09-论文材料冻结-2026-09-22.md](09-论文材料冻结-2026-09-22.md)：W2 主表、paired delta、TRC 参数量、缺导和失败方向材料已经冻结。
- [失败实验归档-2026-09-22.md](失败实验归档-2026-09-22.md)：已关闭/判负方向的统一台账。
- [W1执行报告-2026-09-21-主机A.md](W1执行报告-2026-09-21-主机A.md)：W1 主机 A 执行报告和证据边界。
- [项目总报告-2026-09-22.md](项目总报告-2026-09-22.md)：阶段性总览快照；它记录的是生成时点，不替代最新 W3 台账。

## 已完成的 W3 证据

以下机器可读结果保留在原运行目录，避免改动代码和审计引用：

- [A-0 冻结复核](../../ECG_SSL_LFBT-main/runlog/W3/freeze_audit.json)：11/11 checkpoint SHA 一致，账本无重复。
- [B-0 冻结材料审计](../../ECG_SSL_LFBT-main/runlog/W3/freeze_audit_hostB.json)：审计通过。
- [协议审计](../../ECG_SSL_LFBT-main/runlog/W3/protocol_audit.md)。
- [失败方向索引](../../ECG_SSL_LFBT-main/runlog/W3/failed_directions_index.csv)。
- [A-2 鲁棒性说明](../../ECG_SSL_LFBT-main/runlog/W3/robustness_README.md)及 `robustness_b0/c1/c2.csv`。

最新主线提交：`044820b`。

