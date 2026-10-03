# paper_materials_v3 · CPSC 修正分区合并材料

**CPSC 行为修正分区数据（W5A 审计重建 test=1385，见 `runlog/W5/cpsc_audit.md`），
与 W2/W4 时代的污染分区行（test=2004 新旧并集）不可比。**
W2/W3/W4 冻结文件原样保留作污染分区历史记录，本目录是其"CPSC 行换新"版本：

- `main_table_merged_v3.csv`: PTB/Chapman 行 = v2 原样（源列不动）；CPSC LP 5 方法行全部换
  W5 数（b0 在 v2 的两条历史行——W2/W3 冻结 n=1 行与 W4 外部基线 n=2 行——同为污染分区产物，
  一并替换为 W5 n=3 单行）。
- `paired_stats_v3.csv`: PTB/Chapman 行 = v2 原样；CPSC 四行 = `runlog/W5/stats/paired_stats.csv`
  （RNG 20260923, 10k 次, record-level 配对 bootstrap）。
- `paired_delta_v3.csv` / `seed_sign_v3.csv`: 非 cpsc 行 = W3 原样（补 metric 列标注 auroc）；
  CPSC 行 = W5 账本种子级新算，**auroc 与 auprc 双口径**（B-2 指定交付为 AUPRC 口径：
  C2−C1 = +0.70/+0.59/+1.20 mean +0.83, 3/3⊕, t-CI [+0.02,+1.64]）。
  W3 表中被替换剔除的唯一 cpsc 行（lp/cpsc/C2-C1 auroc +2.30, 污染分区）仍留在 W3 冻结文件中。

统计口径: 全部 record-level 描述性证据, 禁写"统计显著/patient-level"(红线)。
逐记录预测输入: `runlog/W5/predictions/`(15 套, protocol_id=w5-cpscredo)。
