# W9 table4_complete 草表(AUROC / AUPRC mean±SD, 3 seeds {0,2,4})

来源: W9=本批 30 次; 其余引用旧账本(见 source 列, summary.csv 有明细)。
数值 4 位小数(论文口径); 原值 10 位在 summary.csv。

| 方法 | PTB-XL AUROC | PTB-XL AUPRC | CPSC AUROC | CPSC AUPRC | Chapman AUROC | Chapman AUPRC | 来源 |
|---|---|---|---|---|---|---|---|
| S1 SimCLR | 0.8500±0.0029 | 0.6085±0.0025 | 0.8970±0.0018 | 0.6243±0.0021 | 0.9867±0.0009 | 0.9599±0.0030 | baseline_results.csv;w8_simph_results.csv;W9 |
| SimCLR+NSTDB(真实噪声视图) | 0.8533±0.0056 | 0.6096±0.0062 | 0.8986±0.0045 | 0.6284±0.0125 | 0.9863±0.0018 | 0.9576±0.0063 | w8_simph_results.csv;w8_simph_results.csv;W9 |
| S2 CLOCS | 0.8542±0.0028 | 0.6253±0.0041 | 0.9155±0.0049 | 0.6767±0.0092 | 0.9910±0.0017 | 0.9720±0.0057 | baseline_results.csv;W9;W9 |
| TSR 预任务 | 0.8149±0.0159 | 0.5760±0.0208 | 0.8491±0.0137 | 0.4932±0.0168 | 0.9802±0.0004 | 0.9366±0.0006 | w8_tsr_results.csv;w8_tsr_results.csv;W9 |
| M2 LFBT+Beat-RR | 0.8568±0.0019 | 0.6314±0.0020 | 0.9221±0.0015 | 0.6903±0.0049 | 0.9907±0.0018 | 0.9718±0.0061 | dv_results.csv;e1_dv_results.csv;W9;W9 |
| M5 LFBT+NSTDB(时间隔离版) | 0.8555±0.0049 | 0.6288±0.0045 | 0.9209±0.0044 | 0.6933±0.0085 | 0.9909±0.0005 | 0.9718±0.0017 | W9;W9;W9 |
