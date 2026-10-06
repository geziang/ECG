# E2d TRC 机理分析摘要

- 样本: 各语料 384 条(类分层, rng 20261006); c2 seeds=[0, 1, 2, 3, 4]; 通道数 C=64×8导联
- ratio = ||shift_post||_1 / ||shift_pre||_1 (shift=通道时间均值相对 NFH 的偏移; <1 即 TRC 缩小跨域偏移)

| domain | mean_ratio | median_ratio | mean_rho | mean_ratio_lrms | median_ratio_lrms | n_cells |
|---|---|---|---|---|---|---|
| ptbxl | 1.1825 | 1.1786 | -0.0671 | 0.8137 | 0.8144 | 40 |
| cpsc | 1.2125 | 1.2128 | -0.0428 | 0.8591 | 0.8574 | 40 |
| chapman | 1.1990 | 1.1992 | -0.0366 | 0.8522 | 0.8524 | 40 |

- TRC 参数幅值(全 seed×lead×channel, n=2560): |gamma| mean=1.2888 p90=2.1895; |beta| mean=0.1215 p90=0.2572
- 判读: mean_ratio>1=均值层不缩偏(如实记录); ratio_lrms<1=RMS 能量尺度层缩偏(支持范数域适应机制); rho 正=调整与偏移同调
