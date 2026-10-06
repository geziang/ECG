# E4d 心拍分层误差分析摘要

- 记录特征行: 5171; 预测目录: 63; 对账: 全部一致


## 逐桶配对差 (对 seed 取均值, delta=acc_other − acc_b0)

| domain | eval | feature | bucket | pair | n | mean_delta |
|---|---|---|---|---|---|---|
| chapman | ft10 | mean_rr_ms | q1 | b0-c1 | 512 | -0.0059 |
| chapman | ft10 | mean_rr_ms | q1 | b0-c2 | 512 | -0.0026 |
| chapman | ft10 | mean_rr_ms | q2 | b0-c1 | 512 | +0.0059 |
| chapman | ft10 | mean_rr_ms | q2 | b0-c2 | 512 | +0.0078 |
| chapman | ft10 | mean_rr_ms | q3 | b0-c1 | 511 | +0.0059 |
| chapman | ft10 | mean_rr_ms | q3 | b0-c2 | 511 | +0.0130 |
| chapman | ft10 | mean_rr_ms | q4 | b0-c1 | 512 | +0.0007 |
| chapman | ft10 | mean_rr_ms | q4 | b0-c2 | 512 | +0.0059 |
| chapman | ft10 | sdnn_ms | q1 | b0-c1 | 512 | +0.0039 |
| chapman | ft10 | sdnn_ms | q1 | b0-c2 | 512 | +0.0072 |
| chapman | ft10 | sdnn_ms | q2 | b0-c1 | 512 | +0.0007 |
| chapman | ft10 | sdnn_ms | q2 | b0-c2 | 512 | +0.0046 |
| chapman | ft10 | sdnn_ms | q3 | b0-c1 | 511 | +0.0033 |
| chapman | ft10 | sdnn_ms | q3 | b0-c2 | 511 | +0.0111 |
| chapman | ft10 | sdnn_ms | q4 | b0-c1 | 512 | -0.0013 |
| chapman | ft10 | sdnn_ms | q4 | b0-c2 | 512 | +0.0013 |
| chapman | lp | mean_rr_ms | q1 | b0-c1 | 512 | +0.0264 |
| chapman | lp | mean_rr_ms | q1 | b0-c2 | 512 | +0.0234 |
| chapman | lp | mean_rr_ms | q2 | b0-c1 | 512 | +0.0195 |
| chapman | lp | mean_rr_ms | q2 | b0-c2 | 512 | +0.0254 |
| chapman | lp | mean_rr_ms | q3 | b0-c1 | 511 | +0.0088 |
| chapman | lp | mean_rr_ms | q3 | b0-c2 | 511 | +0.0108 |
| chapman | lp | mean_rr_ms | q4 | b0-c1 | 512 | +0.0166 |
| chapman | lp | mean_rr_ms | q4 | b0-c2 | 512 | +0.0166 |
| chapman | lp | sdnn_ms | q1 | b0-c1 | 512 | +0.0254 |
| chapman | lp | sdnn_ms | q1 | b0-c2 | 512 | +0.0244 |
| chapman | lp | sdnn_ms | q2 | b0-c1 | 512 | +0.0283 |
| chapman | lp | sdnn_ms | q2 | b0-c2 | 512 | +0.0293 |
| chapman | lp | sdnn_ms | q3 | b0-c1 | 511 | +0.0117 |
| chapman | lp | sdnn_ms | q3 | b0-c2 | 511 | +0.0147 |
| chapman | lp | sdnn_ms | q4 | b0-c1 | 512 | +0.0059 |
| chapman | lp | sdnn_ms | q4 | b0-c2 | 512 | +0.0078 |
| cpsc | ft10 | mean_rr_ms | q1 | b0-c1 | 347 | +0.0010 |
| cpsc | ft10 | mean_rr_ms | q1 | b0-c2 | 347 | -0.0019 |
| cpsc | ft10 | mean_rr_ms | q2 | b0-c1 | 346 | -0.0048 |
| cpsc | ft10 | mean_rr_ms | q2 | b0-c2 | 346 | +0.0173 |
| cpsc | ft10 | mean_rr_ms | q3 | b0-c1 | 346 | -0.0010 |
| cpsc | ft10 | mean_rr_ms | q3 | b0-c2 | 346 | +0.0058 |
| cpsc | ft10 | mean_rr_ms | q4 | b0-c1 | 346 | -0.0154 |
| cpsc | ft10 | mean_rr_ms | q4 | b0-c2 | 346 | -0.0010 |
| cpsc | ft10 | sdnn_ms | q1 | b0-c1 | 347 | -0.0058 |
| cpsc | ft10 | sdnn_ms | q1 | b0-c2 | 347 | -0.0067 |
| cpsc | ft10 | sdnn_ms | q2 | b0-c1 | 346 | -0.0039 |
| cpsc | ft10 | sdnn_ms | q2 | b0-c2 | 346 | +0.0125 |
| cpsc | ft10 | sdnn_ms | q3 | b0-c1 | 346 | -0.0125 |
| cpsc | ft10 | sdnn_ms | q3 | b0-c2 | 346 | +0.0067 |
| cpsc | ft10 | sdnn_ms | q4 | b0-c1 | 346 | +0.0019 |
| cpsc | ft10 | sdnn_ms | q4 | b0-c2 | 346 | +0.0077 |
| cpsc | lp | mean_rr_ms | q1 | b0-c1 | 347 | +0.0130 |
| cpsc | lp | mean_rr_ms | q1 | b0-c2 | 347 | +0.0101 |
| cpsc | lp | mean_rr_ms | q2 | b0-c1 | 346 | -0.0029 |
| cpsc | lp | mean_rr_ms | q2 | b0-c2 | 346 | -0.0014 |
| cpsc | lp | mean_rr_ms | q3 | b0-c1 | 346 | +0.0188 |
| cpsc | lp | mean_rr_ms | q3 | b0-c2 | 346 | +0.0130 |
| cpsc | lp | mean_rr_ms | q4 | b0-c1 | 346 | -0.0318 |
| cpsc | lp | mean_rr_ms | q4 | b0-c2 | 346 | -0.0130 |
| cpsc | lp | sdnn_ms | q1 | b0-c1 | 347 | -0.0144 |
| cpsc | lp | sdnn_ms | q1 | b0-c2 | 347 | -0.0000 |
| cpsc | lp | sdnn_ms | q2 | b0-c1 | 346 | -0.0029 |
| cpsc | lp | sdnn_ms | q2 | b0-c2 | 346 | -0.0029 |
| cpsc | lp | sdnn_ms | q3 | b0-c1 | 346 | +0.0173 |
| cpsc | lp | sdnn_ms | q3 | b0-c2 | 346 | +0.0000 |
| cpsc | lp | sdnn_ms | q4 | b0-c1 | 346 | -0.0029 |
| cpsc | lp | sdnn_ms | q4 | b0-c2 | 346 | +0.0116 |
| cpsc | lp_w5 | mean_rr_ms | q1 | b0-c1 | 347 | +0.0058 |
| cpsc | lp_w5 | mean_rr_ms | q1 | b0-c2 | 347 | +0.0086 |
| cpsc | lp_w5 | mean_rr_ms | q2 | b0-c1 | 346 | +0.0077 |
| cpsc | lp_w5 | mean_rr_ms | q2 | b0-c2 | 346 | +0.0193 |
| cpsc | lp_w5 | mean_rr_ms | q3 | b0-c1 | 346 | +0.0279 |
| cpsc | lp_w5 | mean_rr_ms | q3 | b0-c2 | 346 | +0.0270 |
| cpsc | lp_w5 | mean_rr_ms | q4 | b0-c1 | 346 | -0.0173 |
| cpsc | lp_w5 | mean_rr_ms | q4 | b0-c2 | 346 | -0.0039 |
| cpsc | lp_w5 | sdnn_ms | q1 | b0-c1 | 347 | -0.0134 |
| cpsc | lp_w5 | sdnn_ms | q1 | b0-c2 | 347 | -0.0029 |
| cpsc | lp_w5 | sdnn_ms | q2 | b0-c1 | 346 | +0.0116 |
| cpsc | lp_w5 | sdnn_ms | q2 | b0-c2 | 346 | +0.0116 |
| cpsc | lp_w5 | sdnn_ms | q3 | b0-c1 | 346 | +0.0106 |
| cpsc | lp_w5 | sdnn_ms | q3 | b0-c2 | 346 | +0.0135 |
| cpsc | lp_w5 | sdnn_ms | q4 | b0-c1 | 346 | +0.0154 |
| cpsc | lp_w5 | sdnn_ms | q4 | b0-c2 | 346 | +0.0289 |
| ptbxl | ft20 | mean_rr_ms | q1 | b0-c1 | 435 | -0.0100 |
| ptbxl | ft20 | mean_rr_ms | q1 | b0-c2 | 435 | -0.0222 |
| ptbxl | ft20 | mean_rr_ms | q2 | b0-c1 | 435 | +0.0092 |
| ptbxl | ft20 | mean_rr_ms | q2 | b0-c2 | 435 | +0.0038 |
| ptbxl | ft20 | mean_rr_ms | q3 | b0-c1 | 435 | -0.0023 |
| ptbxl | ft20 | mean_rr_ms | q3 | b0-c2 | 435 | +0.0015 |
| ptbxl | ft20 | mean_rr_ms | q4 | b0-c1 | 434 | -0.0069 |
| ptbxl | ft20 | mean_rr_ms | q4 | b0-c2 | 434 | +0.0000 |
| ptbxl | ft20 | sdnn_ms | q1 | b0-c1 | 435 | +0.0069 |
| ptbxl | ft20 | sdnn_ms | q1 | b0-c2 | 435 | -0.0107 |
| ptbxl | ft20 | sdnn_ms | q2 | b0-c1 | 435 | -0.0077 |
| ptbxl | ft20 | sdnn_ms | q2 | b0-c2 | 435 | +0.0031 |
| ptbxl | ft20 | sdnn_ms | q3 | b0-c1 | 434 | -0.0077 |
| ptbxl | ft20 | sdnn_ms | q3 | b0-c2 | 434 | -0.0123 |
| ptbxl | ft20 | sdnn_ms | q4 | b0-c1 | 435 | -0.0015 |
| ptbxl | ft20 | sdnn_ms | q4 | b0-c2 | 435 | +0.0031 |
| ptbxl | ft40 | mean_rr_ms | q1 | b0-c1 | 435 | -0.0115 |
| ptbxl | ft40 | mean_rr_ms | q1 | b0-c2 | 435 | -0.0123 |
| ptbxl | ft40 | mean_rr_ms | q2 | b0-c1 | 435 | +0.0092 |
| ptbxl | ft40 | mean_rr_ms | q2 | b0-c2 | 435 | +0.0023 |
| ptbxl | ft40 | mean_rr_ms | q3 | b0-c1 | 435 | -0.0054 |
| ptbxl | ft40 | mean_rr_ms | q3 | b0-c2 | 435 | -0.0069 |
| ptbxl | ft40 | mean_rr_ms | q4 | b0-c1 | 434 | -0.0069 |
| ptbxl | ft40 | mean_rr_ms | q4 | b0-c2 | 434 | -0.0131 |
| ptbxl | ft40 | sdnn_ms | q1 | b0-c1 | 435 | +0.0054 |
| ptbxl | ft40 | sdnn_ms | q1 | b0-c2 | 435 | -0.0100 |
| ptbxl | ft40 | sdnn_ms | q2 | b0-c1 | 435 | -0.0069 |
| ptbxl | ft40 | sdnn_ms | q2 | b0-c2 | 435 | -0.0092 |
| ptbxl | ft40 | sdnn_ms | q3 | b0-c1 | 434 | -0.0154 |
| ptbxl | ft40 | sdnn_ms | q3 | b0-c2 | 434 | -0.0200 |
| ptbxl | ft40 | sdnn_ms | q4 | b0-c1 | 435 | +0.0023 |
| ptbxl | ft40 | sdnn_ms | q4 | b0-c2 | 435 | +0.0092 |
| ptbxl | lp | mean_rr_ms | q1 | b0-c1 | 435 | -0.0253 |
| ptbxl | lp | mean_rr_ms | q1 | b0-c2 | 435 | -0.0253 |
| ptbxl | lp | mean_rr_ms | q2 | b0-c1 | 435 | -0.0149 |
| ptbxl | lp | mean_rr_ms | q2 | b0-c2 | 435 | -0.0287 |
| ptbxl | lp | mean_rr_ms | q3 | b0-c1 | 435 | -0.0069 |
| ptbxl | lp | mean_rr_ms | q3 | b0-c2 | 435 | -0.0103 |
| ptbxl | lp | mean_rr_ms | q4 | b0-c1 | 434 | -0.0161 |
| ptbxl | lp | mean_rr_ms | q4 | b0-c2 | 434 | -0.0173 |
| ptbxl | lp | sdnn_ms | q1 | b0-c1 | 435 | -0.0184 |
| ptbxl | lp | sdnn_ms | q1 | b0-c2 | 435 | -0.0310 |
| ptbxl | lp | sdnn_ms | q2 | b0-c1 | 435 | -0.0138 |
| ptbxl | lp | sdnn_ms | q2 | b0-c2 | 435 | -0.0195 |
| ptbxl | lp | sdnn_ms | q3 | b0-c1 | 434 | -0.0184 |
| ptbxl | lp | sdnn_ms | q3 | b0-c2 | 434 | -0.0230 |
| ptbxl | lp | sdnn_ms | q4 | b0-c1 | 435 | -0.0126 |
| ptbxl | lp | sdnn_ms | q4 | b0-c2 | 435 | -0.0080 |

## cpsc ft10 (mean_rr_ms 四分位桶, acc 按 model 对 seed 取均值)

| bucket | n | b0 | c1 | c2 |
|---|---|---|---|---|
| q1 | 347 | 0.7646 | 0.7656 | 0.7627 |
| q2 | 346 | 0.6917 | 0.6869 | 0.7091 |
| q3 | 346 | 0.7119 | 0.7110 | 0.7177 |
| q4 | 346 | 0.7023 | 0.6869 | 0.7013 |

## cpsc lp (mean_rr_ms 四分位桶, acc 按 model 对 seed 取均值)

| bucket | n | b0 | c1 | c2 |
|---|---|---|---|---|
| q1 | 347 | 0.7983 | 0.8112 | 0.8084 |
| q2 | 346 | 0.7370 | 0.7341 | 0.7355 |
| q3 | 346 | 0.7645 | 0.7832 | 0.7775 |
| q4 | 346 | 0.7587 | 0.7269 | 0.7457 |

## ptbxl ft20 (mean_rr_ms 四分位桶, acc 按 model 对 seed 取均值)

| bucket | n | b0 | c1 | c2 |
|---|---|---|---|---|
| q1 | 435 | 0.7433 | 0.7333 | 0.7211 |
| q2 | 435 | 0.7655 | 0.7747 | 0.7693 |
| q3 | 435 | 0.7870 | 0.7847 | 0.7885 |
| q4 | 434 | 0.7742 | 0.7673 | 0.7742 |

## chapman ft10 (mean_rr_ms 四分位桶, acc 按 model 对 seed 取均值)

| bucket | n | b0 | c1 | c2 |
|---|---|---|---|---|
| q1 | 512 | 0.8900 | 0.8841 | 0.8874 |
| q2 | 512 | 0.9375 | 0.9434 | 0.9453 |
| q3 | 511 | 0.9569 | 0.9628 | 0.9700 |
| q4 | 512 | 0.9355 | 0.9362 | 0.9414 |
