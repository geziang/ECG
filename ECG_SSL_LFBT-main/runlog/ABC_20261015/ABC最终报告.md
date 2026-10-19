# ABC_20261015 联合框架补充实验·最终报告(自动生成 2026-10-19 15:12)

> 预注册方案: ABC联合框架补充实验方案.md(ABC-SUPP-20261015); 冻结协议: protocol_frozen.md。
> 本报告由 run_abc_stats.py --report 自动装配, 结论措辞按方案 §3.2/§12 决策表;
> 证据分级: E1 确认性正增益 / E2 稳定描述性 / E3 局部 / E4 中性或负(总表保留)。

## 一、完成度(方案 §15 验收)

- results.csv 已落 370 行(ABC 新增); 正增益账本 6 条
  (E1×2 / E2×3 / E3×1)。
- 明细对账见 events.md; 失败与重跑记录见同文件。

## 二、主终点(CPSC LP macro-AUPRC, 核心比较族, Holm 校正)

```
ts,task,dataset,pair,metric,family,delta_mean_pt,ci_lo_pt,ci_hi_pt,p_raw,p_adjusted,pos_seeds,total_seeds,per_seed_pt
```
(完整表见 paired_stats.csv; family=core 行为核心族)

## 三、A/B/C 因子交互(方案 §10.3 全式 + 条件边际)

```
ts,name,metric,mean_pt,ci_lo_pt,ci_hi_pt,p_emp,per_seed_pt
2026-10-18 19:58,I_AB|C=0,auprc,-0.622,-1.4,0.142,0.1134,{"0": -0.537, "2": -0.16, "4": -1.169}
2026-10-18 19:58,I_AB|C=1,auprc,-0.133,-0.816,0.521,0.6764,{"0": 0.021, "2": 0.32, "4": -0.739}
2026-10-18 19:58,I_AC|B=0,auprc,0.111,-0.594,0.813,0.7426,{"0": 0.595, "2": -0.011, "4": -0.252}
2026-10-18 19:58,I_AC|B=1,auprc,0.6,-0.176,1.346,0.1278,{"0": 1.152, "2": 0.469, "4": 0.178}
2026-10-18 19:58,I_BC|A=0,auprc,-0.743,-1.479,0.004,0.0512,{"0": -1.385, "2": -1.235, "4": 0.39}
2026-10-18 19:58,I_BC|A=1,auprc,-0.254,-1.081,0.544,0.53,{"0": -0.827, "2": -0.755, "4": 0.82}
2026-10-18 19:58,I_ABC,auprc,0.489,-0.566,1.479,0.367,{"0": 0.557, "2": 0.48, "4": 0.43}
2026-10-18 19:58,A|B+C(m4-m7),auprc,0.808,0.287,1.316,0.0022,{"0": 1.316, "2": 0.899, "4": 0.211}
2026-10-18 19:58,B|A+C(m4-m6),auprc,-0.051,-0.618,0.495,0.8256,{"0": -0.458, "2": 0.352, "4": -0.046}
2026-10-18 19:58,C|A+B(m4-m3),auprc,0.275,-0.526,1.088,0.492,{"0": 0.409, "2": 0.061, "4": 0.354}
2026-10-18 19:58,I_AB|C=0,auroc,-0.102,-0.378,0.151,0.4614,{"0": 0.041, "2": -0.119, "4": -0.227}
```

措辞规则: 交互为正且 CI 不跨 0 → "协同增益得到统计支持"; 为正但跨 0 → "正向三阶交互趋势";
否则 → "联合设计/鲁棒性扩展"(不写超加性协同)。

## 四、B 内部拆分(M4−M8/M4−M9/M8−M6/M9−M6)

见 paired_stats.csv 中 pair=m4:m8 / m4:m9 / m8:m6 / m9:m6。
若 RR 平均贡献≈0 且 Beat 保持正向: B 表述为"以 R 峰对齐 Beat 形态为主要有效成分、
RR 为被验证但贡献有限的辅助目标"(不删除 RR 结果)。

## 五、噪声鲁棒(27 格 last_half, 确定性同噪声配对)

```
model,seed,dataset,noise_type,snr,auroc,auprc,source
m4,0,ptbxl,bw,0,0.8711,0.6264,JOINT
m3,0,ptbxl,bw,0,0.8489,0.5875,JOINT
m4,0,ptbxl,bw,5,0.8836,0.6449,JOINT
m3,0,ptbxl,bw,5,0.8560,0.5992,JOINT
m4,0,ptbxl,bw,10,0.8888,0.6579,JOINT
m3,0,ptbxl,bw,10,0.8650,0.6127,JOINT
m4,0,ptbxl,ma,0,0.8591,0.6115,JOINT
```
逐记录 bootstrap 族(pair=m4:m3 / m7:m6 / m8:m6 / m9:m6 的 auroc_mean9cells 行)
见 paired_stats.csv; 其余噪声对为行级描述性(table_noise27.csv)。

## 六、机制(N1/N2 三种子)

见 paired_stats.csv pair=n1:m3 / n2:m3。
N1−M3 若 3/3 为负且 CI 不跨 0 → "R 峰对齐是有效 Beat 约束的必要条件";
N2−M3≈0 → "当前权重和协议下未观察到 RR 标签信息的独立贡献"(不写"RR 永远无效")。

## 七、第二语料(P1-2, M4b=PTB-XL 预训练)

见 paired_stats.csv pair=m4b:m4(cpsc LP/ft10)与 table_lp_3domains.csv m4b 行。
定位: 支持"框架不只依赖单一预训练语料", 不用于重新搜索损失权重。

## 八、多标签敏感性(P1-3)

```
ts,ckpt,seed,macro_auroc,macro_auprc,macro_f1,macro_sens,macro_spec,protocol_id,git_sha,checkpoint_sha256,note
2026-10-19 13:58,m0,0,0.8343213162098589,0.48854099108807614,0.49364302550333494,0.6007885292461592,0.8727,abc-20261015-multilib,f36e78b,de0c28eedb30,one-hot真值(W6 口径, 敏感性分析不替换主结果)
2026-10-19 13:58,m0,2,0.8340517162292371,0.47433302438828906,0.48166878495006976,0.6014866107556566,0.8899,abc-20261015-multilib,f36e78b,26ebf0e26e5a,one-hot真值(W6 口径, 敏感性分析不替换主结果)
2026-10-19 13:58,m0,4,0.8280649202802445,0.4865942031366664,0.4899498952564507,0.597143450203859,0.8871,abc-20261015-multilib,f36e78b,e599dbcdb693,one-hot真值(W6 口径, 敏感性分析不替换主结果)
2026-10-19 13:58,m1,0,0.854046998659282,0.5262210808924752,0.5293820294790683,0.6122315199238018,0.9141,abc-20261015-multilib,f36e78b,af242a89c340,one-hot真值(W6 口径, 敏感性分析不替换主结果)
2026-10-19 13:58,m1,2,0.8609074782135561,0.530834007239238,0.5250412624398887,0.5911022110697359,0.9227,abc-20261015-multilib,f36e78b,d3f8979f70cd,one-hot真值(W6 口径, 敏感性分析不替换主结果)
2026-10-19 13:58,m1,4,0.8530151102868917,0.5194397862540641,0.5111427498138649,0.5717692107666292,0.9207,abc-20261015-multilib,f36e78b,5979f2301720,one-hot真值(W6 口径, 敏感性分析不替换主结果)
2026-10-19 13:58,m2,0,0.8371457195516293,0.49262180456295673,0.49473612665065997,0.6124435002451889,0.8790,abc-20261015-multilib,f36e78b,123c4018768a,one-hot真值(W6 口径, 敏感性分析不替换主结果)
```
口径: W6 冻结(one-hot 主类真值, 限制如实入账); 主分析仍为单标签口径。

## 九、效率与成本(P0-4)

```
model,params_total,params_encoder,params_head,macs_per_record,macs_method,pt_wall_100ep,pt_epoch_mean,pt_vram_peak,pt_source,ft10_wall_cpsc,infer_latency_b1_ms_med,infer_latency_b1_ms_iqr,infer_throughput_b64_recs_s,infer_vram_peak_mib,env,seeds_pt
m0,630345,625728,4617,145101312,hook Conv1d+Linear(BN/激活不计),n/a,n/a,n/a,n/a(历史批次未记录),n/a,6.37,0.287,9664.4,28.9,NVIDIA GeForce RTX 4090;cuda12.1;torch2.5.1+cu121;len2048;cudnn90100,0
m1,631369,626752,4617,145101312,hook Conv1d+Linear(BN/激活不计),n/a,n/a,n/a,n/a(历史批次未记录),n/a,6.976,0.302,8946.4,29.0,NVIDIA GeForce RTX 4090;cuda12.1;torch2.5.1+cu121;len2048;cudnn90100,0
m2,630345,625728,4617,145101312,hook Conv1d+Linear(BN/激活不计),n/a,n/a,n/a,n/a(历史批次未记录),n/a,6.379,0.267,9510.6,28.9,NVIDIA GeForce RTX 4090;cuda12.1;torch2.5.1+cu121;len2048;cudnn90100,0
m3,631369,626752,4617,145101312,hook Conv1d+Linear(BN/激活不计),n/a,n/a,n/a,n/a(历史批次未记录),30.5min,6.833,0.273,8986.6,29.0,NVIDIA GeForce RTX 4090;cuda12.1;torch2.5.1+cu121;len2048;cudnn90100,0
m4,631369,626752,4617,145101312,hook Conv1d+Linear(BN/激活不计),n/a,n/a,n/a,n/a(历史批次未记录),30.9min,6.843,0.28,8981.4,29.0,NVIDIA GeForce RTX 4090;cuda12.1;torch2.5.1+cu121;len2048;cudnn90100,0
m5,630345,625728,4617,145101312,hook Conv1d+Linear(BN/激活不计),n/a,n/a,n/a,n/a(历史批次未记录),n/a,6.276,0.292,9745.5,28.9,NVIDIA GeForce RTX 4090;cuda12.1;torch2.5.1+cu121;len2048;cudnn90100,0
m6,631369,626752,4617,145101312,hook Conv1d+Linear(BN/激活不计),n/a,n/a,n/a,n/a(历史批次未记录),34.3min,6.834,0.292,9102.8,29.0,NVIDIA GeForce RTX 4090;cuda12.1;torch2.5.1+cu121;len2048;cudnn90100,0
m7,630345,625728,4617,145101312,hook Conv1d+Linear(BN/激活不计),129.0min,1.29min,8648MiB,ABC_20261015,34.3min,6.34,0.253,9058.1,28.9,NVIDIA GeForce RTX 4090;cuda12.1;torch2.5.1+cu121;len2048;cudnn90100,3
m8,631369,626752,4617,145101312,hook Conv1d+Linear(BN/激活不计),162.0min,1.62min,11836MiB,ABC_20261015,35.1min,6.846,0.336,8834.1,29.0,NVIDIA GeForce RTX 4090;cuda12.1;torch2.5.1+cu121;len2048;cudnn90100,3
m9,631369,626752,4617,145101312,hook Conv1d+Linear(BN/激活不计),114.6min,1.15min,8354MiB,ABC_20261015,34.5min,6.841,0.326,9144.1,29.0,NVIDIA GeForce RTX 4090;cuda12.1;torch2.5.1+cu121;len2048;cudnn90100,3
```

## 十、论文决策表对照(方案 §12)

- M4 对基线/单组件显著为正(E1) → 主表+摘要 "A+B+C 获得稳定联合增益"
- 正但 CI 跨 0(E2) → 主表或补表 "描述性正增益", 不写"显著"
- I_ABC>0 跨 0 → 交互表 "正向三阶交互趋势"; 不跨 0 → 主文 "协同增益得到统计支持"
- I_ABC≤0 但总体/鲁棒为正 → "有效联合/鲁棒框架, 无超加性协同证据"
- 负向比较 → 总表保留, 明确代价与边界

## 附:产物索引

results.csv / paired_stats.csv / factorial_stats.csv / positive_gain_ledger.csv /
efficiency.csv / multilib_results.csv / checkpoints_manifest.json /
table_lp_3domains.csv / table_ft10.csv / table_noise27.csv / predictions/ / logs/ / events.md
