# JOINT_20261013 运行前检查记录(protocol §8)

执行:主机A,2026-10-13 19:0x,git @ 5072574(protocol.md 同步版)。

## 数据与缓存

| 检查项 | 结果 |
|---|---|
| NFH 文件数(`data/pt_pretrain_nfh/samples`) | 34,905 ✅(W8 任务书口径一致) |
| `pt_rpeaks_nfh.npz` 条目 | 34,905;与 NFH stem 集合**一一对应**(0 缺 0 多)✅ |
| 有效 R 峰(≥3 峰) | 34,898 / 34,905 = **99.98%** ✅ |
| 0<峰数<3 | 0 条;空(检测失败) 7 条(= `runlog/W6/logs/prep_rpeaks_nfh.log` 统计: total 34905, failed 7, success_rate 0.9998) |
| R 峰<3 记录清单 | prep 日志为汇总口径(不落逐条清单);空键训练时 valid=0 不参与辅助损失 |
| ptbxl / cpsc / chapman 文件数 | 17,092 / 6,877 / 10,216 ✅ 在位 |
| CPSC 分区 | W5A 修正患者级分区 test=1385(冻结口径,未动) |
| `pt_rpeaks.npz`(旧) | **17,418 条 = PTB-XL 语料缓存**——A1/A2 s0 错配证据,见任务书 §二 |

## 实现口径(代码核验)

- **RR z-score 来源**:DVDataset 以缓存全库统计 mu/sd;缓存仅覆盖 NFH(预训练训练集)→ 满足"统计量只能由预训练训练集计算"。
- **噪声两视图独立**:run_pt.py L783-791,t/t2 各自 `Compose([NoiseInjector, …])`,同一 injector 逐调用独立抽取(W7 E5 起 3 种子在用,无状态泄漏)。
- **偏移域**:训练 `--nstdb-offset first_half`(run_pt 旗标);评测 `draw_offset(..., "last_half")`(E6 实现,本批次复用)。短记录回退全域(既有行为)。
- **§4.2 顺序**:R 峰缓存离线生成于干净信号;心拍视图/RR 目标取自 `raw = np.load(...)`(无变换干净信号);噪声仅进两个整段视图的变换链——代码路径天然满足。
- **心拍视图过同一组编码器**:`_embed_beat` 走 `backbone_group[i]`(trc=1 时含 GRN1D),与整段视图共享。

## 待 smoke 关闭项(任务书 §五)

- checkpoint 保存 TRC/RR head/优化器/epoch/RNG;resume 损失与 lr 连续;LP trc=1 加载联合 checkpoint;config.json 完整性(含 rpeak_npz)。
