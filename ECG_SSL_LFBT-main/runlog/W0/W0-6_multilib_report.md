# W0-6 多库入库验收报告(N1 语料轨)

> 日期:2026-09-15|执行:`prepare_multilib.py`(预处理版本 `n1-multilib-1`)|验收脚本:`runlog/W0/verify_multilib.py`

## 1. 入库结果

| 库 | 记录数 | fs | 原始时长 | 状态 |
|---|---:|---:|---|---|
| cpsc_2018 | 6,877 | 500Hz | 6–60s | ✅ |
| cpsc_2018_extra | 3,453 | 500Hz | 6–60s | ✅ |
| georgia | 10,344 | 500Hz | 10s | ✅ |
| ptb | 516 | 1000Hz | 38–115s | ✅ |
| st_petersburg_incart | 74 | 257Hz | ~30min | ✅ |
| **合计** | **21,264** | — | — | **0 错误,724 秒** |

与《01》§1.4 登记逐库一致。**ptb-xl 未入库**(与下游 PTB-XL 同源,红线)。

## 2. 预处理口径(与冻结协议对齐)

1. 逐记录解析 .hea(采样率/导联名/Dx),按**导联名**取 8 导联 [II,III,V1–V6](兼容各库导联排列差异);
2. 中心裁剪 10s(native fs;不足对称补零,valid 区外置零)——ptb 1000Hz→裁 10,000 点、incart 257Hz→裁 2,570 点;
3. **整条记录统一标量 z-score**(valid 区统计,(x−μ)/(σ+1e-5)),与 `prepare_data.py` 冻结口径一致(不换算物理增益,z-score 仿射不变);
4. resample → [8, 2048] float32;常数导联置零并登记 `constant_leads`(如 incart I0002 的 V6 为原始数据真平直)。

## 3. 验收单测(全部 PASS)

- 文件名全局唯一(21,264);ptb-xl 未入库;与 PTB-XL 下游零交集(id 命名空间 `{lib}__{record_id}` 隔离);
- 抽样 200 条:shape (8,2048) / float32 / 有限值 / std≈1.0 全过;
- 抽查 20 条 SHA256 与 manifest 一致;
- 每库 8 条 × 8 导联可视化(`runlog/W0/multilib_viz/*.png`):各导联均为可辨识 P-QRS-T 形态,无噪声/平线/乱码面板(个别样本为快速心律,属数据本身特性)。

## 4. 产物位置

- 数据:`ECG_SSL_LFBT-main/data/multilib/samples/`(1.4GB,21,264 个 npy,**留本地不入 git**)
- 清单:`ECG_SSL_LFBT-main/data/multilib/manifest.json`(逐条 fs/长度/valid 区/常数导联/Dx/SHA256,**入 git**——数据完整性可随时校验)
- 留档:GitHub 无法承载 1.4GB 数据本体(单文件/仓库大小限制),按"代码+manifest+报告入 git、数据留本地"执行。

## 5. 下一步(W0-8)

硬链接合并 `pt_pretrain(17,418) + multilib(21,264) = 38,682` → `data/n1_pretrain/`,原版 LFBT 目标、B0 同配置(seed 0, 200ep, batch 128)预训练;判死活:PTB-XL LP ≥ B0(0.7177)且跨库改善。
