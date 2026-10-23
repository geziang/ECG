# thesis-p0-20261023 events

## 2026-10-23

- **22:57 认领**：HOSTS 行翻 🏃Win4090(2026-10-23 22:57)，commit `3c3c35c` 已推送。下发 HEAD=`08316d9`，pull --rebase 干净，工作树 clean、无 stash。
- **23:0x A0 实测**（preflight.json）：Win4090 / RTX 4090 24,564MiB（驱动 610.74）/ GPU 无训练进程 / conda DL：Python 3.11.9、torch 2.5.1+cu121（CUDA 可用）、numpy 1.26.3、scikit-learn 1.5.0、wfdb 4.1.2。与历史登记一致，本轮为实测值。
- **九 checkpoint 9/9 通过并备份**：M0=`checkpoint/confirm/c1_seed{0,2,4}`、M4=`checkpoint/joint_m4_noise/seed{0,2,4}`、M6=`checkpoint/joint_m6/seed{0,2,4}` 的 `encoder_group.pth`，SHA256 与 source_lock 逐个一致；已复制至 `D:\thesis_p0_20261023_backup\checkpoints\`（含 config.json），副本哈希复核一致。
- **local_assets 6/7 一致**：cpsc_multilabel.csv、cpsc_sample_order.txt、W2 freeze_manifest_v2.json、W5 lp_results.csv、JOINT results.csv、JOINT checkpoints_manifest.json 均与锁一致。**唯一不符：`export_cpsc_multilabel.py`**（实际 `638bcbce...` vs 锁 `c6de8aa4...`）。定因：办公机在 f106744 提交时对该脚本做了字节等价重构（csv.writer QUOTE_ALL、StringIO+write_text，11+/7−），来源锁哈希的是重构前工作文件；其产物 cpsc_multilabel.csv 哈希与锁一致（逐字节送达无误）。判定：标签/数据来源链完好，脚本差异属工具版本记录问题，不停线；实际送达哈希记入 source_and_checkpoint_manifest.json。
- **参考源码三仓库固定 commit 获取**（`D:\thesis_p0_20261023_backup\refs\`）：HeartLang@a08afd8、ecg_tokenizer@ce976ce、shapeefm@9bb1b6e，clone 后 9 个锁定文件 SHA256 9/9 一致。插曲：本机 git autocrlf=true 曾致 checkout 后 7 文件 CRLF 化哈希不符，git blob 哈希与锁一致证实仓库内容无误，已对三仓库设 autocrlf=false + eol=lf 重检出后全过。
- **HeartLang 主 checkpoint 解析**：HF `PKUDigitalHealth/HeartLang` revision `b8118c1bc34a43cc2202b6d30267244ac7fc52c1`（main，lastModified 2025-01-21），文件 `checkpoints/pretrain/MIMIC-IV/checkpoint-200.pth`，534,928,678 bytes，LFS sha256 `9dc89d72a3f1cacd941892103eaa523d44e373164cd4fadfb975907cc65864f6`。同目录 vqhbr/MIMIC-IV/checkpoint-100.pth 为任务书禁用的 VQ-HBR-only 权重，不取。**23:1x 下载完成，实测 SHA256 与 LFS oid 一致，verified=true**（存 `D:\thesis_p0_20261023_backup\heartlang_weights\`，经本机代理 7897）。
- **代码可恢复副本**：`git archive 3c3c35c` → `D:\thesis_p0_20261023_backup\code_snapshot_3c3c35c.zip`（66,775,479 bytes，sha256 `4dbc46f67156d8ac73207b86bde4a83e97cf4420f1a550d096fe6b041f22a6f8`）。A0 时点工作树 clean，无未提交差异需附带。执行代码 commit 在训练前另冻结（protocol_resolved.json）。

## A1–A4 执行期

- **A1 过（89d3d82）**：契约检查全过、train/val 导出隔离副本（SHA 锁）、test 零信号访问；实测归一化=整记录全局 z-score（协议文"逐导联"表述与实测不符，以实测记录，未二次归一化）；train 字节级重复 131 组（记录级观察）。
- **A2 完成**：9/9 作业（特征缓存 512 维、Linear(512,9) BCE 头）；seed0 工程门过；均值 M6 0.7318 > M4 0.7269 > M0 0.7177（三种子内排序一致）。首轮分组 AP 对 0 阳性类走了 sklearn 退化路径（告警），已改为显式分母并重跑 seed0（特征缓存复用，主指标逐位一致）。
- **A3 完成**：峰缓存 5492/5492（V5 5485+II 7+失败 0，无记录触发无峰回退；回退/边界/等距取峰以合成用例验证）；参数量 12713 逐位符合；64 条过拟合 0.684→0.050。12/12 作业：均值 D0 0.5973 > D1 0.5925 > D3 0.5865 > D2 0.5619；工程选线门（ref=D0）D3−D0 = −0.011、0/3 种子为正 → **不过门（有效阴性）**。峰扰动敏感性：±4 样点 Δ≤0.002，±8 样点 Δ∈[−0.004,+0.002]。
- **A4 启动适配（第三方环境排障，各≈5 分钟，训练从未启动故不属"作业失败重试"）**：①PEP420 包优先级——仓库根 utils/（常规包）遮蔽作者 utils/（命名空间包）→ 导入期摘除仓库根路径；②作者 utils.utils 顶部 import tensorboardX（本环境无）→ load/freeze 语义等价复刻（含严格 backbone 记账断言）；③HF 预训练 checkpoint 的 lm_head 词表 8192（VQ）vs 微调构造默认 1024 尺寸不符 → 查证分类器 forward(return_all_tokens=True) 不经过 lm_head，作为预训练专属头排除并记录（excluded_pretrain_only）。协议偏差记录：backbone 全程 eval（作者入口整模型 train()，任务书§6 要求冻结 LP 无 dropout）。
- **A4 数据**：作者原生管线（前 2500 点@500Hz→零填 5000→minmax[-3,3]→100Hz→QRSTokenizer fs100/max256/token96/12 导/xqrs lead I）作用于我们冻结 split+multihot（类序=本项目；作者 csv 列序仅作映射记录）；4809+683 全部分词，零退化记录，数据目录无任何 test_* 文件（启动断言）。
- **10-24 00:4x 收官**：A4 完成（AUPRC 0.4464/AUROC 0.8121/best_ep 92，backbone 冻结核验过，checkpoint 键扫描确认无 mlp_head=头为本批新初始化；150MB 完整模型备份至 D 盘 sha256 88d11a7d...）。22/22 作业收官，summarize 重建 results.csv+三 summary+resume_manifest；FINAL_REPORT.md 落盘；HOSTS 行翻 ✅。总账：GPU 0.24h/12h、CPU 0.06h/6h、工程≈1.6h/8h。按任务书停在 P1，不自动进入新预训练。
