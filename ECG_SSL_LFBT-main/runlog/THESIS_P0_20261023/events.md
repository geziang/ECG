# thesis-p0-20261023 events

## 2026-10-23

- **22:57 认领**：HOSTS 行翻 🏃Win4090(2026-10-23 22:57)，commit `3c3c35c` 已推送。下发 HEAD=`08316d9`，pull --rebase 干净，工作树 clean、无 stash。
- **23:0x A0 实测**（preflight.json）：Win4090 / RTX 4090 24,564MiB（驱动 610.74）/ GPU 无训练进程 / conda DL：Python 3.11.9、torch 2.5.1+cu121（CUDA 可用）、numpy 1.26.3、scikit-learn 1.5.0、wfdb 4.1.2。与历史登记一致，本轮为实测值。
- **九 checkpoint 9/9 通过并备份**：M0=`checkpoint/confirm/c1_seed{0,2,4}`、M4=`checkpoint/joint_m4_noise/seed{0,2,4}`、M6=`checkpoint/joint_m6/seed{0,2,4}` 的 `encoder_group.pth`，SHA256 与 source_lock 逐个一致；已复制至 `D:\thesis_p0_20261023_backup\checkpoints\`（含 config.json），副本哈希复核一致。
- **local_assets 6/7 一致**：cpsc_multilabel.csv、cpsc_sample_order.txt、W2 freeze_manifest_v2.json、W5 lp_results.csv、JOINT results.csv、JOINT checkpoints_manifest.json 均与锁一致。**唯一不符：`export_cpsc_multilabel.py`**（实际 `638bcbce...` vs 锁 `c6de8aa4...`）。定因：办公机在 f106744 提交时对该脚本做了字节等价重构（csv.writer QUOTE_ALL、StringIO+write_text，11+/7−），来源锁哈希的是重构前工作文件；其产物 cpsc_multilabel.csv 哈希与锁一致（逐字节送达无误）。判定：标签/数据来源链完好，脚本差异属工具版本记录问题，不停线；实际送达哈希记入 source_and_checkpoint_manifest.json。
- **参考源码三仓库固定 commit 获取**（`D:\thesis_p0_20261023_backup\refs\`）：HeartLang@a08afd8、ecg_tokenizer@ce976ce、shapeefm@9bb1b6e，clone 后 9 个锁定文件 SHA256 9/9 一致。插曲：本机 git autocrlf=true 曾致 checkout 后 7 文件 CRLF 化哈希不符，git blob 哈希与锁一致证实仓库内容无误，已对三仓库设 autocrlf=false + eol=lf 重检出后全过。
- **HeartLang 主 checkpoint 解析**：HF `PKUDigitalHealth/HeartLang` revision `b8118c1bc34a43cc2202b6d30267244ac7fc52c1`（main，lastModified 2025-01-21），文件 `checkpoints/pretrain/MIMIC-IV/checkpoint-200.pth`，534,928,678 bytes，LFS sha256 `9dc89d72a3f1cacd941892103eaa523d44e373164cd4fadfb975907cc65864f6`。同目录 vqhbr/MIMIC-IV/checkpoint-100.pth 为任务书禁用的 VQ-HBR-only 权重，不取。**23:1x 下载完成，实测 SHA256 与 LFS oid 一致，verified=true**（存 `D:\thesis_p0_20261023_backup\heartlang_weights\`，经本机代理 7897）。
- **代码可恢复副本**：`git archive 3c3c35c` → `D:\thesis_p0_20261023_backup\code_snapshot_3c3c35c.zip`（66,775,479 bytes，sha256 `4dbc46f67156d8ac73207b86bde4a83e97cf4420f1a550d096fe6b041f22a6f8`）。A0 时点工作树 clean，无未提交差异需附带。执行代码 commit 在训练前另冻结（protocol_resolved.json）。
