# JOINT_20261013 事件台账

> 批次=TRC–Beat-RR–Noise 联合验证(protocol.md);执行台账,只追加。判定/协议原文见 protocol.md,执行细则见 任务书.md。

## 2026-10-13 19:1x 批次开工前盘点与两项裁决(主机A)

- git 同步得 protocol.md(5072574);锚点映射完成(任务书 §一):M0=confirm/c1、M1=confirm/c2、M2=w6_dv/seed0+w7_dv/seed{2,4}(rpeak_nfh ✓×3)、M5=w8_noiseiso(时间隔离版,与 protocol §2 口径一致)。
- **preflight 全过**(preflight.md):rpeak_nfh↔NFH 一一对应 34,905;有效 R 峰 99.98%(7 条空);噪声两视图独立/偏移域/RR z-score 来源/CPSC 1385 均核验。
- **裁决一(A1/A2 s0 口径污染, 任务书 §二)**:W8 E5 dvo/rro s0 config 的 rpeak_npz=pt_rpeaks.npz(17,418 条=旧 PTB-XL 语料;E5 链模板漏传 --rpeak-npz)——NFH 前 17,418 条拿错语料 R 峰、后 17,487 条无键。A1/A2 改三种子全部重跑(显式 rpeak_nfh),最低新增 PT 10→12 次;w8_dvsplit_results.csv 冻结只读,本条即更正记录,W8 报告引用此条。
- **裁决二(排程)**:smoke→M3→M4→A1→A2→M6(含)串行;N1/N2 暂缓待 M3 落地。预算 29-31h(不含 M6)~38-40h(含)。

## 2026-10-13 18:2x-18:5x S0 smoke 全过 + 正式链启动

- **smoke 四项验收全过**(任务书 §五):①5ep 全开关(trc1+dv0.01+rr0.1+nstdb0.5/first_half)exit0,每 100 步 JSON 含 loss_dv/loss_rr 且有限(loss_dv 1128→246↓, loss_rr 1.20→0.60↓),[nstdb-aug] banner 出现;②续训 --epochs 6: `[resume] 恢复自 epoch 4, 从 5 续训`,损失连续(ep4 均值 557 → ep5 均值 520);③LP --trc 1 加载联合 checkpoint exit0 出 metrics(6ep 垃圾编码仅验机制);④config.json 全开关落盘(rpeak_npz=pt_rpeaks_nfh.npz)。
- checkpoint 内容核验:train_state 含 model/optimizer/epoch/torch_rng;rr_head 完整模块入 train_state;TRC 权重=encoder `model.5.gamma/beta`(trc=0 加载报 unexpected——LP 必须 --trc 1,链已对齐)。
- 实测定标:**59s/ep(全开关)** → 100ep≈98min/seed;预算更新:最小配置(12 PT)≈26h PT + 评测 ≈ 30h;含 M6 ≈ 33h。
- **18:48 正式链启动**(launch_joint.ps1 detached, pid 30880):m3 s0 PT 开跑,命令与 protocol §4.3 逐字一致;值班 cron automation-9fc9eceb 每小时 :23 看护。
