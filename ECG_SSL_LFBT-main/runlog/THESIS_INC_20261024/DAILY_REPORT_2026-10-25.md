# thesis-inc-20261024 日报（2026-10-25）

本日无新训练。批次全部科学内容已于 10-24 夜间完成（主线 19:31 收束、REF 23:10 收束），本日按收口清单做最终核验并关闭批次；以下为核验结果与最终定案。

## 最终核验（10-25 上午）

1. **作业完整性**：矩阵 50 行中 17 行实际启动并完整完成（R1 12 + R2 2 + REF 3），1 行 partial_infeasible_on_gpu（R-TK patching，两档实测 OOM/单 epoch>120min），32 行 not_entered_by_gate（阴性门），0 行 queued/running。results.csv 与逐作业 DONE.json 一致。
2. **断言复算**：所有 LP/REF 作业 val 预测落盘独立复算差 ≤1e-8（结果字段 recompute_abs_diff_max）。
3. **冻结编码器核验**：每个 LP 作业特征提取前后 backbone 参数指纹一致（frozen_encoder=True）；R-HL 训练前后 backbone 哈希一致。
4. **test 访问审计**：results.csv 全文 0 处 test 引用；R-HL 数据目录无 test_* 文件、test_loader_called=0；R-TK 缓存 fold10 信号零读取（caches_manifest 记录）；主线 LP 入口 split 白名单守卫全程未触发拒绝以外行为。
5. **预算终值**：GPU 3.09h/36h；REF 2.74h/4h；CPU 0.63h/12h；工程 2.25h/12h；作业计数 18（含一次 I1 因 runner 重启的重复计数）；单 GPU 进程纪律全程成立。
6. **备份**：7 个主线 PT 作业 encoder+train_state 双份（F: checkpoint/thesis_inc + D:\thesis_inc_20261024_backup，5.5GB）；REF 产物在 runlog 与 D:。
7. **Git**：全部结果/代码/manifest/报告已提交推送 origin/main；HOSTS 已回写 ✅。

## 最终数字（同 FINAL_REPORT）

- 主线（val macro-AUPRC, seed0/20ep）：I0 0.7367 > I2 0.7363(−0.0003) > I1 0.7314 > I5 0.7293 > I4 0.7268 > I3 0.7235；R2 I2+RR 0.7360(−0.0007)——**全负有效阴性**。
- REF：R-HL 0.4218/0.7947（作者协议）；R-TK median_beat 0.8977、heartlang 0.8971；patching partial_infeasible_on_gpu。

## 遗留与边界

- patching 行在本卡的不可行实测（32×8 OOM、16×16 单 epoch>120min、显存 24G 饱和）可供后续算力规划；如需该行须更大显存卡或按协议另行授权的近似（本批不做）。
- 本批阴性为 20ep/seed0 口径；I2 与 I0 差距在种子噪声量级内，但按冻结门判负，不做事后翻案；后续如重启该方向须先声明鉴别设计。
- 10-26 12:00 截止前无待办；批次关闭。
