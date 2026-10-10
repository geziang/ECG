# W8 事件账本

> 纪律: 判定/sanity 先写后跑; 冻结账本只读; 措辞 record-level。

## 10-10 18:1x W8 开工与 E1 预注册

- 挂号: HOSTS §一-E 认领 E1–E7(71eceda), 顺序 E1→E2→E4→E6→E3→E5→E7。
- 仓库代码变更(仅 1 处, E1 前置): `metrics_ext.py` 预测落盘白名单 +W8(W3–W7 不变, W2 硬拒不变); `tests/test_metrics_ext.py` +1 镜像用例, 22/22 绿。默认关闭行为不变(不传 --save-predictions 不触守卫)。
- E1 实现路径=零新代码: `run_lp.py --checkpoint none`(随机初始化冻结特征+线性头, W4 S3 同款 TFS 路径) + `run_ft.py` 省略 --checkpoint(随机初始化全参微调)。驱动脚本 `runlog/W8/run_e1_rand.py`。

**E1 预注册判定(先写后跑, 任务书 sanity 红线的操作化)**:
1. 无晋级门, 18 跑(LP 9 + FT10 9)如实入 `w8_rand_results.csv`。
2. sanity 红线(任务书): LP(rand) 三域均应显著低于 C1。操作化: 任一域 rand 3-seed mean AUROC 距该域 C1 3-seed mean < 2.0pt → 判"接近", 停下上报办公机, 该域数字冻结待裁决, 不擅自继续用于论文表述。C1 参照: ptbxl=W2/W4 账本, cpsc=W5A 新账本(0.9463), chapman=W4 账本。
3. FT10(rand)=从头训练参照, 无 sanity 门, 如实入表。
4. 统计(逐记录配对 bootstrap 10k + DeLong vs C1)由后续 stats 步骤统一算, E1 出数不等待统计。
