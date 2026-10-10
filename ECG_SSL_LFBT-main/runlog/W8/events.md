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

## 10-10 18:3x E2 预注册与启动

- 实现=零新仓库代码: E2 = W4 S1 全套(`--loss-mode simclr` τ=0.5 默认) + 唯一差异 `--nstdb-aug 0.5`(W7 E5 注入器)。开关纪律单测 `tests/test_w8_e2.py` 4/4: prob=0 逐位恒等 / prob=1 注入生效 / 注入器无状态(两视图独立结构保证) / run_pt 源码守卫断言(nstdb 默认 0.0 + >0 守卫 + loss-mode 默认 bt + τ 默认 0.5)。"默认关==S1 逐位一致"的结构依据=守卫块默认不执行, S1 checkpoint 由同代码路径产出(W4), 无交互项。
- 链: seeds{0,2,4} × PT(NFH 100ep) → LP 三域 → FT10{ptbxl,cpsc}; 主链后补 S1 cpsc FT10 ×3(W4 冻结 simclr checkpoint 纯补格评测, 行 ckpt=simclr-w4frozen, 供 E2 vs S1 在 cpsc FT10 配对——S1 在 W4 只评过 ptbxl FT10)。
- 判定(任务书): 无晋级门如实入表; 配对口径=vs S1(隔离"ECG 增强"单因素, LP 三域+cpsc FT10)+vs C1; 与 C1/Noise(=W7 E5 c1na)/S1 构成 {BT,SimCLR}×{RRC-TO,+NSTDB} 2×2 四格收齐。
- 车道: GPU 车道2(与 E1 车道1 并行, 显存预算 PT~6.7G+LP~2.5G << 24G)。
