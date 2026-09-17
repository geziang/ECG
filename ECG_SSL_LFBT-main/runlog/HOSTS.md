# 多主机并行实验协调表(HOSTS)

> **目的**:多台主机依托本仓库并行执行 M 筛选矩阵与判决实验。**本文档是唯一任务认领台账**:任何主机在启动任何任务前,必须 `git pull --rebase` 并查本文档,凡状态为 🏃(执行中)或 📋(已认领排队)的任务**一律不得重复启动**。
>
> **认领方式**:把对应任务行的状态改为 `🏃<主机名>(开始时间)` 并 commit+push;完成后改为 ✅,并把结果行写入本主机结果文件(§五-3)。
> **更新纪律**:任务状态每次变化(领取/完成/失败)立即 push;长任务不必频繁汇报,但必须写明预计完成时间。
> 文档路径:`runlog/HOSTS.md`;最后更新:**2026-09-17 17:15,主机A(Win4090)**。
>
> **分工总原则(2026-09-17 定)**:存量筛选/判决任务(§二、§三)**全部由主机A认领**,其他主机不得启动;**其他主机的任务是复现基线锚点 + 开创新方法**(§四)。

---

## 一、主机登记表

| 主机 | 硬件 | 关键环境 | 当前状态 |
|---|---|---|---|
| **主机A = `Win4090`**(本条目维护机,F:\新实验) | RTX 4090 24G / 32 逻辑核 / 128G RAM | Win10,Python 3.10.11,torch 2.0.0+cu118,conda env `DL` | 🏃 双车道运行中(§二) |
| 主机B | (待登记) | (待登记) | 🆓 |
| 主机C | (待登记) | (待登记) | 🆓 |

新主机入场四步:①按 §五-6 准备并校验数据;②**先跑本机 B0 锚点**(§五-1 红线);③在上表登记硬件与环境;④按 §四 开展创新方法并在登记表挂号。

## 二、主机A:正在执行(⛔ 其他主机禁止启动)

| 任务 | 类型 | 启动 | 预计完成 | 接力来源 |
|---|---|---|---|---|
| `aug_mask_weak` seed0(`--aug-params 0.5,1.0,0.0,0.3`) | 预训练+LP | 09-17 15:58 | ~09-17 19:00 | 车道A(m_screen batch3c) |
| `aug_crop_085` seed0(`--aug-params 0.85,1.0,0.0,0.5`) | 预训练+LP | 09-17 16:00 | ~09-17 19:00 | 车道B(chainB2) |

## 三、主机A:已认领排队(⛔ 其他主机禁止领取;由本机脚本自动接力)

**车道A(m_screen batch3c 余量,自动执行):**
1. `proj_dim1024` seed0(`--projector 1024-1024-1024`)
2. `proj_dim4096` seed0(`--projector 4096-4096-4096`)

**车道B(chainB2 → chainB3,自动执行):**
3. FT10 评估 ×2(B0 锚点 `checkpoint/ptxl_gamma08`、crop_weak seed0)——轻任务,标注效率轴首批数据
4. `d7_rec01` seed0(`--d7-weight 0.1`,MAE 重建支路)
5. `d7_rec003` seed0(`--d7-weight 0.03`)
6. `n3_prob03` seed0(`--n3-prob 0.3`,患者正对)
7. `common_view01` seed0(`--common-weight 0.1`,共模视图)
8. `D1L×AR-B3` 预训练+LP(先验×鲁棒基座判决实验)

**条件追加(自动)**:以上任何 seed0 探针 Δ>0,主机A将自动补跑该探针 seed2/4 确认轮(3-seed 符号一致门)。

**队尾(⛔ 同样全部由主机A认领;`pipeline_runner.py` 待命,车道槽位空闲后自动接力,当前维持双车道让显存):**

9. `b0fast` 校准(`--fast-backbone`,fast 探针路线锚点)
10. `speed_perturb` seed0(`--speed-perturb 0.85,1.15`)
11. `asym_view` seed0(`--view2-params 0.9,1.0,0.0,0.1`)
12. `lead_swap02` seed0(`--lead-swap-prob 0.2`)
13. `cautious_adam` seed0(`--cautious`)
14. `arb3_base` seed0(AR-B3 基座,`run_pt_ar --variant B0 --fusion mean --fusion-bt-weight 0.2 --lead-mask-prob 0.5`)
15. `psfull_arb3` seed0(**判决实验**,`run_e006_physiospatial --profile physiospatial --base-variant ar_b3 --ablation full`)
16. CPSC 跨库评价(评估轴④,用既有 checkpoint,需先备 CPSC 数据)

**自有机内部分工**:E007/LGA 5-seed 扩展与 η=0.01 复核固定在实验机(E 盘,已有完整 E001 工程)执行,同样不属于其他主机的任务。

## 四、其他主机的分工:复现基线锚点 + 开创新方法

**其他主机不参与 §二/§三 的任何任务**,要做的是:

1. **入场即跑本机 B0 锚点**:按 §五-2 冻结协议预训练+LP(约 2h),结果写 `runlog/M/matrix_results_<主机名>.csv`——这是该主机后续一切 Δ 的分母,没有它你的数字无法解释;
2. **开创新方法**:新模块 / 新损失 / 新增强 / 新预训练目标……方向不限,按 §五 红线执行与报告;建议每个新方法先 seed0 粗探,Δ>0 再升 3-seed 确认(可复用 `m_screen.py --confirm` 逻辑);
3. **在下表登记,避免多主机撞题**(领取时把状态改成 🏃 并 push):

### 新方法登记表

| 方法名 | 主机 | 一句话设计 | 状态 | 结果文件 |
|---|---|---|---|---|
| (示例)xxx 注意力变体 | 主机B | 投影头上加轻量通道注意力 | 🆓 | matrix_results_hostB.csv |

## 五、多主机方法学红线(必读,违反=数据作废)

1. **禁止跨机拼表**:同一模型的绝对 AUPRC 在不同 GPU 上可差 0.8~1.0pt(历史实测)。每台新主机**必须先跑本机 B0 锚点**(预训练+LP,约 2h),本机全部 Δ 一律对本机锚点计算;跨机只比较方向与符号,最终判定实验回单机重跑确认。
2. **协议冻结**:预训练 200ep / batch 128 / Adam 1e-3 / RRC-TO(0.5,1.0,0.0,0.5)/ PTB-XL folds 1–8(17,418 条);下游 LP(冻结+线性,1e-3)与 FT(全量,1e-4)各 100ep,val 选 checkpoint、锁定 test 仅评一次。改动任何一项即不可比。
3. **结果文件分主机**:各写 `runlog/M/matrix_results_<主机名>.csv`(schema 同 matrix_results.csv),**不要共写** `matrix_results.csv`(避免 git 冲突);主机A的历史行已在 matrix_results.csv。汇总合并由主机A定期执行。
4. **checkpoint 不入 git**(体积过大);结果行的 args 列已含完整复现命令。
5. **显存安全**:`run_pt/run_lp/run_ft` 已内置启动自闸门(6500/2500MiB,CUDA 初始化前排队),多车道并发不会 OOM;各主机可放心开 2~3 条车道(单预训练实测峰值 6.74GB)。
6. **数据准备**:`preprocess_ptbxl.py` + `prepare_data.py` 生成 `data/pt_pretrain`(17,418)与 `data/ptbxl`(train/val/test = 13,639/1,714/1,739);以 `data/manifest.json` 与 B0 checkpoint SHA256(`026033d5…`)核对一致性。

## 六、已完成任务速览(✅ 勿重复)

截至 09-17 16:00,`matrix_results.csv` 共 13 行判定:**d1l_05_02 ❌ / d1l_neg(负对照)/ n4_ema999 ❌ / aug_crop_strong(0.4)❌ / b0_gamma07 ❌ / b0_gamma09 ❌ / b0_lambd003 ❌ / b0_lambd01 ❌ / aug_crop_weak(s0 +0.27 → s2 ❌ → s4 ❌,3-seed 门判负,裁剪轴关闭)/ aug_crop_080 ❌ / aug_crop_090 ❌**。另:N1 四库语料 ❌、D9 VICReg 六形态 ❌(详见根目录实验报告)。

---

*维护责任:主机A。其他主机修改本表后请在行末注明自己的主机名与时间。*
