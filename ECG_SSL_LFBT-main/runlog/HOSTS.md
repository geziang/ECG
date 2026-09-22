# 多主机并行实验协调表(HOSTS)

> **目的**:多台主机依托本仓库并行执行 M 筛选矩阵与判决实验。**本文档是唯一任务认领台账**:任何主机在启动任何任务前,必须 `git pull --rebase` 并查本文档,凡状态为 🏃(执行中)或 📋(已认领排队)的任务**一律不得重复启动**。
>
> **认领方式**:把对应任务行的状态改为 `🏃<主机名>(开始时间)` 并 commit+push;完成后改为 ✅,并把结果行写入本主机结果文件(§五-3)。
> **更新纪律**:任务状态每次变化(领取/完成/失败)立即 push;长任务不必频繁汇报,但必须写明预计完成时间。
> 文档路径:`runlog/HOSTS.md`;最后更新:**2026-09-22 20:05,W3 第二批任务已下发**。
>
> **主机A当前状态:🟢 空闲（旧任务已全部完成）**——W3 新批次 A-2 已下发，等待主机 A `git pull --ff-only origin main` 后回写认领；旧 §二/§三 队列仅作历史记录，不得启动。
>
> **本轮再分配三原则(2026-09-17 晚,用户指令)**:①台账只按实际情况分配任务,**实现细节(开关/脚本/单测)由各主机自理**;②**总目标 = 尽快拿到至少一个过 3-seed 门的涨点、出成果**,判决线优先于扫描线;③已判死/无判值任务一律清除,机时不许再花。
>
> **🛠 工具提示(09-18 新增)**:各机 ZCode 的科研 MCP(arxiv / semanticscholar / ssh)部署指南见 **`runlog/MCP部署指南.md`**——含办公机实装验证的配置 JSON 与踩坑记录,工作机 `git pull` 后照装;ssh 服务器装好后即可跨机远程查卡/看日志,替代人肉中转。

---

## 一、主机登记表

| 主机 | 硬件 | 关键环境 | 当前状态 |
|---|---|---|---|
| **主机A = `Win4090`**(F:\新实验) | RTX 4090 24G / 32 逻辑核 / 128G RAM,双车道 | Win10,Python 3.10.11,torch 2.0.0+cu118,conda env `DL` | 🟢 空闲；W3 A-2 已下发，待认领 |
| **主机B = `DESKTOP-0PBLCND`**(E:\GZA) | RTX 3080 10G / 16 逻辑核 / 64G RAM,**单车道** | Win10,Python 3.11.9,torch 2.5.1+cu121;数据已校验(含 5 个损坏 .mat 修复);推送走 `ECG-push-tmp` | 🟢 空闲；W3 B-0→B-1→B-2 已下发，待认领 |
| 主机C | (待登记) | (待登记) | 🆓 | 

## 一-A、W2 冻结后的新任务总览（2026-09-22，优先级高于下方历史队列）

> W2 已完成并冻结于 `2c31a2b`；旧的 §二/§三 队列均视为历史记录，**不得据此重新启动训练**。新任务唯一入口是根目录《下一阶段双机任务书-2026-09-22.md》，结果统一写入 `runlog/W3/`。

| 任务 | 主机 | 当前状态 | 交付物 | 停止线 |
|---|---|---|---|---|
| A-0 冻结哈希与协议复核 | A / Win4090 | ✅ 已完成（93f0379） | `runlog/W3/freeze_audit.json` | 11/11 SHA 一致；账本 28 行且无重复 |
| A-1 TRC 插入位置最小消融 | A / Win4090 | ⏸ 待教授确认 | `runlog/W3/trc_insertion/` | 仅 smoke→seed0；不升 seed2/4 |
| A-2 冻结模型退化评估 | A / Win4090 | 📋 已下发（待 A 回写认领） | `runlog/W3/robustness_*.csv` | 先 smoke；不重训，只复用冻结 checkpoint |
| A-3 FT20/FT40 标签效率 | A / Win4090 | 💤 可选 | `runlog/W3/label_efficiency.csv` | 需保持既定协议，否则延期 |
| B-0 冻结材料审计 | B / DESKTOP-0PBLCND | 📋 已下发（待 B 回写认领） | `runlog/W3/freeze_audit_hostB.json`、协议说明 | 不改 W2 文件；SHA/行数不符即停 |
| B-1 统计与图表复算代码 | B / DESKTOP-0PBLCND | 📋 已下发（B-0 后续） | `runlog/W3/paper_materials/` | 只做 seed-level 描述性统计 |
| B-2 指标/预测保存扩展 | B / DESKTOP-0PBLCND | 📋 已下发（B-0 后续） | `runlog/W3/metrics_schema/` | 先 smoke 与单测，不自动重评 test |
| B-3 NFH 97 条剔除明细 | B / DESKTOP-0PBLCND | ⏳ 待资料 | `runlog/W3/nfh_exclusion_reconciliation.csv` | 无原始明细不得估算 |

认领纪律：先 `git pull --ff-only origin main`，再把本表对应行改为 `🏃<主机>(时间)` 并 push；完成后改 `✅`，附结果路径、代码 SHA、数据/ checkpoint SHA。任何未登记的 GPU 训练均视为禁止。

### W3 第二批重新分配（2026-09-22 20:05）

两台机器旧任务均已完成并空闲；办公机无法直接 SSH，本批次通过本台账下发。A/B 开工前必须执行 `git pull --ff-only origin main`，确认基线为 `93f0379`，再把对应行从“📋已下发”改为“🏃主机名(时间)”并 push。

- **A / Win4090 — A-2 主任务**：只读取 W2 冻结的 B0/C1/C2 encoder 和同一 test split，先做 clean smoke，再做 adjacent lead-swap、连续导联质量衰减、baseline wander、肌电、50/60 Hz 工频（SNR `0/5/10/20 dB`）。固定 clean 线性头，不重训 encoder/分类器，不选择 test 方向；输出 `runlog/W3/robustness_*.csv`，记录模型、seed、扰动、强度、AUROC/AUPRC、clean 相对掉幅及代码/数据/checkpoint SHA。任何 checkpoint/manifest/架构不匹配或 clean smoke 与冻结读数不一致，立即停并写失败记录。
- **A / Win4090 — A-1**：继续暂停，只有教授明确要求才运行旁路 `off/pre_gap/post_gap` smoke；不得启动长训，post-GAP 不得继承 C2 权重冒充公平结果。
- **B / DESKTOP-0PBLCND — B-0→B-1→B-2**：B-0 先生成 `freeze_audit_hostB.json`、`protocol_audit.md`、`failed_directions_index.csv`；B-0 通过后 B-1 复算主表、paired delta、mean±SD、seed 符号表、缺导汇总和失败索引到 `runlog/W3/paper_materials/`，不得覆盖 W2；随后 B-2 只做指标 schema/逐记录预测保存的 smoke 与单测，输出 per-class AP、Macro-F1、Sensitivity/Specificity、混淆矩阵、ECE/Brier 和 SHA 元数据接口，禁止自动重评 test。
- **B-3**：NFH 97 条剔除明细仍待原始记录；没有记录名和排除阶段时只写“待交付”，不得估算。

本批次禁止恢复 D1L、ACL、LGA、Attention、Mixer、JEPA、VICReg、EMA、Sinc、MixUp、HRV、CCM、导联相关矩阵新损失、12 导联恢复、多模态睡眠或任何新预训练扫描。所有新文件只写 `runlog/W3/`，W2 文件只读。

新主机入场四步:①按 §五-6 准备并校验数据;②**先跑本机 B0 锚点**(§五-1 红线);③在上表登记硬件与环境;④按 §四 开展创新方法并在登记表挂号。

## 二、历史队列（已封存，不得启动）

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

> **A-P2 状态(主机A,20:55)**:① 开关全部实现并**8 项单测全过**(`--blur-pool/--pool-power/--whiten/--whiten-ln/--whiten-shuffle`;关态与 git HEAD 参照逐位一致、白化协方差=I、NEG 可逆),h2/t3/h1(预注册 3 seeds)+ LN-NEG + 位置-NEG 共 5 条已入 `pipeline_queue.yaml`;② **秩诊断初版**:旧缓存特征 erank/d=0.29~0.32(逐导联)/0.110(全局)< 0.5 → 判塌缩,**但该缓存标签 22 类、来源存疑**,正在以 SHA256 校验过的 B0 checkpoint 重抽特征复核(`feat/h5_b0_regen`),**复核不塌缩则 h1 三条立即出队**;③ H5 校准脚本就绪(`calibrate_lp.py`:train 上 StandardScaler+Logistic C 网格,val 选 C、test 仅评一次),待重抽特征后出数;④ TFS 基线秩诊断待补。

### A-P3 队尾(有空位再跑)

**自有机内部分工**:E007/LGA 5-seed 扩展与 η=0.01 复核固定在实验机(E 盘,已有完整 E001 工程)执行,同样不属于其他主机的任务。

### W1 状态(主机A,2026-09-20 晚)

- **筛选矩阵收官**: t3_pow3 3-seed 全负关线(−0.40/−0.10/−0.27pt);A-P2 三线(h2/t3/h1)全部关闭,矩阵 36 数据行,仅剩 b0fast 在跑(其 sinc_M 秒败 bug 已修: fast 路径 __init__ 提前 return 漏初始化);
- **双 bug 已修并提交**: ①t3 LP 崩溃真根因=run_lp 用 children()[:-1] 绕过 forward 取特征,T3 时 model 末尾无池化 -> PowerPool1d 挂入 model 末尾修复(键集不变,旧 checkpoint 直接加载);②b0fast sinc_M;
- **W1 任务书已开工(报告归档/历史文档/07-下周科研与代码任务书-2026-09-19.md)**: T1 D1L-fix 完成(τ 入对角/PSD/闭合基线逐位一致,分支 w1-code-tasks);T2 ACL 四区完成(Eq10/11 单测全过,同分支);T4 NFH 预处理完成(34,905/34,905,零排除,manifest 在 runlog/W1/);
- **C1 NFH B0' 手动链在跑**(worktree 运行隔离,100ep matched-updates≈27.3k 步,PT 完自动接 PTB-XL LP 并落 csv);A1/A2/A3 已入 pipeline_queue.yaml,**须待 b0fast 结束、w1-code-tasks 合并 main 后方可由车道执行**(当前 main 的 --d1l 仍是旧错误语义);
- 主机A红线上新增: ECG_SSL_LFBT-wt1 为 worktree 目录(C1 链运行中,勿动);C1 运行不阻塞 main 树合并,main 树 run_pt 存活时仍禁止合并。

## 四、其他主机的分工:复现基线锚点 + 开创新方法

**其他主机不参与 §二/§三 的任何任务**,要做的是:

1. **入场即跑本机 B0 锚点**:按 §五-2 冻结协议预训练+LP(约 2h),结果写 `runlog/M/matrix_results_<主机名>.csv`——这是该主机后续一切 Δ 的分母,没有它你的数字无法解释;
2. **开创新方法**:新模块 / 新损失 / 新增强 / 新预训练目标……方向不限,按 §五 红线执行与报告;建议每个新方法先 seed0 粗探,Δ>0 再升 3-seed 确认(可复用 `m_screen.py --confirm` 逻辑);
3. **在下表登记,避免多主机撞题**(领取时把状态改成 🏃 并 push):

### 新方法登记表

| 方法名 | 主机 | 一句话设计 | 状态 | 结果文件 |
|---|---|---|---|---|
| (示例)xxx 注意力变体 | 主机B | 投影头上加轻量通道注意力 | 🆓 | matrix_results_hostB.csv |
| H3 患者身份不变性(轻量去相关) | 主机B | 患者均值离散度压零(--h3-weight) | ✅ 判负 s0 Δ-0.1498(0.3权重破坏性强;复核可试0.03) | matrix_results_hostB.csv |
| C1 Sinc 带通前端 | 主机B | 第一层换参数化带通组(--sinc-frontend 16) | ✅ 判负 s0 Δ-0.0271(bands.json已存) | matrix_results_hostB.csv |
| C3 准周期 MixUp | 主机B | FFT相位对齐后小比例混合(--mixup-prob/--mixup-align) | ✅ 判负 s0: 对齐Δ-0.0169 / 普通Δ-0.0144 双负(配对结论: 对齐无增益, MixUp轴含对齐一并关闭) | matrix_results_hostB.csv |
| H4 HRV 借口 | 主机B | gqrs R峰->HRV四统计回归辅助头(--hrv-weight) | ✅ 判负 s0 Δ-0.0026(0.1权重下HRV辅助头无增益; 期间R5自愈1次=hrv_head按mask索引, a78e7de) | matrix_results_hostB.csv |
| A1 D1L-fix 翻案(W1 T1) | 主机B | τ 移到 cross-corr 对角目标, off-diag 恒0; PSD 校验+target_matrix.npy 工件 | ✅ 判负 s0 Δ-0.0758(τ对角修正版破坏性强于旧bug; NEG打乱-0.0689≈真实→先验无效, 结构化目标线关闭) | matrix_results_hostB.csv |
| A4 ACL 随机分组 NEG(W1 T2) | 主机B | T2 ACL 完整实现(四区+独立projector+Eq10/11), A4=随机分区×3(101/102/103) | ✅ 判负三分区全负: p101 -0.0072/p102 -0.0049/p103 -0.0069(符号一致; ACL加性随机分区无增益, 待与主机A A3解剖分区跨机符号对照) | matrix_results_hostB.csv |
| B1/B2/B3 重建线(W1 T3) | 主机B | 旧D7归档 / 多段20%两视图 / CCM周期内20%不跨R峰(gqrs缓存) | ✅ B3判负-0.0091 / B2噪声-0.0017(周期遮挡劣于多段遮挡, 重建支路中性; B1旧D7弃跑=旧路径未提速~23h, 归档价值低) | matrix_results_hostB.csv |

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

> 主机B(DESKTOP-0PBLCND)于 09-17 19:10 登记入场:环境详情见 `runlog/HOSTB_环境报告.md`;入场 B0 锚点运行中,完成后结果写 `runlog/M/matrix_results_hostB.csv`;未认领 §二/§三 任何任务,等待 §四 分配。(登记已 push,commit f957df3)

> 主机B 终报(09-18 07:30):§四 seed0 三项+n3 全部完成,Δ 全负,详见 runlog/M/matrix_results_hostB.csv 与 runlog/M/hostB_auto_report.md;机器空闲可接新任务。n3_prob03(拉近)Δ-0.0097 一并判负——患者轴三选一结论=不动(B0)最优。
> 主机B 第二批终报(09-18 19:28):C3配对(对齐-0.0169/普通-0.0144)与 H4(-0.0026)均判负,§四任务书清零;两批六探针Δ全负,均"不动最优"。matrix_results_hostB.csv 与 hostB_auto_report.md 已随本提交入库。机器空闲待分配。

> 主机B W1 启动(09-20 21:30):已 pull 85804dd 拿到《07-下周任务书》,按主机B分工实现 T1(D1L-fix: τ 对角目标/PSD/工件)、T2(models/acl_region.py 四区+独立projector+Eq.10/11, A4 随机分区×3)、T3(data_utils/ccm.py 多段+CCM 两视图独立 mask+masked MSE+梯度比诊断, R峰 gqrs 离线缓存 prep_rpeaks.py)、T0(config 增 git_sha/host/dataset_hash/preprocess_version);tests/test_w1.py **14/14 全过**(含默认关==B0 逐位一致、τ=1 闭合==B0、CCM 不跨 R 峰)。队列 pipeline_queue_hostB.yaml: a1_d1lfix→b3_ccm→a4_rand_p101→b2_multiseg→a1_neg→b1_d7_arch→a4_p102→a4_p103(单车道,每 run ~3.2h,预计 24h 完成前 7 项)。**NFH/Chapman 数据已在本机 E:\GZA\ECG_data**(ningbo 34,905 / chapman 10,247),runlog/prepare_nfh.py 审计+转换脚本已备好(纯 CPU 不占车道),主机A 可直接取用省掉下载与预处理;CPSC2018(icbeb) 本机缺失,待下载补周五跨库评估。

> 主机B W1 终报(09-21 21:50):队列 7 跑 1 弃**全部判负**,无一达 3-seed 门(≥+0.005)。终表(vs 锚点 0.7158):
> a1_d1lfix -0.0758❌ | a1_d1lfix_neg -0.0689❌ | b3_ccm -0.0091❌ | a4_rand p101/p102/p103 = -0.0072❌/-0.0049≈/-0.0069❌(三分区符号一致) | b2_multiseg -0.0017≈噪声; b1_d7 弃跑(旧路径未提速 ~23h, 归档审计价值低, 工程决策让位)。
> **机制结论**:① D1L 翻案实验证伪——修正版(τ对角)比旧bug(-0.006)伤害更大, 且打乱先验(-0.0689)≈真实先验(-0.0758)→伤害源于"多数导联对被强制去相关至0"的机制本身, 非先验对错, 结构化跨导目标线关闭; ② B线——多段遮挡(-0.0017)≈中性而周期遮挡(-0.0091)更差→"周期相位保真"假设帮倒忙, 重建支路本身无增益(η=0.1); ③ ACL加性随机分区三分区一致微负→与LGA负结果同向, 待主机A A3(解剖分区)跨机符号对照, 若同为负则ACL加性形式在本基座关闭。"不动最优(B0)"结论进一步加固(累计13个负探针)。
> 工程事件落账:p102首次尝试40min瞬态崩溃(WDDM显存竞争, b1击杀过渡期), runner自动重试成功; b1弃跑见上。csv已随本提交入库; 代码见 7f421ab 及此前5个提交。(主机B, 09-21 21:50)

> 主机A W3 认领(09-22 19:45):已 pull 至 cbe70f3(含 2c31a2b 基线)拿到《下一阶段双机任务书》。**A-0 冻结复核** 🏃主机A(09-22 19:45);A-1 待教授确认不启动(分支 codex/trc-insertion-ablation@aec79b9 已核验存在);A-2/A-3 第三阶段按需。W2 结果文件只读。
> 主机A A-0 完成 ✅(09-22 19:50,登记补录 19:58):runlog/W3/freeze_audit.json——11/11 checkpoint SHA 独立重算一致;账本 28 行 0 重复键,git_sha 统一 fb08bc5;seed 符号 CPSC+++/FT10+++;三 manifest 在册;limitation 已注明(无逐记录概率,不得声称 patient-level bootstrap)。基线 cbe70f3。A-1 仍待教授确认。(注:✅行曾因 shell cwd 残留误写入嵌套空目录 ECG_SSL_LFBT-main/ECG_SSL_LFBT-main/,已清理并补录至此;freeze_audit.json 本体路径无误)
