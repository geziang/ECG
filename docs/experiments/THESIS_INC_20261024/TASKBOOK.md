# 主机A正式任务书：10月24日至26日中午心电自监督增量实验

任务ID：`thesis-inc-20261024`；协议：`thesis-inc-20261024-v1`；版本v1。**2026-10-24经用户授权正式安排，训练授权仅在HOSTS登记并由主机A完成预检后生效。** 新代码尚未实现，不以任务书存在冒充已启动。总交付截止：**2026-10-26 12:00 Asia/Shanghai（北京时间）**。

同目录 `protocol.json`、`task_matrix.csv`、`source_lock.json` 为机器可读执行范围；`validate_dispatch.py` 可在办公机/主机A用标准Python检验包内一致性，不启动训练。发生不一致先停止相关行，报告并修复实现，不静默改科学定义。

## 1. 本批目标、授权与边界

按用户要求沿LFBT心电自监督路线小步探索，允许已有组件迁移、有限组合、局部调参与负项的有限互补试验；不要求每一行先证明新颖性。真正的无标签预训练是主线，公开模型/模块验证为旁线。已有P0保持冻结，不能把小网络分类或LP头训练再次当新SSL预训练。

允许A机自主实现、预检、运行本书范围内的阶段，按冻结规则生成下一轮配置、补种子、有限修复、checkpoint恢复、汇总及commit+push；无需用户逐行判决。不得扩展到本书外的新主干/新损失家族/新数据下载、FT网格或旧关闭队列。办公机和主机A只经origin/main交接，不使用SSH。

结果独立写 `ECG_SSL_LFBT-main/runlog/THESIS_INC_20261024/`，代码建议 `tools/thesis_inc/`，权重/缓存在A机另建独立目录并记录可恢复备份。旧W0–W9/JOINT/ABC/P0的结果、权重和协议只读。必要时导出冻结代码工作副本，训练期间不在运行目录pull/rebase/checkout。

本批全部下游开发与参考验证限train/val，禁止读取test信号/标签和计算test指标；已有test数量/ID元数据审计可用。作者测试数值复现不纳入本批完成声明。外部最终测试和少标签微调不在本批自动扩展范围。

## 2. 时间、优先级与总预算

当前下发日为2026-10-24，办公机首次核时11:56。距截止约48h，实际可用时间以A机认领时为准；不以历史机器路径/旧空闲状态推断可用环境。

| 时间窗 | 优先工作 | 当窗产物；不是保证全部跑完的耗时承诺 |
|---|---|---|
| 10-24认领后至夜间 | A0交接/源与数据预检、实现与smoke；R1六组20epoch短训+LP | 真实计时、六行结果或明确partial、可续训状态 |
| 10-25 | R2组合、R3单权重三档、R4种子确认；穿插有时限的参考验证 | 每轮配置与决策表，完整种子和阴性记录 |
| 10-26上午 | 满足条件且预计能完成时做R5完整训练；核验/报告 | 正向/阴性/未完成、候选与底座对照、恢复材料 |

硬上限：**累计GPU进程墙钟36h，CPU计算12h，工程排障12h，主训练作业最多50，最多1个GPU进程**。时间可交错，不代表48h内一定完成所有上限。参考验证GPU子预算4h，排障每支线最多1h；主SSL失败不能靠无限重试耗尽窗口。

截止规则：10-26 **10:00后不新开预训练**，**11:00后不新开GPU作业**，**11:30前保存并结束本批GPU进程**，12:00前报告和push。启动前按最近完整epoch/同类行耗时×1.25估计，不能在剩余窗口内完成的行标queued_not_started；运行中每batch检查截止/预算。保存过程预留至少10分钟。只管理本批已登记PID，不停止他人进程。

预算用尽即保存并报告；未跑完的20/100epoch不能改成缩水“正式完成”。不得为了完成50行跳过训练、减样本或伪填DONE。

## 3. A0：认领、交接、保护与工程预检

1. 开工先fetch并读HOSTS顶部。无冲突后本任务行改为`🏃Win4090(实际时间)`并commit+push，再进入训练。若机器不符/另有任务占用，报告冲突，不重复启动。
2. 实测GPU、进程、驱动、Python/PyTorch/CUDA/wfdb等，检查磁盘与独立备份。复用P0的conda DL前先核验，不默认环境未变。
3. 核验source_lock中的实际Git内容哈希、P0 train/val数据SHA、NFH目录及峰缓存。NFH预计34905条；建立record ID、逐记录内容SHA与峰键对应清单。全库峰缓存里的无关键不得进入RR统计。发现数量/身份差异先定因，不能静默换语料。
4. P0交接只补真实权重位置/SHA/备份和已有日志；日志不存在如实标缺失，不重造历史。修复变长CSV记录器用于本新批次，不覆盖原P0账本。
5. 新入口仅允许本批解析后的绝对输出路径。test loader被调用即报错；本地LP的train/val隔离副本仍须对照P0数据锁。所有模型/样本顺序检查先于指标。
6. 每种新代码路径先1epoch smoke、有限损失/梯度、辅助开关、有效mask/边界/无峰、从完整训练状态连续恢复检查；关闭新增项的模型前向/主损失与该批I0一致。smoke不参与选线。
7. 首次正式训练前输出`protocol_resolved.json`（版本、源码commit与dirty diff、全参数、真实路径、种子、预算/截止和实际job清单），完成一个配置后不得静默更改它。新增组合另存job配置和决策来源。

## 4. 公共无标签预训练与下游评价

### 4.1 预训练身份

NFH `data/pt_pretrain_nfh` 与其匹配R峰缓存；原始8导联顺序II,III,V1–V6，float32、(8,2048)、fs204.8。保留原预处理，不二次归一化。全部从同种子的随机初始化开始，不能将已有100epoch模型当20epoch底座。

沿用VGG16(alpha0.125)逐导联主干/投影器、TRC=1，LFBT同/跨导联Barlow Twins，gamma0.8、lambda0.0051。Adam lr0.001、betas(0.9,0.999)、eps1e-8、weight_decay0，无scheduler，batch128、workers4、float32。**本新批次drop_last=True**，避免既有固定batch归一分母与末批不一致；此共同设置对所有新行一致，历史权重仅旁表。

整段两视图保留当前RRC-TO参数与NSTDB(prob0.5、SNR均匀5–20dB、bw/ma/em、first_half)的实际调用顺序；以源代码为准，解析全默认值入resolved。各配方同一seed使用相同epoch样本顺序、整段增强随机流与基础主干初始指纹；辅助视图使用独立RNG，不因多采心拍而改变整段视图。由(record_id,epoch,view_id,seed)固定局部随机流，记录实现版本。

R1–R4固定20epoch，不以PT loss早停；R5固定100epoch。按固定最后epoch导出编码器，不用CPSC val在预训练epoch间挑最优。保存每epoch的train loss分项与实际step，20epoch及最后epoch有完整恢复状态。

### 4.2 LP统一协议

复用P0已隔离CPSC train4809/val683、原始九类multihot及其顺序。仅用共同整段编码器提取512维特征，全部eval/冻结，训练Linear(512,9)，BCEWithLogitsLoss、Adam lr0.001、batch128、最多100epoch、patience15、最小元素加权val BCE选模，无调参/增强/类别权重/scheduler。pretrain_seed=head_seed；末批保留。

辅助心拍分支/RR头在下游弃用，不能通过拼接更多下游特征冒充共同编码器增益。保存val record_id/y_true/y_prob/类序与模型、数据、代码、配置SHA；独立复算macro-AP/AUROC差≤1e-8，核对冻结前后参数与buffer指纹。

## 5. 首轮R1：I0–I5（6次PT+6次LP，seed0）

| 配方 | 固定定义 | 权重 |
|---|---|---|
| I0 | 公共M6设计背景，无新增辅助目标 | 0 |
| I1 | 旧平均心拍模板，整段—模板辅助一致性沿用`_dv_consistency` | morphology=0.01，RR=0 |
| I2 | 与I1相同，仅把模板逐点平均改为中位数 | morphology=0.01，RR=0 |
| I3 | 同拍两局部视图的逐拍BT一致性，定义见下 | local=0.01，RR=0 |
| I4 | 旧RR三统计回归，沿用RR head，输入整段首导特征 | RR=0.1，morphology=0 |
| I5 | I3的时间均匀固定窗对照，匹配每条记录窗数/长度与容量 | local=0.01，RR=0 |

R峰仅来自增强前无标签NFH记录；沿用已核验NFH峰缓存，冻结来源/检测器和采样率，不能拿CPSC缓存顶替。峰数≥3有效；少于3只跳过相关辅助项，整段主目标仍训练且记录保留。

I1/I2对该条记录所有有效缓存峰取W205窗 `[p-102,p+103)`，越界edge padding；不逐拍重归一化，mean/median是唯一模板差别。RR使用所有有效峰，目标[mean_RR,SDNN,RMSSD]秒单位；mean/std只由本批NFH训练记录计算。

### I3/I5的具体局部目标

每条有效记录K=min(峰数,8)，超过8按有序峰索引等距确定性取8；I5使用相同K，但中心为floor((j+0.5)*2048/K)。W205及padding相同，padding槽不参与。检测无效记录I3/I5均不使用局部项。

同一组原始窗生成两份局部视图：各自只使用当前NoiseInjector(prob0.5、first_half、SNR5–20)，不做局部时间裁剪/平移/翻转；对应槽代表同一时间窗。两份均经与整段共享的该导联backbone+projector；无新主干/可学习聚合器。所有局部视图的RNG与整段分开，峰与K不随噪声重算。

每导联聚合有效(b,j)行，以每条记录相同总权重、记录内各窗均分：`w_bj=1/(N_valid*K_b)`。对两个投影矩阵分别按w求均值/方差，使用`sqrt(var+1e-4)`标准化；`C=Z1.T @ diag(w) @ Z2`；局部损失=`sum((diag(C)-1)^2)+0.0051*sum(offdiag(C)^2)`，八导联平均。N_valid<2的batch跳过局部项并计数。总目标=原LFBT+local_weight*局部项（组合时再加RR）。记录局部方差/相关矩阵诊断、辅助与主干梯度范数，确认局部目标不是未接入的零项。

这属于借鉴逐拍自监督思想的本地增量配方，不称完整复现Beat-SSL。I3/I5采用相同实现，只有窗中心定义不同；I1/I2是旧辅助损失的模板替换，损失尺度差别必须在报告说明。

## 6. 自主生成后续轮次（均受时间/预算限制）

### R2组合：最多3个新配方（3次PT+3次LP）

完成首轮后，用val macro-AUPRC相对I0增量>0保留信号，不要求≥0.005、3/3或显著。按降序增量排名，精确相同则按实测训练秒数较小、最后配置ID字典序打破平局。

从I1/I2/I3中选择正向者最多2个，各自叠加RR权重0.1，形态/局部保持0.01。第三个名额允许一次有形态—节律互补理由的“最佳尚未入组合的非正形态项+RR”，记录理由；不存在该项则不凑数。首轮全非正时仍可按同一规则给最佳形态项+RR一次组合机会，不进行无限组合。

I5如有正向作为固定窗有效参照保留，可在R3/R4参与后续母配方选择；不为了心拍故事丢弃它。所有组合保存相对I0及相对母配方增量；组合低于母配方时不宣称协同。

### R3单权重三档：通常最多2个新配方（2次PT+2次LP）

在I1–I5及组合中，选相对I0正向的第一名。若含形态/局部项，仅将该权重乘0.3和3（原1×复用），RR保持0.1；若只有RR，试0.03/0.3，原0.1复用。无正向则不扫权重。不同时改主干、lr和增强。

### R4种子确认：最多6次新PT+6次LP

从全部有效短训结果选正向前2个（不足则按实际数），加I0，全部补seed2/4；只存在一个正向则补4个PT，不存在则阶段结束。不得仅补正向种子；均值按预定种子全体计算，保留类别代价但不作原0.02硬否决。只有seed0不称三种子确认。

### R5时间允许的完整训练：最多6次PT+6次LP

三种子短训平均增量>0时，选均值第一名及I0，在相同完整100epoch预算下各做{0,2,4}。完整候选参数不再变；任务ID区分20与100epoch。若20epoch状态完整保存了模型/优化器/全部RNG/采样器且20epoch前路径完全一致，允许显式parent lineage从20继续至100，仍记作新的full作业；否则从头100，不以best权重伪装连续恢复。须确认完整配对预计能在截止前完成，不能用短训对比历史完整基线。

R1–R4最多17次PT+17次LP；R5最多6次PT+6次LP。完整阶段未进入或只完成部分如实报告，不自动进入FT/外部测试。

## 7. 参考验证旁线（最多4个主训练作业）

优先不挤占首轮SSL/必要LP；可在10-24完成首轮后或新SSL分支实现排障时穿插，GPU仍单进程。子预算4 GPU进程小时，超时保存partial继续主线。

- R-HL：固定作者commit/HF主权重，作者CPSC train/val CSV（4950/551）及标签、前5秒→补零→100Hz→原QRSTokenizer；100%标签、seed0、linear、batch256、AdamW lr0.005/wd0.05/warmup10/100epoch，作者train模式下冻结参数，按val macro-AUROC选模。不加载作者test；与P0本地适配分表，不声称已复现论文test数值。
- R-TK三行：固定作者源码、作者fold≤8/9的train/val、12导联100Hz10秒、train-fitted scaler；同Transformer d128/e3/h8/d_ff96；patch16/stride8、median60、heartlang256×96；position=1，lead-specific行lead=1、median=0；seed2025、fix_seed=1，RAdam lr0.001，最多300epoch，val AUC patience15。无TEST构造/读取。
- RTX4090下先预检三行microbatch32、gradient accumulation8（有效batch256）；如任一行预检OOM，**三行统一**micro16/accum16，正式开跑前冻结。检查模型无受microbatch影响的BN；若存在，记录该适配身份而不能声称严格数值等价。两档均失败则支线partial，不静默改输入或模型。此为已授权的显式资源适配，不混入SSL方法消融。
- 梯度累积按样本/类别元素加权，末组不足累积步不能按满256分母；每个optimizer step的LR调度与作者逻辑一致。运行时限截断不能称完整300epoch训练。
- 作者TRAIN入口的首次缓存构建也可能遍历全库信号；须先按已冻结train/val元数据生成隔离信号清单，再构建两份缓存，不能以“只调用TRAIN”代替test零访问。只读取test ID/fold元数据，不读取其信号或标签列。

## 8. 执行器、恢复与回传

用新runner维护GPU锁、manifest、事件、累计资源、截止和状态。状态至少planned/verified/running/trained/evaluated/recomputed/complete/partial/failed；只有配置、恢复权重引用、日志及预测/独立指标齐全才能complete。DONE复用须重新核验全部身份与文件。

完整断点保存模型、优化器、epoch/global step、Python/NumPy/Torch/CUDA RNG、sampler/数据序、worker/per-record增强状态、已耗资源与配置hash。工程修复每失败job最多一次，科学参数/样本不变；工程重试也计资源；启动前环境问题各支线累计不超工程上限。不得把反复启动失败称零耗时。

结果目录至少：`preflight.json`、`protocol_resolved.json`、`source_manifest.json`、`data_manifest.json`、`job_manifest.json`（含动态行与parent）、`results.csv`（固定schema）、`resource_usage.csv`（固定schema）、`events.md`、`decisions.jsonl`、`resume_manifest.json`、`jobs/<id>/`（配置/日志/val预测/权重SHA与备份引用）、`DAILY_REPORT_2026-10-24.md`、`DAILY_REPORT_2026-10-25.md`、`FINAL_REPORT.md`。

状态/每轮结果及时commit+push；运行目录固定快照与交接仓库分开，推送不替换正在训练的代码。截止/预算到达后HOSTS改✅（完成本批报告，无论阴性/partial均列明），不隐瞒未完成，也不续跑下一天旧队列。

## 9. 给主机A的正式交接文本

请认领`thesis-inc-20261024`并执行本任务书、protocol.json和来源锁。用户已授权10-24至10-26中午的心电自监督增量探索，允许本书内单项→组合→局部调权重→种子确认→条件性完整训练自主推进，不用等人工逐项判决。先预检、实现和验证新runner，首轮六组是真正NFH无标签预训练，统一CPSC train/val LP评价；正向门=固定主验证指标配对增量>0。主预算36 GPU进程小时，最多50个主训练作业（含参考旁线），单GPU串行；10-26 11:30结束GPU、12:00前交报告并停止。全程保护旧冻结材料，禁止test访问、超预算、静默缩水及旧队列恢复。工程细节由A机按已固定算法实现并冻结实际源码/参数；失败和阴性如实回传。
