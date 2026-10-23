# 近期论文代码可用性与发表平台核查

核查日期：2026-10-23。返回 [阅读清单](READING_LIST_2026-10-23.md)、[选线评估](THESIS_DIRECTION_ASSESSMENT_2026-10-23.md)、[项目地图](../../PROJECT_MAP.md)。

用户已要求集中下载参考源码：现已建立 [本地参考代码目录](../../../参考代码/README.md)，每篇独立存放，实际下载情况见 [下载清单](../../../参考代码/下载清单.json)。该归档不表示已安装或运行复现；实施顺序见 [实验推进建议](../../../参考代码/实验推进建议.md)。

后续补充的公开论文 PDF、下载身份核验和来源哈希见[论文与代码总览](../../../参考代码/论文与代码总览.md)及[论文下载清单](../../../参考代码/论文下载清单.json)。原始源码核查记录保留；新增 PDF 不改变代码/权重缺口和未运行复现的状态。

本轮检查作者仓库的实际文件树、README、部分训练/方法源码及权重文件列表；7 个代码/资源仓库用只读用途的临时浅克隆固定了 commit。未安装项目依赖、执行外部代码、下载模型权重或启动训练。因此“较完整”表示主要源码入口可见，不表示已跑通或已复现论文数值。GitHub API 请求遇到限流，后续通过公开页面、原始文件和 Git 对象完成核查。

## 1. 可复现性判断

| 论文 | 实际公开内容 | 权重状态 | 推荐用途 |
|---|---|---|---|
| [HeartLang](https://github.com/PKUDigitalHealth/HeartLang) | QRS tokenizer、VQ 心拍重建、掩码预训练、数据处理和下游分类脚本均可见 | [官方 HF](https://huggingface.co/PKUDigitalHealth/HeartLang/tree/main/checkpoints) 有 pretrain / vqhbr；总目录约 827 MB | 优先：复用权重完成下游对照，借鉴心拍表示。README 的原始预训练示例使用 8 张 4090，单卡复现应先收缩到下游任务 |
| [TolerantECG](https://github.com/Fsoft-AIC/TolerantECG) | 训练、微调、评估、噪声/缺导数据组件、模型和配置均可见 | [作者 HF](https://huggingface.co/ndhuynh02/TolerantECG/tree/main) 有 `TolerantECG_encoder.pth`，约 107 MB | 优先：噪声和缺导联对照；先冻结特征或规范微调。信号—文本训练的监督来源与本项目纯信号 SSL 分开说明 |
| [ECG tokenizer 对照研究](https://github.com/LeeJarvis996/ecg_tokenizer) | 8 类 tokenizer、主干、CPSC 加载、BCE 训练/验证/测试循环可见 | 本轮未见整套分类模型权重；ECG-Byte 的 BPE 文件依赖外部项目 | 最适合先做小规模模板/逐拍/固定窗表示比较；该项目中的 HeartLang 式切片不等于 HeartLang 完整预训练模型 |
| [BeatRhythm-TTA](https://github.com/Vincent-Wenhan/BeatRhythm-TTA) | 源模型训练、峰缓存、主方法、离线与在线评价脚本可见 | 未找到随项目发布的源模型权重；需自行训练或提供 | 选择跨域适配方向时优先。当前方法目录主要有 source-only 和 BeatRhythm，论文其他对比方法不能默认都已随仓库提供 |
| [ECG-LFM](https://github.com/biomed-AI/ECG-LFM) | 数据预处理、fairseq-signals 模型/损失、预训练、微调及推理配置可见 | [Zenodo](https://zenodo.org/records/20388950) 公开 `pretrain.pt`，文件约 7.5 GB | 适合作为高水平开放权重参照；环境和模型较重，不建议首轮从头预训练。权重文件体积不能直接等同推理显存 |
| [ShapeEFM](https://github.com/AQ-MedAI/ShapeEFM) | 可见形态提取、tokenizer、模型和两阶段预训练脚本 | 本轮未见公开 checkpoint 链接 | 适合借鉴核心形态 token 代码；目前未见独立下游 LP/分类训练入口，不能直接标为完整端到端复现包 |
| [Beat-SSL](https://github.com/ilham1012/beat-ssl) | 已找到的作者仓库是论文展示站，顶层是 `static/`、`index.html`、README | 未确认训练代码或权重公开 | 方法可读；当前不列为开箱可复现首选。检索未发现不等于作者没有其他未索引仓库 |
| [IECG](https://arxiv.org/abs/2610.28083) | 本轮未从论文及公开检索中确认对应官方训练仓库 | 未确认 | 用于研究问题和解释评价参考，暂不承诺可直接复现 |
| [CODE-II](https://github.com/PetrusAbreu/code-ii) | 当前 Git 树只有 README，属于数据资源说明仓库；另链接通用预处理项目 | 该仓库未提供训练模型权重 | 外部数据候选；不是完整模型复现工程。README 的 PhysioNet 链接仍为 coming soon，不能仅凭论文的数据可用性声明认为下载已核实 |

### 关键源码入口

- HeartLang：`run_vqhbr_training.py`、`run_heartlang_pretraining.py`、`run_class_finetuning.py`、`QRSTokenizer.py`、`scripts/finetune/`。
- TolerantECG：`src/train.py`、`src/eval.py`、`src/models/dino_clip_module.py`、`src/models/dino_clip_finetune_module.py`、`script/train.sh`、`script/finetune.sh`。
- tokenizer 对照：`run.py`、`models/tokenizers/`、`exp/exp_classification.py`；分类损失实际为 `BCEWithLogitsLoss`。
- BeatRhythm-TTA：`src/method/beatrhythmtta.py`、`src/core/trainer_ecg.py`、`src/evaluation/run_test.py`、`scripts/data_process/precompute_rpeaks.py`、`scripts/train/`、`scripts/test/`。
- ShapeEFM：`tokenizer.py` / `tokenizer.sh`、`train.py` / `train.sh`。已读 `train.py`，其目标是 masked token 预测，不能因 README 提到 LP 就把它视为已公开下游训练链；虽有 LP 参数工具，当前文件树未见独立 LP 主程序。

## 2. 发表平台怎样理解

这组论文中，**ECG-LFM 和 CODE-II 分别发表于 Nature Communications 与 npj Digital Medicine**；其他多数是会议论文或预印本。期刊与会议不应用同一套 SCI 分区/影响因子直接排序。

| 平台 | 对应论文 | 评价与已核实信息 |
|---|---|---|
| Nature Communications | ECG-LFM | 高水平综合性研究期刊；Nature 官方指标列 2025 JIF 为 18.1 |
| npj Digital Medicine | CODE-II | 高水平数字医学专业期刊；Nature 官方指标列 2025 JIF 为 18.0 |
| ICLR | HeartLang | 表征学习/深度学习顶级会议；[正式论文集](https://proceedings.iclr.cc/paper_files/paper/2025/hash/17aa70697d6cd35835f201c6fb0a2fd5-Abstract-Conference.html) 已确认论文身份 |
| ACM Multimedia | TolerantECG | 多媒体领域高水平旗舰会议；[CCF 官方条目](https://www.ccf.org.cn/Academic_Evaluation/CGAndMT/zgjsjxhtjgjxshy/al/2017-03-17/588341.shtml) 列在 A 类 |
| MICCAI | ShapeEFM、BeatRhythm-TTA | 医学图像计算和计算机辅助干预领域高水平会议；两篇均有正式会议论文页面 |
| IEEE ISBI | Beat-SSL | 生物医学成像领域的重要专业会议；属于会议论文，不是期刊论文 |
| arXiv | tokenizer 对照；本轮所读 IECG 版本 | 预印本平台本身不是期刊或会议。IECG 页面有作者关于 WACV 2027 接收的备注，本轮未核验会议正式论文集 |

期刊指标来自 [Nature Portfolio 2025 Journal Metrics](https://www.nature.com/nature-portfolio/about-journals/journal-metrics) 与 [Nature Communications 指标页](https://www.nature.com/ncomms/ncomms/journal-impact)。18.1 与 18.0 的细小差异没有用于本项目选题或复现优先级排序。未用未经核实的中科院分区给这些平台定级。

## 3. 对本项目的选择

1. **心拍方法主线：先 HeartLang。** 平台强、方法邻近、代码入口和权重齐，先做一个既有任务的公开权重下游复现，再拆出可借鉴的 tokenizer/心拍处理。
2. **低成本问题诊断：先 tokenizer 对照代码。** 只选模板、心拍、固定窗几种表示，保持同一任务和预算；先不照抄全网格。
3. **导联/噪声主线：先 TolerantECG。** 利用公开编码器做压力测试与规范微调，再判断本项目是否有新机制空间。
4. **适配主线：BeatRhythm-TTA。** 需要单独定义源域训练和目标域无标签适配协议，不将它直接混入旧 LP/FT 表。
5. **高水平期刊参照：ECG-LFM。** 先解决环境、输入和权重加载，做冻结特征/少量下游适配；不把原始大规模预训练或遗传分析当作当前必做工作。

## 4. 源码快照记录

| 仓库 | 核查 commit | 已跟踪文件数 |
|---|---|---|
| PKUDigitalHealth/HeartLang | `a08afd8117e813fee212a1ac283db4a7140a4669` | 85 |
| AQ-MedAI/ShapeEFM | `9bb1b6e1f61cca5b2ccc5cbfff02f63e57dd2af3` | 39 |
| LeeJarvis996/ecg_tokenizer | `ce976cecaa27912fbba38c28fdda07abcc211b7d` | 53 |
| Vincent-Wenhan/BeatRhythm-TTA | `372f26bc40faa7a488b0e856f14ff1e918b5bfeb` | 53 |
| Fsoft-AIC/TolerantECG | `51a73c7ce3851f949dd075b7ca3ae27aa0f35fff` | 110 |
| PetrusAbreu/code-ii | `9aaa752f811cc81467737ed91a58cd0bd17daecb` | 1 |
| biomed-AI/ECG-LFM | `647369a012dd8d5d362e99e2bf238c91f3e1d03b` | 258 |

文件数只用于标识本轮快照，不作为代码质量或完整性评分。以上判断应在真正实施前重新检查依赖、输入契约、数据权限与 checkpoint 兼容性，并先做最小运行验证。
