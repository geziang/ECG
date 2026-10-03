# LRTC-Net: Lead-Aware Self-Supervised Learning with Minimal Channel-Response Recalibration for Cross-Domain, Label-Efficient Reduced-Lead ECG Classification

> **版本说明（2026-09-25 第2版，工作稿）**：本稿为主论文的统一协议证据版，取代 2026-09-07 的旧稿（`LRTC-Net_Applied_Sciences_English.md`，旧协议 PTB-XL 域内预训练证据，仅作归档）。全部数字逐项取自冻结台账：`runlog/W2/confirm_results.csv`、`runlog/W2/stats/`、`runlog/W3/paper_materials/`、`runlog/W4/baseline_results.csv`、`runlog/W4/paper_materials_v2/`、`runlog/W4/stats/paired_stats.csv`、`runlog/W2/missing_lead_c2.csv`、`runlog/W1/missing_lead_curves.csv`、`runlog/W2/freeze_manifest_v2.json` 及各语料 manifest；CPSC2018 全部数字取自修正分区重评估 `runlog/W5/lp_results.csv`（审计记录 `runlog/W5/cpsc_audit.md`）。与支撑论文（`支撑论文_负结果与边界分析_参考稿.md`）的交叉引用口径一致。
>
> **待办（提交前必须关闭）**：
> 1. ✅ 已闭环（2026-09-25，W5A `runlog/W5/`）：CPSC 测试集审计判定"残留"（两版标签映射先后运行未清理目录，磁盘成 2,004 并集、含 2 个跨类重复 stem），数据重建后干净分区 = 1,385，15 跑 CPSC LP 已用 SHA 校验的冻结 checkpoint 重出；本文全部 CPSC 数字为修正分区口径。
> 2. CPSC 的 record-level bootstrap 为办公机临时复算（10,000 次同步重采样，RNG 20260925），待 B 机官方重算替换（输入 = `runlog/W5/predictions/`）；涉及主文 Table 6、§4.2/§4.4 及 Supplementary Table S3 的 CPSC 四行（含 DeLong 列）。
> 3. S3/S4 补充表已排版并入本文件末尾（2026-09-25），暂存稿已删除。
> 4. ✅ 插图全部闭环（2026-09-28）：图 1 = 新三域叙事版（作者 PPT 绘制导出，12597×7197）；图 4 = matplotlib 重生成（双面板，脚本内置 Table 8 四组数字自动核对通过）；图 2/3 = 作者 9-1 PowerPoint 导出版入稿（12600×7200；源 PPT 尚有三处小字缺陷待修后重导：图 2 归一化框指数"10⁻ ⁵"空隙、图 3 步骤 2"10⁻ ⁶"同病、图 3 公式批次下标 β 应为 b 且步骤 2 公式折行截断——不影响汇报）。Author Contributions / Funding / IRB / Data Availability 占位段待补（需作者信息）。

## Abstract

Self-supervised learning (SSL) for multi-lead electrocardiograms (ECGs) is increasingly pretrained on external hospital corpora and transferred to target domains with few labels, yet the incremental value of small architectural modifications under this cross-corpus regime is rarely isolated. We propose LRTC-Net, which embeds Temporal Response Calibration (TRC)—a zero-initialized, channel-wise recalibration module with 1,024 parameters (0.161% of the encoder)—into each branch of a lead-aware redundancy-reduction encoder (LRRE) and jointly optimizes it during SSL pretraining on an external corpus of 34,905 unlabeled eight-lead ECGs (the Ningbo portion of the Chapman-Shaoxing/Ningbo collection, NFH). Under a preregistered protocol with three seeds and sign-consistency gates, we compared LRTC-Net against its protocol-matched control (C1: identical encoder, corpus, updates, and downstream protocol, without TRC) and an optimizer-update-matched in-domain anchor (B0) across three downstream domains: PTB-XL (five-class; linear probe and 10%-label fine-tuning), CPSC2018 (nine-class linear probe), and Chapman-Shaoxing (four-class linear probe). TRC improved third-domain transfer: on CPSC2018, macro-AUPRC increased by +0.83 points (three of three seeds consistent; seed-level 95% CI [+0.02, +1.64]), and 10%-label fine-tuning on PTB-XL improved by +0.65 points (three of three seeds); the record-level paired bootstrap on macro-AUROC did not resolve the CPSC2018 difference (+0.10 points, p = 0.19), so the third-domain gain is seed-consistent but not record-level confirmed. Gains were absent on the in-domain linear probe and on the lineage-related Chapman domain, delimiting where recalibration helps. Against published baselines under matched budgets without hyperparameter search, the LRRE family exceeded SimCLR and CLOCS in five of six linear-probe comparisons. Fine-tuning with 10% of labels recovered 97.7% of the macro-AUPRC of a fully supervised network. Under lead zero-masking, LRTC-Net retained the smaller degradation of external pretraining (mean −0.59 points per single dropped lead and −1.41 per dropped pair, versus −0.83 and −1.88 for the in-domain anchor) without absolute crossover. These results characterize a minimal, mechanism-targeted modification whose benefits concentrate exactly where domain shift is largest.

**Keywords:** electrocardiogram; self-supervised learning; multi-lead ECG; channel recalibration; cross-domain transfer; label efficiency; PTB-XL; CPSC2018

## 1. Introduction

Electrocardiography (ECG) is routinely used to screen for and diagnose rhythm, conduction, myocardial injury, and repolarization abnormalities, and public datasets such as PTB-XL have facilitated reproducible algorithm development [1, 2]. The expansion of telemedicine, ambulatory monitoring, and wearable ECG acquisition has increased the demand for automated analysis that can prioritize records for expert review [3]. In practice, however, high-quality labels require specialist interpretation, recordings are affected by electrode-contact problems and motion artifacts, and deployment sites rarely coincide with the institutions that produced the training corpora [4-6]. Models intended for low-resource settings must therefore handle two coupled constraints: expert labels are scarce, and the unlabeled corpus available for pretraining differs statistically from the target domain.

Self-supervised learning (SSL) addresses label scarcity by deriving a pretraining signal from unlabeled ECG records [14-16]. Redundancy-reduction objectives such as Barlow Twins [17] have been adapted to multi-lead ECG through lead-fusion formulations [18, 19], and subsequent work has explored masked reconstruction, temporal-spatial consistency, period-aware objectives, and JEPA pretraining [20-30]. In applied deployments, the pretraining corpus is increasingly external to the evaluation corpus—large multi-source corpora are the norm rather than the exception—yet published ablations usually vary the objective or the module while holding an in-domain pretraining corpus fixed. Whether a small architectural modification retains its value under external pretraining and cross-domain transfer is therefore difficult to determine from cross-study comparisons alone. Re-evaluations in metric learning, recommender systems, and computer vision have shown that protocol calibration can eliminate much of the reported progress over simple baselines [40, 41]; ECG SSL is exposed to the same risk because corpus identity, split hygiene, and downstream protocol move results by magnitudes comparable to typical module effects.

Channel-response recalibration offers a parameter-efficient intervention point. Squeeze-and-Excitation and ECA-Net demonstrate that channel statistics can modulate intermediate representations at limited structural cost [31, 32], and ECG-specific studies have applied lightweight channel or temporal attention to classification and signal-quality assessment [33, 34]. What these studies do not establish is whether channel-wise recalibration of high-level temporal representations, jointly optimized during external-corpus SSL pretraining, yields an independent downstream benefit under a protocol-matched comparison—and, critically, in which transfer directions that benefit concentrates. This gap motivates the following testable hypothesis: external pretraining produces temporal representations whose channel-response statistics require a small amount of re-weighting under target-domain statistical shift; such recalibration should help most on a third domain with large distributional distance and under limited-label fine-tuning, and should leave nearly unchanged the domains that are either matched to the pretraining corpus or related to it by acquisition lineage.

To test this hypothesis, we developed LRTC-Net by embedding the channel-wise Temporal Response Calibration (TRC) module in each branch of a lead-aware redundancy-reduction encoder (LRRE) and pretrained it on the NFH external corpus. Three instantiations were compared under an identical downstream protocol: B0, the LRRE anchor pretrained in-domain on PTB-XL with matched optimizer updates; C1, the protocol-matched control pretrained on NFH without TRC; and C2 (LRTC-Net), identical to C1 except for TRC. We evaluated linear probing on PTB-XL, CPSC2018, and Chapman-Shaoxing, and 10%-label fine-tuning on PTB-XL, with preregistered promotion gates, validation-only model selection, and a frozen test evaluation. External baselines (SimCLR, CLOCS, and a fully supervised reference) were run under matched budgets without hyperparameter search. The incremental contribution of TRC was further examined through record-level paired bootstrap statistics, simulated lead dropout by zero-masking, and parameter accounting.

The contributions of this study are fourfold:

1. We isolate, under a preregistered and protocol-matched comparison with matched optimizer updates, the incremental value of a 1,024-parameter channel-wise recalibration module during external-corpus pretraining of a lead-aware SSL encoder, and locate its benefits precisely: third-domain transfer (+0.83 macro-AUPRC points on CPSC2018, three of three seeds, CI [+0.02, +1.64] excluding zero narrowly) and 10%-label fine-tuning (+0.65 points, three of three seeds), with no material change on the in-domain probe or the lineage-related domain.
2. We characterize the pretraining-corpus effect itself: external pretraining costs 5.86 macro-AUPRC points on the in-domain PTB-XL linear probe relative to the update-matched in-domain anchor, a gap that 10%-label fine-tuning narrows by a factor of 3.4, and it confers smaller degradation under lead zero-masking—findings that delimit what "external pretraining" does and does not buy.
3. We provide matched-budget external reference points (SimCLR, CLOCS, supervised-direct) and record-level paired statistics over per-record predictions, which anchor the family's absolute position among established methods without hyperparameter search on either side.
4. We report the evaluation discipline itself—preregistered gates, seed-level paired deltas, frozen single-shot test evaluation, and full ledger provenance for every number—as the framework within which a minimal modification can be credited or rejected; a companion study documents the more than 40 modification families explored around the same anchor that failed these gates.

## 2. Related Work

### 2.1. Multi-Lead ECG Learning and Applied Constraints

Modern ECG analysis has progressed from manually designed features to one-dimensional convolutional, residual, and Inception-style networks that learn multiscale patterns directly from raw signals [7, 9]. Multi-lead models additionally use graph-based or multi-branch mechanisms to preserve relationships among leads [35]. Benchmark analyses on PTB-XL have shown that label hierarchy, patient partitioning, class imbalance, and the selected metric materially affect reported performance [2, 9], so comparisons must specify input leads, label construction, splits, fine-tuning protocol, and primary metric.

Input configuration is an application-level consideration. Reduced-lead classification and teacher-student compression seek to reduce acquisition or computational requirements, whereas lead reconstruction attempts to infer unavailable channels [10-13]; lead importance is task-dependent, and reconstructed waveforms may introduce bias. Electrode misplacement and poor signal quality are different forms of input variation because they alter the recorded signal rather than removing a channel [4-6]. We used a fixed eight-lead configuration (II, III, V1–V6) throughout and treated simulated lead dropout by zero-masking as a controlled stress test, keeping the term distinct from electrode displacement.

### 2.2. ECG Self-Supervised Representation Learning and External Pretraining

ECG SSL methods span contrastive or invariance learning, reconstruction-based learning, and lead-fusion objectives. CLOCS exploits relationships across space, time, and patients [15]; Contrastive Heartbeats uses heartbeat-level sampling [16]; and a comprehensive 12-lead assessment established reproducible label-efficiency baselines [14]. Barlow Twins learns by matching the cross-correlation matrix of two augmented views to the identity [17]; lead-fusion Barlow Twins (LFBT) extends it with intra-lead and inter-lead objectives [18], and LCD formulates lead-correlation and decorrelation objectives [19]. Recent work has added masked reconstruction, temporal-spatial consistency, out-of-distribution evaluation, multiview information bottlenecks, adversarial perturbations, period-aware and long-term objectives, sequential MAE clustering, and JEPA pretraining [20-30].

Two aspects of this literature motivated our design. First, the strongest published lead-fusion results were obtained under external-corpus pretraining on hospital data distinct from the evaluation corpora [18], yet follow-up work usually transplants a module into an in-domain protocol and attributes the outcome to the module name; the protocol—corpus, split, updates, downstream evaluation—is a first-order variable. Second, cross-study gains for small modules are frequently of the same magnitude as cross-seed variability, a problem documented formally in other fields [40, 41]. We therefore held the entire protocol fixed except for the module under test and preregistered the decision rule before the confirmatory runs.

### 2.3. Lightweight Feature Recalibration and Lead Availability

Squeeze-and-Excitation blocks model channel dependencies via global statistics [31]; ECA-Net captures local cross-channel interactions without dimensionality reduction [32]; and global response normalization with zero-initialized gains has been used to stabilize masked-autoencoder training in vision [39]. ECG-specific attention studies have applied channel and temporal attention to arrhythmia classification and signal-quality assessment [33, 34]. These methods motivate response modulation as a design principle, but they do not establish the effect of jointly optimizing such a module during ECG SSL pretraining on an external corpus, nor in which transfer directions the effect concentrates. TRC denotes channel-wise feature recalibration on temporal feature maps; it is not temporal attention and not probability calibration.

Changes in input availability are also relevant. Wearable recordings are susceptible to motion and contact artifacts, and electrode misplacement alters the spatial information of precordial leads [4-6]. SSL studies have used temporal or lead masking, and other work has examined lead reconstruction and reduced lead sets [12, 13, 20, 22, 23]. In the present evaluation, selected normalized leads were zero-masked and suppressed by a valid-lead mask before classification; this is a controlled perturbation of input availability, not a simulation of electrode displacement.

In summary, prior work establishes lead-aware SSL encoders, channel-response recalibration, external-corpus pretraining, and lead-availability analysis as separate threads. What remains limited is a protocol-matched evaluation that combines external pretraining with a parameter-efficient recalibration module, locates the module's benefit across transfer directions, and reports the statistics and provenance needed to interpret a small effect. The present study provides such an evaluation, and a companion paper reports the systematic negative-results program that closed the surrounding modification space.

## 3. Materials and Methods

### 3.1. Study Design, Model Instantiations, and Preregistered Gates

The study used three public corpora in two roles (Table 1). The NFH corpus (the Ningbo portion of the Chapman-Shaoxing/Ningbo collection [38]) supplied unlabeled external pretraining data; PTB-XL [2], CPSC2018 [37], and Chapman-Shaoxing [38] supplied downstream tasks. Three model instantiations of the same architecture family were compared:

- **B0 (in-domain anchor):** LRRE pretrained on PTB-XL folds 1–8 (200 epochs; ≈27,216 optimizer updates), evaluated to bound what in-domain pretraining achieves.
- **C1 (protocol-matched control):** LRRE pretrained on NFH (100 epochs; ≈27,270 optimizer updates), with encoder, objective, augmentations, batch size, downstream protocol, and seeds identical to C2 except for the absence of TRC.
- **C2 (LRTC-Net):** C1 plus TRC, jointly optimized during pretraining.

Because the NFH corpus is twice as large, C1/C2 used half as many epochs as B0 so that all three chains consumed a matched number of optimizer updates (≈27.2k). This step-matching is a precondition for attributing differences to corpus or module rather than to training length.

Confirmatory evaluation followed a preregistered discipline with three seeds (0, 2, 4) per configuration and identical downstream seeds within each pairing. Promotion gates were fixed before the confirmatory runs: for the CPSC2018 linear probe, TRC required same-sign improvement on three of three seeds and a mean paired gain of at least +0.5 macro-AUPRC points; for PTB-XL 10%-label fine-tuning, at least two of three seeds and a mean of at least +0.5 points. Model selection during exploration used validation data only; each frozen checkpoint was evaluated once on its test split. Because macro-AUPRC is not decomposable per record, the primary metric for seed-level gates was macro-AUPRC (mean paired points, exact sign-flip permutation p, and seed-level t intervals), and record-level paired bootstrap was applied to macro-AUROC using per-record predictions; all record-level statements are explicitly record-level and not patient-level.

### 3.2. Data Sources, Splits, and Label Construction

**PTB-XL** [2] comprised 21,799 ten-second 12-lead records at 100/500 Hz across official folds 1–10. Folds 1–8 supplied records for in-domain pretraining (B0) and for downstream training; fold 9 was the validation set; fold 10 (1,739 records) was the locked test set. The five-class single-label task (CD, HYP, MI, NORM, STTC) was constructed from `scp_codes` with the official `diagnostic_class` mapping as in our previous protocol: records mapping to multiple distinct abnormal superclasses were excluded, records with no abnormal superclass were assigned NORM, and an abnormal superclass took precedence over NORM when both were present. A `patient_id` check confirmed no patient occurred in more than one split. The downstream training, validation, and test sets comprised 13,639, 1,714, and 1,739 records, respectively (Table 2); the pretraining and supervised sets overlap by design, and diagnostic labels were not accessed during pretraining.

**NFH (external pretraining corpus).** The Ningbo portion of the Chapman-Shaoxing/Ningbo collection [38] contributed 34,905 ten-second twelve-lead records at 500 Hz, processed with the same eight-lead selection, resampling, and normalization as the downstream corpora. No diagnostic labels were used. The corpus release provides no patient identifiers; duplicate auditing was therefore performed by record-content SHA256, which flagged three duplicate pairs that were retained and marked. Record identifiers of the Ningbo portion (JS10647 onward) occupy a range disjoint from the Chapman-Shaoxing evaluation subset (JS00001–JS10646), but the two corpora share acquisition team and device lineage; this lineage is treated explicitly in the interpretation of the Chapman results.

**CPSC2018** [37] comprised 6,877 twelve-lead records with nine rhythm/morphology classes (NSR, AF, IAVB, LBBB, RBBB, PAC, PVC, STD, STE). Multi-label statements were mapped to a single label by a fixed class-priority list, and a record-level stratified 70/10/20 split (seed 0) produced training, validation, and test partitions; the corrected test partition comprised 1,385 samples (see the data-audit note below). **Chapman-Shaoxing** [38] comprised 10,216 labeled records with four rhythm classes (SB, AFIB, GSVT, SR) under the same record-level stratified split; the test partition comprised 2,047 records. For both corpora, one record corresponds to one patient in the release, but no patient-level identifier was retained for clustering-robust statistics; all statistics are therefore reported at record level.

**Data-audit note (CPSC test partition; resolved).** A post hoc reconciliation of per-record prediction counts against the preparation manifest revealed that the CPSC2018 directory used during the original confirmatory evaluations contained 2,004 test files rather than the 1,385 recorded by the manifest: two label-mapping revisions had been run successively over the same directory without cleaning it, so six classes retained residual files from the first run (619 files in total) and two records appeared in two class folders simultaneously. The directory was rebuilt from source, the corrected partition (test = 1,385; zero cross-class duplicates; per-class counts identical to the manifest) was verified, and every CPSC2018 result in this paper was re-evaluated on the corrected partition from the same SHA256-verified frozen checkpoints (15 of 15 verified). The contaminated-era CPSC2018 scores remain in the frozen ledgers unchanged and are not comparable with the corrected values; the correction reduced the apparent third-domain TRC gain (Section 4.2), and all record-level CPSC2018 statistics reported here are provisional recomputations pending independent confirmation.

**Table 1. Corpora and their roles.**

| Corpus | Role | Records used | Leads | Classes | Test split |
|---|---|---:|---:|---:|---|
| NFH (Ningbo portion of Chapman-Shaoxing/Ningbo) | External SSL pretraining (C1, C2) | 34,905 | 8 | Unlabeled | — |
| PTB-XL folds 1–8 | In-domain SSL pretraining (B0); downstream train | 17,418 / 13,639 | 8 | 5 | Fold 10: 1,739 |
| CPSC2018 | Third-domain downstream (LP) | 6,877 | 8 | 9 | 1,385 (corrected partition) |
| Chapman-Shaoxing | Lineage-domain downstream (LP) | 10,216 | 8 | 4 | 2,047 |

**Table 2. PTB-XL downstream class distribution.**

| Class | Training | Validation | Test | Total |
|---|---:|---:|---:|---:|
| CD (conduction disturbance) | 1,680 | 206 | 229 | 2,115 |
| HYP (hypertrophy) | 416 | 65 | 56 | 537 |
| MI (myocardial infarction-related) | 2,043 | 233 | 256 | 2,532 |
| NORM (normal) | 7,577 | 951 | 952 | 9,480 |
| STTC (ST/T change) | 1,923 | 259 | 246 | 2,428 |

### 3.3. Signal Preprocessing and Simulated Lead Dropout

Preprocessing was identical for all corpora. We did not treat the 12-lead ECG as twelve exchangeable and fully independent channels. Under the ideal standard lead system, the limb leads approximately satisfy the following relations [1]:

$$
I=II-III,
\tag{1}
$$

$$
aVR=-\frac{I+II}{2},\qquad aVL=I-\frac{II}{2},\qquad aVF=II-\frac{I}{2}.
\tag{2}
$$

Although these relations are only approximate in measured recordings because of noise, electrode placement, and acquisition conditions, they indicate partial algebraic redundancy among the limb leads, whereas the precordial leads V1–V6 are acquired from different chest positions and provide complementary spatial projections of cardiac electrical activity [1]. Following the configuration used in related multi-lead self-supervised research [18], we retained leads II, III, and V1–V6 and omitted the algebraically related limb leads I, aVR, aVL, and aVF.

Let $X_{12} \in \mathbb{R}^{12 \times L}$ denote a 12-lead record. The eight-lead input was obtained with a fixed lead-selection matrix $S \in \{0,1\}^{8 \times 12}$:

$$
X_8=SX_{12},
\tag{3}
$$

where $S$ extracts leads II, III, V1, V2, V3, V4, V5, and V6 in that fixed order. We denote the selected signal as:

$$
X=[x_1,\ldots,x_8]\in\mathbb{R}^{8\times L}.
\tag{4}
$$

Each record was resampled along the temporal dimension to 2,048 samples (preserving the ten-second duration rather than cropping), and the mean and standard deviation were computed jointly over all eight leads and all time points of each record before normalization:

$$
\tilde{X}=\frac{X-\mu(X)}{\sigma(X)+10^{-5}},
\tag{5}
$$

where $\mu(X)$ and $\sigma(X)$ denote the global mean and global standard deviation over all $8\times2048$ samples of one record, yielding the final model input

$$
\tilde{X}\in\mathbb{R}^{8\times2048}.
\tag{6}
$$

This record-wise normalization used no aggregate training-set statistics and was not computed per lead. For CPSC2018 and Chapman-Shaoxing, the central ten seconds were cropped at 500 Hz before resampling, with zero-padding for shorter records. No band-pass filtering, baseline-drift correction, or quality-based exclusion was applied.

During pretraining, two augmented views were generated per record: each view randomly retained a continuous segment covering 50%–100% of the duration, resampled it to 2,048 points, and randomly zeroed a continuous segment covering 0%–50%. No amplitude scaling, additive noise, or lead masking was used in pretraining. Simulated lead dropout was applied after normalization at evaluation time by zero-masking the selected lead over all 2,048 samples and suppressing its encoded representation with a `valid_mask` before feature concatenation; throughout this paper, *lead dropout* denotes this controlled zero-masking procedure and not physical electrode displacement. Thirty-seven reproducible conditions were evaluated: complete input, eight single-lead conditions, and 28 double-lead conditions.

### 3.4. Lead-Aware Redundancy-Reduction Encoder

LRRE assigns one independent five-block one-dimensional convolutional branch to each lead, without parameter sharing. Each branch maps a 2,048-sample single-lead signal through five convolutional blocks (13 Conv1d layers in total; channels 8, 16, 32, 64, 64; each layer followed by BatchNorm and ReLU) interleaved with five two-fold max-pooling operations (feature maps 1 × 2048 → 64 × 64), and adaptive average pooling produces a 64-dimensional representation per lead. A lead-specific three-layer projection head (64 → 2048 → 2048 → 2048) maps each representation into the self-supervised space during pretraining.

For each augmented view $v\in\{1,2\}$ and lead $l$, the convolutional backbone generates a high-level temporal feature map, the lead-specific TRC module (present only in C2) recalibrates the channel responses, temporal pooling produces a lead representation, and a lead-specific projection head maps it into the self-supervised space:

$$
F_l^{(v)}=f_{\theta_l}^{\mathrm{backbone}}(\tilde{x}_l^{(v)})\in\mathbb{R}^{C\times T},\qquad
Y_l^{(v)}=\operatorname{TRC}_{\psi_l}\left(F_l^{(v)}\right)\in\mathbb{R}^{C\times T},\qquad
h_l^{(v)}=\operatorname{GAP}_T\left(Y_l^{(v)}\right)\in\mathbb{R}^{C},\qquad
z_l^{(v)}=g_{\phi_l}\left(h_l^{(v)}\right)\in\mathbb{R}^{D}.
\tag{7}
$$

Here $\theta_l$, $\psi_l$, and $\phi_l$ denote the backbone, TRC, and projection-head parameters for lead $l$; in C1, $Y_l^{(v)}=F_l^{(v)}$. Let $Z_{l,a}^{(v,n)}=[z_l^{(v,n)}]_a$ denote the projected value for sample $n$, lead $l$, view $v$, and projection dimension $a$. For a mini-batch of $B$ samples, each projection dimension was centered and standardized to unit variance along the batch:

$$
\hat{Z}_{l,a}^{(v,n)}=
\frac{Z_{l,a}^{(v,n)}-\mu_{l,a}^{(v)}}
{\sqrt{(\sigma_{l,a}^{(v)})^2+\varepsilon}},\qquad
\mu_{l,a}^{(v)}=\frac{1}{B}\sum_{n=1}^{B}Z_{l,a}^{(v,n)},\qquad
(\sigma_{l,a}^{(v)})^2=\frac{1}{B}\sum_{n=1}^{B}
\left(Z_{l,a}^{(v,n)}-\mu_{l,a}^{(v)}\right)^2,
\tag{8}
$$

with $\varepsilon=10^{-6}$. For any ordered lead pair $(i,j)$, the cross-view correlation matrix was defined as

$$
C^{(i,j)}_{ab}=\frac{1}{B}\sum_{n=1}^{B}
\hat{Z}_{i,a}^{(1,n)}\hat{Z}_{j,b}^{(2,n)},
\tag{9}
$$

where the ordered cross-lead set contains both $(i,j)$ and $(j,i)$, so the correlations are computed in both directions; intra-lead terms correspond to $i=j$ and inter-lead terms to $i\ne j$, both under Eq. (9). The redundancy-reduction loss for one lead pair was

$$
\mathcal{L}_{\mathrm{RR}}^{(i,j)}=
\sum_{a=1}^{D}(1-C_{aa}^{(i,j)})^2+
\lambda\sum_{a\ne b}(C_{ab}^{(i,j)})^2,
\tag{10}
$$

with the off-diagonal weight $\lambda=0.0051$. To keep the intra-lead and inter-lead constraints at their prespecified weights, the lead-pair sets and mean losses were

$$
\mathcal{P}_{\mathrm{intra}}=\{(l,l)\mid l=1,\ldots,8\},\qquad
\mathcal{P}_{\mathrm{inter}}=\{(i,j)\mid i,j\in\{1,\ldots,8\},\ i\ne j\},
\tag{11}
$$

$$
\mathcal{L}_{\mathrm{intra}}=
\frac{1}{|\mathcal{P}_{\mathrm{intra}}|}
\sum_{(i,j)\in\mathcal{P}_{\mathrm{intra}}}
\mathcal{L}_{\mathrm{RR}}^{(i,j)},\qquad
\mathcal{L}_{\mathrm{inter}}=
\frac{1}{|\mathcal{P}_{\mathrm{inter}}|}
\sum_{(i,j)\in\mathcal{P}_{\mathrm{inter}}}
\mathcal{L}_{\mathrm{RR}}^{(i,j)},
\tag{12}
$$

with $|\mathcal{P}_{\mathrm{intra}}|=8$ and $|\mathcal{P}_{\mathrm{inter}}|=56$. The total objective is

$$
\mathcal{L}_{\mathrm{SSL}}=0.8\,\mathcal{L}_{\mathrm{intra}}+0.2\,\mathcal{L}_{\mathrm{inter}}.
\tag{13}
$$

The inter-lead term is a redundancy-reduction constraint—not a similarity maximization—so it suppresses cross-dimensional redundancy while preserving lead-specific information; a companion study documents that rewriting it as hard cross-lead alignment consistently damages transfer.

### 3.5. Temporal Response Calibration and Downstream Heads

TRC operates on the feature map produced by each branch after the fifth convolutional block and its max-pooling:

$$
F\in\mathbb{R}^{B\times64\times64}.
\tag{14}
$$

The L2 response of each channel along the temporal dimension was calculated as

$$
r_{b,c}=\left\|F_{b,c,:}\right\|_2,
\tag{15}
$$

and normalized by the mean channel response within the same sample:

$$
n_{b,c}=\frac{r_{b,c}}{\frac{1}{64}\sum_{c'=1}^{64}r_{b,c'}+10^{-6}}.
\tag{16}
$$

The final output was obtained through residual channel modulation

$$
Y=F+\gamma\odot(F\odot n)+\beta,
\tag{17}
$$

with learnable channel-wise parameters $\gamma,\beta\in\mathbb{R}^{1\times64\times1}$ initialized to zero, broadcast across batch and time. Eight independent TRC modules (one per lead) add 128 parameters each, for 1,024 in total. TRC is part of the encoder, jointly optimized during pretraining, and transferred with the encoder weights; it computes channel-wise coefficients from temporal response statistics and applies no pointwise temporal attention. The zero initialization makes the module an identity at the start of training, so C2 begins exactly where C1 does and any divergence must be learned.

Downstream evaluation used two protocols. **Linear probe (LP):** the pretrained encoder was frozen and a linear classification head on the 512-dimensional concatenated representation was trained for 100 epochs (Adam, learning rate 0.001, batch 128) with validation-loss model selection; the head dimension matched each task (5, 9, or 4 classes). **10%-label fine-tuning (FT-10):** a class-stratified 10% subset of the PTB-XL downstream training set was used for full-model fine-tuning (encoder, TRC where present, and head) for 100 epochs (Adam, learning rate 0.0001, batch 128, unweighted cross-entropy), with validation-loss model selection. Under a given seed, all compared models used the identical label subset.

### 3.6. External Baselines, Statistics, and Reproducibility

Three external references were run under matched budgets with no hyperparameter search on either side; all deviations from the C1 configuration were logged in a baseline hyperparameter ledger. **S1 (SimCLR)** [36] replaced the pretraining objective with NT-Xent (temperature 0.5, published default) while keeping the backbone, projector, augmentations, batch size, optimizer, corpus, and update count identical to C1. **S2 (CLOCS)** [15] used the official multi-positive formulation with matched backbone and updates; the implementation was cross-checked against the official repository. **S3 (supervised-direct)** trained the same architecture from random initialization on the full PTB-XL downstream training set with the fine-tuning defaults (learning rate 0.0001, 100 epochs, batch 128), anchoring the fully supervised upper reference. Stronger recent methods (e.g., ST-MEM, MERL, HeartLang) were not included; the comparison scope is stated as same-budget, no-tuning, same-backbone-family references.

**Statistics.** Seed-level paired deltas in macro-AUPRC (points) are reported per seed with exact one-sided sign-flip permutation p (minimum attainable 0.125 at n = 3) and seed-level t intervals (mean ± 4.3027·sd/√3); in accordance with the preregistered wording rules, three-of-three same-direction results are described as sign-consistent, never as statistically significant. Record-level paired bootstrap on macro-AUROC resampled test-record indices 10,000 times with the same index applied synchronously to both methods and to all seeds; two-sided p and percentile 95% CIs are reported as descriptive evidence, with per-class DeLong tests (seed 0) as an auxiliary analysis. All record-level statistics are at the record level; no patient-level significance is claimed.

**Reproducibility.** Random states of Python, NumPy, and PyTorch (CPU/CUDA) were fixed per seed with deterministic cuDNN settings. Confirmatory runs used seeds 0, 2, 4; the B0 PTB-XL rows combine the seed-0 anchor evaluation with seed-2/4 confirmations, and the B0 Chapman rows combine the seed-0 anchor evaluation with a later two-seed extension (a documented +0.0010 macro-AUROC code-version difference affects the PTB-XL seed-0 anchor reading only). The B0 CPSC2018 rows derive entirely, together with all other CPSC2018 rows, from the corrected-partition re-evaluation described in Section 3.2. Checkpoints, manifests, and per-record predictions are hashed (SHA256) in a frozen ledger; 20 of 20 ledger re-evaluations reproduced bitwise. All results in this paper were produced with Python 3.11.9 and PyTorch 2.5.1+cu121 on an NVIDIA RTX 4090, with audits and recomputation on an RTX 3080.

**Table 3. Unified training settings.**

| Item | B0 | C1 / C2 (LRTC-Net) |
|---|---|---|
| Pretraining corpus | PTB-XL folds 1–8 (17,418) | NFH external (34,905) |
| Pretraining epochs × batch | 200 × 128 | 100 × 128 |
| Optimizer updates (matched) | ≈27,216 | ≈27,270 |
| TRC | Absent | Present in C2 only (1,024 parameters) |
| Pretraining optimizer / LR | Adam / 0.001 | Adam / 0.001 |
| Augmentations (two views) | Crop 50–100% + resample; zero-mask 0–50% | Same |
| Objective | Eq. 13 (λ = 0.0051) | Same |
| Downstream seeds | 0, 2, 4 | 0, 2, 4 |
| LP head / optimizer | Linear; Adam, LR 0.001, 100 epochs | Same |
| FT-10 | — (PTB-XL only) | 10% stratified labels; Adam, LR 0.0001, 100 epochs |
| Test evaluation | Frozen checkpoint, single evaluation | Same |

## 4. Results

### 4.1. Primary Results across Three Domains

Table 4 reports macro-AUROC and macro-AUPRC (mean ± sample standard deviation over seeds 0, 2, 4) for the three instantiations. Three patterns organize the table. First, the in-domain anchor B0 achieved the best PTB-XL linear-probe performance (0.7119 macro-AUPRC), confirming the value of matching the pretraining corpus to the target domain. Second, external pretraining (C1/C2) degraded the PTB-XL probe by 5–6 points and left C2 0.49 macro-AUPRC points below the in-domain anchor on CPSC2018 (0.7683 vs 0.7732; per-seed differences inconsistent at −1.16/−0.66/+0.35) despite never having seen PTB-XL or CPSC2018 data during pretraining—recovering most, but not all, of the in-domain advantage on a third domain. Third, on the lineage-related Chapman domain, both NFH-pretrained models exceeded the anchor by roughly one macro-AUPRC point, an advantage interpreted as acquisition lineage rather than generic cross-domain transfer (Section 4.4).

**Table 4. Main results (mean ± sample SD over seeds 0/2/4).**

| Eval / domain | Metric | B0 (in-domain) | C1 (external, no TRC) | C2 = LRTC-Net (external + TRC) |
|---|---|---:|---:|---:|
| LP / PTB-XL | macro-AUROC | **0.9138 ± 0.0013** | 0.8848 ± 0.0056 | 0.8860 ± 0.0016 |
| LP / PTB-XL | macro-AUPRC | **0.7119 ± 0.0073** | 0.6533 ± 0.0107 | 0.6554 ± 0.0032 |
| LP / CPSC2018 | macro-AUROC | **0.9506 ± 0.0006** | 0.9463 ± 0.0016 | 0.9474 ± 0.0022 |
| LP / CPSC2018 | macro-AUPRC | **0.7732 ± 0.0038** | 0.7600 ± 0.0024 | 0.7683 ± 0.0056 |
| LP / Chapman | macro-AUROC | 0.9932 ± 0.0007 | **0.9964 ± 0.0001** | 0.9963 ± 0.0003 |
| LP / Chapman | macro-AUPRC | 0.9795 ± 0.0021 | **0.9889 ± 0.0002** | 0.9886 ± 0.0004 |
| FT-10 / PTB-XL | macro-AUROC | **0.8714 ± 0.0039** | 0.8544 ± 0.0060 | 0.8597 ± 0.0038 |
| FT-10 / PTB-XL | macro-AUPRC | **0.6449 ± 0.0022** | 0.6279 ± 0.0053 | 0.6345 ± 0.0049 |

*Note:* All CPSC2018 rows are three-seed re-evaluations on the corrected partition (Section 3.2). The B0 Chapman rows combine the seed-0 anchor evaluation with a two-seed extension (n = 3; Section 3.6). Bold marks the column-wise best value per row.

### 4.2. Incremental Effect of TRC under the Preregistered Gates

Table 5 reports the seed-level paired deltas (macro-AUPRC, points) of C2 over C1, and Table 6 the record-level paired bootstrap on macro-AUROC. On the CPSC2018 third-domain probe (corrected partition), TRC improved every seed (+0.70 / +0.59 / +1.20; mean +0.83), passing the preregistered promotion gate (3/3 same sign, mean ≥ +0.5), with the seed-level 95% t interval [+0.02, +1.64] excluding zero, though narrowly. The record-level paired bootstrap on macro-AUROC did not resolve the difference (+0.10 points, 95% CI [−0.05, +0.25]; p = 0.19; provisional), so the third-domain gain is seed-consistent but not record-level confirmed. On PTB-XL FT-10, TRC improved every seed (+0.19 / +1.38 / +0.39; mean +0.65), passing the fine-tuning gate, with a seed-level interval that includes zero ([−0.93, +2.24]) and a record-level AUROC difference of +0.12 points (p = 0.48)—the direction is consistent, the magnitude is descriptive. By contrast, TRC left the in-domain linear probe essentially unchanged (+0.21 points, one of three seeds; bootstrap −0.12 points AUROC, p = 0.48) and left the lineage-related Chapman domain unchanged (−0.03 points, zero of three; bootstrap −0.01, p = 0.49). The benefit of channel-response recalibration under external pretraining is therefore directionally concentrated where the domain shift is largest, and is absent where the target domain matches or relates to the pretraining corpus, as the motivating hypothesis anticipated; after the partition correction, the third-domain component is a seed-consistent trend rather than a record-level-confirmed effect.

**Table 5. Seed-level paired deltas, macro-AUPRC (points), C2 − C1 and C1 − B0.**

| Comparison / eval | d(s0) | d(s2) | d(s4) | Mean | Sign | Permutation p (one-sided) | 95% CI (seed-level t) |
|---|---:|---:|---:|---:|---|---:|---:|
| C2 − C1, LP / CPSC2018 | +0.70 | +0.59 | +1.20 | **+0.83** | 3/3 ⊕ | 0.125 | **[+0.02, +1.64]** |
| C2 − C1, FT-10 / PTB-XL | +0.19 | +1.38 | +0.39 | **+0.65** | 3/3 ⊕ | 0.125 | [−0.93, +2.24] |
| C2 − C1, LP / PTB-XL | −0.10 | +1.06 | −0.34 | +0.21 | 1/3 | 0.500 | [−1.65, +2.07] |
| C2 − C1, LP / Chapman | −0.02 | +0.00 | −0.07 | −0.03 | 0/3 | 1.000 | [−0.12, +0.06] |
| C1 − B0, LP / PTB-XL | −6.03 | −7.32 | −4.24 | **−5.86** | 3/3 ⊖ | 0.125 (neg.) | [−9.71, −2.02] |
| C1 − B0, FT-10 / PTB-XL | −0.92 | −2.28 | −1.89 | **−1.70** | 3/3 ⊖ | 0.125 (neg.) | [−3.44, +0.04] |

*Wording rule:* with three seeds the exact permutation p cannot fall below 0.125; results are therefore described as seed-consistent (⊕/⊖) and never as statistically significant. The CI excluding zero for CPSC2018 is the strongest seed-level statement available.

**Table 6. Record-level paired bootstrap on macro-AUROC (10,000 synchronized resamples over test records and seeds 0/2/4; two-sided p; descriptive).**

| Domain | Pair | Δ (points) | 95% CI | p | Interpretation |
|---|---|---:|---|---:|---|
| CPSC2018 | C2 − C1 | +0.10 | [−0.05, +0.25] | 0.19 † | Seed-consistent, not record-level resolved |
| PTB-XL | C2 − C1 | +0.12 | [−0.23, +0.48] | 0.484 | No material change (LP) |
| Chapman | C2 − C1 | −0.01 | [−0.03, +0.01] | 0.494 | No material change |
| CPSC2018 | C1 − SimCLR | **+3.13** | [+2.39, +3.95] | <0.0001 † | LRRE > SimCLR |
| Chapman | C1 − SimCLR | **+0.88** | [+0.68, +1.11] | <0.0001 | LRRE > SimCLR |
| PTB-XL | C1 − SimCLR | +0.57 | [−0.39, +1.54] | 0.247 | Positive, not resolved |
| PTB-XL | C1 − CLOCS | **+1.59** | [+0.87, +2.35] | <0.0001 | LRRE > CLOCS |
| CPSC2018 | C1 − CLOCS | **+0.45** | [+0.16, +0.76] | 0.004 † | LRRE > CLOCS |
| Chapman | C1 − CLOCS | +0.04 | [−0.03, +0.12] | 0.275 | Parity near ceiling |
| PTB-XL | C2 − B0 | **−2.77** | [−3.57, −2.01] | <0.0001 | In-domain anchor better |
| CPSC2018 | C2 − B0 | −0.33 | [−0.71, +0.02] | 0.070 † | Slightly below anchor, not resolved |
| Chapman | C2 − B0 | **+0.31** | [+0.21, +0.43] | <0.0001 | Lineage advantage |

*Note:* Non-CPSC2018 rows are the audited W4 statistics on the original partitions. † CPSC2018 rows are provisional recomputations on the corrected partition (10,000 synchronized record-level resamples over seeds 0/2/4; RNG 20260925) from the re-evaluated per-record predictions; independent recomputation is pending. All rows remain record-level and descriptive.

### 4.3. Comparison with External Baselines

Table 7 places the LRRE family against the matched-budget external references. The LRRE control C1 exceeded SimCLR on all six linear-probe comparisons and exceeded CLOCS on PTB-XL and CPSC2018, with parity on Chapman where all methods approach ceiling (macro-AUROC ≥ 0.987); record-level significance holds for PTB-XL (CLOCS) and Chapman (SimCLR), and provisionally for both CPSC2018 comparisons (Table 6 note). Under 10%-label fine-tuning, C1 exceeded SimCLR and CLOCS by 1.94 and 0.26 macro-AUPRC points on PTB-XL, and by 9.5 and 1.2 points on the corrected CPSC2018 partition. These references anchor the family's position among established methods under a no-search, same-budget policy; they are not a leaderboard claim against recent large-corpus systems, whose pretraining budgets and corpora are not comparable (Section 5).

**Table 7. External baselines under matched budgets (mean ± sample SD over seeds 0/2/4).**

| Eval / domain | Metric | SimCLR (S1) | CLOCS (S2) | Supervised full-label (S3) | C1 | C2 (LRTC-Net) |
|---|---|---:|---:|---:|---:|---:|
| LP / PTB-XL | AUROC | 0.8791 ± 0.0009 | 0.8690 ± 0.0029 | — | 0.8848 ± 0.0056 | 0.8860 ± 0.0016 |
| LP / PTB-XL | AUPRC | 0.6440 ± 0.0047 | 0.6279 ± 0.0019 | — | 0.6533 ± 0.0107 | 0.6554 ± 0.0032 |
| LP / CPSC2018 | AUROC | 0.9150 ± 0.0012 | 0.9418 ± 0.0016 | — | 0.9463 ± 0.0016 | 0.9474 ± 0.0022 |
| LP / CPSC2018 | AUPRC | 0.6651 ± 0.0083 | 0.7475 ± 0.0067 | — | 0.7600 ± 0.0024 | 0.7683 ± 0.0056 |
| LP / Chapman | AUROC | 0.9876 ± 0.0005 | 0.9960 ± 0.0002 | — | 0.9964 ± 0.0001 | 0.9963 ± 0.0003 |
| LP / Chapman | AUPRC | 0.9621 ± 0.0013 | 0.9879 ± 0.0007 | — | 0.9889 ± 0.0002 | 0.9886 ± 0.0004 |
| FT-10 / PTB-XL | AUROC | 0.8500 ± 0.0029 | 0.8542 ± 0.0028 | — | 0.8544 ± 0.0060 | 0.8597 ± 0.0038 |
| FT-10 / PTB-XL | AUPRC | 0.6085 ± 0.0025 | 0.6253 ± 0.0041 | — | 0.6279 ± 0.0053 | 0.6345 ± 0.0049 |
| FT-100 / PTB-XL | AUROC | — | — | 0.8777 ± 0.0080 | — | — |
| FT-100 / PTB-XL | AUPRC | — | — | 0.6497 ± 0.0168 | — | — |

*Note:* CPSC2018 rows are corrected-partition re-evaluations (Section 3.2); all other rows are unchanged from the audited ledgers.

### 4.4. Pretraining Corpus, Data Lineage, and Label Efficiency

The B0 anchor quantifies the corpus effect. External pretraining without TRC cost 5.86 macro-AUPRC points on the PTB-XL linear probe (C1 − B0; three of three seeds negative; CI [−9.71, −2.02]) but only 1.70 points under 10%-label fine-tuning (CI [−3.44, +0.04]): fine-tuning with a small labeled target sample narrows the corpus gap by a factor of 3.4. On CPSC2018, C2 remained 0.49 macro-AUPRC points below the in-domain anchor (record-level macro-AUROC −0.33 points, 95% CI [−0.71, +0.02]; p = 0.070, provisional; not resolved) despite never seeing either corpus's labels—external pretraining recovers most, but not all, of the in-domain advantage on a third domain; on Chapman, both C1 and C2 exceeded the anchor by roughly +0.3 AUROC points (p < 0.0001), which we attribute to the shared acquisition team and device lineage between the NFH pretraining corpus and the Chapman evaluation corpus rather than to generic transfer, consistent with the corpus-disjoint record-ID ranges being an incomplete guarantee of distributional independence.

Label efficiency follows directly. With 10% of the downstream labels, LRTC-Net reached a macro-AUPRC of 0.6345, i.e., 97.7% of the fully supervised reference trained from random initialization on 100% of the labels (0.6497), and a macro-AUROC of 0.8597 (98.0% of 0.8777). The in-domain anchor with the same 10% budget reached 0.6449, so external pretraining closes most—but not all—of the gap to in-domain pretraining when a small labeled target sample is available.

### 4.5. Robustness under Simulated Lead Dropout

Table 8 summarizes macro-AUPRC degradation under the 37 zero-masking conditions on the PTB-XL linear probe, using the same checkpoints as Table 4. External pretraining retained a smaller mean degradation than the in-domain anchor—C2 dropped −0.59 points per single dropped lead and −1.41 per dropped pair, versus −0.83 and −1.88 for B0—and the most informative double-lead condition, simultaneous absence of the inferior limb leads II and III, degraded C2 by −3.79 points versus −6.13 for B0. No condition produced an absolute crossover above the complete-input performance of the in-domain anchor, and the C2-versus-C1 per-condition comparison was non-systematic (C2 better in 20 of 36 dropout conditions), so the robustness statement is about preserved degradation profiles of external pretraining, not about TRC adding robustness and not about absolute superiority. Per-lead sensitivity followed clinical expectations, with the largest single-lead drops for leads III and V2 and the smallest for V5/V6.

**Table 8. Macro-AUPRC drop from complete input under lead zero-masking (points).**

| Condition | B0 (in-domain) | C1 (external) | C2 = LRTC-Net |
|---|---:|---:|---:|
| Mean over 8 single-lead conditions | −0.83 | −0.60 | −0.59 |
| Mean over 28 double-lead conditions | −1.88 | −1.47 | −1.41 |
| Worst informative pair (II + III) | −6.13 | −3.64 | −3.79 |
| Precordial pair (V5 + V6) | −2.48 | −0.93 | −0.97 |

*Note:* B0 and C1 entries derive from the seed-0 anchor evaluation; C2 entries are means over seeds 0/2/4. Range across single-lead conditions for C2: −1.83 to +0.41; across double-lead conditions: −4.68 to +0.36.

### 4.6. Parameter and Compute Overhead

Measured from the frozen encoder checkpoints, C1 contains 634,280 parameters and C2 contains 635,304; TRC therefore adds exactly 1,024 parameters (0.161% of the encoder). Counting the eight pretraining projection heads, the pretraining models contain 68,848,704 and 68,849,728 parameters, respectively (+0.0015%). C1 and C2 consumed the same number of optimizer updates (≈27,270) with a wall-clock difference below 1%, and the downstream heads add a single linear layer (512 → number of classes). The gains reported in Section 4.2—+0.83 macro-AUPRC points on third-domain transfer and +0.65 on 10%-label fine-tuning—were obtained at this overhead.

![Figure 1. LRTC-Net training, transfer, and three-domain evaluation workflow.](figures/figure_1_three_domain_workflow.png)

**Figure 1. LRTC-Net training, transfer, and three-domain evaluation workflow.** Unlabeled eight-lead ECGs from the external NFH corpus are processed by an LRRE encoder containing TRC, jointly optimized during self-supervised pretraining. The pretrained encoder is transferred to three downstream domains (PTB-XL linear probe and 10%-label fine-tuning; CPSC2018 and Chapman linear probes), with an optimizer-update-matched in-domain anchor (B0) and matched-budget external baselines (SimCLR, CLOCS, supervised-direct). Evaluation covers primary performance, seed-level paired deltas with preregistered gates, record-level paired statistics, controlled lead zero-masking, and parameter overhead.

![Figure 2. ECG signal preprocessing and controlled lead-dropout construction.](figures/figure_2_preprocessing_missing_leads_replaced.png)

**Figure 2. ECG signal preprocessing and simulated lead-dropout workflow.** Twelve-lead records from all corpora undergo eight-lead selection (II, III, V1–V6), frequency-domain resampling to 2,048 samples, and record-wise global normalization to form [8, 2048] inputs. For lead-dropout evaluation, the selected lead is zero-masked over the full time range after normalization and suppressed by a valid-lead mask after encoding.

![Figure 3. TRC local structure and downstream classification path.](figures/figure_3_trc_structure_replaced.png)

**Figure 3. TRC local structure and downstream classification path.** Each lead branch produces a [B, 64, 64] feature map after the fifth convolutional block. TRC computes channel-wise L2 responses along time, normalizes them across channels within each record, and applies zero-initialized residual channel modulation (Eq. 17). Representations are globally average-pooled and concatenated in fixed lead order into a 512-dimensional vector for the downstream head.

![Figure 4. Macro-AUPRC under complete input and simulated lead dropout.](figures/figure_4_lead_dropout.png)

**Figure 4. Macro-AUPRC degradation under simulated lead dropout on the PTB-XL linear probe.** (a) Single-lead zero-masking: Δ macro-AUPRC relative to each model's complete-input performance for the eight leads (B0 and C1: seed 0; C2: mean over seeds 0/2/4, error bars show the seed range). (b) The 28 double-lead conditions ordered by the in-domain anchor's degradation; the worst informative pair (II+III) and the precordial pair (V5+V6) are marked. External pretraining retains a flatter degradation profile than the in-domain anchor, and no condition crosses the anchor's complete-input performance.

## 5. Discussion

### 5.1. Principal Findings

Under a preregistered, protocol-matched comparison with matched optimizer updates, adding TRC to a lead-aware SSL encoder during external-corpus pretraining produced seed-consistent gains where the motivating hypothesis predicted them: +0.83 macro-AUPRC points on the third-domain CPSC2018 probe (three of three seeds; seed-level CI [+0.02, +1.64] excluding zero narrowly; record-level difference not resolved, p = 0.19) and +0.65 points on PTB-XL 10%-label fine-tuning (three of three seeds). The gains were absent on the in-domain linear probe and on the lineage-related Chapman domain. Because the encoder, corpus, augmentations, updates, splits, and downstream protocol were identical between C1 and C2, these coordinated results isolate the incremental contribution of a 1,024-parameter, zero-initialized channel-response recalibration. The interpretation is mechanistically coherent: when the pretraining corpus differs statistically from the target domain, the learned channel-response weighting of high-level temporal features is miscalibrated for the target, and a residual, per-channel re-weighting learned during pretraining partially corrects this—directionally most visibly where the shift is largest (CPSC2018), absent where the target matches the corpus statistics that fine-tuning can already adapt (PTB-XL probe), and unmeasurable near ceiling (Chapman). The magnitude of the third-domain component should be read with its evidentiary status in mind: seed-consistent and gate-passing, but not resolved at record level.

### 5.2. The Corpus Effect and Its Interaction with the Module

The B0 anchor shows that external pretraining is not free: it cost 5.86 linear-probe macro-AUPRC points on PTB-XL relative to update-matched in-domain pretraining. Two practical consequences follow. First, 10%-label fine-tuning narrows this gap by a factor of 3.4 (to −1.70 points), so the deployment-relevant cost of external pretraining is much smaller than its probe-level cost; second, on the third domain, the external-pretrained model with TRC closed most of the gap to the in-domain anchor (0.49 macro-AUPRC points below; record-level macro-AUROC −0.33 points, p = 0.070, provisional; not resolved) despite training on neither corpus's labels—external pretraining buys breadth, TRC recovers a substantial part of what it loses in specificity. The Chapman domain illustrates the lineage confound: the +0.31-point advantage over the anchor there is better explained by shared acquisition team and devices between the NFH corpus and the Chapman evaluation set than by transfer, and we therefore do not describe it as cross-hospital generalization.

### 5.3. Position among External Methods and the Role of the Evaluation Discipline

Against matched-budget references, the LRRE family exceeded SimCLR and CLOCS on five of six linear-probe comparisons (four significant at record level) and by 3.0–4.5 macro-AUPRC points under 10%-label fine-tuning, consistent with the general finding that generic contrastive objectives trail ECG-specific multi-lead designs on this class of tasks. We deliberately restricted this comparison to same-budget, no-tuning, same-backbone-family references; recent systems pretrained on larger multi-source corpora are outside its scope and are not claimed against.

The evaluation discipline is itself part of the contribution. With a strong anchor whose cross-seed spread is roughly 0.8 macro-AUPRC points, single-run gains of 0.2–0.4 points are indistinguishable from noise; a companion study documents that a candidate gaining +0.27 points at seed 0 reversed sign at seeds 2 and 4, and that more than 40 modification families explored around the same anchor—including objective rewrites, architectural add-ons, reconstruction auxiliaries, and corpus mixing—closed negative or indistinguishable under the same gates. TRC is the sole survivor of that closed search space, which is the evidentiary context in which its small but consistently located gains should be read: not as the best of a garden of working modules, but as the only modification that survived a preregistered filter. The data-lineage audit reinforced the same lesson at the partition level: a preparation flaw that had silently enlarged the CPSC2018 test partition inflated the apparent third-domain gain from +0.83 to +2.30 points, and the discrepancy was caught only by reconciling per-record prediction counts against the preparation manifest (Section 3.2)—a direct demonstration that partition integrity moves results at least as much as module identity.

### 5.4. Limitations

Several limitations bound these conclusions. All results rest on three seeds; the exact permutation floor is p = 0.125, so seed-level statements are sign-consistency claims, and the record-level bootstrap—while resolved—is descriptive, treats records rather than patients as units, and does not model seed-level variance fully. CPSC2018 and Chapman-Shaoxing were split record-level stratified 70/10/20 rather than by official patient partitions, and the NFH corpus provides no patient identifiers (content-hash duplicate auditing only); no patient-level statistical claim is made. All CPSC2018 results derive from an audited, corrected test partition (Section 3.2), and their record-level statistics are provisional pending independent recomputation. Lead dropout was simulated by zero-masking with post-encoding suppression, which is not electrode displacement, lead reversal, or signal degradation. The pretraining corpus is a single external center; corpus identity moves results by magnitudes comparable to module effects, so the TRC gains should not be assumed to replicate verbatim under different external corpora without the same protocol. Label efficiency was evaluated at the 10% fraction (plus the fully supervised reference) rather than a full label-efficiency curve. Finally, deployment characteristics—latency, memory, FLOPs, probability calibration, and prospective clinical utility—were not evaluated.

### 5.5. Future Work

Priorities include: re-running the confirmatory matrix on additional external pretraining corpora to test whether the CPSC-pattern of TRC gains tracks distributional distance; full label-fraction curves and repeated label subsampling under the final protocol; patient-level splits and clustered statistics where identifiers permit; realistic lead-availability perturbations (lead reversal, noise, partial disconnection) beyond zero-masking; and mechanistic analysis of the learned γ/β distributions and channel-response statistics to explain why recalibration transfers. The closed modification map in the companion paper defines the axes that do not warrant further single-point sweeps under this protocol.

## 6. Conclusions

LRTC-Net embeds a zero-initialized, channel-wise Temporal Response Calibration module—1,024 parameters, 0.161% of the encoder—into each branch of a lead-aware redundancy-reduction encoder and jointly optimizes it during self-supervised pretraining on an external hospital corpus of 34,905 eight-lead ECGs. Under a preregistered three-seed protocol with matched optimizer updates and a protocol-matched control, TRC improved third-domain transfer (CPSC2018 linear probe: +0.83 macro-AUPRC points, three of three seeds, seed-level CI [+0.02, +1.64]; record-level difference not resolved) and 10%-label fine-tuning on PTB-XL (+0.65 points, three of three seeds), while leaving the in-domain probe and the lineage-related domain unchanged. With 10% of downstream labels, the model reached 97.7% of the macro-AUPRC of a fully supervised reference trained on all labels, retained the smaller lead-zero-masking degradation of external pretraining (−0.59/−1.41 points per dropped lead/pair versus −0.83/−1.88 for the in-domain anchor), and closed most of the update-matched in-domain anchor's third-domain advantage (0.49 macro-AUPRC points below, not seed-consistent). The corpus effect itself was quantified: external pretraining costs 5.86 linear-probe points in-domain, a gap that limited-label fine-tuning narrows 3.4-fold. These results, together with the companion negative-results program in which TRC was the sole modification to pass the preregistered gates, characterize a minimal, mechanism-targeted recalibration whose benefits concentrate exactly where domain shift is largest—within the explicit scope of the evaluated corpora, splits, and protocols, and without claims of cross-hospital generalization, patient-level significance, or robustness to electrode displacement.

## Author Contributions

[To be completed.]

## Funding

[To be completed: if none, state "This research received no external funding."]

## Institutional Review Board Statement

[To be completed: all corpora are public and de-identified; confirm "Not applicable" or supply approval details.]

## Informed Consent Statement

[To be completed.]

## Data Availability Statement

PTB-XL, CPSC2018, and the Chapman-Shaoxing/Ningbo collection are publicly available through PhysioNet. The per-record experimental ledgers, manifests, checkpoint hashes, and per-record predictions supporting every number in this paper are described in the project repository. [To be completed: repository URL and access statements.]

## Conflicts of Interest

[To be completed.]

## Acknowledgments

[To be completed.]

## Supplementary Materials

**Supplementary Table S1. PTB-XL data filtering workflow.**

| Fold/range | Original records | Records with signals | Records with mappable labels | Multi-superclass exclusions | Final retained |
|---|---:|---:|---:|---:|---:|
| Folds 1–8 (pretraining) | 17,418 | 17,418 | — | — | 17,418 |
| Folds 1–8 total | 17,418 | 17,418 | 17,418 | 3,779 | 13,639 |
| Fold 9 | 2,183 | 2,183 | 2,183 | 469 | 1,714 |
| Fold 10 | 2,198 | 2,198 | 2,198 | 459 | 1,739 |

**Supplementary Table S2. Per-seed values underlying Tables 4 and 7.** (macro-AUROC / macro-AUPRC per seed; sources: `runlog/W2/confirm_results.csv`, `runlog/W1/c3_evals.csv`, `runlog/W4/baseline_results.csv`, and `runlog/W5/lp_results.csv` for the corrected-partition CPSC2018 rows.)

| Config | Eval / domain | s0 | s2 | s4 |
|---|---|---|---|---|
| B0 | LP / PTB-XL | 0.9126 / 0.7177 | 0.9152 / 0.7144 | 0.9135 / 0.7037 |
| C1 | LP / PTB-XL | 0.8907 / 0.6574 | 0.8795 / 0.6412 | 0.8842 / 0.6613 |
| C2 | LP / PTB-XL | 0.8860 / 0.6564 | 0.8845 / 0.6518 | 0.8876 / 0.6579 |
| B0 | FT-10 / PTB-XL | 0.8756 / 0.6433 | 0.8706 / 0.6474 | 0.8679 / 0.6440 |
| C1 | FT-10 / PTB-XL | 0.8604 / 0.6341 | 0.8484 / 0.6246 | 0.8544 / 0.6251 |
| C2 | FT-10 / PTB-XL | 0.8627 / 0.6360 | 0.8611 / 0.6384 | 0.8554 / 0.6290 |
| B0 | LP / CPSC2018 | 0.9506 / 0.7775 | 0.9501 / 0.7708 | 0.9512 / 0.7712 |
| C1 | LP / CPSC2018 | 0.9455 / 0.7589 | 0.9453 / 0.7583 | 0.9481 / 0.7627 |
| C2 | LP / CPSC2018 | 0.9455 / 0.7659 | 0.9468 / 0.7642 | 0.9498 / 0.7747 |
| B0 | LP / Chapman | 0.9932 / 0.9796 | 0.9925 / 0.9774 | 0.9939 / 0.9816 |
| C1 | LP / Chapman | 0.9964 / 0.9891 | 0.9964 / 0.9888 | 0.9965 / 0.9888 |
| C2 | LP / Chapman | 0.9966 / 0.9889 | 0.9964 / 0.9888 | 0.9960 / 0.9881 |

**Supplementary Table S3. Full record-level paired statistics** (macro-AUROC; 10,000 synchronized record-level bootstrap resamples over test records and seeds 0/2/4; two-sided p; per-class DeLong at seed 0 as auxiliary; descriptive, record-level).

| Domain | Comparison | Seeds | n (records) | AUROC (A) | AUROC (B) | Δ (pt) | 95% CI | p (bootstrap) | DeLong p min / median (seed 0) | Classes |
|---|---|---|---:|---:|---:|---:|---|---:|---|---:|
| PTB-XL | C1 (LRRE) vs C2 (LRTC-Net) | 0;2;4 | 1,739 | 0.8848 | 0.8860 | −0.12 | [−0.48, +0.23] | 0.4842 | 0.2344 / 0.3456 | 5 |
| Chapman | C1 (LRRE) vs C2 (LRTC-Net) | 0;2;4 | 2,047 | 0.9964 | 0.9963 | +0.01 | [−0.01, +0.03] | 0.4934 | 0.0000 / 0.5807 | 4 |
| CPSC2018 † | C1 (LRRE) vs C2 (LRTC-Net) | 0;2;4 | 1,385 | 0.9463 | 0.9474 | −0.10 | [−0.25, +0.05] | 0.1876 | — / — | 9 |
| PTB-XL | C1 (LRRE) vs S1 (SimCLR) | 0;2;4 | 1,739 | 0.8848 | 0.8791 | +0.57 | [−0.39, +1.54] | 0.2472 | 0.0072 / 0.1300 | 5 |
| Chapman | C1 (LRRE) vs S1 (SimCLR) | 0;2;4 | 2,047 | 0.9964 | 0.9876 | +0.88 | [+0.68, +1.11] | <0.0001 | 0.0000 / 0.0004 | 4 |
| CPSC2018 † | C1 (LRRE) vs S1 (SimCLR) | 0;2;4 | 1,385 | 0.9463 | 0.9150 | +3.13 | [+2.39, +3.95] | <0.0001 | — / — | 9 |
| PTB-XL | C1 (LRRE) vs S2 (CLOCS) | 0;2;4 | 1,739 | 0.8848 | 0.8689 | +1.59 | [+0.87, +2.35] | <0.0001 | 0.0000 / 0.0012 | 5 |
| Chapman | C1 (LRRE) vs S2 (CLOCS) | 0;2;4 | 2,047 | 0.9964 | 0.9960 | +0.04 | [−0.03, +0.12] | 0.2750 | 0.0592 / 0.0711 | 4 |
| CPSC2018 † | C1 (LRRE) vs S2 (CLOCS) | 0;2;4 | 1,385 | 0.9463 | 0.9418 | +0.45 | [+0.16, +0.76] | 0.0038 | — / — | 9 |
| PTB-XL | C2 (LRTC-Net) vs B0 (anchor) | 0;2;4 | 1,739 | 0.8860 | 0.9138 | −2.77 | [−3.57, −2.01] | <0.0001 | 0.0000 / 0.0179 | 5 |
| Chapman | C2 (LRTC-Net) vs B0 (anchor) | 0;2;4 | 2,047 | 0.9963 | 0.9932 | +0.31 | [+0.21, +0.43] | <0.0001 | 0.0000 / 0.0283 | 4 |
| CPSC2018 † | C2 (LRTC-Net) vs B0 (anchor) | 0;2;4 | 1,385 | 0.9474 | 0.9506 | −0.33 | [−0.70, +0.02] | 0.0696 | — / — | 9 |

*Notes.* (1) Δ = AUROC(A) − AUROC(B), in points (×100). PTB-XL and Chapman rows are the audited W4 statistics on the original partitions (source: `runlog/W4/stats/paired_stats.csv`; method details in `runlog/W4/stats/stats_methods.md`). (2) † CPSC2018 rows use the corrected test partition (Section 3.2; n = 1,385) and the re-evaluated per-record predictions (`runlog/W5/predictions/`); their bootstrap and p values are provisional office recomputations (10,000 synchronized resamples, RNG 20260925) pending the official recomputation, and per-class DeLong tests are not yet available for these rows; macro-AUROC point estimates derive from `runlog/W5/lp_results.csv`. (3) All rows are record-level and descriptive; no patient-level significance is claimed, and per the preregistered wording rules, p values are supporting evidence only—the primary decision rule remains the three-seed sign-consistency gate (Section 3.1). (4) AUROC columns are three-seed means over seeds 0/2/4.

**Supplementary Table S4. Per-condition macro-AUPRC under simulated lead dropout by zero-masking** (PTB-XL linear probe; 37 conditions; C2 = LRTC-Net as three-seed means; B0 and C1 seed-0 anchor evaluation shown as context).

| Condition | Dropped lead(s) | C2 macro-AUPRC (3-seed mean) | C2 Δ vs complete (pt) | B0 Δ (pt, seed 0) | C1 Δ (pt, seed 0) |
|---|---|---:|---:|---:|---:|
| Complete input (reference) | — | 0.6554 | — | — | — |
| Single | II | 0.6482 | −0.72 | −1.52 | −0.65 |
| Single | III | 0.6425 | −1.28 | −1.81 | −1.02 |
| Single | V1 | 0.6454 | −1.00 | −1.55 | −1.05 |
| Single | V2 | 0.6425 | −1.29 | −0.44 | −0.96 |
| Single | V3 | 0.6522 | −0.32 | −0.25 | −0.52 |
| Single | V4 | 0.6548 | −0.05 | −0.06 | −0.02 |
| Single | V5 | 0.6560 | +0.06 | −0.59 | −0.30 |
| Single | V6 | 0.6540 | −0.13 | −0.40 | −0.27 |
| Double | II + III | 0.6174 | −3.79 | −6.13 | −3.64 |
| Double | II + V1 | 0.6370 | −1.84 | −3.68 | −1.52 |
| Double | II + V2 | 0.6347 | −2.06 | −2.28 | −2.04 |
| Double | II + V3 | 0.6446 | −1.07 | −1.98 | −1.21 |
| Double | II + V4 | 0.6468 | −0.86 | −1.41 | −0.80 |
| Double | II + V5 | 0.6485 | −0.69 | −1.81 | −1.08 |
| Double | II + V6 | 0.6433 | −1.20 | −2.25 | −1.31 |
| Double | III + V1 | 0.6263 | −2.91 | −3.69 | −2.35 |
| Double | III + V2 | 0.6315 | −2.39 | −2.34 | −2.47 |
| Double | III + V3 | 0.6397 | −1.57 | −2.07 | −1.46 |
| Double | III + V4 | 0.6407 | −1.47 | −1.91 | −1.31 |
| Double | III + V5 | 0.6411 | −1.42 | −2.07 | −1.45 |
| Double | III + V6 | 0.6413 | −1.41 | −2.55 | −1.44 |
| Double | V1 + V2 | 0.6230 | −3.24 | −2.61 | −3.00 |
| Double | V1 + V3 | 0.6411 | −1.43 | −1.77 | −1.93 |
| Double | V1 + V4 | 0.6444 | −1.09 | −1.43 | −0.79 |
| Double | V1 + V5 | 0.6450 | −1.03 | −2.07 | −1.56 |
| Double | V1 + V6 | 0.6430 | −1.24 | −1.61 | −1.52 |
| Double | V2 + V3 | 0.6345 | −2.08 | −1.16 | −2.08 |
| Double | V2 + V4 | 0.6412 | −1.41 | −0.58 | −1.10 |
| Double | V2 + V5 | 0.6417 | −1.37 | −0.85 | −1.48 |
| Double | V2 + V6 | 0.6420 | −1.34 | −0.86 | −1.41 |
| Double | V3 + V4 | 0.6485 | −0.69 | −0.67 | −1.01 |
| Double | V3 + V5 | 0.6521 | −0.32 | −1.01 | −0.96 |
| Double | V3 + V6 | 0.6519 | −0.34 | −0.68 | −0.80 |
| Double | V4 + V5 | 0.6525 | −0.29 | −0.63 | −0.41 |
| Double | V4 + V6 | 0.6548 | −0.06 | −0.15 | −0.16 |
| Double | V5 + V6 | 0.6456 | −0.97 | −2.48 | −0.93 |

*Summary rows* (matching Table 8): mean over the eight single-lead conditions: C2 −0.59 pt (B0 −0.83, C1 −0.60); mean over the 28 double-lead conditions: C2 −1.41 pt (B0 −1.88, C1 −1.47).

*Notes.* (1) *Lead dropout* denotes controlled zero-masking of the selected normalized lead over all 2,048 samples with post-encoding suppression by the valid-lead mask; it is not electrode displacement, lead reversal, or signal degradation. (2) C2 values are means over seeds 0/2/4 of the same frozen checkpoints as the main-table linear probe; per-seed values are retained in `runlog/W2/missing_lead_c2.csv`. B0 and C1 columns derive from the seed-0 anchor evaluation (`runlog/W1/missing_lead_curves.csv`) and are shown as context only. (3) Per-seed C2 deltas across the 36 dropout conditions span −1.83 to +0.41 (single-lead) and −4.68 to +0.36 (double-lead). No condition produced an absolute crossover above the complete-input performance of the in-domain anchor, and the C2-versus-C1 per-condition comparison is non-systematic (C2 better in 20 of 36 conditions), consistent with Section 4.5. (4) Single-lead sensitivity follows clinical expectations in the aggregate—largest mean drops for leads III (−1.28) and V2 (−1.29), smallest for V5 (+0.06) and V4 (−0.05)—while the most informative double-lead absence is II + III (−3.79), the condition also most damaging for the in-domain anchor (−6.13).

## References

[1] Kligfield, P.; Gettes, L.S.; Bailey, J.J.; Childers, R.; Deal, B.J.; Hancock, E.W.; van Herpen, G.; Kors, J.A.; Macfarlane, P.; Mirvis, D.M.; Pahlm, O.; Rautaharju, P.M.; Wagner, G.S. Recommendations for the standardization and interpretation of the electrocardiogram: Part I: The electrocardiogram and its technology. *Circulation* **2007**, *115*, 1306-1324. https://doi.org/10.1161/CIRCULATIONAHA.106.180200.

[2] Wagner, P.; Strodthoff, N.; Bousseljot, R.-D.; Kreiseler, D.; Lunze, F.I.; Samek, W.; Schaeffter, T. PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data* **2020**, *7*, 154. https://doi.org/10.1038/s41597-020-0495-6.

[3] Lai, J.; Tan, H.; Wang, J.; Ji, L.; Guo, J.; Han, B.; Shi, Y.; Feng, Q.; Yang, W. Practical intelligent diagnostic algorithm for wearable 12-lead ECG via self-supervised learning on large-scale dataset. *Nature Communications* **2023**, *14*, 3741. https://doi.org/10.1038/s41467-023-39472-8.

[4] Satija, U.; Ramkumar, B.; Manikandan, M.S. A review of signal processing techniques for electrocardiogram signal quality assessment. *IEEE Reviews in Biomedical Engineering* **2018**, *11*, 36-52. https://doi.org/10.1109/RBME.2018.2810957.

[5] Zhang, X.; Li, J.; Cai, Z.; Zhao, L.; Liu, C. Deep learning-based signal quality assessment for wearable ECGs. *IEEE Instrumentation & Measurement Magazine* **2022**, *25*, 41-52. https://doi.org/10.1109/MIM.2022.9832823.

[6] Rjoob, K.; Bond, R.; Finlay, D.; McGilligan, V.; Leslie, S.J.; Rababah, A.; Iftikhar, A.; Guldenring, D.; Knoery, C.; McShane, A.; Peace, A. Reliable deep learning-based detection of misplaced chest electrodes during electrocardiogram recording: Algorithm development and validation. *JMIR Medical Informatics* **2021**, *9*, e25347. https://doi.org/10.2196/25347.

[7] Ribeiro, A.H.; Ribeiro, M.H.; Paixão, G.M.M.; Oliveira, D.M.; Gomes, P.R.; Canazart, J.A.; Ferreira, M.P.S.; Andersson, C.R.; Macfarlane, P.W.; Meira, W.; Schön, T.B.; Ribeiro, A.L.P. Automatic diagnosis of the 12-lead ECG using a deep neural network. *Nature Communications* **2020**, *11*, 1760. https://doi.org/10.1038/s41467-020-15432-4.

[8] Hannun, A.Y.; Rajpurkar, P.; Haghpanahi, M.; Tison, G.H.; Bourn, C.; Turakhia, M.P.; Ng, A.Y. Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network. *Nature Medicine* **2019**, *25*, 65-69. https://doi.org/10.1038/s41591-018-0268-3.

[9] Strodthoff, N.; Wagner, P.; Schaeffter, T.; Samek, W. Deep learning for ECG analysis: Benchmarks and insights from PTB-XL. *IEEE Journal of Biomedical and Health Informatics* **2021**, *25*, 1519-1528. https://doi.org/10.1109/JBHI.2020.3022989.

[10] Sepahvand, M.; Abdali-Mohammadi, F. A novel method for reducing arrhythmia classification from 12-lead ECG signals to single-lead ECG with minimal loss of accuracy through teacher-student knowledge distillation. *Information Sciences* **2022**, *593*, 64-77. https://doi.org/10.1016/j.ins.2022.01.030.

[11] Reznichenko, S.; Whitaker, J.; Ni, Z.; Zhou, S. Comparing ECG lead subsets for heart arrhythmia/ECG pattern classification: Convolutional neural networks and random forest. *CJC Open* **2025**, *7*, 176-186. https://doi.org/10.1016/j.cjco.2024.10.012.

[12] Chen, J.; Wu, W.; Liu, T.; Hong, S. Multi-channel masked autoencoder and comprehensive evaluations for reconstructing 12-lead ECG from arbitrary single-lead ECG. *npj Cardiovascular Health* **2024**, *1*, 34. https://doi.org/10.1038/s44325-024-00036-4.

[13] Presacan, O.; Dorobanţiu, A.; Isaksen, J.L.; Willi, T.; Graff, C.; Riegler, M.A.; Sridhar, A.R.; Kanters, J.K.; Thambawita, V. Evaluating the feasibility of 12-lead electrocardiogram reconstruction from limited leads using deep learning. *Communications Medicine* **2025**, *5*, 139. https://doi.org/10.1038/s43856-025-00814-w.

[14] Mehari, T.; Strodthoff, N. Self-supervised representation learning from 12-lead ECG data. *Computers in Biology and Medicine* **2022**, *141*, 105114. https://doi.org/10.1016/j.compbiomed.2021.105114.

[15] Kiyasseh, D.; Zhu, T.; Clifton, D.A. CLOCS: Contrastive learning of cardiac signals across space, time, and patients. In *Proceedings of the 38th International Conference on Machine Learning*; PMLR: 2021; Volume 139, pp. 5606-5615.

[16] Wei, C.-T.; Hsieh, M.-E.; Liu, C.-L.; Tseng, V.S. Contrastive heartbeats: Contrastive learning for self-supervised ECG representation and phenotyping. In *Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing*; IEEE: 2022; pp. 1126-1130. https://doi.org/10.1109/ICASSP43922.2022.9746887.

[17] Zbontar, J.; Jing, L.; Misra, I.; LeCun, Y.; Deny, S. Barlow Twins: Self-supervised learning via redundancy reduction. In *Proceedings of the 38th International Conference on Machine Learning*; PMLR: 2021; Volume 139, pp. 12310-12320.

[18] Liu, W.; Pan, S.; Li, Z.; Chang, S.; Huang, Q.; Jiang, N. Lead-fusion Barlow twins: A fused self-supervised learning method for multi-lead electrocardiograms. *Information Fusion* **2025**, *114*, 102698. https://doi.org/10.1016/j.inffus.2024.102698.

[19] Liu, W.; Pan, S.; Chang, S.; Huang, Q.; Jiang, N. Self-supervised learning for electrocardiogram classification using lead correlation and decorrelation. *Applied Soft Computing* **2025**, *172*, 112871. https://doi.org/10.1016/j.asoc.2025.112871.

[20] Hu, R.; Chen, J.; Zhou, L. Spatiotemporal self-supervised representation learning from multi-lead ECG signals. *Biomedical Signal Processing and Control* **2023**, *84*, 104772. https://doi.org/10.1016/j.bspc.2023.104772.

[21] Chen, W.; Wang, H.; Zhang, L.; Zhang, M. Temporal and spatial self-supervised learning methods for electrocardiograms. *Scientific Reports* **2025**, *15*, 6029. https://doi.org/10.1038/s41598-025-90084-2.

[22] Soltanieh, S.; Hashemi, J.; Etemad, A. In-distribution and out-of-distribution self-supervised ECG representation learning for arrhythmia detection. *IEEE Journal of Biomedical and Health Informatics* **2024**, *28*, 789-800. https://doi.org/10.1109/JBHI.2023.3331626.

[23] Yang, S.; Lian, C.; Zeng, Z.; Xu, B.; Su, Y.; Xue, C. Masked self-supervised ECG representation learning via multiview information bottleneck. *Neural Computing and Applications* **2024**, *36*, 7625-7637. https://doi.org/10.1007/s00521-024-09486-4.

[24] Wang, N.; Feng, P.; Ge, Z.; Zhou, Y.; Zhou, B.; Wang, Z. Adversarial spatiotemporal contrastive learning for electrocardiogram signals. *IEEE Transactions on Neural Networks and Learning Systems* **2024**, *35*, 13845-13859. https://doi.org/10.1109/TNNLS.2023.3272153.

[25] Ma, K.; Zhang, T.; Zhang, H.; Huang, W. Self-supervised contrastive learning achieves 12-lead ECG classification. *Biomedical Signal Processing and Control* **2026**, *112*, 108420. https://doi.org/10.1016/j.bspc.2025.108420.

[26] Xu, P.; Li, L.; Xie, X.; Lv, J.; Di, C.; Gu, X.; Chen, Y. Multi-stage temporal and cross-view contrastive learning for self-supervised multi-lead ECG classification. *Biomedical Signal Processing and Control* **2026**, *114*, 109317. https://doi.org/10.1016/j.bspc.2025.109317.

[27] Zhu, X.; Shi, M.; Yu, X.; Liu, C.; Lian, X.; Fei, J.; Luo, J.; Jin, X.; Zhang, P.; Ji, X. Self-supervised inter-intra period-aware ECG representation learning for detecting atrial fibrillation. *Biomedical Signal Processing and Control* **2025**, *100*, 106939. https://doi.org/10.1016/j.bspc.2024.106939.

[28] Li, Z.; Tian, Y.; Jin, Y.; Wei, X.; Wang, M.; Liu, J.; Zhao, L.; Liu, C. An early warning method for arrhythmias in long-term ECGs based on self-supervised learning and LSTM. *Knowledge-Based Systems* **2025**, *327*, 114137. https://doi.org/10.1016/j.knosys.2025.114137.

[29] Zhang, Y.; Li, X.; Zhang, L.; Wang, J.; Jiang, S.; Ma, Y.; Li, D. A sequential MAE-clustering self-supervised learning method for arrhythmia detection. *Expert Systems with Applications* **2025**, *269*, 126379. https://doi.org/10.1016/j.eswa.2025.126379.

[30] Weimann, K.; Conrad, T.O.F. Self-supervised pre-training with joint-embedding predictive architecture boosts ECG classification performance. *Computers in Biology and Medicine* **2025**, *196*, 110809. https://doi.org/10.1016/j.compbiomed.2025.110809.

[31] Hu, J.; Shen, L.; Sun, G. Squeeze-and-Excitation Networks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*; IEEE: 2018; pp. 7132-7141. https://doi.org/10.1109/CVPR.2018.00745.

[32] Wang, Q.; Wu, B.; Zhu, P.; Li, P.; Zuo, W.; Hu, Q. ECA-Net: Efficient channel attention for deep convolutional neural networks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*; IEEE: 2020; pp. 11531-11539. https://doi.org/10.1109/CVPR42600.2020.01155.

[33] Zeng, Y.; Lv, H.; Jiang, M.; Zhang, J.; Xia, L.; Wang, Y.; Wang, Z. Deep arrhythmia classification based on SENet and lightweight context transform. *Mathematical Biosciences and Engineering* **2023**, *20*, 1-17. https://doi.org/10.3934/mbe.2023001.

[34] Jin, Y.; Li, Z.; Qin, C.; Liu, J.; Liu, Y.; Zhao, L.; Liu, C. A novel attentional deep neural network-based assessment method for ECG quality. *Biomedical Signal Processing and Control* **2023**, *79*, 104064. https://doi.org/10.1016/j.bspc.2022.104064.

[35] Zhong, M.; Li, F.; Chen, W. Automatic arrhythmia detection with multi-lead ECG signals based on heterogeneous graph attention networks. *Mathematical Biosciences and Engineering* **2022**, *19*, 12448-12471. https://doi.org/10.3934/mbe.2022581.

[36] Chen, T.; Kornblith, S.; Norouzi, M.; Hinton, G. A simple framework for contrastive learning of visual representations. In *Proceedings of the 37th International Conference on Machine Learning*; PMLR: 2020; Volume 119, pp. 1597-1607.

[37] Liu, F.; Liu, C.; Zhao, L.; Zhang, X.; Wu, X.; Xu, X.; Liu, Y.; Ma, C.; Wei, S.; He, Z.; Li, J.; Ng, E.Y.K. An open access database for evaluating the algorithms of electrocardiogram rhythm and morphology abnormality detection. *Journal of Medical Imaging and Health Informatics* **2018**, *8*, 1368-1373. https://doi.org/10.1166/jmihi.2018.2442.

[38] Zheng, J.; Zhang, J.; Danioko, S.; Yao, M.; Guo, H.; Rakovski, C. A 12-lead electrocardiogram database for arrhythmia research: The Chapman-Shaoxing and Ningbo Database. *Computing in Cardiology* **2020**, *47*. https://doi.org/10.22489/CinC.2020.083.

[39] Woo, S.; Debnath, S.; Hu, R.; Chen, X.; Liu, Z.; Kweon, I.S.; Xie, S. ConvNeXt V2: Co-designing and scaling ConvNets with masked autoencoders. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*; IEEE: 2023; pp. 16133-16142. https://doi.org/10.1109/CVPR52729.2023.01561.

[40] Musgrave, K.; Belongie, S.; Lim, S.-N. A metric learning reality check. In *Proceedings of the European Conference on Computer Vision (ECCV)*; Springer: 2020; pp. 681-699.

[41] Recht, B.; Roelofs, R.; Schmidt, L.; Shankar, V. Do ImageNet classifiers generalize to ImageNet? In *Proceedings of the 36th International Conference on Machine Learning*; PMLR: 2019; pp. 5389-5400.
