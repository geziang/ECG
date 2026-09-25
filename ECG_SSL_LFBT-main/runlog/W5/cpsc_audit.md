# W5 A-1: CPSC test 残留审计 (cpsc_audit.md)

- 执行: 主机A, 2026-09-25 10:0x (任务书 W5A-主机A-CPSC清查与重评估任务书-2026-09-24.md §①)
- 审计对象: `data/cpsc/` (ECG_SSL_LFBT-main 仓库根)
- 工作目录: `F:\新实验\ECG_SSL_LFBT-main`, 命令原文照录, 输出完整保留

## 1. 四条审计命令与完整输出

### 命令1: 三个 split 总数

```powershell
(Get-ChildItem data\cpsc\test  -Recurse -Filter *.npy).Count
(Get-ChildItem data\cpsc\train -Recurse -Filter *.npy).Count
(Get-ChildItem data\cpsc\val   -Recurse -Filter *.npy).Count
```

输出:

```
test:  2004
train: 5602
val:   1028
```

### 命令2: test 分类计数

```powershell
Get-ChildItem data\cpsc\test -Directory | ForEach-Object { "{0} {1}" -f $_.Name, (Get-ChildItem $_.FullName -Filter *.npy).Count }
```

输出:

```
AF    245
IAVB  239
LBBB  68
NSR   185
PAC   193
PVC   135
RBBB  563
STD   296
STE   80
```

合计 2004。

### 命令3: 跨类重复 stem

```powershell
Get-ChildItem data\cpsc\test -Recurse -Filter *.npy | Group-Object Name | Where-Object Count -gt 1 | ForEach-Object { $_.Name }
```

输出:

```
A0308.npy
A2877.npy
```

(共 2 个重复 stem)

### 命令4: 时间戳分层 (LastWriteTime, 按小时)

```powershell
Get-ChildItem data\cpsc\test -Recurse -Filter *.npy | Group-Object { $_.LastWriteTime.ToString("yyyy-MM-dd HH") } | Sort-Object Name | Format-Table Name,Count
```

输出:

```
Name          Count
----          -----
2026-09-21 04  2004
```

(单一小时分层; 分钟级证据见 §2)

## 2. 补充取证: 重复 stem 的落点与分钟级时间戳

```
A0308.npy  RBBB  2026-09-21 04:50:41
A2877.npy  RBBB  2026-09-21 04:50:42
A0308.npy  PVC   2026-09-21 04:55:05
A2877.npy  PVC   2026-09-21 04:55:05
```

同一 stem 在 RBBB(04:50) 与 PVC(04:55) 两个类目录各有一份, 写入时刻相差约 4.5 分钟
—— 小时分层为单批, 但分钟级呈现两个写入波次。

## 3. 对照 W1 manifest (runlog/W1/cpsc_manifest.json, csc-v1)

| 类 | manifest.test | 实测 | 差值 |
|----|----|----|----|
| NSR | 185 | 185 | 0 |
| AF | 245 | 245 | 0 |
| IAVB | 138 | 239 | +101 |
| LBBB | 39 | 68 | +29 |
| RBBB | 309 | 563 | +254 |
| PAC | 121 | 193 | +72 |
| PVC | 135 | 135 | 0 |
| STD | 168 | 296 | +128 |
| STE | 45 | 80 | +35 |
| **合计** | **1385** | **2004** | **+619** |

6 个类多出 619 个文件, 1385+619=2004 对账吻合; NSR/AF/PVC 三类与 manifest 逐个一致。
A0308/A2877 均带 {RBBB, PVC} 双码, 恰落在两个有增差的类里。

## 4. 判定 (按任务书判定表)

> 总数 ≈2004 且有重复 stem 或时间戳两批 → 判"残留"

实测: **总数 = 2004 (精确命中) 且存在 2 个跨类重复 stem** → **判定: 残留**, 进入 §② A-2 重评估。

## 5. 机制分析 (供复盘, 不影响判定)

`prepare_cpsc_chapman.py` 写出时 `mkdir(exist_ok=True)` 且逐文件 `np.save`, **不清理既有目录**。
04:50 与 04:55 两个写入波次 + 标签映射修订痕迹(CPSC_MAP 中 `164884008=VEB: cpsc_2018 打包版
PVC 实际用码` 为后补注释)与以下过程一致: 2026-09-21 04 时同一小时内以新旧两版标签映射先后
运行, 第二次运行只增写/覆盖同名文件, 第一次运行独有 stem 的文件全部残留, 磁盘成为两次
标注的并集(2004), 而 manifest 只反映最后一次运行(1385)。
A0308/A2877 两次运行被映射到不同类(RBBB→PVC 优先级/PVC 用码变动), 旧份未删, 形成跨类重复。

**推论**: W2/W4 期间所有 CPSC LP/FT 评测实际读取的是这份 2004 并集 test(含 619 个多出的
文件与 2 个跨类重复记录), 故 CPSC 统计行全部待重算(B 机知会项); PTB/Chapman 不受影响
(本次审计未涉及, 亦不重跑——任务书红线)。

## 6. 后续

- 旧目录整体改名备份: `data/cpsc` → `data/cpsc_bak_2004` (保留证据, 不删除)
- 重跑 `python prepare_cpsc_chapman.py --which cpsc --manifest-dir runlog/W5`, 新 manifest 落 `runlog/W5/cpsc_manifest.json`
- 自检新 test 总数 ≈ 1385 后执行 15 跑 CPSC LP 重评估 (b0/c1/c2/simclr/clocs × seeds{0,2,4})

## 7. A-2 执行记录 (2026-09-25 10:0x–10:39, 主机A)

1. **备份**: `data/cpsc` → `data/cpsc_bak_2004` (gitignore 内, 仅本地磁盘保留)。
2. **重跑 prepare** (conda env DL, 95.1s): 6877 全 ok / 0 剔除, 分类计数与 W1 manifest(csc-v1)
   逐类一致; 新 manifest = `runlog/W5/cpsc_manifest.json`。
   **磁盘自检: test=1385, train=4809, val=683, 跨类重复 stem=0** ✓
3. **checkpoint SHA256 反查**: 15/15 与 freeze_manifest_v2 + W4 baseline_results 逐位一致
   (b0 s0 = `checkpoint/ptxl_gamma08` 即 W1 锚点, 明细 `runlog/W5/ckpt_sha_check.json`)。
4. **15 跑 CPSC LP 重评估** (`runlog/W5/run_w5_cpscredo.py`, 协议与 W2/W4 LP 同构:
   run_lp.py 默认超参 100ep/bs128/lr1e-3, c2 `--trc 1` 其余 `--trc 0`,
   `--save-predictions runlog/W5/predictions/`, `--protocol-id w5-cpscredo`):
   15/15 完成, 账本 = `runlog/W5/lp_results.csv`(15 行, 含 git_sha/checkpoint_sha256),
   预测 15 套齐全(y_prob (1385,9))。

### 新 test(1385) LP 结果 (macro, mean±SD over seeds{0,2,4})

| ckpt | AUROC (s0/s2/s4) | AUROC mean±SD | AUPRC (s0/s2/s4) | AUPRC mean±SD |
|---|---|---|---|---|
| b0 | 0.9506/0.9501/0.9512 | 0.9506±0.0006 | 0.7775/0.7708/0.7712 | 0.7732±0.0038 |
| c1 | 0.9455/0.9453/0.9481 | 0.9463±0.0016 | 0.7589/0.7583/0.7627 | 0.7600±0.0024 |
| c2 | 0.9455/0.9468/0.9498 | 0.9474±0.0022 | 0.7659/0.7642/0.7747 | 0.7683±0.0056 |
| simclr | 0.9164/0.9142/0.9143 | 0.9150±0.0012 | 0.6728/0.6563/0.6661 | 0.6651±0.0083 |
| clocs | 0.9401/0.9421/0.9433 | 0.9418±0.0016 | 0.7412/0.7468/0.7545 | 0.7475±0.0067 |

- 描述性排序(AUROC): b0 > c2 > c1 > clocs > simclr; c2−c1 逐seed: AUROC 平/正/正, AUPRC 3/3 正。
  **门判定与统计检验不在本任务范围**(3-seed 门纪律与配对统计归 B 机重算, 见 §8)。
- **旧 CPSC 数字(W2 confirm_results / W4 baseline_results 相应行)不与新数可比**:
  旧数出自 2004 并集 test(含 619 个多出文件+2 跨类重复), 冻结账本原样保留不改。

## 8. 事故与处置 (透明记录)

- 首启 b0 s0 在评测完成后被 `metrics_ext.py::_assert_w3_dir` 拦截——逐记录预测白名单
  仅 runlog/W3、W4, 不含 W5(守卫为 W4 A-8 时代实现)。处置: 白名单扩展至 runlog/W5/
  (W2 硬拒不变), 同步新增镜像单测 `test_guard_allows_w5_and_still_rejects_w2`,
  `tests/test_metrics_ext.py` 20/20 全绿后重启矩阵, 15/15 收口。
- 涉及仓库代码变更: `metrics_ext.py`(守卫白名单+注释)、`tests/test_metrics_ext.py`(+1 用例)。
- **知会 B 机: CPSC 统计行(paired_stats/paper_materials_v2 中 cpsc 各行)全部待重算**,
  输入 = `runlog/W5/predictions/`(15 套逐记录预测, protocol_id=w5-cpscredo);
  PTB/Chapman 统计行不受影响, 无需重跑。
