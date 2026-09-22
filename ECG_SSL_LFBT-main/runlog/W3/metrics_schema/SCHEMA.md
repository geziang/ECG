# W3 B-2 评估指标扩展 Schema（主机B）

> 交付：`metrics_ext.py`（共享库）+ `run_lp.py`/`run_ft.py` 集成（**默认全关**，不加参数时两脚本
> 行为与 metrics.json 内容与 W2 逐字节一致）+ 单测 18/18 + 本目录 smoke。
> 红线遵守：未重评任何真实 test；逐记录落盘只允许 `runlog/W3/` 之下（W2 目录写入直接抛错）。

## 1. 新增 CLI（run_lp.py / run_ft.py 一致）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--extended-metrics` | 0 | 1=metrics.json 追加 §2 全部扩展字段（原有字段不动） |
| `--save-predictions DIR` | '' | 逐记录 y_true/y_pred/y_prob + metrics_ext.json 落盘到 DIR；DIR 必须位于 `runlog/W3/` 下 |
| `--protocol-id STR` | '' | 协议标识（如 `w3-robustness-v1`）；保存逐记录时**必填** |
| `--data-manifest-sha STR` | '' | 数据 manifest SHA256 元数据 |
| `--checkpoint-sha STR` | '' | checkpoint SHA256；留空且给了 --checkpoint 时自动计算 |

示例（W3 评估调用形态）：
```bash
python run_lp.py --data-dir data/ptbxl --num-classes 22 --feat-dir feat/w3_x \
  --checkpoint checkpoint/confirm/c2_seed0.pth --trc 1 --seed 0 \
  --extended-metrics 1 \
  --save-predictions runlog/W3/robustness_c2_s0/records \
  --protocol-id w3-robustness-v1 --data-manifest-sha <sha> 
```

## 2. metrics_ext.json 字段（smoke_out/ 有真实示例）

- `metadata`: `{protocol_id, data_manifest_sha, checkpoint, checkpoint_sha256, eval, num_classes, seed, zero_leads(仅LP), code_sha}`
- `confusion_matrix`: C×C int 列表
- 逐类：`ap_class_{c}`、`class_{c}_precision/_sensitivity/_specificity/_f1`（空类=NaN 安全）
- 汇总：`macro_ap`（与 sklearn 2D average_precision_score 默认 macro 逐位一致，smoke 已验）、
  `macro_f1`、`macro_sensitivity`、`macro_specificity`、`accuracy`
- 校准：`ece`、`mce`（等宽 15 桶，置信度=max-prob）、`brier_multiclass`
  （公式 `mean_samples mean_classes (p_c−y_c)²`，字段 `brier_formula` 内自述）、
  `reliability_table`（每桶 n/avg_conf/avg_acc）

## 3. 逐记录文件（--save-predictions DIR）

| 文件 | 内容 | 顺序 |
|---|---|---|
| `y_true.npy` | int 真标签 (N,) | test loader 顺序（不 shuffle） |
| `y_pred.npy` | int 预测类 (N,) | 同上 |
| `y_prob.npy` | float softmax 概率 (N,C) | 同上 |
| `metrics_ext.json` | §2 全量 + metadata | — |

## 4. 单测与 smoke

- `tests/test_metrics_ext.py`：18 用例全过——AP/混淆/校准手工验算、sklearn 无并列交叉、
  W2 落盘守卫（拒绝 runlog/W2 与 W3 之外路径、protocol_id 必填）、run_lp/run_ft py_compile
  与默认关闭静态检查。
- `run_smoke.py`：合成 3 类 400 样本走完整链路，macro_ap/macro_f1 与 sklearn 双交叉一致
  （`SMOKE PASS`），示例产物在 `smoke_out/`。

## 5. 统计口径备注

逐记录概率自 W3 起可保存 → 未来 W3 新评估可做 **record-level bootstrap**；
W2 存量结果仍只有聚合指标，其统计表述维持 seed-level（见 `runlog/W3/protocol_audit.md` §2-1）。
