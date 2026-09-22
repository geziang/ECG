# -*- coding: utf-8 -*-
"""W3 B-2 smoke: 在合成小数据上走 metrics_ext 完整链路(compute_all + save_eval_artifacts),
产出 schema 示例文件到 runlog/W3/metrics_schema/smoke_out/。不触碰任何真实 test 集,
不重评 test(任务书 B-2 红线)。CPU-only。
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import metrics_ext as mx  # noqa: E402

rng = np.random.default_rng(20260922)
N, C = 400, 3
y_true = rng.integers(0, C, N)
logits = rng.normal(0, 2, (N, C))
# 让预测与标签弱相关, 指标落在有意义的区间
logits[np.arange(N), y_true] += 1.5
prob = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
y_pred = prob.argmax(1)

meta = {
    "protocol_id": "w3-metrics-smoke-v1",
    "data_manifest_sha": "synthetic-rng-20260922(no real data)",
    "checkpoint": "smoke(no ckpt)",
    "checkpoint_sha256": "",
    "eval": "smoke", "num_classes": C, "seed": 20260922,
    "code_sha": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip(),
}
out_dir = Path(__file__).parent / "smoke_out"
mx.save_eval_artifacts(out_dir, y_true, y_pred, prob, meta)

payload = json.loads((out_dir / "metrics_ext.json").read_text(encoding="utf-8"))
print("smoke artifacts:", sorted(p.name for p in out_dir.iterdir()))
print(f'macro_ap={payload["macro_ap"]:.4f} macro_f1={payload["macro_f1"]:.4f} '
      f'ece={payload["ece"]:.4f} brier={payload["brier_multiclass"]:.4f}')

# 与 sklearn 双交叉(口径一致性): macro AP 与 macro F1
from sklearn.metrics import average_precision_score, f1_score  # noqa: E402
sk_ap = average_precision_score(np.eye(C)[y_true], prob)
sk_f1 = f1_score(y_true, y_pred, average="macro")
assert abs(payload["macro_ap"] - sk_ap) < 1e-10, (payload["macro_ap"], sk_ap)
assert abs(payload["macro_f1"] - sk_f1) < 1e-12, (payload["macro_f1"], sk_f1)
print(f"sklearn cross-check OK: macro_ap {sk_ap:.6f} == ours; macro_f1 {sk_f1:.6f} == ours")
print("SMOKE PASS")
