# -*- coding: utf-8 -*-
"""metrics_ext.py —— W3 B-2 评估指标扩展(主机B, 任务书 B-2)。

为 run_lp.py / run_ft.py 提供可选扩展指标与逐记录保存, 全部默认关闭,
不改变 W2 既有路径与 metrics.json 默认内容。

提供:
  per_class_ap(y_true, prob)         逐类 AP + macro(与 sklearn average_precision_score
                                      2D 默认 macro 口径一致, smoke 有交叉验证)
  conf_mat_metrics(conf_mat)         per-class precision/recall(=sens)/specificity/F1,
                                      macro_f1/sens/spec, accuracy
  calibration(y_true, prob, n_bins)  ECE/MCE(等宽分桶, max-prob 置信度), 多类 Brier
                                      (mean_samp mean_c (p-y)^2), 分桶校准表
  compute_all(y_true, prob, n_bins)  上述汇总 dict(JSON 可序列化)
  sha256_of(path)                    文件 SHA256
  save_eval_artifacts(...)           逐记录 y_true/y_pred/prob 落盘(仅允许 runlog/W3/,
                                      拒绝 runlog/W2 与仓库外默认; 元数据必填 protocol_id)

依赖: numpy(仅此); 单测见 tests/test_metrics_ext.py。
"""
import hashlib
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- AP
def average_precision(y_true_bin, score):
    """单类 AP: 按分数降序逐步precision的加权和(与 sklearn 定义一致, 无ties特殊处理)。"""
    y_true_bin = np.asarray(y_true_bin, dtype=float)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="stable")
    y_sorted = y_true_bin[order]
    n_pos = y_sorted.sum()
    if n_pos == 0:
        return float("nan")
    cum_tp = np.cumsum(y_sorted)
    ranks = np.arange(1, len(y_sorted) + 1)
    precision = cum_tp / ranks
    return float((precision * y_sorted).sum() / n_pos)


def per_class_ap(y_true, prob):
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    n_classes = prob.shape[1]
    one_hot = np.eye(n_classes)[y_true]
    aps = {f"ap_class_{c}": average_precision(one_hot[:, c], prob[:, c])
           for c in range(n_classes)}
    aps["macro_ap"] = float(np.mean([aps[f"ap_class_{c}"] for c in range(n_classes)]))
    return aps


# ---------------------------------------------------------------- 混淆矩阵派生
def conf_mat_metrics(conf_mat):
    cm = np.asarray(conf_mat, dtype=float)
    n_classes = cm.shape[0]
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - tp - fn - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        recall_sens = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        specificity = np.where(tn + fp > 0, tn / (tn + fp), np.nan)
        f1 = np.where(precision + recall_sens > 0,
                      2 * precision * recall_sens / (precision + recall_sens), np.nan)
    out = {f"class_{c}_precision": float(precision[c]) for c in range(n_classes)}
    out.update({f"class_{c}_sensitivity": float(recall_sens[c]) for c in range(n_classes)})
    out.update({f"class_{c}_specificity": float(specificity[c]) for c in range(n_classes)})
    out.update({f"class_{c}_f1": float(f1[c]) for c in range(n_classes)})
    out["macro_f1"] = float(np.nanmean(f1))
    out["macro_sensitivity"] = float(np.nanmean(recall_sens))
    out["macro_specificity"] = float(np.nanmean(specificity))
    out["accuracy"] = float(tp.sum() / cm.sum())
    return out


# ---------------------------------------------------------------- 校准
def calibration(y_true, prob, n_bins=15):
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    n_classes = prob.shape[1]
    one_hot = np.eye(n_classes)[y_true]
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    table, ece, mce = [], 0.0, 0.0
    for b in range(n_bins):
        m = bins == b
        n_b = int(m.sum())
        row = {"bin": b, "range": f"[{b / n_bins:.3f},{(b + 1) / n_bins:.3f})",
               "n": n_b, "avg_conf": None, "avg_acc": None}
        if n_b:
            row["avg_conf"] = float(conf[m].mean())
            row["avg_acc"] = float(correct[m].mean())
            gap = abs(row["avg_conf"] - row["avg_acc"])
            ece += n_b / len(y_true) * gap
            mce = max(mce, gap)
        table.append(row)
    brier_samples = ((prob - one_hot) ** 2).mean(axis=1)  # 样本内对类平均
    return {"ece": float(ece), "mce": float(mce), "n_bins": n_bins,
            "brier_multiclass": float(brier_samples.mean()),
            "brier_formula": "mean_over_samples( mean_over_classes (p_c - y_c)^2 )",
            "reliability_table": table}


# ---------------------------------------------------------------- 汇总
def compute_all(y_true, prob, n_bins=15):
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    n_classes = prob.shape[1]
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, prob.argmax(axis=1)):
        cm[t, p] += 1
    out = {"confusion_matrix": cm.tolist()}
    out.update(per_class_ap(y_true, prob))
    out.update(conf_mat_metrics(cm))
    out.update(calibration(y_true, prob, n_bins))
    return out


# ---------------------------------------------------------------- 元数据与落盘
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_w3_dir(out_dir: Path):
    """逐记录预测落盘守卫(任务书 B-2; W4 A-8 扩展)。

    允许 runlog/W3/ 与 runlog/W4/ 之下; runlog/W2/ 硬拒(冻结只读);
    其余路径拒绝。不传 --save-predictions 时本函数不会被调用(默认关闭不变)。
    """
    s = str(out_dir.resolve()).replace("\\", "/").lower()
    if "/runlog/w2/" in s:
        raise ValueError(f"W2 目录只读, 拒绝写入: {out_dir}")
    if "/runlog/w3/" not in s and "/runlog/w4/" not in s:
        raise ValueError(f"逐记录预测只允许写 runlog/W3/ 或 runlog/W4/ 之下, 收到: {out_dir}")


def save_eval_artifacts(out_dir, y_true, y_pred, prob, metadata, n_bins=15):
    """保存逐记录预测与扩展指标; out_dir 必须位于 runlog/W3/ 之下(任务书 B-2)。

    输出文件:
      y_true.npy / y_pred.npy / y_prob.npy   (逐记录, 顺序=test loader 顺序)
      metrics_ext.json                        (compute_all 全量 + metadata)
    """
    out_dir = Path(out_dir)
    _assert_w3_dir(out_dir)
    if not metadata.get("protocol_id"):
        raise ValueError("metadata.protocol_id 必填(任务书 B-2 元数据字段)")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "y_true.npy", np.asarray(y_true, dtype=int))
    np.save(out_dir / "y_pred.npy", np.asarray(y_pred, dtype=int))
    np.save(out_dir / "y_prob.npy", np.asarray(prob, dtype=float))
    payload = {"metadata": metadata}
    payload.update(compute_all(y_true, prob, n_bins))
    (out_dir / "metrics_ext.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return out_dir
