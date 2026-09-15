"""Multi-label metrics used by the CPSC2018 external validation in E004.

AUPRC/AUROC are computed from continuous prediction probabilities (no
threshold). Macro-F1 / Macro-Recall / Accuracy require per-class thresholds,
which are selected on the validation split only and then applied once to the
locked test split.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


OFFICIAL_SCORED_CPSC_CLASSES = ("IAVB", "AF", "NSR", "PAC", "LBBB", "RBBB")


def _as_arrays(y_true, y_prob, num_classes):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = np.eye(num_classes, dtype=np.int64)[y_true]
    y_true = y_true.astype(np.int64)
    if y_prob.shape != y_true.shape:
        raise ValueError(
            f"label/probability shape mismatch: {y_true.shape} vs {y_prob.shape}"
        )
    return y_true, y_prob


def _macro_auprc(y_true, y_prob):
    scores = []
    for column in range(y_true.shape[1]):
        target = y_true[:, column]
        if np.unique(target).size < 2:
            continue
        scores.append(average_precision_score(target, y_prob[:, column]))
    return float(np.mean(scores)) if scores else None


def _macro_auroc(y_true, y_prob):
    scores = []
    for column in range(y_true.shape[1]):
        target = y_true[:, column]
        if np.unique(target).size < 2:
            continue
        scores.append(roc_auc_score(target, y_prob[:, column]))
    return float(np.mean(scores)) if scores else None


def compute_multilabel_scores(y_true, y_prob, class_names):
    """Return continuous-probability macro metrics and per-class AUPRC/AUROC."""
    num_classes = len(class_names)
    y_true, y_prob = _as_arrays(y_true, y_prob, num_classes)
    per_class = {}
    for column, class_name in enumerate(class_names):
        target = y_true[:, column]
        if np.unique(target).size < 2:
            per_class[class_name] = {
                "support": int(target.sum()),
                "auroc": None,
                "auprc": None,
            }
        else:
            per_class[class_name] = {
                "support": int(target.sum()),
                "auroc": float(roc_auc_score(target, y_prob[:, column])),
                "auprc": float(average_precision_score(target, y_prob[:, column])),
            }
    official_indices = [
        class_names.index(name)
        for name in OFFICIAL_SCORED_CPSC_CLASSES
        if name in class_names
    ]
    official_auprc = None
    official_auroc = None
    if official_indices:
        official_auprc = _macro_auprc(y_true[:, official_indices], y_prob[:, official_indices])
        official_auroc = _macro_auroc(y_true[:, official_indices], y_prob[:, official_indices])
    return {
        "macro_auprc": _macro_auprc(y_true, y_prob),
        "macro_auroc": _macro_auroc(y_true, y_prob),
        "official_six_macro_auprc": official_auprc,
        "official_six_macro_auroc": official_auroc,
        "num_samples": int(len(y_true)),
        "per_class": per_class,
    }


def select_thresholds(y_true, y_prob, class_names):
    """Select one threshold per class maximizing that class's F1 on the given split.

    Must only be called with validation predictions; the returned thresholds are
    then applied unchanged to the test split.
    """
    from sklearn.metrics import precision_recall_curve

    num_classes = len(class_names)
    y_true, y_prob = _as_arrays(y_true, y_prob, num_classes)
    thresholds = {}
    for column, class_name in enumerate(class_names):
        target = y_true[:, column]
        if np.unique(target).size < 2:
            thresholds[class_name] = 0.5
            continue
        precision, recall, curve = precision_recall_curve(target, y_prob[:, column])
        denominator = precision + recall
        f1 = np.divide(
            2 * precision * recall,
            denominator,
            out=np.zeros_like(precision, dtype=np.float64),
            where=denominator > 0,
        )
        best = int(np.argmax(f1))
        thresholds[class_name] = float(curve[best]) if best < len(curve) else 0.5
    return thresholds


def apply_thresholds(y_true, y_prob, class_names, thresholds):
    """Compute thresholded macro-F1 / macro-Recall / Accuracy using fixed thresholds."""
    num_classes = len(class_names)
    y_true, y_prob = _as_arrays(y_true, y_prob, num_classes)
    y_pred = np.zeros_like(y_true)
    for column, class_name in enumerate(class_names):
        y_pred[:, column] = (y_prob[:, column] >= thresholds[class_name]).astype(np.int64)
    class_precision = precision_score(
        y_true, y_pred, average=None, zero_division=0
    )
    class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            class_name: {
                "precision": float(class_precision[column]),
                "recall": float(class_recall[column]),
                "f1": float(class_f1[column]),
                "threshold": float(thresholds[class_name]),
            }
            for column, class_name in enumerate(class_names)
        },
    }


def save_multilabel(output_dir, y_true, y_prob, class_names, thresholds=None, metadata=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = compute_multilabel_scores(y_true, y_prob, class_names)
    payload = {
        "macro_auprc": scores["macro_auprc"],
        "macro_auroc": scores["macro_auroc"],
        "official_six_macro_auprc": scores["official_six_macro_auprc"],
        "official_six_macro_auroc": scores["official_six_macro_auroc"],
        "num_samples": scores["num_samples"],
        "class_names": list(class_names),
        "per_class": scores["per_class"],
        "thresholds": thresholds,
    }
    if thresholds is not None:
        thresholded = apply_thresholds(y_true, y_prob, class_names, thresholds)
        per_class_thresholded = thresholded.pop("per_class")
        payload.update(thresholded)
        for class_name, values in per_class_thresholded.items():
            scores["per_class"][class_name].update(values)
    if metadata:
        payload.update(_json_ready(metadata))
    np.save(output_dir / "y_true.npy", np.asarray(y_true))
    np.save(output_dir / "y_prob.npy", np.asarray(y_prob, dtype=np.float32))
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    with (output_dir / "per_class_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(scores["per_class"], stream, ensure_ascii=False, indent=2)
    return payload


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
