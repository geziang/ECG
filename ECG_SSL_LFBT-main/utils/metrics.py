import json
import hashlib
import platform
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


def runtime_metadata():
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(json_ready(payload), stream, ensure_ascii=False, indent=2)


def compute_metrics(y_true, y_prob, class_names):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    num_classes = len(class_names)
    if y_prob.ndim != 2 or y_prob.shape[1] != num_classes:
        raise ValueError(
            f"probability shape {y_prob.shape} does not match {num_classes} classes"
        )
    y_pred = y_prob.argmax(axis=1)
    y_one_hot = np.eye(num_classes, dtype=np.float64)[y_true]
    per_class = {}
    aurocs = []
    auprcs = []
    recalls = recall_score(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        average=None,
        zero_division=0,
    )
    for class_index, class_name in enumerate(class_names):
        target = y_one_hot[:, class_index]
        if np.unique(target).size < 2:
            auroc = None
            auprc = None
        else:
            auroc = float(roc_auc_score(target, y_prob[:, class_index]))
            auprc = float(average_precision_score(target, y_prob[:, class_index]))
            aurocs.append(auroc)
            auprcs.append(auprc)
        per_class[class_name] = {
            "support": int(target.sum()),
            "auroc": auroc,
            "auprc": auprc,
            "recall": float(recalls[class_index]),
        }
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    metrics = {
        "auroc": float(np.mean(aurocs)) if aurocs else None,
        "auprc": float(np.mean(auprcs)) if auprcs else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=np.arange(num_classes),
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=np.arange(num_classes),
                average="macro",
                zero_division=0,
            )
        ),
        "num_samples": int(len(y_true)),
        "class_names": list(class_names),
    }
    return metrics, per_class, matrix, y_pred


def save_evaluation(output_dir, y_true, y_prob, class_names, metadata=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, per_class, matrix, y_pred = compute_metrics(y_true, y_prob, class_names)
    if metadata:
        metrics.update(json_ready(metadata))
    np.save(output_dir / "y_true.npy", np.asarray(y_true, dtype=np.int64))
    np.save(output_dir / "y_prob.npy", np.asarray(y_prob, dtype=np.float32))
    np.save(output_dir / "y_pred.npy", np.asarray(y_pred, dtype=np.int64))
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "per_class_metrics.json", per_class)
    write_json(
        output_dir / "confusion_matrix.json",
        {"class_names": class_names, "matrix": matrix.tolist()},
    )
    return metrics
