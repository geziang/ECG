"""thesis-p0-20261023 shared helpers: paths, hashing, metrics, job IO, budget log."""
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

GIT_ROOT = Path(r"F:\新实验")
CODE_ROOT = GIT_ROOT / "ECG_SSL_LFBT-main"
RESULTS = CODE_ROOT / "runlog" / "THESIS_P0_20261023"
ADAPT = Path(r"D:\thesis_p0_20261023_backup\adaptation_data")
FEAT_CACHE = ADAPT / "feat_cache"
JOBS = RESULTS / "jobs"
PROTOCOL = json.loads((GIT_ROOT / "docs/experiments/THESIS_P0_20261023/protocol.json").read_text(encoding="utf-8"))

CLASS_ORDER = PROTOCOL["data"]["class_order"]
CKPT_PATHS = {
    ("M0", 0): CODE_ROOT / "checkpoint/confirm/c1_seed0/encoder_group.pth",
    ("M0", 2): CODE_ROOT / "checkpoint/confirm/c1_seed2/encoder_group.pth",
    ("M0", 4): CODE_ROOT / "checkpoint/confirm/c1_seed4/encoder_group.pth",
    ("M4", 0): CODE_ROOT / "checkpoint/joint_m4_noise/seed0/encoder_group.pth",
    ("M4", 2): CODE_ROOT / "checkpoint/joint_m4_noise/seed2/encoder_group.pth",
    ("M4", 4): CODE_ROOT / "checkpoint/joint_m4_noise/seed4/encoder_group.pth",
    ("M6", 0): CODE_ROOT / "checkpoint/joint_m6/seed0/encoder_group.pth",
    ("M6", 2): CODE_ROOT / "checkpoint/joint_m6/seed2/encoder_group.pth",
    ("M6", 4): CODE_ROOT / "checkpoint/joint_m6/seed4/encoder_group.pth",
}
CKPT_TRC = {"M0": 0, "M4": 1, "M6": 1}


def sha256_of(p) -> str:
    p = Path(p)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
        return h.hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "-C", str(GIT_ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def code_diff_dirty() -> list:
    out = subprocess.run(["git", "-C", str(GIT_ROOT), "status", "--porcelain"],
                         capture_output=True, text=True).stdout
    return out.splitlines()


def set_seed(seed: int):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dev_data(split: str):
    """唯一开发数据入口: 只读 A1 导出的 train/val 副本(测试隔离守卫)。"""
    if split not in ("train", "val"):
        raise PermissionError(f"dev entry refuses split={split!r}")
    z = np.load(ADAPT / f"{split}_data.npz", allow_pickle=False)
    return z["X"], z["record_ids"].astype(str), z["y"].astype(np.float32)


def data_sha() -> dict:
    m = json.loads((RESULTS / "data_manifest.json").read_text(encoding="utf-8"))
    return {sp: m["export"][sp]["npz_sha256"] for sp in ("train", "val")}


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """y_true (N,9) 0/1, y_prob (N,9) sigmoid in (0,1). All classes kept."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    n_classes = y_true.shape[1]
    per_ap, per_auc = [], []
    for c in range(n_classes):
        col_true, col_prob = y_true[:, c], y_prob[:, c]
        per_ap.append(float(average_precision_score(col_true, col_prob)))
        try:
            per_auc.append(float(roc_auc_score(col_true, col_prob)))
        except ValueError:
            per_auc.append(None)  # no positives & no negatives -> N/A, macro mean keeps denominator 9
    eps = 1e-12
    p = np.clip(y_prob, eps, 1 - eps)
    bce = float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())
    single = y_true.sum(1) == 1
    multi = y_true.sum(1) >= 2

    def group_auprc(mask):
        """组内宏 AP: 仅对组内阳性数>=1 的类计算, 显式记录分母(不悄悄改)."""
        if not mask.any():
            return None, 0
        aps = [float(average_precision_score(y_true[mask][:, c], y_prob[mask][:, c]))
               for c in range(n_classes) if y_true[mask, c].sum() >= 1]
        return (float(np.mean(aps)), len(aps)) if aps else (None, 0)

    auprc_single, n_single_used = group_auprc(single)
    auprc_multi, n_multi_used = group_auprc(multi)
    out = {
        "macro_auprc": float(np.mean(per_ap)),
        "macro_auroc": float(np.mean([a if a is not None else np.nan for a in per_auc])),
        "per_class_ap": per_ap,
        "per_class_auroc": per_auc,
        "support": [int(v) for v in y_true.sum(0)],
        "val_bce_from_prob": bce,
        "n_single": int(single.sum()),
        "n_multi": int(multi.sum()),
        "macro_auprc_single": auprc_single,
        "group_single_ap_denominator_classes": n_single_used,
        "macro_auprc_multi": auprc_multi,
        "group_multi_ap_denominator_classes": n_multi_used,
    }
    return out


def append_csv(path: Path, row: dict):
    path = Path(path)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def log_resource(job_id: str, phase: str, wall_s: float, gpu: bool, extra: dict | None = None):
    row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "job_id": job_id, "phase": phase,
           "wall_s": round(wall_s, 1), "counts_as_gpu_process": int(gpu)}
    if extra:
        row.update(extra)
    append_csv(RESULTS / "resource_usage.csv", row)


def log_event(line: str):
    with open(RESULTS / "events.md", "a", encoding="utf-8") as f:
        f.write(f"- **{time.strftime('%Y-%m-%d %H:%M')}** {line}\n")
