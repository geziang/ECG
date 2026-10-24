# -*- coding: utf-8 -*-
"""thesis-inc-20261024 shared helpers: paths, budget ledger, deadlines, logs.

独立新批次; 不 import thesis_p0.common (P0 结果目录只读), 只复用其模式。
结果目录 = runlog/THESIS_INC_20261024/; 权重 = checkpoint/thesis_inc/<job_id>/。
"""
import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

GIT_ROOT = Path(r"F:\新实验")
CODE_ROOT = GIT_ROOT / "ECG_SSL_LFBT-main"
PKG = GIT_ROOT / "docs" / "experiments" / "THESIS_INC_20261024"
RESULTS = CODE_ROOT / "runlog" / "THESIS_INC_20261024"
JOBS = RESULTS / "jobs"
CKPT_ROOT = CODE_ROOT / "checkpoint" / "thesis_inc"
BACKUP_ROOT = Path(r"D:\thesis_inc_20261024_backup")
ADAPT = Path(r"D:\thesis_p0_20261023_backup\adaptation_data")  # P0 隔离 LP 数据(只读)

PROTOCOL = json.loads((PKG / "protocol.json").read_text(encoding="utf-8-sig"))
CST = timedelta(hours=8)  # Asia/Shanghai


def now_bj() -> datetime:
    return datetime.now(timezone.utc) + CST


def bj_str(dt: datetime | None = None) -> str:
    dt = dt or now_bj()
    return dt.strftime("%Y-%m-%d %H:%M:%S+08:00")


def deadline(name: str) -> datetime:
    return datetime.fromisoformat(PROTOCOL["window"][name])


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


def git_status_porcelain() -> list:
    out = subprocess.run(["git", "-C", str(GIT_ROOT), "status", "--porcelain"],
                         capture_output=True, text=True).stdout
    return out.splitlines()


# ---------------- 预算账本 ----------------
STATE_F = RESULTS / "budget_state.json"


def load_state() -> dict:
    if STATE_F.exists():
        return json.loads(STATE_F.read_text(encoding="utf-8"))
    return {"gpu_s": 0.0, "ref_gpu_s": 0.0, "cpu_s": 0.0, "eng_s": 0.0,
            "jobs_started": 0, "jobs_started_by_stage": {},
            "repairs": {}, "updated": bj_str()}


def save_state(st: dict):
    st["updated"] = bj_str()
    STATE_F.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_F.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(STATE_F)


def add_resource(kind: str, seconds: float, job_id: str = ""):
    """kind: gpu|ref_gpu|cpu|eng; 累加进账本并写 resource_usage.csv。"""
    st = load_state()
    key = {"gpu": "gpu_s", "ref_gpu": "ref_gpu_s", "cpu": "cpu_s", "eng": "eng_s"}[kind]
    st[key] = st.get(key, 0.0) + max(0.0, seconds)
    save_state(st)
    append_csv(RESULTS / "resource_usage.csv", {
        "ts": bj_str(), "job_id": job_id, "kind": kind,
        "add_s": round(max(0.0, seconds), 1),
        "cum_gpu_h": round(st["gpu_s"] / 3600, 3), "cum_ref_gpu_h": round(st["ref_gpu_s"] / 3600, 3),
        "cum_cpu_h": round(st["cpu_s"] / 3600, 3), "cum_eng_h": round(st["eng_s"] / 3600, 3)})


def budget_left_hours() -> dict:
    st = load_state()
    b = PROTOCOL["budgets"]
    return {
        "gpu": (b["max_gpu_process_hours"] - st["gpu_s"] / 3600),
        "ref_gpu": (b["max_reference_gpu_hours"] - st["ref_gpu_s"] / 3600),
        "cpu": (b["max_cpu_compute_hours"] - st["cpu_s"] / 3600),
        "eng": (b["max_engineering_hours"] - st["eng_s"] / 3600),
        "jobs": b["max_primary_training_jobs"] - st["jobs_started"],
    }


def count_job_start(job_id: str, stage: str, repair: bool = False):
    st = load_state()
    if not repair:  # 修复重跑不重复计作业数(资源仍照计)
        st["jobs_started"] += 1
        st["jobs_started_by_stage"][stage] = st["jobs_started_by_stage"].get(stage, 0) + 1
    else:
        st["repairs"][job_id] = st["repairs"].get(job_id, 0) + 1
    save_state(st)


# ---------------- 日志 ----------------
def log_event(line: str):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "events.md", "a", encoding="utf-8") as f:
        f.write(f"- **{bj_str()}** {line}\n")


def log_decision(payload: dict):
    payload = {"ts": bj_str(), **payload}
    with open(RESULTS / "decisions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_csv(path: Path, row: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def set_seed(seed: int):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def backup_job_artifacts(job_id: str):
    """PT 产出(编码器+完整恢复状态)复制到 D 盘独立备份目录; 返回清单。"""
    import shutil
    src = CKPT_ROOT / job_id
    dst = BACKUP_ROOT / job_id
    if not src.exists():
        return {"job_id": job_id, "skipped": "no_ckpt_dir"}
    dst.mkdir(parents=True, exist_ok=True)
    out = {"job_id": job_id, "ts": bj_str(), "files": {}}
    for name in ("encoder_group.pth", "train_state.pth"):
        p = src / name
        if p.exists():
            q = dst / name
            shutil.copy2(p, q)
            out["files"][name] = {"bytes": p.stat().st_size, "sha256": sha256_of(q)}
    return out


# ---------------- LP 数据(唯一开发数据入口, test 隔离守卫) ----------------
def load_dev_data(split: str):
    if split not in ("train", "val"):
        raise PermissionError(f"dev entry refuses split={split!r}")
    z = np.load(ADAPT / f"{split}_data.npz", allow_pickle=False)
    return z["X"], z["record_ids"].astype(str), z["y"].astype(np.float32)


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """与 P0 同口径: 宏 AP/AUROC(9类全保留), 元素加权 val BCE 由调用方另算。"""
    from sklearn.metrics import average_precision_score, roc_auc_score
    n_classes = y_true.shape[1]
    per_ap, per_auc = [], []
    for c in range(n_classes):
        per_ap.append(float(average_precision_score(y_true[:, c], y_prob[:, c])))
        try:
            per_auc.append(float(roc_auc_score(y_true[:, c], y_prob[:, c])))
        except ValueError:
            per_auc.append(None)
    eps = 1e-12
    p = np.clip(y_prob, eps, 1 - eps)
    bce = float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())
    return {
        "macro_auprc": float(np.mean(per_ap)),
        "macro_auroc": float(np.mean([a if a is not None else np.nan for a in per_auc])),
        "per_class_ap": per_ap, "per_class_auroc": per_auc,
        "support": [int(v) for v in y_true.sum(0)],
        "val_bce_from_prob": bce,
    }
