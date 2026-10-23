"""thesis-p0-20261023-v1 A0 preflight: machine/env实测, 9 checkpoints SHA256 vs
source_lock, 7 local_assets hash check, checkpoint backup copy + verify.

Writes runlog/THESIS_P0_20261023/preflight.json. Local checks only; reference
source fetch and HeartLang weight resolution are handled separately.
"""
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

GIT_ROOT = Path(r"F:\新实验")
CODE_ROOT = GIT_ROOT / "ECG_SSL_LFBT-main"
LOCK_PATH = GIT_ROOT / "docs/experiments/THESIS_P0_20261023/source_lock.json"
RESULTS_DIR = CODE_ROOT / "runlog" / "THESIS_P0_20261023"
BACKUP_DIR = Path(r"D:\thesis_p0_20261023_backup")

CKPT_CANDIDATES = {
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


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info():
    def run(*args):
        return subprocess.run(
            ["git", "-C", str(GIT_ROOT), *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()
    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": run("status", "--porcelain").splitlines(),
    }


def main():
    t0 = time.time()
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "protocol_id": lock["protocol_id"],
        "stage": "A0_preflight",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": {
            "hostname": platform.node(),
            "os": platform.platform(),
            "processor_cores_logical": os_cpu_count() if (os_cpu_count := __import__("os").cpu_count) else None,
        },
        "git": git_info(),
        "environment": {},
        "checkpoints": [],
        "local_assets": [],
        "backup_dir": str(BACKUP_DIR),
        "issues": [],
    }

    # --- environment 实测 ---
    env = report["environment"]
    env["python_executable"] = sys.executable
    env["python_version"] = sys.version
    try:
        import numpy
        env["numpy"] = numpy.__version__
        import sklearn
        env["scikit_learn"] = sklearn.__version__
        import wfdb
        env["wfdb"] = wfdb.__version__
        import torch
        env["torch"] = torch.__version__
        env["torch_cuda_available"] = torch.cuda.is_available()
        env["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_total_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    except Exception as e:  # pragma: no cover
        report["issues"].append(f"environment probe error: {e!r}")

    # --- nine checkpoints vs source_lock ---
    lock_ckpt = {(c["model"], c["pretrain_seed"]): c for c in lock["local_checkpoints"]}
    assert set(lock_ckpt) == set(CKPT_CANDIDATES), "candidate/lock key mismatch"
    ckpt_backup_dir = BACKUP_DIR / "checkpoints"
    ckpt_backup_dir.mkdir(parents=True, exist_ok=True)
    for key, cand in CKPT_CANDIDATES.items():
        entry = dict(zip(["model", "pretrain_seed"], key))
        expected = lock_ckpt[key]
        entry["expected_sha256"] = expected["sha256"]
        entry["source_lock_trc"] = expected["trc"]
        entry["candidate_path"] = str(cand)
        if not cand.exists():
            entry.update(status="missing", actual_sha256=None)
            report["issues"].append(f"checkpoint missing: {key} at {cand}")
            report["checkpoints"].append(entry)
            continue
        actual = sha256_of(cand)
        entry["actual_sha256"] = actual
        entry["actual_bytes"] = cand.stat().st_size
        if actual != expected["sha256"]:
            entry.update(status="sha_mismatch")
            report["issues"].append(f"checkpoint sha mismatch: {key}")
            report["checkpoints"].append(entry)
            continue
        # verified -> backup copy + re-verify
        dest = ckpt_backup_dir / f"{key[0]}_s{key[1]}_encoder_group.pth"
        shutil.copy2(cand, dest)
        copy_sha = sha256_of(dest)
        cfg = cand.parent / "config.json"
        cfg_dest = None
        if cfg.exists():
            cfg_dest = ckpt_backup_dir / f"{key[0]}_s{key[1]}_config.json"
            shutil.copy2(cfg, cfg_dest)
        entry.update(
            status="ok",
            backup_path=str(dest),
            backup_sha256=copy_sha,
            backup_verified=(copy_sha == expected["sha256"]),
            backup_config_path=str(cfg_dest) if cfg_dest else None,
        )
        if copy_sha != expected["sha256"]:
            report["issues"].append(f"backup verify failed: {key}")
        report["checkpoints"].append(entry)

    # --- local_assets delivered-byte check ---
    for asset in lock["local_assets"]:
        p = GIT_ROOT / asset["path_relative_to_git_root"]
        entry = {"path": asset["path_relative_to_git_root"], "expected_sha256": asset["sha256"]}
        if not p.exists():
            entry.update(status="missing")
            report["issues"].append(f"asset missing: {p}")
        else:
            actual = sha256_of(p)
            entry.update(
                status="ok" if actual == asset["sha256"] else "sha_mismatch",
                actual_sha256=actual,
                actual_bytes=p.stat().st_size,
            )
            if actual != asset["sha256"]:
                report["issues"].append(f"asset sha mismatch: {p}")
        report["local_assets"].append(entry)

    report["passed"] = not report["issues"]
    report["elapsed_s"] = round(time.time() - t0, 1)
    out = RESULTS_DIR / "preflight.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    n_ok = sum(1 for c in report["checkpoints"] if c["status"] == "ok" and c.get("backup_verified"))
    n_asset_ok = sum(1 for a in report["local_assets"] if a["status"] == "ok")
    print(f"preflight: checkpoints {n_ok}/9 verified+backed-up, assets {n_asset_ok}/{len(report['local_assets'])} ok")
    print(f"passed={report['passed']} -> {out}")
    for issue in report["issues"]:
        print("ISSUE:", issue)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
