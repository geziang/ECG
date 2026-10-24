# -*- coding: utf-8 -*-
"""thesis-inc A0 预检: 环境/源码/数据/NFH峰缓存/LP数据/只读checkpoint/磁盘。

产出 preflight.json, source_manifest.json, data_manifest.json。
只读操作; 不启动训练; 不触碰任何 test 数据。
"""
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.thesis_inc import common as C


def main() -> int:
    t0 = time.time()
    out = {"protocol_id": C.PROTOCOL["protocol_id"], "ts": C.bj_str(), "host": platform.node()}
    issues = []

    # --- 环境 ---
    import torch
    out["env"] = {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
    }
    try:
        import wfdb  # noqa: F401
        out["env"]["wfdb"] = wfdb.__version__
    except Exception as e:  # pragma: no cover
        issues.append(f"wfdb import fail: {e}")
    q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    out["gpu_state"] = q
    for drive, label in (("F:\\", "code"), ("D:\\", "backup")):
        try:
            u = subprocess.run(["fsutil", "volume", "diskfree", drive],
                               capture_output=True, text=True).stdout
            out.setdefault("disk_free_bytes", {})[label] = u.splitlines()[0] if u else "?"
        except Exception:
            out.setdefault("disk_free_bytes", {})[label] = "query_fail"

    # --- 源码锁 ---
    lock = json.loads((C.PKG / "source_lock.json").read_text(encoding="utf-8-sig"))
    src_ok = 0
    for item in lock["local_assets"]:
        data = subprocess.check_output(
            ["git", "-C", str(C.GIT_ROOT), "show",
             lock["observed_git_base"] + ":" + item["path_relative_to_git_root"]])
        import hashlib
        sha = hashlib.sha256(data).hexdigest()
        assert sha == item["canonical_git_blob_sha256"], item["path_relative_to_git_root"]
        src_ok += 1
    out["source_lock_hashes_ok"] = src_ok
    out["git_head_at_preflight"] = C.git_head()
    out["git_dirty_at_preflight"] = C.git_status_porcelain()

    # --- 只读 checkpoint 复核 ---
    ck_ok = []
    for ck in lock["existing_checkpoints_read_only"]:
        actual = C.sha256_of(ck["candidate_path"])
        ck_ok.append({"model": ck["model"], "seed": ck["pretrain_seed"],
                      "match": actual == ck["expected_sha256"],
                      "path": ck["candidate_path"], "backup_exists": Path(ck["backup_path"]).exists(),
                      "backup_match": C.sha256_of(ck["backup_path"]) == ck["expected_sha256"]})
    out["readonly_checkpoints"] = ck_ok
    if not all(r["match"] and r["backup_match"] for r in ck_ok):
        issues.append("readonly checkpoint SHA mismatch")

    # --- NFH 数据与峰缓存 ---
    samples_dir = C.CODE_ROOT / "data" / "pt_pretrain_nfh" / "samples"
    paths = sorted(samples_dir.glob("*.npy"))
    out["nfh_records"] = len(paths)
    if len(paths) != C.PROTOCOL["pretrain"]["expected_records"]:
        issues.append(f"NFH records {len(paths)} != expected 34905")
    rp = np.load(C.CODE_ROOT / "data" / "pt_rpeaks_nfh.npz", allow_pickle=True)
    rp_keys = set(rp.keys())
    stems = {p.stem for p in paths}
    out["rpeak_keys"] = len(rp_keys)
    out["rpeak_key_match_sample_stems"] = rp_keys == stems
    if rp_keys != stems:
        issues.append("rpeak cache keys != sample stems")
    # 抽检 + 统计
    import hashlib
    rec_shas = {}
    n_valid = 0
    peak_counts = {}
    t1 = time.time()
    for i, p in enumerate(paths):
        arr = np.load(p)
        if arr.shape != (8, 2048) or arr.dtype != np.float32:
            issues.append(f"{p.name} shape/dtype {arr.shape}/{arr.dtype}")
            break
        pk = rp[p.stem]
        peak_counts[p.stem] = int(len(pk))
        if len(pk) >= 3:
            n_valid += 1
        rec_shas[p.stem] = hashlib.sha256(arr.tobytes()).hexdigest()
        if i % 5000 == 0:
            print(f"  hashed {i}/{len(paths)} ({time.time()-t1:.0f}s)", flush=True)
    cont_manifest_sha = hashlib.sha256(
        "\n".join(f"{k} {rec_shas[k]}" for k in sorted(rec_shas)).encode()).hexdigest()
    out["nfh_valid_records_peaks_ge3"] = n_valid
    out["nfh_content_manifest_sha256"] = cont_manifest_sha
    out["rpeak_npz_sha256"] = C.sha256_of(C.CODE_ROOT / "data" / "pt_rpeaks_nfh.npz")
    out["nfh_hash_elapsed_s"] = round(time.time() - t1, 1)

    # RR 归一化统计(数据清单级记录; 训练数据集内同式复算)
    from tools.thesis_inc.pt_dataset import rr_stats
    stats = [rr_stats(rp[s]) for s in sorted(rp_keys)]
    stats = [s for s in stats if s is not None]
    arr = np.asarray(stats)
    out["rr_mu"] = arr.mean(axis=0).tolist()
    out["rr_sd"] = arr.std(axis=0).tolist()
    out["rr_stats_n"] = len(stats)

    # --- P0 隔离 LP 数据 ---
    lp = {}
    for sp, exp in (("train", lock["p0_isolated_dev_data"]["train"]),
                    ("val", lock["p0_isolated_dev_data"]["val"])):
        p = C.ADAPT / f"{sp}_data.npz"
        sha = C.sha256_of(p)
        z = np.load(p, allow_pickle=False)
        lp[sp] = {"sha_match": sha == exp["npz_sha256"], "n": z["X"].shape[0],
                  "n_expected": exp["n"]}
        if not lp[sp]["sha_match"] or lp[sp]["n"] != exp["n"]:
            issues.append(f"LP data {sp} mismatch")
    out["lp_data"] = lp

    out["issues"] = issues
    out["elapsed_s"] = round(time.time() - t0, 1)
    out["passed"] = not issues
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "preflight.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # source_manifest: 本批实现涉及的新文件 + 基线 commit
    (C.RESULTS / "source_manifest.json").write_text(json.dumps({
        "protocol_id": C.PROTOCOL["protocol_id"],
        "git_head": C.git_head(), "git_dirty": C.git_status_porcelain(),
        "baseline_source_lock_verified": src_ok,
        "new_impl_files": [str(p.relative_to(C.GIT_ROOT)) for p in
                           sorted((C.CODE_ROOT / "tools" / "thesis_inc").glob("*.py"))],
        "ts": C.bj_str()}, ensure_ascii=False, indent=1), encoding="utf-8")

    # data_manifest: record ID <-> 内容 SHA <-> 峰键 对应清单(抽样核验入口 + 全量CSV)
    dm = {
        "protocol_id": C.PROTOCOL["protocol_id"], "ts": C.bj_str(),
        "nfh": {"records": len(paths), "content_manifest_sha256": cont_manifest_sha,
                "rpeak_npz_sha256": out["rpeak_npz_sha256"],
                "keys_match": rp_keys == stems,
                "valid_records_peaks_ge3": n_valid,
                "rr_mu": out["rr_mu"], "rr_sd": out["rr_sd"], "rr_stats_n": len(stats)},
        "lp_dev_data": lp,
        "per_record_csv": "nfh_record_manifest.csv"}
    (C.RESULTS / "data_manifest.json").write_text(
        json.dumps(dm, ensure_ascii=False, indent=1), encoding="utf-8")
    import csv as _csv
    with open(C.RESULTS / "nfh_record_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["record_id", "content_sha256", "n_peaks", "valid_ge3"])
        for s in sorted(rec_shas):
            w.writerow([s, rec_shas[s], peak_counts[s], int(peak_counts[s] >= 3)])

    print(json.dumps({"passed": out["passed"], "issues": issues,
                      "nfh_records": out["nfh_records"], "valid": n_valid,
                      "elapsed_s": out["elapsed_s"]}, ensure_ascii=False))
    return 0 if out["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
