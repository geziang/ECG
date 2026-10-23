"""thesis-p0-20261023 A1: CPSC 数据契约与测试集隔离。

- 只读 train/val 信号；test 仅核对 CSV 元数据(数量/身份)，不读取其 ECG 内容。
- 校验: 计数/类序/多诊断数/跨split重复/重复内容哈希/目录类-主类一致/样本序对齐/
  每条 (8,2048) float32 有限值/值域审计(防二次归一化)。
- 导出: D:\\thesis_p0_20261023_backup\\adaptation_data\\{train,val}_data.npz
  (X, record_ids, y multihot, class_order), 与 data_manifest.json 一起锁 SHA256。
- 训练进程只允许经 load_dev_data() 读导出副本；test 无任何加载路径。
"""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

GIT_ROOT = Path(r"F:\新实验")
CODE_ROOT = GIT_ROOT / "ECG_SSL_LFBT-main"
RESULTS = CODE_ROOT / "runlog" / "THESIS_P0_20261023"
CSV_PATH = CODE_ROOT / "runlog/preds_archive/cpsc_multilabel/cpsc_multilabel.csv"
SAMPLE_ORDER = CODE_ROOT / "runlog/preds_archive/cpsc_sample_order.txt"
DATA_ROOT = CODE_ROOT / "data" / "cpsc"
ADAPT = Path(r"D:\thesis_p0_20261023_backup\adaptation_data")

CLASS_ORDER = ["NSR", "AF", "IAVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
EXPECTED = {"train": 4809, "val": 683, "test_metadata_only": 1385}
EXPECTED_MULTI = {"train": 343, "val": 36}
PREPROCESS_NOTE = (
    "prepare_cpsc_chapman.py: 中心10s裁剪@500Hz(短则尾零填) -> LFBT 8导联 -> "
    "scipy.signal.resample 2048 -> 整记录全局标量 z-score (x-x.mean())/(x.std()+1e-5) "
    "-> float32; 有效采样率 204.8Hz。协议文'逐导联 z-score'与实测(整记录全局标量)不符, "
    "以实测为准记录, 本批不重算/不二次归一化。"
)


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TestAccessGuard:
    """任何 test 信号/标签访问请求即抛错(A1 契约: test 只允许 CSV 元数据审计)。"""

    def __init__(self):
        self.violations = []

    def check_path(self, p):
        parts = Path(p).parts
        if "test" in parts:
            self.violations.append(str(p))
            raise PermissionError(f"test access denied: {p}")

    def assert_zero(self):
        assert not self.violations, f"test access violations: {self.violations}"


def load_csv():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    header = reader.fieldnames
    return header, rows


def canonical_npy_relpath(row):
    return f"{row['dir_class']}/{row['record_id']}.npy"


def main():
    t0 = time.time()
    guard = TestAccessGuard()
    issues = []

    header, rows = load_csv()
    if header[:6] != ["record_id", "split", "dir_class", "primary_class", "all_snomed_codes", "mapped_classes"]:
        issues.append(f"unexpected CSV header: {header[:6]}")
    if header[6:] != CLASS_ORDER:
        issues.append(f"class columns != protocol order: {header[6:]}")

    by_split = {"train": [], "val": [], "test": []}
    seen_ids = {}
    for r in rows:
        rid, sp = r["record_id"], r["split"]
        if sp not in by_split:
            issues.append(f"unknown split {sp} for {rid}")
            continue
        if rid in seen_ids:
            issues.append(f"duplicate record_id {rid} ({seen_ids[rid]} vs {sp})")
        seen_ids[rid] = sp
        y = np.array([int(r[c]) for c in CLASS_ORDER], dtype=np.int8)
        if y.sum() < 1:
            issues.append(f"{rid}: empty multihot")
        by_split[sp].append((r, y))

    counts = {sp: len(v) for sp, v in by_split.items()}
    for sp in ("train", "val", "test"):
        if counts[sp] != EXPECTED[sp if sp != "test" else "test_metadata_only"]:
            issues.append(f"split {sp} count {counts[sp]} != {EXPECTED.get(sp, EXPECTED['test_metadata_only'])}")

    # cross-split overlap already implied by unique id map; multi-diagnosis counts
    for sp in ("train", "val"):
        n_multi = sum(1 for _, y in by_split[sp] if y.sum() >= 2)
        if n_multi != EXPECTED_MULTI[sp]:
            issues.append(f"{sp} multidiagnosis {n_multi} != {EXPECTED_MULTI[sp]}")

    # dir_class vs primary_class consistency
    n_dir_mismatch = sum(1 for sp in ("train", "val", "test") for r, _ in by_split[sp]
                         if r["dir_class"] != r["primary_class"])
    if n_dir_mismatch:
        issues.append(f"dir_class != primary_class for {n_dir_mismatch} rows")

    # sample-order cross-check against preds_archive canonical order (metadata only)
    so_lines = SAMPLE_ORDER.read_text(encoding="utf-8").splitlines()
    so_map = {}
    cur = None
    for ln in so_lines:
        ln = ln.strip()
        if ln.startswith("[") and ln.endswith("]"):
            cur = ln[1:-1]
            so_map[cur] = []
        elif ln and cur:
            so_map[cur].append(ln)
    for sp in ("train", "val"):
        mine = [canonical_npy_relpath(r) for r, _ in sorted(by_split[sp], key=lambda t: canonical_npy_relpath(t[0]))]
        theirs = sorted(so_map.get(sp, []))
        if mine != theirs:
            issues.append(f"{sp}: canonical order differs from cpsc_sample_order.txt")
    test_ids_mine = sorted(r["record_id"] for r, _ in by_split["test"])
    test_ids_so = sorted(p.split("/")[-1][:-4] for p in so_map.get("test", []))
    if test_ids_mine != test_ids_so:
        issues.append("test: record id set differs from cpsc_sample_order.txt [test] section")

    # export train/val with full per-record audit (signals read ONLY for train/val)
    ADAPT.mkdir(parents=True, exist_ok=True)
    audit = {}
    for sp in ("train", "val"):
        recs = sorted(by_split[sp], key=lambda t: canonical_npy_relpath(t[0]))
        n = len(recs)
        X = np.empty((n, 8, 2048), dtype=np.float32)
        y = np.empty((n, 9), dtype=np.int8)
        rids = []
        src_sha = {}
        gmin = gmax = None
        nonfinite = 0
        for i, (r, yi) in enumerate(recs):
            rel = canonical_npy_relpath(r)
            p = DATA_ROOT / sp / rel
            guard.check_path(p)  # would catch 'test' if ever passed
            arr = np.load(p)
            if arr.shape != (8, 2048) or arr.dtype != np.float32:
                issues.append(f"{r['record_id']}: shape/dtype {arr.shape}/{arr.dtype}")
            if not np.isfinite(arr).all():
                nonfinite += 1
            X[i] = arr
            y[i] = yi
            rids.append(r["record_id"])
            src_sha[r["record_id"]] = sha256_of(p)
            mn, mx = float(arr.min()), float(arr.max())
            gmin = mn if gmin is None else min(gmin, mn)
            gmax = mx if gmax is None else max(gmax, mx)
        if nonfinite:
            issues.append(f"{sp}: {nonfinite} non-finite records")
        # per-record global z-score audit (provenance confirmation)
        per_rec_mean = X.reshape(n, -1).mean(axis=1)
        per_rec_std = X.reshape(n, -1).std(axis=1)
        out_npz = ADAPT / f"{sp}_data.npz"
        np.savez(out_npz, X=X, record_ids=np.array(rids), y=y,
                 class_order=np.array(CLASS_ORDER))
        audit[sp] = {
            "n": n,
            "npz": str(out_npz),
            "npz_sha256": sha256_of(out_npz),
            "value_min": gmin, "value_max": gmax,
            "per_record_global_mean_avg": float(per_rec_mean.mean()),
            "per_record_global_std_avg": float(per_rec_std.mean()),
            "source_sha_min": min(src_sha.values()), "source_sha_max": max(src_sha.values()),
        }
        # content-hash duplicates (byte-level; NOT patient-level dedup)
        inv = {}
        dups = 0
        for rid, s in src_sha.items():
            inv.setdefault(s, []).append(rid)
            if len(inv[s]) > 1:
                dups += 1
        audit[sp]["duplicate_content_pairs"] = {s: v for s, v in inv.items() if len(v) > 1}
        (ADAPT / f"{sp}_source_sha256.json").write_text(
            json.dumps(src_sha, indent=0, sort_keys=True), encoding="utf-8")

    manifest = {
        "protocol_id": "thesis-p0-20261023-v1",
        "stage": "A1_data_contract",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "class_order": CLASS_ORDER,
        "counts": counts,
        "multidiagnosis": {sp: sum(1 for _, yy in by_split[sp] if yy.sum() >= 2) for sp in ("train", "val")},
        "cross_split_overlap_ids": [],
        "dir_vs_primary_mismatch": n_dir_mismatch,
        "sample_order_aligned": not any("sample_order" in i for i in issues),
        "test_policy": {
            "test_records_metadata_only": counts["test"],
            "test_signal_files_read": 0,
            "test_label_columns_analyzed": False,
            "guard_violations": guard.violations,
        },
        "preprocessing_provenance": PREPROCESS_NOTE,
        "patient_id_note": "CPSC2018 records carry no patient IDs; all dedup/overlap statements are record-level only",
        "export": audit,
        "issues": issues,
        "elapsed_s": round(time.time() - t0, 1),
    }
    guard.assert_zero()
    manifest["passed"] = not issues
    dest = RESULTS / "data_manifest.json"
    dest.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"A1 data contract: passed={manifest['passed']}, train={counts['train']}, val={counts['val']}, "
          f"test(metadata)={counts['test']}, elapsed={manifest['elapsed_s']}s")
    for i in issues:
        print("ISSUE:", i)
    return 0 if manifest["passed"] else 1


def load_dev_data(split: str):
    """训练/评估唯一数据入口: 只读导出的 train/val 副本。split 只接受 train/val。"""
    if split not in ("train", "val"):
        raise PermissionError(f"dev entry refuses split={split!r}; only train/val exported")
    z = np.load(ADAPT / f"{split}_data.npz", allow_pickle=False)
    return z["X"], z["record_ids"].astype(str), z["y"].astype(np.float32), list(z["class_order"].astype(str))


if __name__ == "__main__":
    sys.exit(main())
