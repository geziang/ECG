"""prepare_multilib.py — W0-6: Challenge2020 四库(五个子库)入库为 N1 预训练语料。

复用源仓库 prepare_cpsc.py 的 header 解析 / 中心裁剪逻辑;
z-score 口径与冻结协议 prepare_data.py 完全一致(不换算物理单位, 整条记录统一标量统计),
保证与 data/pt_pretrain 语料同分布处理。

预处理 (每条记录, 确定性):
    1. 解析 .hea: 采样率 / 导联名 / Dx SNOMED 码 (逐记录 fs, 兼容 incart 257Hz / ptb 1000Hz)
    2. loadmat 取 8 导联 (按导联名, 不按数组位): II, III, V1..V6
    3. 中心裁剪 10s (native fs; 不足对称补零, 记录 valid 区)
    4. z-score: valid 区整条记录统一标量统计 (x-mean)/(std+1e-5), 常数导联置 0
    5. resample -> [8, 2048] float32
输出:
    data/multilib/samples/{lib}__{record_id}.npy
    data/multilib/manifest.json   (逐条 fs/长度/常数导联/Dx/SHA256, manifest v2)
红线: ptb-xl 禁入 (与下游同源); 不做 train/test 划分 (纯预训练语料)。
"""
import argparse
import hashlib
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample

LEAD_ORDER = ["II", "III", "V1", "V2", "V3", "V4", "V5", "V6"]
TARGET_LENGTH = 2048
CROP_SECONDS = 10.0
PREPROCESS_VERSION = "n1-multilib-1"
LIBS = {
    "cpsc_2018": None,          # 值为 None: 从 source-dir 下子目录名识别
    "cpsc_2018_extra": None,
    "georgia": None,
    "ptb": None,
    "st_petersburg_incart": None,
}
EXCLUDED = {"ptb-xl": "与下游 PTB-XL 同源, 禁入预训练 (红线)"}

LABEL_CODES = (
    270492004, 164889003, 426783006, 429622005, 164884008,
    284470004, 164909002, 164931005, 59118001,
)
CODE_TO_INDEX = {c: i for i, c in enumerate(LABEL_CODES)}


def parse_header(path: Path):
    leads, dx_codes = [], []
    fs = sample_count = None
    gains_ok = True
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_no, line in enumerate(stream):
            s = line.strip()
            if line_no == 0:
                tokens = s.split()
                fs = float(tokens[2].split("/")[0])
                sample_count = int(tokens[3])
            elif s.startswith("#"):
                if s.startswith("# Dx:"):
                    dx_codes = [int(t) for t in s.split("# Dx:", 1)[1].replace(",", " ").split() if t]
            elif s:
                parts = s.split()
                if len(parts) >= 9 and parts[0].endswith(".mat"):
                    leads.append(parts[-1])
                    try:
                        gain = float(parts[2].split("/")[0].split("(")[0])
                        if not np.isfinite(gain) or gain <= 0:
                            gains_ok = False
                    except ValueError:
                        gains_ok = False
    if fs is None or sample_count is None:
        raise ValueError(f"missing metadata: {path}")
    return leads, dx_codes, fs, sample_count, gains_ok


def label_vector(dx_codes):
    v = [0] * len(LABEL_CODES)
    for c in dx_codes:
        if c in CODE_TO_INDEX:
            v[CODE_TO_INDEX[c]] = 1
    return v


def center_crop_or_pad(sig, target_len):
    n = sig.shape[1]
    if n >= target_len:
        start = (n - target_len) // 2
        return sig[:, start:start + target_len], 0, target_len
    pad = target_len - n
    left = pad // 2
    out = np.pad(sig, ((0, 0), (left, pad - left)), mode="constant")
    return out, left, n


def process_one(job):
    lib, hea_path, out_dir = job
    rid = hea_path.stem
    leads, dx_codes, fs, hdr_n, gains_ok = parse_header(hea_path)
    idx = {n: i for i, n in enumerate(leads)}
    missing = [n for n in LEAD_ORDER if n not in idx]
    if missing:
        return {"lib": lib, "record_id": rid, "error": f"missing leads {missing}"}
    raw = loadmat(str(hea_path.with_suffix(".mat")))["val"]
    if raw.ndim != 2 or raw.shape[1] != hdr_n:
        return {"lib": lib, "record_id": rid, "error": f"shape {raw.shape} vs header {hdr_n}"}
    x = raw[[idx[n] for n in LEAD_ORDER]].astype(np.float64)
    if not np.isfinite(x).all():
        return {"lib": lib, "record_id": rid, "error": "NaN/Inf in raw"}

    crop_n = int(round(CROP_SECONDS * fs))
    cropped, valid_start, valid_n = center_crop_or_pad(x, crop_n)
    valid = cropped[:, valid_start:valid_start + valid_n]
    std = valid.std()
    if std < 1e-8:
        out = np.zeros_like(cropped)
        const_leads = list(LEAD_ORDER)
    else:
        mu = valid.mean()
        out = (cropped - mu) / (std + 1e-5)
        out[:, :valid_start] = 0.0
        out[:, valid_start + valid_n:] = 0.0
        lead_std = valid.std(axis=1)
        const_leads = [LEAD_ORDER[i] for i in np.where(lead_std < 1e-8)[0]]
        for i in np.where(lead_std < 1e-8)[0]:
            out[i, :] = 0.0
    y = resample(out, TARGET_LENGTH, axis=1)
    if not np.isfinite(y).all():
        return {"lib": lib, "record_id": rid, "error": "NaN/Inf after resample"}

    out_path = Path(out_dir) / f"{lib}__{rid}.npy"
    np.save(out_path, y.astype(np.float32))
    digest = hashlib.sha256()
    with out_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "lib": lib, "record_id": rid, "file": out_path.name,
        "fs": fs, "original_length": int(hdr_n),
        "crop_samples": crop_n, "valid_length": int(valid_n),
        "gains_ok": bool(gains_ok), "constant_leads": const_leads,
        "dx": dx_codes, "labels": label_vector(dx_codes),
        "sha256": digest.hexdigest(),
    }


def main():
    ap = argparse.ArgumentParser(description="N1 multilib pretrain corpus preparation")
    ap.add_argument("--source-dir", type=Path, required=True,
                    help="Challenge2020 training 根目录 (含 cpsc_2018/ 等子库)")
    ap.add_argument("--output-dir", type=Path, default=Path("data/multilib"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true", help="跳过已有 npy (仍登记 manifest)")
    args = ap.parse_args()

    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    jobs, skipped = [], []
    for lib in LIBS:
        lib_dir = args.source_dir / lib
        if not lib_dir.is_dir():
            raise FileNotFoundError(lib_dir)
        for hea in sorted(lib_dir.rglob("*.hea")):
            out = samples_dir / f"{lib}__{hea.stem}.npy"
            if args.resume and out.exists():
                skipped.append((lib, hea.stem))
                continue
            jobs.append((lib, hea, samples_dir))

    print(f"待处理 {len(jobs)} 条 (resume 跳过 {len(skipped)})", flush=True)
    t0 = time.time()
    results, errors = [], []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_one, jobs, chunksize=32), 1):
            if "error" in r:
                errors.append(r)
            else:
                results.append(r)
            if i % 500 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] elapsed {time.time()-t0:.0f}s errors={len(errors)}", flush=True)

    # resume 的旧文件补登记
    for lib, rid in skipped:
        out = samples_dir / f"{lib}__{rid}.npy"
        digest = hashlib.sha256()
        with out.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        arr = np.load(out)
        results.append({"lib": lib, "record_id": rid, "file": out.name,
                        "sha256": digest.hexdigest(), "resume": True,
                        "shape": list(arr.shape)})

    by_lib = {}
    for r in results:
        by_lib.setdefault(r["lib"], 0)
        by_lib[r["lib"]] += 1
    manifest = {
        "purpose": "N1 multilib pretrain corpus (W0-6)",
        "preprocess_version": PREPROCESS_VERSION,
        "input_shape": [len(LEAD_ORDER), TARGET_LENGTH],
        "lead_order": LEAD_ORDER,
        "crop_seconds": CROP_SECONDS,
        "zscore": "whole-record scalar on valid region (与冻结协议 prepare_data.py 一致, 不换算增益)",
        "excluded": EXCLUDED,
        "label_codes": list(LABEL_CODES),
        "counts": by_lib,
        "total": len(results),
        "errors": errors,
        "records": results,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"counts": by_lib, "total": len(results),
                      "errors": len(errors)}, ensure_ascii=False))
    if errors:
        print("首批错误:", [f"{e['lib']}/{e['record_id']}: {e['error']}" for e in errors[:10]])


if __name__ == "__main__":
    main()
