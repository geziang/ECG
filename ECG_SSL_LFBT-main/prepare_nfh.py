"""prepare_nfh.py — 外部 NFH(Ningbo First Hospital) 预训练库预处理 (任务书 T4 / C1)。

用法:
    python prepare_nfh.py --src "F:/数据集/ECG_data/ningbo" [--workers 14] [--limit 0]

输出 (与 prepare_data.py 的 pt_pretrain 契约逐位一致, run_pt 无需改动):
    data/pt_pretrain_nfh/samples/sample_XXXXX.npy   [8, 2048] float32, z-score
    runlog/W1/nfh_manifest.json                      审计清单(计数/导联映射/哈希摘要/排除明细)
    runlog/W1/nfh_sha256.txt                         每条记录 .mat 内容 SHA256 (查重用)

预处理 (每一条记录, 与 PTB-XL v1 口径一致):
    1. 读 .mat['val'] (12, 5000) int16 @ 500Hz (标准导联序 I,II,III,aVR,aVL,aVF,V1..V6)
    2. 取 LFBT 8 导联: II, III, V1..V6 (idx [1,2,6..11])
    3. scipy.signal.resample -> (8, 2048)
    4. 整条记录 z-score: x = (x - mean(x)) / (std(x) + 1e-5)
    5. 存 float32 .npy

排除规则 (预注册, 只剔硬失败; 计数全部入 manifest):
    - .mat 读取失败 / 'val' 缺失            -> read_fail
    - 形状不是 (12, 5000)                    -> bad_shape
    - 含非有限值 (nan/inf)                   -> non_finite
    - 全记录幅值 std < 1e-6 (全平)           -> flat_signal

matched-updates 协议 (任务书 §5.2): NFH 34,905 条 ≈ PTB 17,418 的 2.004 倍,
C1 预训练用 --epochs 100 (NFH) 对齐 B0 的 200 epochs (PTB) 的 optimizer updates
(≈27.2k steps, 差 <0.3%)。manifest 记录该换算。

NFH 无患者 ID (.hea 仅 Age/Sex/Dx): 患者交叉审计降级为内容级查重——
SHA256 全库查重 + 与 chapman_shaoxing (JS00001~JS10646 区间互斥) 的范围审计。
"""
import argparse
import hashlib
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.io
import scipy.signal

PREPROCESS_VERSION = "nfh-v1"
TARGET_LENGTH = 2048
SOURCE_FS = 500
# 标准 12 导联顺序 (0-based): I, II, III, aVR, aVL, aVF, V1..V6
LFBT_LEAD_IDX = [1, 2, 6, 7, 8, 9, 10, 11]  # II, III, V1, V2, V3, V4, V5, V6
LFBT_LEAD_NAMES = ["II", "III", "V1", "V2", "V3", "V4", "V5", "V6"]


def process_one(args):
    """返回 (record_name, sha256, status, out_array_or_None)。status: ok|read_fail|bad_shape|non_finite|flat_signal"""
    mat_path, rel = args
    try:
        h = hashlib.sha256()
        with open(mat_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest()
        d = scipy.io.loadmat(str(mat_path))
        if "val" not in d:
            return rel, sha, "read_fail", None
        v = d["val"]
        if v.shape != (12, 5000):
            return rel, sha, "bad_shape", None
        x = v[LFBT_LEAD_IDX].astype(np.float64)
        if not np.isfinite(x).all():
            return rel, sha, "non_finite", None
        if x.std() < 1e-6:
            return rel, sha, "flat_signal", None
        x = scipy.signal.resample(x, TARGET_LENGTH, axis=1)
        x = (x - x.mean()) / (x.std() + 1e-5)
        return rel, sha, "ok", x.astype(np.float32)
    except Exception:
        return rel, None, "read_fail", None


def main():
    ap = argparse.ArgumentParser(description="NFH 预训练库预处理 (T4/C1)")
    ap.add_argument("--src", type=str, required=True, help="ningbo 根目录 (含 g1..g35)")
    ap.add_argument("--out", type=str, default="data/pt_pretrain_nfh")
    ap.add_argument("--manifest-dir", type=str, default="runlog/W1")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只处理前 N 条 (冒烟用)")
    args = ap.parse_args()

    src = Path(args.src)
    mats = sorted(src.glob("g*/*.mat"))
    if args.limit > 0:
        mats = mats[: args.limit]
    print(f"[nfh] 源记录数: {len(mats)}")

    out_dir = Path(args.out) / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.manifest_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    jobs = [(m, m.stem) for m in mats]
    stats = {"ok": 0, "read_fail": 0, "bad_shape": 0, "non_finite": 0, "flat_signal": 0}
    excl = {k: [] for k in stats if k != "ok"}
    sha_list = []
    n_ok = 0
    with Pool(args.workers) as pool:
        for i, (rel, sha, status, arr) in enumerate(pool.imap_unordered(process_one, jobs, chunksize=64)):
            stats[status] += 1
            if status != "ok":
                excl[status].append(rel)
                continue
            np.save(out_dir / f"sample_{n_ok + 1:05d}.npy", arr)
            sha_list.append((rel, sha))
            n_ok += 1
            if (i + 1) % 5000 == 0:
                print(f"[nfh] {i + 1}/{len(jobs)} 完成, 用时 {time.time() - t0:.0f}s")

    # 内容级查重 (NFH 无患者 ID, 降级为 SHA256 查重)
    seen, dups = {}, []
    for rel, sha in sha_list:
        if sha in seen:
            dups.append(f"{seen[sha]}=={rel}")
        else:
            seen[sha] = rel

    manifest = {
        "preprocess_version": PREPROCESS_VERSION,
        "source": str(src),
        "source_fs_hz": SOURCE_FS,
        "target_length": TARGET_LENGTH,
        "lead_mapping": {n: i for n, i in zip(LFBT_LEAD_NAMES, LFBT_LEAD_IDX)},
        "normalization": "per-record z-score (mean/std over full 8x2048), eps=1e-5, float32",
        "n_source": len(mats),
        "n_ok": stats["ok"],
        "n_excluded": {k: v for k, v in stats.items() if k != "ok"},
        "excluded_records": {k: v[:200] for k, v in excl.items()},  # 明细截断, 全量计数为准
        "n_sha256_duplicates": len(dups),
        "sha256_duplicates": dups[:200],
        "patient_audit_note": "NFH .hea 无患者 ID; 以 SHA256 内容查重替代; "
                              "chapman_shaoxing(JS00001~JS10646) 与 ningbo(JS10647+) 编号区间互斥",
        "matched_updates": {
            "ptb_b0": "200 epochs x 17418/128 ~= 27216 steps",
            "nfh_c1": "100 epochs x {}/128 ~= {:.0f} steps".format(stats["ok"], 100 * stats["ok"] / 128),
            "note": "C1 预训练用 --epochs 100 对齐 optimizer updates",
        },
        "wall_time_s": round(time.time() - t0, 1),
    }
    with open(Path(args.manifest_dir) / "nfh_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    with open(Path(args.manifest_dir) / "nfh_sha256.txt", "w") as f:
        for rel, sha in sha_list:
            f.write(f"{sha}  {rel}\n")

    print(f"[nfh] 完成: ok={stats['ok']} 排除={manifest['n_excluded']} 查重dup={len(dups)} "
          f"用时 {manifest['wall_time_s']}s -> {out_dir}")


if __name__ == "__main__":
    main()
