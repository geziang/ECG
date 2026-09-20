"""prepare_cpsc_chapman.py — C3 外部下游库预处理 (任务书 T4/C3, 2026-09-21 晨)。

用法:
    python prepare_cpsc_chapman.py --which both   # cpsc | chapman | both

输出 (与 prepare_data.py 的 ptbxl 契约一致, run_lp 可直接消费):
    data/cpsc/{train,val,test}/{NSR,AF,IAVB,LBBB,RBBB,PAC,PVC,STD,STE}/*.npy   9 类
    data/chapman/{train,val,test}/{SB,AFIB,GSVT,SR}/*.npy                     4 类
    runlog/W1/cpsc_manifest.json / chapman_manifest.json

预处理: 中心 10s(5000 样本@500Hz)裁剪(不足则尾部零填) -> LFBT 8 导联 -> resample 2048
        -> 整条 z-score -> float32。标签取自 .hea 的 Dx(SNOMED), 依官方 Dx_map.csv
        (physionetchallenges 2020, 2026-09-21 经 raw.githubusercontent 核对原文)。

标签映射(预注册):
  CPSC 9 类(官方 CPSC2018 类别; 多类命中按 异常优先 Normal 最后):
    NSR=426783006; AF=164889003; IAVB=270492004; LBBB=164909002;
    RBBB={59118001,713427006}; PAC={284470004,63593006}; PVC={427172004,17338001};
    STD=429622005; STE=164931005
    优先级: AF>STE>STD>PVC>PAC>RBBB>LBBB>IAVB>NSR; 无可映射码 -> 剔除计数
  Chapman 4 类(基于官方缩写预注册; SB 计数 3889 与 Zheng et al. 2020 精确吻合):
    SB=426177001; AFIB={164889003,164890007}; GSVT={427084000,426761007,67198005,713422000};
    SR=426783006; 优先级: AFIB>GSVT>SB>SR
划分: 类别分层 70/10/20, seed 0, 记录级(两库均每记录一患者)。
"""
import argparse
import collections
import json
import random
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.io
import scipy.signal

PREPROCESS_VERSION = "csc-v1"
LFBT_LEAD_IDX = [1, 2, 6, 7, 8, 9, 10, 11]

CPSC_CLASSES = ["NSR", "AF", "IAVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
CPSC_MAP = {
    "426783006": "NSR", "164889003": "AF", "270492004": "IAVB", "164909002": "LBBB",
    "59118001": "RBBB", "713427006": "RBBB", "284470004": "PAC", "63593006": "PAC",
    "427172004": "PVC", "17338001": "PVC", "164884008": "PVC",  # 164884008=VEB: cpsc_2018 打包版 PVC 实际用码(全库仅 9 码核实)
    "429622005": "STD", "164931005": "STE",
}
CPSC_PRIORITY = ["AF", "STE", "STD", "PVC", "PAC", "RBBB", "LBBB", "IAVB", "NSR"]

CHAP_CLASSES = ["SB", "AFIB", "GSVT", "SR"]
CHAP_MAP = {
    "426177001": "SB", "164889003": "AFIB", "164890007": "AFIB",
    "427084000": "GSVT", "426761007": "GSVT", "67198005": "GSVT", "713422000": "GSVT",
    "426783006": "SR",
}
CHAP_PRIORITY = ["AFIB", "GSVT", "SB", "SR"]

DX_PAT = re.compile(r"#\s*Dx:\s*(.*)")


def parse_args_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["cpsc", "chapman", "both"], default="both")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out-root", default="data")
    ap.add_argument("--manifest-dir", default="runlog/W1")
    return ap.parse_args()


def label_of(hea_txt, cmap, priority):
    m = DX_PAT.search(hea_txt)
    if not m:
        return None
    classes = {cmap[c.strip()] for c in m.group(1).split(",") if c.strip() in cmap}
    if not classes:
        return None
    for p in priority:
        if p in classes:
            return p
    return None


def process_one(job):
    mat_path, cls = job
    try:
        v = scipy.io.loadmat(str(mat_path))["val"]
        if v.shape[0] != 12:
            return cls, None, "bad_shape"
        x = v[LFBT_LEAD_IDX].astype(np.float64)
        T = x.shape[1]
        if T >= 5000:                      # 中心 10s
            s = (T - 5000) // 2
            x = x[:, s:s + 5000]
        else:                              # 不足 10s 尾部零填
            x = np.pad(x, ((0, 0), (0, 5000 - T)))
        if not np.isfinite(x).all():
            return cls, None, "non_finite"
        x = scipy.signal.resample(x, 2048, axis=1)
        x = (x - x.mean()) / (x.std() + 1e-5)
        return cls, x.astype(np.float32), "ok"
    except Exception:
        return cls, None, "read_fail"


def build(root_glob, cmap, priority, classes, name, args):
    files = sorted(Path(p) for p in __import__("glob").glob(str(root_glob), recursive=True))
    t0 = time.time()
    jobs, labels = [], {}
    excl_unmapped = 0
    for f in files:
        if f.suffix != ".hea":
            continue
        hea = open(f, encoding="ascii", errors="ignore").read()
        cls = label_of(hea, cmap, priority)
        if cls is None:
            excl_unmapped += 1
            continue
        labels[f.stem] = cls
        jobs.append((f.with_suffix(".mat"), cls))
    stats = collections.Counter()
    with Pool(args.workers) as pool:
        results = pool.map(process_one, jobs, chunksize=64)
    by_cls = collections.defaultdict(list)
    for (mat_path, cls), (cls2, arr, status) in zip(jobs, results):
        stats[status] += 1
        if arr is not None:
            by_cls[cls].append((mat_path.stem, arr))
    rng = random.Random(0)
    counts = {"train": collections.Counter(), "val": collections.Counter(), "test": collections.Counter()}
    for cls in classes:
        items = by_cls.get(cls, [])
        rng.shuffle(items)
        n = len(items)
        n_tr, n_va = int(n * 0.7), int(n * 0.1)
        for split, seg in (("train", items[:n_tr]), ("val", items[n_tr:n_tr + n_va]),
                           ("test", items[n_tr + n_va:])):
            d = Path(args.out_root) / name / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for stem, arr in seg:
                np.save(d / f"{stem}.npy", arr)
            counts[split][cls] = len(seg)
    manifest = {
        "preprocess_version": PREPROCESS_VERSION,
        "n_hea": len(jobs) + excl_unmapped, "n_labeled": len(jobs),
        "n_unmapped_excluded": excl_unmapped,
        "n_processed": dict(stats),
        "class_counts": {c: {"train": counts["train"][c], "val": counts["val"][c],
                             "test": counts["test"][c]} for c in classes},
        "split": "stratified 70/10/20 seed=0, record-level (1 record = 1 patient)",
        "signal": "central 10s crop @500Hz (short->zero-pad), LFBT 8 leads, resample 2048, z-score",
        "wall_time_s": round(time.time() - t0, 1),
    }
    Path(args.manifest_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.manifest_dir) / f"{name}_manifest.json", "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=1)
    print(f"[{name}] ok={stats['ok']} excl_unmapped={excl_unmapped} "
          f"other={dict((k,v) for k,v in stats.items() if k!='ok')} 用时{manifest['wall_time_s']}s")
    for c in classes:
        print(f"    {c}: {manifest['class_counts'][c]}")


def main():
    args = parse_args_()
    if args.which in ("cpsc", "both"):
        build(r"F:\数据集\classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2\training\cpsc_2018\g*\*.hea",
              CPSC_MAP, CPSC_PRIORITY, CPSC_CLASSES, "cpsc", args)
    if args.which in ("chapman", "both"):
        build(r"F:\数据集\ECG_data\chapman_shaoxing\g*\*.hea",
              CHAP_MAP, CHAP_PRIORITY, CHAP_CLASSES, "chapman", args)


if __name__ == "__main__":
    main()
