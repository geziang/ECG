"""prepare_data.py — 将原始 PTB-XL 数据转换为 LFBT 论文要求的 [8, 2048] npy 格式。

用法:
    python prepare_data.py --dataset ptbxl [--subset 5000]

输出 (与 lfbt_reproduction_guide.md 第 3 节契约一致):
    data/ptbxl/{train,val,test}/{NORM,CD,HYP,MI,STTC}/*.npy   下游数据 (官方 strat_fold 划分)
    data/pt_pretrain/samples/*.npy                            预训练数据 (folds 1-8, 无标签)
    data/manifest.json                                        数据清单

预处理 (每一条记录):
    1. 读取 (12, 5000) int16 @ 500Hz
    2. 取 LFBT 8 导联: II, III, V1, V2, V3, V4, V5, V6
    3. scipy.signal.resample -> (8, 2048)
    4. 整条记录 z-score: x = (x - mean(x)) / (std(x) + 1e-5)
    5. 存 float32 .npy
"""
import argparse
import ast
import hashlib

from utils.pathguard import open_out, safe_out_path
import json
import os
import re
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.io
import scipy.signal

PREPROCESS_VERSION = "v1"
TARGET_LENGTH = 2048
SOURCE_FS = 500

# 标准 12 导联顺序: I, II, III, aVR, aVL, aVF, V1..V6 (0-based)
LFBT_LEAD_IDX = [1, 2, 6, 7, 8, 9, 10, 11]  # II, III, V1, V2, V3, V4, V5, V6
LFBT_LEAD_NAMES = ["II", "III", "V1", "V2", "V3", "V4", "V5", "V6"]

SUPERCLASSES = ["NORM", "CD", "HYP", "MI", "STTC"]
HR_PAT = re.compile(r"^HR(\d{5})\.mat$")


def build_id_to_path(ptbxl_root):
    """扫描 g1..g22 目录, 建立 ecg_id -> 文件路径 映射 (不依赖目录分组规则)。"""
    id2path = {}
    for name in sorted(os.listdir(ptbxl_root)):
        sub = os.path.join(ptbxl_root, name)
        if not (os.path.isdir(sub) and name.startswith("g")):
            continue
        for f in os.listdir(sub):
            m = HR_PAT.match(f)
            if m:
                id2path[int(m.group(1))] = os.path.join(sub, f)
    return id2path


def load_superclass_map(scp_statements_path):
    """从 scp_statements.csv 构建 scp-code -> diagnostic_class 映射。"""
    scp = pd.read_csv(scp_statements_path)
    return scp.set_index(scp.columns[0])["diagnostic_class"].to_dict()


def superclass_of(scp_codes_str, diag_map):
    """把一条记录的 scp_codes 映射为唯一的 superclass; 多标签返回 None。"""
    try:
        codes = ast.literal_eval(scp_codes_str)
    except (ValueError, SyntaxError):
        return None
    classes = set()
    for c in codes:
        cc = str(c).split(":")[0]
        if cc in diag_map and pd.notna(diag_map[cc]):
            classes.add(diag_map[cc])
    classes.discard("NORM")
    if not classes:
        return "NORM"
    if len(classes) == 1:
        return classes.pop()
    return None  # 多标签, 剔除


def process_record(args):
    """加载原始记录并转换为 [8, 2048] float32 (z-score 整条记录统一统计)。"""
    ecg_id, path = args
    raw = scipy.io.loadmat(path)["val"]  # (12, N) int16
    x = raw[LFBT_LEAD_IDX].astype(np.float64)
    if x.shape[1] != TARGET_LENGTH:
        x = scipy.signal.resample(x, TARGET_LENGTH, axis=1)
    x = (x - x.mean()) / (x.std() + 1e-5)  # 整条记录统一统计
    return ecg_id, x.astype(np.float32)


def write_npy(out_dir, filename, arr):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    np.save(path, arr)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ptbxl", choices=["ptbxl", "cpsc", "chapman"])
    ap.add_argument("--ptbxl-root", default="ptb-xl")
    ap.add_argument("--out-root", default="data")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--subset", type=int, default=None,
                    help="预训练集采样条数 (seed=0), 默认使用 folds 1-8 全部")
    args = ap.parse_args()

    if args.dataset != "ptbxl":
        raise NotImplementedError(
            f"{args.dataset} 原始数据尚未就绪; 目前仅支持 --dataset ptbxl")

    ptbxl_root = args.ptbxl_root
    db_path = os.path.join(ptbxl_root, "ptbxl_database.csv")
    scp_path = os.path.join(ptbxl_root, "scp_statements.csv")
    for p in (db_path, scp_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少官方 CSV: {p}")

    print("[1/5] 扫描数据文件 ...")
    t0 = time.time()
    id2path = build_id_to_path(ptbxl_root)
    print(f"      on-disk .mat 文件数: {len(id2path)} ({time.time()-t0:.1f}s)")

    print("[2/5] 加载官方 CSV + 标签映射 ...")
    db = pd.read_csv(db_path)
    diag_map = load_superclass_map(scp_path)
    db["superclass"] = db["scp_codes"].map(lambda s: superclass_of(s, diag_map))
    db["has_file"] = db["ecg_id"].isin(id2path)
    print(f"      CSV 行数: {len(db)}, 缺文件: {(~db['has_file']).sum()}")

    # ---- 预训练集: folds 1-8 (无标签, 排除 fold 9/10 下游患者) ----
    print("[3/5] 生成预训练集 (folds 1-8) ...")
    pretrain_df = db[(db["strat_fold"] >= 1) & (db["strat_fold"] <= 8) & db["has_file"]]
    if args.subset:
        rng = np.random.RandomState(0)
        pretrain_df = pretrain_df.iloc[rng.choice(len(pretrain_df), args.subset, replace=False)]
    pretrain_dir = os.path.join(args.out_root, "pt_pretrain", "samples")
    os.makedirs(pretrain_dir, exist_ok=True)
    pretrain_items = [(int(e), id2path[e]) for e in pretrain_df["ecg_id"]]
    print(f"      预训练样本数: {len(pretrain_items)}")

    # ---- 下游: 单标签 superclass + 官方 fold 划分 ----
    print("[4/5] 生成下游数据 (单标签, 官方 strat_fold 划分) ...")
    down_df = db[(db["superclass"].notna()) & db["has_file"]].copy()
    split_map = {"train": [1, 2, 3, 4, 5, 6, 7, 8], "val": [9], "test": [10]}
    down_items = []  # (ecg_id, path, superclass, split)
    for _, row in down_df.iterrows():
        for split, folds in split_map.items():
            if row["strat_fold"] in folds:
                down_items.append((int(row["ecg_id"]), id2path[row["ecg_id"]],
                                   row["superclass"], split))
                break
    print(f"      下游样本数: {len(down_items)}")

    # ---- 并行转换 ----
    print("[5/5] 转换 + 写入 (workers=%d) ..." % args.workers)
    down_map = {e: (p, sup, split) for e, p, sup, split in down_items}
    pretrain_ids = set(pretrain_df["ecg_id"])
    patient_map = dict(zip(db["ecg_id"], db["patient_id"]))
    conv_ids = set(down_map) | pretrain_ids
    all_items = [(int(e), id2path[e]) for e in conv_ids]

    # 跳过已存在的输出文件 (重复运行秒级完成)
    existing_ids = set()
    for split in ["train", "val", "test"]:
        for sup in SUPERCLASSES:
            d = os.path.join(args.out_root, "ptbxl", split, sup)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.endswith(".npy"):
                        existing_ids.add(int(f.split("_")[1].split(".")[0]))
    if os.path.isdir(pretrain_dir):
        for f in os.listdir(pretrain_dir):
            if f.endswith(".npy"):
                existing_ids.add(int(f.split("_")[1].split(".")[0]))
    todo_items = [it for it in all_items if it[0] not in existing_ids]
    print(f"      待转换 {len(todo_items)} (已存在 {len(all_items)-len(todo_items)})")

    t0 = time.time()
    manifest_records = []
    converted = set()
    if todo_items:
        with Pool(args.workers) as pool:
            for n, (ecg_id, x) in enumerate(pool.imap_unordered(process_record, todo_items, chunksize=16)):
                if ecg_id in down_map:
                    _, sup, split = down_map[ecg_id]
                    write_npy(os.path.join(args.out_root, "ptbxl", split, sup),
                              f"sample_{ecg_id:05d}.npy", x)
                if ecg_id in pretrain_ids:
                    write_npy(pretrain_dir, f"sample_{ecg_id:05d}.npy", x)
                converted.add(ecg_id)
                if (n + 1) % 2000 == 0:
                    print(f"      已转换 {n+1}/{len(todo_items)} ({time.time()-t0:.0f}s)")

    # ---- manifest.json (覆盖全部 conv_ids, 不依赖是否重新转换) ----
    for ecg_id in sorted(conv_ids):
        rec = dict(ecg_id=int(ecg_id), patient_id=int(patient_map[ecg_id]),
                   source_file=id2path[ecg_id], shape=[8, 2048],
                   sampling_rate_hz=SOURCE_FS, preprocess_version=PREPROCESS_VERSION)
        if ecg_id in down_map:
            _, sup, split = down_map[ecg_id]
            rec.update(split=split, label=sup)
        else:
            rec.update(split="pretrain", label=None)
        rec["in_pretrain"] = ecg_id in pretrain_ids
        manifest_records.append(rec)

    # ---- manifest.json ----
    manifest = dict(
        dataset="PTB-XL", csv_version="1.0.3", on_disk_version="1.0.1 (500Hz)",
        input_shape=[8, 2048], lead_order=LFBT_LEAD_NAMES, sampling_rate_hz=SOURCE_FS,
        duration_seconds=10, zscore="whole-record", preprocess_version=PREPROCESS_VERSION,
        class_order=sorted(SUPERCLASSES),  # DatasetFolder 按排序目录名分配 class index
        pretrain_folds=[1, 2, 3, 4, 5, 6, 7, 8], downstream_folds={"train": [1, 2, 3, 4, 5, 6, 7, 8],
        "val": [9], "test": [10]},
        records=manifest_records,
    )
    manifest_path = safe_out_path(args.out_root, "manifest.json")
    with open_out(args.out_root, "manifest.json", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    # 汇总
    counts = {}
    for split in ["train", "val", "test"]:
        counts[split] = sum(1 for r in manifest_records if r["split"] == split)
    counts["pretrain"] = sum(1 for r in manifest_records if r["in_pretrain"])
    print("\n===== 完成 =====")
    print(f"耗时 {time.time()-t0:.0f}s | manifest: {manifest_path}")
    print("样本数:", counts)
    for split in ["train", "val", "test"]:
        sub = {s: sum(1 for r in manifest_records if r["split"] == split and r["label"] == s)
               for s in SUPERCLASSES}
        print(f"  {split}: {sub}")


if __name__ == "__main__":
    main()
