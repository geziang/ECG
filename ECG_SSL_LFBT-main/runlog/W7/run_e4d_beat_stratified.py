# -*- coding: utf-8 -*-
"""W7 E4d: 心拍先验分析层——三域 test 逐记录 RR 特征 × W6/W5 预测的分层误差。

阶段 A: 三域 test R 峰检测缓存 data/pt_rpeaks_test_{domain}.npz (幂等可续, 键=stem 值=int64 索引,
        口径=W1 prep_rpeaks: gqrs V5(idx6)→II(idx0) 回退, fs=204.8, 峰数<4 标空)。
阶段 B: 逐预测目录 (W6 {b0,c1,c2}_{dom}_{eval}_seed{S} + seedext_* + W5 {m}_cpsc_seed{S})
        重建 DatasetFolder 顺序 → 对齐 → RR 特征分桶(meanRR/SDNN 四分位+no_rr) → 逐桶 acc。
产物: runlog/W7/e4d_rr_features.csv, e4d_bucket_summary.csv, e4d_delta_summary.csv, e4d_summary.md
性质: 描述性分析(只读预测与数据, 不训练不评新 test 格, 无晋级门)。
用法: python runlog/W7/run_e4d_beat_stratified.py [--skip-cache]
"""
import argparse
import csv
import io
import json
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd().resolve()
if not (ROOT / "runlog/W7").is_dir():  # 文档用法=从仓库根启动(见文件头)
    raise SystemExit(f"please run from repo root, cwd={ROOT}")
sys_path_hack = ROOT  # noqa
import sys
sys.path.insert(0, str(ROOT))

DOMAINS = ["ptbxl", "cpsc", "chapman"]
W6_PRED = ROOT / "runlog/W6/predictions"
W5_PRED = ROOT / "runlog/W5/predictions"
OUTD = ROOT / "runlog/W7"


def _safe(p):
    q = Path(p).resolve()
    if not q.is_relative_to(ROOT):
        raise ValueError(f"path escapes repo root: {q}")
    return q


FS = 204.8  # 2048 点 / 10 s


def log(msg):
    print(f"[e4d {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---------- 阶段 A: R 峰缓存 ----------

def detect_one(sig):
    import wfdb.processing as wp
    for lead_idx in (6, 0):  # V5 优先, II 回退 (W1 口径)
        try:
            pk = wp.gqrs_detect(sig[lead_idx].astype(np.float64), fs=FS)
        except Exception:
            pk = np.empty(0, dtype=np.int64)
        if len(pk) >= 4:
            return np.sort(pk.astype(np.int64))
    return np.empty(0, dtype=np.int64)


def build_cache(domain):
    out = ROOT / f"data/pt_rpeaks_test_{domain}.npz"
    cache = {}
    if out.exists():  # 幂等: 已有则只补缺
        cache = dict(np.load(out, allow_pickle=True))
    test_root = ROOT / f"data/{domain}/test"
    files = sorted(test_root.rglob("*.npy"))
    todo = [f for f in files if f.stem not in cache]
    log(f"cache {domain}: {len(files)} test files, {len(todo)} to detect")
    t0 = time.time()
    for i, f in enumerate(todo):
        sig = np.load(f)
        cache[f.stem] = detect_one(sig)
        if (i + 1) % 500 == 0:
            log(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)")
    np.savez(out, **cache)
    n_empty = sum(1 for v in cache.values() if len(v) == 0)
    log(f"cache {domain}: done {len(cache)} records, {n_empty} no-rpeak")
    return cache, n_empty


def rr_features(pk):
    """pk: R 峰索引(2048 坐标)。返回 dict(ms 单位) 或 None(峰不足)。"""
    if pk is None or len(pk) < 4:
        return None
    rr = np.diff(pk) / FS * 1000.0  # ms
    rr = rr[(rr > 300) & (rr < 2000)]  # 生理范围过滤(30-200bpm)
    if len(rr) < 3:
        return None
    d = np.abs(np.diff(rr))
    return {
        "n_beats": len(pk),
        "mean_rr_ms": float(np.mean(rr)),
        "sdnn_ms": float(np.std(rr, ddof=1)),
        "rmssd_ms": float(np.sqrt(np.mean(d ** 2))) if len(d) else float("nan"),
        "hr_bpm": float(60000.0 / np.mean(rr)),
    }


# ---------- 阶段 B: 预测对齐与分桶 ----------

def test_order(domain):
    """重建 DatasetFolder 顺序: 类字母序 × 类内文件序。返回 [(stem, class_name)]。"""
    root = ROOT / f"data/{domain}/test"
    classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    order = []
    for c in classes:
        for f in sorted((root / c).glob("*.npy")):
            order.append((f.stem, c))
    return order, classes


def quartile_edges(values):
    """对有效值取四分位边界(去重保序)。"""
    q = np.unique(np.nanpercentile(values, [25, 50, 75]))
    return list(q)


def bucket_of(v, edges):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "no_rr"
    for i, e in enumerate(edges):
        if v <= e:
            return f"q{i + 1}"
    return f"q{len(edges) + 1}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cache", action="store_true")
    args = ap.parse_args()

    _safe(OUTD).mkdir(exist_ok=True)
    # ---- 阶段 A
    caches, features = {}, {}
    feat_rows = []
    for dom in DOMAINS:
        out_npz = ROOT / f"data/pt_rpeaks_test_{dom}.npz"
        if args.skip_cache and out_npz.exists():
            cache = dict(np.load(out_npz, allow_pickle=True))
            log(f"cache {dom}: reuse existing ({len(cache)})")
        else:
            cache, _ = build_cache(dom)
        caches[dom] = cache
        order, _ = test_order(dom)
        for stem, cls in order:
            f = rr_features(cache.get(stem))
            if f is None:
                f = {"n_beats": 0, "mean_rr_ms": float("nan"), "sdnn_ms": float("nan"),
                     "rmssd_ms": float("nan"), "hr_bpm": float("nan")}
            features[(dom, stem)] = f
            feat_rows.append(dict(domain=dom, stem=stem, cls=cls, **f))
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(feat_rows[0].keys()))
    w.writeheader()
    w.writerows(feat_rows)
    _safe(OUTD / "e4d_rr_features.csv").write_text(buf.getvalue(), encoding="utf-8", newline="")
    log(f"features csv: {len(feat_rows)} rows")

    # 桶边界(按域, 用全体 test 记录分布; no_rr 单列)
    edges = {}
    for dom in DOMAINS:
        order, _ = test_order(dom)
        for feat in ("mean_rr_ms", "sdnn_ms"):
            vals = [features[(dom, s)][feat] for s, _ in order]
            edges[(dom, feat)] = quartile_edges([v for v in vals if not np.isnan(v)])
    log("edges: " + json.dumps({f"{k[0]}:{k[1]}": [round(x, 1) for x in v] for k, v in edges.items()}))

    # ---- 收集预测目录
    dirs = []
    if W6_PRED.exists():
        for d in sorted(W6_PRED.iterdir()):
            m = re.match(r"^(b0|c1|c2)_(ptbxl|cpsc|chapman)_(lp|ft10|ft20|ft40)_seed(\d+)$", d.name)
            if m:
                dirs.append((d, m.group(1), m.group(2), m.group(3), m.group(4)))
            m2 = re.match(r"^seedext_(b0|c1|c2)_(ptbxl|cpsc|chapman)_lp_seed(\d+)$", d.name)
            if m2:
                dirs.append((d, m2.group(1), m2.group(2), "lp", m2.group(3)))
    if W5_PRED.exists():
        for d in sorted(W5_PRED.iterdir()):
            m = re.match(r"^(b0|c1|c2)_cpsc_seed(\d+)$", d.name)
            if m:
                dirs.append((d, m.group(1), "cpsc", "lp_w5", m.group(2)))
    log(f"prediction dirs: {len(dirs)}")

    rows, order_cache = [], {}

    def get_order(dom):
        if dom not in order_cache:
            order_cache[dom] = test_order(dom)
        return order_cache[dom]

    acc_check = []
    for d, model, dom, ev, seed in dirs:
        try:
            y_true = np.load(d / "y_true.npy")
            y_pred = np.load(d / "y_pred.npy")
        except Exception as e:
            log(f"SKIP {d.name}: {e}")
            continue
        order, _ = get_order(dom)
        if len(order) != len(y_true):
            log(f"SKIP {d.name}: len mismatch {len(order)} vs {len(y_true)}")
            continue
        mj = json.loads((d / "metrics_ext.json").read_text(encoding="utf-8")) if (d / "metrics_ext.json").exists() else {}
        ref_acc = mj.get("accuracy")
        acc = float(np.mean(y_true == y_pred))
        if ref_acc is not None and abs(acc - ref_acc) > 1e-9:
            acc_check.append(f"{d.name}: ours={acc:.6f} ext={ref_acc:.6f} MISMATCH")
        correct = (y_true == y_pred).astype(float)
        for feat in ("mean_rr_ms", "sdnn_ms"):
            ed = edges[(dom, feat)]
            buckets = [bucket_of(features[(dom, s)][feat], ed) for s, _ in order]
            for b in sorted(set(buckets)):
                idx = [i for i, bb in enumerate(buckets) if bb == b]
                rows.append(dict(
                    pred_dir=d.name, model=model, domain=dom, eval=ev, seed=seed,
                    feature=feat, bucket=b, n=len(idx),
                    acc=float(np.mean(correct[idx])),
                ))
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    _safe(OUTD / "e4d_bucket_summary.csv").write_text(buf.getvalue(), encoding="utf-8", newline="")
    log(f"bucket rows: {len(rows)}")

    # ---- 配对差: b0 vs c2 / c2 vs c1 (同 domain/eval/seed) 逐桶
    acc_map = {}
    for r in rows:
        acc_map[(r["domain"], r["eval"], r["seed"], r["feature"], r["bucket"], r["model"])] = (r["acc"], r["n"])
    deltas = []
    for (dom, ev, seed, feat, b, m), (a, n) in acc_map.items():
        for other in ("c2", "c1"):
            if m != "b0":
                continue
            hit = acc_map.get((dom, ev, seed, feat, b, other))
            if hit:
                deltas.append(dict(domain=dom, eval=ev, seed=seed, feature=feat, bucket=b,
                                   n=n, pair=f"b0-{other}", acc_b0=a, acc_other=hit[0],
                                   delta=hit[0] - a))
    if deltas:
        buf = io.StringIO(newline="")
        w = csv.DictWriter(buf, fieldnames=list(deltas[0].keys()))
        w.writeheader()
        w.writerows(deltas)
        _safe(OUTD / "e4d_delta_summary.csv").write_text(buf.getvalue(), encoding="utf-8", newline="")
    log(f"delta rows: {len(deltas)}")
    # delta 汇总(暂存, 摘要块拼接): 按 (domain, eval, feature, bucket, pair) 对 seed 取均值
    delta_md = []
    if deltas:
        delta_md += ["\n## 逐桶配对差 (对 seed 取均值, delta=acc_other − acc_b0)\n",
                     "| domain | eval | feature | bucket | pair | n | mean_delta |",
                     "|---|---|---|---|---|---|---|"]
        groups, nmap = {}, {}
        for d0 in deltas:
            k5 = (d0["domain"], d0["eval"], d0["feature"], d0["bucket"], d0["pair"])
            groups.setdefault(k5, []).append(d0["delta"])
            nmap[k5] = d0["n"]
        for k in sorted(groups):
            delta_md.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {nmap[k]} | {np.mean(groups[k]):+.4f} |")

    # ---- 摘要
    lines = ["# E4d 心拍分层误差分析摘要", "",
             f"- 记录特征行: {len(feat_rows)}; 预测目录: {len(dirs)}; 对账: " +
             ("全部一致" if not acc_check else f"{len(acc_check)} 处不一致(见下)")]
    lines += [f"  - {x}" for x in acc_check[:8]]
    lines.append("")
    lines += delta_md
    # 代表读数: cpsc ft10, mean_rr 四桶, b0/c2 各 seed 均值
    for dom, ev in [("cpsc", "ft10"), ("cpsc", "lp"), ("ptbxl", "ft20"), ("chapman", "ft10")]:
        sub = [r for r in rows if r["domain"] == dom and r["eval"] == ev and r["feature"] == "mean_rr_ms"]
        if not sub:
            continue
        lines.append(f"\n## {dom} {ev} (mean_rr_ms 四分位桶, acc 按 model 对 seed 取均值)\n")
        lines.append("| bucket | n | " + " | ".join(sorted({r["model"] for r in sub})) + " |")
        lines.append("|---|---|" + "---|" * len({r["model"] for r in sub}))
        buckets = sorted({r["bucket"] for r in sub})
        for b in buckets:
            cells = []
            for m in sorted({r["model"] for r in sub}):
                v = [r["acc"] for r in sub if r["model"] == m and r["bucket"] == b]
                cells.append(f"{np.mean(v):.4f}" if v else "-")
            ns = [r["n"] for r in sub if r["bucket"] == b]
            lines.append(f"| {b} | {int(np.mean(ns))} | " + " | ".join(cells) + " |")
    (OUTD / "e4d_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("summary written")
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    main()
