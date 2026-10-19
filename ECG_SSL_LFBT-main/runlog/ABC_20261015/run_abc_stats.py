# -*- coding: utf-8 -*-
"""run_abc_stats.py — ABC_20261015 统计阶段(方案 §10/§11, protocol_frozen §6, CPU-only)。

读各方逐记录预测, 产出(整文件重写, 幂等可反复跑):
  paired_stats.csv        — 配对 bootstrap 10k(多 seed 同步重采样, RNG 20261015)+DeLong(仅 cpsc)
                            p_adj: 核心族=Holm, 探索族=BH-FDR
  factorial_stats.csv     — §10.3 全部 7 式交互 + 3 条件边际(cpsc LP, 逐种子+bootstrap CI)
  positive_gain_ledger.csv— 正增益账本(E1-E4 分级+措辞+论文位置, 方案 §3.2/§12)
  table_lp_3domains.csv / table_ft10.csv / table_noise27.csv
  ABC最终报告.md          — --report 时生成(缺数据段标"待补")

预测目录映射(protocol_frozen §1): m0/m1=c1/c2(cpsc=W5修正分区, ptbxl/chapman=confirm
重放->predictions_anchor/), m2=dv(W6 s0/W7 s24), m5=noiseiso(W8), m3/m4/m6=JOINT,
m7/m8/m9/n1n2(s2,s4)/m4b=本目录; n1/n2 s0=JOINT。缺哪方跳哪对, 全齐出全表。

用法:
  python runlog/ABC_20261015/run_abc_stats.py --families p0    # P0 快照
  python runlog/ABC_20261015/run_abc_stats.py --families all   # 全量
  python runlog/ABC_20261015/run_abc_stats.py --families all --report
  python runlog/ABC_20261015/run_abc_stats.py --smoke          # m4:m3/cpsc/500boot 验机制
"""
import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUTD = ROOT / "runlog/ABC_20261015"
JOUT = ROOT / "runlog/JOINT_20261013"
SEEDS = [0, 2, 4]
DOMAINS = [("ptbxl", 5), ("cpsc", 9), ("chapman", 4)]
N_BOOT = 10000
RNG_SEED = 20261015  # 方案 §11 固定值

_spec = importlib.util.spec_from_file_location(
    "paired_stats_w4", ROOT / "runlog/W4/stats/paired_stats.py")
W4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W4)  # 复用: load_pred_dir/macro_auroc/delong_paired_class


def log(msg):
    print(f"[stats {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def _avg_precision(y_bin, score):
    order = np.argsort(-score, kind="mergesort")
    y = y_bin[order].astype(np.float64)
    if y.sum() == 0:
        return float("nan")
    tp = np.cumsum(y)
    prec = tp / np.maximum(np.arange(1, len(y) + 1), 1)
    rec = tp / y.sum()
    rec_prev = np.concatenate(([0.0], rec[:-1]))
    return float(np.sum((rec - rec_prev) * prec))


def macro_auprc(y_true, prob):
    nc = prob.shape[1]
    one_hot = np.eye(nc)[y_true]
    per = np.array([_avg_precision(one_hot[:, c], prob[:, c]) for c in range(nc)])
    return float(np.nanmean(per)), per


def _m_auprc(y, p):
    return macro_auprc(y, p)[0]


def _m_auroc(y, p):
    return W4.macro_auroc(y, p)[0]


# ---------------------------------------------------------------- 预测解析
def confirm_replay(model, ds, seed, nc):
    base = "c1" if model == "m0" else "c2"
    src = ROOT / f"results/confirm/{base}_{ds}_seed{seed}"
    dst = OUTD / "predictions_anchor" / f"{model}_{ds}_seed{seed}"
    if not (dst / "y_prob.npy").exists():
        import torch
        from models.linear import LinearClassifier
        X = np.load(src / "X_test.npy")
        y = np.load(src / "y_test.npy").astype(int)
        head = LinearClassifier(feat_dim=512, num_classes=nc)
        head.load_state_dict(torch.load(src / "classifier_best_ckpt.pth",
                                        map_location="cpu", weights_only=True))
        head.eval()
        with torch.no_grad():
            prob = torch.softmax(head(torch.from_numpy(X)), dim=1).numpy()
        au, _ = W4.macro_auroc(y, prob)
        ref = json.loads((src / "metrics.json").read_text(encoding="utf-8"))["auroc"]
        assert abs(au - ref) < 5e-3, f"{dst.name}: 回放 auroc {au:.4f} vs 冻结 {ref:.4f}"
        dst.mkdir(parents=True, exist_ok=True)
        np.save(dst / "y_true.npy", y)
        np.save(dst / "y_prob.npy", prob)
        np.save(dst / "y_pred.npy", prob.argmax(axis=1).astype(np.int64))
        log(f"  锚点重放 {model}/{ds}/s{seed}: auroc {au:.4f} ≈ 冻结 {ref:.4f} OK")
    return dst


def pred_dir(model, ds, seed, nc):
    if model in ("m0", "m1"):
        if ds == "cpsc":
            base = "c1" if model == "m0" else "c2"
            return ROOT / f"runlog/W5/predictions/{base}_cpsc_seed{seed}"
        return confirm_replay(model, ds, seed, nc)
    if model == "m2":
        w = "W6" if seed == 0 else "W7"
        return ROOT / f"runlog/{w}/predictions/dv_{ds}_lp_seed{seed}"
    if model == "m5":
        return ROOT / f"runlog/W8/predictions/noiseiso_{ds}_lp_seed{seed}"
    if model in ("m3", "m4", "m6"):
        return JOUT / "predictions" / f"{model}_{ds}_lp_seed{seed}"
    if model in ("n1", "n2") and seed == 0:
        return JOUT / "predictions" / f"{model}_{ds}_lp_seed{seed}"
    return OUTD / "predictions" / f"{model}_{ds}_lp_seed{seed}"


def ft_pred_dir(model, ds, seed):
    if model in ("m0", "m1"):
        base = "c1" if model == "m0" else "c2"
        return ROOT / f"runlog/W6/predictions/{base}_{ds}_ft10_seed{seed}"
    if model in ("m3", "m4", "m6") and ds == "ptbxl":
        return JOUT / "predictions" / f"{model}_ptbxl_ft10_seed{seed}"
    return OUTD / "predictions" / f"{model}_{ds}_ft10_seed{seed}"


def noise_pred_dir(model, ds, seed, cond):
    if model in ("m3", "m4", "m6"):
        tag = "m3-lasteval" if model == "m3" else model
        return JOUT / "predictions" / f"{tag}_{ds}_lpnoise_seed{seed}" / cond
    return OUTD / "predictions" / f"{model}_{ds}_lpnoise_seed{seed}" / cond


def load_preds(dir_fn, model, ds, seeds, nc):
    out = {}
    for s in seeds:
        d = dir_fn(model, ds, s, nc) if dir_fn is pred_dir else dir_fn(model, ds, s)
        if not (d / "y_prob.npy").exists():
            return None
        y, _, p = W4.load_pred_dir(d)
        out[s] = (y, p.astype(np.float64))
    return out


# ---------------------------------------------------------------- 检验
def paired_bootstrap(preds_a, preds_b, metric, n_boot, seed):
    seeds = sorted(preds_a)
    n = len(preds_a[seeds[0]][0])
    for s in seeds:
        assert np.array_equal(preds_a[s][0], preds_b[s][0]), f"y_true 不对齐 seed{s}"
        assert len(preds_a[s][0]) == n
    d0 = np.array([metric(preds_a[s][0], preds_a[s][1]) - metric(preds_b[s][0], preds_b[s][1])
                   for s in seeds])
    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        per = [metric(preds_a[s][0][idx], preds_a[s][1][idx])
               - metric(preds_b[s][0][idx], preds_b[s][1][idx]) for s in seeds]
        boots[b] = np.mean(per)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    return dict(delta_mean=float(d0.mean()),
                per_seed={s: float(v) for s, v in zip(seeds, d0)},
                pos_seeds=int(sum(v > 0 for v in d0)),
                ci_lo=float(lo), ci_hi=float(hi),
                p_emp=float(min(max(p, 1.0 / n_boot), 1.0)))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    mx = 0.0
    for rank, i in enumerate(order):
        v = (m - rank) * pvals[i]
        mx = max(mx, v)
        adj[i] = min(mx, 1.0)
    return adj


def bh_fdr(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        v = min(prev, (m / (rank + 1)) * pvals[i])
        adj[i] = v
        prev = v
    return adj


# ---------------------------------------------------------------- 对族定义
CORE = [("m4", "m0"), ("m4", "m1"), ("m4", "m2"), ("m4", "m3"),
        ("m4", "m5"), ("m4", "m6"), ("m4", "m7")]           # 方案 §10.1(主终点族)
BSPLIT = [("m4", "m8"), ("m4", "m9"), ("m8", "m6"), ("m9", "m6")]  # §5.2 B 内部拆分
MECH = [("n1", "m3"), ("n2", "m3")]                          # §P1-1 机制
CORPUS = [("m4b", "m4")]                                     # §P1-2 跨语料
FT_PAIRS = [("m4", "m3"), ("m4", "m6"), ("m3", "m6"), ("m4", "m0"), ("m4", "m1")]
NOISE_BOOT = [("m4", "m3"), ("m7", "m6"), ("m8", "m6"), ("m9", "m6")]  # 逐记录噪声族
NOISE_ROW = [("m4", "m7"), ("m4", "m8"), ("m4", "m9"), ("m6", "m3")]  # 行级表(描述性)
NOISE_TYPES = ("bw", "ma", "em")
SNRS = ("0", "5", "10")


def run_lp_pairs(pairs_list, n_boot, skip=None):
    """三域 LP; 返回行列表。skip={(task,ds,pair,metric)...} 已有结果的对整体跳过(增量)。"""
    fam_core, fam_explore = [], []
    for ds, nc in DOMAINS:
        for a, b in pairs_list:
            _pk = ("lp", ds, f"{a}:{b}")
            if skip and (_pk + ("auprc",)) in skip and (_pk + ("auroc",)) in skip \
                    and (ds != "cpsc" or (_pk + ("delong_auroc_p_min",)) in skip):
                continue
            pa = load_preds(pred_dir, a, ds, SEEDS, nc)
            pb = load_preds(pred_dir, b, ds, SEEDS, nc)
            if pa is None or pb is None:
                log(f"  [skip] {ds} {a}:{b}: 预测不全")
                continue
            for metric_name, mf in (("auprc", _m_auprc), ("auroc", _m_auroc)):
                r = paired_bootstrap(pa, pb, mf, n_boot, RNG_SEED)
                row = dict(task="lp", dataset=ds, pair=f"{a}:{b}", metric=metric_name,
                           family="core" if (ds == "cpsc" and metric_name == "auprc"
                                             and (a, b) in CORE) else "explore",
                           **{k: r[k] for k in
                              ("delta_mean", "per_seed", "pos_seeds",
                               "ci_lo", "ci_hi", "p_emp")})
                (fam_core if row["family"] == "core" else fam_explore).append(row)
                log(f"  {ds} {a}:{b} [{metric_name}] Δ={r['delta_mean']*100:+.3f}pt "
                    f"CI[{r['ci_lo']*100:+.3f},{r['ci_hi']*100:+.3f}] "
                    f"p={r['p_emp']:.4f} seeds+{r['pos_seeds']}/3")
            if ds == "cpsc":
                for s in SEEDS:
                    ya, pa_s = pa[s]
                    _, pb_s = pb[s]
                    dl = [W4.delong_paired_class(ya, pa_s, pb_s, c)
                          for c in range(pa_s.shape[1])]
                    ps = [d[4] for d in dl if np.isfinite(d[4])]
                    if ps:
                        fam_explore.append(dict(
                            task="lp", dataset=ds, pair=f"{a}:{b}",
                            metric="delong_auroc_p_min", family="explore",
                            delta_mean=None, per_seed={str(s): None}, pos_seeds=None,
                            ci_lo=None, ci_hi=None,
                            p_emp=round(float(np.min(ps)), 4)))
    return fam_core, fam_explore


def run_ft_pairs(n_boot):
    rows = []
    for ds, nc in (("cpsc", 9), ("chapman", 4), ("ptbxl", 5)):
        for a, b in FT_PAIRS:
            pa, pb = {}, {}
            for s in SEEDS:
                da, db = ft_pred_dir(a, ds, s), ft_pred_dir(b, ds, s)
                if not (da / "y_prob.npy").exists() or not (db / "y_prob.npy").exists():
                    pa = None
                    break
                y, _, p = W4.load_pred_dir(da)
                pa[s] = (y, p.astype(np.float64))
                y2, _, p2 = W4.load_pred_dir(db)
                pb[s] = (y2, p2.astype(np.float64))
            if pa is None:
                log(f"  [skip] ft10 {ds} {a}:{b}: 预测不全")
                continue
            for metric_name, mf in (("auprc", _m_auprc), ("auroc", _m_auroc)):
                r = paired_bootstrap(pa, pb, mf, n_boot, RNG_SEED + 2)
                rows.append(dict(task="ft10", dataset=ds, pair=f"{a}:{b}",
                                 metric=metric_name, family="explore",
                                 **{k: r[k] for k in
                                    ("delta_mean", "per_seed", "pos_seeds",
                                     "ci_lo", "ci_hi", "p_emp")}))
                log(f"  ft10 {ds} {a}:{b} [{metric_name}] "
                    f"Δ={r['delta_mean']*100:+.3f}pt p={r['p_emp']:.4f}")
    return rows


def _load_noise_preds(d):
    """噪声预测目录只存 y_prob/y_true/noise_offsets(无 y_pred), 直接加载。
    W4.load_pred_dir 要求 y_pred.npy 会在噪声目录上 FileNotFoundError(10-17 22:00 崩因)。"""
    y = np.load(d / "y_true.npy").astype(int)
    p = np.load(d / "y_prob.npy").astype(np.float64)
    return y, p


def run_noise_pairs(n_boot):
    """逐记录噪声族: 每(ds, pair) 9 格均值 delta(AUROC 主) + bootstrap;
    行级族只进表不做 bootstrap。"""
    rows = []
    for ds, _ in DOMAINS:
        for a, b in NOISE_BOOT:
            cell_metrics = {}
            ok = True
            for nt in NOISE_TYPES:
                for snr in SNRS:
                    cond = f"{nt}@{snr}"
                    da, db = noise_pred_dir(a, ds, SEEDS[0], cond), \
                        noise_pred_dir(b, ds, SEEDS[0], cond)
                    if not (da / "y_prob.npy").exists() or not (db / "y_prob.npy").exists():
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                log(f"  [skip] noise {ds} {a}:{b}: 9 格预测不全")
                continue
            # 收集 (seed -> 每 cell y/prob)
            per_seed = {}
            for s in SEEDS:
                cells = []
                for nt in NOISE_TYPES:
                    for snr in SNRS:
                        cond = f"{nt}@{snr}"
                        ya, pa_ = _load_noise_preds(noise_pred_dir(a, ds, s, cond))
                        yb, pb_ = _load_noise_preds(noise_pred_dir(b, ds, s, cond))
                        assert np.array_equal(ya, yb)
                        cells.append((ya, pa_, pb_))
                per_seed[s] = cells
            n = len(per_seed[SEEDS[0]][0][0])
            d0 = np.array([np.mean([_m_auroc(y, pa_) - _m_auroc(y, pb_)
                                    for y, pa_, pb_ in per_seed[s]]) for s in SEEDS])
            rng = np.random.RandomState(RNG_SEED + 3)
            boots = np.empty(n_boot)
            for bi in range(n_boot):
                idx = rng.randint(0, n, n)
                per = []
                for s in SEEDS:
                    vals = [_m_auroc(y[idx], pa_[idx]) - _m_auroc(y[idx], pb_[idx])
                            for y, pa_, pb_ in per_seed[s]]
                    per.append(np.mean(vals))
                boots[bi] = np.mean(per)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
            rows.append(dict(task="lpnoise9mean", dataset=ds, pair=f"{a}:{b}",
                             metric="auroc_mean9cells", family="explore",
                             delta_mean=float(d0.mean()),
                             per_seed={str(s): float(v) for s, v in zip(SEEDS, d0)},
                             pos_seeds=int(sum(v > 0 for v in d0)),
                             ci_lo=float(lo), ci_hi=float(hi),
                             p_emp=float(min(max(p, 1.0 / n_boot), 1.0))))
            log(f"  noise {ds} {a}:{b} mean9 Δ={d0.mean()*100:+.3f}pt "
                f"CI[{lo*100:+.3f},{hi*100:+.3f}] p={rows[-1]['p_emp']:.4f}")
    return rows


# ---------------------------------------------------------------- 因子交互(§10.3)
FACT = {
    "I_AB|C=0": {"m3": 1, "m1": -1, "m2": -1, "m0": 1},
    "I_AB|C=1": {"m4": 1, "m6": -1, "m7": -1, "m5": 1},
    "I_AC|B=0": {"m6": 1, "m1": -1, "m5": -1, "m0": 1},
    "I_AC|B=1": {"m4": 1, "m3": -1, "m7": -1, "m2": 1},
    "I_BC|A=0": {"m7": 1, "m2": -1, "m5": -1, "m0": 1},
    "I_BC|A=1": {"m4": 1, "m3": -1, "m6": -1, "m1": 1},
    "I_ABC": {"m4": 1, "m3": -1, "m6": -1, "m7": -1, "m1": 1, "m2": 1, "m5": 1, "m0": -1},
    "A|B+C(m4-m7)": {"m4": 1, "m7": -1},
    "B|A+C(m4-m6)": {"m4": 1, "m6": -1},
    "C|A+B(m4-m3)": {"m4": 1, "m3": -1},
}


def run_factorial(n_boot):
    rows = []
    for metric_name, mf in (("auprc", _m_auprc), ("auroc", _m_auroc)):
        preds = {}
        for fname, combo in FACT.items():
            need = set(combo)
            for m in need - set(preds):
                preds[m] = load_preds(pred_dir, m, "cpsc", SEEDS, 9)
        for fname, combo in FACT.items():
            if any(preds.get(m) is None for m in combo):
                log(f"  [skip] factorial {fname}/{metric_name}: 预测不全")
                continue
            n = len(preds[combo.keys().__iter__().__next__()][SEEDS[0]][0])
            def combo_val(s, idx=None):
                v = 0.0
                for m, w in combo.items():
                    y, p = preds[m][s]
                    if idx is not None:
                        y, p = y[idx], p[idx]
                    v += w * mf(y, p)
                return v
            per_seed = {s: combo_val(s) for s in SEEDS}
            rng = np.random.RandomState(RNG_SEED + 1)
            boots = np.empty(n_boot)
            for b in range(n_boot):
                idx = rng.randint(0, n, n)
                boots[b] = np.mean([combo_val(s, idx) for s in SEEDS])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
            rows.append(dict(name=fname, metric=metric_name,
                             mean_pt=round(float(np.mean(list(per_seed.values()))) * 100, 3),
                             ci_lo_pt=round(lo * 100, 3), ci_hi_pt=round(hi * 100, 3),
                             p_emp=round(float(min(max(p, 1.0 / n_boot), 1.0)), 4),
                             per_seed_pt={str(s): round(v * 100, 3)
                                          for s, v in per_seed.items()}))
            log(f"  factorial {fname}/{metric_name}: {rows[-1]['mean_pt']:+.3f}pt "
                f"CI[{rows[-1]['ci_lo_pt']:+.3f},{rows[-1]['ci_hi_pt']:+.3f}] "
                f"p={rows[-1]['p_emp']}")
    return rows


# ---------------------------------------------------------------- 表与账本
def write_paired(rows_core, rows_explore, path):
    allr = rows_core + rows_explore
    if not allr:
        return
    pc = [r["p_emp"] for r in rows_core]
    ac = holm(np.array(pc)) if pc else []
    pe = [r["p_emp"] for r in rows_explore if r["p_emp"] is not None]
    ae = bh_fdr(np.array(pe)) if pe else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts", "task", "dataset", "pair", "metric", "family",
                    "delta_mean_pt", "ci_lo_pt", "ci_hi_pt", "p_raw", "p_adjusted",
                    "pos_seeds", "total_seeds", "per_seed_pt"])
        i_c = i_e = 0
        for r in rows_core + rows_explore:
            if r["family"] == "core":
                padj = round(float(ac[i_c]), 4) if r["p_emp"] is not None else None
                i_c += 1
            else:
                padj = round(float(ae[i_e]), 4) if r["p_emp"] is not None else None
                i_e += 1
            w.writerow([time.strftime("%Y-%m-%d %H:%M"), r["task"], r["dataset"],
                        r["pair"], r["metric"], r["family"],
                        round(r["delta_mean"] * 100, 3) if r["delta_mean"] is not None else None,
                        round(r["ci_lo"] * 100, 3) if r["ci_lo"] is not None else None,
                        round(r["ci_hi"] * 100, 3) if r["ci_hi"] is not None else None,
                        r["p_emp"], padj,
                        r["pos_seeds"], len(r["per_seed"]),
                        json.dumps({k: (round(v * 100, 3) if v is not None else None)
                                    for k, v in r["per_seed"].items()}, ensure_ascii=False)])


def build_ledger(rows_core, rows_explore):
    ledger = []
    for r in rows_core + rows_explore:
        if "delta_mean" in r:  # 新算行: 浮点小数
            if r["delta_mean"] is None or r["delta_mean"] <= 0:
                continue
            delta_pt = r["delta_mean"] * 100
            ci_lo_pt = r["ci_lo"] * 100 if r["ci_lo"] is not None else None
            ci_hi_pt = r["ci_hi"] * 100 if r["ci_hi"] is not None else None
            pos = r["pos_seeds"]
            tot = len(r["per_seed"])
            praw = r["p_emp"]
        else:  # CSV 复用行(10-19 增量): 键带 _pt 后缀, 值已是百分点
            try:
                delta_pt = float(r.get("delta_mean_pt") or 0)
            except (TypeError, ValueError):
                continue
            if delta_pt <= 0:
                continue
            ci_lo_pt = float(r["ci_lo_pt"]) if r.get("ci_lo_pt") else None
            ci_hi_pt = float(r["ci_hi_pt"]) if r.get("ci_hi_pt") else None
            pos = int(r["pos_seeds"]) if r.get("pos_seeds") else 0
            tot = int(r["total_seeds"]) if r.get("total_seeds") else 3
            praw = r.get("p_raw")
        scope = f"{r['task']}/{r['dataset']}/{r['metric']}"
        if ci_lo_pt is not None and ci_lo_pt > 0:
            tier, wording, loc = "E1", "显著提升(95% CI 不跨 0)", "主表候选"
        elif (pos or 0) >= 2:
            tier, wording, loc = "E2", "观察到稳定正向趋势(描述性增益, CI 跨 0)", "主表或补表"
        else:
            tier, wording, loc = "E3", f"局部正增益(限定范围: {scope})", "补表(限定措辞)"
        ledger.append(dict(comparison=r["pair"], task=r["task"], dataset=r["dataset"],
                           metric=r["metric"], scope=scope,
                           delta_pt=round(delta_pt, 3),
                           ci_low_pt=round(ci_lo_pt, 3) if ci_lo_pt is not None else None,
                           ci_high_pt=round(ci_hi_pt, 3) if ci_hi_pt is not None else None,
                           p_raw=praw, p_adjusted=None,
                           positive_seed_count=pos, total_seed_count=tot,
                           evidence_tier=tier, paper_location=loc, wording=wording))
    # p_adjusted 从 paired_stats 关联回填
    if (OUTD / "paired_stats.csv").exists():
        with open(OUTD / "paired_stats.csv", encoding="utf-8") as f:
            pads = {(x["task"], x["dataset"], x["pair"], x["metric"]): x["p_adjusted"]
                    for x in csv.DictReader(f)}
        for row in ledger:
            row["p_adjusted"] = pads.get((row["task"], row["dataset"],
                                          row["comparison"], row["metric"]))
    return ledger


def write_tables():
    # 表1: LP 三域
    lines = ["model,dataset,auroc_mean,auroc_sd,auprc_mean,auprc_sd,n_seeds"]
    for model in ("m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9",
                  "m4b", "n1", "n2"):
        for ds, nc in DOMAINS:
            preds = load_preds(pred_dir, model, ds, SEEDS, nc)
            if preds is None:
                continue
            au = [_m_auroc(*preds[s]) for s in SEEDS if s in preds]
            ap = [_m_auprc(*preds[s]) for s in SEEDS if s in preds]
            lines.append(f"{model},{ds},{np.mean(au):.4f},{np.std(au):.4f},"
                         f"{np.mean(ap):.4f},{np.std(ap):.4f},{len(au)}")
    (OUTD / "table_lp_3domains.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # 表2: FT10(读 ABC results.csv + 历史 ft10grid/JOINT 账)
    lines2 = ["model,dataset,seed,auroc,auprc,source"]
    for src, proto in ((OUTD / "results.csv", "abc"),):
        if src.exists():
            with open(src, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r["task"] == "ft10":
                        lines2.append(f"{r['model']},{r['dataset']},{r['seed']},"
                                      f"{r['auroc']},{r['auprc']},ABC")
    for hist in (ROOT / "runlog/W6/ft10grid_results_L1.csv", JOUT / "results.csv"):
        if hist.exists():
            with open(hist, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("eval") == "ft10":
                        model = {"b0": "b0", "c1": "m0", "c2": "m1"}.get(
                            r["ckpt"], r["ckpt"])
                        lines2.append(f"{model},{r['downstream']},{r['seed']},"
                                      f"{r['auroc']},{r['auprc']},{hist.parent.name}")
    (OUTD / "table_ft10.csv").write_text("\n".join(lines2) + "\n", encoding="utf-8")
    # 表3: 噪声 27 格行级(ABC + JOINT 账, 每行=model×seed×ds×cond 的 auroc)
    lines3 = ["model,seed,dataset,noise_type,snr,auroc,auprc,source"]
    for src in ((OUTD / "results.csv", "ABC"), (JOUT / "results.csv", "JOINT")):
        if src[0].exists():
            with open(src[0], encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("eval") == "lpnoise":
                        model = "m3" if r["ckpt"] == "m3-lasteval" else (
                            "m1" if r["ckpt"] == "m1-lasteval" else r["ckpt"])
                        nt = r.get("noise_type") or (r.get("cond", "").split("@")[0]
                                                     if r.get("cond") else "")
                        snr = r.get("snr") or (r.get("cond", "").split("@")[1]
                                               if "@" in r.get("cond", "") else "")
                        lines3.append(f"{model},{r['seed']},{r['downstream']},"
                                      f"{nt},{snr},"
                                      f"{r['auroc']},{r['auprc']},{src[1]}")
    (OUTD / "table_noise27.csv").write_text("\n".join(lines3) + "\n", encoding="utf-8")
    log("三表写出(lp_3domains/ft10/noise27)")


# ---------------------------------------------------------------- 报告
def gen_report(ledger):
    def _csv_head(p, n=6):
        if not Path(p).exists():
            return "（待补：未生成）"
        with open(p, encoding="utf-8") as f:
            return "\n".join(",".join(row) for row in
                             [r for _, r in zip(range(n), csv.reader(f))])
    tiers = {}
    for row in ledger:
        tiers[row["evidence_tier"]] = tiers.get(row["evidence_tier"], 0) + 1
    res_rows = 0
    if (OUTD / "results.csv").exists():
        with open(OUTD / "results.csv", encoding="utf-8") as f:
            res_rows = sum(1 for _ in csv.DictReader(f))
    md = f"""# ABC_20261015 联合框架补充实验·最终报告(自动生成 {time.strftime('%Y-%m-%d %H:%M')})

> 预注册方案: ABC联合框架补充实验方案.md(ABC-SUPP-20261015); 冻结协议: protocol_frozen.md。
> 本报告由 run_abc_stats.py --report 自动装配, 结论措辞按方案 §3.2/§12 决策表;
> 证据分级: E1 确认性正增益 / E2 稳定描述性 / E3 局部 / E4 中性或负(总表保留)。

## 一、完成度(方案 §15 验收)

- results.csv 已落 {res_rows} 行(ABC 新增); 正增益账本 {len(ledger)} 条
  (E1×{tiers.get('E1', 0)} / E2×{tiers.get('E2', 0)} / E3×{tiers.get('E3', 0)})。
- 明细对账见 events.md; 失败与重跑记录见同文件。

## 二、主终点(CPSC LP macro-AUPRC, 核心比较族, Holm 校正)

```
{_csv_head(OUTD / 'paired_stats.csv', 1)}
```
(完整表见 paired_stats.csv; family=core 行为核心族)

## 三、A/B/C 因子交互(方案 §10.3 全式 + 条件边际)

```
{_csv_head(OUTD / 'factorial_stats.csv', 12)}
```

措辞规则: 交互为正且 CI 不跨 0 → "协同增益得到统计支持"; 为正但跨 0 → "正向三阶交互趋势";
否则 → "联合设计/鲁棒性扩展"(不写超加性协同)。

## 四、B 内部拆分(M4−M8/M4−M9/M8−M6/M9−M6)

见 paired_stats.csv 中 pair=m4:m8 / m4:m9 / m8:m6 / m9:m6。
若 RR 平均贡献≈0 且 Beat 保持正向: B 表述为"以 R 峰对齐 Beat 形态为主要有效成分、
RR 为被验证但贡献有限的辅助目标"(不删除 RR 结果)。

## 五、噪声鲁棒(27 格 last_half, 确定性同噪声配对)

```
{_csv_head(OUTD / 'table_noise27.csv', 8)}
```
逐记录 bootstrap 族(pair=m4:m3 / m7:m6 / m8:m6 / m9:m6 的 auroc_mean9cells 行)
见 paired_stats.csv; 其余噪声对为行级描述性(table_noise27.csv)。

## 六、机制(N1/N2 三种子)

见 paired_stats.csv pair=n1:m3 / n2:m3。
N1−M3 若 3/3 为负且 CI 不跨 0 → "R 峰对齐是有效 Beat 约束的必要条件";
N2−M3≈0 → "当前权重和协议下未观察到 RR 标签信息的独立贡献"(不写"RR 永远无效")。

## 七、第二语料(P1-2, M4b=PTB-XL 预训练)

见 paired_stats.csv pair=m4b:m4(cpsc LP/ft10)与 table_lp_3domains.csv m4b 行。
定位: 支持"框架不只依赖单一预训练语料", 不用于重新搜索损失权重。

## 八、多标签敏感性(P1-3)

```
{_csv_head(OUTD / 'multilib_results.csv', 8)}
```
口径: W6 冻结(one-hot 主类真值, 限制如实入账); 主分析仍为单标签口径。

## 九、效率与成本(P0-4)

```
{_csv_head(OUTD / 'efficiency.csv', 11)}
```

## 十、论文决策表对照(方案 §12)

- M4 对基线/单组件显著为正(E1) → 主表+摘要 "A+B+C 获得稳定联合增益"
- 正但 CI 跨 0(E2) → 主表或补表 "描述性正增益", 不写"显著"
- I_ABC>0 跨 0 → 交互表 "正向三阶交互趋势"; 不跨 0 → 主文 "协同增益得到统计支持"
- I_ABC≤0 但总体/鲁棒为正 → "有效联合/鲁棒框架, 无超加性协同证据"
- 负向比较 → 总表保留, 明确代价与边界

## 附:产物索引

results.csv / paired_stats.csv / factorial_stats.csv / positive_gain_ledger.csv /
efficiency.csv / multilib_results.csv / checkpoints_manifest.json /
table_lp_3domains.csv / table_ft10.csv / table_noise27.csv / predictions/ / logs/ / events.md
"""
    (OUTD / "ABC最终报告.md").write_text(md, encoding="utf-8")
    log("ABC最终报告.md 生成")


# ---------------------------------------------------------------- 主流程
def load_csv_rows(path):
    """paired_stats.csv -> 行 dict(保留 CSV 原值, 复用时不重推导只重算 p_adjusted)。"""
    rows_c, rows_e = [], []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tgt = rows_c if r["family"] == "core" else rows_e
            tgt.append(r)
    return rows_c, rows_e


def incremental_stats(args):
    """10-18 增量模式(events 记录): 终局统计复用既有 paired_stats.csv 行(输入/RNG 冻结,
    p0 族结果逐位不变), 只补算缺失对(m4b 跨语料/n1n2 机制), 重算全族校正与账本三表。
    factoriaL_stats.csv 原样保留(8 模型预测未变, 结果逐位同)。"""
    csv_path = OUTD / "paired_stats.csv"
    rows_core, rows_explore = load_csv_rows(csv_path)
    have = {(r["task"], r["dataset"], r["pair"], r["metric"])
            for r in rows_core + rows_explore}
    log(f"增量模式: 复用既有 {len(rows_core)+len(rows_explore)} 行, 补算缺失对")
    pairs_list = list(CORE) + list(BSPLIT) + list(MECH) + list(CORPUS)
    new_c, new_e = run_lp_pairs(pairs_list, args.nboot, skip=have)  # 已有对跳过/缺预测自动跳过
    kept_c = [r for r in new_c
              if (r["task"], r["dataset"], r["pair"], r["metric"]) not in have]
    kept_e = [r for r in new_e
              if (r["task"], r["dataset"], r["pair"], r["metric"]) not in have
              and r["metric"] != "delong_auroc_p_min"]
    log(f"新增 {len(kept_c)+len(kept_e)} 行(其余已存在/仍缺预测跳过)")
    # FT/噪声族输入与 stats0 完全一致 -> 全部复用, 不重算
    write_paired_csv(rows_core + kept_c, rows_explore + kept_e, csv_path)
    ledger = build_ledger(rows_core + kept_c, rows_explore + kept_e)
    with open(OUTD / "positive_gain_ledger.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["comparison", "task", "dataset", "metric",
                                          "scope", "delta_pt", "ci_low_pt", "ci_high_pt",
                                          "p_raw", "p_adjusted", "positive_seed_count",
                                          "total_seed_count", "evidence_tier",
                                          "paper_location", "wording"])
        w.writeheader()
        w.writerows(ledger)
    write_tables()
    log(f"增量统计完成: paired 总行={len(rows_core)+len(kept_c)+len(rows_explore)+len(kept_e)}, "
        f"ledger={len(ledger)} 条(factorial 沿用 stats0 版本)")


def write_paired_csv(rows_core, rows_explore, path):
    """写 CSV: 兼容两类行——dict(新算, 浮点) 与 DictReader 行(复用, 字符串原样)。"""
    def _fmt(r, k, scale=True):
        # 双键兼容: 新算行=delta_mean(小数), CSV 复用行=delta_mean_pt(百分点, 10-19 修复)
        v = r.get(k)
        if (v is None or v == "") and not k.endswith("_pt"):
            v = r.get(k + "_pt")
        if v is None or v == "":
            return None
        if isinstance(v, str):
            return v  # 复用行: CSV 原值直通
        return round(v * 100, 3) if scale else v
    def _praw(r):
        v = r.get("p_emp")
        if v in (None, ""):
            v = r.get("p_raw")  # CSV 复用行键名
        return v
    pc = [float(_praw(r)) for r in rows_core if _praw(r) not in (None, "")]
    ac = holm(np.array(pc)) if pc else []
    pe = [float(_praw(r)) for r in rows_explore if _praw(r) not in (None, "")]
    ae = bh_fdr(np.array(pe)) if pe else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts", "task", "dataset", "pair", "metric", "family",
                    "delta_mean_pt", "ci_lo_pt", "ci_hi_pt", "p_raw", "p_adjusted",
                    "pos_seeds", "total_seeds", "per_seed_pt"])
        i_c = i_e = 0
        for r in rows_core + rows_explore:
            praw = _praw(r)
            if praw in (None, ""):
                padj = None
            elif r["family"] == "core":
                padj = round(float(ac[i_c]), 4)
                i_c += 1
            else:
                padj = round(float(ae[i_e]), 4)
                i_e += 1
            w.writerow([time.strftime("%Y-%m-%d %H:%M"), r["task"], r["dataset"],
                        r["pair"], r["metric"], r["family"],
                        _fmt(r, "delta_mean"), _fmt(r, "ci_lo"), _fmt(r, "ci_hi"),
                        praw, padj,
                        r.get("pos_seeds"), r.get("total_seeds") or len(
                            json.loads(r["per_seed_pt"]) if isinstance(r.get("per_seed_pt"), str)
                            else r.get("per_seed", {})),
                        r.get("per_seed_pt") if isinstance(r.get("per_seed_pt"), str)
                        else json.dumps({k: round(v * 100, 3) for k, v in
                                         r.get("per_seed", {}).items()})])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="all", choices=["p0", "all"])
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="10-17 优化: 不重算统计, 只从既有 CSV 重新装配报告")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nboot", type=int, default=N_BOOT)
    ap.add_argument("--force-full", action="store_true",
                    help="10-18: 忽略既有 CSV 强制全量重算(默认增量)")
    args = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        ledger = []
        p = OUTD / "positive_gain_ledger.csv"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                ledger = list(csv.DictReader(f))
        for need in ("paired_stats.csv", "factorial_stats.csv",
                     "positive_gain_ledger.csv"):
            assert (OUTD / need).exists(), f"--report-only 缺 {need}(先跑全量统计)"
        gen_report(ledger)
        return
    if args.smoke:
        pa = load_preds(pred_dir, "m4", "cpsc", SEEDS, 9)
        pb = load_preds(pred_dir, "m3", "cpsc", SEEDS, 9)
        r = paired_bootstrap(pa, pb, _m_auprc, 500, RNG_SEED)
        log(f"smoke m4:m3/cpsc Δ={r['delta_mean']*100:+.3f}pt "
            f"CI[{r['ci_lo']*100:+.3f},{r['ci_hi']*100:+.3f}]")
        return
    pairs_list = list(CORE) + list(BSPLIT) + list(MECH)
    if args.families == "all":
        pairs_list += list(CORPUS)
    # 10-18 增量模式: 全量族 + 已有 stats0 产物 → 只补算缺失对(省 ~17h 重算)
    if args.families == "all" and not args.force_full and \
            (OUTD / "paired_stats.csv").exists() and \
            (OUTD / "factorial_stats.csv").exists():
        incremental_stats(args)
        if args.report:
            ledger = list(csv.DictReader(
                open(OUTD / "positive_gain_ledger.csv", encoding="utf-8")))
            gen_report(ledger)
        return
    log(f"统计启动: families={args.families} nboot={args.nboot} "
        f"lp对={len(pairs_list)}")
    rows_core, rows_explore = run_lp_pairs(pairs_list, args.nboot)
    ft_rows = run_ft_pairs(args.nboot)
    noise_rows = run_noise_pairs(args.nboot) if args.families == "all" else \
        run_noise_pairs(args.nboot)  # P0 快照也含 m4:m3 等 P0 对
    rows_explore += ft_rows + noise_rows
    write_paired(rows_core, rows_explore, OUTD / "paired_stats.csv")
    fact = run_factorial(args.nboot)
    with open(OUTD / "factorial_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts", "name", "metric", "mean_pt", "ci_lo_pt", "ci_hi_pt",
                    "p_emp", "per_seed_pt"])
        for r in fact:
            w.writerow([time.strftime("%Y-%m-%d %H:%M"), r["name"], r["metric"],
                        r["mean_pt"], r["ci_lo_pt"], r["ci_hi_pt"], r["p_emp"],
                        json.dumps(r["per_seed_pt"], ensure_ascii=False)])
    ledger = build_ledger(rows_core, rows_explore)
    with open(OUTD / "positive_gain_ledger.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["comparison", "task", "dataset", "metric",
                                          "scope", "delta_pt", "ci_low_pt", "ci_high_pt",
                                          "p_raw", "p_adjusted", "positive_seed_count",
                                          "total_seed_count", "evidence_tier",
                                          "paper_location", "wording"])
        w.writeheader()
        w.writerows(ledger)
    write_tables()
    log(f"统计完成: paired={len(rows_core)+len(rows_explore)} 行, "
        f"factorial={len(fact)} 行, ledger={len(ledger)} 条")
    if args.report:
        gen_report(ledger)


if __name__ == "__main__":
    main()
