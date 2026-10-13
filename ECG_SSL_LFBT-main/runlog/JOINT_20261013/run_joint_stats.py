# -*- coding: utf-8 -*-
"""run_joint_stats.py — JOINT_20261013 统计阶段(protocol §6/§10, CPU-only)。

读各方逐记录预测(LP 产物 y_true/y_pred/y_prob), 产出:
  paired_stats.csv      — 配对 bootstrap(主=macro-AUPRC, 次=macro-AUROC)+DeLong(AUROC 辅助)
  interaction_stats.csv — I=(M3−M1)−(M2−M0) 逐种子值 + 同步重采样 bootstrap CI
  table1_lp_3domains.csv — §10-1: M0/M1/M2/M3 × 三域 LP mean±sd
  table2_ablation_cpsc.csv — §10-2: M0/A1/A2/M2/M1/M3 CPSC 消融逐种子+差值
  table3_noise_paired.csv — §10-3: M3/M4 clean+27 格配对差(读 results.csv)

锚点预测目录映射: m0/m1=c1/c2(cpsc=W5 修正分区版, ptbxl/chapman=results/confirm
需重放——X_test.npy+classifier 前向, 存 predictions_anchor/), m2=dv(W6 s0/W7 s24),
m5=noiseiso(W8)。新批次 m3/m4/a1/a2/m6 = 本目录 predictions/。
缺哪方就跳哪对(不报错), 全齐后出全表——幂等可反复跑。

用法:
  python runlog/JOINT_20261013/run_joint_stats.py --smoke   # 只跑 m1:m0/cpsc 验机制
  python runlog/JOINT_20261013/run_joint_stats.py            # 全量(收官后)
"""
import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUTD = ROOT / "runlog" / "JOINT_20261013"
SEEDS = [0, 2, 4]
DOMAINS = [("ptbxl", 5), ("cpsc", 9), ("chapman", 4)]
N_BOOT = 10000
RNG_SEED = 20261013  # 本批次独立种子(方法学注记), 逐对固定

_spec = importlib.util.spec_from_file_location(
    "paired_stats_w4", ROOT / "runlog" / "W4" / "stats" / "paired_stats.py")
W4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W4)  # 复用: load_pred_dir/macro_auroc/delong_paired_class


def log(msg):
    print(f"[stats {__import__('time').strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 指标
def _avg_precision(y_bin, score):
    """AP = Σ (R_i − R_{i−1})·P_i, 降序 mergesort 稳定(与 sklearn AP 同口径步长和)。"""
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


# ---------------------------------------------------------------- 预测解析
def confirm_replay(model, ds, seed, nc):
    """results/confirm 无 y_prob → 用 X_test+classifier CPU 前向重放(W4 predreplay 先例)。

    回放目录 predictions_anchor/{model}_{ds}_seed{S}; 与 metrics.json 对拍(|Δauroc|<5e-3
    否则 raise)。幂等: 已存在直接读。
    """
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
        import json
        ref = json.loads((src / "metrics.json").read_text(encoding="utf-8"))["auroc"]
        assert abs(au - ref) < 5e-3, f"{dst.name}: 回放 auroc {au:.4f} vs 冻结 {ref:.4f} 偏差过大"
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
    return OUTD / "predictions" / f"{model}_{ds}_lp_seed{seed}"


def load_preds(model, ds, seeds, nc):
    """{seed: (y_true, prob)}; 任一种子缺失返回 None(整对跳过)。"""
    out = {}
    for s in seeds:
        d = pred_dir(model, ds, s, nc)
        if not (d / "y_prob.npy").exists():
            log(f"  [skip] {model}/{ds}/s{s}: 预测缺({d})")
            return None
        y, _, p = W4.load_pred_dir(d)
        out[s] = (y, p.astype(np.float64))
    return out


# ---------------------------------------------------------------- 检验
def paired_bootstrap(preds_a, preds_b, metric, n_boot=N_BOOT, seed=RNG_SEED):
    """同步重采样: 记录索引逐 boot 抽一次, 各种子共用; delta=mean_s(delta_s)。"""
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
    return dict(delta_mean=float(d0.mean()), per_seed={s: float(v) for s, v in zip(seeds, d0)},
                ci_lo=float(lo), ci_hi=float(hi), p_emp=float(min(max(p, 1.0 / n_boot), 1.0)))


def _m_auprc(y, p):
    return macro_auprc(y, p)[0]


def _m_auroc(y, p):
    return W4.macro_auroc(y, p)[0]


# ---------------------------------------------------------------- 主流程
PAIRS_PRIMARY = [("m3", "m1"), ("m3", "m2"), ("m3", "m0"), ("m4", "m3"),
                 ("a1", "m0"), ("a2", "m0"), ("m2", "m0"), ("m3", "a1"),
                 ("m3", "a2"), ("m6", "m1"), ("m6", "m5"), ("m5", "m0")]


def run_pairs(datasets, pairs, n_boot):
    out_rows = []
    for ds, nc in datasets:
        for a, b in pairs:
            pa, pb = load_preds(a, ds, SEEDS, nc), load_preds(b, ds, SEEDS, nc)
            if pa is None or pb is None:
                continue
            for metric_name, mf in (("auprc", _m_auprc), ("auroc", _m_auroc)):
                r = paired_bootstrap(pa, pb, mf, n_boot=n_boot)
                row = dict(dataset=ds, pair=f"{a}:{b}", metric=metric_name,
                           delta_mean_pt=round(r["delta_mean"] * 100, 3),
                           ci_lo_pt=round(r["ci_lo"] * 100, 3),
                           ci_hi_pt=round(r["ci_hi"] * 100, 3),
                           p_emp=round(r["p_emp"], 4),
                           per_seed_pt={str(k): round(v * 100, 3)
                                        for k, v in r["per_seed"].items()})
                out_rows.append(row)
                log(f"  {ds} {a}:{b} [{metric_name}] Δ={row['delta_mean_pt']:+.3f}pt "
                    f"CI[{row['ci_lo_pt']:+.3f},{row['ci_hi_pt']:+.3f}] p={row['p_emp']}")
            # DeLong(AUROC 辅助, 仅 cpsc)
            if ds == "cpsc":
                for s in SEEDS:
                    ya, pa_s = pa[s]
                    _, pb_s = pb[s]
                    nc_ = pa_s.shape[1]
                    dl = [W4.delong_paired_class(ya, pa_s, pb_s, c) for c in range(nc_)]
                    ps = [d[4] for d in dl if np.isfinite(d[4])]
                    if ps:
                        out_rows.append(dict(dataset=ds, pair=f"{a}:{b}",
                                             metric="delong_auroc_p_min",
                                             delta_mean_pt=None, ci_lo_pt=None,
                                             ci_hi_pt=None,
                                             p_emp=round(float(np.min(ps)), 4),
                                             per_seed_pt={str(s): None}))
    return out_rows


def write_csv(rows, path):
    import json
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "dataset", "pair", "metric", "delta_mean_pt",
                        "ci_lo_pt", "ci_hi_pt", "p_emp", "per_seed_pt"])
        import time
        for r in rows:
            w.writerow([time.strftime("%Y-%m-%d %H:%M"), r["dataset"], r["pair"],
                        r["metric"], r["delta_mean_pt"], r["ci_lo_pt"], r["ci_hi_pt"],
                        r["p_emp"], json.dumps(r["per_seed_pt"], ensure_ascii=False)])


def run_interaction():
    """I=(M3−M1)−(M2−M0), cpsc, AUPRC 主+AUROC 次, 逐种子+bootstrap CI。"""
    import json
    import time
    rows = []
    path = OUTD / "interaction_stats.csv"
    for ds, nc in DOMAINS:
        if ds != "cpsc":
            continue
        for mname, mf in (("auprc", _m_auprc), ("auroc", _m_auroc)):
            preds = {m: load_preds(m, ds, SEEDS, nc) for m in ("m3", "m1", "m2", "m0")}
            if any(v is None for v in preds.values()):
                log(f"  [skip] interaction {ds}/{mname}: 预测不全")
                continue
            n = len(preds["m0"][SEEDS[0]][0])
            per_seed = {s: (mf(preds["m3"][s][0], preds["m3"][s][1])
                            - mf(preds["m1"][s][0], preds["m1"][s][1]))
                        - (mf(preds["m2"][s][0], preds["m2"][s][1])
                           - mf(preds["m0"][s][0], preds["m0"][s][1])) for s in SEEDS}
            rng = np.random.RandomState(RNG_SEED + 1)
            boots = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = rng.randint(0, n, n)
                per = [((mf(preds["m3"][s][0][idx], preds["m3"][s][1][idx])
                         - mf(preds["m1"][s][0][idx], preds["m1"][s][1][idx]))
                        - (mf(preds["m2"][s][0][idx], preds["m2"][s][1][idx])
                           - mf(preds["m0"][s][0][idx], preds["m0"][s][1][idx])))
                       for s in SEEDS]
                boots[b] = np.mean(per)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
            row = [time.strftime("%Y-%m-%d %H:%M"), ds, mname,
                   round(float(np.mean(list(per_seed.values()))) * 100, 3),
                   round(float(lo) * 100, 3), round(float(hi) * 100, 3),
                   round(float(min(max(p, 1.0 / N_BOOT), 1.0)), 4),
                   json.dumps({str(s): round(v * 100, 3) for s, v in per_seed.items()})]
            need_header = not path.exists()
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if need_header:
                    w.writerow(["ts", "dataset", "metric", "I_mean_pt", "ci_lo_pt",
                                "ci_hi_pt", "p_emp", "I_per_seed_pt"])
                w.writerow(row)
            log(f"  interaction {ds}/{mname}: I={row[3]:+.3f}pt CI[{row[4]:+.3f},"
                f"{row[5]:+.3f}] per-seed {row[7]}")


def run_tables():
    """§10 三张表(从预测与 results.csv 汇总)。"""
    # 表1: M0/M1/M2/M3 × 三域 LP
    lines = ["model,dataset,auroc_mean,auroc_sd,auprc_mean,auprc_sd,n_seeds"]
    for model in ("m0", "m1", "m2", "m3", "m4", "a1", "a2", "m6"):
        for ds, nc in DOMAINS:
            preds = load_preds(model, ds, SEEDS, nc)
            if preds is None:
                continue
            au = [W4.macro_auroc(*preds[s]) for s in SEEDS if s in preds]
            ap = [macro_auprc(*preds[s])[0] for s in SEEDS if s in preds]
            lines.append(f"{model},{ds},{np.mean(au):.4f},{np.std(au):.4f},"
                         f"{np.mean(ap):.4f},{np.std(ap):.4f},{len(au)}")
    (OUTD / "table1_lp_3domains.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"table1 写出({len(lines) - 1} 行)")
    # 表2: CPSC 消融逐种子(AUPRC 主)
    lines2 = ["model,seed,auprc,delta_vs_m0_pt"]
    m0 = load_preds("m0", "cpsc", SEEDS, 9)
    if m0 is not None:
        base0 = {s: macro_auprc(*m0[s])[0] for s in SEEDS}
        for model in ("m0", "a1", "a2", "m2", "m1", "m3"):
            preds = load_preds(model, "cpsc", SEEDS, 9)
            if preds is None:
                continue
            for s in SEEDS:
                v = macro_auprc(*preds[s])[0]
                lines2.append(f"{model},{s},{v:.4f},"
                              f"{(v - base0[s]) * 100:+.3f}")
        (OUTD / "table2_ablation_cpsc.csv").write_text("\n".join(lines2) + "\n",
                                                       encoding="utf-8")
        log("table2 写出")
    # 表3: M3/M4 clean+27 格配对差(读 results.csv)
    res = OUTD / "results.csv"
    if res.exists():
        import collections
        cells = collections.defaultdict(dict)
        with open(res, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["eval"] == "lpnoise" and r["ckpt"] in ("m4", "m3-lasteval"):
                    cells[(r["seed"], r["downstream"], r["cond"])][r["ckpt"]] = \
                        float(r["auroc"])
        lines3 = ["seed,dataset,cond,m4_auroc,m3ref_auroc,delta_pt"]
        per_seed_mean = collections.defaultdict(list)
        for (s, ds, cond), v in sorted(cells.items()):
            if "m4" in v and "m3-lasteval" in v:
                d = (v["m4"] - v["m3-lasteval"]) * 100
                lines3.append(f"{s},{ds},{cond},{v['m4']:.4f},{v['m3-lasteval']:.4f},{d:+.3f}")
                per_seed_mean[s].append(d)
        for s, ds in per_seed_mean.items():
            lines3.append(f"{s},MEAN27,all27,,,"
                          f"{np.mean(ds):+.3f}")
        (OUTD / "table3_noise_paired.csv").write_text("\n".join(lines3) + "\n",
                                                      encoding="utf-8")
        log(f"table3 写出({len(per_seed_mean)} 个种子有 27 格数据)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="只跑 m1:m0/cpsc/1000boot")
    args = ap.parse_args()
    if args.smoke:
        rows = run_pairs([("cpsc", 9)], [("m1", "m0")], n_boot=1000)
        write_csv(rows, OUTD / "paired_stats_smoke.csv")
        log("smoke 完成")
        return
    write_csv(run_pairs(DOMAINS, PAIRS_PRIMARY, N_BOOT), OUTD / "paired_stats.csv")
    run_interaction()
    run_tables()
    log("全量统计完成")


if __name__ == "__main__":
    main()
