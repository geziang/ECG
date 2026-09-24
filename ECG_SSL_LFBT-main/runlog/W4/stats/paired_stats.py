# -*- coding: utf-8 -*-
"""paired_stats.py —— W4 B-4 逐记录配对统计(主机B, CPU-only)。

读 runlog/W4/predictions/ 下 metrics_ext.save_eval_artifacts 落盘的
y_true.npy / y_pred.npy / y_prob.npy(A-8 预测重放与 A-4/A-5/A-6 下游评估产物),
对同一批 test 记录上的两个方法做配对检验, 产出 runlog/W4/stats/paired_stats.csv。

检验方法(与 stats_methods.md 一致):
  1) paired bootstrap on macro-AUROC(主检验): 记录级重采样, 多 seed 同步重采样,
     delta = mean_over_seeds(AUROC_a - AUROC_b), 双侧经验 p + 95% CI;
  2) DeLong per-class(辅助): 每类 one-vs-rest AUROC 的配对 DeLong 检验,
     报 per-class p 的 min/median(macro 层面不做解析合并, 方差独立性无保证)。

措辞边界(任务书 B-4 停止线): 全部为 record-level 统计, 同一患者的多条记录
视为相关样本未做 patient-level 聚类校正, 论文不得声称 patient-level 显著性。

目录布局约定(A-8 落盘时遵循): runlog/W4/predictions/{method}_{dataset}_seed{N}/
若实际布局不同, 仅需改 DISCOVER_RE 一处。

用法(在 ECG_SSL_LFBT-main/ 下运行):
  python runlog/W4/stats/paired_stats.py \
      --pred-root runlog/W4/predictions --out-dir runlog/W4/stats \
      --pairs c1:c2 c1:s1 c1:s2 c2:b0 --datasets ptbxl cpsc chapman --seeds 0 2 4
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils.pathguard import open_out  # noqa: E402

DISCOVER_RE = re.compile(r"^(?P<method>[a-z0-9]+)_(?P<dataset>[a-z0-9]+)_seed(?P<seed>\d+)$")

# 任务书 B-4 指定的默认对比对(各下游)
DEFAULT_PAIRS = ["c1:c2", "c1:s1", "c1:s2", "c2:b0"]
DEFAULT_DATASETS = ["ptbxl", "cpsc", "chapman"]
DEFAULT_SEEDS = [0, 2, 4]
N_BOOT = 10000
RNG_SEED = 20260923


# ---------------------------------------------------------------- 预测加载
def load_pred_dir(d: Path):
    """读一个 save_eval_artifacts 目录, 做一致性校验后返回 (y_true, y_pred, y_prob)。"""
    y_true = np.load(d / "y_true.npy")
    y_pred = np.load(d / "y_pred.npy")
    y_prob = np.load(d / "y_prob.npy")
    if not (len(y_true) == len(y_pred) == len(y_prob)):
        raise ValueError(f"{d}: 三数组长度不一致")
    if not np.array_equal(y_pred, y_prob.argmax(axis=1).astype(y_pred.dtype)):
        raise ValueError(f"{d}: y_pred 与 argmax(y_prob) 不一致(目录可能被改动)")
    return y_true.astype(int), y_pred.astype(int), y_prob.astype(float)


def discover(pred_root: Path):
    """扫描预测根目录 -> {(method, dataset, seed): Path}; 布局不符的目录跳过并警告。"""
    found, skipped = {}, []
    for d in sorted(p.iterdir() if (p := pred_root).is_dir() else []):
        if not d.is_dir() or not (d / "y_prob.npy").exists():
            continue
        m = DISCOVER_RE.match(d.name)
        if m:
            found[(m["method"], m["dataset"], int(m["seed"]))] = d
        else:
            skipped.append(d.name)
    return found, skipped


def check_pair_aligned(y_true_a, y_true_b, tag):
    """配对前提: 两组预测必须来自同一 test 顺序(同 y_true)。"""
    if y_true_a.shape != y_true_b.shape or not np.array_equal(y_true_a, y_true_b):
        raise ValueError(f"{tag}: 两组预测的 y_true 不一致, 不能做配对检验"
                         f"(shape {y_true_a.shape} vs {y_true_b.shape})")


# ---------------------------------------------------------------- AUROC
def _binary_auroc(y_bin, score):
    """秩基二类 AUROC(Mann-Whitney, 平均秩处理 ties)。"""
    y_bin = np.asarray(y_bin, dtype=float)
    score = np.asarray(score, dtype=float)
    n_pos, n_neg = y_bin.sum(), (1 - y_bin).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):  # 平均秩(ties)
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    sum_pos = ranks[y_bin == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def macro_auroc(y_true, prob):
    """macro one-vs-rest AUROC + 每类分量(空类返回 nan 并在均值中剔除)。"""
    n_classes = prob.shape[1]
    one_hot = np.eye(n_classes)[y_true]
    per = np.array([_binary_auroc(one_hot[:, c], prob[:, c]) for c in range(n_classes)])
    return float(np.nanmean(per)), per


# ---------------------------------------------------------------- DeLong(per-class, 配对)
def _delong_components(y_bin, score):
    """DeLong 结构成分: V10(i)=dA/dx_i(阳性侧), V01(i)(阴性侧)。

    V10_i = (rank_i - 1 - #(同分阴性在 i 前)) / n_neg  (i 为阳性)
    V01_i = #((阳性分 > s_i) + 0.5*#(同分阳性)) / n_pos (i 为阴性)
    AUC = mean(V10 over 阳性) = mean(V01 over 阴性)。
    """
    y_bin = np.asarray(y_bin, dtype=float)
    score = np.asarray(score, dtype=float)
    pos, neg = y_bin == 1, y_bin == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return None, None
    v10 = np.zeros(len(y_bin))
    v01 = np.zeros(len(y_bin))
    order = np.argsort(score, kind="mergesort")
    s_sorted = score[order]
    y_sorted = y_bin[order]
    # 累积: 走过排序序列, 记录严格小于当前分的阴性计数与同分计数
    cum_neg_before = 0
    i = 0
    neg_lt = np.zeros(len(s_sorted))   # 分数严格小于 s_i 的阴性数
    tie_neg = np.zeros(len(s_sorted))  # 与 s_i 同分的阴性数
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        blk_neg_lt, blk_tie_neg = 0, 0
        for k in range(i, j + 1):
            if y_sorted[k] == 0:
                blk_tie_neg += 1
            neg_lt[k] = cum_neg_before
            tie_neg[k] = None  # 占位, 组内同分阴性数一致, 下一步回填
        # 回填组内 tie_neg
        for k in range(i, j + 1):
            tie_neg[k] = blk_tie_neg
        cum_neg_before += blk_tie_neg
        i = j + 1
    # 阳性侧: V10_i = (n_neg_before_i + 0.5*tie_neg_i) / n_neg  (等于平均秩公式)
    inv_order = np.empty(len(order), dtype=int)
    inv_order[order] = np.arange(len(order))
    nl, tn = neg_lt[inv_order], tie_neg[inv_order]
    v10[pos] = (nl[pos] + 0.5 * tn[pos]) / n_neg
    # 阴性侧: V01_i = (n_pos_above_i + 0.5*tie_pos_i) / n_pos
    # n_pos_above_i = n_pos - (n_pos_leq_i); 用对称的从高到低扫描等价于:
    # V01_i = 1 - (n_pos_below + 0.5*tie_pos)/n_pos - ... 直接用 AUC 对称式:
    # AUC = P(s_pos > s_neg) + 0.5 P(tie); V01_i = P(随机阳性胜过 i)
    pos_scores = score[pos]
    # 对每个阴性 i: 严格大于 s_i 的阳性数 + 0.5*同分阳性数
    ps_sorted = np.sort(pos_scores)
    idx = np.searchsorted(ps_sorted, score[neg], side="left")   # < s_i 的阳性数
    gt = n_pos - np.searchsorted(ps_sorted, score[neg], side="right")
    tie_p = n_pos - gt - idx
    v01[neg] = (gt + 0.5 * tie_p) / n_pos
    return v10, v01


def delong_paired_class(y_true, prob_a, prob_b, c):
    """同一记录上两模型第 c 类 OvR AUROC 的配对 DeLong 检验。

    返回 (auc_a, auc_b, delta, z, p); z 经标准正态双侧 p。
    """
    n_classes = prob_a.shape[1]
    one_hot = np.eye(n_classes)[y_true]
    y_bin = one_hot[:, c]
    va10, va01 = _delong_components(y_bin, prob_a[:, c])
    vb10, vb01 = _delong_components(y_bin, prob_b[:, c])
    if va10 is None or vb10 is None:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    pos, neg = y_bin == 1, y_bin == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    auc_a = va10[pos].mean()
    auc_b = vb10[pos].mean()
    delta = auc_a - auc_b
    # 配对协方差(同一批记录): 结构成分按记录拼成两列再算协方差
    xa = np.where(pos, va10, va01)
    xb = np.where(pos, vb10, vb01)
    va = xa.var(ddof=1) / n_pos + xa[neg].var(ddof=1) / n_neg  # A 的方差成分
    vb = xb.var(ddof=1) / n_pos + xb[neg].var(ddof=1) / n_neg
    cov = np.cov(xa, xb, ddof=1)[0, 1] * (1.0 / n_pos + 1.0 / n_neg)
    var_d = va + vb - 2.0 * cov
    if var_d <= 0:
        z = 0.0 if delta == 0 else np.inf * np.sign(delta)
    else:
        z = delta / np.sqrt(var_d)
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))
    return float(auc_a), float(auc_b), float(delta), float(z), float(min(max(p, 0.0), 1.0))


# ---------------------------------------------------------------- paired bootstrap(主检验)
def _auroc_rows(y_bin_mat, scores_mat):
    """向量化逐行 AUROC: y_bin_mat (B, n) 0/1, scores_mat (B, n) -> (B,)。

    rankdata 沿 axis=1 平均秩处理 ties, Mann-Whitney 公式与 _binary_auroc
    单样本口径一致; 标签与得分必须同为重采样后的行(乘掩码支持逐行不同标签)。
    """
    from scipy.stats import rankdata
    y_bin_mat = y_bin_mat.astype(float)
    ranks = rankdata(scores_mat, axis=1)
    n_pos = y_bin_mat.sum(axis=1)
    n_neg = (1.0 - y_bin_mat).sum(axis=1)
    ok = (n_pos > 0) & (n_neg > 0)
    r_pos = (ranks * y_bin_mat).sum(axis=1)
    out = np.full(scores_mat.shape[0], np.nan)
    out[ok] = ((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))[ok]
    return out


def _macro_auroc_boot(y_true, prob_mat, idx):
    """重采样行集上的 macro AUROC。

    idx (B, n) 重采样索引同时作用于标签与得分; prob_mat 为重采样后得分
    (B, n, C) = prob[idx]; y_true (n,) 原始标签。返回 (B,)。
    """
    n_classes = prob_mat.shape[2]
    one_hot = np.eye(n_classes)[y_true]          # (n, C)
    per = np.stack([_auroc_rows(one_hot[:, c][idx], prob_mat[:, :, c])
                    for c in range(n_classes)], axis=1)
    return np.nanmean(per, axis=1)


def paired_bootstrap_macro(y_true, prob_a, prob_b, n_boot=N_BOOT, rng=None,
                           block=1000):
    """记录级配对 bootstrap on macro-AUROC 差。

    重采样记录索引, 同一索引同时作用于标签/方法A/方法B(配对), delta 分布给
    双侧 p 与 95% CI。多 seed 合并模式: 传入 list[prob_a]/list[prob_b], 同一
    bootstrap 索引同步作用于全部 seed, delta_boot = mean_over_seeds。
    实现: 按块生成索引矩阵 (block, n) 后向量化(scipy rankdata 沿批次轴,
    Mann-Whitney 平均秩口径与 _binary_auroc 逐次循环一致)。
    """
    rng = rng or np.random.default_rng(RNG_SEED)
    probs_a = prob_a if isinstance(prob_a, list) else [prob_a]
    probs_b = prob_b if isinstance(prob_b, list) else [prob_b]
    n = len(y_true)
    base = [macro_auroc(y_true, pa)[0] - macro_auroc(y_true, pb)[0]
            for pa, pb in zip(probs_a, probs_b)]
    delta_obs = float(np.mean(base))
    deltas = np.empty(n_boot)
    done = 0
    while done < n_boot:
        b = min(block, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        d = np.zeros(b)
        for pa, pb in zip(probs_a, probs_b):
            d += _macro_auroc_boot(y_true, pa[idx], idx) \
                - _macro_auroc_boot(y_true, pb[idx], idx)
        deltas[done:done + b] = d / len(probs_a)
        done += b
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return delta_obs, float(min(max(p, 0.0), 1.0)), float(lo), float(hi)


# ---------------------------------------------------------------- 主流程
def run(args):
    pred_root = Path(args.pred_root)
    found, skipped = discover(pred_root)
    if skipped:
        print(f"[warn] 布局未识别已跳过: {skipped}", file=sys.stderr)
    out_dir = Path(args.out_dir)
    rows = []
    for pair in args.pairs:
        ma, mb = pair.split(":")
        for ds in args.datasets:
            # 收集两方法在该下游的全部 seed
            dirs_a = {s: found.get((ma, ds, s)) for s in args.seeds if (ma, ds, s) in found}
            dirs_b = {s: found.get((mb, ds, s)) for s in args.seeds if (mb, ds, s) in found}
            common = sorted(set(dirs_a) & set(dirs_b))
            if not common:
                print(f"[skip] {pair}@{ds}: 无共同 seed (A={sorted(dirs_a)} B={sorted(dirs_b)})")
                continue
            data_a = [load_pred_dir(dirs_a[s]) for s in common]
            data_b = [load_pred_dir(dirs_b[s]) for s in common]
            for (yt_a, _, pa), (yt_b, _, pb) in zip(data_a, data_b):
                check_pair_aligned(yt_a, yt_b, f"{pair}@{ds}")
            y_true = data_a[0][0]
            # ---- 主检验: paired bootstrap on macro-AUROC(seeds 同步合并) ----
            probs_a = [d[2] for d in data_a]
            probs_b = [d[2] for d in data_b]
            delta, p_boot, lo, hi = paired_bootstrap_macro(
                y_true, probs_a, probs_b, args.n_boot,
                np.random.default_rng(RNG_SEED))
            aucs_a = [macro_auroc(y_true, pa)[0] for pa in probs_a]
            aucs_b = [macro_auroc(y_true, pb)[0] for pb in probs_b]
            # ---- 辅助: DeLong per-class(取 seed0; per-class p 的 min/median) ----
            ps = [delong_paired_class(y_true, probs_a[0], probs_b[0], c)[4]
                  for c in range(probs_a[0].shape[1])]
            ps = [x for x in ps if not np.isnan(x)]
            rows.append({
                "dataset": ds, "method_a": ma, "method_b": mb,
                "seeds": ";".join(map(str, common)), "n_records": len(y_true),
                "auroc_macro_a": round(float(np.mean(aucs_a)), 6),
                "auroc_macro_b": round(float(np.mean(aucs_b)), 6),
                "delta_macro": round(delta, 6),
                "boot_p": round(p_boot, 5),
                "boot_ci95_low": round(lo, 6), "boot_ci95_high": round(hi, 6),
                "delong_p_min(seed0)": round(min(ps), 5) if ps else "",
                "delong_p_median(seed0)": round(float(np.median(ps)), 5) if ps else "",
                "n_boot": args.n_boot, "n_classes": probs_a[0].shape[1],
                "level": "record", "note": "record-level; 多seed=bootstrap同步重采样均值",
            })
            print(f"[done] {pair}@{ds}: delta={delta:+.4f} boot_p={p_boot:.4f}")
    if not rows:
        print(f"[empty] {pred_root} 下无可对比预测(等待 A-8 落盘后重跑)", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open_out(out_dir, "paired_stats.csv", mode="w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] {len(rows)} 行 -> {out_dir / 'paired_stats.csv'}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred-root", default="runlog/W4/predictions")
    ap.add_argument("--out-dir", default="runlog/W4/stats")
    ap.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
