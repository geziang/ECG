# -*- coding: utf-8 -*-
"""有效秩诊断 (A-P2 第2项): 裁决 H1(分组白化)立项与否。

用法(直接吃 run_lp 缓存特征, 秒级, 不占 GPU):
  python diagnose_rank.py --feat-dir feat/ptxl_gamma08        # B0
  python diagnose_rank.py --feat-dir feat/ptxl_lp_tfs         # TFS 随机初始化对照

口径(06 §v2.4 冻结口径的实现):
  - 输入: LP 缓存的 X_train.npy [N, 512] (8 导联 × 64 维 concat)
  - 每导联 64 维子块分别计算, 再给全 512 维总口径:
    erank   = exp(H(p)), p_i = s_i^2 / Σ s_j^2, s 为中心化特征矩阵奇异值 (Roy-Vetterli 有效秩)
    pr      = 参与率 (Σs)^2 / Σ s^2
    rankme  = RankMe 口径(未中心化) exp(H(p̃)), p̃ 基于未中心化奇异值
  - 判读(06 判据): erank/d 比值显著低于 1(如 <0.5)视为有效维度塌缩 -> H1 才立项
输出: runlog/M/rank_diag_<name>.json + 控制台表
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def erank_from_svd(X, center=True):
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    p = s ** 2 / (s ** 2).sum()
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def participation_ratio(X, center=True):
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    return float((s.sum() ** 2) / (s ** 2).sum())


def diagnose(feat_dir: Path):
    X = np.load(feat_dir / "X_train.npy", allow_pickle=False).astype(np.float64)
    N, D = X.shape
    out = {"feat_dir": str(feat_dir), "N": int(N), "D": int(D),
           "per_lead": {}, "ts": time.strftime("%m-%d %H:%M")}
    rows = []
    for li in range(D // 64):
        sub = X[:, li * 64:(li + 1) * 64]
        er, rm, pr = erank_from_svd(sub), erank_from_svd(sub, center=False), participation_ratio(sub)
        lead = ["II", "III", "V1", "V2", "V3", "V4", "V5", "V6"][li]
        out["per_lead"][lead] = {"erank": round(er, 2), "erank/d": round(er / 64, 3),
                                 "rankme": round(rm, 2), "pr": round(pr, 2)}
        rows.append((lead, er, er / 64, rm, pr))
    er, rm, pr = erank_from_svd(X), erank_from_svd(X, center=False), participation_ratio(X)
    out["overall"] = {"erank": round(er, 2), "erank/d": round(er / D, 3),
                      "rankme": round(rm, 2), "pr": round(pr, 2)}
    out["h1_gate"] = "塌缩(H1 立项)" if er / D < 0.5 else "未塌缩(H1 不立项)"

    print(f"\n== 有效秩诊断: {feat_dir} (N={N}, D={D}) ==")
    print(f"{'导联':>4} {'erank':>7} {'erank/d':>8} {'rankme':>7} {'PR':>6}")
    for lead, e, r, m, p in rows:
        print(f"{lead:>4} {e:>7.2f} {r:>8.3f} {m:>7.2f} {p:>6.2f}")
    print(f"{'总体':>4} {er:>7.2f} {er / D:>8.3f} {rm:>7.2f} {pr:>6.2f}")
    print(f"H1 门: {out['h1_gate']}")
    name = feat_dir.name.replace("/", "_")
    dst = ROOT / "runlog" / "M" / f"rank_diag_{name}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已写 {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True, type=Path)
    diagnose(ap.parse_args().feat_dir.resolve())
