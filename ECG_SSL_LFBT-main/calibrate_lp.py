# -*- coding: utf-8 -*-
"""H5 评测校准 (A-P2 第1项): LP 特征标准化 + 正则小网格, 不重训编码器。

用法(吃 run_lp 缓存特征, CPU 数分钟):
  python calibrate_lp.py --feat-dir feat/ptxl_gamma08 --name b0

协议(冻结): train 上 StandardScaler + LogisticRegression(C 网格), **val 选 C, test 仅评一次**;
输出新旧口径对照与推荐新锚点, 写 runlog/M/h5_calib_<name>.json。
注意(HOSTS §五-2): 新锚点须经确认冻结后方可作为 Δ 分母, 在此之前一切 Δ 仍用旧口径。
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, label_binarize

ROOT = Path(__file__).resolve().parent
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def macro_auprc(y_true, score):
    Y = label_binarize(y_true, classes=np.arange(score.shape[1]))
    return float(np.mean([average_precision_score(Y[:, i], score[:, i])
                          for i in range(score.shape[1]) if Y[:, i].sum() > 0]))


def macro_auroc(y_true, score):
    return float(roc_auc_score(y_true, score, multi_class="ovr", average="macro"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True, type=Path)
    ap.add_argument("--name", default="b0")
    a = ap.parse_args()
    fd = a.feat_dir.resolve()
    Xtr, ytr = np.load(fd / "X_train.npy"), np.load(fd / "y_train.npy")
    Xv, yv = np.load(fd / "X_val.npy"), np.load(fd / "y_val.npy")
    Xte, yte = np.load(fd / "X_test.npy"), np.load(fd / "y_test.npy")
    ytr, yv, yte = ytr.ravel(), yv.ravel(), yte.ravel()

    sc = StandardScaler().fit(Xtr)
    results = {"feat_dir": str(fd), "grid": C_GRID, "ts": time.strftime("%m-%d %H:%M")}
    best = None
    for standardize in (False, True):
        A = sc.transform if standardize else (lambda x: x)
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=3000, n_jobs=8)
            clf.fit(A(Xtr), ytr)
            sv = clf.predict_proba(A(Xv))
            au = macro_auprc(yv, sv)
            tag = "std" if standardize else "raw"
            results[f"{tag}_C{C}"] = {"val_auprc": round(au, 4),
                                      "val_auroc": round(macro_auroc(yv, sv), 4)}
            if best is None or au > best[0]:
                best = (au, standardize, C)
    _, use_std, use_C = best
    # test 仅评一次(用 val 选出的唯一配置)
    A = sc.transform if use_std else (lambda x: x)
    clf = LogisticRegression(C=use_C, max_iter=3000, n_jobs=8).fit(A(Xtr), ytr)
    st = clf.predict_proba(A(Xte))
    results["chosen"] = {"standardize": use_std, "C": use_C}
    results["test"] = {"auprc": round(macro_auprc(yte, st), 4),
                       "auroc": round(macro_auroc(yte, st), 4)}
    old = json.loads((fd / "metrics.json").read_text(encoding="utf-8")) \
        if (fd / "metrics.json").exists() else {}
    results["old_head"] = {k: old.get(k) for k in ("auroc", "auprc")}

    print(json.dumps(results, ensure_ascii=False, indent=1))
    dst = ROOT / "runlog" / "M" / f"h5_calib_{a.name}.json"
    dst.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已写 {dst}\n提示: 新锚点冻结前, 一切 Δ 仍按旧口径; 冻结需在 HOSTS §五-2 登记生效时间。")


if __name__ == "__main__":
    main()
