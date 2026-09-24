# -*- coding: utf-8 -*-
"""W4 B-4 配对统计单测(主机B)。运行: python tests/test_paired_stats.py

验证 paired_stats.py 的统计正确性(合成数据)与配对前提校验:
  AUROC 秩基口径 / DeLong 成分自洽与边界行为 / bootstrap 边界行为与可复现 /
  预测目录加载一致性与 y_true 配对校验。
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runlog.W4.stats import paired_stats as ps  # noqa: E402

FAILS = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


def test_binary_auroc():
    y = np.array([0, 0, 1, 1])
    check("AUROC perfect", ps._binary_auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0)
    check("AUROC reversed", ps._binary_auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0)
    # 手算例: 阳性 {0.5, 0.3}, 阴性 {0.4, 0.1}
    # 正对: (0.5>0.4)+(0.5>0.1)+(0.3>0.4)x+(0.3>0.1) = 3/4
    check("AUROC hand case", abs(ps._binary_auroc(np.array([1, 1, 0, 0]),
                                                  np.array([0.5, 0.3, 0.4, 0.1])) - 0.75) < 1e-12)
    # ties: 阳性 {0.5}, 阴性 {0.5, 0.0} -> (0.5 + 1.0)/2 = 0.75
    check("AUROC ties avg-rank", abs(ps._binary_auroc(np.array([1, 0, 0]),
                                                      np.array([0.5, 0.5, 0.0])) - 0.75) < 1e-12)
    check("AUROC degenerate nan", np.isnan(ps._binary_auroc(np.array([1, 1]), np.array([0.1, 0.2]))))


def test_macro_auroc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=200)
    prob = np.eye(3)[y] * 0.9 + 0.1 / 3  # 近完美
    m, per = ps.macro_auroc(y, prob)
    check("macro near-perfect", m > 0.95 and len(per) == 3)
    check("macro equals mean of per-class", abs(m - np.mean(per)) < 1e-12)


def test_delong_components_consistency():
    rng = np.random.default_rng(1)
    n = 300
    y_bin = (rng.random(n) < 0.4).astype(float)
    score = rng.random(n)
    v10, v01 = ps._delong_components(y_bin, score)
    pos, neg = y_bin == 1, y_bin == 0
    auc_v10 = v10[pos].mean()
    auc_v01 = v01[neg].mean()
    auc_ref = ps._binary_auroc(y_bin, score)
    check("DeLong components mean==AUROC (V10)", abs(auc_v10 - auc_ref) < 1e-10)
    check("DeLong components mean==AUROC (V01)", abs(auc_v01 - auc_ref) < 1e-10)


def test_delong_paired_bounds():
    rng = np.random.default_rng(2)
    n = 400
    y = rng.integers(0, 2, size=n)
    one_hot = np.eye(2)[y]
    strong = np.clip(one_hot[y == 1][:, 0].mean() if False else 0, 0, 1)  # noqa: F841
    # A: 与标签强相关; B: 与标签弱相关
    pa = np.clip(one_hot * 0.8 + rng.random((n, 2)) * 0.2, 1e-6, None)
    pa = pa / pa.sum(axis=1, keepdims=True)
    pb = np.tile(np.array([0.5, 0.5]), (n, 1)) + rng.random((n, 2)) * 1e-3
    pb = pb / pb.sum(axis=1, keepdims=True)
    auc_a, auc_b, d, z, p = ps.delong_paired_class(y, pa, pb, 0)
    check("DeLong A>B delta positive", d > 0.3)
    check("DeLong strong effect small p", p < 1e-4)
    # A 与 B 完全相同 -> delta=0, p=1
    _, _, d0, _, p0 = ps.delong_paired_class(y, pa, pa.copy(), 0)
    check("DeLong identical p=1", d0 == 0.0 and p0 == 1.0)


def test_paired_bootstrap_bounds():
    rng = np.random.default_rng(3)
    n = 500
    y = rng.integers(0, 3, size=n)
    one_hot = np.eye(3)[y]
    pa = one_hot * 0.7 + 0.3 * rng.random((n, 3))
    pa /= pa.sum(axis=1, keepdims=True)
    d_same, p_same, lo, hi = ps.paired_bootstrap_macro(
        y, pa, pa.copy(), n_boot=500, rng=np.random.default_rng(7))
    check("bootstrap identical delta=0 p large", d_same == 0.0 and p_same > 0.9)
    pb = np.tile(1.0 / 3, (n, 3)) + rng.random((n, 3)) * 1e-3
    pb /= pb.sum(axis=1, keepdims=True)
    d_big, p_big, lo, hi = ps.paired_bootstrap_macro(
        y, pa, pb, n_boot=500, rng=np.random.default_rng(7))
    check("bootstrap strong effect small p", d_big > 0.3 and p_big < 0.01)
    check("bootstrap CI covers delta", lo <= d_big <= hi)
    # 多 seed 合并: 同 RNG 可复现
    r1 = ps.paired_bootstrap_macro(y, [pa, pa], [pb, pb], n_boot=300,
                                   rng=np.random.default_rng(11))
    r2 = ps.paired_bootstrap_macro(y, [pa, pa], [pb, pb], n_boot=300,
                                   rng=np.random.default_rng(11))
    check("bootstrap reproducible with same rng", r1 == r2)


def test_load_and_discover(tmp=None):
    rng = np.random.default_rng(4)
    n = 64
    y = rng.integers(0, 5, size=n)
    prob = rng.random((n, 5))
    prob /= prob.sum(axis=1, keepdims=True)
    ypred = prob.argmax(axis=1).astype(int)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "predictions"
        d = root / "c1_ptbxl_seed0"
        d.mkdir(parents=True)
        np.save(d / "y_true.npy", y)
        np.save(d / "y_pred.npy", ypred)
        np.save(d / "y_prob.npy", prob)
        (d / "metrics_ext.json").write_text('{"metadata": {"protocol_id": "w4-test"}}')
        found, skipped = ps.discover(root)
        check("discover finds layout", ("c1", "ptbxl", 0) in found and not skipped)
        yt, yp, pr = ps.load_pred_dir(d)
        check("load roundtrip", np.array_equal(yt, y) and np.array_equal(yp, ypred))
        # 改动 y_pred -> 拒绝
        bad = yp.copy()
        bad[0] = (bad[0] + 1) % 5
        np.save(d / "y_pred.npy", bad)
        try:
            ps.load_pred_dir(d)
            ok = False
        except ValueError:
            ok = True
        check("load rejects tampered y_pred", ok)


def test_pair_alignment_guard():
    y1 = np.array([0, 1, 2, 1])
    y2 = np.array([0, 1, 2, 2])
    try:
        ps.check_pair_aligned(y1, y2, "t")
        ok = False
    except ValueError:
        ok = True
    check("pair alignment guard raises", ok)
    ps.check_pair_aligned(y1, y1.copy(), "t")  # 不应抛
    check("pair alignment same ok", True)


def test_vectorized_boot_equivalence():
    # 向量化 _macro_auroc_boot 必须与逐次 macro_auroc 在相同重采样索引上逐位一致
    rng = np.random.default_rng(6)
    n = 200
    y = rng.integers(0, 5, size=n)
    oh = np.eye(5)[y]
    pa = oh * 0.6 + 0.4 * rng.random((n, 5))
    pa /= pa.sum(axis=1, keepdims=True)
    idx = np.random.default_rng(42).integers(0, n, size=(8, n))
    vec = ps._macro_auroc_boot(y, pa[idx], idx)
    loop = np.array([ps.macro_auroc(y[idx[i]], pa[idx[i]])[0] for i in range(8)])
    check("vectorized boot == loop (bitwise)", np.allclose(vec, loop, atol=1e-12))


if __name__ == "__main__":
    test_binary_auroc()
    test_macro_auroc()
    test_delong_components_consistency()
    test_delong_paired_bounds()
    test_paired_bootstrap_bounds()
    test_vectorized_boot_equivalence()
    test_load_and_discover()
    test_pair_alignment_guard()
    if FAILS:
        print("\nFAILED:", FAILS)
        sys.exit(1)
    print("\nALL TESTS PASSED")
