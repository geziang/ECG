# -*- coding: utf-8 -*-
"""W3 B-2 单测: metrics_ext.py 全指标手工可验算例 + 落盘守卫 + run_lp/run_ft 默认关闭。

运行: python -m pytest tests/test_metrics_ext.py -q   (或 python tests/test_metrics_ext.py)
无需 GPU/数据; metrics_ext 仅依赖 numpy。
"""
import json
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import metrics_ext as mx  # noqa: E402


class TestAveragePrecision(unittest.TestCase):
    def test_perfect_ranking(self):
        self.assertEqual(mx.average_precision([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]), 1.0)

    def test_worst_ranking(self):
        # 正类全排最后: 逐步 precision = {1/3, 2/4} -> AP=(1/3+1/2)/2=5/12
        self.assertAlmostEqual(mx.average_precision([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]),
                               5.0 / 12.0, places=12)

    def test_no_positive(self):
        self.assertTrue(np.isnan(mx.average_precision([0, 0], [0.5, 0.4])))

    def test_matches_sklearn_no_ties(self):
        try:
            from sklearn.metrics import average_precision_score
        except ImportError:
            self.skipTest("sklearn 不可用")
        rng = np.random.default_rng(7)
        y = rng.integers(0, 2, 200)
        s = rng.random(200)  # 连续分数无并列
        self.assertAlmostEqual(mx.average_precision(y, s),
                               average_precision_score(y, s), places=10)


class TestPerClassAP(unittest.TestCase):
    def test_two_class_macro(self):
        # y=[0,1,1,0], p(class1)=[0.2,0.9,0.6,0.3] -> class1 AP: 排序 y=[1,1,0,0] AP=1.0
        # class0 等价于 1-p 排序 -> 同样完美 AP=1.0 -> macro=1.0
        out = mx.per_class_ap([0, 1, 1, 0], [[0.8, 0.2], [0.1, 0.9], [0.4, 0.6], [0.7, 0.3]])
        self.assertAlmostEqual(out["ap_class_0"], 1.0)
        self.assertAlmostEqual(out["ap_class_1"], 1.0)
        self.assertAlmostEqual(out["macro_ap"], 1.0)

    def test_matches_sklearn_macro(self):
        try:
            from sklearn.metrics import average_precision_score
        except ImportError:
            self.skipTest("sklearn 不可用")
        rng = np.random.default_rng(3)
        y = rng.integers(0, 3, 300)
        prob = rng.random((300, 3))
        prob /= prob.sum(1, keepdims=True)
        out = mx.per_class_ap(y, prob)
        sk = average_precision_score(np.eye(3)[y], prob)  # 2D 默认 macro
        self.assertAlmostEqual(out["macro_ap"], sk, places=10)


class TestConfMatMetrics(unittest.TestCase):
    def test_hand_case(self):
        cm = [[8, 2], [3, 7]]
        m = mx.conf_mat_metrics(cm)
        self.assertAlmostEqual(m["class_0_precision"], 8 / 11)
        self.assertAlmostEqual(m["class_0_sensitivity"], 8 / 10)
        self.assertAlmostEqual(m["class_0_specificity"], 7 / 10)
        self.assertAlmostEqual(m["class_1_precision"], 7 / 9)
        self.assertAlmostEqual(m["class_1_sensitivity"], 7 / 10)
        self.assertAlmostEqual(m["class_1_specificity"], 8 / 10)
        self.assertAlmostEqual(m["accuracy"], 15 / 20)
        f1_0 = 2 * (8 / 11) * (8 / 10) / ((8 / 11) + (8 / 10))
        self.assertAlmostEqual(m["class_0_f1"], f1_0)
        self.assertAlmostEqual(m["macro_f1"], (f1_0 + 2 * (7 / 9) * 0.7 / (7 / 9 + 0.7)) / 2)

    def test_empty_class_no_crash(self):
        m = mx.conf_mat_metrics([[0, 0], [5, 5]])  # 类0无样本
        self.assertTrue(np.isnan(m["class_0_sensitivity"]))
        self.assertAlmostEqual(m["class_1_sensitivity"], 0.5)


class TestCalibration(unittest.TestCase):
    def test_perfect(self):
        m = mx.calibration([0, 1], [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(m["ece"], 0.0)
        self.assertEqual(m["brier_multiclass"], 0.0)

    def test_confidently_wrong(self):
        m = mx.calibration([0], [[0.0, 1.0]])
        self.assertAlmostEqual(m["ece"], 1.0)
        self.assertAlmostEqual(m["brier_multiclass"], 1.0)

    def test_overconfident(self):
        m = mx.calibration([0, 1], [[0.75, 0.25], [0.25, 0.75]])
        self.assertAlmostEqual(m["ece"], 0.25)
        self.assertAlmostEqual(m["brier_multiclass"], 0.0625)

    def test_bin_edges(self):
        m = mx.calibration([0], [[1.0, 0.0]], n_bins=4)  # conf=1.0 -> clip 到最后一桶
        last = m["reliability_table"][-1]
        self.assertEqual(last["n"], 1)
        self.assertEqual(last["range"], "[0.750,1.000)")
        self.assertEqual(len(m["reliability_table"]), 4)


class TestSaveArtifacts(unittest.TestCase):
    TMP = ROOT / "runlog" / "W3" / "metrics_schema" / "_selftest_tmp"

    def tearDown(self):
        if self.TMP.exists():
            shutil.rmtree(self.TMP)

    def test_guard_rejects_w2(self):
        with self.assertRaises(ValueError):
            mx.save_eval_artifacts(ROOT / "runlog" / "W2" / "x", [0], [0], [[1.0]],
                                   {"protocol_id": "t"})

    def test_guard_rejects_outside_w3(self):
        with self.assertRaises(ValueError):
            mx.save_eval_artifacts(ROOT / "results" / "x", [0], [0], [[1.0]],
                                   {"protocol_id": "t"})

    def test_guard_allows_w4_and_still_rejects_w2(self):
        """W4 A-8 扩展: 允许 runlog/W4/ 之下; W2 硬拒与仓库外拒绝不变。"""
        w4 = ROOT / "runlog" / "W4" / "metrics_schema" / "_selftest_tmp"
        try:
            mx.save_eval_artifacts(w4, [0], [0], [[1.0]], {"protocol_id": "t"})
            self.assertTrue((w4 / "metrics_ext.json").exists())
        finally:
            if w4.exists():
                shutil.rmtree(w4)
        with self.assertRaises(ValueError):
            mx.save_eval_artifacts(ROOT / "runlog" / "W2" / "x", [0], [0], [[1.0]],
                                   {"protocol_id": "t"})
        with self.assertRaises(ValueError):
            mx.save_eval_artifacts(ROOT / "results" / "x", [0], [0], [[1.0]],
                                   {"protocol_id": "t"})

    def test_guard_allows_w5_and_still_rejects_w2(self):
        """W5 A-2 扩展(CPSC重评估): 允许 runlog/W5/ 之下; W2 硬拒与仓库外拒绝不变。"""
        w5 = ROOT / "runlog" / "W5" / "metrics_schema" / "_selftest_tmp"
        try:
            mx.save_eval_artifacts(w5, [0], [0], [[1.0]], {"protocol_id": "t"})
            self.assertTrue((w5 / "metrics_ext.json").exists())
        finally:
            if w5.exists():
                shutil.rmtree(w5)
        with self.assertRaises(ValueError):
            mx.save_eval_artifacts(ROOT / "runlog" / "W2" / "x", [0], [0], [[1.0]],
                                   {"protocol_id": "t"})
        with self.assertRaises(ValueError):
            mx.save_eval_artifacts(ROOT / "results" / "x", [0], [0], [[1.0]],
                                   {"protocol_id": "t"})

    def test_guard_requires_protocol_id(self):
        with self.assertRaises(ValueError):
            mx.save_eval_artifacts(self.TMP, [0], [0], [[1.0]], {})

    def test_roundtrip(self):
        y = [0, 1, 2]
        prob = [[0.8, 0.1, 0.1], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]]
        mx.save_eval_artifacts(self.TMP, y, np.array(prob).argmax(1), prob,
                               {"protocol_id": "selftest"})
        self.assertTrue((self.TMP / "y_true.npy").exists())
        self.assertTrue((self.TMP / "y_prob.npy").exists())
        payload = json.loads((self.TMP / "metrics_ext.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["metadata"]["protocol_id"], "selftest")
        self.assertIn("macro_f1", payload)
        json.dumps(payload)  # 全量 JSON 可序列化


class TestEntryScriptsUnchangedByDefault(unittest.TestCase):
    """B-2 红线: 新开关默认全关; 脚本可编译。"""

    def test_py_compile(self):
        import py_compile
        py_compile.compile(str(ROOT / "run_lp.py"), doraise=True)
        py_compile.compile(str(ROOT / "run_ft.py"), doraise=True)
        py_compile.compile(str(ROOT / "metrics_ext.py"), doraise=True)

    def test_argparse_defaults_off(self):
        import argparse
        import importlib.util
        for name in ("run_lp", "run_ft"):
            spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
            # 不执行模块(torch 重): 只静态检查源码中默认值为关
            src = (ROOT / f"{name}.py").read_text(encoding="utf-8")
            self.assertIn("'--extended-metrics', default=0", src, name)
            self.assertIn("'--save-predictions', default=''", src, name)
            self.assertIn("'--protocol-id', default=''", src, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
