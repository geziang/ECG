# -*- coding: utf-8 -*-
"""W8 E6 (w8-noiseiso) 单测: 噪声时间片段隔离(训练前 50% / 评测后 50%)。

核心断言:
  1. draw_offset(full) 与历史行为同式(randint(0, Ln-L)), 默认参数即 full;
  2. draw_offset(first_half) 全部落在 [0, Ln//2-L); last_half 全部落在 [Ln//2, Ln-L);
  3. 非法 offset_range 拒绝; 记录过短回退全域不炸;
  4. NoiseInjector 默认构造 offset_range='full'(历史行为), 透传生效;
  5. run_pt 源码守卫: --nstdb-offset 默认 full 且 choices 限三值, 注入器带 offset_range 构造。
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_utils.nstdb_aug import NoiseInjector, draw_offset


def draws(rng, n, Ln, L, mode):
    return [draw_offset(rng, Ln, L, mode) for _ in range(n)]


class TestW8E6(unittest.TestCase):

    def test_full_matches_legacy(self):
        np.random.seed(1)
        a = draws(np.random, 50, 100000, 2048, "full")
        np.random.seed(1)
        b = [np.random.randint(0, 100000 - 2048) for _ in range(50)]
        self.assertEqual(a, b, "full 必须与历史 randint(0, Ln-L) 同分布同调用")

    def test_first_half_range(self):
        np.random.seed(2)
        Ln, L = 100000, 2048
        os_ = draws(np.random, 200, Ln, L, "first_half")
        self.assertTrue(all(0 <= o < Ln // 2 - L for o in os_))

    def test_last_half_range(self):
        np.random.seed(3)
        Ln, L = 100000, 2048
        os_ = draws(np.random, 200, Ln, L, "last_half")
        self.assertTrue(all(Ln // 2 <= o < Ln - L for o in os_))

    def test_halves_disjoint_and_coverable(self):
        np.random.seed(4)
        Ln, L = 50000, 2048
        f = set(draws(np.random, 100, Ln, L, "first_half"))
        l = set(draws(np.random, 100, Ln, L, "last_half"))
        self.assertFalse(f & l, "前后半段采样域不得重叠")

    def test_invalid_range_rejected(self):
        with self.assertRaises(ValueError):
            draw_offset(np.random, 1000, 100, "middle")

    def test_short_record_fallback(self):
        # 半段装不下窗口时回退全域(Ln>L 前提下), 不抛异常
        Ln, L = 3000, 2048  # Ln//2-L<1 且 Ln//2 > Ln-L -> 两分支均触发回退
        for mode in ("first_half", "last_half"):
            o = draw_offset(np.random.RandomState(0), Ln, L, mode)
            self.assertTrue(0 <= o < Ln - L)

    def test_injector_default_and_passthrough(self):
        nj = NoiseInjector(prob=0.5)
        self.assertEqual(nj.offset_range, "full", "默认必须 full(历史行为)")
        nj6 = NoiseInjector(prob=0.5, offset_range="first_half")
        self.assertEqual(nj6.offset_range, "first_half")
        # 状态字典断言顺带守护(两视图独立结构)
        self.assertEqual(vars(nj6)["offset_range"], "first_half")

    def test_runpt_source_guards(self):
        src = (ROOT / "run_pt.py").read_text(encoding="utf-8")
        self.assertIn("'--nstdb-offset', default='full'", src, "开关默认必须 full")
        self.assertIn("choices=['full', 'first_half', 'last_half']", src)
        self.assertIn("offset_range=str(getattr(args, 'nstdb_offset', 'full') or 'full')", src,
                      "注入器必须显式透传 offset_range")


if __name__ == "__main__":
    unittest.main(verbosity=2)
