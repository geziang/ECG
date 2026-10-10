# -*- coding: utf-8 -*-
"""W8 E2 (w8-simph) 开关纪律单测: SimCLR-Physio = W4 S1 全套 + 唯一差异 --nstdb-aug 0.5。

E2 未新增任何仓库代码(两个既有独立开关的组合), 本测试把"默认关==S1 逐位一致"的
结构依据钉死:
  1. NoiseInjector(prob=0) 严格恒等(逐位)——关态不触碰样本;
  2. prob=1 时注入确实发生(形状/类型保持, 数值改变)——开态生效;
  3. run_pt.py 源码结构断言: --nstdb-aug 默认 0.0 且守卫块以 >0 为条件(默认不执行);
     --loss-mode 默认 bt; --simclr-temp 默认 0.5(S1 审计口径);
  4. 注入器为无状态对象(两视图复用同一实例但每次调用独立抽随机)——"两视图独立"成立。
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(name, cond):
    if not cond:
        raise AssertionError(name)


class TestW8E2(unittest.TestCase):

    def test_prob0_identity_bitwise(self):
        from data_utils.nstdb_aug import NoiseInjector
        rng = np.random.RandomState(7)
        x = rng.randn(8, 2048).astype(np.float32)
        nj = NoiseInjector(prob=0.0)
        y = nj(x)
        self.assertTrue(np.array_equal(x, y), "prob=0 必须逐位原样返回")

    def test_prob1_injects(self):
        from data_utils.nstdb_aug import NoiseInjector
        d = ROOT / "data" / "nstdb"
        if not (d / "bw.dat").exists():
            self.skipTest("本机无 data/nstdb(注入路径需真实噪声源)")
        np.random.seed(123)
        rng = np.random.RandomState(7)
        x = rng.randn(8, 2048).astype(np.float32)
        nj = NoiseInjector(prob=1.0, snr_lo=5.0, snr_hi=20.0)
        y = nj(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, np.float32)
        self.assertFalse(np.allclose(x, y), "prob=1 必须改变样本")

    def test_injector_stateless_two_views_independent(self):
        from data_utils.nstdb_aug import NoiseInjector
        # 两视图复用同一实例: 每个 __call__ 内部独立抽 random()/噪声类型/偏移/SNR,
        # 无实例级状态 -> "两视图独立注入"由结构保证。
        nj = NoiseInjector(prob=0.5)
        self.assertEqual(vars(nj), {"prob": 0.5, "snr_lo": 5.0, "snr_hi": 20.0,
                                    "offset_range": "full"})

    def test_runpt_source_guards(self):
        src = (ROOT / "run_pt.py").read_text(encoding="utf-8")
        self.assertIn("'--nstdb-aug', default=0.0", src, "nstdb 开关默认必须为 0")
        self.assertIn("if float(getattr(args, 'nstdb_aug', 0.0) or 0.0) > 0:", src,
                      "nstdb 守卫块必须以 >0 为条件(默认不执行)")
        self.assertIn("'--loss-mode', default='bt'", src, "loss-mode 默认必须 bt")
        self.assertIn("'--simclr-temp', default=0.5", src, "S1 审计口径 tau=0.5")


if __name__ == "__main__":
    unittest.main(verbosity=2)
