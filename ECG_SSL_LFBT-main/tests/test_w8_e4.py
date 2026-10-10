# -*- coding: utf-8 -*-
"""W8 E4 (w8-trcaff) 单测: VGG16 trc=2 = 参数匹配仿射对照(去 ⊙n 门控)。

核心断言:
  1. 默认关(trc=0)不含 gamma/beta 键, 前向可跑(结构不变);
  2. trc=2 零初始化时, 与 trc=0 同权重前向逐位一致(开态初始=关态, 同 GRN1D 设计);
  3. trc=2 的 gamma/beta 有梯度(可学习, 非死参数);
  4. trc=2 与 trc=1 同非零 gamma 下输出不同(仿射确实去掉 n 门控);
  5. GRN1D 公式回归守卫(手动公式逐位一致, 证明本改动未触碰 TRC 原路径);
  6. trc 非法值拒绝。
"""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.vgg_1d import VGG16, GRN1D, AffineChannel1D


def make_input(seed=3, B=4, C=64, L=32):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, C, L, generator=g) * 3 + 1


class TestW8E4(unittest.TestCase):

    def test_trc0_no_calib_keys(self):
        m = VGG16(ch_in=1, alpha=0.125, trc=0)
        keys = set(m.state_dict().keys())
        self.assertFalse(any(("gamma" in k or "beta" in k) for k in keys),
                         "trc=0 不得含 gamma/beta 校准键")
        y = m(torch.randn(2, 1, 2048))
        self.assertEqual(tuple(y.shape), (2, 1000))

    def test_trc2_zero_init_bitwise_equal_trc0(self):
        torch.manual_seed(11)
        m0 = VGG16(ch_in=1, alpha=0.125, trc=0).eval()
        torch.manual_seed(11)
        m2 = VGG16(ch_in=1, alpha=0.125, trc=2).eval()
        x = torch.randn(3, 1, 2048)
        with torch.no_grad():
            y0, y2 = m0(x), m2(x)
        self.assertTrue(torch.equal(y0, y2), "trc=2 零初始化前向必须与 trc=0 逐位一致")

    def test_trc2_grads_flow(self):
        m2 = VGG16(ch_in=1, alpha=0.125, trc=2)
        m2(torch.randn(2, 1, 2048)).sum().backward()
        calib = [m for m in m2.model if isinstance(m, AffineChannel1D)][0]
        self.assertIsNotNone(calib.gamma.grad)
        self.assertGreater(float(calib.gamma.grad.abs().sum()), 0, "gamma 必须有非零梯度")
        self.assertGreater(float(calib.beta.grad.abs().sum()) + 1, 1.0)  # beta 梯度可为常数项, 仅验不炸

    def test_affine_differs_from_grn(self):
        x = make_input()
        g = torch.full((1, 64, 1), 0.5)
        b = torch.zeros(1, 64, 1)
        grn, aff = GRN1D(64), AffineChannel1D(64)
        with torch.no_grad():
            grn.gamma.copy_(g); aff.gamma.copy_(g)
            grn.beta.copy_(b); aff.beta.copy_(b)
        self.assertFalse(torch.allclose(grn(x), aff(x)),
                         "同 gamma 下 GRN(带 n 门控)与仿射必须不同")

    def test_grn_formula_regression(self):
        x = make_input(seed=5)
        m = GRN1D(64)
        with torch.no_grad():
            m.gamma.normal_(0, 0.3); m.beta.normal_(0, 0.3)
        n = x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        manual = x + m.gamma * (x * n) + m.beta
        self.assertTrue(torch.equal(m(x), manual), "GRN1D 公式回归: 手动计算逐位一致")

    def test_affine_formula_manual(self):
        x = make_input(seed=9)
        m = AffineChannel1D(64)
        with torch.no_grad():
            m.gamma.normal_(0, 0.3); m.beta.normal_(0, 0.3)
        manual = x + m.gamma * x + m.beta
        self.assertTrue(torch.equal(m(x), manual), "仿射=逐位手动公式")

    def test_trc_invalid_rejected(self):
        with self.assertRaises(AssertionError):
            VGG16(ch_in=1, alpha=0.125, trc=3)

    def test_module_object_roundtrip_safeload(self):
        """run_pt 双格式保存(完整模块对象) -> load_torch_checkpoint(weights_only=True)。"""
        import tempfile
        from utils.checkpoint import load_torch_checkpoint
        m2 = VGG16(ch_in=1, alpha=0.125, trc=2).eval()
        x = torch.randn(2, 1, 2048)
        with torch.no_grad():
            y_ref = m2(x)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "encoder_group.pth"
            torch.save({"backbone_state_dict": [m2]}, p)  # run_pt §762 同款完整模块格式
            loaded = load_torch_checkpoint(p)
            m2b = loaded["backbone_state_dict"][0]
            self.assertIsInstance(m2b, VGG16)
            with torch.no_grad():
                y2 = m2b(x)
        self.assertTrue(torch.equal(y_ref, y2), "模块对象经受限加载后前向必须逐位一致")


if __name__ == "__main__":
    unittest.main(verbosity=2)
