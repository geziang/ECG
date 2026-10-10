# -*- coding: utf-8 -*-
"""W8 E3 (w8-tsr) 单测: TSR 官方四变体 pretext 适配 (BobZwr/ReverseECG)。

覆盖:
  1. build_variant 与官方 generate_reverse 前 3 块逐位一致(原图/幅值反转/时间反转);
  2. k=3 双重反转=官方标签意图(记录官方 snippet [::-1] 轴位笔误, 按标签语义实现);
  3. y2 槽位映射 [时间反转?,幅值反转?] 与官方块顺序一致;
  4. TSReverseDataset 合成目录: 形状/标签合法;
  5. LeadFusionBT(loss_mode='tsr') forward: 损失有限、与手工 BCE 一致、梯度达
     tsr_heads 与 backbone;
  6. 默认关: loss-mode 默认 bt; --wd 默认 0.0 且 Adam 透传(历史行为不变)。
"""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_utils.tsr_dataset import build_variant, TSReverseDataset


def official_generate_reverse(oridata):
    """官方 pretraining.py generate_reverse 原样抄录(对照基准)。"""
    mu = np.mean(oridata, -1, keepdims=True)
    spatial_reverse = -1 * (oridata - mu) + mu
    temporal_reverse = oridata[:, ::-1]
    ts_reverse = spatial_reverse[::-1]  # 官方原样(记录轴, 笔误)
    x = np.concatenate((oridata, spatial_reverse, temporal_reverse, ts_reverse))
    n = oridata.shape[0]
    y = np.array([0, 0] * n + [0, 1] * n + [1, 0] * n + [1, 1] * n)
    return x, y


class TestW8E3(unittest.TestCase):

    def _x(self, seed=7):
        rng = np.random.RandomState(seed)
        return (rng.randn(8, 2048) * 2 + 1).astype(np.float32)

    def test_blocks_match_official(self):
        x = self._x()
        # 官方代码按 [N, length] 单导联 2D 处理 -> 逐导联喂官方函数再对拍
        # (时间反转=末轴翻转的 8 导联推广, 见 tsr_dataset 文档)
        for k in (0, 1, 2):
            xv, y2 = build_variant(x, k)
            for j in range(8):
                ox, oy = official_generate_reverse(x[j][None])
                self.assertTrue(np.array_equal(xv[j], ox[k]),
                                f"k={k} lead{j} 变体须与官方逐位一致")
                self.assertEqual(list(y2), [int(v) for v in oy[2 * k:2 * k + 2]],
                                 f"k={k} 标签须与官方一致")

    def test_ts_block_follows_label_intent(self):
        x = self._x()
        mu = x.mean(axis=-1, keepdims=True)
        intended = (-(x - mu) + mu)[:, ::-1]
        xv, y2 = build_variant(x, 3)
        self.assertTrue(np.array_equal(xv, np.ascontiguousarray(intended)),
                        "k=3 按标签意图=双重反转(官方 snippet [::-1] 反记录轴系笔误)")
        self.assertEqual(list(y2), [1.0, 1.0])

    def test_label_slots(self):
        for k, want in {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}.items():
            _, y2 = build_variant(self._x(seed=k), k)
            self.assertEqual(list(y2), want)

    def test_dataset_synthetic(self):
        td = tempfile.mkdtemp()
        try:
            for i in range(3):
                np.save(Path(td) / f"r{i}.npy", self._x(seed=i))
            ds = TSReverseDataset(td)
            self.assertEqual(len(ds), 3)
            np.random.seed(0)
            views, y2 = ds[0]
            self.assertEqual(views[0].shape, (8, 2048))
            self.assertEqual(tuple(y2.shape), (2,))
            self.assertIn(tuple(y2.tolist()), [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)])
            self.assertTrue(torch.equal(views[0], views[1]), "双视图槽为同一变体")
        finally:
            shutil.rmtree(td)

    def test_tsr_forward_and_grads(self):
        from argparse import Namespace
        from run_pt import LeadFusionBT
        d = dict(num_leads=8, batch_size=4, gamma=0.8, lambd=0.0051,
                 projector='64-64', loss_mode='tsr',
                 vicreg_sim=25.0, vicreg_var=25.0, vicreg_cov=1.0,
                 vicreg_var_eps=1e-4, vicreg_keep_bn=False,
                 projector_norm='batchnorm', lcm_weight=0.0,
                 dualview_weight=0.0, rr_weight=0.0)
        torch.manual_seed(0)
        m = LeadFusionBT(Namespace(**d))
        m.device = 'cpu'
        for g in m.backbone_group + m.projector_group + m.bn_group:
            g.to('cpu')
        m.tsr_heads.to('cpu')
        x = torch.randn(4, 8, 512)
        tgt = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        loss, lr_, lt = m.forward(x, tgt)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.equal(loss, lr_))
        self.assertTrue(torch.equal(lt, torch.zeros_like(lt)))
        # 手工 BCE 对拍(逐导联头均值)
        import torch.nn.functional as F
        _, f_list, _ = m._embed_full(x)
        manual = sum(F.binary_cross_entropy_with_logits(m.tsr_heads[i](f_list[i]), tgt)
                     for i in range(8)) / 8
        self.assertTrue(torch.allclose(loss, manual, atol=1e-7))
        loss.backward()
        self.assertGreater(float(m.tsr_heads[0].weight.grad.abs().sum()), 0)
        bb_grad = sum(float(p.grad.abs().sum()) for p in m.backbone_group[0].parameters()
                      if p.grad is not None)
        self.assertGreater(bb_grad, 0, "梯度须达 backbone")

    def test_default_off_source_guards(self):
        src = (ROOT / "run_pt.py").read_text(encoding="utf-8")
        self.assertIn("'--loss-mode', default='bt'", src, "loss-mode 默认必须 bt")
        self.assertIn("'--wd', default=0.0", src, "wd 默认必须 0(历史行为)")
        self.assertIn("weight_decay=float(getattr(args, 'wd', 0.0) or 0.0)", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
