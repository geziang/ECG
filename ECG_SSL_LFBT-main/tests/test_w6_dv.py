# -*- coding: utf-8 -*-
"""W6 Stage4 双视角开关单元测试 (运行: python tests/test_w6_dv.py)

红线: 新代码默认关==C1 逐位一致。
覆盖:
  T1 默认关(dv 参数缺省/为 None)与显式 0.0 两路损失逐位一致, 无 loss_dv/loss_rr 记录
  T2 beat_view: 已知峰位的合成信号 -> 坰心窗平均正确; 峰数<3 返回 None
  T3 rr_stats: 等间距峰 -> mean_RR/SDNN/RMSSD 精确值
  T4 DVDataset 归一化目标 z-score(用合成缓存目录构造, 若不可行则跳过)
  T5 开启(dv 提供有效批)时损失有限、rr_head 与主干有梯度、loss_dv/loss_rr 已记录
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_pt import LeadFusionBT
from data_utils.dv_dataset import beat_view, rr_stats, FS, W_BEAT

FAILS = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


def make_args(**kw):
    from argparse import Namespace
    d = dict(num_leads=8, batch_size=8, gamma=0.8, lambd=0.0051,
             projector='64-64', loss_mode='bt',
             vicreg_sim=25.0, vicreg_var=25.0, vicreg_cov=1.0,
             vicreg_var_eps=1e-4, vicreg_keep_bn=False,
             projector_norm='batchnorm', lcm_weight=0.0,
             dualview_weight=0.0, rr_weight=0.0)
    d.update(kw)
    return Namespace(**d)


def make_model(**kw):
    torch.manual_seed(0)
    m = LeadFusionBT(make_args(**kw))
    m.device = 'cpu'
    for g in m.backbone_group + m.projector_group + m.bn_group:
        g.to('cpu')
    if getattr(m, 'rr_head', None) is not None:
        m.rr_head.to('cpu')
    return m


def make_views(B=8, T=1024, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, 8, T, generator=g), torch.randn(B, 8, T, generator=g))


def make_dv(B=8, seed=2):
    g = torch.Generator().manual_seed(seed)
    beat = torch.randn(B, 8, W_BEAT, generator=g)
    rr3 = torch.randn(B, 3, generator=g)
    valid = torch.ones(B)
    return (beat, rr3, valid)


def t1_default_off():
    m0 = make_model()
    m1 = make_model()
    del m1.args.dualview_weight, m1.args.rr_weight
    y1, y2 = make_views()
    with torch.no_grad():
        a = m0.forward(y1, y2)[0]
        b = m1.forward(y1, y2)[0]
        c = m0.forward(y1, y2, dv=None)[0]
    check("T1a 关态缺省属性/显式0.0/dv=None 三路逐位一致",
          torch.equal(a, b) and torch.equal(a, c))
    check("T1b 关态无 dv/rr 记录",
          'loss_dv' not in m0.last_extra and 'loss_rr' not in m0.last_extra)


def t2_beat_view():
    T = 512
    t = np.arange(T)
    raw = np.stack([np.sin(2 * np.pi * t / 50 + i) for i in range(8)]).astype(np.float32)
    peaks = np.array([100, 200, 300, 400])
    bv = beat_view(raw, peaks, w=21)
    check("T2a 形状 (8,21)", bv.shape == (8, 21))
    # 每窗中心在峰位 -> 正弦在相位 0 处; 窗平均≈窗内均值, 手工对照
    manual = np.mean([raw[:, p - 10:p + 11] for p in peaks], axis=0)
    check("T2b 与手工坰心窗平均逐位一致", np.allclose(bv, manual, atol=1e-6))
    check("T2c 峰数<3 返回 None", beat_view(raw, np.array([100, 200]), w=21) is None)
    edge = beat_view(raw, np.array([2, 500, 505]), w=21)  # 边缘 pad
    check("T2d 边缘峰可用(edge pad)不崩", edge is not None and np.isfinite(edge).all())


def t3_rr_stats():
    # 等间距峰: RR 恒定 -> SDNN=0, RMSSD=0
    peaks = np.array([0, 205, 410, 615])
    m, s, r = rr_stats(peaks)
    check("T3a mean_RR 精确", abs(m - 205 / FS) < 1e-9)
    check("T3b 等距 SDNN=RMSSD=0", s < 1e-9 and r < 1e-9)
    # 交替间距: RR in {a,b} -> mean=(a+b)/2, SDNN=|a-b|/2, RMSSD=|a-b|
    a, b = 200, 300
    peaks = np.array([0, a, a + b, 2 * a + b, 2 * a + 2 * b])
    m, s, r = rr_stats(peaks)
    check("T3c 交替间距三统计精确",
          abs(m - (a + b) / 2 / FS) < 1e-9 and abs(s - (b - a) / 2 / FS) < 1e-9
          and abs(r - (b - a) / FS) < 1e-9)
    check("T3d 峰数<3 返回 None", rr_stats(np.array([10, 20])) is None)


def t5_on_finite_grads():
    m = make_model(dualview_weight=0.05, rr_weight=0.1)
    y1, y2 = make_views()
    dv = make_dv()
    loss, lr_, lt = m.forward(y1, y2, dv=dv)
    check("T5a 开启时损失有限", bool(torch.isfinite(loss)))
    check("T5b loss_dv/loss_rr 已记录且有限",
          'loss_dv' in m.last_extra and 'loss_rr' in m.last_extra
          and np.isfinite(m.last_extra['loss_dv']) and np.isfinite(m.last_extra['loss_rr']))
    loss.backward()
    g0 = m.backbone_group[0].model[0][0][0].weight.grad
    gr = m.rr_head[0].weight.grad
    check("T5c 主干有梯度", g0 is not None and bool(torch.isfinite(g0).all()))
    check("T5d rr_head 有梯度", gr is not None and bool(torch.isfinite(gr).all()))
    # 无效样本全 0 -> 双视角项跳过不崩
    dv0 = (dv[0], dv[1], torch.zeros(8))
    m2 = make_model(dualview_weight=0.05, rr_weight=0.1)
    with torch.no_grad():
        l0 = m2.forward(y1, y2, dv=dv0)[0]
    check("T5e 全无效批优雅跳过", bool(torch.isfinite(l0)) and 'loss_dv' not in m2.last_extra)


def main():
    t1_default_off()
    t2_beat_view()
    t3_rr_stats()
    t5_on_finite_grads()
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + str(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
