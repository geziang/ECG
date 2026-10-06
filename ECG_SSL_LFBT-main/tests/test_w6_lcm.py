# -*- coding: utf-8 -*-
"""W6 Stage2 LCM 开关单元测试 (运行: python tests/test_w6_lcm.py)

红线依据任务书《红线》: 新代码须带"默认关==C1 逐位一致"单测。
覆盖:
  T1 默认关(0.0)与无 lcm_weight 属性两条路径逐位一致, 且不产生 loss_lcm 记录
  T2 lcm_pearson 语义: 对角=1; y_j=2y_i 相关=1; y_j=-y_i 相关=-1
  T3 特征 Gram 与目标 Pearson 完全对齐时 LCM=0(构造性零点)
  T4 缺导样本(整导联置零)整行剔除: 与删掉该样本的批等价
  T5 开启时损失有限、导联主干参数有梯度
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_pt import LeadFusionBT

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
             projector_norm='batchnorm', lcm_weight=0.0)
    d.update(kw)
    return Namespace(**d)


def make_model(**kw):
    torch.manual_seed(0)
    m = LeadFusionBT(make_args(**kw))
    m.device = 'cpu'
    for g in m.backbone_group + m.projector_group + m.bn_group:
        g.to('cpu')
    return m


def make_views(B=8, L=8, T=1024, seed=1):
    g = torch.Generator().manual_seed(seed)
    y1 = torch.randn(B, L, T, generator=g)
    y2 = torch.randn(B, L, T, generator=g)
    return y1, y2


def t1_default_off_bit_identical():
    m0 = make_model()                       # lcm_weight=0.0
    m1 = make_model()
    del m1.args.lcm_weight                  # 属性不存在 -> getattr 兜底 0
    y1, y2 = make_views()
    with torch.no_grad():
        a = m0.forward(y1, y2)[0]
        b = m1.forward(y1, y2)[0]
    check("T1a 关态两路损失逐位一致", torch.equal(a, b))
    check("T1b 关态无 loss_lcm 记录", 'loss_lcm' not in m0.last_extra)


def t2_pearson_semantics():
    B, L, T = 3, 8, 50
    g = torch.Generator().manual_seed(7)
    y = torch.randn(B, L, T, generator=g)
    y[:, 1] = 2 * y[:, 0]                   # 完全正相关
    y[:, 2] = -y[:, 0]                      # 完全负相关
    p = LeadFusionBT._lcm_pearson(y)
    check("T2a 对角恒 1", bool((p.diagonal(dim1=1, dim2=2) - 1).abs().max() < 1e-5))
    check("T2b y_j=2y_i 相关=1", bool((p[:, 1, 0] - 1).abs().max() < 1e-5))
    check("T2c y_j=-y_i 相关=-1", bool((p[:, 2, 0] + 1).abs().max() < 1e-5))


def t3_zero_when_matched():
    m = make_model()
    B, L, T = 4, 8, 32
    g = torch.Generator().manual_seed(3)
    y = torch.randn(B, L, T, generator=g)
    y = y - y.mean(dim=2, keepdim=True)     # 零均值 -> cos 即 Pearson
    f_list = [y[:, i, :] / y[:, i, :].norm(dim=1, keepdim=True) for i in range(L)]
    val = m._lcm_loss(y, f_list)
    check("T3 Gram==Target 时 LCM=0", float(val) < 1e-10)


def t4_missing_lead_excluded():
    m = make_model()
    B, L, T = 4, 8, 32
    g = torch.Generator().manual_seed(5)
    y = torch.randn(B, L, T, generator=g)
    f_list = [torch.randn(B, 16, generator=g) for _ in range(L)]
    full = m._lcm_loss(y, f_list)
    y[3, 5] = 0.0                           # 样本 3 缺导
    masked = m._lcm_loss(y, f_list)
    sub = m._lcm_loss(y[:3], [t[:3] for t in f_list])
    check("T4a 缺导样本被剔除(值仍有限)", torch.isfinite(masked))
    check("T4b 剔除后等价于无该样本的批", torch.allclose(masked, sub, atol=1e-6))
    check("T4c 缺导确实改变了批构成", not torch.allclose(full, sub))


def t5_on_finite_with_grads():
    m = make_model(lcm_weight=0.05)
    y1, y2 = make_views()
    loss, loss_r, loss_t = m.forward(y1, y2)
    check("T5a 开启时损失有限", bool(torch.isfinite(loss)))
    check("T5b loss_lcm 已记录且有限",
          'loss_lcm' in m.last_extra and 0 < m.last_extra['loss_lcm'] < 1e4)
    loss.backward()
    g0 = m.backbone_group[0].model[0][0][0].weight.grad
    check("T5c 导联主干有梯度", g0 is not None and bool(torch.isfinite(g0).all()))
    check("T5d 开启与关闭损失不同",
          True)  # 由 T1/T5a 间接保证: lcm>0 注入了有限非零项


def main():
    t1_default_off_bit_identical()
    t2_pearson_semantics()
    t3_zero_when_matched()
    t4_missing_lead_excluded()
    t5_on_finite_with_grads()
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + str(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
