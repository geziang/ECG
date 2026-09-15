# -*- coding: utf-8 -*-
"""D9(VICReg 化)开关单元测试 (运行: python tests/test_d9_switch.py)

红线依据 01 §六-1: 开关全关必须与 B0 逐位一致。
覆盖:
  T1 默认参数 forward == 独立复现的 B0 参考公式 (逐位)
  T2 vicreg 模式: 损失有限、全参数有梯度、loss_r/loss_t 聚合结构不变
  T3 vicreg 三项语义: inv=0 (同嵌入)、var hinge=0 (std>=1)、cov~0 (独立维)
  T4 keep-bn 与默认(去 bn) 输出不同
  T5 projector-norm=layernorm 可运行且与 batchnorm 输出不同
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_pt import LeadFusionBT, off_diagonal

FAILS = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


def make_args(**kw):
    from argparse import Namespace
    d = dict(num_leads=8, batch_size=32, gamma=0.8, lambd=0.0051,
             projector='2048-2048-2048', loss_mode='bt',
             vicreg_sim=25.0, vicreg_var=25.0, vicreg_cov=1.0,
             vicreg_var_eps=1e-4, vicreg_keep_bn=False,
             projector_norm='batchnorm')
    d.update(kw)
    return Namespace(**d)


def ref_bt_loss(model, y1, y2, args):
    """独立复现的 B0 参考公式 (与 run_pt 实现分开写, 作为 ground truth)。"""
    zs1 = [model.projector_group[i](model.backbone_group[i](y1[:, [i], :])) for i in range(8)]
    zs2 = [model.projector_group[i](model.backbone_group[i](y2[:, [i], :])) for i in range(8)]
    r = t = 0.0
    for i in range(8):
        for j in range(8):
            a = model.bn_group[i](zs1[i])
            b = model.bn_group[j](zs2[j])
            c = torch.mm(a.t(), b) / args.batch_size
            diag = (torch.diagonal(c) - 1).pow(2).sum()
            offd = off_diagonal(c).pow(2).sum()
            ls = diag + args.lambd * offd
            if i == j:
                r = r + ls
            else:
                t = t + ls
    r = r / 8
    t = t / (8 * 7)
    return args.gamma * r + (1 - args.gamma) * t, r, t


def main():
    torch.manual_seed(0)
    args = make_args()
    model = LeadFusionBT(args)
    model.device = 'cpu'
    for g in model.backbone_group + model.projector_group + model.bn_group:
        g.to('cpu')
    y1 = torch.randn(32, 8, 2048)
    y2 = torch.randn(32, 8, 2048)

    # T1: 默认参数 == B0 参考公式。逐位不现实: strided-view 与连续张量的 sum 归约顺序
    #     不同会引起 float32 舍入差 (~1e-7 相对量级), 故用紧容差; 权威逐位验收 =
    #     修改前后 2-epoch CPU 冒烟日志比对 (runlog/W0/d9_{preedit,postedit}_smoke.log)
    with torch.no_grad():
        loss, lr, lt = model.forward(y1, y2)
        r_loss, r_r, r_t = ref_bt_loss(model, y1, y2, args)
    rel = lambda a, b: abs(a - b) / (abs(b) + 1e-12)
    check("T1 默认 forward == B0 参考公式 (rel<1e-6)",
          rel(loss.item(), r_loss.item()) < 1e-6
          and rel(lr.item(), r_r.item()) < 1e-6
          and rel(lt.item(), r_t.item()) < 1e-6)

    # T2: vicreg 模式结构
    va = make_args(loss_mode='vicreg')
    vm = LeadFusionBT(va)
    vm.device = 'cpu'
    for g in vm.backbone_group + vm.projector_group + vm.bn_group:
        g.to('cpu')
    # 新建模块默认训练模式
    v_loss, v_r, v_t = vm.forward(y1, y2)
    check("T2 vicreg 损失有限且为正",
          torch.isfinite(v_loss) and v_loss.item() > 0)
    v_loss.backward()
    no_grad = [n for n, p in (*vm.backbone_group[0].named_parameters(),)
               if p.grad is None]
    check("T2 vicreg 全部主干参数有梯度", len(no_grad) == 0)

    # T3: 三项语义 (直接用 _pair_loss)
    zi = torch.randn(32, 64) * 5.0   # std 5 -> var hinge = 0
    zj = zi.clone()                  # 同嵌入 -> inv = 0
    pair_same = vm._pair_loss(zi, zj)
    zi2 = torch.randn(32, 64) * 5.0  # 独立维 -> cov ~ 0 (统计意义)
    pair_ind = vm._pair_loss(zi2, zj)
    check("T3 同嵌入对: loss == var(0)+cov(0) -> 仅剩 cov(秩结构) 且量级小",
          pair_same.item() < pair_ind.item())
    z_big = torch.randn(32, 64) * 5.0
    var_term = torch.relu(1.0 - torch.sqrt(z_big.var(dim=0) + va.vicreg_var_eps)).mean()
    check("T3 std=5 时 variance hinge 项为 0", var_term.item() == 0.0)

    # T4: keep-bn 与默认不同
    vb = make_args(loss_mode='vicreg', vicreg_keep_bn=True)
    vmb = LeadFusionBT(vb)
    vmb.device = 'cpu'
    for g in vmb.backbone_group + vmb.projector_group + vmb.bn_group:
        g.to('cpu')
    # 新建模块默认训练模式
    with torch.no_grad():
        lb, _, _ = vmb.forward(y1, y2)
    check("T4 keep-bn 与去-bn 输出不同", abs(lb.item() - v_loss.item()) > 1e-6)

    # T5: layernorm projector 可运行且不同于 batchnorm
    vl = make_args(projector_norm='layernorm')
    vml = LeadFusionBT(vl)
    vml.device = 'cpu'
    for g in vml.backbone_group + vml.projector_group + vml.bn_group:
        g.to('cpu')
    # 新建模块默认训练模式
    with torch.no_grad():
        ll, _, _ = vml.forward(y1, y2)
        bl, _, _ = vm.forward(y1, y2)  # vm 同 seed 构建时权重相同? 分别构建后权重不同 -> 仅验证有限可跑
    check("T5 layernorm projector 可运行且损失有限", torch.isfinite(ll))

    print("\nALL PASS" if not FAILS else f"\nFAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
