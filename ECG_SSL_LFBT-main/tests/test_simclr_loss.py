# -*- coding: utf-8 -*-
"""W4 A-4 单测: S1 SimCLR(NT-Xent) 损失正确性 + 默认 bt 路径不受实现影响。

运行: python tests/test_simclr_loss.py
- test_nt_xent_orthogonal_analytic: 正交基小例的解析值 ln(1+2e^{-1/τ·1}) 形核对
- test_nt_xent_matches_naive:      与逐对朴素双循环 InfoNCE 实现逐位一致
- test_bt_default_matches_manual:  默认 bt forward 与手工 BT 公式重算逐位一致
- test_loss_mode_cli_accepts_simclr: argparse 接受 simclr 且默认仍为 bt
"""
import math
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_pt  # noqa: E402


def _small_model(num_leads=2, seed=0):
    """小尺寸 LFBT 模型(结构同构, 维度缩小), 固定种子可复现。"""
    sys.argv = ["run_pt.py"]
    args = run_pt.parser.parse_args(["--data-dir", "data/ptbxl"])
    args.num_leads = num_leads
    args.projector = "16-16"
    args.batch_size = 8
    torch.manual_seed(seed)
    return run_pt.LeadFusionBT(args), args


def test_nt_xent_orthogonal_analytic():
    m, args = _small_model()
    args.simclr_temp = 0.5
    zi = torch.eye(2)
    zj = torch.eye(2)  # 两视图相同且样本正交: 解析 loss = ln(1 + 2*exp(-1/τ))
    got = m._nt_xent(zi, zj).item()
    want = math.log(1 + 2 * math.exp(-1 / 0.5))
    assert abs(got - want) < 1e-6, f"解析例不符: got={got} want={want}"


def test_nt_xent_matches_naive():
    """朴素实现: 逐视图 i, 对全部 k≠i 求 -log softmax(sim(i, pos(i)))。"""
    torch.manual_seed(3)
    m, args = _small_model()
    args.simclr_temp = 0.3
    N, d = 5, 7
    zi, zj = torch.randn(N, d), torch.randn(N, d)
    got = m._nt_xent(zi, zj)
    z = torch.cat([zi, zj])
    z = z / z.norm(dim=1, keepdim=True)
    total = 0.0
    for i in range(2 * N):
        pos = (i + N) % (2 * N)
        sims = [(z[i] @ z[k]).item() / 0.3 for k in range(2 * N) if k != i]
        ps = [math.exp(s) for s in sims]
        total = total - math.log(ps[[k for k in range(2 * N) if k != i].index(pos)]
                                 / sum(ps))
    want = total / (2 * N)
    assert abs(got.item() - want) < 1e-5, f"朴素对照不符: got={got.item()} want={want}"


def test_bt_default_matches_manual():
    """默认 bt: forward 主路径与手工 BT 公式(重算相关矩阵)逐位一致。"""
    m, args = _small_model(num_leads=2, seed=7)
    # bn_group 训练态批统计(affine=False), forward 与手工重算同态
    torch.manual_seed(11)
    y1 = torch.randn(8, 2, 2048).to(m.device)
    y2 = torch.randn(8, 2, 2048).to(m.device)
    loss, loss_r, loss_t = m.forward(y1, y2)
    # 手工重算: 用模型自己的 embed + bn_group(训练态批统计), 聚合按 forward 公式
    with torch.no_grad():
        z1 = m._embed(y1)
        z2 = m._embed(y2)
        lr_acc, lt_acc = 0.0, 0.0
        for i in range(2):
            for j in range(2):
                c1 = m.bn_group[i](z1[i])
                c2 = m.bn_group[j](z2[j])
                c = c1.T @ c2
                c = c / args.batch_size
                on = (torch.diagonal(c) - 1).pow(2).sum()
                off = run_pt.off_diagonal(c).pow(2).sum()
                ls = on + args.lambd * off
                if i == j:
                    lr_acc = lr_acc + ls
                else:
                    lt_acc = lt_acc + ls
        lr_acc = lr_acc / 2
        lt_acc = lt_acc / 2
        manual = args.gamma * lr_acc + (1 - args.gamma) * lt_acc
    assert abs(loss.item() - manual.item()) < 1e-6, \
        f"bt 手工重算不符: forward={loss.item()} manual={manual.item()}"
    assert abs(loss_r.item() - lr_acc.item()) < 1e-6
    assert abs(loss_t.item() - lt_acc.item()) < 1e-6


def test_loss_mode_cli_accepts_simclr():
    sys.argv = ["run_pt.py"]
    a = run_pt.parser.parse_args(["--data-dir", "data/ptbxl"])
    assert a.loss_mode == "bt", "默认必须仍为 bt(B0 路径)"
    b = run_pt.parser.parse_args(["--data-dir", "data/ptbxl", "--loss-mode", "simclr"])
    assert b.loss_mode == "simclr" and b.simclr_temp == 0.5


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {type(e).__name__} {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
