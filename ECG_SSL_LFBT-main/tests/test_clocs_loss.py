# -*- coding: utf-8 -*-
"""W4 A-5 单测: S2 CLOCS diag 双向损失(官方口径) 正确性。

运行: python tests/test_clocs_loss.py
- test_diag_denominator_includes_self: 正交基解析例 = 2·ln(1+e^{-1/τ}) —
  证明分母含正对自身(官方 obtain_contrastive_loss 口径; 与 SimCLR 排除自身相反)
- test_clocs_forward_matches_manual: clocs 分支与逐对手工重算逐位一致
- test_cli: choices 含 clocs, --clocs-temp 默认 0.1, 默认 loss-mode 仍 bt
"""
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_pt  # noqa: E402


def _small_model(num_leads=2, seed=0):
    argv_bak = sys.argv
    sys.argv = ["x"]
    args = run_pt.parser.parse_args(["--data-dir", "data/ptbxl"])
    sys.argv = argv_bak
    args.num_leads = num_leads
    args.projector = "16-16"
    args.batch_size = 8
    torch.manual_seed(seed)
    return run_pt.LeadFusionBT(args), args


def test_diag_denominator_includes_self():
    m, args = _small_model()
    args.clocs_temp = 0.1
    za = torch.eye(2)
    zb = torch.eye(2)  # 样本正交且两视图相同: cos 对角=1, 非对角=0
    got = m._clocs_diag_loss(za, zb).item()
    want = 2 * math.log(1 + math.exp(-1 / 0.1))  # 分母=行和(e^{10}+e^0) 含正对自身
    assert abs(got - want) < 1e-6, f"解析例不符: got={got} want={want}(若分母排除自身应为 20)"


def test_clocs_forward_matches_manual():
    m, args = _small_model(num_leads=2, seed=7)
    args.loss_mode = "clocs"
    args.clocs_temp = 0.1
    torch.manual_seed(11)
    y1 = torch.randn(8, 2, 2048).to(m.device)
    y2 = torch.randn(8, 2, 2048).to(m.device)
    loss, loss_r, loss_t = m.forward(y1, y2)

    def manual_pair(za, zb):
        a = za / za.norm(dim=1, keepdim=True)
        b = zb / zb.norm(dim=1, keepdim=True)
        e = torch.exp(a @ b.T / 0.1)
        d = torch.diagonal(e)
        return (-torch.mean(torch.log(d / e.sum(1)))
                - torch.mean(torch.log(d / e.sum(0))))

    with torch.no_grad():
        z1 = m._embed(y1)
        z2 = m._embed(y2)
        temporal = manual_pair(z1[0], z2[0]) + manual_pair(z1[1], z2[1])
        spatial = manual_pair(z1[0], z1[1]) + manual_pair(z2[0], z2[1])
        n_pairs = 2 + 2 * 1
        manual_loss = (temporal + spatial) / (2 * n_pairs)
    assert abs(loss.item() - manual_loss.item()) < 1e-6, \
        f"clocs 手工重算不符: forward={loss.item()} manual={manual_loss.item()}"
    assert abs(loss_r.item() - (temporal / 4).item()) < 1e-6
    assert abs(loss_t.item() - (spatial / 4).item()) < 1e-6  # 2 对 spatial × 2 方向


def test_cli():
    a = run_pt.parser.parse_args(["--data-dir", "data/ptbxl"])
    assert a.loss_mode == "bt" and a.clocs_temp == 0.1 and a.simclr_temp == 0.5
    b = run_pt.parser.parse_args(["--data-dir", "data/ptbxl", "--loss-mode", "clocs"])
    assert b.loss_mode == "clocs"


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
