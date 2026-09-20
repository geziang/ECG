"""T1 D1L-fix 单测 (任务书 §七-T1 完成条件)。

运行: python tests/test_d1l_fix.py   (在 ECG_SSL_LFBT-main 下)
覆盖:
  1. 目标矩阵契约: 对称 / diag=1 / PSD(eig >= -1e-6), '0.5,0.2' 生理拓扑保真
  2. 闭合基线: d1l='1,1' (全1矩阵) 的 loss 与 d1l 关闭(B0) 逐位一致
  3. NEG 确定性: d1l-shuffle 两次构建矩阵完全相同
  4. 关态: d1l='' 时 d1l_P is None, 走 B0 原路径
  5. 语义差: 生理目标 (0.5,0.2) 的 loss != 闭合基线 (τ 确实进了对角目标)
"""
import argparse
import sys
import torch

sys.path.insert(0, ".")
import run_pt  # noqa: E402


def make_model(d1l="", shuffle=False, full=False):
    args = argparse.Namespace(
        num_leads=8, projector="2048-2048-2048", fast_backbone=True,
        blur_pool=0, pool_power=0.0, d7_weight=0.0, sinc_frontend=0,
        whiten=0, whiten_ln=False, whiten_shuffle=False, loss_mode="bt",
        gamma=0.8, lambd=0.0051, bt_var_hinge=0.0, h3_weight=0.0,
        mixup_prob=0.0, hrv_weight=0.0, d1l=d1l, d1l_shuffle=shuffle,
        d1l_full=full, batch_size=128)
    return run_pt.LeadFusionBT(args)


def main():
    # ---- 1. 矩阵契约 ----
    m = make_model("0.5,0.2")
    P = m.d1l_P.cpu()
    assert torch.allclose(P, P.T, atol=1e-6), "P 不对称"
    assert torch.allclose(torch.diagonal(P), torch.ones(8), atol=1e-6), "diag != 1"
    assert torch.linalg.eigvalsh(P).min() >= -1e-6, "P 非 PSD"
    assert abs(P[0, 1].item() - 0.5) < 1e-6, f"II-III tau={P[0,1].item()} != 0.5"
    assert abs(P[2, 3].item() - 0.2) < 1e-6, f"V1-V2 tau={P[2,3].item()} != 0.2"
    assert abs(P[0, 2].item()) < 1e-6, "非相邻对应为 0"
    print("[1] 矩阵契约 OK: 对称/diag=1/PSD/II-III=0.5/V1-V2=0.2")

    # ---- 2. 闭合基线: --d1l-full 全1矩阵 loss == B0 关态 loss 逐位一致 ----
    m11 = make_model(full=True)
    assert torch.allclose(m11.d1l_P.cpu(), torch.ones(8, 8), atol=1e-6), "full 矩阵应为全1"
    assert torch.linalg.eigvalsh(m11.d1l_P.cpu()).min() >= -1e-6, "全1矩阵非 PSD"
    torch.manual_seed(0)
    y1 = torch.randn(16, 8, 1000, device="cuda")
    y2 = torch.randn(16, 8, 1000, device="cuda")
    m11.bn_all.eval()  # BN eval: 同输入同输出, 消除 running-stat 更新干扰
    loss_closure, _, _ = m11.forward(y1, y2)
    m11.d1l_P = None  # 同一模型同一权重同一批次, 只关掉 P
    loss_b0, _, _ = m11.forward(y1, y2)
    assert torch.equal(loss_closure, loss_b0), \
        f"闭合基线不一致: {loss_closure.item()} vs {loss_b0.item()}"
    print(f"[2] 闭合基线 OK: --d1l-full loss == B0 loss 逐位一致 ({loss_b0.item():.6f})")

    # ---- 3. NEG 确定性 ----
    ma, mb = make_model("0.5,0.2", shuffle=True), make_model("0.5,0.2", shuffle=True)
    assert torch.equal(ma.d1l_P, mb.d1l_P), "d1l-shuffle 两次构建不一致"
    mc = make_model("0.5,0.2")
    assert not torch.equal(ma.d1l_P, mc.d1l_P), "shuffle 后应不同于真实分组"
    print("[3] NEG 确定性 OK: 同 seed 置换可复现, 且 != 真实分组")

    # ---- 4. 关态 ----
    m0 = make_model("")
    assert m0.d1l_P is None and m0.d1l_meta is None
    print("[4] 关态 OK: d1l='' -> P=None (B0 原路径)")

    # ---- 5. 语义差 ----
    m11b = make_model(full=True)
    m11b.bn_all.eval()
    loss_tau, _, _ = m11b.forward(y1, y2)
    m11b.d1l_P = make_model("0.5,0.2").d1l_P  # 换生理目标
    loss_physio, _, _ = m11b.forward(y1, y2)
    assert not torch.equal(loss_tau, loss_physio), "生理目标应改变跨导联对角项 loss"
    print(f"[5] 语义差 OK: tau(1,1)={loss_tau.item():.6f} != tau(0.5,0.2)={loss_physio.item():.6f}")

    print("ALL_D1L_FIX_TESTS_PASSED")


if __name__ == "__main__":
    main()
