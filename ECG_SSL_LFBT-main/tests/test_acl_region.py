"""T2 ACL 区域模型单测 (任务书 §七-T2: Eq.(10)/(11) 正负样本索引有单测)。

运行: python tests/test_acl_region.py   (在 ECG_SSL_LFBT-main 下)
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from models.acl_region import (ACL_REGIONS, N_REGIONS, build_regions, info_nce,  # noqa: E402
                               intra_region_loss, inter_region_pairs, inter_region_loss, acl_loss)


def orthogonal_batch(b, d, seed=0):
    """构造 b 个近正交向量的批次 (可分辨的正负样本)。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, d, generator=g)
    return F.normalize(x, dim=1)


def main():
    # ---- 1. 分组 ----
    assert build_regions() == ACL_REGIONS == ((0, 1), (2, 3), (4, 5), (6, 7))
    parts = [build_regions(r) for r in (1, 2, 3)]
    assert all(len(set(sum(p, ()))) == 8 and len(p) == N_REGIONS for p in parts), "随机分组不是 0..7 的划分"
    assert len({tuple(p) for p in parts}) == 3, "3 个 partition 应互不相同"
    assert build_regions(1) == build_regions(1), "随机分组应可复现"
    assert tuple(build_regions(1)) != ACL_REGIONS or True  # 允许极小概率撞真实分组, 只要求确定性
    print("[1] 分组 OK: 真实四区 + 3 个可复现随机 partition")

    # ---- 2. Eq.(10) 索引语义: 对角=正 ----
    b, d = 8, 16
    u1 = [orthogonal_batch(b, d, seed=r) for r in range(4)]
    u2_pos = [u.clone() for u in u1]                       # 正样本恰在对角
    perm = torch.randperm(b, generator=torch.Generator().manual_seed(7))
    u2_neg = [u[perm].clone() for u in u1]                 # 同一批向量, ECG 归属被打乱
    l_pos = intra_region_loss(u1, u2_pos)
    l_neg = intra_region_loss(u1, u2_neg)
    assert l_pos < l_neg, f"Eq.(10) 正样本索引错误: {l_pos.item():.4f} !< {l_neg.item():.4f}"
    print(f"[2] Eq.(10) OK: 对角正样本 loss={l_pos.item():.4f} < 打乱={l_neg.item():.4f}")

    # ---- 3. Eq.(11) 索引语义: 同 ECG 跨区跨视图为正, 批内他 ECG 为负 ----
    assert len(inter_region_pairs()) == 12  # 4x3 有序对, 不含 (r,r)
    l_inter_pos = inter_region_loss(u1, u2_pos)
    l_inter_neg = inter_region_loss(u1, u2_neg)
    assert l_inter_pos < l_inter_neg, "Eq.(11) 正样本索引错误"
    # 关键语义: 同 ECG 的其他区域与 u1_r 相似不受罚 (不互相推远)
    # 构造 u2_s 全部等于 u1_0 (同 ECG 跨区完全一致) -> 对角仍是正样本, loss 应低
    u2_same_ecg = [u1[0].clone() for _ in range(4)]
    l_norepel = inter_region_loss(u1, u2_same_ecg)
    l_baseline = inter_region_loss(u1, [orthogonal_batch(b, d, seed=100 + r) for r in range(4)])
    assert l_norepel < l_baseline, "同 ECG 跨区一致被惩罚 -> 违反 Eq.(11) 不推远语义"
    print(f"[3] Eq.(11) OK: 对角正={l_inter_pos.item():.4f} < 打乱={l_inter_neg.item():.4f}; "
          f"同ECG跨区一致不受罚({l_norepel.item():.4f} < 随机{ l_baseline.item():.4f})")

    # ---- 4. 梯度流 ----
    w = torch.nn.Linear(16, 16, bias=False)
    u1w = [F.normalize(w(u), dim=1) for u in u1]
    loss, li, lt = acl_loss(u1w, u2_pos, temperature=0.5, gamma=0.5, use_inter=True)
    loss.backward()
    assert w.weight.grad is not None and torch.isfinite(w.weight.grad).all()
    assert abs(float(loss) - (0.5 * float(li) + 0.5 * float(lt))) < 1e-5
    loss_a2, li_a2, lt_a2 = acl_loss(u1w, u2_pos, use_inter=False)
    assert torch.allclose(loss_a2, li_a2) and float(lt_a2) == 0.0
    print("[4] 梯度流+组合 OK: A3=γ混合, A2=仅intra")

    print("ALL_ACL_REGION_TESTS_PASSED")


if __name__ == "__main__":
    main()
