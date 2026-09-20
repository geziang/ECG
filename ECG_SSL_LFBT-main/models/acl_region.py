# -*- coding: utf-8 -*-
"""acl_region.py — T2: ACL-ECG 式解剖区域关系目标 (任务书 07 §三 / 06 §三)。

机制(对齐 ACL-ECG, Liu-Wu-Yuan 2026):
  1. 四区域固定分组(lead 序 ii,iii,v1..v6 = 0..7):
       下壁 (II,III)=(0,1) / 间隔 (V1,V2)=(2,3) / 前壁 (V3,V4)=(4,5) / 侧壁 (V5,V6)=(6,7)
  2. 区域表示: 组内两条导联的 64 维 GAP 特征 concat(128) -> 独立区域 projector(2048)。
     不做三组均值(与已判负的 LGA 硬拉近切割)。
  3. L_intra (Eq.10 区域内一致): 同区域、同记录、跨视图为正; 批内其他记录(两视图)为负。
  4. L_inter (Eq.11 区域间判别): d!=d' 时, 同一记录的区域 d(视图1) 与区域 d'(视图2) 仍是正
     样本; 负样本只来自批内其他记录 —— 同一记录的其它区域特征绝不入负样本池,
     即"不是把不同区域互相推远"(任务书 §3.1 红线)。
  5. 随机分组 NEG (A4): --acl-partition random + --acl-rand-seed, ≥3 个 partition。

默认关闭(不实例化)时对 B0 主路径零影响。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 解剖四区(固定, 任务书 §3.2)
REGIONS_ANATOMY = ((0, 1), (2, 3), (4, 5), (6, 7))
LEAD_NAMES = ("II", "III", "V1", "V2", "V3", "V4", "V5", "V6")


def build_partition(kind="anatomy", rand_seed=101, num_leads=8):
    """返回 4 个二元组分区。random: 固定种子打乱 8 导联再两两成组(组内排序保确定性)。"""
    if kind == "anatomy":
        return list(REGIONS_ANATOMY)
    if kind == "random":
        g = torch.Generator().manual_seed(int(rand_seed))
        perm = torch.randperm(num_leads, generator=g).tolist()
        parts = [tuple(sorted(perm[k * 2:k * 2 + 2])) for k in range(num_leads // 2)]
        return parts
    raise ValueError(f"未知 acl-partition: {kind}")


class RegionProjectors(nn.Module):
    """每区域一个 MLP projector(结构镜像导联 projector: Linear-BN-ReLU 堆叠)。"""

    def __init__(self, regions, feat_per_lead=64, spec="128-2048-2048-2048"):
        super().__init__()
        self.regions = [tuple(r) for r in regions]
        sizes = list(map(int, spec.split('-')))
        assert sizes[0] == feat_per_lead * 2, \
            f"区域 projector 输入维 {sizes[0]} != 2×{feat_per_lead}(两导联 concat)"
        self.nets = nn.ModuleList()
        for _ in self.regions:
            layers = []
            for j in range(len(sizes) - 2):
                layers.append(nn.Linear(sizes[j], sizes[j + 1], bias=False))
                layers.append(nn.BatchNorm1d(sizes[j + 1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
            self.nets.append(nn.Sequential(*layers))

    def forward(self, lead_feats):
        """lead_feats: list[(B, 64) × 8] -> list[(B, out) × 4](区域 projector 原始输出)。"""
        outs = []
        for r, net in zip(self.regions, self.nets):
            x = torch.cat([lead_feats[r[0]], lead_feats[r[1]]], dim=1)
            outs.append(net(x))
        return outs


def nt_xent_pair(z_a, z_b, tau=0.5):
    """标准 NT-Xent(两视图对齐形态)。z_a/z_b: (B, d) 已按样本对齐, 内部 L2 归一。

    正样本: z_a[i] <-> z_b[i](跨视图同记录); 负样本: 批内其他记录的 z_a/z_b(2B-2 个)。
    """
    B = z_a.shape[0]
    za = F.normalize(z_a, dim=1)
    zb = F.normalize(z_b, dim=1)
    z = torch.cat([za, zb], dim=0)                      # (2B, d)
    sim = z @ z.t() / tau                               # (2B, 2B)
    self_mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(self_mask, float('-inf'))          # 排除自身
    pos = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)], dim=0).to(z.device)
    # anchor i 的正样本 = 跨视图同记录; 对角 -inf 后每行剩 2B-1 个有限值
    return F.cross_entropy(sim, pos)


def acl_losses(z1_regions, z2_regions, tau=0.5):
    """返回 (L_intra, L_inter)。

    L_intra: 同区域跨视图 NT-Xent, 4 区域求平均。
    L_inter: d!=d' 有序对 (z1_d, z2_d') NT-Xent, 12 对求平均。
      负样本池只含 (z1_d ∪ z2_d') 的批内其他记录 —— 同一记录的 z2_d / z1_d' 不在池中,
      因此不同区域永不被推远(等价实现 ACL Eq.(11) 的 d!=d' 正样本语义)。
    """
    R = len(z1_regions)
    l_intra = 0
    for d in range(R):
        l_intra = l_intra + nt_xent_pair(z1_regions[d], z2_regions[d], tau)
    l_intra = l_intra / R
    l_inter = 0
    n_pair = 0
    for d in range(R):
        for dp in range(R):
            if d == dp:
                continue
            l_inter = l_inter + nt_xent_pair(z1_regions[d], z2_regions[dp], tau)
            n_pair += 1
    l_inter = l_inter / n_pair
    return l_intra, l_inter
