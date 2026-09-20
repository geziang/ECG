"""acl_region.py — ACL-ECG 四解剖区域受控实现 (任务书 T2 / A2 / A3 / A4)。

机制 (对照 ACL-ECG Sensors 2026, 26(3):1080):
  - 固定四区: (II,III) / (V1,V2) / (V3,V4) / (V5,V6) (lead 序 0..7)
  - 区域输入 = 区内两导联编码器输出 h(64) concat -> 128, 过独立 region projector -> u_r
  - Eq.(10) 区域内 InfoNCE (loss_intra): 同区两视图为正, 批内其他 ECG 为负
  - Eq.(11) 跨区域 InfoNCE (loss_inter): 同一 ECG 的区域 d(视图1) 与区域 d'≠d(视图2)
    为正样本, 批内其他 ECG 为负样本 —— 注意它不把不同区域互相推远
  - A4 NEG: 随机四区分组 (>=3 个 partition), 证明收益来自解剖结构而非参数量

设计约束:
  - 默认关 (--acl-region 0) 时 run_pt 不经过本模块任何路径, B0 逐位一致
  - InfoNCE 用对称 NT-Xent, 温度可配; loss = gamma*intra + (1-gamma)*inter (A3 gamma=0.5)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 固定四区 (lead 序: ii,iii,v1..v6 = 0..7)
ACL_REGIONS = ((0, 1), (2, 3), (4, 5), (6, 7))  # (II,III) (V1,V2) (V3,V4) (V5,V6)
N_REGIONS = 4


def build_regions(random_partition=0):
    """返回区域分组元组。random_partition>0 时用固定种子生成第 N 个随机分组(A4 NEG)。"""
    if random_partition <= 0:
        return ACL_REGIONS
    leads = list(range(8))
    g = torch.Generator().manual_seed(1000 + random_partition)
    perm = torch.randperm(8, generator=g).tolist()
    leads = [leads[p] for p in perm]
    return tuple((leads[2 * r], leads[2 * r + 1]) for r in range(N_REGIONS))


class RegionProjector(nn.Module):
    """区域 projector: 128 (2x64 concat) -> 2048 -> 2048, 与主 projector 同容量级。"""

    def __init__(self, in_dim=128, sizes=(2048, 2048)):
        super().__init__()
        layers, d = [], in_dim
        for s in sizes:
            layers += [nn.Linear(d, s, bias=False), nn.BatchNorm1d(s), nn.ReLU(inplace=True)]
            d = s
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def region_embed(h_list, regions, projectors):
    """h_list: 每导联编码器输出 [(B,64)]*8 -> 每区 concat 后投影 + L2 归一化 [(B,d)]*4。"""
    u = []
    for r, (i, j) in enumerate(regions):
        x = torch.cat([h_list[i], h_list[j]], dim=1)
        u.append(F.normalize(projectors[r](x), dim=1))
    return u


def info_nce(z_a, z_b, temperature=0.5):
    """对称 NT-Xent (Eq.10 口径): 对角为正 (同 ECG 两视图), 批内其余为负。
    z_a/z_b: (B, d) 已 L2 归一化。返回标量 loss。"""
    b = z_a.size(0)
    logits = z_a @ z_b.T / temperature          # (B,B), logits[k,l] = sim(a_k, b_l)
    labels = torch.arange(b, device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def intra_region_loss(u1, u2, temperature=0.5):
    """Eq.(10): 每区两视图 InfoNCE, 区平均。u1/u2: [(B,d)]*4 (同区序)。"""
    return torch.stack([info_nce(u1[r], u2[r], temperature) for r in range(len(u1))]).mean()


def inter_region_pairs(num_regions=N_REGIONS):
    """Eq.(11) 的区域对枚举 (r, r'), r != r' (同 ECG 跨区正样本; 不含推远项)。"""
    return [(r, s) for r in range(num_regions) for s in range(num_regions) if r != s]


def inter_region_loss(u1, u2, temperature=0.5):
    """Eq.(11): 对每对 r!=r', InfoNCE(u1_r, u2_r') —— 同一 ECG 的跨区跨视图为正,
    批内其他 ECG 为负。对对平均。"""
    pairs = inter_region_pairs(len(u1))
    return torch.stack([info_nce(u1[r], u2[s], temperature) for r, s in pairs]).mean()


def acl_loss(u1, u2, temperature=0.5, gamma=0.5, use_inter=True):
    """总损失: A3 = gamma*intra + (1-gamma)*inter; A2 (use_inter=False) = 仅 intra。"""
    li = intra_region_loss(u1, u2, temperature)
    if not use_inter:
        return li, li, torch.zeros_like(li)
    lt = inter_region_loss(u1, u2, temperature)
    return gamma * li + (1.0 - gamma) * lt, li, lt
