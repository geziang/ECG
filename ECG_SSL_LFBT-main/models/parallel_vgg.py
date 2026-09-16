# -*- coding: utf-8 -*-
"""parallel_vgg.py — M0 提速基建: 8 导联独立 VGG16 合并为 groups=num_leads 的单模型。

数学等价性:
  - 分组卷积: weight (L*Cout, Cin, K), 第 g 组 = 导联 g 的原卷积权重 (cat dim=0)
  - BatchNorm1d(L*C): 逐通道统计 == 每导联独立 BN 逐通道统计
  - MaxPool / AdaptiveAvgPool: 逐通道, 天然等价
  输入 (B, L, T) [L=num_leads], 输出 (B, L*feat_dim), view 成 (B, L, feat) 即每导联 64 维表征。

用途: run_pt --fast-backbone 开关(粗筛提速); 与逐导联循环版数值等价(容差 1e-4 级,
归约顺序差异)。B0 锚点与配对基准仍用原路径, fast 路径仅用于 M 矩阵候选粗探。
"""
import torch
import torch.nn as nn

from models.vgg_1d import VGG16


def _group_block(ch_in, ch_out, groups):
    return nn.Sequential(
        nn.Conv1d(ch_in * groups, ch_out * groups, kernel_size=3, padding=1,
                  groups=groups),
        nn.BatchNorm1d(ch_out * groups),
        nn.ReLU(),
    )


class ParallelVGG16(nn.Module):
    """L 个独立 VGG16(alpha 缩放) 的分组卷积合并版, 参数量与 L 个独立版完全一致。"""

    def __init__(self, num_leads=8, ch_in=1, alpha=0.125):
        super().__init__()
        g = num_leads
        c = lambda x: int(x * alpha)
        self.num_leads = g
        self.alpha = alpha
        self.feat_dim = c(512)
        blocks = [
            (ch_in, c(64)), (c(64), c(64)),
            (c(64), c(128)), (c(128), c(128)),
            (c(128), c(256)), (c(256), c(256)), (c(256), c(256)),
            (c(256), c(512)), (c(512), c(512)), (c(512), c(512)),
            (c(512), c(512)), (c(512), c(512)), (c(512), c(512)),
        ]
        layers = []
        conv_i = 0
        for ci, co in blocks:
            layers.append(_group_block(ci, co, g))
            conv_i += 1
            # 与 VGG16 相同的池化节奏: 每 2/2/3/3/3 个卷积后一个 maxpool
            if conv_i in (2, 4, 7, 10, 13):
                layers.append(nn.MaxPool1d(2, 2))
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, num_leads, T)
        assert x.dim() == 3 and x.shape[1] == self.num_leads
        return self.model(x).squeeze(-1)  # (B, num_leads * feat_dim)

    def conv_bns(self):
        """按顺序返回 [(conv, bn), ...](modules() 深度优先序与构建序一致)。"""
        convs = [m for m in self.model.modules() if isinstance(m, nn.Conv1d)]
        bns = [m for m in self.model.modules() if isinstance(m, nn.BatchNorm1d)]
        return list(zip(convs, bns))


def _vgg_conv_bns(vgg):
    convs = [m for m in vgg.model.modules() if isinstance(m, nn.Conv1d)]
    bns = [m for m in vgg.model.modules() if isinstance(m, nn.BatchNorm1d)]
    return list(zip(convs, bns))


def build_parallel_from_list(vgg_modules):
    """从 L 个 VGG16 模块构建数值等价的 ParallelVGG16。"""
    L = len(vgg_modules)
    alpha = vgg_modules[0].alpha
    p = ParallelVGG16(num_leads=L, alpha=alpha)
    refs = [c.weight for c, _ in _vgg_conv_bns(vgg_modules[0])]
    pcs = p.conv_bns()
    assert len(refs) == len(pcs), "层数不一致"
    with torch.no_grad():
        for pi, (pc, pb) in enumerate(pcs):
            per_lead = [_vgg_conv_bns(v)[pi] for v in vgg_modules]
            pc.weight.copy_(torch.cat([c.weight for c, _ in per_lead], dim=0))
            if pc.bias is not None:
                pc.bias.copy_(torch.cat([c.bias for c, _ in per_lead], dim=0))
            pb.weight.copy_(torch.cat([b.weight for _, b in per_lead], dim=0))
            pb.bias.copy_(torch.cat([b.bias for _, b in per_lead], dim=0))
            pb.running_mean.copy_(torch.cat([b.running_mean for _, b in per_lead], dim=0))
            pb.running_var.copy_(torch.cat([b.running_var for _, b in per_lead], dim=0))
    return p.to(next(vgg_modules[0].parameters()).device)


def export_parallel_to_state_dicts(p):
    """ParallelVGG16 -> L 个 VGG16 state_dict 列表(兼容 run_lp 的 checkpoint 加载)。"""
    L, alpha = p.num_leads, p.alpha
    outs = []
    pcs = p.conv_bns()
    for l in range(L):
        vgg = VGG16(ch_in=1, n_classes=1000, alpha=alpha).to(
            next(p.parameters()).device)
        vgg.fc = torch.nn.Identity()  # 与 run_pt 原版一致: 无 fc 键, LP 侧 strict=False 加载
        vcs = _vgg_conv_bns(vgg)
        with torch.no_grad():
            for (pc, pb), (vc, vb) in zip(pcs, vcs):
                cout_g = pc.weight.shape[0] // L
                vc.weight.copy_(pc.weight.view(L, cout_g, *pc.weight.shape[1:])[l])
                if pc.bias is not None:
                    vc.bias.copy_(pc.bias.view(L, cout_g)[l])
                vb.weight.copy_(pb.weight.view(L, cout_g)[l])
                vb.bias.copy_(pb.bias.view(L, cout_g)[l])
                vb.running_mean.copy_(pb.running_mean.view(L, cout_g)[l])
                vb.running_var.copy_(pb.running_var.view(L, cout_g)[l])
        outs.append(vgg.state_dict())
    return outs


class ParallelProjector(nn.Module):
    """8 导联投影头合并版: 参数张量 (L, out, in), 前向单次 bmm。

    对应原版 projector_group: 每导联 Linear(bias=False)-BN-ReLU ×2 + Linear(bias=False)。
    输入 feat (B, L, d_in) -> 输出 z (B, L, d_out)。
    """

    def __init__(self, num_leads=8, sizes=(64, 2048, 2048, 2048)):
        super().__init__()
        self.num_leads = num_leads
        self.sizes = list(sizes)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(num_leads, sizes[i + 1], sizes[i]))
            for i in range(len(sizes) - 1)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(sizes[i + 1] * num_leads)
            for i in range(len(sizes) - 2)
        ])
        for w in self.weights:
            nn.init.kaiming_uniform_(w, a=5 ** 0.5)  # 与 nn.Linear 默认一致

    def forward(self, feat):
        # feat: (B, L, d) -> (L, d, B) 按导联 bmm -> (L, out, B) -> (B, L, out)
        B, L, d = feat.shape
        h = feat
        for li, w in enumerate(self.weights):
            h = torch.bmm(w, h.transpose(0, 1).transpose(1, 2))  # (L, out, B)
            h = h.permute(2, 0, 1)  # (B, L, out)
            if li < len(self.bns):
                h = self.bns[li](h.reshape(B, -1)).reshape(B, L, -1)
                h = torch.relu(h)
        return h

    def export_to_per_lead(self):
        """-> L 个 [Linear.weight/bn params...] 的 state_dict 列表(与原版 projector 结构对应)。"""
        outs = []
        for l in range(self.num_leads):
            sd = {}
            for li, w in enumerate(self.weights):
                sd[f"{li}.weight"] = w[l]
            for bi, bn in enumerate(self.bns):
                c = bn.num_features // self.num_leads
                for k in ("weight", "bias", "running_mean", "running_var"):
                    sd[f"bn{bi}.{k}"] = getattr(bn, k).view(self.num_leads, c)[l]
                sd[f"bn{bi}.num_batches_tracked"] = bn.num_batches_tracked
            outs.append(sd)
        return outs
