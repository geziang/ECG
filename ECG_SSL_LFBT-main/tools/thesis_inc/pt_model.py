# -*- coding: utf-8 -*-
"""thesis-inc 预训练模型: 复刻 run_pt.LeadFusionBT 慢路径结构 + 新局部目标。

- 逐导联 VGG16(alpha=0.125, trc=1, fc=Identity) + projector(64-2048-2048-2048,
  BN+ReLU) + 输出 BN(affine=False), 与 run_pt 同构同初始化顺序。
- 主损失 = LFBT BT (gamma*loss_r+(1-gamma)*loss_t, c/batch_size 归一)。
- I1/I2 模板一致性: 复刻 _embed_beat + _dv_consistency(逐对重算 BN, on-diag)。
- I4 RR 头: Linear(64,32)+ReLU+Linear(32,3), 输入视图1首导(II) GAP 特征。
- I3/I5 局部加权 BT: 共享 backbone+projector, 权重 1/(N_valid*K_b),
  加权标准化 sqrt(var+1e-4), C=Z1^T diag(w) Z2, 逐导联 on/off-diag, 八导联平均。
"""
import torch
import torch.nn as nn

from models.vgg_1d import VGG16


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class IncBT(nn.Module):
    def __init__(self, num_leads: int = 8, trc: int = 1, projector: str = "2048-2048-2048",
                 gamma: float = 0.8, lambd: float = 0.0051, use_rr: bool = False):
        super().__init__()
        self.num_leads = num_leads
        self.gamma = gamma
        self.lambd = lambd
        sizes = [64] + list(map(int, projector.split("-")))
        self.backbone_group = nn.ModuleList()
        for _ in range(num_leads):
            bb = VGG16(ch_in=1, alpha=0.125, trc=trc)
            bb.fc = nn.Identity()
            self.backbone_group.append(bb)
        self.projector_group = nn.ModuleList()
        for _ in range(num_leads):
            layers = []
            for j in range(len(sizes) - 2):
                layers.append(nn.Linear(sizes[j], sizes[j + 1], bias=False))
                layers.append(nn.BatchNorm1d(sizes[j + 1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
            self.projector_group.append(nn.Sequential(*layers))
        self.bn_group = nn.ModuleList(
            [nn.BatchNorm1d(sizes[-1], affine=False) for _ in range(num_leads)])
        self.rr_head = None
        if use_rr:
            self.rr_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))

    # ---- 前向 ----
    def _embed(self, y):
        """(B,L,T) -> projector 原始输出 z 列表。"""
        return [self.projector_group[i](self.backbone_group[i](y[:, [i], :]))
                for i in range(self.num_leads)]

    def _embed_feat(self, y):
        """(B,L,T) -> (z 列表, 每导联 64 维 GAP 特征列表)。"""
        z_list, f_list = [], []
        for i in range(self.num_leads):
            bb = self.backbone_group[i]
            fm = bb.model[:-1](y[:, [i], :])
            f = bb.model[-1:](fm).view(fm.shape[0], -1)
            z_list.append(self.projector_group[i](bb.fc(f)))
            f_list.append(f)
        return z_list, f_list

    # ---- 损失 ----
    def bt_loss(self, z1_list, z2_list, batch_size: int):
        """LFBT 主目标(逐对重算 BN, c/batch_size 归一; 与 run_pt bt 路径一致)。"""
        loss_r = 0
        loss_t = 0
        for i in range(self.num_leads):
            for j in range(self.num_leads):
                c1 = self.bn_group[i](z1_list[i])
                c2 = self.bn_group[j](z2_list[j])
                c = c1.T @ c2
                c.div_(batch_size)
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = off_diagonal(c).pow(2).sum()
                ls = on_diag + self.lambd * off_diag
                if i == j:
                    loss_r += ls
                else:
                    loss_t += ls
        loss_r = loss_r / self.num_leads
        loss_t = loss_t / (self.num_leads * (self.num_leads - 1))
        return self.gamma * loss_r + (1 - self.gamma) * loss_t, loss_r, loss_t

    def template_loss(self, z1_list, z2_list, template, valid):
        """I1/I2: 整段 vs 心拍模板 BT 式 on-diag 一致性(复刻 _dv_consistency)。"""
        m = valid > 0.5
        if not bool(m.any()):
            return z1_list[0].new_zeros(())
        zb_list = self._embed(template[m])

        def _one(z_list):
            tot = 0
            for i in range(self.num_leads):
                c1 = self.bn_group[i](z_list[i][m])
                c2 = self.bn_group[i](zb_list[i])
                c = c1.T @ c2
                c.div_(int(m.sum()))
                tot = tot + torch.diagonal(c).add_(-1).pow_(2).sum()
            return tot / self.num_leads

        return (_one(z1_list) + _one(z2_list)) / 2

    def rr_loss(self, f1_list, rr3, valid):
        """I4: 视图1首导(II)特征回归 z-scored [mean_RR, SDNN, RMSSD]。"""
        m = valid > 0.5
        if not bool(m.any()):
            return f1_list[0].new_zeros(())
        return nn.functional.mse_loss(self.rr_head(f1_list[0][m]), rr3[m])

    def local_loss(self, w1, w2, wmask):
        """I3/I5: 局部窗两视图加权 BT(独立于 bn_group; 权重=1/(N_valid*K_b))。

        w1/w2: (B,K,8,205); wmask: (B,K) bool。返回 (loss, diag_mean)。
        N_valid<2 时返回 0(调用方另行计数)。
        """
        Kb = wmask.sum(dim=1).float()
        n_valid = int((Kb > 0).sum())
        if n_valid < 2:
            return w1.new_zeros(()), None
        w_rec = torch.where(wmask, 1.0 / (n_valid * Kb[:, None]),
                            torch.zeros((), device=w1.device, dtype=w1.dtype))
        w = w_rec[wmask]                                   # (M,)
        x1 = w1[wmask]                                     # (M,8,205)
        x2 = w2[wmask]
        total = 0
        diag_sum = 0
        for i in range(self.num_leads):
            z1 = self.projector_group[i](self.backbone_group[i](x1[:, [i], :]))
            z2 = self.projector_group[i](self.backbone_group[i](x2[:, [i], :]))
            mu1 = (z1 * w[:, None]).sum(dim=0)
            var1 = ((z1 - mu1) ** 2 * w[:, None]).sum(dim=0)
            z1s = (z1 - mu1) / torch.sqrt(var1 + 1e-4)
            mu2 = (z2 * w[:, None]).sum(dim=0)
            var2 = ((z2 - mu2) ** 2 * w[:, None]).sum(dim=0)
            z2s = (z2 - mu2) / torch.sqrt(var2 + 1e-4)
            c = (z1s * w[:, None]).T @ z2s
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow(2).sum()
            total = total + on_diag + self.lambd * off_diag
            diag_sum = diag_sum + torch.diagonal(c).mean()
        return total / self.num_leads, (diag_sum / self.num_leads).detach()

    def param_groups_pb(self):
        """run_pt 同口径: ndim>1 权重在前, ndim==1 偏置在后。"""
        weights, biases = [], []
        for md in (self.backbone_group, self.projector_group, self.bn_group):
            for p in md.parameters():
                (biases if p.ndim == 1 else weights).append(p)
        if self.rr_head is not None:
            for p in self.rr_head.parameters():
                (biases if p.ndim == 1 else weights).append(p)
        return weights + biases


def init_fingerprint(model: IncBT) -> str:
    """共享主干+投影+BN(不含 rr_head)的初始化指纹(跨配方一致性检查用)。"""
    import hashlib
    h = hashlib.sha256()
    for md in (model.backbone_group, model.projector_group, model.bn_group):
        for p in md.parameters():
            h.update(p.detach().cpu().numpy().tobytes())
        for n, b in md.named_buffers():
            h.update(b.detach().cpu().numpy().tobytes())
    return h.hexdigest()
