from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
import numpy as np
from torch import nn, optim
import torch
import torchvision.transforms as transforms
from data_utils.data_folder import ECGDatasetFolder
from data_utils.multi_view_data_injector import MultiViewDataInjector
from data_utils.augmentations import RandomResizeCropTimeOut, ToTensor
from data_utils.seed_utils import set_seed
from models.vgg_1d import VGG16, group_whiten, group_ln

parser = argparse.ArgumentParser(description='Lead-Fusion Barlow Twins Pretraining')
parser.add_argument('--data-dir', type=Path, required=True,
                    metavar='DIR', help='data path')
parser.add_argument('--num-leads', default=8, type=int, metavar='N', help="the number of leads")
parser.add_argument('--workers', default=6, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=0.001, type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--gamma', default=0.8, type=float, metavar='L',
                    help='balance parameter of the loss')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='2048-2048-2048', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--seed', default=0, type=int, metavar='N', help='random seed')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
# ===== S1/D9: VICReg 化开关 (默认全关 == B0) =====
parser.add_argument('--loss-mode', default='bt', choices=['bt', 'vicreg'],
                    help='目标函数: bt=原版 Barlow Twins (B0), vicreg=D9 三项化')
parser.add_argument('--vicreg-sim', default=25.0, type=float, help='VICReg invariance(MSE) 系数')
parser.add_argument('--vicreg-var', default=25.0, type=float, help='VICReg variance hinge 系数')
parser.add_argument('--vicreg-cov', default=1.0, type=float, help='VICReg covariance 系数')
parser.add_argument('--vicreg-var-eps', default=1e-4, type=float, help='variance hinge 的 sqrt 内 eps')
parser.add_argument('--vicreg-keep-bn', action='store_true',
                    help='vicreg 模式下保留输出 BN(affine=False); 默认去除, 方差交给 hinge')
parser.add_argument('--projector-norm', default='batchnorm', choices=['batchnorm', 'layernorm'],
                    help='projector 隐层归一化: batchnorm=B0 原版, layernorm=D9 消融')
parser.add_argument('--bt-var-hinge', default=0.0, type=float,
                    help='D9-lite: BT 目标之上对 projector 原始输出加 variance hinge 的权重 (0=关闭,逐位等于 B0)')
parser.add_argument('--fast-backbone', action='store_true',
                    help='M0 提速: 分组卷积合并 8 导联主干+投影头(数值等价 rel<1e-6);默认关闭=原路径')
# ===== M 矩阵候选开关 (N4/D1L) =====
parser.add_argument('--ema-decay', default=0.0, type=float,
                    help='N4: 权重 EMA 衰减系数(如 0.999);0=关闭。开启时 checkpoint 存 EMA 权重')
parser.add_argument('--d1l', default='', type=str,
                    help='D1L: 结构化目标矩阵 "ii_iii,adjacent" (如 "0.5,0.2");空=关闭(B0)。'
                         'inter-loss 的跨导联相关目标从 0 改为生理拓扑设定值')
parser.add_argument('--d1l-shuffle', action='store_true',
                    help='D1L NEG 负对照: 打乱目标矩阵的导联归属(同值随机重排), 应不涨点才有效')
parser.add_argument('--d1l-full', action='store_true',
                    help='D1L 闭合基线: 全1目标矩阵(所有跨导联对角目标=1==B0 隐式目标), '
                         'loss 应与 B0 逐位一致; 仅用于单测/审计, 不入实验矩阵')
parser.add_argument('--aug-params', default='0.5,1.0,0.0,0.5', type=str,
                    help='RRC-TO 增强参数 crop_low,crop_up,mask_low,mask_up (默认=论文原值,从未扫过)')
# ===== 跨域迁移候选 (batch4) =====
parser.add_argument('--speed-perturb', default='1.0,1.0', type=str,
                    help='速度扰动(语音迁移) low,high 因子;默认 1.0,1.0=关闭')
# ===== 06 文献第一批开关 (A-P2, 默认关==B0 逐位一致; 2026-09-20 合并恢复) =====
parser.add_argument('--blur-pool', default=0, type=int,
                    help='H2 抗混叠下采样: >0 时每个下采样点前加 filt=N 固定二项式低通(零参数)')
parser.add_argument('--pool-power', default=0.0, type=float,
                    help='T3 幂均值池化: >0 时末端池化换 Q=N 广义幂均值(带符号稳定版,零参数)')
parser.add_argument('--whiten', default=0, type=int,
                    help='H1 分组白化: >0 时编码器 64 维 h 分 N 组组内白化(零参数,仅预训练 forward)')
parser.add_argument('--whiten-ln', action='store_true',
                    help='H1 LN-NEG: 分组 LayerNorm 替代白化(只标准化不去相关), 须与 --whiten 同用')
parser.add_argument('--whiten-shuffle', action='store_true',
                    help='H1 位置-NEG: 白化前固定随机置换维度(白化错误分组), 须与 --whiten 同用')
parser.add_argument('--view2-params', default='', type=str,
                    help='非对称增强(半监督视觉迁移): 视图2 独立 RRC-TO 参数;空=两视图同分布(B0)')
parser.add_argument('--lead-swap-prob', default=0.0, type=float,
                    help='相邻导联互换增强(阵列迁移, 电极错位模拟)概率;0=关闭')
parser.add_argument('--cautious', action='store_true',
                    help='Cautious Adam(优化器前沿迁移): 屏蔽与梯度符号相反的动量更新')
parser.add_argument('--d7-weight', default=0.0, type=float,
                    help='D7 掩码重建支路权重 η(0=关闭;推荐 0.1 起)')
parser.add_argument('--d7-mask', default=0.5, type=float,
                    help='D7 时间掩码比例(跨导联同掩码,重建被掩段)')
parser.add_argument('--n3-prob', default=0.0, type=float,
                    help='N3 患者级正对: 第二视图换成同患者另一记录的概率(0=关闭)')
parser.add_argument('--common-weight', default=0.0, type=float,
                    help='共模视图: 各导联与跨导联均值信号的 BT 对齐项权重(0=关闭)')
# ===== 主机B 任务开关 (HOSTS §四; 默认全关 == B0) =====
parser.add_argument('--h3-weight', default=0.0, type=float,
                    help='H3 患者身份不变性(轻量去相关): 患者共享方向偏离批均值的惩罚权重 (0=关闭)')
parser.add_argument('--h3-prob', default=0.3, type=float,
                    help='H3 三元组中同患者伙伴视图的触发概率(与 n3_prob 同默认值保持两臂对称)')
parser.add_argument('--sinc-frontend', default=0, type=int,
                    help='C1 Sinc 带通前端: 第一层替换为参数化带通滤波器组的通道数 M(16/32; 0=关闭)')
parser.add_argument('--sinc-reg', default=0.01, type=float,
                    help='C1 频带正则权重(带宽约束 [1,15]Hz, 防退化全带=普通卷积)')
parser.add_argument('--sinc-kernel', default=101, type=int,
                    help='C1 Sinc 核长(点数; 有效采样率 204.8Hz)')
# ===== 主机B P3 开关 (HOSTS §四; 默认全关 == B0) =====
parser.add_argument('--mixup-prob', default=0.0, type=float,
                    help='C3 准周期 MixUp: 每样本与随机他样本混合的概率(0=关闭)')
parser.add_argument('--mixup-align', default=1, type=int, choices=[0, 1],
                    help='C3 相位对齐: 1=FFT互相关对齐(准周期版), 0=普通 MixUp 对照')
parser.add_argument('--hrv-weight', default=0.0, type=float,
                    help='H4 HRV 借口: lead-II 表征回归 HRV 统计的辅助头权重(0=关闭; 需 data/pt_hrv.npz)')
# ===== ACL 四区受控实现 (任务书 T2/A2/A3/A4; 默认关==B0 逐位一致) =====
parser.add_argument('--acl-region', default=0, type=int,
                    help='ACL-ECG 四区: 0=关闭(B0); 1=真实解剖分组; 2/3/4=随机分组 partition 1/2/3(A4 NEG)')
parser.add_argument('--acl-inter', default=1, type=int, choices=[0, 1],
                    help='A2=0 仅区域内 InfoNCE; A3=1 加跨区域 Eq.(11)')
parser.add_argument('--acl-gamma', default=0.5, type=float, help='A3: loss = g*intra + (1-g)*inter')
parser.add_argument('--acl-tau', default=0.5, type=float, help='ACL InfoNCE 温度')


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class CautiousAdam(optim.Adam):
    """Cautious Optimizer (2024, 迁移版): 屏蔽与当前梯度符号相反的动量更新。"""

    @torch.no_grad()
    def step(self, closure=None):
        prevs = [(p, p.detach().clone())
                 for g in self.param_groups for p in g['params'] if p.grad is not None]
        super().step(closure)
        for p, prev in prevs:
            update = p - prev
            mask = (update * p.grad) < 0
            if mask.any():
                p.mul_(~mask).add_(prev * mask)


class LeadFusionBT(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fast = getattr(args, 'fast_backbone', False)
        # A-P2 开关守卫: H2/T3 不支持 fast 分组卷积路径, 也不与 D7 组合(D7 取 model[:-1] 特征图)
        _bp, _pp = int(getattr(args, 'blur_pool', 0)), float(getattr(args, 'pool_power', 0.0))
        assert not (self.fast and (_bp or _pp)), "H2/T3 暂不支持 fast 路径"
        assert not (getattr(args, 'd7_weight', 0.0) > 0 and (_bp or _pp)), "H2/T3 暂不与 D7 组合"
        self.sinc_M = 0  # 2026-09-20 修复: fast 路径在下方提前 return, forward 无条件访问 sinc_M,
        #  b0fast 因此 AttributeError 秒败 3 轮; 初始化必须提到 return 之前(升级逻辑仍在下方慢路径)
        self.whiten_g = int(getattr(args, 'whiten', 0))
        self.whiten_ln_on = bool(getattr(args, 'whiten_ln', False))
        # 位置-NEG: 固定种子置换(与训练 seed 无关, 保证可复现)
        _perm = torch.randperm(64, generator=torch.Generator().manual_seed(0))
        self.whiten_perm = _perm
        self.whiten_inv = torch.empty_like(_perm)
        self.whiten_inv[_perm] = torch.arange(64)
        self.whiten_shuffle_on = bool(getattr(args, 'whiten_shuffle', False))
        sizes = [64] + list(map(int, args.projector.split('-')))
        # D1L-fix (任务书 T1/A1, 2026-09-20): 结构化目标矩阵, lead 序: ii,iii,v1..v6 = 0..7
        # 语义修正: τ 只进 cross-correlation 的 diagonal target (跨导联同坐标对齐),
        # off-diagonal target 恒为 0 (旧实现把 τ 放 off_diagonal(c) 制造维度间冗余, 已撤回)。
        # 矩阵契约: 对称 / diag=1 / PSD(eig >= -1e-6); '1,1' 全 1 矩阵为闭合基线
        # (对角目标全 1 == B0 隐式目标, loss 应与 B0 逐位一致, 见单测)。
        self.d1l_P = None
        self.d1l_meta = None
        if getattr(args, 'd1l', '') or getattr(args, 'd1l_full', False):
            if getattr(args, 'd1l_full', False):
                # 闭合基线: 全1矩阵 (所有跨导联对角目标=1 == B0 隐式目标), 仅用于单测/审计
                P = torch.ones(args.num_leads, args.num_leads)
            else:
                a, b = map(float, args.d1l.split(','))
                P = torch.eye(args.num_leads)
                P[0, 1] = P[1, 0] = a          # II-III (Einthoven 相邻)
                for k in range(2, args.num_leads - 1):  # 相邻胸导 V1-V2 ... V5-V6
                    P[k, k + 1] = P[k + 1, k] = b
            if getattr(args, 'd1l_shuffle', False):  # NEG 负对照: 固定种子打乱导联归属
                g = torch.Generator().manual_seed(12345)
                perm = torch.randperm(args.num_leads, generator=g)
                P = P[perm][:, perm]
            P_raw = P.clone()
            # PSD 投影(最近相关矩阵): 特征值裁剪到 >=0 后对角归一, 迭代至收敛
            for _ in range(3):
                if (torch.linalg.eigvalsh(P).min() >= -1e-6
                        and torch.allclose(torch.diagonal(P), torch.ones(args.num_leads), atol=1e-6)):
                    break
                eig, vec = torch.linalg.eigh(P)
                eig = eig.clamp_min(0.0)
                P = vec @ torch.diag(eig) @ vec.T
                d = torch.sqrt(torch.clamp(torch.diagonal(P), min=1e-12))
                P = P / d[:, None] / d[None, :]
            eigvals = torch.linalg.eigvalsh(P)
            assert eigvals.min() >= -1e-6, f"D1L 目标矩阵非 PSD: min eig = {eigvals.min():.2e}"
            self.d1l_P = P.to(self.device)
            self.d1l_meta = {
                "raw": P_raw.cpu().numpy().tolist(),
                "psd_projected": P.cpu().numpy().tolist(),
                "eigenvalues": eigvals.cpu().numpy().tolist(),
                "shuffle": bool(getattr(args, 'd1l_shuffle', False)),
                "full": bool(getattr(args, 'd1l_full', False)),
                "note": "tau -> cross-corr diagonal target; off-diag target = 0",
            }
        # ACL 四区受控实现 (T2): 替换式区域目标, h(两导联 concat)->独立 region projector
        self.acl_on = int(getattr(args, 'acl_region', 0)) > 0
        self.acl_regions = None
        self.acl_projectors = None
        if self.acl_on:
            assert not self.fast, "ACL 区域路径暂不支持 fast"
            from models.acl_region import build_regions, RegionProjector
            part = 0 if int(args.acl_region) == 1 else int(args.acl_region) - 1
            self.acl_regions = build_regions(part)
            self.acl_projectors = nn.ModuleList(
                [RegionProjector(in_dim=128).to(self.device) for _ in range(4)])
        if self.fast:
            from models.parallel_vgg import ParallelVGG16, ParallelProjector
            self.p_vgg = ParallelVGG16(num_leads=args.num_leads, ch_in=1).to(self.device)
            self.p_proj = ParallelProjector(num_leads=args.num_leads, sizes=sizes).to(self.device)
            self.bn_all = nn.BatchNorm1d(sizes[-1] * args.num_leads, affine=False).to(self.device)
            # 兼容原字段(优化器构建/保存路径判空用)
            self.backbone_group = []
            self.projector_group = []
            self.bn_group = []
            return
        self.backbone_group = list()
        for i in range(args.num_leads):
            backbone = VGG16(ch_in=1, alpha=0.125,
                             blur_pool=int(getattr(args, 'blur_pool', 0)),
                             pool_power=float(getattr(args, 'pool_power', 0.0)))
            backbone.fc = nn.Identity()
            self.backbone_group.append(backbone.to(self.device))

        self.projector_group = list()
        for i in range(args.num_leads):
            layers = []
            for j in range(len(sizes) - 2):
                layers.append(nn.Linear(sizes[j], sizes[j + 1], bias=False))
                if args.projector_norm == 'layernorm':
                    layers.append(nn.LayerNorm(sizes[j + 1]))
                else:
                    layers.append(nn.BatchNorm1d(sizes[j + 1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
            self.projector_group.append(nn.Sequential(*layers).to(self.device))
        self.bn_group = list()
        for i in range(args.num_leads):
            self.bn_group.append(nn.BatchNorm1d(sizes[-1], affine=False).to(self.device))
        # C1: Sinc 带通前端替换(慢路径; model[0][0]=Conv1d(1,C1,k3)+BN+ReLU -> SincConv(1,M)+BN+ReLU)
        self.sinc_M = 0
        if getattr(args, 'sinc_frontend', 0) > 0:
            assert not self.fast, "C1 Sinc 前端暂不支持 fast 路径"
            from models.sinc_conv import apply_sinc_frontend
            self.sinc_M = int(args.sinc_frontend)
            for backbone in self.backbone_group:
                apply_sinc_frontend(backbone, self.sinc_M,
                                    kernel_size=int(getattr(args, 'sinc_kernel', 101)))
                backbone.to(self.device)  # 手术在 .to(device) 之后发生, 需补搬运
        # H4: HRV 回归辅助头(lead-II 64 维表征 -> 4 统计量; 仅慢路径)
        self.hrv_head = None
        if getattr(args, 'hrv_weight', 0.0) > 0:
            assert not self.fast, "H4 暂不支持 fast 路径"
            self.hrv_head = nn.Sequential(          # 64 = int(512*alpha), alpha=0.125
                nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4)).to(self.device)
        # D7: 并联重建支路(仅慢路径; 共享 backbone, 池化前特征接轻量解码器)
        self.d7_decoders = None
        if getattr(args, 'd7_weight', 0.0) > 0:
            assert not self.fast, "D7 暂不支持 fast 路径"
            from models.decoders import SmallDecoder
            self.d7_decoders = nn.ModuleList(
                [SmallDecoder().to(self.device) for _ in range(args.num_leads)])

    def _d7_recon_loss(self, y1):
        """对第一视图做跨导联同掩码, 重建被掩段(MSE 仅在被掩位置)。"""
        B, L, T = y1.shape
        mlen = max(1, int(self.args.d7_mask * T))
        mstart = torch.randint(0, T - mlen, (1,)).item()
        mask = torch.zeros(1, 1, T, device=y1.device)
        mask[..., mstart:mstart + mlen] = 1.0
        xm = y1 * (1 - mask)
        rec_loss = 0
        for i in range(L):
            feat_map = self.backbone_group[i].model[:-1](xm[:, [i], :])
            rec = self.d7_decoders[i](feat_map)  # (B, 1, T)
            rec_loss = rec_loss + nn.functional.mse_loss(
                rec[:, 0, mstart:mstart + mlen],
                y1[:, i, mstart:mstart + mlen])
        return rec_loss / L

    def _encode_h(self, y):
        """(B, L, T) -> 每导联编码器输出 h 列表 [(B,64)] (含 whiten, 与 projector 输入同口径)。
        ACL 区域路径复用 (region 输入 = 区内两导联 h concat)。"""
        hs = list()
        for i in range(self.args.num_leads):
            h = self.backbone_group[i](y[:, [i], :])
            if self.whiten_g > 0:
                if self.whiten_shuffle_on:
                    h = h[:, self.whiten_perm.to(h.device)]
                    h = group_whiten(h, self.whiten_g)
                    h = h[:, self.whiten_inv.to(h.device)]  # H1 位置-NEG: 白化错误分组
                elif self.whiten_ln_on:
                    h = group_ln(h, self.whiten_g)  # H1 LN-NEG: 只标准化不去相关
                else:
                    h = group_whiten(h, self.whiten_g)  # H1: 组内白化后再进 projector
            hs.append(h)
        return hs

    def _embed(self, y):
        """(B, L, T) -> 原始 projector 输出 z 列表 (每导联 (B, d))。"""
        if not self.fast:
            return [self.projector_group[i](h) for i, h in enumerate(self._encode_h(y))]
        feat = self.p_vgg(y).view(y.shape[0], self.args.num_leads, -1)
        z = self.p_proj(feat)  # (B, L, d)
        return [z[:, i, :] for i in range(self.args.num_leads)]

    def _bn_embed(self, z_list):
        """BT 路径的输出 BN: 每导联 affine=False。"""
        if not self.fast:
            return [self.bn_group[i](z_list[i]) for i in range(self.args.num_leads)]
        z = torch.stack(z_list, dim=1)  # (B, L, d)
        zb = self.bn_all(z.reshape(z.shape[0], -1)).reshape_as(z)
        return [zb[:, i, :] for i in range(self.args.num_leads)]

    def _pair_loss(self, zi, zj):
        """D9: 单导联对的 VICReg 三项损失 (官方口径: MSE + std hinge + off-diag cov/d)。

        zi/zj 为 projector 原始输出 (keep-bn 模式下已过输出 BN)。
        """
        inv = nn.functional.mse_loss(zi, zj)
        var = (torch.relu(1.0 - torch.sqrt(zi.var(dim=0) + self.args.vicreg_var_eps)).mean()
               + torch.relu(1.0 - torch.sqrt(zj.var(dim=0) + self.args.vicreg_var_eps)).mean()) / 2
        cov = 0
        for z in (zi, zj):
            zc = z - z.mean(dim=0)
            c = (zc.T @ zc) / z.shape[0]
            cov = cov + off_diagonal(c).pow_(2).sum() / z.shape[1]
        cov = cov / 2
        return (self.args.vicreg_sim * inv
                + self.args.vicreg_var * var
                + self.args.vicreg_cov * cov)

    def forward(self, y1, y2, y_pair=None, pair_mask=None,
                hrv_target=None, hrv_valid=None):
        if self.acl_on:
            # A2/A3 替换式区域目标 (任务书: 不与旧 LGA/inter 叠加): loss_r=intra, loss_t=inter
            from models.acl_region import region_embed, acl_loss
            u1 = region_embed(self._encode_h(y1), self.acl_regions, self.acl_projectors)
            u2 = region_embed(self._encode_h(y2), self.acl_regions, self.acl_projectors)
            loss, li, lt = acl_loss(u1, u2, float(self.args.acl_tau),
                                    float(self.args.acl_gamma), bool(self.args.acl_inter))
            return loss, li, lt
        z1_list = self._embed(y1)
        z2_list = self._embed(y2)
        if self.args.loss_mode == 'vicreg':
            # D9: bn 语义交给 variance hinge; keep-bn 开关可保留输出 BN(whitening-lite 消融)
            if self.args.vicreg_keep_bn:
                z1_list = self._bn_embed(z1_list)
                z2_list = self._bn_embed(z2_list)
            loss_r = 0
            loss_t = 0
            for i in range(self.args.num_leads):
                for j in range(self.args.num_leads):
                    ls = self._pair_loss(z1_list[i], z2_list[j])
                    if i == j:
                        loss_r += ls
                    else:
                        loss_t += ls
            loss_r = loss_r / self.args.num_leads
            loss_t = loss_t / (self.args.num_leads * (self.args.num_leads - 1))
            loss = self.args.gamma * loss_r + (1 - self.args.gamma) * loss_t
            return loss, loss_r, loss_t
        z1b = self._bn_embed(z1_list) if self.fast else None
        z2b = self._bn_embed(z2_list) if self.fast else None
        loss_r = 0
        loss_t = 0
        for i in range(self.args.num_leads):
            for j in range(self.args.num_leads):
                # 注意: 保持 B0 原始结构(每对重算 BN)——梯度求和顺序影响逐位一致性
                c1 = z1b[i] if self.fast else self.bn_group[i](z1_list[i])
                c2 = z2b[j] if self.fast else self.bn_group[j](z2_list[j])
                c = c1.T @ c2
                c.div_(self.args.batch_size)
                # D1L-fix: τ 只进对角目标(跨导联同坐标对齐), off-diag 目标恒为 0;
                # 关态 (i==j 或 P=None) 走原 B0 算子与顺序, 逐位一致
                if self.d1l_P is not None and i != j:
                    on_diag = (torch.diagonal(c) - self.d1l_P[i, j]).pow(2).sum()
                else:
                    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = off_diagonal(c).pow_(2).sum()
                ls = on_diag + self.args.lambd * off_diag
                if i == j:
                    loss_r += ls
                else:
                    loss_t += ls
        loss_r = loss_r / self.args.num_leads
        loss_t = loss_t / (self.args.num_leads * (self.args.num_leads - 1))
        loss = self.args.gamma * loss_r + (1 - self.args.gamma) * loss_t
        if getattr(self.args, 'bt_var_hinge', 0.0) > 0:
            # D9-lite: 最小防塌补丁, 仅对 16 个原始嵌入 (8导联x2视图) 加 variance hinge
            hinge = 0
            for z_list in (z1_list, z2_list):
                for z in z_list:
                    hinge = hinge + torch.relu(
                        1.0 - torch.sqrt(z.var(dim=0) + self.args.vicreg_var_eps)).mean()
            loss = loss + self.args.bt_var_hinge * hinge / (2 * self.args.num_leads)
            loss_r = loss_r + self.args.bt_var_hinge * hinge / (2 * self.args.num_leads)
        if getattr(self, 'd7_decoders', None) is not None:
            # D7: 并联掩码重建支路 L = L_BT + η·L_rec
            rec = self._d7_recon_loss(y1)
            loss = loss + self.args.d7_weight * rec
            loss_r = loss_r + self.args.d7_weight * rec
        if getattr(self.args, 'common_weight', 0.0) > 0:
            # 共模视图: 各导联与跨导联均值信号(噪声抵消)的 BT 对齐(相关锚定式,非拉近)
            y_common = y1.mean(dim=1, keepdim=True)  # (B,1,T)
            zc_list = self._embed(y_common.repeat(1, self.args.num_leads, 1))
            cm = 0
            for i in range(self.args.num_leads):
                c = self.bn_group[i](z1_list[i]).T @ self.bn_group[i](zc_list[i])
                c.div_(self.args.batch_size)
                cm = cm + torch.diagonal(c).add_(-1).pow_(2).sum() \
                    + self.args.lambd * off_diagonal(c).pow_(2).sum()
            cm = cm / self.args.num_leads
            loss = loss + self.args.common_weight * cm
            loss_r = loss_r + self.args.common_weight * cm
        if getattr(self.args, 'h3_weight', 0.0) > 0 and y_pair is not None and pair_mask is not None:
            # H3 患者身份不变性(轻量去相关, 主机B): 患者共享方向 mu_p=(z1+z_pair)/2,
            # 惩罚其在批内的离散度(患者间协方差 off-block -> 0), 尺度按批内 std(stopgrad)
            # 归一保持与 BT on-diagonal 同量级; 仅对有同患者伙伴的行前向(flag 过滤)。
            # 仅当带伙伴样本数 >= 2 时计算(批大小 1 无法过训练态 BatchNorm;
            # ~6.4% 样本带 flag, 每 batch 恰 1 个的概率 ~0.2%, 跳过无碍统计)
            if int(pair_mask.sum()) >= 2:
                idx = pair_mask.nonzero(as_tuple=True)[0]
                yp = y_pair[idx]
                h3 = 0
                for i in range(self.args.num_leads):
                    z1s = z1_list[i][idx]
                    zps = self.projector_group[i](self.backbone_group[i](yp[:, [i], :]))
                    mu = (z1s + zps) / 2
                    mu_bar = mu.mean(dim=0, keepdim=True)
                    sigma = z1s.std(dim=0, keepdim=True).detach() + 1e-4
                    h3 = h3 + ((mu - mu_bar) / sigma).pow(2).mean()
                h3 = h3 / self.args.num_leads
                loss = loss + self.args.h3_weight * h3
                loss_r = loss_r + self.args.h3_weight * h3
        if self.sinc_M > 0:
            # C1 频带正则: 带宽压在 [1,15]Hz, 防退化全带=普通卷积(主机B)
            from models.sinc_conv import sinc_band_penalty
            sreg = 0
            for backbone in self.backbone_group:
                sreg = sreg + sinc_band_penalty(backbone.model[0][0][0])
            sreg = sreg / self.args.num_leads
            loss = loss + self.args.sinc_reg * sreg
            loss_r = loss_r + self.args.sinc_reg * sreg
        if getattr(self.args, 'hrv_weight', 0.0) > 0 and self.hrv_head is not None \
                and hrv_target is not None and hrv_valid is not None:
            # H4 借口: lead-II 表征回归全库归一化的 HRV 统计(仅有效样本行)
            m = hrv_valid > 0.5
            if bool(m.any()):
                feat = self.backbone_group[0](y1[:, [0], :])   # (B, 64) lead II
                l4 = nn.functional.mse_loss(self.hrv_head(feat[m]), hrv_target[m])
                loss = loss + self.args.hrv_weight * l4
                loss_r = loss_r + self.args.hrv_weight * l4
        return loss, loss_r, loss_t


def main_worker(gpu, args):

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = LeadFusionBT(args)
    if getattr(model, 'd1l_meta', None) is not None:
        # T1 契约: 导出目标矩阵/特征值/配置, 供审计与闭合基线复核
        np.save(args.checkpoint_dir / "target_matrix.npy",
                torch.tensor(model.d1l_meta["psd_projected"]))
        with open(args.checkpoint_dir / "d1l_config.json", "w", encoding="utf-8") as f:
            json.dump(model.d1l_meta, f, ensure_ascii=False, indent=1)

    if model.fast:
        param_weights = [p for p in model.p_vgg.parameters() if p.ndim > 1]
        param_biases = [p for p in model.p_vgg.parameters() if p.ndim == 1]
        for md in (model.p_proj, model.bn_all):
            for param in md.parameters():
                (param_biases if param.ndim == 1 else param_weights).append(param)
    else:
        param_weights = []
        param_biases = []
        for md in model.backbone_group:
            for param in md.parameters():
                if param.ndim == 1:
                    param_biases.append(param)
                else:
                    param_weights.append(param)

        for md in model.projector_group:
            for param in md.parameters():
                if param.ndim == 1:
                    param_biases.append(param)
                else:
                    param_weights.append(param)

        for md in model.bn_group:
            for param in md.parameters():
                if param.ndim == 1:
                    param_biases.append(param)
                else:
                    param_weights.append(param)

        if getattr(model, 'd7_decoders', None) is not None:
            for param in model.d7_decoders.parameters():
                if param.ndim == 1:
                    param_biases.append(param)
                else:
                    param_weights.append(param)

        if getattr(model, 'hrv_head', None) is not None:
            for param in model.hrv_head.parameters():
                if param.ndim == 1:
                    param_biases.append(param)
                else:
                    param_weights.append(param)

        if getattr(model, 'acl_projectors', None) is not None:
            for param in model.acl_projectors.parameters():
                if param.ndim == 1:
                    param_biases.append(param)
                else:
                    param_weights.append(param)

    parameters = param_weights + param_biases
    if getattr(args, 'cautious', False):
        optimizer = CautiousAdam(parameters, lr=args.learning_rate)
    else:
        optimizer = optim.Adam(parameters, lr=args.learning_rate)

    # N4: 权重 EMA(不含 BN 统计, 标准 weight-EMA 口径)
    ema_shadows = None
    if args.ema_decay > 0:
        ema_shadows = [p.detach().clone() for p in parameters]

    aug_p = [float(v) for v in args.aug_params.split(',')]
    from data_utils.augmentations import SpeedPerturbRRC_TO, AdjacentLeadSwap
    sp = tuple(float(v) for v in args.speed_perturb.split(','))
    if (sp[0], sp[1]) == (1.0, 1.0) and args.lead_swap_prob <= 0:
        t = transforms.Compose([RandomResizeCropTimeOut(params=aug_p), ToTensor()])
    else:
        t = transforms.Compose([
            AdjacentLeadSwap(prob=args.lead_swap_prob),
            SpeedPerturbRRC_TO(params=aug_p, speed=sp),
            ToTensor()])
    if args.view2_params:
        t2 = transforms.Compose([
            AdjacentLeadSwap(prob=args.lead_swap_prob),
            SpeedPerturbRRC_TO(
                params=[float(v) for v in args.view2_params.split(',')], speed=sp),
            ToTensor()])
    else:
        t2 = t
    if getattr(args, 'mixup_prob', 0.0) > 0:
        from data_utils.c3_mixup import MixUpDataset
        dataset = MixUpDataset(args.data_dir, t, t2,
                               prob=args.mixup_prob, align=bool(args.mixup_align))
    elif getattr(args, 'hrv_weight', 0.0) > 0:
        from data_utils.h4_dataset import HRVDataset
        hrv_npz = Path('data/pt_hrv.npz')
        if not hrv_npz.exists():
            raise FileNotFoundError('H4 需先运行 runlog/prep_hrv.py 生成 data/pt_hrv.npz')
        dataset = HRVDataset(args.data_dir, t, t2, str(hrv_npz))
    elif getattr(args, 'h3_weight', 0.0) > 0:
        from data_utils.h3_dataset import H3TripletDataset
        from data_utils.n3_dataset import load_patient_map
        dataset = H3TripletDataset(args.data_dir, t, t2,
                                   load_patient_map(), prob=args.h3_prob)
    elif getattr(args, 'n3_prob', 0.0) > 0:
        from data_utils.n3_dataset import N3PairsDataset, load_patient_map
        dataset = N3PairsDataset(args.data_dir, t,
                                 load_patient_map(), prob=args.n3_prob)
    else:
        dataset = ECGDatasetFolder(args.data_dir, transform=MultiViewDataInjector([t, t2]))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True,
        pin_memory=True)

    for epoch in range(0, args.epochs):
        total_loss = 0
        total_loss_r = 0
        total_loss_t = 0
        ep_start_time = time.time()
        for step, batch in enumerate(loader, start=epoch * len(loader)):
            (views, flag) = batch
            y1 = views[0].to(model.device, non_blocking=True)
            y2 = views[1].to(model.device, non_blocking=True)
            y_pair = pair_mask = hrv_t = hrv_v = None
            if isinstance(flag, (tuple, list)):
                # H4: flag = (targets(B,4), valid(B,))
                hrv_t = flag[0].to(model.device, non_blocking=True)
                hrv_v = flag[1].to(model.device, non_blocking=True)
            elif len(views) == 3:
                # H3 三元组: y_pair=同患者伙伴视图, flag=是否有效
                y_pair = views[2].to(model.device, non_blocking=True)
                pair_mask = flag.to(model.device).float() > 0.5
            optimizer.zero_grad()
            loss, loss_r, loss_t = model.forward(y1, y2, y_pair, pair_mask,
                                                 hrv_target=hrv_t, hrv_valid=hrv_v)
            loss.backward()
            optimizer.step()
            if ema_shadows is not None:
                with torch.no_grad():
                    for p, s in zip(parameters, ema_shadows):
                        s.mul_(args.ema_decay).add_(p.detach(), alpha=1 - args.ema_decay)
            if step % args.print_freq == 0:
                stats = dict(epoch=epoch, step=step,
                             loss=loss.item(),
                             loss_r=loss_r.item(),
                             loss_t=loss_t.item())
                print(json.dumps(stats))
            total_loss += loss.item()
            total_loss_r += loss_r.item()
            total_loss_t += loss_t.item()

        total_loss /= len(loader)
        total_loss_r /= len(loader)
        total_loss_t /= len(loader)
        ep_end_time = time.time()

        print("\nEpoch end. Time: %f - Average loss %f - loss_r %f - loss_t %f.\n" % (
            ep_end_time - ep_start_time, total_loss, total_loss_r, total_loss_t))

    # N4: 保存前把 EMA 影子权重换入(EMA 开启时 checkpoint 即 EMA 权重)
    if ema_shadows is not None:
        with torch.no_grad():
            for p, s in zip(parameters, ema_shadows):
                p.copy_(s)

    # 双格式保存: 兼容旧版(模块对象列表) + 纯 state_dict (跨版本稳定, 指南 §5 P1)
    ckpt_path = args.checkpoint_dir / 'encoder_group.pth'
    if model.fast:
        from models.parallel_vgg import export_parallel_to_state_dicts
        sd_list = export_parallel_to_state_dicts(model.p_vgg)
        torch.save({
            'backbone_state_dict': sd_list,
            'backbone_state_dict_list': sd_list,
        }, ckpt_path)
    else:
        torch.save({
            'backbone_state_dict': model.backbone_group,
            'backbone_state_dict_list': [enc.state_dict() for enc in model.backbone_group],
        }, ckpt_path)
    sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    print("Checkpoint saved:", ckpt_path)
    print("SHA256:", sha)

    # C1: 导出学到的截止频率(可解释性: 直方图数据; HOSTS §四 P2)
    if model.sinc_M > 0:
        from models.sinc_conv import sinc_bands
        bands = {f"lead{i}": sinc_bands(model.backbone_group[i].model[0][0][0])
                 for i in range(args.num_leads)}
        with open(args.checkpoint_dir / "sinc_bands.json", "w") as f:
            json.dump(bands, f, indent=1)
        print("Sinc bands exported:", args.checkpoint_dir / "sinc_bands.json")

    # 环境与配置记录 (指南 §6-6)
    info = dict(
        args=vars(args),
        python=platform.python_version(),
        torch=torch.__version__,
        torchvision=torchvision_version(),
        cuda_build=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        seed=args.seed,
        checkpoint_sha256=sha,
        pretrain_dataset="PTB-XL folds 1-8 (替代 NFH, 非论文原始设置)",
    )
    with open(args.checkpoint_dir / "config.json", "w") as f:
        json.dump(info, f, indent=1, default=str)


def torchvision_version():
    import torchvision
    return torchvision.__version__


def wait_for_gpu_memory(need_mib=6500, poll_s=60):
    """启动显存闸门(2026-09-17): 在 CUDA 初始化前等待空闲显存。
    背景: 双链+多车道并发时曾出现 4 个预训练同时启动把 24.5G 显存占爆。
    本闸门保证任何来源(m_screen/chain/pipeline_runner)启动的预训练, 在空闲显存
    不足时于加载模型前排队等待, 物理上杜绝超发 OOM。nvidia-smi 不可用时直接放行。"""
    import subprocess as _sp
    while True:
        try:
            out = _sp.run(['nvidia-smi', '--query-gpu=memory.free',
                           '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, timeout=60).stdout
            free = int(out.strip().splitlines()[0])
        except Exception:
            return
        if free >= need_mib:
            return
        print(f"[vram-gate] 等待显存: 需要 {need_mib}MiB, 当前空闲 {free}MiB ({poll_s}s 后重查)", flush=True)
        time.sleep(poll_s)


def main():
    args = parser.parse_args()
    wait_for_gpu_memory()
    set_seed(args.seed)
    print("Pretraining Setting ======================================")
    print(args)
    print("==========================================================")
    main_worker(0, args)


if __name__ == '__main__':
    main()
