from pathlib import Path
import argparse
import hashlib

from utils.pathguard import open_out, safe_out_path
import json
import platform
import time
from torch import nn, optim
import torch
import torchvision.transforms as transforms
from data_utils.data_folder import ECGDatasetFolder
from data_utils.multi_view_data_injector import MultiViewDataInjector
from data_utils.augmentations import RandomResizeCropTimeOut, ToTensor
from data_utils.seed_utils import set_seed
from models.vgg_1d import VGG16

parser = argparse.ArgumentParser(description='Lead-Fusion Barlow Twins Pretraining')
parser.add_argument('--data-dir', type=Path, required=True,
                    metavar='DIR', help='data path')
parser.add_argument('--num-leads', default=8, type=int, metavar='N', help="the number of leads")
parser.add_argument('--resume', action='store_true',
                    help='断点续训: checkpoint_dir/train_state.pth 存在时恢复 epoch/权重/优化器/RNG')
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
parser.add_argument('--loss-mode', default='bt', choices=['bt', 'vicreg', 'simclr', 'clocs'],
                    help='目标函数: bt=原版 Barlow Twins (B0), vicreg=D9 三项化, '
                         'simclr=W4 A-4 外部基线 S1(NT-Xent), clocs=W4 A-5 外部基线 S2')
parser.add_argument('--simclr-temp', default=0.5, type=float,
                    help='S1: NT-Xent 温度(SimCLR 发表默认 0.5, 禁调参)')
parser.add_argument('--clocs-temp', default=0.1, type=float,
                    help='S2: CLOCS 温度(官方 obtain_contrastive_loss 默认 0.1, 禁调参)')
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
parser.add_argument('--aug-params', default='0.5,1.0,0.0,0.5', type=str,
                    help='RRC-TO 增强参数 crop_low,crop_up,mask_low,mask_up (默认=论文原值,从未扫过)')
# ===== 跨域迁移候选 (batch4) =====
parser.add_argument('--speed-perturb', default='1.0,1.0', type=str,
                    help='速度扰动(语音迁移) low,high 因子;默认 1.0,1.0=关闭')
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
# ===== W1 任务书(07) T1/T2/T3 开关 (默认全关 == B0 逐位一致) =====
parser.add_argument('--rec-style', default='none',
                    choices=['none', 'd7', 'multiseg', 'ccm'],
                    help='T3 重建支路: none=关(B0); d7=旧版单段50%%仅视图1(B1归档审计); '
                         'multiseg=B2 多段总20%%两视图独立; ccm=B3 周期内20%%不跨R峰(失败回退multiseg)')
parser.add_argument('--rec-ratio', default=0.2, type=float,
                    help='B2/B3 目标遮挡比例(任务书 §4.2: 约 20%%)')
parser.add_argument('--rpeak-npz', default='data/pt_rpeaks.npz', type=str,
                    help='R 峰缓存文件(gqrs 预计算, 见 runlog/prep_rpeaks.py)')
parser.add_argument('--acl-mode', default='off', choices=['off', 'intra', 'full'],
                    help='T2 ACL 区域目标: off=关(B0); intra=A2 仅区域内一致; full=A3 区域内+区域间 Eq.(11)')
parser.add_argument('--acl-partition', default='anatomy', choices=['anatomy', 'random'],
                    help='四区分组: anatomy=(II,III)/(V1,V2)/(V3,V4)/(V5,V6); random=A4 NEG')
parser.add_argument('--acl-rand-seed', default=101, type=int,
                    help='A4 随机分组种子(任务书 T2: >=3 个 partition, 如 101/102/103)')
parser.add_argument('--acl-projector', default='128-2048-2048-2048', type=str,
                    help='区域 projector MLP 规格(输入 128 = 两导联 64 维 concat)')
parser.add_argument('--acl-tau', default=0.1, type=float,
                    help='InfoNCE temperature(ACL-ECG 论文: tau=0.1, 09-20 按 PDF 核对修正)')
# ===== C2: TRC/GRN1D 时序适配器 (W1 §5.2, E001 Eq.14-17 忠实移植, 默认关==B0 逐位一致) =====
parser.add_argument('--trc', default=0, type=int,
                    help='C2: >0 时每导联 block5 后 GAP 前插 GRN1D(零初始化, +128参数/导联)')
parser.add_argument('--acl-eta1', default=0.5, type=float,
                    help='区域内一致项权重 η1')
parser.add_argument('--acl-eta2', default=0.5, type=float,
                    help='区域间判别项权重 η2(ACL γ=0.5 当量: η1=η2)')


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
        sizes = [64] + list(map(int, args.projector.split('-')))
        # D1L: 结构化目标矩阵 (lead 序: ii,iii,v1..v6 = 0..7)
        # W1 T1 修正(任务书 07 §2.5/06 §二): τ 属于 cross-correlation 的【对角】目标
        # (同一投影维度跨导联相关), 非对角目标恒 0。目标矩阵 T = I + P 需对称/对角1/PSD。
        self.d1l_P = None
        self.d1l_eigs = None
        if getattr(args, 'd1l', ''):
            parts = [float(v) for v in args.d1l.split(',')]
            P = torch.zeros(args.num_leads, args.num_leads)
            if len(parts) == 1:
                # τ≡v 全矩阵(闭合基线检查用, 如 "1" == B0 / "0" == 跨导完全去相关)
                P.fill_(parts[0])
                P.fill_diagonal_(0.0)
                a = b = parts[0]
            else:
                a, b = parts
                P[0, 1] = P[1, 0] = a          # II-III (Einthoven 相邻)
                for k in range(2, args.num_leads - 1):  # 相邻胸导 V1-V2 ... V5-V6
                    P[k, k + 1] = P[k + 1, k] = b
            if getattr(args, 'd1l_shuffle', False):  # NEG 负对照: 固定种子打乱导联归属
                g = torch.Generator().manual_seed(12345)
                perm = torch.randperm(args.num_leads, generator=g)
                P = P[perm][:, perm]
            T_target = P.clone()
            T_target.fill_diagonal_(1.0)
            eigs = torch.linalg.eigvalsh(T_target)
            # PSD 纪律(任务书 07 §3.2): 结构化先验矩阵必须 PSD。
            # 例外: tau≡1 闭合基线(等价 B0 的逐对对角目标, 全矩阵是路径图邻接+I,
            # 数学上非 PSD 但损失逐对物化、不作为先验矩阵使用)——只记录不拦截。
            if not (a == 1.0 and b == 1.0):
                assert float(eigs.min()) >= -1e-6, \
                    f"D1L 目标矩阵非 PSD: min eig = {float(eigs.min()):.3e}"
            self.d1l_eigs = [round(float(v), 6) for v in eigs]
            self.d1l_P = P.to(self.device)
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
            backbone = VGG16(ch_in=1, alpha=0.125, trc=int(getattr(args, 'trc', 0)))
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
        # W1 T3: multiseg/ccm 复用同一解码器池(masked MSE 见 _masked_recon_loss)
        self.rec_style = getattr(args, 'rec_style', 'none')
        self.d7_decoders = None
        if getattr(args, 'd7_weight', 0.0) > 0 or self.rec_style in ('d7', 'multiseg', 'ccm'):
            assert not self.fast, "D7/重建支路暂不支持 fast 路径"
            from models.decoders import SmallDecoder
            self.d7_decoders = nn.ModuleList(
                [SmallDecoder().to(self.device) for _ in range(args.num_leads)])
        # T2: ACL 区域 projector(仅慢路径; 默认 off 不实例化, B0 主路径零影响)
        self.acl_nets = None
        self.acl_partition = None
        if getattr(args, 'acl_mode', 'off') != 'off':
            assert not self.fast, "ACL 区域支路暂不支持 fast 路径"
            from models.acl_region import RegionProjectors, build_partition
            self.acl_partition = build_partition(getattr(args, 'acl_partition', 'anatomy'),
                                                 getattr(args, 'acl_rand_seed', 101),
                                                 args.num_leads)
            self.acl_nets = RegionProjectors(
                self.acl_partition,
                spec=getattr(args, 'acl_projector', '128-2048-2048-2048')).to(self.device)

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

    def _masked_recon_loss(self, fmap_list, y_orig, mask):
        """T3 B2/B3: masked MSE —— 解码器直接吃 _embed_full 缓存的池化前特征图
        (消除重建路径的第二次骨干前向, 09-20 提速: 融合前 epoch 565s)。
        重建目标为遮挡前原波形, 仅在被掩点计损; mask: (B,1,T) 跨导联同步。
        重建损失监督视图1(mask 仍两视图独立生成, 见 data_utils/ccm.py)。"""
        m = mask[:, :1, :]                       # (B,1,T) 广播到全部导联
        npts = m.sum().clamp(min=1.0)
        rec_loss = 0
        for i, fm in enumerate(fmap_list):
            rec = self.d7_decoders[i](fm)        # (B, 1, T)
            rec_loss = rec_loss + ((rec - y_orig[:, [i], :]).pow(2) * m).sum()
        return rec_loss / (npts * len(fmap_list))

    def _embed_full(self, y):
        """_embed 的融合版本: 额外返回每导联 64 维 GAP 特征(T2)与池化前特征图(T3)。
        算子顺序与 VGG16.forward 完全一致(model[:-1] -> model[-1:] -> view -> fc),
        z 数值与 _embed 逐位相同; 仅 ACL/重建开启时调用以复用主干计算。"""
        z_list, f_list, fm_list = list(), list(), list()
        for i in range(self.args.num_leads):
            bb = self.backbone_group[i]
            fm = bb.model[:-1](y[:, [i], :])
            fm_list.append(fm)
            f = bb.model[-1:](fm).view(fm.shape[0], -1)
            f_list.append(f)
            z_list.append(self.projector_group[i](bb.fc(f)))
        return z_list, f_list, fm_list

    def _embed(self, y):
        """(B, L, T) -> 原始 projector 输出 z 列表 (每导联 (B, d))。"""
        if not self.fast:
            z_list = list()
            for i in range(self.args.num_leads):
                z_list.append(self.projector_group[i](self.backbone_group[i](y[:, [i], :])))
            return z_list
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

    def _nt_xent(self, zi, zj):
        """S1 (W4 A-4): 标准 SimCLR NT-Xent (Chen et al. 2020, Algorithm 1)。

        zi/zj = 同 batch 两视图的 projector 原始输出 (N, d); 正对=同索引配对,
        负对=批内其余 2N-2 视图; L2 归一化 + 温度缩放 + 对角掩码 + 交叉熵。
        """
        N = zi.shape[0]
        z = torch.cat([zi, zj], dim=0)
        z = nn.functional.normalize(z, dim=1)
        sim = z @ z.T / self.args.simclr_temp
        sim.fill_diagonal_(float('-inf'))
        targets = torch.arange(2 * N, device=z.device)
        targets = (targets + N) % (2 * N)
        return nn.functional.cross_entropy(sim, targets)

    def _clocs_diag_loss(self, za, zb):
        """S2 (W4 A-5): CLOCS 官方 obtain_contrastive_loss 的 diag 双向项。

        官方口径(danikiyasseh/CLOCS prepare_miscellaneous.py): s=cos/τ, τ=0.1;
        loss_term1/2 = -mean(log(diag/整行和)), -mean(log(diag/整列和))——
        **分母含正对自身(与 SimCLR 排除自身相反, 审计关键点②)**。
        官方在无患者 id 时 loss_terms=2 即纯本函数; 本仓 NFH 管线无 pid,
        off-diag(同患者跨实例)项不适用=官方"无 pid"自然路径。
        """
        za_n = za / za.norm(dim=1, keepdim=True)
        zb_n = zb / zb.norm(dim=1, keepdim=True)
        e = torch.exp(za_n @ zb_n.T / self.args.clocs_temp)
        diag = torch.diagonal(e)
        t1 = -torch.mean(torch.log(diag / e.sum(dim=1)))
        t2 = -torch.mean(torch.log(diag / e.sum(dim=0)))
        return t1 + t2

    def forward(self, y1, y2, y_pair=None, pair_mask=None,
                hrv_target=None, hrv_valid=None, rec=None):
        self.last_extra = {}
        if self.acl_nets is not None or rec is not None:
            z1_list, f1_list, fm1_list = self._embed_full(y1)
            z2_list, f2_list, fm2_list = self._embed_full(y2)
        else:
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
        if self.args.loss_mode == 'simclr':
            # S1 (W4 A-4): 逐导联独立 SimCLR——正对=同记录同导联两视图, 负对=批内
            # 其余 2N-2 视图; 与 C1 的结构/增强/优化器完全同构(见 baseline_hparams.csv)。
            # NT-Xent 以 L2 归一化+温度实现尺度不变, BT 特有的输出白化 BN 不适用
            # (与 vicreg 分支同模式绕过 bn_group); 跨导联项不适用于 NT-Xent(loss_t=0)。
            loss_r = 0
            for i in range(self.args.num_leads):
                loss_r = loss_r + self._nt_xent(z1_list[i], z2_list[i])
            loss_r = loss_r / self.args.num_leads
            loss_t = torch.zeros_like(loss_r)
            return loss_r, loss_r, loss_t
        if self.args.loss_mode == 'clocs':
            # S2 (W4 A-5): 逐导联框架下的 CLOCS 正对映射——temporal(CMSC 语义)=
            # 同导联两时间窗 (z1[i], z2[i]); spatial(CMLC 语义)=同窗跨导联
            # (z1[i], z1[j]) 与 (z2[i], z2[j]), i<j。每对算官方 diag 双向损失,
            # 按官方归一化 loss=Σ/(2×n_pairs)(loss_terms=2 × ncombinations)。
            # projector 输出直接 L2 归一化(官方无独立 projector, 以编码器输出
            # embedding 参算; 本仓以 C1 等价 projector 输出对齐, 差异入 hparams)。
            L = self.args.num_leads
            temporal, spatial = 0.0, 0.0
            for i in range(L):
                temporal = temporal + self._clocs_diag_loss(z1_list[i], z2_list[i])
            for i in range(L):
                for j in range(i + 1, L):
                    spatial = spatial + self._clocs_diag_loss(z1_list[i], z1_list[j])
                    spatial = spatial + self._clocs_diag_loss(z2_list[i], z2_list[j])
            n_pairs = L + 2 * (L * (L - 1) // 2)
            loss = (temporal + spatial) / (2 * n_pairs)
            loss_r = temporal / (2 * L)
            loss_t = spatial / (2 * (n_pairs - L)) if n_pairs > L else torch.zeros_like(loss)
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
                if self.d1l_P is not None and i != j:
                    # D1L-fix (W1 T1, 任务书 07 §3.2): τ 进入对角目标(跨导同坐标相关),
                    # 非对角目标恒 0(去冗余语义不变)。旧实现把 τ 放非对角 = 主动制造
                    # 投影维度间交叉协方差, 旧 -0.60pt 结果撤回待翻案(06 §二)。
                    tau = self.d1l_P[i, j]
                    on_diag = (torch.diagonal(c) - tau).pow(2).sum()
                    off_diag = off_diagonal(c).pow(2).sum()
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
        if rec is not None:
            # T3 B2/B3 (任务书 07 §4.2): 数据管线提供两视图独立 mask + 原波形目标,
            # masked MSE 按被掩点归一; L = L_BT + η·L_rec(η 即 --d7-weight)。
            # 重建监督视图1(两视图 mask 均独立生成; 解码器吃融合特征图, 09-20 提速)
            loss_bt_term = loss
            o1, o2, m1, m2 = rec
            rec_val = self._masked_recon_loss(fm1_list, o1, m1)
            loss = loss + self.args.d7_weight * rec_val
            loss_r = loss_r + self.args.d7_weight * rec_val
            self.last_extra['loss_rec'] = rec_val.item()
            self._diag_tensors = (loss_bt_term, self.args.d7_weight * rec_val)
        elif getattr(self, 'd7_decoders', None) is not None \
                and self.rec_style in ('d7', 'none'):
            # D7 旧版(B1 归档审计 / 兼容仅 --d7-weight 的旧命令): 单段 50%, 仅视图 1
            rec_d7 = self._d7_recon_loss(y1)
            loss = loss + self.args.d7_weight * rec_d7
            loss_r = loss_r + self.args.d7_weight * rec_d7
            self.last_extra['loss_rec'] = rec_d7.item()
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
        if self.acl_nets is not None:
            # T2 ACL 区域目标(任务书 07 §三): 区域 projector 输入 = 两导联 64 维 GAP concat;
            # L = L_LFBT + η1·L_intra(区域内一致 Eq.10) [+ η2·L_inter(区域间判别 Eq.11)]
            # 负样本只来自批内其他记录, 同一记录其它区域特征不入负样本池(不推远区域)。
            from models.acl_region import acl_losses
            z1r = self.acl_nets(f1_list)
            z2r = self.acl_nets(f2_list)
            l_intra, l_inter = acl_losses(z1r, z2r, tau=self.args.acl_tau)
            acl_term = self.args.acl_eta1 * l_intra
            self.last_extra['acl_intra'] = l_intra.item()
            if self.args.acl_mode == 'full':
                acl_term = acl_term + self.args.acl_eta2 * l_inter
                self.last_extra['acl_inter'] = l_inter.item()
            loss = loss + acl_term
            loss_r = loss_r + acl_term
        return loss, loss_r, loss_t


def main_worker(gpu, args):

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = LeadFusionBT(args)

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

        if getattr(model, 'acl_nets', None) is not None:
            for param in model.acl_nets.parameters():
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
    if getattr(args, 'rec_style', 'none') in ('multiseg', 'ccm'):
        # T3 B2/B3: 遮挡即增强 —— RRC 裁剪保留(aug-params 前两参数), 随机 TO 由
        # CCM/多段 mask 替代; 两视图独立; 附带原波形与 mask 供 masked MSE。
        from data_utils.ccm import CCMDataset
        rpeak_npz = Path(args.rpeak_npz)
        if args.rec_style == 'ccm' and not rpeak_npz.exists():
            raise FileNotFoundError(
                'B3 CCM 需先运行 runlog/prep_rpeaks.py 生成 data/pt_rpeaks.npz')
        crop = tuple(float(v) for v in args.aug_params.split(',')[:2])
        dataset = CCMDataset(args.data_dir, rpeak_npz, style=args.rec_style,
                             ratio=args.rec_ratio, crop=crop)
    elif getattr(args, 'mixup_prob', 0.0) > 0:
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

    start_epoch = 0
    state_path = args.checkpoint_dir / 'train_state.pth'
    if getattr(args, 'resume', False) and state_path.exists():
        from utils.checkpoint import load_torch_checkpoint
        st = load_torch_checkpoint(state_path, map_location='cpu')
        for _name, _sd in st['model'].items():
            _mod = getattr(model, _name)
            if isinstance(_mod, list):
                for _m, _s in zip(_mod, _sd):
                    _m.load_state_dict(_s)
            else:
                _mod.load_state_dict(_sd)
        optimizer.load_state_dict(st['optimizer'])
        if ema_shadows is not None and st.get('ema_shadows') is not None:
            for s_old, s_new in zip(ema_shadows, st['ema_shadows']):
                s_old.copy_(s_new)
        torch.set_rng_state(st['torch_rng'].cpu().to(torch.uint8)
                            if hasattr(st['torch_rng'], 'cpu') else st['torch_rng'])
        start_epoch = st['epoch'] + 1
        print(f"[resume] 恢复自 epoch {st['epoch']}, 从 {start_epoch} 续训", flush=True)

    for epoch in range(start_epoch, args.epochs):
        total_loss = 0
        total_loss_r = 0
        total_loss_t = 0
        ep_start_time = time.time()
        for step, batch in enumerate(loader, start=epoch * len(loader)):
            (views, flag) = batch
            y1 = views[0].to(model.device, non_blocking=True)
            y2 = views[1].to(model.device, non_blocking=True)
            y_pair = pair_mask = hrv_t = hrv_v = None
            rec_info = None
            if len(views) == 4:
                # T3 B2/B3: (masked1, masked2, orig1, orig2) + (mask1, mask2)
                rec_info = (views[2].to(model.device, non_blocking=True),
                            views[3].to(model.device, non_blocking=True),
                            flag[0].to(model.device, non_blocking=True),
                            flag[1].to(model.device, non_blocking=True))
            elif isinstance(flag, (tuple, list)):
                # H4: flag = (targets(B,4), valid(B,))
                hrv_t = flag[0].to(model.device, non_blocking=True)
                hrv_v = flag[1].to(model.device, non_blocking=True)
            elif len(views) == 3:
                # H3 三元组: y_pair=同患者伙伴视图, flag=是否有效
                y_pair = views[2].to(model.device, non_blocking=True)
                pair_mask = flag.to(model.device).float() > 0.5
            optimizer.zero_grad()
            loss, loss_r, loss_t = model.forward(y1, y2, y_pair, pair_mask,
                                                 hrv_target=hrv_t, hrv_valid=hrv_v,
                                                 rec=rec_info)
            grad_ratio = None
            if rec_info is not None and step % args.print_freq == 0 \
                    and getattr(model, '_diag_tensors', None) is not None:
                # T3 诊断(任务书 §4.3): 记录两路梯度范数比 ||∇L_rec||/||∇L_BT||
                # (以 lead-II backbone 为探针, 每 print_freq 步一次, +2 次反向 ~2% 开销)
                try:
                    bt_t, rc_t = model._diag_tensors
                    probe = [p for p in model.backbone_group[0].parameters()]
                    g_bt = torch.autograd.grad(bt_t, probe, retain_graph=True,
                                               allow_unused=True)
                    g_rc = torch.autograd.grad(rc_t, probe, retain_graph=True,
                                               allow_unused=True)
                    n_bt = torch.sqrt(sum(g.pow(2).sum() for g in g_bt if g is not None))
                    n_rc = torch.sqrt(sum(g.pow(2).sum() for g in g_rc if g is not None))
                    grad_ratio = (n_rc / n_bt.clamp(min=1e-12)).item()
                except Exception:
                    grad_ratio = float('nan')
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
                stats.update({k: round(v, 5) for k, v in
                              getattr(model, 'last_extra', {}).items()})
                if grad_ratio is not None:
                    stats['grad_rec_bt'] = round(grad_ratio, 5)
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

        # 断点续训状态: 每 5 epoch 落盘(权重+优化器+EMA+RNG)。
        # 2026-09-21 事故修正: GPU 张量直接 torch.save 在多进程并发下病态慢(三实例互踩 2.5h 未完成一次),
        # 改为先整体 .cpu() 拷贝再序列化(CPU-only 字节流, D2H 一次性完成); 原子替换防半写。
        import os as _os

        def _to_cpu(sd):
            if isinstance(sd, list):
                return [{k: v.cpu() if torch.is_tensor(v) else v for k, v in s.items()} for s in sd]
            return {k: v.cpu() if torch.is_tensor(v) else v for k, v in sd.items()}

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            _snap = {}
            for _name in ('backbone_group', 'projector_group', 'bn_group', 'p_vgg', 'p_proj',
                          'bn_all', 'acl_projectors', 'hrv_head', 'd7_decoders'):
                _mod = getattr(model, _name, None)
                if _mod is None or (isinstance(_mod, list) and len(_mod) == 0):
                    continue
                _snap[_name] = _to_cpu([m.state_dict() for m in _mod] if isinstance(_mod, list)
                                       else _mod.state_dict())
            _opt = optimizer.state_dict()
            _opt['state'] = {k: {k2: (v2.cpu() if torch.is_tensor(v2) else v2)
                                 for k2, v2 in v.items()} for k, v in _opt['state'].items()}
            _opt['param_groups'] = _opt['param_groups']
            _st = {'epoch': epoch, 'model': _snap, 'optimizer': _opt,
                   'torch_rng': torch.get_rng_state()}
            if ema_shadows is not None:
                _st['ema_shadows'] = [s.detach().cpu().clone() for s in ema_shadows]
            _tmp = state_path.with_suffix('.tmp')
            torch.save(_st, _tmp)
            _os.replace(_tmp, state_path)

        if getattr(args, 'rec_style', 'none') == 'ccm':
            # T3 诊断: 回退率(init 抽样估计; 逐视图计数在 worker 侧不回传)
            print(json.dumps({'epoch': epoch, 'ccm_stats': {
                'init_fallback_rate': getattr(dataset, 'init_fallback', None)}}))

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
        with open_out(args.checkpoint_dir, "sinc_bands.json") as f:
            json.dump(bands, f, indent=1)
        print("Sinc bands exported:", args.checkpoint_dir / "sinc_bands.json")

    # T1 工件(任务书 07 §T1): target_matrix.npy + 特征值 + 配置 JSON
    if model.d1l_P is not None:
        import numpy as _np
        T_full = model.d1l_P.cpu().clone()
        T_full.fill_diagonal_(1.0)
        _np.save(safe_out_path(args.checkpoint_dir, "target_matrix.npy"), T_full.numpy())
        with open_out(args.checkpoint_dir, "d1l_check.json") as f:
            json.dump(dict(d1l=args.d1l, shuffle=bool(args.d1l_shuffle),
                           eigenvalues=model.d1l_eigs,
                           min_eig=min(model.d1l_eigs),
                           symmetric=True), f, indent=1)
        print("D1L-fix artifacts saved: target_matrix.npy / d1l_check.json")

    # 环境与配置记录 (指南 §6-6; W1 T0 增补 git_sha/host/dataset_hash/preprocess_version)
    def _git_sha():
        for cand in (Path("runlog/GIT_SHA.txt"), Path("../runlog/GIT_SHA.txt")):
            if cand.exists():
                return cand.read_text(encoding="utf-8").strip()
        return "unknown"

    def _dataset_hash():
        m = Path(args.data_dir).parent / "manifest.json"
        if m.exists():
            return hashlib.sha256(m.read_bytes()).hexdigest()[:16]
        return "no-manifest"

    info = dict(
        args=vars(args),
        python=platform.python_version(),
        torch=torch.__version__,
        torchvision=torchvision_version(),
        cuda_build=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        host=platform.node(),
        git_sha=_git_sha(),
        dataset_hash=_dataset_hash(),
        preprocess_version="v1-frozen-2026-09-17",
        seed=args.seed,
        checkpoint_sha256=sha,
        pretrain_dataset="PTB-XL folds 1-8 (替代 NFH, 非论文原始设置)",
    )
    if model.acl_partition is not None:
        info["acl_partition"] = [list(r) for r in model.acl_partition]
    with open_out(args.checkpoint_dir, "config.json") as f:
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
