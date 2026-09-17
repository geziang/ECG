from pathlib import Path
import argparse
import hashlib
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
        self.d1l_P = None
        if getattr(args, 'd1l', ''):
            a, b = map(float, args.d1l.split(','))
            P = torch.zeros(args.num_leads, args.num_leads)
            P[0, 1] = P[1, 0] = a          # II-III (Einthoven 相邻)
            for k in range(2, args.num_leads - 1):  # 相邻胸导 V1-V2 ... V5-V6
                P[k, k + 1] = P[k + 1, k] = b
            if getattr(args, 'd1l_shuffle', False):  # NEG 负对照: 固定种子打乱导联归属
                g = torch.Generator().manual_seed(12345)
                perm = torch.randperm(args.num_leads, generator=g)
                P = P[perm][:, perm]
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
            backbone = VGG16(ch_in=1, alpha=0.125)
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

    def forward(self, y1, y2):
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
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                if self.d1l_P is not None and i != j:
                    tau = self.d1l_P[i, j]
                    off_diag = (off_diagonal(c) - tau).pow(2).sum()
                else:
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
    dataset = ECGDatasetFolder(args.data_dir, transform=MultiViewDataInjector([t, t2]))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True,
        pin_memory=True)

    for epoch in range(0, args.epochs):
        total_loss = 0
        total_loss_r = 0
        total_loss_t = 0
        ep_start_time = time.time()
        for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.to(model.device, non_blocking=True)
            y2 = y2.to(model.device, non_blocking=True)
            optimizer.zero_grad()
            loss, loss_r, loss_t = model.forward(y1, y2)
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


def main():
    args = parser.parse_args()
    set_seed(args.seed)
    print("Pretraining Setting ======================================")
    print(args)
    print("==========================================================")
    main_worker(0, args)


if __name__ == '__main__':
    main()
