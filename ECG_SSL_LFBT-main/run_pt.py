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


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class LeadFusionBT(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.backbone_group = list()
        for i in range(args.num_leads):
            backbone = VGG16(ch_in=1, alpha=0.125)
            backbone.fc = nn.Identity()
            self.backbone_group.append(backbone.to(self.device))

        sizes = [64] + list(map(int, args.projector.split('-')))
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
        z1_list = list()
        z2_list = list()
        for i in range(self.args.num_leads):
            z1_list.append(self.projector_group[i](self.backbone_group[i](y1[:, [i], :])))
            z2_list.append(self.projector_group[i](self.backbone_group[i](y2[:, [i], :])))
        if self.args.loss_mode == 'vicreg':
            # D9: bn 语义交给 variance hinge; keep-bn 开关可保留输出 BN(whitening-lite 消融)
            if self.args.vicreg_keep_bn:
                z1_list = [self.bn_group[i](z1_list[i]) for i in range(self.args.num_leads)]
                z2_list = [self.bn_group[i](z2_list[i]) for i in range(self.args.num_leads)]
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
        loss_r = 0
        loss_t = 0
        for i in range(self.args.num_leads):
            for j in range(self.args.num_leads):
                c = self.bn_group[i](z1_list[i]).T @ self.bn_group[j](z2_list[j])
                c.div_(self.args.batch_size)
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
        return loss, loss_r, loss_t


def main_worker(gpu, args):

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = LeadFusionBT(args)

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

    parameters = param_weights + param_biases
    optimizer = optim.Adam(parameters, lr=args.learning_rate)

    t = transforms.Compose([
        RandomResizeCropTimeOut(),
        ToTensor()
    ])
    dataset = ECGDatasetFolder(args.data_dir, transform=MultiViewDataInjector([t, t]))
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

    # 双格式保存: 兼容旧版(模块对象列表) + 纯 state_dict (跨版本稳定, 指南 §5 P1)
    ckpt_path = args.checkpoint_dir / 'encoder_group.pth'
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
