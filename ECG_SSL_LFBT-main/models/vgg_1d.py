import math

import torch
import torch.nn as nn


def conv_layer(ch_in, ch_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv1d(ch_in, ch_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm1d(ch_out),
        nn.ReLU()
    )
    return layer


class BlurPool1d(nn.Module):
    """H2 抗混叠下采样(零参数): 固定二项式低通核 depthwise 卷积后接原 MaxPool。"""

    def __init__(self, channels, filt_size=3):
        super().__init__()
        k = torch.tensor([math.comb(filt_size - 1, i) for i in range(filt_size)],
                         dtype=torch.float32)
        k = k / k.sum()
        self.register_buffer('kernel', k.view(1, 1, filt_size).repeat(channels, 1, 1))
        self.channels = channels

    def forward(self, x):
        return nn.functional.conv1d(x, self.kernel, padding=self.kernel.size(-1) // 2,
                                    groups=self.channels)


def power_pool(x, q=3.0, eps=1e-6):
    """T3 广义幂均值池化(带符号稳定版): y = (mean(x^q))^(1/q)。
    通过 maxabs 归一防溢出; q 为奇整数时保号保序。x: [B,C,T] -> [B,C,1]。"""
    m = x.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    s = x / m
    inner = s.pow(q).mean(dim=-1, keepdim=True)
    out = inner.abs().clamp_min(eps).pow(1.0 / q) * inner.sign() * m
    return out


class PowerPool1d(nn.Module):
    """T3 幂均值池化的模块版(零参数): 挂在 model 末尾, 输出 [B,C,1] 与 AdaptiveAvgPool1d 同形。
    2026-09-20 修复: 原实现池化只在 forward 里补, run_lp/run_ft 用 children()[:-1] 直接调
    self.model 取特征, T3 时 model 末尾无池化 -> 特征带时间维 -> LP 的 CE 崩。挂进 Sequential
    后所有取特征路径口径一致; 无参数故 state_dict 键不变, 旧 checkpoint 直接加载。"""

    def __init__(self, q=3.0):
        super().__init__()
        self.q = float(q)

    def forward(self, x):
        return power_pool(x, self.q)


def group_whiten(h, g=8, eps=1e-5):
    """H1 分组白化(零参数): [B,D] 分 g 组, 组内 batch 统计白化(协方差->I)。
    Cholesky 失败时退化为组内标准化。仅预训练 forward 使用, 不入 checkpoint。"""
    B, D = h.shape
    d = D // g
    x = h.view(B, g, d)
    xc = x - x.mean(dim=0, keepdim=True)
    try:
        cov = torch.einsum('bgd,bge->gde', xc, xc) / max(B - 1, 1)
        cov = cov + eps * torch.eye(d, device=h.device, dtype=h.dtype)
        L = torch.linalg.cholesky(cov)
        y = torch.linalg.solve_triangular(L.transpose(-1, -2), xc.permute(1, 2, 0), upper=False)
        return y.permute(2, 0, 1).reshape(B, D)
    except Exception:
        return ((x - x.mean(dim=0, keepdim=True)) /
                (x.std(dim=0, keepdim=True) + eps)).reshape(B, D)


def group_ln(h, g=8):
    """H1 的 LN-NEG: 同分组只做逐样本标准化(不去相关)。
    若 H1 增益来自'去相关'本身, LN-NEG 应明显弱于白化。"""
    B, D = h.shape
    return nn.functional.layer_norm(h.view(B, g, D // g), (D // g,)).view(B, D)


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s, pool_module=None):
    layers = [conv_layer(int(in_list[i]), int(out_list[i]), k_list[i], p_list[i]) for i in range(len(in_list))]
    if pool_module is None:
        layers += [nn.MaxPool1d(kernel_size=pooling_k, stride=pooling_s)]
    else:
        layers += [pool_module]  # H2: BlurPool + MaxPool 组合
    return nn.Sequential(*layers)


class GRN1D(nn.Module):
    """TRC 导联内通道响应重标定 (E001 论文 Eq.14-17, 2026-09-21 忠实移植, W1-C2 comparator)。
    Y = F + gamma*(F*n) + beta, n = F / ||F||_2(通道内时间维 L2); gamma/beta [1,C,1] 零初始化
    -> 开态初始前向与关态逐位一致, 每导联 2C=128 参数(alpha=0.125 时), 8 导联共 1024。
    插入点: block5+maxpool 后、全局池化前 (证据链 E001_2026-09-08 §128 行规格)。"""

    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        r = x.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        n = x / r
        return x + self.gamma * (x * n) + self.beta


class VGG16(nn.Module):
    def __init__(self, ch_in=8, n_classes=1000, alpha=0.5, blur_pool=0, pool_power=0.0, trc=0):
        super(VGG16, self).__init__()
        self.alpha = alpha
        self.blur_pool = int(blur_pool)    # H2: >0 时每个下采样点前加 filt=blur_pool 二项式低通
        self.pool_power = float(pool_power)  # T3: >0 时末端池化换广义幂均值 Q=pool_power
        self.trc = int(trc)                # C2: >0 时 block5 后、GAP 前插 GRN1D(零初始化)
        c = 512 * alpha

        def _pool(c_out):
            if self.blur_pool > 0:
                return nn.Sequential(BlurPool1d(int(c_out), self.blur_pool),
                                     nn.MaxPool1d(kernel_size=2, stride=2))
            return None  # 走 vgg_conv_block 默认路径(与原版逐位一致)

        _blocks = [
            vgg_conv_block([ch_in, 64 * alpha], [64 * alpha, 64 * alpha], [3, 3], [1, 1], 2, 2,
                           pool_module=_pool(64 * alpha)),
            vgg_conv_block([64 * alpha, 128 * alpha], [128 * alpha, 128 * alpha], [3, 3], [1, 1], 2, 2,
                           pool_module=_pool(128 * alpha)),
            vgg_conv_block([128 * alpha, 256 * alpha, 256 * alpha], [256 * alpha, 256 * alpha, 256 * alpha], [3, 3, 3],
                           [1, 1, 1], 2, 2, pool_module=_pool(256 * alpha)),
            vgg_conv_block([256 * alpha, 512 * alpha, 512 * alpha], [512 * alpha, 512 * alpha, 512 * alpha], [3, 3, 3],
                           [1, 1, 1], 2, 2, pool_module=_pool(512 * alpha)),
            vgg_conv_block([512 * alpha, 512 * alpha, 512 * alpha], [512 * alpha, 512 * alpha, 512 * alpha], [3, 3, 3],
                           [1, 1, 1], 2, 2, pool_module=_pool(c)),
        ]
        if self.trc > 0:
            _blocks.append(GRN1D(int(c)))  # C2/TRC: block5+maxpool 后、全局池化前
        if self.pool_power == 0.0:
            _blocks.append(nn.AdaptiveAvgPool1d(1))  # 关态结构与原版逐层一致(state_dict 键名不变)
        else:
            _blocks.append(PowerPool1d(self.pool_power))  # T3: 全局池化入 model, LP/FT 同口径(见类注释)
        self.model = nn.Sequential(*_blocks)
        self.fc = nn.Linear(int(512 * alpha), n_classes)
        self.output_dim = int(512 * alpha)  # 恢复属性(mbn.py/run_ft.py 引用;纯属性声明,不影响任何计算路径)

    def forward(self, x):
        x = self.model(x)  # T3 池化已入 model(PowerPool1d), 关态 AdaptiveAvgPool1d 原位不变
        x = x.view(-1, int(512 * self.alpha))
        x = self.fc(x)
        return x
