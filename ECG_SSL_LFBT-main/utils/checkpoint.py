from pathlib import Path

import torch

# W2 冻结的 encoder_group.pth 采用双格式保存(run_pt.py §755): 'backbone_state_dict'
# 字段是完整模块对象列表而非纯 state_dict, weights_only=True 的默认白名单不含
# 自定义 nn.Module 类, 2026-09-23 W4 安全门②实测加载失败。此处将该类注册进
# torch 官方 safe-globals 白名单(仍保持 weights_only=True 受限反序列化, 非
# weights_only=False 绕过); 注册是进程级一次性的, 本模块被 import 即生效。
from torch.nn import Identity, Linear, Module
from torch.nn.modules import activation, batchnorm, container, conv, pooling

from models.vgg_1d import GRN1D, VGG16

# 白名单 = 冻结 checkpoint 实际引用的全部自定义/nn 全局类(2026-09-23 对
# b0/c1/c2 全系 encoder_group.pth 做 pickle opcode 静态扫描得出, 见
# runlog/W4/diag_ckpt_globals2.py): 本仓库 VGG16/GRN1D + nn 基础模块 + 基础容器。
# 均为数据结构/网络模块类, 不含任意可调用; 仍保持 weights_only=True 受限加载。
_W4_SAFE_GLOBALS = [
    VGG16, GRN1D,
    container.Sequential,
    conv.Conv1d,
    batchnorm.BatchNorm1d,
    activation.ReLU,
    pooling.MaxPool1d, pooling.AdaptiveAvgPool1d,
    Identity, Linear,
    set, frozenset,
]
torch.serialization.add_safe_globals(_W4_SAFE_GLOBALS)
_SAFE_GLOBALS_REGISTERED = list(_W4_SAFE_GLOBALS)


def register_safe_globals(classes):
    """追加注册可信类(如未来 checkpoint 出现新的自定义模块类型)."""
    torch.serialization.add_safe_globals(list(classes))
    _SAFE_GLOBALS_REGISTERED.extend(classes)


def load_torch_checkpoint(path, map_location="cpu"):
    """Load trusted local checkpoints safely under weights_only=True.

    CWE-502 整改: 拒绝任意 pickle 对象; 冻结 checkpoint 内的 models.vgg_1d.VGG16
    完整模块对象经模块级 add_safe_globals 白名单放行, 其余自定义全局仍被拒绝。
    """
    return torch.load(Path(path), map_location=map_location, weights_only=True)
