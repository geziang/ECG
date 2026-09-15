"""统一种子工具: 保证预训练/下游实验可复现 (指南 §5 P1)。"""
import random

import numpy as np
import torch


def set_seed(seed):
    """固定 Python / NumPy / PyTorch / CUDA 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
