from pathlib import Path

import torch


def load_torch_checkpoint(path, map_location="cpu"):
    """Load trusted local checkpoints (tensor-only state dicts) safely.

    weights_only=True 拒绝任意 pickle 对象(CWE-502 整改); 本仓库 checkpoint
    只含张量/基础类型字典, 两机 torch 2.0/2.5 均支持该参数。
    """
    return torch.load(Path(path), map_location=map_location, weights_only=True)
