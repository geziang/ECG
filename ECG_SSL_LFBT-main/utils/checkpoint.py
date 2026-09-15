from pathlib import Path

import torch


def load_torch_checkpoint(path, map_location="cpu"):
    """Load trusted local checkpoints across PyTorch 2.0-2.5."""
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
