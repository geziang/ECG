"""最小单元测试: 验证指南 §5 修复项 (运行: python tests/test_repro.py)"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils.augmentations import RandomResizeCropTimeOut
from data_utils.cls_datasets import get_data_loaders
from models.vgg_1d import VGG16
from prepare_data import superclass_of, load_superclass_map

FAILS = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


def test_path_join_without_trailing_slash():
    # P0: 不带末尾分隔符也能工作
    loaders = get_data_loaders("data/ptbxl", batch_size=32, num_workers=0)
    sizes = [len(l.dataset) for l in loaders]
    check("P0 pathlib join (no trailing slash)", sizes == [13639, 1714, 1739])
    print("     train/val/test sizes:", sizes)


def test_checkpoint_dual_format():
    # P1: checkpoint 双格式 (模块列表 + state_dict), 且 state_dict 可加载
    enc = VGG16(ch_in=1, alpha=0.125)
    enc.fc = torch.nn.Identity()  # 与 run_pt 一致: 预训练前 fc 被替换为 Identity
    sd = {k: v for k, v in enc.state_dict().items()}
    torch.save({'backbone_state_dict': [enc], 'backbone_state_dict_list': [sd]},
               "data/_ckpt_test.pth")
    from utils.checkpoint import load_torch_checkpoint  # import 即注册 safe-globals 白名单
    ckpt = load_torch_checkpoint("data/_ckpt_test.pth", map_location="cpu")
    check("P1 dual keys present",
          'backbone_state_dict' in ckpt and 'backbone_state_dict_list' in ckpt)
    enc2 = VGG16(ch_in=1, alpha=0.125)  # 带 fc=Linear(64,1000), 与 run_lp 加载路径一致
    missing, unexpected = enc2.load_state_dict(ckpt['backbone_state_dict_list'][0], strict=False)
    check("P1 state_dict load ok (missing only fc)",
          missing == ['fc.weight', 'fc.bias'] and unexpected == [])
    os.remove("data/_ckpt_test.pth")


def test_augmentation_no_inplace():
    # P2: 增强不原地修改输入
    rng = np.random.RandomState(0)
    x = rng.randn(8, 2048).astype(np.float32)
    x_copy = x.copy()
    t = RandomResizeCropTimeOut()
    _ = t(x)
    _ = t(x)
    check("P2 augmentation no in-place", np.array_equal(x, x_copy))


def test_superclass_mapping():
    diag_map = load_superclass_map("ptb-xl/scp_statements.csv")
    check("labels: single MI", superclass_of("{'IMI': 100.0, 'SR': 0.0}", diag_map) == "MI")
    check("labels: multi excluded", superclass_of("{'IMI': 100.0, 'ISCAL': 80.0}", diag_map) is None)
    check("labels: NORM", superclass_of("{'SR': 0.0}", diag_map) == "NORM")


def test_random_split_reproducible():
    # P1: random_split 种子固定 -> 可复现
    _, _, _ = get_data_loaders("data/ptbxl", batch_size=32, num_workers=0, train_ratio=0.1, seed=0)
    _, _, _ = get_data_loaders("data/ptbxl", batch_size=32, num_workers=0, train_ratio=0.1, seed=0)
    l1 = get_data_loaders("data/ptbxl", batch_size=32, num_workers=0, train_ratio=0.1, seed=0)[0]
    l2 = get_data_loaders("data/ptbxl", batch_size=32, num_workers=0, train_ratio=0.1, seed=0)[0]
    i1 = [l1.dataset.indices[i] for i in range(50)]
    i2 = [l2.dataset.indices[i] for i in range(50)]
    check("P1 random_split same seed reproducible", i1 == i2)


if __name__ == "__main__":
    test_path_join_without_trailing_slash()
    test_checkpoint_dual_format()
    test_augmentation_no_inplace()
    test_superclass_mapping()
    test_random_split_reproducible()
    if FAILS:
        print("\nFAILED:", FAILS)
        sys.exit(1)
    print("\nALL TESTS PASSED")
