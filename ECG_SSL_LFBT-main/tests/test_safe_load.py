# -*- coding: utf-8 -*-
"""W4 安全门②回归: weights_only=True + safe-globals 白名单加载冻结 checkpoint。

背景(2026-09-23): W2 冻结 encoder_group.pth 的 'backbone_state_dict' 字段为完整
模块对象列表, CWE-502 整改后裸 weights_only=True 加载失败; 修复 = utils/checkpoint.py
模块级 add_safe_globals 注册实际引用的 nn/自定义类。本测保证该白名单不回归。

运行: python tests/test_safe_load.py
"""
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.vgg_1d import GRN1D, VGG16  # noqa: E402
from utils.checkpoint import load_torch_checkpoint  # noqa: E402 (import 即注册白名单)


def test_roundtrip_module_object():
    """含完整 VGG16 对象(与冻结物同构)的 checkpoint 往返加载。"""
    torch.manual_seed(0)
    m = VGG16(ch_in=1, alpha=0.125, trc=1)
    m.grn = GRN1D(64)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "obj.pth"
        torch.save({"backbone_state_dict": [m], "flags": {1, 2}}, p)
        d = load_torch_checkpoint(p)
        assert isinstance(d["backbone_state_dict"][0], VGG16), "VGG16 对象未按白名单还原"
        assert d["flags"] == {1, 2}, "builtins.set 未按白名单还原"


def test_frozen_checkpoint_if_present():
    """若本地存在 W2 冻结 checkpoint, 实测加载(缺文件跳过, 两机均可用)。"""
    p = ROOT / "checkpoint/confirm/c2_seed0/encoder_group.pth"
    if not p.exists():
        print("  (skip: 无冻结 c2_seed0, 本地无冻结物环境)")
        return
    d = load_torch_checkpoint(p)
    assert isinstance(d, dict), "冻结 encoder_group.pth 应为 dict(双格式)"
    mods = d["backbone_state_dict"]
    assert isinstance(mods, list) and all(isinstance(x, VGG16) for x in mods), \
        "backbone_state_dict 应为 VGG16 对象列表"


def test_arbitrary_global_still_rejected():
    """未注册的自定义全局仍被拒绝(白名单不是 weights_only=False 的后门)。"""
    class Evil:  # noqa: D401
        def __reduce__(self):
            return (print, ("should_not_run",))

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "evil.pth"
        torch.save({"x": Evil()}, p)
        try:
            load_torch_checkpoint(p)
        except Exception:
            return
    raise AssertionError("未注册自定义类竟被 weights_only=True 放行, 白名单失效")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {type(e).__name__} {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
