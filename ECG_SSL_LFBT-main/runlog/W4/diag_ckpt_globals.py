# -*- coding: utf-8 -*-
"""安全门②诊断: 冻结 checkpoint 内 pickle 的自定义全局类清单(只读, 不改任何冻结物)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CKPTS = [
    "checkpoint/confirm/c2_seed0/encoder_group.pth",
    "checkpoint/confirm/c2_seed0/encoder_best.pth",
    "checkpoint/confirm/b0_seed2/encoder_group.pth",
    "checkpoint/confirm/c1_seed0/encoder_group.pth",
    "results/confirm/c2_ptbxl_seed0/classifier_best_ckpt.pth",
]


def walk(o, seen, depth=0):
    mod = type(o).__module__
    if mod not in ("builtins",) and not mod.startswith("torch") and mod != "collections" and mod != "ordereddict":
        seen.add(f"{mod}.{type(o).__name__}")
    if depth > 5:
        return
    if isinstance(o, dict):
        for v in o.values():
            walk(v, seen, depth + 1)
    elif isinstance(o, (list, tuple)):
        for v in o:
            walk(v, seen, depth + 1)


for rel in CKPTS:
    p = ROOT / rel
    if not p.exists():
        print(rel, "-> MISSING")
        continue
    d = torch.load(p, map_location="cpu", weights_only=False)  # 诊断: 本地可信冻结物, 只读
    s = set()
    walk(d, s)
    print(rel, "->", type(d).__name__, sorted(s))
    sys.stdout.flush()
print("DONE")
