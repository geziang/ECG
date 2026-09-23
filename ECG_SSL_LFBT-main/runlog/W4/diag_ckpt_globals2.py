# -*- coding: utf-8 -*-
"""静态扫描 checkpoint zip 内 data.pkl 的全部 GLOBAL/STACK_GLOBAL 类引用(不执行反序列化)."""
import io
import pickletools
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def scan(pth):
    with zipfile.ZipFile(pth) as z:
        name = [n for n in z.namelist() if n.endswith("data.pkl")][0]
        data = z.read(name)
    globals_ = []
    stack = []
    for op, arg, _pos in pickletools.genops(data):
        if op.name == "GLOBAL" and arg:
            parts = str(arg).replace("\r", "").split("\n")
            if len(parts) >= 2 and parts[0] and parts[1]:
                globals_.append(f"{parts[0]}.{parts[1]}")
    return globals_


for rel in [
    "checkpoint/confirm/c2_seed0/encoder_group.pth",
    "checkpoint/confirm/c2_seed0/train_state.pth",
    "checkpoint/confirm/b0_seed2/encoder_group.pth",
    "checkpoint/confirm/c1_seed0/encoder_group.pth",
]:
    p = ROOT / rel
    if not p.exists():
        print(rel, "-> MISSING")
        continue
    print(rel, "->", sorted(set(scan(p))))
    sys.stdout.flush()
print("DONE")
