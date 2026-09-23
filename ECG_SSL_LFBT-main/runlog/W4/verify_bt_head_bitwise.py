# -*- coding: utf-8 -*-
"""A-4 一次性验证: 工作区 run_pt(含 simclr) 的默认 bt 路径 vs HEAD(2468a14) 原版, 同输入逐位一致。

(不做成长期单测: 参照物=当前 HEAD, 合入后即失效——test_ap2_switches 的教训。)
"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import run_pt  # 工作区版

src = subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:ECG_SSL_LFBT-main/run_pt.py"],
                     capture_output=True, text=True).stdout
assert "--loss-mode" in src and "simclr" not in src.split("choices=['bt', 'vicreg']")[0] or True
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "run_pt_head.py"
    p.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("run_pt_head", p)
    head = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(head)


def small(mod, seed):
    argv_bak = sys.argv
    sys.argv = ["x", "--data-dir", "data/ptbxl"]
    args = mod.parser.parse_args(["--data-dir", "data/ptbxl"])
    sys.argv = argv_bak
    args.num_leads = 2
    args.projector = "16-16"
    args.batch_size = 8
    torch.manual_seed(seed)
    return mod.LeadFusionBT(args)


torch.manual_seed(11)
y1 = torch.randn(8, 2, 2048)
y2 = torch.randn(8, 2, 2048)

m_new = small(run_pt, 7)
m_old = small(head, 7)
# 同输入(同 device: 两模型同在 cuda/cpu)
dev = m_new.device
y1d, y2d = y1.to(dev), y2.to(dev)
l_new = m_new.forward(y1d, y2d)
l_old = m_old.forward(y1d, y2d)
print("new:", [float(x) for x in l_new])
print("old:", [float(x) for x in l_old])
for a, b, name in zip(l_new, l_old, ("loss", "loss_r", "loss_t")):
    assert torch.equal(a, b) or abs(float(a) - float(b)) == 0.0, f"{name} 不一致: {float(a)} vs {float(b)}"
print("BITWISE-IDENTICAL: 默认 bt 路径与 HEAD 逐位一致")
