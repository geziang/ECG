# -*- coding: utf-8 -*-
"""gpu_smoke_hostB.py — 主机B 新增代码 GPU 冒烟(小 batch, ~1GB, 与在跑训练共存):
验证 H3/C1 在真实训练路径(CUDA)上前向+反向; 默认关 == B0 逐位一致。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch

assert torch.cuda.is_available(), "需要 GPU"
dev = 'cuda'


def make_args(**kw):
    d = dict(num_leads=8, projector='64-64', batch_size=4, gamma=0.8, lambd=0.0051,
             loss_mode='bt', projector_norm='batchnorm', fast_backbone=False,
             d1l='', d1l_shuffle=False, ema_decay=0.0, bt_var_hinge=0.0,
             d7_weight=0.0, d7_mask=0.5, common_weight=0.0,
             h3_weight=0.0, h3_prob=1.0, sinc_frontend=0, sinc_reg=0.01, sinc_kernel=101,
             vicreg_sim=25., vicreg_var=25., vicreg_cov=1., vicreg_var_eps=1e-4, vicreg_keep_bn=False)
    d.update(kw)
    return argparse.Namespace(**d)


import run_pt
from models.sinc_conv import sinc_bands

torch.manual_seed(0)
xb = torch.randn(4, 8, 2048, device=dev)
print("[1] 默认关 == B0", flush=True)
m0 = run_pt.LeadFusionBT(make_args())
l_a, lr_a, lt_a = m0.forward(xb, xb)
l_b, lr_b, lt_b = m0.forward(xb, xb, None, None)
assert torch.equal(l_a, l_b) and torch.equal(lr_a, lr_b) and torch.equal(lt_a, lt_b)
print("    forward(y1,y2) == forward(y1,y2,None,None) 逐位一致 OK", flush=True)
l_a.backward()
print("    B0 路径 backward OK, loss=%.3f" % l_a.item(), flush=True)

print("[2] H3 开(h3_weight=0.1, 全 mask)", flush=True)
m1 = run_pt.LeadFusionBT(make_args(h3_weight=0.1))
l1, _, _ = m1.forward(xb, xb, xb.flip(0), torch.ones(4, dtype=torch.bool, device=dev))
l1.backward()
g = m1.backbone_group[0].model[0][0][0].weight.grad
assert g is not None and torch.isfinite(g).all()
print("    loss=%.3f 有限, 反向 OK" % l1.item(), flush=True)

print("[3] C1 开(sinc_frontend=16)", flush=True)
m2 = run_pt.LeadFusionBT(make_args(sinc_frontend=16))
l2, _, _ = m2.forward(xb, xb)
l2.backward()
sl = m2.backbone_group[0].model[0][0][0]
assert sl.low_hz.grad is not None and torch.isfinite(sl.low_hz.grad).all()
b = sinc_bands(sl)
print("    loss=%.3f 有限, sinc 梯度 OK, f1∈[%.1f,%.1f] f2∈[%.1f,%.1f]"
      % (l2.item(), min(b['low']), max(b['low']), min(b['high']), max(b['high'])), flush=True)

print("[4] H3 数据集(H3TripletDataset) 一个批次", flush=True)
from data_utils.augmentations import RandomResizeCropTimeOut, ToTensor
from torchvision.transforms import Compose
from data_utils.h3_dataset import H3TripletDataset
from data_utils.n3_dataset import load_patient_map
t = Compose([RandomResizeCropTimeOut(params=[0.5, 1.0, 0.0, 0.5]), ToTensor()])
ds = H3TripletDataset("data/pt_pretrain", t, t, load_patient_map(), prob=1.0)
n_with = sum(1 for i in range(0, len(ds), max(1, len(ds) // 300)) if ds.partner_of[i])
(views, flags) = next(iter(torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)))
assert len(views) == 3 and views[0].shape == (4, 8, 2048)
print("    len=%d, 抽样~300条中有同患者伙伴: %d, batch views=3 OK (flag.sum=%.0f)"
      % (len(ds), n_with, flags.sum()), flush=True)

print("ALL GPU SMOKE TESTS PASSED", flush=True)
