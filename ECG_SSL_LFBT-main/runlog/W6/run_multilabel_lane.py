# -*- coding: utf-8 -*-
"""W6 Stage 7: CPSC 多标签评估 — 冻结 encoder + BCE 多标签 LP 头(协议 w6-multilabel)。

真值口径(限制, 如实入账): A 机仅存修正分区的主类映射(one-hot), CPSC 原始多标签
引用(.hea/Refs)在 W5A 重建后未留存——BCE 头在 one-hot 真值上训练与评估,
连续概率 macro-AUROC/AUPRC(one-vs-rest, 单标签真值下有效) + val 选 per-class
阈值后 test 只评一次(per-class F1/Sens/Spec, utils/multilabel.py=E004 工具)。
与单标签主表分列, 不混写不替换主结论。

组合: {b0(s0=ptxl_gamma08,s2,s4), c1(s0,2,4), c2(s0,2,4)} × data/cpsc(修正分区)。
头 = Linear(512→9)+sigmoid, 全批 Adam 1e-3 100ep, val macro-AUROC 选 checkpoint
(与 LP 协议同型: 冻结特征上的线性头, val 选点 test 一次)。
产物: runlog/W6/multilabel_results.csv + runlog/W6/multilabel/{kind}_s{seed}/(阈值+明细)。
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run_robustness import load_test, build_lp_encoders, lp_features, ckpt_sha  # noqa: E402
from utils.multilabel import (compute_multilabel_scores, select_thresholds,  # noqa: E402
                              apply_thresholds, save_multilabel)

CLS = ["AF", "IAVB", "LBBB", "NSR", "PAC", "PVC", "RBBB", "STD", "STE"]  # 与目录序(排序后)对齐由 load_test 保证
OUT = ROOT / "runlog/W6/multilabel_results.csv"
ART = ROOT / "runlog/W6/multilabel"
PAIRS = [("b0", 0), ("b0", 2), ("b0", 4), ("c1", 0), ("c1", 2), ("c1", 4),
         ("c2", 0), ("c2", 2), ("c2", 4)]


def _macro_spec(y_true, y_prob, th):
    """one-hot 真值下的 per-class specificity 宏平均(1-负类检出为阳性)。"""
    yt = np.eye(len(CLS))[y_true]
    specs = []
    for j, c in enumerate(CLS):
        neg = yt[:, j] == 0
        if neg.any():
            specs.append(1.0 - float((y_prob[neg, j] >= th[c]).mean()))
    return f"{np.mean(specs):.4f}"


def build_encs(kind, seed):
    """同 run_robustness.build_lp_encoders 但支持 b0 seed0=ptxl_gamma08。"""
    from models.vgg_1d import VGG16
    ckdir = enc_dir(kind, seed)
    params = torch.load(ckdir / "encoder_group.pth", map_location="cuda", weights_only=True)
    trc = {"c2": 1, "c1": 0, "b0": 0}[kind]
    encs = []
    for i in range(8):
        enc = VGG16(ch_in=1, alpha=0.125, blur_pool=0, pool_power=0.0, trc=trc)
        sd = params["backbone_state_dict_list"][i]
        missing, unexpected = enc.load_state_dict(sd, strict=False)
        assert missing == ["fc.weight", "fc.bias"] and unexpected == [], (kind, seed)
        encs.append(torch.nn.Sequential(*list(enc.children())[:-1]).cuda().eval())
    return encs, ckdir


def log(m):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}", flush=True)


def git_sha():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def enc_dir(kind, seed):
    if (kind, seed) == ("b0", 0):
        return ROOT / "checkpoint/ptxl_gamma08"
    return ROOT / f"checkpoint/confirm/{kind}_seed{seed}"


def load_split(split):
    if split == "test":
        x, y, sha, _ = load_test("cpsc")
    else:
        x, y, sha = _load_trval(split)
    return x, y, sha


def _load_trval(split):
    import os
    root = ROOT / "data/cpsc" / split
    classes = sorted(d for d in os.listdir(root) if (root / d).is_dir())
    xs, ys = [], []
    for ci, c in enumerate(classes):
        for f in sorted(os.listdir(root / c)):
            if f.endswith(".npy"):
                xs.append(np.load(root / c / f))
                ys.append(ci)
    return np.stack(xs), np.array(ys), None


def train_head(ftr, ytr, fva, yva, nc=9, epochs=100, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    head = nn.Linear(ftr.shape[1], nc).cuda()
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    yt = torch.eye(nc, device="cuda")[torch.as_tensor(ytr, device="cuda")]
    xv = torch.as_tensor(fva, device="cuda")
    best, best_sd = -1.0, None
    for ep in range(epochs):
        head.train()
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(head(ftr), yt)
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            prob = torch.sigmoid(head(xv)).cpu().numpy()
        s = compute_multilabel_scores(yva, prob, CLS)
        au = s["macro_auroc"]
        if au > best:
            best, best_sd = au, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_sd)
    return head, best


def main():
    xtr, ytr, _ = load_split("train")
    xva, yva, _ = load_split("val")
    xte, yte, sha = load_split("test")
    log(f"cpsc 修正分区: train {len(ytr)} / val {len(yva)} / test {len(yte)} (data_sha={sha})")
    new = not OUT.exists()
    for kind, seed in PAIRS:
        encs, ckdir = build_encs(kind, seed)
        ftr = lp_features(encs, torch.from_numpy(xtr).cuda()).cuda()
        fva = lp_features(encs, torch.from_numpy(xva).cuda()).cuda()
        fte = lp_features(encs, torch.from_numpy(xte).cuda()).cuda()
        head, val_au = train_head(ftr, ytr, fva, yva, seed=seed)
        with torch.no_grad():
            pva = torch.sigmoid(head(fva)).cpu().numpy()
            pte = torch.sigmoid(head(fte)).cpu().numpy()
        th = select_thresholds(yva, pva, CLS)
        s = compute_multilabel_scores(yte, pte, CLS)
        t = apply_thresholds(yte, pte, CLS, th)  # macro_recall=macro_sens; spec 由 one-hot 反类算
        art = ART / f"{kind}_s{seed}"
        save_multilabel(art, yte, pte, CLS, thresholds=th, metadata={
            "protocol_id": "w6-multilabel", "kind": kind, "seed": seed,
            "val_macro_auroc": val_au, "data_sha": sha,
            "truth": "primary-class one-hot(原始多标签真值未留存, 见脚本头)"})
        with open(OUT, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts", "ckpt", "seed", "macro_auroc", "macro_auprc",
                            "macro_f1", "macro_sens", "macro_spec",
                            "protocol_id", "git_sha", "checkpoint_sha256", "note"])
                new = False
            w.writerow([time.strftime("%Y-%m-%d %H:%M"), kind, seed,
                        f"{s['macro_auroc']:.4f}", f"{s['macro_auprc']:.4f}",
                        f"{t['macro_f1']:.4f}", f"{t['macro_recall']:.4f}",
                        _macro_spec(yte, pte, th),
                        "w6-multilabel", git_sha(),
                        ckpt_sha(kind, seed)[:12] if (kind, seed) != ("b0", 0) else "026033d532d4",
                        "one-hot真值(原始多标签未留存)"])
        log(f"  + {kind} s{seed}: auroc {s['macro_auroc']:.4f} auprc {s['macro_auprc']:.4f} "
            f"f1 {t['macro_f1']:.4f} (val {val_au:.4f})")
    log("MULTILABEL_DONE")


if __name__ == "__main__":
    main()
