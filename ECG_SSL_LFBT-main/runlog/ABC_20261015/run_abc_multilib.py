# -*- coding: utf-8 -*-
"""run_abc_multilib.py — ABC_20261015 P1-3 CPSC 多标签敏感性(方案 P1-3, protocol_frozen §8)。

口径 = W6 run_multilabel_lane.py 冻结协议逐字移植, 仅扩组合表:
  M0(c1)/M1(c2)/M2(dv)/M3/M4/M6/M7/M8/M9 × seeds{0,2,4} × data/cpsc(修正分区)。
真值限制(如实入账, 与 W6 同): A 机仅存修正分区主类映射(one-hot), BCE 头在 one-hot
真值上训练与评估; 连续概率 macro-AUROC/AUPRC(one-vs-rest) + val 选 per-class 阈值
test 只评一次。主分析仍为单标签口径, 本项为敏感性分析, 不替换主结果。
幂等: 已有 csv 行跳过。产物 multilib_results.csv + multilib/{model}_s{seed}/。
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

from run_robustness import load_test, lp_features  # noqa: E402
from utils.multilabel import (compute_multilabel_scores, select_thresholds,  # noqa: E402
                              apply_thresholds, save_multilabel)

CLS = ["AF", "IAVB", "LBBB", "NSR", "PAC", "PVC", "RBBB", "STD", "STE"]
OUTD = ROOT / "runlog/ABC_20261015"
OUT = OUTD / "multilib_results.csv"
ART = OUTD / "multilib"
SEEDS = [0, 2, 4]
# 模型 -> (seed0-2 ckpt 目录模板, trc); m2 跨批次特判
MODEL_CK = {
    "m0": ("checkpoint/confirm/c1_seed{seed}", 0),
    "m1": ("checkpoint/confirm/c2_seed{seed}", 1),
    "m2": ("checkpoint/w6_dv/seed0|checkpoint/w7_dv/seed{seed}", 0),
    "m3": ("checkpoint/joint_m3/seed{seed}", 1),
    "m4": ("checkpoint/joint_m4_noise/seed{seed}", 1),
    "m6": ("checkpoint/joint_m6/seed{seed}", 1),
    "m7": ("checkpoint/abc_supp/m7_bc/seed{seed}", 0),
    "m8": ("checkpoint/abc_supp/m8_a_beat_c/seed{seed}", 1),
    "m9": ("checkpoint/abc_supp/m9_a_rr_c/seed{seed}", 1),
}


def log(m):
    print(f"[multilib {time.strftime('%m-%d %H:%M:%S')}] {m}", flush=True)


def git_sha():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def ck_of(model, seed):
    if model == "m2":
        return ROOT / ("checkpoint/w6_dv/seed0" if seed == 0
                       else f"checkpoint/w7_dv/seed{seed}")
    return ROOT / MODEL_CK[model][0].format(seed=seed)


def build_encs(model, seed):
    from models.vgg_1d import VGG16
    ckdir = ck_of(model, seed)
    params = torch.load(ckdir / "encoder_group.pth", map_location="cuda",
                        weights_only=True)
    trc = MODEL_CK[model][1]
    encs = []
    for i in range(8):
        enc = VGG16(ch_in=1, alpha=0.125, blur_pool=0, pool_power=0.0, trc=trc)
        sd = params["backbone_state_dict_list"][i]
        missing, unexpected = enc.load_state_dict(sd, strict=False)
        assert missing == ["fc.weight", "fc.bias"] and unexpected == [], (model, seed)
        encs.append(torch.nn.Sequential(*list(enc.children())[:-1]).cuda().eval())
    return encs, ckdir


def load_split(split):
    import os
    if split == "test":
        x, y, sha, _ = load_test("cpsc")
        return x, y, sha
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
    for _ep in range(epochs):
        head.train()
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(head(ftr), yt)
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            prob = torch.sigmoid(head(xv)).cpu().numpy()
        s = compute_multilabel_scores(yva, prob, CLS)
        if s["macro_auroc"] > best:
            best, best_sd = s["macro_auroc"], {k: v.clone()
                                               for k, v in head.state_dict().items()}
    head.load_state_dict(best_sd)
    return head, best


def _macro_spec(y_true, y_prob, th):
    yt = np.eye(len(CLS))[y_true]
    specs = []
    for j, c in enumerate(CLS):
        neg = yt[:, j] == 0
        if neg.any():
            specs.append(1.0 - float((y_prob[neg, j] >= th[c]).mean()))
    return f"{np.mean(specs):.4f}"


def row_done(model, seed):
    if not OUT.exists():
        return False
    with open(OUT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["ckpt"] == model and r["seed"] == str(seed):
                return True
    return False


def main():
    import hashlib
    OUTD.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    xtr, ytr, _ = load_split("train")
    xva, yva, _ = load_split("val")
    xte, yte, sha = load_split("test")
    log(f"cpsc 修正分区: train {len(ytr)} / val {len(yva)} / test {len(yte)} "
        f"(data_sha={sha})")
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "ckpt", "seed", "macro_auroc", "macro_auprc",
                        "macro_f1", "macro_sens", "macro_spec", "protocol_id",
                        "git_sha", "checkpoint_sha256", "note"])
        for model in ("m0", "m1", "m2", "m3", "m4", "m6", "m7", "m8", "m9"):
            for seed in SEEDS:
                if row_done(model, seed):
                    log(f"{model} s{seed}: 已有账, 跳过")
                    continue
                encs, ckdir = build_encs(model, seed)
                ftr = lp_features(encs, torch.from_numpy(xtr).cuda()).cuda()
                fva = lp_features(encs, torch.from_numpy(xva).cuda()).cuda()
                fte = lp_features(encs, torch.from_numpy(xte).cuda()).cuda()
                head, val_au = train_head(ftr, ytr, fva, yva, seed=seed)
                with torch.no_grad():
                    pva = torch.sigmoid(head(fva)).cpu().numpy()
                    pte = torch.sigmoid(head(fte)).cpu().numpy()
                th = select_thresholds(yva, pva, CLS)
                s = compute_multilabel_scores(yte, pte, CLS)
                t = apply_thresholds(yte, pte, CLS, th)
                cksha = hashlib.sha256(
                    (ckdir / "encoder_group.pth").read_bytes()).hexdigest()
                save_multilabel(ART / f"{model}_s{seed}", yte, pte, CLS,
                                thresholds=th, metadata={
                    "protocol_id": "abc-20261015-multilib", "kind": model,
                    "seed": seed, "val_macro_auroc": val_au, "data_sha": sha,
                    "truth": "primary-class one-hot(原始多标签真值未留存, 见脚本头)"})
                w.writerow([time.strftime("%Y-%m-%d %H:%M"), model, seed,
                            s["macro_auroc"], s["macro_auprc"], t["macro_f1"],
                            t["macro_recall"], _macro_spec(yte, pte, th),
                            "abc-20261015-multilib", git_sha(), cksha[:12],
                            "one-hot真值(W6 口径, 敏感性分析不替换主结果)"])
                f.flush()
                log(f"  + {model} s{seed}: auroc {s['macro_auroc']} "
                    f"auprc {s['macro_auprc']}")
                del encs, ftr, fva, fte, head
                torch.cuda.empty_cache()
    log("multilib 完成")


if __name__ == "__main__":
    main()
