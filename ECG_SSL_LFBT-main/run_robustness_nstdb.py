# -*- coding: utf-8 -*-
"""W6 Stage 6: 真实噪声鲁棒性(MIT-BIH NSTDB) — test-time 扰动, 零重训。

任务书口径: 冻结 b0/c1/c2 + 已存 LP 头, 三域 LP @ SNR{0,5,10,20}dB;
噪声源 = NSTDB 真实记录 data/nstdb/{bw,ma,em}(wfdb, 2ch@360Hz, 重采样至 FS=204.8);
clean 门先行: ptbxl/chapman 对 W2 账本, cpsc 对 W5A 修正分区账本(runlog/W5/lp_results.csv),
逐位差 |Δ|>5e-4 中止(口径错)。只评不改训。

复用 W3 run_robustness.py 的数据加载/编码器/头/metrics(_match_snr/_rng_for/load_test/
build_lp_encoders/lp_features/metrics_from_probs), 仅噪声生成与组合表为 W6 新增;
W3 文件只读不改。输出 runlog/W6/robustness_real_{b0,c1,c2}.csv(schema 同 W3)。

用法:
  DL_PY run_robustness_nstdb.py --stage clean   # 只跑 clean 复现门
  DL_PY run_robustness_nstdb.py --stage full    # clean 门过后全量 3噪声x4SNR
  DL_PY run_robustness_nstdb.py --stage smoke --limit 96
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import run_robustness as R3
from run_robustness import (FS, ROOT, _match_snr, _rng_for, load_test, lp_features,
                            metrics_from_probs, ckpt_sha, log, git_sha)
from models.vgg_1d import VGG16
from models.linear import LinearClassifier

OUT_DIR = ROOT / "runlog/W6"
NSTDB = ROOT / "data/nstdb"
TRC_OF = {"c2": 1, "c1": 0, "b0": 0}
NUM_CLASSES = {"ptbxl": 5, "cpsc": 9, "chapman": 4}

# 组合表: LP 头位置 (W2=results/confirm; W5A cpsc 头=runlog/W5/feat); b0 s0 encoder=ptxl_gamma08
COMBOS = [
    ("ptbxl", [("b0", 2), ("b0", 4), ("c1", 0), ("c1", 2), ("c1", 4),
               ("c2", 0), ("c2", 2), ("c2", 4)], "w2"),
    ("cpsc", [("b0", 0), ("b0", 2), ("b0", 4), ("c1", 0), ("c1", 2), ("c1", 4),
              ("c2", 0), ("c2", 2), ("c2", 4)], "w5"),
    ("chapman", [("c1", 0), ("c1", 2), ("c1", 4), ("c2", 0), ("c2", 2), ("c2", 4)], "w2"),
]
COND_REAL = [("nstb_" + n, s) for n in ("bw", "ma", "em") for s in ("0", "5", "10", "20")]

_NOISE_CACHE = {}


def load_nstdb(name):
    """name in {bw,ma,em} -> (2, Ln) float32 @204.8Hz(从 360Hz resample_poly)。"""
    if name not in _NOISE_CACHE:
        import wfdb
        from scipy.signal import resample_poly
        rec = wfdb.rdrecord(str(NSTDB / name))
        x = rec.p_signal.T.astype(np.float32)          # (2, 650000) @360Hz
        x = resample_poly(x, up=2048, down=3600, axis=1)  # 360->204.8 Hz (×512/900)
        _NOISE_CACHE[name] = x
        log(f"NSTDB {name}: {rec.p_signal.shape[0]}样本@360Hz -> {x.shape[1]}@204.8Hz")
    return _NOISE_CACHE[name]


def enc_dir(kind, seed):
    if (kind, seed) == ("b0", 0):
        return ROOT / "checkpoint/ptxl_gamma08"
    return ROOT / f"checkpoint/confirm/{kind}_seed{seed}"


def build_lp(kind, seed, ds, ref_epoch):
    ckdir = enc_dir(kind, seed)
    params = torch.load(ckdir / "encoder_group.pth", map_location="cuda", weights_only=True)
    encs = []
    for i in range(8):
        enc = VGG16(ch_in=1, alpha=0.125, blur_pool=0, pool_power=0.0, trc=TRC_OF[kind])
        sd = params["backbone_state_dict_list"][i]
        missing, unexpected = enc.load_state_dict(sd, strict=False)
        assert missing == ["fc.weight", "fc.bias"] and unexpected == [], (kind, seed, missing)
        encs.append(torch.nn.Sequential(*list(enc.children())[:-1]).cuda().eval())
    head_path = (ROOT / f"runlog/W5/feat/{kind}_{ds}_seed{seed}/classifier_best_ckpt.pth"
                 if ref_epoch == "w5"
                 else ROOT / f"results/confirm/{kind}_{ds}_seed{seed}/classifier_best_ckpt.pth")
    head = LinearClassifier(feat_dim=512, num_classes=NUM_CLASSES[ds]).cuda().eval()
    head.load_state_dict(torch.load(head_path, map_location="cuda", weights_only=True))
    return encs, head, ckdir


def make_perturbed_real(x, ds, name, intensity):
    """NSTDB 加性噪声: 每 (ds,name,intensity) 独立 rng; 每记录每导联取随机段,
    通道映射 lead->ch[lead%2]; SNR 口径同 W3(_match_snr 逐记录逐导联)。"""
    noise = load_nstdb(name.split("_")[1])             # (2, Ln)
    Ln = noise.shape[1]
    L = x.shape[2]
    rng = _rng_for(ds, name, intensity)
    offs = rng.integers(0, Ln - L, size=(x.shape[0],))
    ch = np.array([i % 2 for i in range(x.shape[1])])
    n = np.stack([noise[ch, o:o + L] for o in offs])   # (N,8,L)
    return x + _match_snr(x, n, float(intensity))


def refs(ds, ref_epoch):
    ref = {}
    if ref_epoch == "w5":
        with open(ROOT / "runlog/W5/lp_results.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ref[(r["ckpt"], r["seed"], "lp", r["downstream"])] = (
                    float(r["auroc"]), float(r["auprc"]))
    else:
        with open(ROOT / "runlog/W2/confirm_results.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ref[(r["ckpt"], r["seed"], r["eval"], r["downstream"])] = (
                    float(r["auroc"]), float(r["auprc"]))
    return ref


def append_row(model_csv, row):
    new = not model_csv.exists()
    with open(model_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "dataset", "eval", "seed", "perturbation", "intensity",
                        "auroc", "auprc", "clean_auroc", "clean_auprc",
                        "delta_auroc_pt", "delta_auprc_pt",
                        "n_test", "git_sha", "checkpoint_sha256", "data_sha", "noise_source"])
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="full", choices=["clean", "full", "smoke"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gsha = git_sha()
    smoke = args.stage == "smoke"
    conds = ([("clean", "1.0")] if args.stage == "clean" else
             [("clean", "1.0"), ("nstb_bw", "10"), ("nstb_em", "5")] if smoke
             else [("clean", "1.0")] + COND_REAL)
    limit = args.limit or (96 if smoke else None)
    log(f"stage={args.stage} conds={len(conds)} git={gsha}")

    for ds, pairs, ref_epoch in COMBOS:
        nc = NUM_CLASSES[ds]
        x, y, data_sha, n_cls = load_test(ds, limit)
        assert n_cls == nc, (ds, n_cls, nc)
        ref = refs(ds, ref_epoch)
        log(f"== {ds}/lp: N={len(y)} classes={nc} ref={ref_epoch} pairs={len(pairs)}")
        models, shas, clean_m = {}, {}, {}
        for kind, seed in pairs:
            encs, head, ckdir = build_lp(kind, seed, ds, ref_epoch)
            models[(kind, seed)] = (encs, head)
            shas[(kind, seed)] = ckpt_sha(kind, seed) if (kind, seed) != ("b0", 0) else "026033d532d4"
        for name, inten in conds:
            xp = x.copy() if name == "clean" else make_perturbed_real(x, ds, name, inten)
            xp_t = torch.from_numpy(np.ascontiguousarray(xp)).cuda()
            for kind, seed in pairs:
                encs, head = models[(kind, seed)]
                feat = lp_features(encs, xp_t)
                with torch.no_grad():
                    probs = F.softmax(head(feat.cuda()), dim=1).cpu().numpy()
                au, ap_ = metrics_from_probs(y, probs, nc)
                if name == "clean":
                    clean_m[(kind, seed)] = (au, ap_)
                    key = (kind, str(seed), "lp", ds)
                    if key in ref and not smoke:
                        rau, rap = ref[key]
                        if abs(au - rau) > 5e-4 or abs(ap_ - rap) > 5e-4:
                            log(f"  ✗ CLEAN 门失败 {key}: {au:.4f}/{ap_:.4f} vs 账本 {rau:.4f}/{rap:.4f} -> 中止")
                            return 2
                        log(f"  ✔ clean {key}: {au:.4f}/{ap_:.4f} = 账本({rau:.4f}/{rap:.4f})")
                    else:
                        log(f"  · clean {key}: {au:.4f}/{ap_:.4f} (无账本参照)")
                    continue
                cau, cap = clean_m[(kind, seed)]
                if not smoke:
                    append_row(OUT_DIR / f"robustness_real_{kind}.csv",
                               [time.strftime("%m-%d %H:%M"), ds, "lp", seed, name, inten,
                                f"{au:.4f}", f"{ap_:.4f}", f"{cau:.4f}", f"{cap:.4f}",
                                f"{(au - cau) * 100:+.2f}", f"{(ap_ - cap) * 100:+.2f}",
                                len(y), gsha, shas[(kind, seed)][:12], data_sha,
                                f"NSTDB-{name.split('_')[1]}"])
                log(f"  · {kind} s{seed} {ds} {name}@{inten}: {au:.4f}/{ap_:.4f} "
                    f"(Δauroc {(au - cau) * 100:+.2f}pt)")
            del xp, xp_t
            torch.cuda.empty_cache()
        if args.stage == "clean":
            log(f"== {ds} clean 门通过")
    log("NSTDB_ROBUSTNESS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
