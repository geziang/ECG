"""run_robustness.py — A-2 冻结 checkpoint 输入退化评估 (任务书 2026-09-22 §A-2, 预注册)

目的: 用 W2 冻结的 B0/C1/C2 encoder + W2 已保存的下游头(LP 线性头/FT10 全模型),
      在同一 test 记录上做 test-time 输入扰动, 量化相对 clean 的退化。
纪律: 零重训、零新损失、不改 W2 任何文件; 每 (kind,seed,ds) 先评 clean,
      与 runlog/W2/confirm_results.csv 4 位小数核对, |Δ|>0.0005 即中止(口径错)。

预注册扰动套件(全部加在 z-score 后的 (8,2048) 波形上; 有效 fs=204.8 Hz:
500 Hz×10s 5000 样本重采样至 2048):
  A. 加性噪声 × SNR{0,5,10,20} dB, SNR 按每记录每导联信号功率定义:
     - bw     基线漂移 = 0.2/0.5/1.0 Hz 三正弦混合, 每导联独立随机相位
     - emg    肌电 = 20–100 Hz butterworth(阶4, filtfilt) 带通白噪声
     - pl50 / pl60 工频正弦, 每导联随机初相
  B. 连续导联质量衰减(区别于 W2 的离散置零):
     - qual_amp   全导联幅度缩放 ×{0.8,0.6,0.4,0.2}
     - qual_noise 全导联宽带白噪声 SNR{20,15,10,5} dB
  C. adjacent lead-swap(仅 test-time): swap01..swap67 相邻下标对交换(II,III,V1..V6)
     + swap_roll 循环左移 1(全导联同时被相邻导联替换的上界情形)

噪声种子只依赖 (dataset, perturbation, intensity) -> 所有模型/种子看到同一扰动实例(配对比较)。
评估组合(只用 W2 已存下游头, 不补训):
  LP  ptbxl: b0(s2,4) c1/c2(s0,2,4)   头=results/confirm/{k}_{ds}_seed{s}/classifier_best_ckpt.pth
  LP  cpsc : c1/c2(s0,2,4)            (b0 无 W2 cpsc 账, 不补; 纪律: 未登记训练不启动)
  FT10 ptbxl: b0(s2,4) c1/c2(s0,2,4)  头=ft_models/confirm/{k}_seed{s}/ft_best_ckpt.pth
  chapman 跳过(AUROC≈0.996 近天花板, 血缘域最低优先级, 见 README)。
输出: runlog/W3/robustness_{b0,c1,c2}.csv

用法:
  DL_PY run_robustness.py --stage clean     # 只跑 clean 复现门
  DL_PY run_robustness.py --stage full      # clean 门过后全量扰动(默认)
  DL_PY run_robustness.py --stage smoke --limit 256   # 快速口径自检
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*")

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import butter, filtfilt
from sklearn.metrics import roc_auc_score, average_precision_score

from models.linear import LinearClassifier
from models.mbn import MultiBranchNet
from models.vgg_1d import VGG16

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "runlog/W3"
LEDGER = ROOT / "runlog/W2/confirm_results.csv"
FS = 204.8          # 有效采样率: 500Hz 10s(5000 样本) 重采样至 2048
SIGMA_NYQ = FS / 2
TRC_OF = {"c2": 1, "c1": 0, "b0": 0}
NUM_CLASSES = {"ptbxl": 5, "cpsc": 9}
LP_SEEDS = {"ptbxl": {"b0": (2, 4), "c1": (0, 2, 4), "c2": (0, 2, 4)},
            "cpsc": {"b0": (), "c1": (0, 2, 4), "c2": (0, 2, 4)}}
FT_SEEDS = {"b0": (2, 4), "c1": (0, 2, 4), "c2": (0, 2, 4)}
LEADS = ["II", "III", "V1", "V2", "V3", "V4", "V5", "V6"]


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)


def git_sha():
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT))
    return r.stdout.strip() or "?"


# ---------------- 扰动套件(预注册, 与模型无关) ----------------

def _rng_for(ds, name, intensity):
    h = hashlib.sha256(f"{ds}|{name}|{intensity}".encode()).hexdigest()[:12]
    return np.random.default_rng(int(h, 16))


def _match_snr(x, n, snr_db):
    """缩放噪声 n 使每记录每导联 mean(n^2)=P_x/10^(snr/10)。x,n: (N,8,L)"""
    px = (x.astype(np.float64) ** 2).mean(axis=2, keepdims=True)
    pn = (n.astype(np.float64) ** 2).mean(axis=2, keepdims=True)
    return (n * np.sqrt(px / (pn * 10 ** (snr_db / 10.0) + 1e-12))).astype(np.float32)


_B, _A = butter(4, [20.0 / SIGMA_NYQ, 100.0 / SIGMA_NYQ], btype="band")
_T = np.arange(2048, dtype=np.float32) / FS


def make_perturbed(x, ds, name, intensity):
    """返回扰动后副本; x: (N,8,2048) float32。clean 原样复制。"""
    n_rec = x.shape[0]
    if name == "clean":
        return x.copy()
    if name == "bw":
        rng = _rng_for(ds, name, intensity)
        ph = rng.uniform(0, 2 * np.pi, size=(n_rec, 8, 3))
        n = (np.sin(2 * np.pi * 0.2 * _T + ph[..., 0:1]) +
             np.sin(2 * np.pi * 0.5 * _T + ph[..., 1:2]) +
             np.sin(2 * np.pi * 1.0 * _T + ph[..., 2:3])).astype(np.float32)
        n = np.broadcast_to(n, x.shape)
        return x + _match_snr(x, n, float(intensity))
    if name == "emg":
        rng = _rng_for(ds, name, intensity)
        w = rng.standard_normal((n_rec * 8, 2048))
        n = filtfilt(_B, _A, w, axis=-1).reshape(n_rec, 8, 2048).astype(np.float32)
        return x + _match_snr(x, n, float(intensity))
    if name in ("pl50", "pl60"):
        f = 50.0 if name == "pl50" else 60.0
        rng = _rng_for(ds, name, intensity)
        ph = rng.uniform(0, 2 * np.pi, size=(n_rec, 8, 1))
        n = np.broadcast_to(np.sin(2 * np.pi * f * _T + ph).astype(np.float32), x.shape)
        return x + _match_snr(x, n, float(intensity))
    if name == "qual_noise":
        rng = _rng_for(ds, name, intensity)
        n = rng.standard_normal(x.shape).astype(np.float32)
        return x + _match_snr(x, n, float(intensity))
    if name == "qual_amp":
        return (x * float(intensity)).astype(np.float32)
    if name == "swap_roll":
        return x[:, list(range(1, 8)) + [0], :].copy()
    if name.startswith("swap"):
        i, j = int(name[4]), int(name[5])
        out = x.copy()
        out[:, [i, j], :] = x[:, [j, i], :]
        return out
    raise ValueError(f"unknown perturbation {name}")


# 预注册条件表: (name, intensity_label)
CONDITIONS = [("clean", "1.0")] + \
    [(n, snr) for n in ("bw", "emg", "pl50", "pl60") for snr in ("0", "5", "10", "20")] + \
    [("qual_amp", s) for s in ("0.8", "0.6", "0.4", "0.2")] + \
    [("qual_noise", s) for s in ("20", "15", "10", "5")] + \
    [(f"swap{i}{i+1}", LEADS[i] + "<->" + LEADS[i + 1]) for i in range(7)] + \
    [("swap_roll", "roll1")]


# ---------------- 数据与模型 ----------------

def load_test(ds, limit=None):
    root = ROOT / "data" / ds / "test"
    classes = sorted(d for d in os.listdir(root) if (root / d).is_dir())
    xs, ys, names = [], [], []
    for ci, c in enumerate(classes):
        for f in sorted(os.listdir(root / c)):
            if f.endswith(".npy"):
                xs.append(np.load(root / c / f))
                ys.append(ci)
                names.append(f"{c}/{f}")
    x = np.stack(xs)
    if limit:  # 仅 smoke 自检: 固定种子随机抽样(类文件夹有序, 截断会单类)
        idx = np.sort(np.random.default_rng(12345).choice(len(ys), size=min(limit, len(ys)), replace=False))
        x, ys, names = x[idx], np.array(ys)[idx], [names[i] for i in idx]
    data_sha = hashlib.sha256(("|".join(names)).encode()).hexdigest()[:12]
    return x, np.array(ys), data_sha, len(classes)


def build_lp_encoders(kind, seed):
    ckdir = ROOT / f"checkpoint/confirm/{kind}_seed{seed}"
    load_params = torch.load(ckdir / "encoder_group.pth", map_location="cuda", weights_only=True)
    encs = []
    for i in range(8):
        enc = VGG16(ch_in=1, alpha=0.125, blur_pool=0, pool_power=0.0, trc=TRC_OF[kind])
        sd = load_params["backbone_state_dict_list"][i]
        missing, unexpected = enc.load_state_dict(sd, strict=False)
        assert missing == ["fc.weight", "fc.bias"] and unexpected == [], (kind, seed, missing)
        encs.append(torch.nn.Sequential(*list(enc.children())[:-1]).cuda().eval())
    return encs, ckdir


def lp_features(encs, x_t, bs=256):
    """x_t: (N,8,L) cuda tensor -> (N,512) 特征, 与 run_lp.infer_feature 口径一致。"""
    feats = []
    with torch.no_grad():
        for b0 in range(0, x_t.shape[0], bs):
            xb = x_t[b0:b0 + bs]
            feats.append(torch.cat([encs[j](xb[:, [j], :]).squeeze(-1) for j in range(8)], dim=1).cpu())
    return torch.cat(feats)


def metrics_from_probs(y_true, probs, nc):
    y1 = np.eye(nc)[y_true]
    return (float(roc_auc_score(y1, probs, average="macro")),
            float(average_precision_score(y1, probs)))


def eval_ft_probs(model, x_t):
    outs = []
    with torch.no_grad():
        for b0 in range(0, x_t.shape[0], 256):
            outs.append(F.softmax(model(x_t[b0:b0 + 256]), dim=1).cpu().numpy())
    return np.concatenate(outs)


def build_ft(kind, seed, nc):
    model = MultiBranchNet(nc, trc=TRC_OF[kind]).cuda().eval()
    sd = torch.load(ROOT / f"ft_models/confirm/{kind}_seed{seed}/ft_best_ckpt.pth", map_location="cuda", weights_only=True)
    model.load_state_dict(sd)  # strict=True: 全模型 state_dict, 错配硬拒
    return model


# ---------------- 账本 ----------------

def ledger_clean():
    ref = {}
    for r in csv.DictReader(open(LEDGER, encoding="utf-8")):
        ref[(r["ckpt"], r["seed"], r["eval"], r["downstream"])] = (
            float(r["auroc"]), float(r["auprc"]), r["checkpoint_sha256"])
    return ref


def ckpt_sha(kind, seed):
    cfg = json.loads((ROOT / f"checkpoint/confirm/{kind}_seed{seed}/config.json").read_text(encoding="utf-8"))
    return cfg.get("checkpoint_sha256", "?")


def append_row(model_csv, row):
    new = not model_csv.exists()
    with open(model_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "dataset", "eval", "seed", "perturbation", "intensity",
                        "auroc", "auprc", "clean_auroc", "clean_auprc",
                        "delta_auroc_pt", "delta_auprc_pt",
                        "n_test", "git_sha", "checkpoint_sha256", "data_sha"])
        w.writerow(row)


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="full", choices=["clean", "full", "smoke"])
    ap.add_argument("--limit", type=int, default=None, help="仅自检: 截断 test 记录数")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gsha = git_sha()
    ref = ledger_clean()
    smoke = args.stage == "smoke"
    conds = ([("clean", "1.0")] if args.stage == "clean" else
             [("clean", "1.0"), ("bw", "5"), ("qual_amp", "0.4"), ("swap12", "III<->V1")] if smoke
             else CONDITIONS)
    limit = args.limit or (256 if smoke else None)
    log(f"stage={args.stage} conds={len(conds)} git={gsha}")

    # 组合表: (ds, eval, [(kind, seed), ...])
    combos = []
    for ds in ("ptbxl", "cpsc"):
        for kind in ("c2", "c1", "b0"):
            seeds = LP_SEEDS[ds][kind]
            if seeds:
                combos.append((ds, "lp", [(kind, s) for s in seeds]))
    combos.append(("ptbxl", "ft10", [(k, s) for k in ("c2", "c1", "b0") for s in FT_SEEDS[k]]))

    for ds, eval_t, pairs in combos:
        nc = NUM_CLASSES[ds]
        x, y, data_sha, n_cls = load_test(ds, limit)
        assert n_cls == nc, (ds, n_cls, nc)
        log(f"== {ds}/{eval_t}: N={len(y)} classes={nc} data_sha={data_sha} combos={len(pairs)}")
        # 每 (kind,seed) 加载模型一次, clean 门先行
        models, shas, clean_m = {}, {}, {}
        for kind, seed in pairs:
            if eval_t == "lp":
                encs, ckdir = build_lp_encoders(kind, seed)
                head = LinearClassifier(feat_dim=512, num_classes=nc).cuda().eval()
                head.load_state_dict(torch.load(
                    ROOT / f"results/confirm/{kind}_{ds}_seed{seed}/classifier_best_ckpt.pth",
                    map_location="cuda", weights_only=True))
                models[(kind, seed)] = (encs, head)
            else:
                models[(kind, seed)] = build_ft(kind, seed, nc)
            shas[(kind, seed)] = ckpt_sha(kind, seed)
        for name, inten in conds:
            xp = make_perturbed(x, ds, name, inten)
            xp_t = torch.from_numpy(np.ascontiguousarray(xp)).cuda()
            for kind, seed in pairs:
                if eval_t == "lp":
                    encs, head = models[(kind, seed)]
                    feat = lp_features(encs, xp_t)
                    with torch.no_grad():
                        probs = F.softmax(head(feat.cuda()), dim=1).cpu().numpy()
                else:
                    probs = eval_ft_probs(models[(kind, seed)], xp_t)
                au, ap_ = metrics_from_probs(y, probs, nc)
                if name == "clean":
                    clean_m[(kind, seed)] = (au, ap_)
                    key = (kind, str(seed), eval_t, ds)
                    if key in ref and not smoke:
                        rau, rap, _ = ref[key]
                        if abs(au - rau) > 5e-4 or abs(ap_ - rap) > 5e-4:
                            log(f"  ✗ CLEAN 门失败 {key}: 本机 {au:.4f}/{ap_:.4f} vs 账本 {rau:.4f}/{rap:.4f} -> 中止")
                            return 2
                        log(f"  ✔ clean {key}: {au:.4f}/{ap_:.4f} = 账本({rau:.4f}/{rap:.4f})")
                    else:
                        log(f"  · clean {key}: {au:.4f}/{ap_:.4f} (无账本参照)")
                    continue
                cau, cap = clean_m[(kind, seed)]
                if not smoke:
                    append_row(OUT_DIR / f"robustness_{kind}.csv",
                           [time.strftime("%m-%d %H:%M"), ds, eval_t, seed, name, inten,
                            f"{au:.4f}", f"{ap_:.4f}", f"{cau:.4f}", f"{cap:.4f}",
                            f"{(au - cau) * 100:+.2f}", f"{(ap_ - cap) * 100:+.2f}",
                            len(y), gsha, shas[(kind, seed)][:12], data_sha])
                log(f"  · {kind} s{seed} {ds}/{eval_t} {name}@{inten}: {au:.4f}/{ap_:.4f} "
                    f"(Δauroc {(au - cau) * 100:+.2f}pt)")
            del xp, xp_t
            torch.cuda.empty_cache()
        if args.stage == "clean":
            log(f"== {ds}/{eval_t} clean 门全部通过")
    log("ROBUSTNESS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
