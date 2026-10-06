# -*- coding: utf-8 -*-
"""W7 E2d: TRC 机理分析——C2 冻结 checkpoint 上检验"TRC 重标定缩小通道统计跨域偏移"假设。

对象: c2 seeds{0,1,2,3,4} (confirm/{0,2,4} + w6_seedext/{1,3}), 8 导联 VGG16(alpha=0.125, trc=1)。
语料: NFH(预训练分布, data/pt_pretrain_nfh 抽样) + 三域 test 抽样, 各 N=384(类分层, rng 20261006)。
机理量(per seed × domain × lead × channel):
  - mu_pre / mu_post: GRN1D 前后时间维通道均值(NFH 参照与目标域各算一份)
  - shift_pre(D)=mu_pre(D)−mu_pre(NFH); shift_post(D)=mu_post(D)−mu_post(NFH)
  - 缩偏比 ratio(D) = ||shift_post(D)||_1 / ||shift_pre(D)||_1  (<1 支持"TRC 缩小跨域偏移")
  - 通道相关: spearman(|shift_pre_c|, |Δmu_c|) (Δmu=mu_post−mu_pre, TRC 调整量)
  - TRC 参数: gamma/beta 逐通道幅值(直接读 state_dict)
产物: runlog/W7/e2d_channel_stats.csv, e2d_gamma_beta.csv, e2d_summary.md
性质: 描述性机理分析(只读冻结 checkpoint 与数据, 不训练)。
用法: python runlog/W7/run_e2d_trc_mechanism.py
"""
import csv
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from models.vgg_1d import VGG16  # noqa: E402

OUTD = ROOT / "runlog/W7"
LEADS = ["ii", "iii", "v1", "v2", "v3", "v4", "v5", "v6"]
C2_CKS = {0: "checkpoint/confirm/c2_seed0", 2: "checkpoint/confirm/c2_seed2", 4: "checkpoint/confirm/c2_seed4",
          1: "checkpoint/w6_seedext/c2_seed1", 3: "checkpoint/w6_seedext/c2_seed3"}
N_SAMPLE = 384
RNG = np.random.default_rng(20261006)


def log(msg):
    print(f"[e2d {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def sample_corpus(kind):
    """类分层抽样 N_SAMPLE 条 (8,2048) 信号。kind: nfh/ptbxl/cpsc/chapman。"""
    if kind == "nfh":
        files = sorted((ROOT / "data/pt_pretrain_nfh").rglob("*.npy"))
        idx = RNG.choice(len(files), size=min(N_SAMPLE, len(files)), replace=False)
        return [files[i] for i in idx]
    root = ROOT / f"data/{kind}/test"
    classes = sorted([p for p in root.iterdir() if p.is_dir()])
    per = max(1, N_SAMPLE // len(classes))
    picked = []
    for c in classes:
        fs = sorted(c.glob("*.npy"))
        take = min(per, len(fs))
        if take < len(fs):
            picked += [fs[i] for i in RNG.choice(len(fs), size=take, replace=False)]
        else:
            picked += fs
    return picked


def build_encoder(ckdir, device):
    """返回 per-lead [conv1-5 序列, GRN1D, pool], trc=1 结构(已置 device)。"""
    from utils.checkpoint import load_torch_checkpoint
    ck = load_torch_checkpoint(Path(ckdir) / "encoder_group.pth", map_location="cpu")
    encs = []
    for i in range(8):
        v = VGG16(ch_in=1, alpha=0.125, trc=1)
        if "backbone_state_dict_list" in ck:
            sd = ck["backbone_state_dict_list"][i]
        else:
            sd = ck["backbone_state_dict"][i].state_dict()
        v.load_state_dict(sd, strict=False)
        model = v.model.to(device).eval()  # Sequential: [b1..b5, GRN1D, pool] (trc=1)
        assert len(model) == 7, f"unexpected model len {len(model)}"
        encs.append((model[:5], model[5], model[6]))
    return encs


@torch.no_grad()
def channel_moments(encs, files, device):
    """返回 per-lead per-channel 时间维均值(对全部样本聚合)。"""
    conv, grn, _ = encs[0]
    C = None
    acc = {j: None for j in range(8)}  # sum over samples of per-channel time-mean
    rms = {j: None for j in range(8)}  # sum over samples of per-channel RMS
    n_tot = 0
    bs = 64
    for b0 in range(0, len(files), bs):
        batch = np.stack([np.load(f) for f in files[b0:b0 + bs]])
        x = torch.from_numpy(batch).float().to(device)
        n_tot += x.shape[0]
        for j in range(8):
            conv, grn, _ = encs[j]
            f = conv(x[:, j:j + 1, :])          # (B,C,T') pre-TRC
            g = grn(f)                          # post-TRC
            m_pre = f.mean(dim=-1).sum(dim=0)   # (C,)
            m_post = g.mean(dim=-1).sum(dim=0)
            r_pre = f.pow(2).mean(dim=-1).sqrt().sum(dim=0)   # per-channel RMS
            r_post = g.pow(2).mean(dim=-1).sqrt().sum(dim=0)
            quad = torch.stack([m_pre, m_post, r_pre, r_post])
            acc[j] = quad if acc[j] is None else acc[j] + quad
        if C is None:
            C = f.shape[1]
    mu_pre = {j: (acc[j][0] / n_tot).cpu().numpy() for j in range(8)}
    mu_post = {j: (acc[j][1] / n_tot).cpu().numpy() for j in range(8)}
    rms_pre = {j: (acc[j][2] / n_tot).cpu().numpy() for j in range(8)}
    rms_post = {j: (acc[j][3] / n_tot).cpu().numpy() for j in range(8)}
    return mu_pre, mu_post, rms_pre, rms_post


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTD.mkdir(exist_ok=True)
    kinds = ["nfh", "ptbxl", "cpsc", "chapman"]
    files = {k: sample_corpus(k) for k in kinds}
    log("samples: " + ", ".join(f"{k}={len(v)}" for k, v in files.items()))

    stat_rows, gb_rows, ratio_rows = [], [], []
    for seed, ckdir in sorted(C2_CKS.items()):
        encs = build_encoder(ckdir, device)
        # TRC 参数幅值
        for j in range(8):
            grn = encs[j][1]
            gm = grn.gamma.detach().cpu().numpy().ravel()
            bt = grn.beta.detach().cpu().numpy().ravel()
            for c in range(len(gm)):
                gb_rows.append(dict(seed=seed, lead=LEADS[j], channel=c,
                                    gamma=float(gm[c]), beta=float(bt[c])))
        mus = {}
        for k in kinds:
            mu_pre, mu_post, rms_pre, rms_post = channel_moments(encs, files[k], device)
            mus[k] = (mu_pre, mu_post, rms_pre, rms_post)
        for dom in ["ptbxl", "cpsc", "chapman"]:
            for j in range(8):
                pre_nfh, post_nfh = mus["nfh"][0][j], mus["nfh"][1][j]
                pre_d, post_d = mus[dom][0][j], mus[dom][1][j]
                rp_nfh, po_nfh = mus["nfh"][2][j], mus["nfh"][3][j]
                rp_d, po_d = mus[dom][2][j], mus[dom][3][j]
                sh_pre, sh_post = pre_d - pre_nfh, post_d - post_nfh
                d_mu = post_d - pre_d
                # RMS 尺度层: 通道能量的对数偏移(log-ratio)
                lr_pre = np.log(rp_d / rp_nfh)
                lr_post = np.log(po_d / po_nfh)
                for c in range(len(pre_d)):
                    stat_rows.append(dict(
                        seed=seed, domain=dom, lead=LEADS[j], channel=c,
                        mu_pre_nfh=float(pre_nfh[c]), mu_post_nfh=float(post_nfh[c]),
                        mu_pre_dom=float(pre_d[c]), mu_post_dom=float(post_d[c]),
                        shift_pre=float(sh_pre[c]), shift_post=float(sh_post[c]),
                        d_mu=float(d_mu[c]),
                        logrms_pre=float(lr_pre[c]), logrms_post=float(lr_post[c])))
                ratio_rows.append(dict(
                    seed=seed, domain=dom, lead=LEADS[j],
                    l1_pre=float(np.abs(sh_pre).sum()), l1_post=float(np.abs(sh_post).sum()),
                    ratio=float(np.abs(sh_post).sum() / max(np.abs(sh_pre).sum(), 1e-12)),
                    rho_shift_mod=spearman(np.abs(sh_pre), np.abs(d_mu)),
                    l1_lrms_pre=float(np.abs(lr_pre).sum()), l1_lrms_post=float(np.abs(lr_post).sum()),
                    ratio_lrms=float(np.abs(lr_post).sum() / max(np.abs(lr_pre).sum(), 1e-12))))
        log(f"seed {seed}: done")

    with open(OUTD / "e2d_channel_stats.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(stat_rows[0].keys()))
        w.writeheader()
        w.writerows(stat_rows)
    with open(OUTD / "e2d_gamma_beta.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(gb_rows[0].keys()))
        w.writeheader()
        w.writerows(gb_rows)

    # 摘要: ratio 按 domain 对 seed×lead 取均值; 相关同
    lines = ["# E2d TRC 机理分析摘要", "",
             f"- 样本: 各语料 {N_SAMPLE} 条(类分层, rng 20261006); c2 seeds={sorted(C2_CKS)}; 通道数 C=64×8导联",
             "- ratio = ||shift_post||_1 / ||shift_pre||_1 (shift=通道时间均值相对 NFH 的偏移; <1 即 TRC 缩小跨域偏移)",
             ""]
    lines += ["| domain | mean_ratio | median_ratio | mean_rho | mean_ratio_lrms | median_ratio_lrms | n_cells |",
              "|---|---|---|---|---|---|---|"]
    for dom in ["ptbxl", "cpsc", "chapman"]:
        sub = [r for r in ratio_rows if r["domain"] == dom]
        rs = [r["ratio"] for r in sub]
        rl = [r["ratio_lrms"] for r in sub]
        rh = [r["rho_shift_mod"] for r in sub if not np.isnan(r["rho_shift_mod"])]
        lines.append(f"| {dom} | {np.mean(rs):.4f} | {np.median(rs):.4f} | {np.mean(rh):+.4f} | "
                     f"{np.mean(rl):.4f} | {np.median(rl):.4f} | {len(rs)} |")
    gm = np.array([r["gamma"] for r in gb_rows])
    bt = np.array([r["beta"] for r in gb_rows])
    lines += ["", f"- TRC 参数幅值(全 seed×lead×channel, n={len(gm)}): |gamma| mean={np.abs(gm).mean():.4f} "
              f"p90={np.percentile(np.abs(gm), 90):.4f}; |beta| mean={np.abs(bt).mean():.4f} p90={np.percentile(np.abs(bt), 90):.4f}",
              "- 判读: mean_ratio>1=均值层不缩偏(如实记录); ratio_lrms<1=RMS 能量尺度层缩偏(支持范数域适应机制); rho 正=调整与偏移同调"]
    (OUTD / "e2d_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("summary written")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
