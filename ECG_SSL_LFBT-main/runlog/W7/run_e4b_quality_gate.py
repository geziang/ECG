# -*- coding: utf-8 -*-
"""W7 E4b: 质量门控推理——生理频带(5-40Hz)功率比质量分, 低分导联置零, 零训练推理期改进。

机制: 复用 W6 NSTDB 机器(build_lp/make_perturbed_real/refs, W3 只读)。置零导联的 per-lead
编码输出为常量向量(zero_feat), 预计算后各 τ 档门控=特征替换, 无需重复前向。
预注册规则(队列 Q4 规格):
  - τ 档 {0.3,0.4,0.5,0.6} 在各域 val 上扫: 约束 clean val Δ(对无门控) ≥ −0.10pt,
    约束内取噪声 val(3噪声×SNR{0,5,10}) 均值最大的 τ*; 无 τ 满足→该域关线。
  - test: seed0×{b0,c1,c2} 无门控 clean 先对 W6/W5 账本(±5e-4); 门控 clean Δ≥−0.10pt;
    扰动格=门控−无门控(无门控参照=W6 robustness_real_{kind}.csv seed0 行, 缺格现算)。
  - 总门: 噪声条件 3 域平均 ΔAUROC ≥ +0.5pt 判正。
产物: runlog/W7/e4b_val_sweep.csv, e4b_quality_gate.csv, e4b_summary.md
用法: DL_PY runlog/W7/run_e4b_quality_gate.py --stage val|test|smoke
"""
import argparse
import csv
import io
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))

import run_robustness as R3  # noqa: E402
from run_robustness import FS, ROOT, load_test, lp_features, metrics_from_probs, log, git_sha, _rng_for
from run_robustness_nstdb import build_lp, make_perturbed_real, refs

OUT_DIR = ROOT / "runlog/W7"
DOMAINS = ["ptbxl", "cpsc", "chapman"]
NUM_CLASSES = {"ptbxl": 5, "cpsc": 9, "chapman": 4}
REF_EPOCH = {"ptbxl": "w2", "cpsc": "w5", "chapman": "w2"}
TAUS = [0.3, 0.4, 0.5, 0.6]
NOISE_CONDS = [("nstb_" + n, s) for n in ("bw", "ma", "em") for s in ("0", "5", "10")]
SOS_Q = butter(4, [5.0 / (FS / 2), 40.0 / (FS / 2)], btype="band", output="sos")


def load_val(ds):
    import os
    root = ROOT / "data" / ds / "val"
    classes = sorted(d for d in os.listdir(root) if (root / d).is_dir())
    xs, ys = [], []
    for ci, c in enumerate(classes):
        for f in sorted(os.listdir(root / c)):
            if f.endswith(".npy"):
                xs.append(np.load(root / c / f))
                ys.append(ci)
    return np.stack(xs), np.array(ys)


def quality_scores(x):
    """(N,8,L) -> (N,8) 带内(5-40Hz)功率占比。"""
    xd = x.astype(np.float64)
    xf = sosfiltfilt(SOS_Q, xd, axis=-1)
    return (xf ** 2).mean(-1) / ((xd ** 2).mean(-1) + 1e-12)


def per_lead_feats(encs, x_t, bs=256):
    """(N,8,L) cuda -> (N,8,64) per-lead 特征 + 各导联全零常量特征 (8,64)。"""
    feats = []
    with torch.no_grad():
        for b0 in range(0, x_t.shape[0], bs):
            xb = x_t[b0:b0 + bs]
            feats.append(torch.stack([encs[j](xb[:, [j], :]).squeeze(-1) for j in range(8)], dim=1).cpu())
    zf = torch.stack([encs[j](torch.zeros(1, 1, x_t.shape[-1]).cuda()).squeeze(-1).cpu() for j in range(8)]).squeeze(1)
    return torch.cat(feats), zf


def assemble(pf, zf, q, tau):
    """τ 门控: q<tau 的导联特征替换为全零特征。pf (N,8,64), zf (8,64), q (N,8) -> (N,512)。"""
    keep = torch.from_numpy(q >= tau)
    out = pf.clone()
    out[~keep] = zf[torch.nonzero(~keep)[:, 1]]  # 逐 (i,j) 行替换
    return out.reshape(pf.shape[0], -1)


def w6_ref(ds, kind):
    """W6 robustness_real_{kind}.csv seed0 无门控参照 {(cond,inten): (auroc, auprc)}。"""
    p = ROOT / f"runlog/W6/robustness_real_{kind}.csv"
    out = {}
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["downstream"] == ds and r["seed"] == "0":
                    out[(r["noise"], r["intensity"])] = (float(r["auroc"]), float(r["auprc"]))
    return out


def eval_feats(head, feat, y, nc):
    with torch.no_grad():
        probs = F.softmax(head(feat.cuda()), dim=1).cpu().numpy()
    return metrics_from_probs(y, probs, nc)


def stage_val(models_by_ds, limit=None):
    rows = []
    sel = {}
    for ds in DOMAINS:
        nc = NUM_CLASSES[ds]
        x, y = load_val(ds)
        if limit:
            x, y = x[:limit], y[:limit]
        models = models_by_ds[ds]
        log(f"== val {ds}: N={len(y)}")
        # 各扰动变体的 per-lead 特征逐模型算一次
        variants = [("clean", "1.0")] + NOISE_CONDS
        per = {}  # (variant, kind) -> (pf, zf, q)
        for name, inten in variants:
            xv = x.copy() if name == "clean" else make_perturbed_real(x, ds, name, inten)
            xv_t = torch.from_numpy(np.ascontiguousarray(xv)).cuda()
            q = quality_scores(xv)
            for kind, (encs, head, _) in models.items():
                pf, zf = per_lead_feats(encs, xv_t)
                per[(name, inten, kind)] = (pf, zf, q)
            del xv, xv_t
        base = {k: eval_feats(models[k][1], per[("clean", "1.0", k)][0].reshape(len(y), -1), y, nc)
                for k in models}
        # 各 τ: clean 约束 + 噪声目标
        for tau in TAUS:
            cd = [eval_feats(models[k][1], assemble(*per[("clean", "1.0", k)], tau), y, nc)[0] - base[k][0]
                  for k in models]
            noisy = []
            for name, inten in NOISE_CONDS:
                for k in models:
                    noisy.append(eval_feats(models[k][1], assemble(*per[(name, inten, k)], tau), y, nc)[0])
            rows.append(dict(domain=ds, tau=tau, clean_delta_mean=float(np.mean(cd)) * 100,
                             noisy_mean=float(np.mean(noisy))))
            log(f"  tau={tau}: cleanΔ={np.mean(cd)*100:+.3f}pt noisy={np.mean(noisy):.4f}")
        ok = [r for r in rows if r["domain"] == ds and r["clean_delta_mean"] >= -0.10]
        if ok:
            best = max(ok, key=lambda r: r["noisy_mean"])
            sel[ds] = best["tau"]
            log(f"  -> tau*={best['tau']}")
        else:
            sel[ds] = None
            log(f"  -> 无 τ 满足 clean 约束, 该域关线")
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=["domain", "tau", "clean_delta_mean", "noisy_mean"])
    w.writeheader()
    w.writerows(rows)
    (OUT_DIR / "e4b_val_sweep.csv").write_text(buf.getvalue(), encoding="utf-8", newline="")
    (OUT_DIR / "e4b_tau_selection.json").write_text(
        __import__("json").dumps(sel, indent=1), encoding="utf-8")
    log(f"tau selection: {sel}")
    return sel


def stage_test(models_by_ds, sel, limit=None):
    gsha = git_sha()
    rows = []
    for ds in DOMAINS:
        if sel.get(ds) is None:
            log(f"== {ds}: 已在 val 关线, 跳过")
            continue
        tau = sel[ds]
        nc = NUM_CLASSES[ds]
        x, y, data_sha, _ = load_test(ds, limit)
        models = models_by_ds[ds]
        ref_ledger = refs(ds, REF_EPOCH[ds])
        log(f"== test {ds}: N={len(y)} tau*={tau}")
        variants = [("clean", "1.0")] + NOISE_CONDS
        for name, inten in variants:
            xv = x.copy() if name == "clean" else make_perturbed_real(x, ds, name, inten)
            xv_t = torch.from_numpy(np.ascontiguousarray(xv)).cuda()
            q = quality_scores(xv)
            n_gate = int((q < tau).sum())
            for kind, (encs, head, _) in models.items():
                pf, zf = per_lead_feats(encs, xv_t)
                pf_flat = pf.reshape(len(y), -1)
                au_u, ap_u = eval_feats(head, pf_flat, y, nc)      # 无门控
                au_g, ap_g = eval_feats(head, assemble(pf, zf, q, tau), y, nc)  # 门控
                if name == "clean":
                    key = (kind, "0", "lp", ds)
                    if key in ref_ledger:
                        rau, rap = ref_ledger[key]
                        if abs(au_u - rau) > 5e-4 or abs(ap_u - rap) > 5e-4:
                            log(f"  ✗ CLEAN 复现门失败 {key}: {au_u:.4f} vs 账本 {rau:.4f} -> 中止")
                            return 2
                    log(f"  ✔ clean {kind}: 无门控={au_u:.4f} 门控={au_g:.4f} (置零导联数 {n_gate}/({len(y)}×8))")
                rows.append(dict(ts=time.strftime("%m-%d %H:%M"), domain=ds, kind=kind, seed=0,
                                 cond=name, inten=inten, tau=tau, n_gated_leads=n_gate,
                                 auroc_ungated=round(au_u, 4), auroc_gated=round(au_g, 4),
                                 delta=round((au_g - au_u) * 100, 2),
                                 auprc_ungated=round(ap_u, 4), auprc_gated=round(ap_g, 4),
                                 git_sha=gsha, data_sha=data_sha, protocol_id="w7-e4b"))
                log(f"  · {kind} {name}@{inten}: 无门控 {au_u:.4f} -> 门控 {au_g:.4f} (Δ{(au_g-au_u)*100:+.2f}pt)")
            del xv, xv_t
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    (OUT_DIR / "e4b_quality_gate.csv").write_text(buf.getvalue(), encoding="utf-8", newline="")
    # 汇总
    import json
    noisy = [r for r in rows if r["cond"] != "clean"]
    clean = [r for r in rows if r["cond"] == "clean"]
    lines = ["# E4b 质量门控推理摘要", "",
             f"- τ*(val 选档): {sel}",
             f"- clean 门: " + ("全部 Δ≥−0.10pt" if all(r["delta"] >= -0.10 for r in clean) else "存在违例(见CSV)"),
             f"- 噪声格: 无门控 vs 门控 平均 Δ = {np.mean([r['delta'] for r in noisy]):+.2f}pt "
             f"(格数 {len(noisy)}, 总门≥+0.5pt 判正)", "",
             "| domain | kind | 平均Δ(9噪声格) |", "|---|---|---|"]
    for ds in DOMAINS:
        for kind in ("b0", "c1", "c2"):
            sub = [r["delta"] for r in noisy if r["domain"] == ds and r["kind"] == kind]
            if sub:
                lines.append(f"| {ds} | {kind} | {np.mean(sub):+.2f}pt |")
    (OUT_DIR / "e4b_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("E4B_DONE")
    return 0


def build_models():
    out = {}
    for ds in DOMAINS:
        models = {}
        for kind in ("b0", "c1", "c2"):
            try:
                encs, head, ckdir = build_lp(kind, 0, ds, REF_EPOCH[ds])
                models[kind] = (encs, head, ckdir)
            except Exception as e:
                log(f"  (跳过 {ds}/{kind}: {type(e).__name__}: {e})")
        out[ds] = models
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="val", choices=["val", "test", "smoke"])
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = build_models()
    if args.stage == "smoke":
        sel = stage_val(models, limit=96)
        return stage_test(models, sel, limit=96)
    if args.stage == "val":
        stage_val(models)
        return 0
    import json
    sel_p = OUT_DIR / "e4b_tau_selection.json"
    sel = json.loads(sel_p.read_text(encoding="utf-8")) if sel_p.exists() else {}
    return stage_test(models, sel)


if __name__ == "__main__":
    raise SystemExit(main())
