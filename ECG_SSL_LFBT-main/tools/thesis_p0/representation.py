"""thesis-p0-20261023 A3: 心拍表示诊断 D0-D3 (4 配置 × 3 种子).

分片约定(与协议/本地 beat_view 对齐, 记录源与差异):
- gqrs_detect(wfdb 4.1.2 默认参数), fs=204.8, V5=idx6 优先, 峰<4 回退 II=idx0;
  仍<4 → beat_valid=False, 四配置统一 16 个均匀固定窗。
- 窗 W=205, [p-102,p+103), 越界 edge padding; 无逐拍重归一化。
- K_r=min(峰数,16); 峰数>16 时按有序峰索引等距确定性取 16。
- 均匀窗中心 c_i = floor((i+0.5)*2048/K) (D0 与无峰回退共用该公式)。

输入物化: adaptation_data/repr_inputs/{mode}_{split}.npy (N,16,8,205) +
mask (N,16) + beat_valid + K_r — 种子无关, SHA 锁定后训练作业只读它。
工程检查(先于主作业): 参数量=12713 / 64 条过拟合 / mask-边界 / 无峰回退。
峰扰动敏感性(训练后仅验证推理): ±4/±8 样点, seed=20261023, 逐记录单偏移
跨配置共享, 峰数/掩码不变, 中心裁 [0,2047], D0 不变。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.thesis_p0 import common as C

HP = C.PROTOCOL["local_and_diagnostic_training"]
RD = C.PROTOCOL["representation_diagnostic"]
FS = 204.8
W = RD["window_samples"]              # 205
OFF_LO = RD["window_start_offset"]    # -102
OFF_HI = RD["window_end_offset_exclusive"]  # 103
KMAX = RD["max_windows"]              # 16
MIN_PEAKS = RD["beat_valid_min_peaks"]  # 4
T_LEN = 2048
REPR_DIR = C.ADAPT / "repr_inputs"
MODES = ["D0_matched_fixed", "D1_mean_template", "D2_median_template", "D3_beat_set"]


def uniform_centers(k: int, t: int = T_LEN) -> np.ndarray:
    return np.floor((np.arange(k) + 0.5) * t / k).astype(np.int64)


def select_peaks(peaks: np.ndarray):
    """返回 (kept_centers, k_r, beat_valid)。峰数>16 等距确定性取 16。"""
    if peaks is None or len(peaks) < MIN_PEAKS:
        return uniform_centers(KMAX), KMAX, False
    if len(peaks) <= KMAX:
        return np.asarray(peaks, dtype=np.int64), len(peaks), True
    idx = np.linspace(0, len(peaks) - 1, KMAX)
    idx = np.rint(idx).astype(np.int64)
    idx = np.unique(idx)  # len>16 时 rint 不会塌缩, 保险去重保序
    return np.asarray(peaks)[idx], len(idx), True


def cut_window(raw: np.ndarray, center: int) -> np.ndarray:
    """raw (8,T) -> (8,205), [c-102,c+103), 越界 edge padding。"""
    lo, hi = int(center) + OFF_LO, int(center) + OFF_HI
    seg = raw[:, max(lo, 0):min(hi, raw.shape[1])]
    pad_lo, pad_hi = max(0, -lo), max(0, hi - raw.shape[1])
    if pad_lo or pad_hi:
        seg = np.pad(seg, ((0, 0), (pad_lo, pad_hi)), mode="edge")
    return seg


def build_peak_cache():
    """train/val 峰缓存(离线一次), 存 FEAT_CACHE/rpeaks_cpsc_trainval.npz。"""
    import wfdb
    import wfdb.processing as wp

    def detect_one(sig):
        for lead_idx in (6, 0):
            try:
                pk = wp.gqrs_detect(sig[lead_idx].astype(np.float64), fs=FS)
            except Exception:
                pk = np.empty(0, dtype=np.int64)
            if len(pk) >= MIN_PEAKS:
                return pk.astype(np.int64), lead_idx
        return np.empty(0, dtype=np.int64), -1

    t0 = time.perf_counter()
    out, stats = {}, {"train": {"v5": 0, "ii": 0, "fail": 0}, "val": {"v5": 0, "ii": 0, "fail": 0}}
    for split in ("train", "val"):
        X, rids, _ = C.load_dev_data(split)
        for i, rid in enumerate(rids):
            pk, lead = detect_one(X[i])
            out[f"{split}/{rid}"] = pk
            key = {6: "v5", 0: "ii", -1: "fail"}[lead]
            stats[split][key] += 1
    path = C.FEAT_CACHE / "rpeaks_cpsc_trainval.npz"
    np.savez(path, **out)
    wall = time.perf_counter() - t0
    stats.update({"wfdb_version": wfdb.__version__, "detector": "gqrs_detect(defaults)",
                  "fs": FS, "primary_lead": "V5(idx6)", "fallback_lead": "II(idx0)",
                  "wall_s": round(wall, 1)})
    (REPR_DIR).mkdir(parents=True, exist_ok=True)
    (REPR_DIR / "peak_cache_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    C.log_resource("a3_peak_cache", "peak_detection", wall, gpu=False, extra=stats)
    print("peak cache:", stats, "->", path)
    return path


def load_peaks():
    return np.load(C.FEAT_CACHE / "rpeaks_cpsc_trainval.npz", allow_pickle=True)


def build_inputs():
    """物化四配置 train/val 输入张量 + mask/beat_valid/K_r (种子无关)。"""
    REPR_DIR.mkdir(parents=True, exist_ok=True)
    peaks_npz = load_peaks()
    for split in ("train", "val"):
        X, rids, _ = C.load_dev_data(split)
        n = len(rids)
        store = {m: np.zeros((n, KMAX, 8, W), dtype=np.float32) for m in MODES}
        mask = np.zeros((n, KMAX), dtype=np.float32)
        beat_valid = np.zeros(n, dtype=bool)
        k_r_arr = np.zeros(n, dtype=np.int64)
        for i, rid in enumerate(rids):
            centers, k_r, valid = select_peaks(peaks_npz[f"{split}/{rid}"])
            beat_valid[i], k_r_arr[i] = valid, k_r
            mask[i, :k_r] = 1.0
            if not valid:  # 无峰回退: 四配置共用 16 均匀固定窗
                centers = uniform_centers(KMAX)
                k_r = KMAX
            wins = np.stack([cut_window(X[i], c) for c in centers[:k_r]])  # (k_r,8,205)
            # D0: 均匀窗(共享峰计数, 不用峰位置)
            ucenters = uniform_centers(k_r)
            store["D0_matched_fixed"][i, :k_r] = np.stack([cut_window(X[i], c) for c in ucenters])
            tpl_mean = wins.mean(0)
            tpl_median = np.median(wins, axis=0)
            store["D1_mean_template"][i, :k_r] = np.broadcast_to(tpl_mean, (k_r, 8, W))
            store["D2_median_template"][i, :k_r] = np.broadcast_to(tpl_median, (k_r, 8, W))
            store["D3_beat_set"][i, :k_r] = wins
        for m in MODES:
            np.save(REPR_DIR / f"{m}_{split}.npy", store[m])
        np.save(REPR_DIR / f"mask_{split}.npy", mask)
        np.save(REPR_DIR / f"beat_valid_{split}.npy", beat_valid)
        np.save(REPR_DIR / f"k_r_{split}.npy", k_r_arr)
        print(f"materialized {split}: n={n} valid={int(beat_valid.sum())} "
              f"K_r median={int(np.median(k_r_arr))}")
    manifest = {"inputs_sha256": {}, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    for f in sorted(REPR_DIR.glob("*.npy")):
        manifest["inputs_sha256"][f.name] = C.sha256_of(f)
    (REPR_DIR / "repr_inputs_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


def load_mode(mode: str, split: str):
    X = np.load(REPR_DIR / f"{mode}_{split}.npy", mmap_mode="r")
    mask = np.load(REPR_DIR / f"mask_{split}.npy")
    beat_valid = np.load(REPR_DIR / f"beat_valid_{split}.npy")
    _, rids, y = C.load_dev_data(split)
    return X, mask, beat_valid, rids, y


class BeatEnc(nn.Module):
    """协议固定小网络: 共享窗编码器 + masked 均值聚合 + Linear(64,9)。"""

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=7, padding=3, bias=True), nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=True), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, 9)

    def forward(self, wins, mask):
        # wins (B,K,8,205) mask (B,K)
        B, K = mask.shape
        f = self.enc(wins.reshape(B * K, 8, W)).squeeze(-1)  # (B*K,64)
        f = f.view(B, K, 64)
        agg = (f * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)
        return self.head(agg)


MODE_SLUG = {"D0_matched_fixed": "d0_matched_fixed", "D1_mean_template": "d1_mean_template",
             "D2_median_template": "d2_median_template", "D3_beat_set": "d3_beat_set"}


def train_mode(mode: str, seed: int) -> dict:
    job_id = f"{MODE_SLUG[mode]}_s{seed}"
    job_dir = C.JOBS / job_id
    done = job_dir / "DONE.json"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    job_dir.mkdir(parents=True, exist_ok=True)
    C.set_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    Xtr, mtr, bvtr, rid_tr, ytr = load_mode(mode, "train")
    Xva, mva, bvva, rid_va, yva = load_mode(mode, "val")
    model = BeatEnc().to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == RD["expected_trainable_parameters"], f"param count {n_params} != 12713"
    opt = torch.optim.Adam(model.parameters(), lr=HP["learning_rate"],
                           betas=tuple(HP["betas"]), eps=HP["eps"], weight_decay=HP["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    n = Xtr.shape[0]
    bs = HP["batch_size"]
    Xtr_np, mtr_np = np.asarray(Xtr), mtr
    best = {"epoch": -1, "val": float("inf"), "state": None}
    stop_reason = "max_epochs"
    no_improve = 0
    t0 = time.perf_counter()
    for ep in range(1, HP["epochs_max"] + 1):
        model.train()
        perm = np.random.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            xb = torch.from_numpy(Xtr_np[idx]).float().to(dev)
            mb = torch.from_numpy(mtr_np[idx]).to(dev)
            yb = torch.from_numpy(ytr[idx]).to(dev)
            loss = loss_fn(model(xb, mb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            tot, el = 0.0, 0
            for s in range(0, Xva.shape[0], 512):
                xb = torch.from_numpy(np.asarray(Xva[s:s + 512])).float().to(dev)
                mb = torch.from_numpy(mva[s:s + 512]).to(dev)
                yb = torch.from_numpy(yva[s:s + 512]).to(dev)
                lg = model(xb, mb)
                tot += float(nn.functional.binary_cross_entropy_with_logits(
                    lg, yb, reduction="sum"))
                el += lg.numel()
            val_bce = tot / el
        if val_bce < best["val"]:
            best = {"epoch": ep, "val": val_bce,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= HP["early_stop_patience"]:
                stop_reason = f"early_stop_patience{HP['early_stop_patience']}"
                break
    train_wall = time.perf_counter() - t0
    model.load_state_dict(best["state"]); model.eval()
    with torch.no_grad():
        probs = []
        for s in range(0, Xva.shape[0], 512):
            lg = model(torch.from_numpy(np.asarray(Xva[s:s + 512])).float().to(dev),
                       torch.from_numpy(mva[s:s + 512]).to(dev))
            probs.append(torch.sigmoid(lg).cpu().numpy())
    probs = np.concatenate(probs)
    vram = round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else 0.0
    metrics = C.multilabel_metrics(yva, probs)
    metrics_bv = C.multilabel_metrics(yva[bvva], probs[bvva]) if bvva.sum() >= 20 else None
    np.savez_compressed(job_dir / "val_predictions.npz", record_ids=rid_va,
                        y_true=yva.astype(np.int8), y_prob=probs.astype(np.float32),
                        beat_valid=bvva, class_order=np.array(C.CLASS_ORDER))
    zj = np.load(job_dir / "val_predictions.npz", allow_pickle=False)
    recompute = C.multilabel_metrics(zj["y_true"].astype(np.float32), zj["y_prob"].astype(np.float32))
    for k in ("macro_auprc", "macro_auroc", "val_bce_from_prob"):
        assert abs(metrics[k] - recompute[k]) <= 1e-8, (k, metrics[k], recompute[k])
    torch.save({"model_state": best["state"], "best_epoch": best["epoch"]},
               job_dir / "model_best.pth")
    result = {
        "job_id": job_id, "stage": "A3_representation", "mode": mode,
        "training_seed": seed, "pretrain_seed": "none",
        "status": "complete", "trainable_params": n_params,
        "macro_auprc": metrics["macro_auprc"], "macro_auroc": metrics["macro_auroc"],
        "val_bce": best["val"], "best_epoch": best["epoch"], "stop_reason": stop_reason,
        "train_wall_s": round(train_wall, 1), "gpu_wall_s": round(train_wall, 1),
        "vram_peak_mib": vram,
        "macro_auprc_beat_valid": metrics_bv["macro_auprc"] if metrics_bv else None,
        "n_beat_valid": int(bvva.sum()), "n_beat_invalid": int((~bvva).sum()),
        "code_sha": C.git_head(),
        "recompute_abs_diff_max": max(abs(metrics[k] - recompute[k])
                                      for k in ("macro_auprc", "macro_auroc", "val_bce_from_prob")),
    }
    (job_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    done.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    C.log_resource(job_id, "train", train_wall, gpu=True, extra={"vram_peak_mib": vram})
    print(f"[{job_id}] auprc={metrics['macro_auprc']:.4f} auroc={metrics['macro_auroc']:.4f} "
          f"bce={best['val']:.4f} best_ep={best['epoch']} stop={stop_reason}")
    return result


def engineering_checks() -> bool:
    """参数量 / 64条过拟合 / mask-边界 / 无峰回退, 先于主作业。"""
    ok = True
    model = BeatEnc()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"param count = {n} (expect 12713)")
    ok &= n == RD["expected_trainable_parameters"]

    # mask/边界: 中心在 0 与 2047 的窗
    raw = np.random.RandomState(0).randn(8, T_LEN).astype(np.float32)
    w0, wT = cut_window(raw, 0), cut_window(raw, 2047)
    edge_ok = (w0.shape == (8, W) and wT.shape == (8, W)
               and np.allclose(w0[:, 0], raw[:, 0]) and np.allclose(wT[:, -1], raw[:, -1]))
    print("edge window check:", edge_ok)
    ok &= edge_ok

    # 无峰回退
    c, k_r, valid = select_peaks(np.array([], dtype=np.int64))
    fb_ok = (not valid) and k_r == KMAX and len(c) == KMAX
    c2, k2, v2 = select_peaks(np.arange(0, 2048, 8))  # 256 峰 -> 等距取 16
    sub_ok = v2 and k2 == KMAX and len(np.unique(c2)) == KMAX
    print("no-peak fallback:", fb_ok, "| >16 equal-spacing:", sub_ok)
    ok &= fb_ok and sub_ok

    # 64 条有限过拟合(D3, seed0): 训练 BCE 应显著下降
    C.set_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, m, bv, rids, y = load_mode("D3_beat_set", "train")
    idx = np.arange(64)
    Xs = torch.from_numpy(np.asarray(X[idx])).float().to(dev)
    ms = torch.from_numpy(m[idx]).to(dev)
    ys = torch.from_numpy(y[idx]).to(dev)
    m2 = BeatEnc().to(dev)
    opt = torch.optim.Adam(m2.parameters(), lr=HP["learning_rate"])
    lf = nn.BCEWithLogitsLoss()
    first = last = None
    for ep in range(60):
        opt.zero_grad()
        loss = lf(m2(Xs, ms), ys)
        loss.backward(); opt.step()
        val = float(loss)
        first = first if first is not None else val
        last = val
    overfit_ok = last < first * 0.5
    print(f"64-rec overfit: first={first:.4f} last={last:.4f} ok={overfit_ok}")
    ok &= overfit_ok
    C.log_resource("a3_engineering_checks", "checks", 0.0, gpu=True,
                   extra={"param_count": n, "overfit_first": first, "overfit_last": last,
                          "passed": bool(ok)})
    print("A3 engineering checks:", "PASS" if ok else "FAIL")
    return ok


def jitter_sensitivity():
    """训练后验证推理: ±4/±8 峰位扰动, 逐记录配对输出(仅 D1/D2/D3 重构输入)。"""
    rng = np.random.RandomState(RD["jitter_seed"])
    Xva, rid_va, yva = C.load_dev_data("val")
    peaks = load_peaks()
    out_dir = C.RESULTS / "peak_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_records = len(rid_va)
    # 先抽完全部偏移(记录序=canonical), 保证跨模式/档位同记录同偏移语义
    offsets = {a: rng.randint(-a, a + 1, size=n_records) for a in RD["validation_peak_jitter_samples"] if a > 0}
    summary = []
    for mode in MODES[1:]:  # D0 不受扰动影响
        for seed in (0, 2, 4):
            job_id = f"{ {'D1_mean_template':'d1_mean_template','D2_median_template':'d2_median_template','D3_beat_set':'d3_beat_set'}[mode] }_s{seed}"
            ck = C.JOBS / job_id / "model_best.pth"
            if not ck.exists():
                continue
            model = BeatEnc()
            model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True)["model_state"])
            if torch.cuda.is_available():
                model = model.to("cuda").eval()
            for a, offs in offsets.items():
                probs = np.zeros((n_records, 9), dtype=np.float32)
                masks = np.zeros((n_records, KMAX), dtype=np.float32)
                with torch.no_grad():
                    for i, rid in enumerate(rid_va):
                        centers, k_r, valid = select_peaks(peaks[f"val/{rid}"])
                        if not valid:
                            centers = uniform_centers(KMAX); k_r = KMAX
                        else:
                            centers = np.clip(centers[:k_r] + int(offs[i]), 0, T_LEN - 1)
                        wins = np.stack([cut_window(Xva[i], c) for c in centers[:k_r]])
                        if mode == "D1_mean_template":
                            wins = np.broadcast_to(wins.mean(0), (k_r, 8, W))
                        elif mode == "D2_median_template":
                            wins = np.broadcast_to(np.median(wins, axis=0), (k_r, 8, W))
                        full = np.zeros((1, KMAX, 8, W), dtype=np.float32)
                        full[0, :k_r] = wins
                        m = np.zeros((1, KMAX), dtype=np.float32); m[0, :k_r] = 1.0
                        xb = torch.from_numpy(full)
                        if torch.cuda.is_available():
                            xb = xb.cuda(); m_t = torch.from_numpy(m).cuda()
                        else:
                            m_t = torch.from_numpy(m)
                        probs[i] = torch.sigmoid(model(xb, m_t)).cpu().numpy()[0]
                        masks[i] = m[0]
                orig = np.load(C.JOBS / job_id / "val_predictions.npz")["y_prob"]
                np.savez_compressed(out_dir / f"{job_id}_jitter{a}.npz",
                                    record_ids=rid_va, y_true=yva.astype(np.int8),
                                    y_prob_orig=orig, y_prob_jitter=probs)
                from sklearn.metrics import average_precision_score
                auprc_j = float(np.mean([average_precision_score(yva[:, c], probs[:, c])
                                         for c in range(9)]))
                auprc_o = float(np.mean([average_precision_score(yva[:, c], orig[:, c])
                                         for c in range(9)]))
                summary.append({"job_id": job_id, "jitter_samples": a,
                                "macro_auprc_orig": auprc_o, "macro_auprc_jitter": auprc_j,
                                "mean_abs_prob_delta": float(np.abs(probs - orig).mean())})
                print(f"[jitter ±{a}] {job_id}: auprc {auprc_o:.4f} -> {auprc_j:.4f}")
    (out_dir / "jitter_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-inputs", "check", "train", "jitter"])
    ap.add_argument("--phase", choices=["seed0", "rest", "all"], default="all")
    args = ap.parse_args()
    if args.cmd == "build-inputs":
        build_peak_cache()
        build_inputs()
        return 0
    if args.cmd == "check":
        return 0 if engineering_checks() else 2
    if args.cmd == "train":
        jobs = MODES if args.phase != "rest" else MODES
        seeds = [0] if args.phase == "seed0" else ([2, 4] if args.phase == "rest" else [0, 2, 4])
        results = [train_mode(m, s) for m in MODES for s in seeds]
        if args.phase == "seed0":
            print("A3 seed0:", [(r["job_id"], round(r["macro_auprc"], 4)) for r in results])
        return 0
    if args.cmd == "jitter":
        jitter_sensitivity()
        return 0


if __name__ == "__main__":
    sys.exit(main())
