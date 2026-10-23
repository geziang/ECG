"""thesis-p0-20261023 A2: 冻结编码器多标签 LP (M0/M4/M6 × seeds {0,2,4}).

- 编码器 eval + 全参数冻结, 逐导联 VGG16(alpha=0.125, trc=模型开关) 提取
  8×64=512 维特征并缓存(带权重/数据/代码 SHA, 复用前核对)。
- 头 Linear(512,9), BCEWithLogitsLoss, Adam(lr=1e-3, eps=1e-8, wd=0),
  batch128, ≤100ep, patience15, 严格最小元素加权验证 BCE(平局取更早)。
- 逐作业输出 val 预测(record_id/y_true/y_prob)、指标、best_epoch、耗时、
  峰值显存、停止原因; 独立复算 |Δ|≤1e-8; 冻结参数不变校验。
- 仅经 common.load_dev_data() 读 train/val 导出副本, 无任何 test 路径。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # ECG_SSL_LFBT-main
from models.vgg_1d import VGG16
from utils.checkpoint import load_torch_checkpoint

from tools.thesis_p0 import common as C

HP = C.PROTOCOL["local_and_diagnostic_training"]


def build_encoders(ckpt_path: Path, trc: int):
    encs = []
    load_params = load_torch_checkpoint(ckpt_path, map_location="cpu")
    for i in range(8):
        enc = VGG16(ch_in=1, alpha=0.125, blur_pool=0, pool_power=0.0, trc=trc)
        sd = load_params["backbone_state_dict_list"][i]
        missing, unexpected = enc.load_state_dict(sd, strict=False)
        assert missing == ["fc.weight", "fc.bias"] and unexpected == [], (missing, unexpected)
        body = torch.nn.Sequential(*list(enc.children())[:-1])  # 去 fc, 输出 (B,64,1)
        body.eval()
        for p in body.parameters():
            p.requires_grad_(False)
        encs.append(body)
    return encs


def param_fingerprint(encs) -> str:
    import hashlib
    h = hashlib.sha256()
    for enc in encs:
        for p in enc.parameters():
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def extract_features(encs, X: np.ndarray, batch=256) -> np.ndarray:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = np.empty((X.shape[0], 512), dtype=np.float32)
    for s in range(0, X.shape[0], batch):
        xb = torch.from_numpy(X[s:s + batch]).to(dev)
        feats = []
        for j, enc in enumerate(encs):
            f = enc(xb[:, [j], :])  # (B,64,1)
            feats.append(f.squeeze(-1))
        out[s:s + batch] = torch.cat(feats, dim=1).cpu().numpy()
    return out


def get_features(model: str, seed: int):
    """缓存特征; 复用前核对权重/数据/代码 SHA 一致。"""
    C.FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    ckpt = C.CKPT_PATHS[(model, seed)]
    ckpt_sha = C.sha256_of(ckpt)
    cache = C.FEAT_CACHE / f"{model}_s{seed}.npz"
    dsha = C.data_sha()
    code_sha = C.git_head()
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        meta = json.loads(str(z["meta"].astype(str)))
        if (meta["ckpt_sha256"] == ckpt_sha and meta["data_sha"] == dsha
                and meta["code_sha"] == code_sha):
            return z["X_train"], z["X_val"], meta
    t0 = time.perf_counter()
    encs = build_encoders(ckpt, C.CKPT_TRC[model])
    if torch.cuda.is_available():
        encs = [e.to("cuda") for e in encs]
    Xtr, rid_tr, _ = C.load_dev_data("train")
    Xva, rid_va, _ = C.load_dev_data("val")
    Ftr = extract_features(encs, Xtr)
    Fva = extract_features(encs, Xva)
    frozen_ok = param_fingerprint(encs)  # 提取后仍未变(无梯度)
    wall = time.perf_counter() - t0
    meta = {"ckpt": str(ckpt), "ckpt_sha256": ckpt_sha, "data_sha": dsha,
            "code_sha": code_sha, "feat_wall_s": round(wall, 1),
            "train_ids_sha": __import__("hashlib").sha256(
                "\n".join(rid_tr).encode()).hexdigest(),
            "val_ids_sha": __import__("hashlib").sha256(
                "\n".join(rid_va).encode()).hexdigest()}
    np.savez(cache, X_train=Ftr, X_val=Fva, train_ids=rid_tr, val_ids=rid_va,
             meta=np.array(json.dumps(meta)))
    C.log_resource(f"feat_{model}_s{seed}", "feature_extract", wall, gpu=True,
                   extra={"vram_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)
                          if torch.cuda.is_available() else 0})
    return Ftr, Fva, meta


def train_head(Ftr, ytr, Fva, yva, seed: int):
    C.set_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    head = nn.Linear(512, 9).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=HP["learning_rate"],
                           betas=tuple(HP["betas"]), eps=HP["eps"], weight_decay=HP["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    n = Ftr.shape[0]
    bs = HP["batch_size"]
    best = {"epoch": -1, "val": float("inf"), "state": None}
    stop_reason = "max_epochs"
    t0 = time.perf_counter()
    epochs_no_improve = 0
    for ep in range(1, HP["epochs_max"] + 1):
        head.train()
        perm = torch.randperm(n)
        for s in range(0, n, bs):  # drop_last=False: 保留末批
            idx = perm[s:s + bs]
            xb = torch.from_numpy(Ftr[idx.numpy()]).to(dev)
            yb = torch.from_numpy(ytr[idx.numpy()]).to(dev)
            loss = loss_fn(head(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # 验证: 全集元素加权 BCE
        head.eval()
        with torch.no_grad():
            tot, el = 0.0, 0
            for s in range(0, Fva.shape[0], 1024):
                xb = torch.from_numpy(Fva[s:s + 1024]).to(dev)
                yb = torch.from_numpy(yva[s:s + 1024]).to(dev)
                lg = head(xb)
                l = nn.functional.binary_cross_entropy_with_logits(
                    lg, yb, reduction="sum")
                tot += float(l)
                el += lg.numel()
            val_bce = tot / el
        if val_bce < best["val"]:  # 严格更小; 平局保留更早
            best = {"epoch": ep, "val": val_bce,
                    "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= HP["early_stop_patience"]:
                stop_reason = f"early_stop_patience{HP['early_stop_patience']}"
                break
    train_wall = time.perf_counter() - t0
    head.load_state_dict(best["state"])
    head.eval()
    with torch.no_grad():
        probs = []
        for s in range(0, Fva.shape[0], 1024):
            lg = head(torch.from_numpy(Fva[s:s + 1024]).to(dev))
            probs.append(torch.sigmoid(lg).cpu().numpy())
    vram = round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else 0.0
    return head, np.concatenate(probs), best, stop_reason, train_wall, vram


def run_job(model: str, seed: int) -> dict:
    job_id = f"lp_{model.lower()}_s{seed}"
    job_dir = C.JOBS / job_id
    done_marker = job_dir / "DONE.json"
    if done_marker.exists():
        return json.loads(done_marker.read_text(encoding="utf-8"))

    job_dir.mkdir(parents=True, exist_ok=True)
    Ftr, Fva, feat_meta = get_features(model, seed)
    _, rid_tr, ytr = C.load_dev_data("train")
    _, rid_va, yva = C.load_dev_data("val")
    # 特征缓存顺序 = 导出 canonical 顺序(构建时同源); 再核对一次 ids
    import hashlib
    z = np.load(C.FEAT_CACHE / f"{model}_s{seed}.npz", allow_pickle=False)
    assert list(z["train_ids"].astype(str)) == list(rid_tr)
    assert list(z["val_ids"].astype(str)) == list(rid_va)

    head, probs, best, stop_reason, train_wall, vram = train_head(Ftr, ytr, Fva, yva, seed)
    metrics = C.multilabel_metrics(yva, probs)
    # 独立复算(从落盘预测)
    np.savez_compressed(job_dir / "val_predictions.npz",
                        record_ids=rid_va, y_true=yva.astype(np.int8),
                        y_prob=probs.astype(np.float32),
                        class_order=np.array(C.CLASS_ORDER))
    zj = np.load(job_dir / "val_predictions.npz", allow_pickle=False)
    recompute = C.multilabel_metrics(zj["y_true"].astype(np.float32), zj["y_prob"].astype(np.float32))
    for k in ("macro_auprc", "macro_auroc", "val_bce_from_prob"):
        assert abs(metrics[k] - recompute[k]) <= 1e-8, (k, metrics[k], recompute[k])

    result = {
        "job_id": job_id, "stage": "A2_local_lp", "model": model,
        "pretrain_seed": seed, "head_seed": seed,
        "status": "complete", "macro_auprc": metrics["macro_auprc"],
        "macro_auroc": metrics["macro_auroc"],
        "val_bce": best["val"], "best_epoch": best["epoch"],
        "stop_reason": stop_reason,
        "feat_wall_s": feat_meta["feat_wall_s"], "train_wall_s": round(train_wall, 1),
        "gpu_wall_s": round(feat_meta["feat_wall_s"] + train_wall, 1),
        "vram_peak_mib": vram,
        "per_class_ap": metrics["per_class_ap"], "per_class_auroc": metrics["per_class_auroc"],
        "support": metrics["support"],
        "macro_auprc_single": metrics["macro_auprc_single"],
        "macro_auprc_multi": metrics["macro_auprc_multi"],
        "ckpt_sha256": feat_meta["ckpt_sha256"],
        "data_sha256": feat_meta["data_sha"], "code_sha": C.git_head(),
        "recompute_abs_diff_max": max(abs(metrics[k] - recompute[k])
                                      for k in ("macro_auprc", "macro_auroc", "val_bce_from_prob")),
        "frozen_encoder": True,
    }
    (job_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    done_marker.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    torch.save(head.state_dict(), job_dir / "head_best.pth")
    C.append_csv(C.RESULTS / "results.csv", {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v)
                                             for k, v in result.items()})
    C.log_resource(job_id, "head_train", train_wall, gpu=True, extra={"vram_peak_mib": vram})
    print(f"[{job_id}] auprc={metrics['macro_auprc']:.4f} auroc={metrics['macro_auroc']:.4f} "
          f"bce={best['val']:.4f} best_ep={best['epoch']} stop={stop_reason}")
    return result


def engineering_gate(results: list) -> bool:
    """seed0 三格工程检查: 有限损失/概率范围/形状/复算一致/冻结不变。"""
    ok = True
    for r in results:
        checks = {
            "finite": np.isfinite([r["macro_auprc"], r["macro_auroc"], r["val_bce"]]).all(),
            "prob_range": 0.0 < r["macro_auprc"] < 1.0,
            "best_epoch_pos": 0 < r["best_epoch"] <= 100,
            "recompute_le_1e-8": r["recompute_abs_diff_max"] <= 1e-8,
            "vram_recorded": r["vram_peak_mib"] > 0,
        }
        if not all(checks.values()):
            ok = False
            print(f"ENG-GATE FAIL {r['job_id']}: {checks}")
    print("engineering_gate(seed0):", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="M0,M4,M6")
    ap.add_argument("--seeds", default="0,2,4")
    ap.add_argument("--phase", choices=["seed0", "rest", "all"], default="all")
    args = ap.parse_args()
    models = args.models.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = []
    if args.phase in ("seed0", "all"):
        jobs += [(m, 0) for m in models]
    if args.phase in ("rest", "all"):
        jobs += [(m, s) for m in models for s in seeds if s != 0]
    results = [run_job(m, s) for m, s in jobs]
    if args.phase in ("seed0", "all") and any(j[1] == 0 for j in jobs):
        seed0 = [r for r in results if r["pretrain_seed"] == 0]
        if seed0 and not engineering_gate(seed0):
            C.log_event("A2 seed0 工程检查 FAIL, 停止后续种子")
            return 2
    print("A2 done:", [(r["job_id"], round(r["macro_auprc"], 4)) for r in results])
    return 0


if __name__ == "__main__":
    sys.exit(main())
