# -*- coding: utf-8 -*-
"""thesis-inc 统一线性探查 (LP): P0 隔离 CPSC train/val, 原生九类 multihot。

协议 (protocol.json lp): 冻结编码器 eval, 共同整段编码器 8x64=512 维特征,
Linear(512,9), BCEWithLogitsLoss, Adam lr1e-3/b(0.9,0.999)/eps1e-8/wd0,
batch128(末批保留), <=100ep, patience15, 严格最小元素加权 val BCE(平局取早),
head_seed=pt_seed。独立复算 |Δ|<=1e-8; 逐记录预测落盘; 无 test 访问路径。
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.vgg_1d import VGG16
from tools.thesis_inc import common as C

HP = {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8, "wd": 0.0,
      "batch_size": 128, "epochs_max": 100, "patience": 15}
CLASS_ORDER = ["NSR", "AF", "IAVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]


def build_encoders(ckpt_path: Path, trc: int = 1):
    blob = torch.load(ckpt_path, map_location="cpu")
    encs = []
    for i in range(8):
        enc = VGG16(ch_in=1, alpha=0.125, blur_pool=0, pool_power=0.0, trc=trc)
        missing, unexpected = enc.load_state_dict(
            blob["backbone_state_dict_list"][i], strict=False)
        assert missing == ["fc.weight", "fc.bias"] and unexpected == [], (missing, unexpected)
        body = torch.nn.Sequential(*list(enc.children())[:-1])
        body.eval()
        for p in body.parameters():
            p.requires_grad_(False)
        encs.append(body)
    return encs


def param_fingerprint(encs) -> str:
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
            f = enc(xb[:, [j], :])
            feats.append(f.squeeze(-1))
        out[s:s + batch] = torch.cat(feats, dim=1).cpu().numpy()
    return out


def train_head(Ftr, ytr, Fva, yva, seed: int):
    C.set_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    head = nn.Linear(512, 9).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=HP["lr"], betas=HP["betas"],
                           eps=HP["eps"], weight_decay=HP["wd"])
    loss_fn = nn.BCEWithLogitsLoss()
    n = Ftr.shape[0]
    bs = HP["batch_size"]
    best = {"epoch": -1, "val": float("inf"), "state": None}
    stop_reason = "max_epochs"
    t0 = time.perf_counter()
    no_improve = 0
    for ep in range(1, HP["epochs_max"] + 1):
        head.train()
        perm = torch.randperm(n)
        for s in range(0, n, bs):  # drop_last=False
            idx = perm[s:s + bs]
            xb = torch.from_numpy(Ftr[idx.numpy()]).to(dev)
            yb = torch.from_numpy(ytr[idx.numpy()]).to(dev)
            loss = loss_fn(head(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            tot, el = 0.0, 0
            for s in range(0, Fva.shape[0], 1024):
                xb = torch.from_numpy(Fva[s:s + 1024]).to(dev)
                yb = torch.from_numpy(yva[s:s + 1024]).to(dev)
                lg = head(xb)
                tot += float(nn.functional.binary_cross_entropy_with_logits(
                    lg, yb, reduction="sum"))
                el += lg.numel()
            val_bce = tot / el
        if val_bce < best["val"]:
            best = {"epoch": ep, "val": val_bce,
                    "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= HP["patience"]:
                stop_reason = f"early_stop_patience{HP['patience']}"
                break
    wall = time.perf_counter() - t0
    head.load_state_dict(best["state"])
    head.eval()
    with torch.no_grad():
        probs = []
        for s in range(0, Fva.shape[0], 1024):
            lg = head(torch.from_numpy(Fva[s:s + 1024]).to(dev))
            probs.append(torch.sigmoid(lg).cpu().numpy())
    vram = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if torch.cuda.is_available() else 0.0
    return head, np.concatenate(probs), best, stop_reason, wall, vram


def run_lp_job(cfg: dict) -> dict:
    """cfg: {job_id, stage, pt_job_id, recipe, seed, encoder_path, encoder_sha256}。"""
    job_id = cfg["job_id"]
    job_dir = C.JOBS / job_id
    done = job_dir / "DONE.json"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    job_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc_path = Path(cfg["encoder_path"])
    enc_sha = C.sha256_of(enc_path)
    assert enc_sha == cfg["encoder_sha256"], "编码器 SHA 与 PT 产出不一致"

    feat_cache = job_dir / "feat.npz"
    Xtr, rid_tr, ytr = C.load_dev_data("train")
    Xva, rid_va, yva = C.load_dev_data("val")
    dsha = {sp: C.sha256_of(C.ADAPT / f"{sp}_data.npz") for sp in ("train", "val")}
    if feat_cache.exists():
        z = np.load(feat_cache, allow_pickle=False)
        meta = json.loads(str(z["meta"].astype(str)))
        assert meta["encoder_sha256"] == enc_sha and meta["data_sha"] == dsha
        Ftr, Fva = z["X_train"], z["X_val"]
        feat_wall = meta["feat_wall_s"]
    else:
        tt = time.perf_counter()
        encs = build_encoders(enc_path)
        if dev == "cuda":
            encs = [e.to("cuda") for e in encs]
        fp_before = param_fingerprint(encs)
        Ftr = extract_features(encs, Xtr)
        Fva = extract_features(encs, Xva)
        assert param_fingerprint(encs) == fp_before, "冻结编码器参数被改动"
        feat_wall = time.perf_counter() - tt
        meta = {"encoder_sha256": enc_sha, "data_sha": dsha,
                "feat_wall_s": round(feat_wall, 1),
                "train_ids_sha": hashlib.sha256("\n".join(rid_tr).encode()).hexdigest(),
                "val_ids_sha": hashlib.sha256("\n".join(rid_va).encode()).hexdigest()}
        np.savez(feat_cache, X_train=Ftr, X_val=Fva,
                 train_ids=rid_tr, val_ids=rid_va, meta=np.array(json.dumps(meta)))
    assert list(np.load(feat_cache, allow_pickle=False)["train_ids"].astype(str)) == list(rid_tr)
    assert list(np.load(feat_cache, allow_pickle=False)["val_ids"].astype(str)) == list(rid_va)

    head, probs, best, stop_reason, train_wall, vram = train_head(Ftr, ytr, Fva, yva, cfg["seed"])
    metrics = C.multilabel_metrics(yva, probs)
    np.savez_compressed(job_dir / "val_predictions.npz",
                        record_ids=rid_va, y_true=yva.astype(np.int8),
                        y_prob=probs.astype(np.float32),
                        class_order=np.array(CLASS_ORDER))
    zj = np.load(job_dir / "val_predictions.npz", allow_pickle=False)
    recompute = C.multilabel_metrics(zj["y_true"].astype(np.float32),
                                     zj["y_prob"].astype(np.float32))
    diffs = {k: abs(metrics[k] - recompute[k])
             for k in ("macro_auprc", "macro_auroc", "val_bce_from_prob")}
    assert max(diffs.values()) <= 1e-8, diffs

    wall = time.time() - t0
    result = {"job_id": job_id, "stage": cfg["stage"], "kind": "linear_probe",
              "recipe": cfg["recipe"], "seed": cfg["seed"],
              "pt_job_id": cfg["pt_job_id"], "status": "complete",
              "macro_auprc": metrics["macro_auprc"],
              "macro_auroc": metrics["macro_auroc"],
              "per_class_ap": metrics["per_class_ap"],
              "support": metrics["support"],
              "val_bce": best["val"], "val_bce_from_prob": metrics["val_bce_from_prob"],
              "best_epoch": best["epoch"], "stop_reason": stop_reason,
              "feat_wall_s": round(feat_wall, 1), "train_wall_s": round(train_wall, 1),
              "wall_s": round(wall, 1), "vram_peak_mib": vram,
              "encoder_sha256": enc_sha, "data_sha256": dsha,
              "recompute_abs_diff_max": max(diffs.values()),
              "frozen_encoder": True, "git_head": C.git_head()}
    torch.save(head.state_dict(), job_dir / "head_best.pth")
    (job_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    done.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    C.add_resource("gpu", wall, job_id)
    C.append_csv(C.RESULTS / "results.csv", {
        "job_id": job_id, "stage": cfg["stage"], "kind": "linear_probe",
        "recipe": cfg["recipe"], "seed": cfg["seed"], "pt_epochs": cfg.get("pt_epochs", ""),
        "status": "complete", "stop_reason": stop_reason,
        "wall_s": result["wall_s"],
        "macro_auprc": result["macro_auprc"], "macro_auroc": result["macro_auroc"],
        "encoder_sha256": enc_sha})
    print(f"[{job_id}] auprc={result['macro_auprc']:.4f} auroc={result['macro_auroc']:.4f} "
          f"bce={best['val']:.4f} best_ep={best['epoch']} stop={stop_reason}")
    return result
