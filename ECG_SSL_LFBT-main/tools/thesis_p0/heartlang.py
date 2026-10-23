"""thesis-p0-20261023 A4: HeartLang 公开预训练权重的单任务 LP 验证(1 作业).

作者原生管线(commit a08afd8, 来源锁 9/9 核验):
- 信号: 原始 12 导联 @500Hz -> 前 2500 点 -> 零填 5000 -> 整记录 min-max 到 [-3,3]
  -> resample 100Hz(1000 点); QRSTokenizer(fs=100, max_len=256, token_len=96,
  used_channels=12 全导, xqrs 检 lead I) 直接复用作者类(同 clone 导入, 不改码)。
- 标签/split: 用本项目冻结 split(4809/683)与 multihot(类序=本项目 NSR..STE;
  作者 csv 列序 AFIB,VPC,NORM,1AVB,CRBBB,STE,PAC,CLBBB,STD 仅作映射记录,
  头为新初始化, 列序等价于置换)。
- 训练: trainable=linear(仅 mlp_head), AdamW(lr5e-3 调度=warmup10ep 1e-6->5e-3
  再 cosine->1e-5, betas(0.9,0.999) eps1e-8, wd0.05 仅 weight 组), batch256,
  <=100ep, 选模=最小元素加权验证 BCE(平局更早), patience15, seed0, float32。
- 协议要求的偏差(相对作者): backbone 全程 eval(dropout 关闭) —— 任务书§6
  "不能以随机 dropout 状态冒充冻结 LP"; 作者入口整模型 train()。
- 测试隔离: 数据目录仅含 train/val 文件, 启动断言无 test_* 文件; 训练进程
  无任何 test 加载路径; 结果记录 test_loader_called=0。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from scipy.signal import resample

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.thesis_p0 import common as C

HL = C.PROTOCOL["heartlang"]
REFS = Path(r"D:\thesis_p0_20261023_backup\refs\HeartLang")
WEIGHT = Path(r"D:\thesis_p0_20261023_backup\heartlang_weights\checkpoint-200.pth")
WEIGHT_SHA = "9dc89d72a3f1cacd941892103eaa523d44e373164cd4fadfb975907cc65864f6"
RAW_ROOT = Path(r"F:\数据集\classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2\training\cpsc_2018")
OUT = C.ADAPT / "heartlang"
AUTHOR_CLASS_ORDER = ["AF", "PVC", "NSR", "IAVB", "RBBB", "STE", "PAC", "LBBB", "STD"]  # csv: AFIB,VPC,NORM,1AVB,CRBBB,STE,PAC,CLBBB,STD
OUR_TO_AUTHOR = {c: AUTHOR_CLASS_ORDER.index(c) for c in C.CLASS_ORDER}
JOB_ID = "heartlang_lp_s0"


def raw_mat_path(rid: str) -> Path:
    cands = list(RAW_ROOT.glob(f"g*/{rid}.mat"))
    if len(cands) != 1:
        raise FileNotFoundError(f"raw mat for {rid}: {cands}")
    return cands[0]


def author_pipeline(v_raw: np.ndarray) -> np.ndarray:
    """make_cpsc_dataset.py 逐行等价: [:, :2500] -> pad 5000 -> minmax[-3,3] -> 100Hz。"""
    v = v_raw[:, :2500]
    v = np.pad(v, ((0, 0), (0, 5000 - v.shape[1])), "constant", constant_values=0)
    mn, mx = np.min(v), np.max(v)
    v = (3 - (-3)) * (v - mn) / (mx - mn) + (-3)
    n_target = int(v.shape[1] * (100 / 500))
    out = np.zeros((v.shape[0], n_target))
    for i in range(v.shape[0]):
        out[i] = resample(v[i], n_target)
    return out


def build_data():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    stats = {}
    n_degenerate = 0
    for split in ("train", "val"):
        _, rids, y = C.load_dev_data(split)
        arrs = np.zeros((len(rids), 12, 1000))
        for i, rid in enumerate(rids):
            v_raw = loadmat(str(raw_mat_path(rid)))["val"].astype(np.float64)
            if float(v_raw.max()) == float(v_raw.min()):
                n_degenerate += 1
            arrs[i] = author_pipeline(v_raw)
        assert np.isfinite(arrs).all(), "non-finite after author normalize (flat-record degenerate)"
        assert n_degenerate == 0, f"{n_degenerate} flat raw records (min==max) would divide by zero"
        sys.path.insert(0, str(REFS))
        from QRSTokenizer import QRSTokenizer
        tok = QRSTokenizer(fs=100, max_len=256, token_len=96,
                           save_path=str(OUT) + "\\", stage=split,
                           used_channels=list(range(12)))
        tok.forward(arrs, plot=False)  # 写 {split}_data/_in_chans/_in_times .npy
        np.save(OUT / f"{split}_labels.npy", y.astype(np.float32))  # 本项目类序
        np.save(OUT / f"{split}_record_ids.npy", np.array(rids))
        stats[split] = {"n": len(rids)}
        print(f"[heartlang build] {split}: {len(rids)} tokenized")
    test_leak = list(OUT.glob("test_*"))
    assert not test_leak, f"test files present in adaptation dir: {test_leak}"
    sha = {f.name: C.sha256_of(f) for f in sorted(OUT.glob("*.npy"))}
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "author_commit": "a08afd8117e813fee212a1ac283db4a7140a4669",
        "signal_pipeline": "first 2500 @500Hz -> zero-pad 5000 -> record minmax [-3,3] -> resample 100Hz -> QRSTokenizer(fs100,max256,token96,12ch,xqrs lead I)",
        "labels": "our frozen split + multihot, class order NSR,AF,IAVB,LBBB,RBBB,PAC,PVC,STD,STE",
        "author_csv_class_order": AUTHOR_CLASS_ORDER,
        "files_sha256": sha, "wall_s": round(time.perf_counter() - t0, 1),
    }
    (OUT / "heartlang_data_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    C.log_resource("a4_build_data", "tokenize_train_val", manifest["wall_s"], gpu=False,
                   extra={k: v["n"] for k, v in stats.items()})
    print("build done", stats)


class HLValTrain:
    """train/val-only 包装: 无 test 数据集构造路径。"""

    @staticmethod
    def tensors(split: str):
        data = np.load(OUT / f"{split}_data.npy")
        labels = np.load(OUT / f"{split}_labels.npy")
        chans = np.load(OUT / f"{split}_data_in_chans.npy").astype(np.int64)
        times = np.load(OUT / f"{split}_data_in_times.npy").astype(np.int64)
        rids = np.load(OUT / f"{split}_record_ids.npy")
        return (torch.from_numpy(data).float(), torch.from_numpy(labels).float(),
                torch.from_numpy(chans), torch.from_numpy(times), rids.astype(str))


def cosine_schedule(base, final, epochs, niter, warmup_epochs, start_warm):
    import math
    warm = np.linspace(start_warm, base, warmup_epochs * niter) if warmup_epochs else np.array([])
    iters = np.arange(epochs * niter - len(warm))
    sched = np.array([final + 0.5 * (base - final) * (1 + math.cos(math.pi * i / len(iters)))
                      for i in iters])
    return np.concatenate([warm, sched])


def train():
    assert C.sha256_of(WEIGHT) == WEIGHT_SHA, "HeartLang weight sha mismatch"
    leak = list(OUT.glob("test_*"))
    assert not leak, f"test files present: {leak}"
    job_dir = C.JOBS / JOB_ID
    if (job_dir / "DONE.json").exists():
        return json.loads((job_dir / "DONE.json").read_text(encoding="utf-8"))
    job_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REFS))
    import modeling_finetune as mf
    from utils.utils import load_state_dict, freeze_except_prefix

    C.set_seed(HL["training_seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    model = mf.HeartLang_finetune_base(num_classes=9).to(dev)
    ckpt = torch.load(WEIGHT, map_location="cpu", weights_only=False)
    model_key = next((k for k in ("model", "model_ema", "module", "state_dict") if k in ckpt), None)
    ckpt_model = ckpt[model_key] if model_key else ckpt
    load_state_dict(model, ckpt_model, prefix="")
    # 严格核对: 除 mlp_head(新头)外, backbone 权重必须全部来自 ckpt
    want = {k: v for k, v in model.state_dict().items() if not k.startswith("mlp_head")}
    src = {f"backbone.{k}": v for k, v in ckpt_model.items()}
    missing = [k for k in want if k not in src]
    assert not missing, f"backbone keys missing from ckpt: {missing[:8]}"
    n_loaded = sum(1 for k in want if k in src and want[k].shape == src[k].shape)
    freeze_except_prefix(model, "mlp_head")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_fp0 = hash_backbone(model)

    Xtr, ytr, ctr, ttr, rid_tr = HLValTrain.tensors("train")
    Xva, yva, cva, tva, rid_va = HLValTrain.tensors("val")
    assert len(rid_tr) == 4809 and len(rid_va) == 683

    from torch.utils.data import DataLoader, TensorDataset
    bs = HL["batch_size"]
    train_loader = DataLoader(TensorDataset(Xtr, ytr, ctr, ttr), batch_size=bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, yva, cva, tva), batch_size=512, shuffle=False)

    head_w = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim > 1]
    head_b = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim <= 1]
    opt = torch.optim.AdamW([
        {"params": head_w, "weight_decay": HL["weight_decay"]},
        {"params": head_b, "weight_decay": 0.0},
    ], lr=HL["learning_rate"], betas=(0.9, 0.999), eps=1e-8)
    niter = len(train_loader)
    lr_sched = cosine_schedule(HL["learning_rate"], 1e-5, HL["epochs_max"], niter,
                               HL["warmup_epochs"], 1e-6)
    loss_fn = nn.BCEWithLogitsLoss()
    best = {"epoch": -1, "val": float("inf"), "state": None}
    stop_reason = "max_epochs"
    no_improve = 0
    gstep = 0
    t0 = time.perf_counter()
    for ep in range(1, HL["epochs_max"] + 1):
        model.mlp_head.train()
        model.backbone.eval()  # 协议要求: 冻结 LP 下 backbone 无 dropout 更新
        for xb, yb, cb, tb in train_loader:
            for gp in opt.param_groups:
                gp["lr"] = float(lr_sched[gstep])
            xb, yb, cb, tb = xb.to(dev), yb.to(dev), cb.to(dev), tb.to(dev)
            loss = loss_fn(model(xb, cb, tb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            gstep += 1
        model.eval()
        with torch.no_grad():
            tot, el = 0.0, 0
            for xb, yb, cb, tb in val_loader:
                lg = model(xb.to(dev), cb.to(dev), tb.to(dev))
                tot += float(nn.functional.binary_cross_entropy_with_logits(
                    lg, yb.to(dev), reduction="sum"))
                el += lg.numel()
            val_bce = tot / el
        if val_bce < best["val"]:
            best = {"epoch": ep, "val": val_bce,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 15:
                stop_reason = "early_stop_patience15"
                break
    train_wall = time.perf_counter() - t0
    model.load_state_dict(best["state"]); model.eval()
    assert hash_backbone(model) == backbone_fp0, "backbone weights changed during LP"
    with torch.no_grad():
        probs = []
        for xb, yb, cb, tb in val_loader:
            probs.append(torch.sigmoid(model(xb.to(dev), cb.to(dev), tb.to(dev))).cpu().numpy())
    probs = np.concatenate(probs)
    vram = round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else 0.0
    yva_np = yva.numpy()
    metrics = C.multilabel_metrics(yva_np, probs)
    np.savez_compressed(job_dir / "val_predictions.npz", record_ids=rid_va,
                        y_true=yva_np.astype(np.int8), y_prob=probs.astype(np.float32),
                        class_order=np.array(C.CLASS_ORDER))
    zj = np.load(job_dir / "val_predictions.npz", allow_pickle=False)
    recompute = C.multilabel_metrics(zj["y_true"].astype(np.float32), zj["y_prob"].astype(np.float32))
    for k in ("macro_auprc", "macro_auroc", "val_bce_from_prob"):
        assert abs(metrics[k] - recompute[k]) <= 1e-8, (k, metrics[k], recompute[k])
    torch.save({"full_state": best["state"], "best_epoch": best["epoch"]},
               job_dir / "model_best.pth")
    result = {
        "job_id": JOB_ID, "stage": "A4_heartlang", "model": "HeartLang_pretrained",
        "training_seed": HL["training_seed"], "label_fraction": HL["label_fraction"],
        "trainable": "linear(mlp_head)", "trainable_params": n_trainable,
        "status": "complete", "macro_auprc": metrics["macro_auprc"],
        "macro_auroc": metrics["macro_auroc"], "val_bce": best["val"],
        "best_epoch": best["epoch"], "stop_reason": stop_reason,
        "train_wall_s": round(train_wall, 1), "gpu_wall_s": round(train_wall, 1),
        "vram_peak_mib": vram,
        "weight_sha256": WEIGHT_SHA, "author_commit": "a08afd8",
        "backbone_keys_loaded": n_loaded, "backbone_missing": missing,
        "backbone_frozen_verified": True, "test_loader_called": 0,
        "selection": "min val BCE (deviation from author AUC selection, protocol-fixed)",
        "code_sha": C.git_head(),
        "recompute_abs_diff_max": max(abs(metrics[k] - recompute[k])
                                      for k in ("macro_auprc", "macro_auroc", "val_bce_from_prob")),
    }
    (job_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    (job_dir / "DONE.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    C.log_resource(JOB_ID, "train", train_wall, gpu=True, extra={"vram_peak_mib": vram})
    print(f"[{JOB_ID}] auprc={metrics['macro_auprc']:.4f} auroc={metrics['macro_auroc']:.4f} "
          f"bce={best['val']:.4f} best_ep={best['epoch']} stop={stop_reason}")
    return result


def hash_backbone(model) -> str:
    import hashlib
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        if not k.startswith("mlp_head"):
            h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-data", "train"])
    args = ap.parse_args()
    if args.cmd == "build-data":
        build_data()
        return 0
    train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
