# -*- coding: utf-8 -*-
"""thesis-inc R-HL: HeartLang 作者协议线性探查(REF 旁线, 1 作业)。

与 P0 A4 的区别(按本批 protocol.references.heartlang):
- split/标签: 作者 icbeb_train.csv(4950)/icbeb_val.csv(551) 与作者列序
  AFIB,VPC,NORM,1AVB,CRBBB,STE,PAC,CLBBB,STD; 100% 标签, seed0。
- backbone train 模式(dropout 开)但参数冻结(作者 train() 语义);
  验证/导出用 eval()。
- 选模 = max val macro-AUROC(作者口径); 100ep 固定, 无早停。
- AdamW lr0.005/wd0.05(wd 仅 weight 组), warmup10(1e-6 起)+cosine(至1e-5), batch256。
- 不加载作者 test; 数据目录断言无 test_* 文件; 分表记录, 不并入主 SSL 结果。
信号管线复用 P0 复刻: 前2500点->零填5000->minmax[-3,3]->100Hz->QRSTokenizer(作者类)。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.thesis_inc import common as C
from tools.thesis_p0 import heartlang as HL0  # 复用已核验的复刻件

CSV_DIR = HL0.REFS / "datasets" / "dataset_preprocess" / "CPSC2018"
OUT = C.CKPT_ROOT / "ref_heartlang_data"
JOB_ID = "ref_heartlang_s0"
CSV_COLS = ["AFIB", "VPC", "NORM", "1AVB", "CRBBB", "STE", "PAC", "CLBBB", "STD"]  # 作者 csv 原始列序


def read_author_csv(split: str):
    import csv
    f = CSV_DIR / f"icbeb_{split}.csv"
    rids, labels = [], []
    with open(f, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rids.append(row["filename"])
            labels.append([int(row[c]) for c in CSV_COLS])
    return rids, np.asarray(labels, dtype=np.float32)


def build_data():
    import scipy.io
    from scipy.io import loadmat
    if (OUT / "train_labels.npy").exists():
        print("ref_hl data already built")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    stats = {}
    for split in ("train", "val"):
        rids, y = read_author_csv(split)
        arrs = np.zeros((len(rids), 12, 1000))
        for i, rid in enumerate(rids):
            v_raw = loadmat(str(HL0.raw_mat_path(rid)))["val"].astype(np.float64)
            assert float(v_raw.max()) != float(v_raw.min()), f"flat record {rid}"
            arrs[i] = HL0.author_pipeline(v_raw)
        assert np.isfinite(arrs).all()
        sys.path.insert(0, str(HL0.REFS))
        from QRSTokenizer import QRSTokenizer
        tok = QRSTokenizer(fs=100, max_len=256, token_len=96,
                           save_path=str(OUT) + "\\", stage=split,
                           used_channels=list(range(12)))
        tok.forward(arrs, plot=False)
        np.save(OUT / f"{split}_labels.npy", y)
        np.save(OUT / f"{split}_record_ids.npy", np.array(rids))
        stats[split] = {"n": len(rids)}
        print(f"[ref_hl build] {split}: {len(rids)} tokenized")
    leak = list(OUT.glob("test_*"))
    assert not leak, f"test files present: {leak}"
    (OUT / "manifest.json").write_text(json.dumps({
        "ts": C.bj_str(), "author_commit": "a08afd8117e813fee212a1ac283db4a7140a4669",
        "split": {"train": 4950, "val": 551}, "class_order": CSV_COLS,
        "csv_sha256": {s: C.sha256_of(CSV_DIR / f"icbeb_{s}.csv") for s in ("train", "val")},
        "files_sha256": {f.name: C.sha256_of(f) for f in sorted(OUT.glob("*.npy"))},
        "wall_s": round(time.perf_counter() - t0, 1)}, ensure_ascii=False, indent=1),
        encoding="utf-8")


def tensors(split: str):
    data = np.load(OUT / f"{split}_data.npy")
    labels = np.load(OUT / f"{split}_labels.npy")
    chans = np.load(OUT / f"{split}_data_in_chans.npy").astype(np.int64)
    times = np.load(OUT / f"{split}_data_in_times.npy").astype(np.int64)
    rids = np.load(OUT / f"{split}_record_ids.npy")
    return (torch.from_numpy(data).float(), torch.from_numpy(labels).float(),
            torch.from_numpy(chans), torch.from_numpy(times), rids.astype(str))


def train():
    job_dir = C.JOBS / JOB_ID
    if (job_dir / "DONE.json").exists():
        return json.loads((job_dir / "DONE.json").read_text(encoding="utf-8"))
    job_dir.mkdir(parents=True, exist_ok=True)
    assert C.sha256_of(HL0.WEIGHT) == HL0.WEIGHT_SHA
    assert not list(OUT.glob("test_*"))
    mf = HL0.import_heartlang()
    C.set_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = mf.HeartLang_finetune_base(num_classes=9).to(dev)
    ckpt = torch.load(HL0.WEIGHT, map_location="cpu", weights_only=False)
    model_key = next((k for k in ("model", "model_ema", "module", "state_dict") if k in ckpt), None)
    ckpt_model = ckpt[model_key] if model_key else ckpt
    missing, unexpected, excluded = HL0.load_state_dict_backbone(model, ckpt_model)
    HL0.freeze_except_prefix(model, "mlp_head")
    backbone_fp0 = HL0.hash_backbone(model)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    Xtr, ytr, ctr, ttr, rid_tr = tensors("train")
    Xva, yva, cva, tva, rid_va = tensors("val")
    assert len(rid_tr) == 4950 and len(rid_va) == 551
    from torch.utils.data import DataLoader, TensorDataset
    train_loader = DataLoader(TensorDataset(Xtr, ytr, ctr, ttr), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, yva, cva, tva), batch_size=512, shuffle=False)

    head_w = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim > 1]
    head_b = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": head_w, "weight_decay": 0.05},
                             {"params": head_b, "weight_decay": 0.0}],
                            lr=0.005, betas=(0.9, 0.999), eps=1e-8)
    niter = len(train_loader)
    lr_sched = HL0.cosine_schedule(0.005, 1e-5, 100, niter, 10, 1e-6)
    loss_fn = nn.BCEWithLogitsLoss()
    best = {"epoch": -1, "auroc": -1.0, "state": None}
    gstep = 0
    t0 = time.perf_counter()
    for ep in range(1, 101):
        model.train()  # 作者 train 模式: backbone dropout 开(参数仍冻结)
        for xb, yb, cb, tb in train_loader:
            for gp in opt.param_groups:
                gp["lr"] = float(lr_sched[gstep])
            xb, yb, cb, tb = xb.to(dev), yb.to(dev), cb.to(dev), tb.to(dev)
            loss = loss_fn(model(xb, cb, tb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            gstep += 1
        model.eval()
        with torch.no_grad():
            probs = []
            for xb, yb, cb, tb in val_loader:
                probs.append(torch.sigmoid(model(xb.to(dev), cb.to(dev), tb.to(dev))).cpu().numpy())
        probs = np.concatenate(probs)
        auroc = C.multilabel_metrics(yva.numpy(), probs)["macro_auroc"]
        if auroc > best["auroc"]:  # max val macro-AUROC, 平局取早
            best = {"epoch": ep, "auroc": auroc,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "probs": probs}
        print(json.dumps({"ep": ep, "val_macro_auroc": round(auroc, 5),
                          "best_ep": best["epoch"]}), flush=True)
    wall = time.perf_counter() - t0
    model.load_state_dict(best["state"])
    model.eval()
    assert HL0.hash_backbone(model) == backbone_fp0, "backbone weights changed"
    metrics = C.multilabel_metrics(yva.numpy(), best["probs"])
    np.savez_compressed(job_dir / "val_predictions.npz", record_ids=rid_va,
                        y_true=yva.numpy().astype(np.int8),
                        y_prob=best["probs"].astype(np.float32),
                        class_order=np.array(CSV_COLS))
    zj = np.load(job_dir / "val_predictions.npz", allow_pickle=False)
    recompute = C.multilabel_metrics(zj["y_true"].astype(np.float32), zj["y_prob"].astype(np.float32))
    diffs = max(abs(metrics[k] - recompute[k])
                for k in ("macro_auprc", "macro_auroc", "val_bce_from_prob"))
    assert diffs <= 1e-8
    vram = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if torch.cuda.is_available() else 0.0
    torch.save({"full_state": best["state"], "best_epoch": best["epoch"]}, job_dir / "model_best.pth")
    result = {"job_id": JOB_ID, "stage": "REF", "kind": "reference_linear_probe",
              "model": "HeartLang_author_protocol", "status": "complete",
              "train_records": 4950, "val_records": 551,
              "class_order": CSV_COLS, "seed": 0, "label_fraction": 1.0,
              "trainable": "linear(mlp_head)", "trainable_params": n_trainable,
              "macro_auprc": metrics["macro_auprc"], "macro_auroc": metrics["macro_auroc"],
              "val_bce_from_prob": metrics["val_bce_from_prob"],
              "best_epoch": best["epoch"], "stop_reason": "fixed_100_epochs",
              "train_wall_s": round(wall, 1), "vram_peak_mib": vram,
              "weight_sha256": HL0.WEIGHT_SHA, "author_commit": "a08afd8",
              "backbone_mode": "train(dropout on) during training, eval for val/export; params frozen+verified",
              "selection": "max val macro-AUROC (author), tie->earlier",
              "recompute_abs_diff_max": diffs, "test_loader_called": 0,
              "git_head": C.git_head()}
    (job_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    (job_dir / "DONE.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    C.add_resource("ref_gpu", wall, JOB_ID)
    C.count_job_start(JOB_ID, "REF")
    C.append_csv(C.RESULTS / "results.csv", {
        "job_id": JOB_ID, "stage": "REF", "kind": "reference_linear_probe",
        "recipe": "HeartLang_author", "seed": 0, "pt_epochs": "",
        "status": "complete", "stop_reason": "fixed_100_epochs",
        "wall_s": result["train_wall_s"],
        "macro_auprc": result["macro_auprc"], "macro_auroc": result["macro_auroc"],
        "encoder_sha256": HL0.WEIGHT_SHA})
    print(f"[{JOB_ID}] auprc={result['macro_auprc']:.4f} auroc={result['macro_auroc']:.4f} "
          f"best_ep={result['best_epoch']}")
    return result


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build-data":
        build_data()
    elif cmd == "train":
        train()
    else:
        print("usage: ref_heartlang.py build-data|train")
