# -*- coding: utf-8 -*-
"""thesis-inc R-TK: ecg_tokenizer 三行参考验证(patching/median_beat/heartlang)。

数据身份(记录性偏差, 见 decisions/events):
- 作者 fold 复现: ecg_id/age/sex/标签取自 HeartLang 仓库三份 icbeb CSV
  (6877 条, ecg_id 为 1..6877 双射; 与本地 2020 镜像 .hea Age/Sex 逐条核对
  294/300 全一致, 6 处差异均为 >89 岁审查伪影(镜像封顶 92 vs 原值 90-94),
  分层年龄桶同落 '>=80', fold 分配不受影响)。
- 作者 stratisfy_df(含 Sechidis 多标签分层, np.random.seed(0))原样导入运行。
- 信号取自本地 2020 PhysioNet 镜像(同一 CPSC 源重编码, wfdb mat), 未走作者
  records100 的 16bit wfdb 往返(浮点路径, 记录偏差); zoom(0.2) 500->100Hz、
  data_slice(->1000x12)、select_data、train-fitted StandardScaler 均作者原码。
- 隔离信号清单: 仅 fold<=9 记录的信号被读取; fold10 信号零读取; fold10 的
  标签/年龄/性别元数据仅作为作者 fold 复现的分层输入, 不进任何训练/选模/评价。
训练(作者语义): Transformer d128/e3/h8/d_ff96/gelu/dropout0.1; RAdam lr1e-3;
BCEWithLogits; clip 4.0; <=300ep; EarlyStopping(作者语义, val macro-AUC,
patience15); seed2025 fix_seed(作者: random/torch/np 三处); 无 scheduler;
有效 batch 256 = microbatch32 x accum8(协议预授权, 末组按实际元素数分母;
模型无 BatchNorm, microbatch 无统计影响); OOM 则三行统一 micro16 x accum16。
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.thesis_inc import common as C

REF = Path(r"D:\thesis_p0_20261023_backup\refs\ecg_tokenizer")
HL_CSV = Path(r"D:\thesis_p0_20261023_backup\refs\HeartLang\datasets\dataset_preprocess\CPSC2018")
MIRROR = Path(r"F:\数据集\classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2"
              r"\training\cpsc_2018")
DATA = C.CKPT_ROOT / "ref_tokenizer_data"
TOKENIZERS = ("patching", "median_beat", "heartlang")
QRS_CACHE = {}  # 内容寻址 QRS/峰检测缓存(确定性输出, 逐位一致)


def import_author():
    """导入作者仓库模块(models/layers 包; 期间摘除本仓根避免 utils 遮蔽)。"""
    our_root = Path(__file__).resolve().parents[2]
    saved = sys.path[:]
    sys.path[:] = [str(REF)] + [p for p in sys.path
                                if Path(p or ".").resolve() != our_root.resolve()]
    try:
        import models  # noqa: F401
        from data_provider import cpsc as author_cpsc
        from models.tokenized_model import TokenizedModel
        import models.Transformer as Transformer
        return author_cpsc, TokenizedModel, Transformer
    finally:
        sys.path[:] = saved


def build_metadata():
    """复现作者 cpsc_database(strat_fold)。仅元数据; 不读信号。"""
    out = DATA / "folds.csv"
    if out.exists():
        return pd.read_csv(out)
    DATA.mkdir(parents=True, exist_ok=True)
    rows = []
    for sp in ("train", "val", "test"):
        df = pd.read_csv(HL_CSV / f"icbeb_{sp}.csv")
        df["hl_split"] = sp
        rows.append(df)
    full = pd.concat(rows, ignore_index=True)
    assert len(full) == 6877 and sorted(full.ecg_id) == list(range(1, 6878))
    full = full.sort_values("ecg_id").reset_index(drop=True)
    scp = []
    for _, r in full.iterrows():
        codes = {c: 100 for c in ("AFIB", "VPC", "NORM", "1AVB", "CRBBB", "STE", "PAC",
                                  "CLBBB", "STD") if int(r[c]) == 1}
        scp.append(codes)
    meta = pd.DataFrame({
        "ecg_id": full.ecg_id, "patient_id": full.ecg_id, "filename": full.filename,
        "age": pd.to_numeric(full.age, errors="coerce"),  # 5 条缺失 -> NaN(作者 NaN 年龄不加桶)
        "sex": full.sex.astype(int),
        "scp_codes": scp, "hl_split": full.hl_split})
    author_cpsc, _, _ = import_author()
    meta = author_cpsc.stratisfy_df(meta, "strat_fold")  # 内部 np.random.seed(0)
    keep = meta[["ecg_id", "filename", "strat_fold"]]
    keep.to_csv(out, index=False)
    n = meta.strat_fold.value_counts().sort_index().to_dict()
    (DATA / "folds_manifest.json").write_text(json.dumps({
        "ts": C.bj_str(), "records": len(meta), "fold_counts": n,
        "train_fold_le8": int((meta.strat_fold <= 8).sum()),
        "val_fold_9": int((meta.strat_fold == 9).sum()),
        "age_missing": int(meta.age.isna().sum()),
        "identity_notes": __doc__.splitlines()[2:12]}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("fold counts:", n)
    return keep


def build_caches():
    """隔离信号清单(fold<=9) -> X_train/y_train/X_val/y_val 缓存。"""
    if (DATA / "X_train.npy").exists():
        print("caches exist")
        return
    from scipy.io import loadmat
    from scipy.ndimage import zoom
    author_cpsc, _, _ = import_author()
    folds = build_metadata()
    meta = folds  # ecg_id/filename/strat_fold
    # 信号仅读 fold<=9
    keep_ids = set(meta[meta.strat_fold <= 9].ecg_id)
    name2fold = dict(zip(meta.filename, meta.strat_fold))
    t0 = time.time()
    sigs = {}
    for rid in sorted(keep_ids):
        name = meta[meta.ecg_id == rid].filename.iloc[0]
        cands = list(MIRROR.glob(f"g*/{name}.mat"))
        assert len(cands) == 1, name
        v = loadmat(str(cands[0]))["val"].astype(np.float64)  # (12, L) @500Hz
        down = np.array([zoom(ch, 0.2) for ch in v]).T  # (L/5, 12) @100Hz, 作者 zoom 语义
        sigs[rid] = down
    data_obj = np.array([sigs[r] for r in sorted(keep_ids)], dtype=object)
    labels_full = {}
    for sp in ("train", "val", "test"):
        df = pd.read_csv(HL_CSV / f"icbeb_{sp}.csv")
        for _, r in df.iterrows():
            labels_full[r["filename"]] = r
    rows_meta = meta[meta.ecg_id.isin(keep_ids)].copy()
    scp = [{c: 100 for c in ("AFIB", "VPC", "NORM", "1AVB", "CRBBB", "STE", "PAC", "CLBBB", "STD")
            if int(labels_full[r.filename][c]) == 1} for r in rows_meta.itertuples()]
    ld = pd.DataFrame({"scp_codes": scp})
    ld["all_scp"] = ld.scp_codes.apply(lambda x: list(x.keys()))
    # select_data 语义(counts>0 过滤 + mlb)
    counts = pd.Series(np.concatenate(ld.all_scp.values)).value_counts()
    counts = counts[counts > 0]
    assert len(counts) == 9, counts
    ld["all_scp"] = ld.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index))))
    ld["all_scp_len"] = ld.all_scp.apply(len)
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    mlb.fit(ld.all_scp.values)  # 作者 select_data: 先 fit 再 transform(列序=排序)
    y_all = mlb.transform(ld.all_scp.values)
    keep_mask = ld.all_scp_len.values > 0
    sl = author_cpsc.data_slice(data_obj[keep_mask])  # (N,1000,12)
    folds_kept = rows_meta.strat_fold.values[keep_mask]
    ids_kept = rows_meta.ecg_id.values[keep_mask]
    names_kept = rows_meta.filename.values[keep_mask]
    tr = folds_kept <= 8
    va = folds_kept == 9
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(np.vstack(sl[tr]).flatten()[:, np.newaxis].astype(float))
    def std(a):
        return np.array([scaler.transform(s.flatten()[:, np.newaxis]).reshape(s.shape) for s in a])
    Xtr, Xva = std(sl[tr]), std(sl[va])
    ytr, yva = y_all[tr], y_all[va]
    np.save(DATA / "X_train.npy", Xtr.astype(np.float32))
    np.save(DATA / "y_train.npy", ytr.astype(np.float32))
    np.save(DATA / "X_val.npy", Xva.astype(np.float32))
    np.save(DATA / "y_val.npy", yva.astype(np.float32))
    np.save(DATA / "train_ids.npy", names_kept[tr])
    np.save(DATA / "val_ids.npy", names_kept[va])
    import pickle
    with open(DATA / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    (DATA / "caches_manifest.json").write_text(json.dumps({
        "ts": C.bj_str(), "class_order_mlb": list(mlb.classes_),
        "train_n": int(tr.sum()), "val_n": int(va.sum()),
        "dropped_no_label": int((~keep_mask).sum()),
        "signals_read": "fold<=9 only; fold10 信号零读取",
        "wall_s": round(time.time() - t0, 1)}, ensure_ascii=False, indent=1), encoding="utf-8")
    C.add_resource("cpu", time.time() - t0, "ref_tk_build")
    print(f"caches: train {tr.sum()} val {va.sum()} mlb={list(mlb.classes_)}")


def base_args(tokenizer: str):
    """run.py 默认 + 协议覆盖。"""
    d = dict(task_name="classification", data="CPSC2018", model_id="CPSC2018",
             tokenizer=tokenizer, model="Transformer", ecg_sample_rate=100,
             seq_len=1000, label_len=48, pred_len=0, enc_in=12, dec_in=12, c_out=12,
             d_model=128, n_heads=8, e_layers=3, d_layers=1, d_ff=96, moving_avg=25,
             factor=1, dropout=0.1, activation="gelu", embed="fixed", freq="s",
             distil=True, num_workers=0, train_epochs=300, batch_size=256,
             patience=15, learning_rate=0.001, use_lead_embedding=0, use_position_embedding=1,
             patch_size=16, patch_stride=8, heartlang_max_len=256, heartlang_token_len=96,
             heartlang_used_chans=",".join(str(i) for i in range(12)),
             median_beat_window_size=60,
             median_beat_used_chans=",".join(str(i) for i in range(12)),
             median_beat_detect_lead_index=1, ecg_byte_padding_idx=0, seed=2025)
    if tokenizer == "patching":
        d["use_lead_embedding"] = 1
    elif tokenizer == "heartlang":
        d["use_lead_embedding"] = 1
    else:  # median_beat
        d["use_lead_embedding"] = 0
    return d


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def macro_auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p, average="macro"))


def train(tokenizer: str, micro_bs: int, accum: int, stop_deadline):
    job_id = f"ref_tokenizer_{tokenizer}_s2025"
    job_dir = C.JOBS / job_id
    if (job_dir / "DONE.json").exists():
        return json.loads((job_dir / "DONE.json").read_text(encoding="utf-8"))
    job_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(job_dir / "train_log.jsonl", "a", encoding="utf-8")

    def log(**kw):
        log_f.write(json.dumps({"ts": C.bj_str(), "job_id": job_id, **kw},
                               ensure_ascii=False) + "\n")
        log_f.flush()

    _, TokenizedModel, Transformer = import_author()
    random.seed(2025)
    torch.manual_seed(2025)
    np.random.seed(2025)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args = DotDict(base_args(tokenizer))
    t0 = time.time()
    model = TokenizedModel(Transformer, args, raw_seq_len=1000, raw_channels=12).float().to(dev)
    Xtr = torch.from_numpy(np.load(DATA / "X_train.npy"))
    ytr = torch.from_numpy(np.load(DATA / "y_train.npy"))
    Xva = torch.from_numpy(np.load(DATA / "X_val.npy"))
    yva = torch.from_numpy(np.load(DATA / "y_val.npy"))
    rid_va = np.load(DATA / "val_ids.npy").astype(str)
    log(event="data", n_train=int(Xtr.shape[0]), n_val=int(Xva.shape[0]),
        micro_bs=micro_bs, accum=accum)

    class DS(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X, self.y = X, y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.X[i], self.y[i]

    def collate(batch):  # 作者 uea.collate_fn 语义(等长数据无实际填充)
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        pm = torch.ones(xs.shape[0], xs.shape[1], dtype=torch.bool)
        return xs, ys, pm

    g = torch.Generator()
    g.manual_seed(2025)
    train_loader = torch.utils.data.DataLoader(
        DS(Xtr, ytr), batch_size=micro_bs, shuffle=True, num_workers=0,
        drop_last=False, generator=g)
    opt = torch.optim.RAdam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    best = {"epoch": -1, "auc": -1.0, "state": None}
    counter = 0
    stop_reason = "max_epochs_300"
    state_path = job_dir / "train_state.pth"
    start_ep = 0
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        random.setstate(st["py"]); np.random.set_state(st["np"]); torch.set_rng_state(st["torch"])
        g.set_state(st["loader_gen"])
        start_ep = st["epoch"] + 1
        best = st["best"]; counter = st["counter"]
        log(event="resume", from_epoch=start_ep)

    # microbatch OOM 预检(一个完整累积组)
    try:
        model.train()
        it = iter(train_loader)
        for _ in range(min(accum, len(train_loader))):
            xb, yb, _ = next(it)
            loss = criterion(model(xb.float().to(dev), None, None, None), yb.to(dev))
            (loss / (micro_bs * accum * 9)).backward()
        opt.zero_grad()
        log(event="microbatch_precheck", ok=True, micro_bs=micro_bs)
    except torch.cuda.OutOfMemoryError:
        opt.zero_grad()
        log(event="microbatch_precheck", ok=False, micro_bs=micro_bs)
        return {"job_id": job_id, "status": "failed", "stop_reason": "OOM at precheck"}

    last_flush = time.time()
    # QRS 检测内容缓存: 相同标准化信号(逐 epoch 不变)的 xqrs/峰检测结果缓存,
    # 输出与逐次检测逐位一致(检测器确定性), 仅消除重复检测耗时(identity 记录)
    import tools.thesis_inc.ref_tokenizer as _self
    tok = model.tokenizer
    if hasattr(tok, "qrs_detection"):
        orig = tok.qrs_detection

        def cached_qrs(ecg_signal, _orig=orig):
            key = np.ascontiguousarray(ecg_signal[0]).tobytes()
            if key not in _self.QRS_CACHE:
                _self.QRS_CACHE[key] = _orig(ecg_signal)
            return _self.QRS_CACHE[key]
        tok.qrs_detection = cached_qrs
    if hasattr(tok, "detect_rpeaks"):
        orig2 = tok.detect_rpeaks

        def cached_rp(signal, _orig=orig2):
            key = np.ascontiguousarray(signal).tobytes()
            if key not in _self.QRS_CACHE:
                _self.QRS_CACHE[key] = _orig(signal)
            return _self.QRS_CACHE[key]
        tok.detect_rpeaks = cached_rp
    try:
        for ep in range(start_ep, 300):
            if C.now_bj() >= stop_deadline:
                stop_reason = f"deadline_partial_at_ep{ep}"
                torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                            "epoch": ep, "best": best, "counter": counter,
                            "py": random.getstate(), "np": np.random.get_state(),
                            "torch": torch.get_rng_state(), "loader_gen": g.get_state()},
                           state_path)
                log(event="deadline_stop", epoch=ep)
                break
            model.train()
            # 按 accum 微批分组: 组内先收集, 以实际元素总数为分母 backward(协议要求)
            micro = [b for b in train_loader]
            groups = [micro[i:i + accum] for i in range(0, len(micro), accum)]
            train_loss_sum, train_el = 0.0, 0
            for grp in groups:
                grp_loss, grp_el = 0.0, 0
                for xb, yb, _ in grp:
                    xb, yb = xb.float().to(dev), yb.to(dev)
                    out = model(xb, None, None, None)
                    grp_loss = grp_loss + criterion(out, yb)
                    grp_el += out.numel()
                (grp_loss / grp_el).backward()
                train_loss_sum += float(grp_loss)
                train_el += grp_el
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
                opt.step()
                opt.zero_grad()
            model.eval()
            with torch.no_grad():
                probs = []
                for s in range(0, len(Xva), 256):
                    lg = model(Xva[s:s + 256].float().to(dev), None, None, None)
                    probs.append(torch.sigmoid(lg).cpu().numpy())
            probs = np.concatenate(probs)
            auc = macro_auc(yva.numpy(), probs)
            if auc > best["auc"]:
                best = {"epoch": ep + 1, "auc": auc,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "probs": probs}
                counter = 0
            else:
                counter += 1
            log(event="epoch_end", epoch=ep + 1, val_macro_auc=round(auc, 5),
                best_ep=best["epoch"], counter=counter,
                train_loss=round(train_loss_sum / max(train_el, 1), 6))
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "epoch": ep, "best": best, "counter": counter,
                        "py": random.getstate(), "np": np.random.get_state(),
                        "torch": torch.get_rng_state(), "loader_gen": g.get_state()},
                       state_path)
            if counter >= 15:
                stop_reason = "early_stop_patience15"
                break
            now = time.time()
            if now - last_flush > 900:
                C.add_resource("ref_gpu", now - last_flush, job_id)
                last_flush = now
    finally:
        C.add_resource("ref_gpu", time.time() - last_flush, job_id)
    wall = time.time() - t0
    if best["state"] is None:
        return {"job_id": job_id, "status": "failed", "stop_reason": "no_epoch_completed"}
    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        probs = []
        for s in range(0, len(Xva), 256):
            probs.append(torch.sigmoid(model(Xva[s:s + 256].float().to(dev), None, None, None)).cpu().numpy())
    probs = np.concatenate(probs)
    metrics = C.multilabel_metrics(yva.numpy(), probs)
    class_order = json.loads((DATA / "caches_manifest.json").read_text(encoding="utf-8"))["class_order_mlb"]
    np.savez_compressed(job_dir / "val_predictions.npz", record_ids=rid_va,
                        y_true=yva.numpy().astype(np.int8), y_prob=probs.astype(np.float32),
                        class_order=np.array(class_order))
    zj = np.load(job_dir / "val_predictions.npz", allow_pickle=False)
    from sklearn.metrics import roc_auc_score
    recompute_auc = float(roc_auc_score(zj["y_true"].astype(np.float32),
                                        zj["y_prob"].astype(np.float32), average="macro"))
    assert abs(recompute_auc - best["auc"]) <= 1e-8, (recompute_auc, best["auc"])
    status = "complete"
    if stop_reason.startswith("deadline_partial"):
        status = "partial"
    result = {"job_id": job_id, "stage": "REF", "kind": "reference_supervised",
              "tokenizer": tokenizer, "status": status, "stop_reason": stop_reason,
              "val_macro_auroc": best["auc"], "macro_auprc": metrics["macro_auprc"],
              "macro_auroc": metrics["macro_auroc"], "best_epoch": best["epoch"],
              "train_wall_s": round(wall, 1), "seed": 2025,
              "effective_batch": 256, "micro": f"{micro_bs}x{accum}",
              "class_order_mlb": class_order, "git_head": C.git_head(),
              "identity": "resource-adapted subset verification; recorded deviations see decisions/events",
              "recompute_abs_diff": abs(recompute_auc - best["auc"])}
    (job_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    if status == "complete":
        (job_dir / "DONE.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    C.count_job_start(job_id, "REF")
    C.append_csv(C.RESULTS / "results.csv", {
        "job_id": job_id, "stage": "REF", "kind": "reference_supervised",
        "recipe": f"Tokenizer_{tokenizer}", "seed": 2025, "pt_epochs": "",
        "status": status, "stop_reason": stop_reason, "wall_s": result["train_wall_s"],
        "macro_auprc": result["macro_auprc"], "macro_auroc": result["macro_auroc"],
        "encoder_sha256": ""})
    log_f.close()
    print(f"[{job_id}] auroc={best['auc']:.4f} best_ep={best['epoch']} stop={stop_reason}")
    return result


def run_all(stop_deadline):
    build_metadata()
    build_caches()
    # microbatch 预检三行统一 32x8; OOM 则统一 16x16(冻结后不再改)
    for micro_bs, accum in ((32, 8), (16, 16)):
        try:
            r = train(TOKENIZERS[0], micro_bs, accum, stop_deadline)
            if r.get("stop_reason") == "OOM at precheck":
                C.log_decision({"gate": "R-TK_microbatch", "choice": f"{micro_bs}x{accum}",
                                "note": "首行预检OOM, 三行统一降档"})
                continue
            C.log_decision({"gate": "R-TK_microbatch", "choice": f"{micro_bs}x{accum}",
                            "note": "预检通过, 三行冻结此档"})
            for tkz in TOKENIZERS:
                train(tkz, micro_bs, accum, stop_deadline)
            return
        except torch.cuda.OutOfMemoryError:
            continue
    C.log_event("R-TK: 32x8 与 16x16 两档预检均 OOM, 支线 partial")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    a = ap.parse_args()
    if a.cmd == "build":
        build_metadata()
        build_caches()
    else:
        run_all(C.deadline("gpu_stop_and_save_by"))
