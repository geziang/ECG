# -*- coding: utf-8 -*-
"""W7 E4c: PTB-XL 患者级重分层敏感性——K=3 次患者级分层 70/10/20 重划 × {b0,c1,c2} × seed0 LP。

阶段 1: 读 data/manifest.json (含 ecg_id/patient_id/split/label), 患者为单位、患者多数标签分层,
        seeds {101,102,103} 重划 train/val/test(患者不跨 split, 硬链接建 data/w7_e4c/ptbxl_k{i}/)。
阶段 2: 每个新 split 跑冻结 checkpoint LP (b0=ptxl_gamma08, c1=confirm/c1_seed0, c2=confirm/c2_seed0,
        c2 需 --trc 1), protocol_id=w7-e4c-k{i}。GPU 轻负载, 可与 E1 PT 并行。
阶段 3: 账本 e4c_patient_lp.csv + e4c_summary.md (与官方 fold split 的 W2 seed0 参照对比;
        官方 strat_fold 本身患者不重叠——本实验检验的是"结论对重划的稳健性", 措辞用敏感性分析)。
性质: 新 split=新 test 格, 每格只评一次; 不改任何协议超参。
用法: python runlog/W7/run_e4c_patient_lp.py [--phase splits|lp|all]
"""
import argparse
import csv
import io
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PY = "C:/Users/admin/.conda/envs/DL/python.exe"
OUTD = ROOT / "runlog/W7"
LOGD = OUTD / "logs"
K_SEEDS = {1: 101, 2: 102, 3: 103}
CKPTS = {
    "b0": (ROOT / "checkpoint/ptxl_gamma08/encoder_group.pth", "0"),
    "c1": (ROOT / "checkpoint/confirm/c1_seed0/encoder_group.pth", "0"),
    "c2": (ROOT / "checkpoint/confirm/c2_seed0/encoder_group.pth", "1"),
}
W2_REF_AUPRC = {"b0": 0.7177, "c1": 0.6574, "c2": 0.6564}  # W2 main_table per_seed 首位(seed0), ptbxl lp


def log(msg):
    print(f"[e4c {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def phase_splits():
    mani = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    recs = mani["records"] if "records" in mani else mani
    rows = [r for r in recs if r.get("split") in ("train", "val", "test")]
    log(f"downstream records: {len(rows)}")
    by_pat = {}
    for r in rows:
        by_pat.setdefault(int(r["patient_id"]), []).append(r)
    pats = sorted(by_pat)
    from collections import Counter
    pat_label = {}
    for p, rs in by_pat.items():
        pat_label[p] = Counter(r["label"] for r in rs).most_common(1)[0][0]
    labels = sorted({r["label"] for r in rows})
    log(f"patients: {len(pats)}, labels: {labels}")

    from sklearn.model_selection import train_test_split
    for k, seed in K_SEEDS.items():
        dest = ROOT / f"data/w7_e4c/ptbxl_k{k}"
        if (dest / "_done.txt").exists():
            log(f"k{k}: already built")
            continue
        p_tr, p_tmp = train_test_split(pats, test_size=0.30, random_state=seed,
                                       stratify=[pat_label[p] for p in pats])
        p_va, p_te = train_test_split(p_tmp, test_size=2 / 3, random_state=seed,
                                      stratify=[pat_label[p] for p in p_tmp])
        assign = {}
        for p in p_tr:
            assign[p] = "train"
        for p in p_va:
            assign[p] = "val"
        for p in p_te:
            assign[p] = "test"
        # 硬链接建目录
        import os
        n_by = {}
        for r in rows:
            split = assign[int(r["patient_id"])]
            cls_dir = dest / split / r["label"]
            cls_dir.mkdir(parents=True, exist_ok=True)
            src = ROOT / "data/ptbxl" / r["split"] / r["label"] / f"sample_{int(r['ecg_id']):05d}.npy"
            dst = cls_dir / src.name
            if not dst.exists():
                try:
                    os.link(src, dst)
                except OSError:
                    import shutil
                    shutil.copyfile(src, dst)
            n_by[(split, r["label"])] = n_by.get((split, r["label"]), 0) + 1
        # 校验: 患者不跨 split(构造保证); 各 split 比例
        tot = sum(n_by.values())
        props = {s: round(sum(v for (ss, _), v in n_by.items() if ss == s) / tot, 3) for s in ("train", "val", "test")}
        (dest / "_done.txt").write_text(f"seed={seed} patients={len(pats)} props={props}\n", encoding="utf-8")
        log(f"k{k} (seed {seed}): props={props}, test_n={sum(v for (ss, _), v in n_by.items() if ss == 'test')}")


def phase_lp():
    for k in K_SEEDS:
        for model, (ck, trc) in CKPTS.items():
            tag = f"e4c_{model}_k{k}"
            feat = OUTD / "feat" / tag
            metrics = feat / "metrics.json"
            if metrics.exists():
                log(f"{tag}: 已有 metrics, 跳过")
                continue
            feat.mkdir(parents=True, exist_ok=True)
            from utils.pathguard import open_out
            with open_out(LOGD, f"{tag}.log", encoding="utf-8") as f:
                rc = subprocess.run(
                    [PY, "-u", "run_lp.py", "--data-dir", f"data/w7_e4c/ptbxl_k{k}",
                     "--num-classes", "5", "--checkpoint", str(ck), "--feat-dir", str(feat),
                     "--seed", "0", "--workers", "6", "--trc", trc,
                     "--extended-metrics", "1",
                     "--save-predictions", str(OUTD / "predictions" / tag),
                     "--protocol-id", f"w7-e4c-k{k}"],
                    stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
            log(f"{tag}: rc={rc}")


def phase_summary():
    import hashlib

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fp:
            for c in iter(lambda: fp.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()[:12]

    out = OUTD / "e4c_patient_lp.csv"
    rows = []
    for k in K_SEEDS:
        for model, (ck, trc) in CKPTS.items():
            mfile = OUTD / "feat" / f"e4c_{model}_k{k}" / "metrics.json"
            if not mfile.exists():
                continue
            m = json.loads(mfile.read_text(encoding="utf-8"))
            rows.append(dict(k=k, model=model, eval="lp_patient_k", auroc=round(m["auroc"], 4),
                             auprc=round(m["auprc"], 4), w2_seed0_auprc_ref=W2_REF_AUPRC[model],
                             auprc_delta_vs_official=round(m["auprc"] - W2_REF_AUPRC[model], 4),
                             ckpt_sha=sha(ck), protocol_id=f"w7-e4c-k{k}"))
    if not rows:
        log("no metrics yet, skip summary")
        return
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    out.write_text(buf.getvalue(), encoding="utf-8", newline="")
    import numpy as np
    lines = ["# E4c 患者级重分层敏感性摘要", "",
             "- 官方 strat_fold 本身患者不重叠; 本实验=K=3 次患者级分层重划(70/10/20, seeds 101/102/103)后冻结 ckpt LP(seed 0),",
             "- 检验结论对 split 重划的稳健性(敏感性分析, 非主表); 参照=W2 官方 split seed0 AUPRC。", "",
             "| model | k1 auprc | k2 auprc | k3 auprc | mean±sd | 官方split参照 | Δrange |", "|---|---|---|---|---|---|---|"]
    for model in CKPTS:
        vs = [r["auprc"] for r in rows if r["model"] == model]
        if len(vs) < 3:
            lines.append(f"| {model} | (待补) | | | | {W2_REF_AUPRC[model]:.4f} | |")
            continue
        ds = [r["auprc_delta_vs_official"] for r in rows if r["model"] == model]
        lines.append(f"| {model} | {vs[0]:.4f} | {vs[1]:.4f} | {vs[2]:.4f} | "
                     f"{np.mean(vs):.4f}±{np.std(vs, ddof=1):.4f} | {W2_REF_AUPRC[model]:.4f} | "
                     f"[{min(ds):+.4f},{max(ds):+.4f}] |")
    (OUTD / "e4c_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"summary written, rows={len(rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["splits", "lp", "all", "summary"])
    a = ap.parse_args()
    if a.phase in ("splits", "all"):
        phase_splits()
    if a.phase in ("lp", "all"):
        phase_lp()
    phase_summary()
