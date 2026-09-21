"""c2_chain.py — W1 C2 = C1 + TRC/GRN1D 适配器跨域稳定性 (2026-09-21)。

链: NFH 预训练(100ep matched-updates, --trc 1) -> LP on PTB/CPSC/Chapman(均 --trc 1)
   -> 追加 runlog/W1/c3_evals.csv (幂等: 已有 c2 行则跳过记账)。
状态: runlog/W1/c2_chain_status.txt
"""
import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(r"F:\新实验\ECG_SSL_LFBT-main")
PY = r"C:\Users\admin\.conda\envs\DL\python.exe"
CKPT = ROOT / "checkpoint/M/c2_nfh_trc_seed0/encoder_group.pth"
EVALS = ROOT / "runlog/W1/c3_evals.csv"
STATUS = ROOT / "runlog/W1/c2_chain_status.txt"


def log(msg):
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    return r


def lp_eval(ds, num_classes, tag):
    feat = ROOT / f"feat/W1_c2trc_{tag}"
    r = run([PY, str(ROOT / "run_lp.py"), "--data-dir", str(ROOT / f"data/{ds}"),
             "--num-classes", str(num_classes), "--checkpoint", str(CKPT),
             "--feat-dir", str(feat), "--seed", "0", "--workers", "6", "--trc", "1"])
    m = json.load(open(feat / "metrics.json"))
    with open(EVALS, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["c2_nfh_trc", ds.replace("ptbxl", "ptbxl"),
                                f"{m['auroc']:.4f}", f"{m['auprc']:.4f}"])
    log(f"[c2] {tag}: auroc={m['auroc']:.4f} auprc={m['auprc']:.4f} rc={r.returncode}")
    return m


def main():
    if any(r["ckpt"] == "c2_nfh_trc" for r in csv.DictReader(open(EVALS, encoding="utf-8"))):
        log("[c2] csv 已有 c2 行, 幂等退出")
        return
    log(f"[c2] PT start {__import__('time').strftime('%m-%d %H:%M')}")
    r = run([PY, str(ROOT / "run_pt.py"), "--data-dir", str(ROOT / "data/pt_pretrain_nfh"),
             "--epochs", "100", "--batch-size", "128", "--workers", "4", "--seed", "0",
             "--trc", "1", "--checkpoint-dir", str(CKPT.parent)])
    log(f"[c2] PT_EXIT={r.returncode}")
    if r.returncode != 0:
        log("[c2] PT 失败: " + (r.stdout or r.stderr or "")[-400:])
        return
    for ds, nc, tag in (("ptbxl", 5, "ptbxl"), ("cpsc", 9, "cpsc"), ("chapman", 4, "chapman")):
        lp_eval(ds, nc, tag)
    log(f"[c2] DONE {__import__('time').strftime('%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
