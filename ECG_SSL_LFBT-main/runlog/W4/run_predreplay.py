# -*- coding: utf-8 -*-
"""W4 A-8b: LP 预测重放 = 冻结 {b0,c1,c2} × {ptbxl,cpsc,chapman} × seeds, 复现门自动比对。

复现门(任务书 A-8): AUROC 四位一致且 AUPRC |Δ|<=0.0001, 超差即停并写失败记录。
账本来源: W2 confirm_results.csv(b0 seed2/4@ptbxl, c1/c2 全网格) + W1 c3_evals.csv(b0_anchor@3库)。
A-7 今日产物(b0 seed2/4 @ cpsc/chapman)已有预测, 自动跳过。
GPU 时机: 在 A-6 FT 与 A-4 PT 双车道之外的空位运行; 幂等可续。
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
PRED = ROOT / "runlog/W4/predictions"
FEAT = ROOT / "runlog/W4/feat"
GATE_LOG = ROOT / "runlog/W4/predreplay_gate.csv"

CK_OF = {("b0", 0): "checkpoint/ptxl_gamma08"}
for s in (2, 4):
    CK_OF[("b0", s)] = f"checkpoint/confirm/b0_seed{s}"
for k in ("c1", "c2"):
    for s in (0, 2, 4):
        CK_OF[(k, s)] = f"checkpoint/confirm/{k}_seed{s}"
TRC = {"c2": 1, "c1": 0, "b0": 0}
NC = {"ptbxl": 5, "cpsc": 9, "chapman": 4}


def ledger(kind, seed, ds):
    """返回 (auroc, auprc, 来源) 账本对照值; 无账本返回 None。"""
    for path, kcol in ((ROOT / "runlog/W2/confirm_results.csv", None),
                       (ROOT / "runlog/W1/c3_evals.csv", None)):
        if not path.exists():
            continue
        for r in csv.DictReader(open(path, encoding="utf-8")):
            m_kind = {"b0_anchor": "b0", "c1_nfh_b0": "c1w1", "c2_nfh_trc": "c2w1"}.get(r["ckpt"], r["ckpt"])
            if (m_kind, r["seed"], r["eval"], r["downstream"]) == (kind, str(seed), "lp", ds):
                if path.name.startswith("W1") and kind in ("c1", "c2"):
                    continue  # W1 的 c1/c2 与 W2 confirm 不同 checkpoint, 只对照 W2
                return float(r["auroc"]), float(r["auprc"]), path.parent.name
    return None


def run_one(kind, seed, ds):
    tag = f"{kind}_{ds}_seed{seed}"
    if (PRED / tag / "y_prob.npy").exists():
        return ("SKIP", tag, "已有预测产物")
    ref = ledger(kind, seed, ds)
    ck = CK_OF[(kind, seed)]
    feat = FEAT / f"replay_{tag}"
    cmd = [PY, "-u", "run_lp.py", "--data-dir", f"data/{ds}", "--num-classes", str(NC[ds]),
           "--checkpoint", f"{ck}/encoder_group.pth", "--feat-dir", str(feat),
           "--seed", str(seed), "--workers", "6", "--trc", str(TRC[kind]),
           "--extended-metrics", "1", "--save-predictions", f"runlog/W4/predictions/{tag}",
           "--protocol-id", "w4-predreplay"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0:
        return ("FAIL", tag, (r.stdout or "")[-200:])
    m = json.loads((feat / "metrics.json").read_text(encoding="utf-8"))
    if ref is None:
        return ("NOREF", tag, f"auroc={m['auroc']:.4f} auprc={m['auprc']:.4f} (无账本行)")
    au_ref, ap_ref, src = ref
    d_au = round(m["auroc"], 4) - round(au_ref, 4)
    d_ap = m["auprc"] - ap_ref
    ok = abs(d_au) < 5e-5 and abs(d_ap) <= 1e-4
    return ("PASS" if ok else "GATE_FAIL", tag,
            f"auroc {m['auroc']:.4f} vs {au_ref:.4f} (d={d_au:+.4f}); "
            f"auprc {m['auprc']:.4f} vs {ap_ref:.4f} (d={d_ap:+.4f}); ref={src}")


def main():
    jobs = []
    for kind in ("b0", "c1", "c2"):
        for ds in ("ptbxl", "cpsc", "chapman"):
            for s in (0, 2, 4):
                jobs.append((kind, s, ds))
    new = not GATE_LOG.exists()
    with open(GATE_LOG, "a", newline="", encoding="utf-8") as gf:
        w = csv.writer(gf)
        if new:
            w.writerow(["ts", "status", "tag", "detail"])
        n_fail = 0
        for kind, s, ds in jobs:
            status, tag, detail = run_one(kind, s, ds)
            w.writerow([time.strftime("%m-%d %H:%M"), status, tag, detail])
            gf.flush()
            print(f"[{status}] {tag}: {detail}", flush=True)
            if status in ("FAIL", "GATE_FAIL"):
                n_fail += 1
                if status == "GATE_FAIL":
                    print("复现门超差, 按任务书停止重放并保留失败记录", flush=True)
                    sys.exit(1)
        print(f"REPLAY DONE: fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
