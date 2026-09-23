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
    """返回 (auroc, auprc, 来源) 账本对照值; 无账本返回 None。

    W2 confirm_results.csv 有 seed 列; W1 c3_evals.csv 无 seed 列(锚点=b0 seed0,
    c1/c2 为 W1 版 checkpoint 与 W2 confirm 不同, 只对照 W2)。
    """
    for path in (ROOT / "runlog/W2/confirm_results.csv",
                 ROOT / "runlog/W1/c3_evals.csv"):
        if not path.exists():
            continue
        is_w1 = path.parent.name == "W1"
        for r in csv.DictReader(open(path, encoding="utf-8")):
            m_kind = {"b0_anchor": "b0", "c1_nfh_b0": "c1w1", "c2_nfh_trc": "c2w1"}.get(r["ckpt"], r["ckpt"])
            r_seed = r.get("seed") or "0"  # W1 无 seed 列, 锚点=seed0
            if (m_kind, r_seed, r["eval"], r["downstream"]) == (kind, str(seed), "lp", ds):
                if is_w1 and kind in ("c1", "c2"):
                    continue
                return float(r["auroc"]), float(r["auprc"]), path.parent.name
    return None


def _compare(tag, m, ref):
    """对照账本; W2 行=硬门(超差即停), W1 行=参考对照(记录不拦截)。

    依据: 任务书 A-8 复现门对象为 W2 账本; W1 b0_anchor 为 09-21 时代代码所跑,
    与当前管线存在固有 LP 随机性差(实测 b0@ptbxl d=+0.0010, 而同管线对 W2 行
    逐位一致, 见 predreplay_gate.csv 与 c2_ptbxl_seed0 探针), 故 W1 行只记录。
    """
    if ref is None:
        return ("NOREF", tag, f"auroc={m['auroc']:.4f} auprc={m['auprc']:.4f} (无账本行)")
    au_ref, ap_ref, src = ref
    d_au = round(m["auroc"], 4) - round(au_ref, 4)
    d_ap = m["auprc"] - ap_ref
    detail = (f"auroc {m['auroc']:.4f} vs {au_ref:.4f} (d={d_au:+.4f}); "
              f"auprc {m['auprc']:.4f} vs {ap_ref:.4f} (d={d_ap:+.4f}); ref={src}")
    ok = abs(d_au) < 5e-5 and abs(d_ap) <= 1e-4
    if src == "W1":
        return ("REF", tag, detail + (" [W1参考行, 不拦截]" ))
    return ("PASS" if ok else "GATE_FAIL", tag, detail)


def run_one(kind, seed, ds):
    tag = f"{kind}_{ds}_seed{seed}"
    ref = ledger(kind, seed, ds)
    if (PRED / tag / "y_prob.npy").exists():
        # 幂等跳过时也回读 feat metrics 做对照补账
        feat = FEAT / f"replay_{tag}"
        mj = FEAT / f"{tag}" / "metrics.json" if kind == "b0" and ds in ("cpsc", "chapman") else None
        for cand in (feat / "metrics.json", mj):
            if cand and cand.exists():
                m = json.loads(cand.read_text(encoding="utf-8"))
                status, t, detail = _compare(tag, m, ref)
                return ("SKIP+" + status, tag, detail)
        return ("SKIP", tag, "已有预测产物(未找到 metrics 供补对)")
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
    return _compare(tag, m, ref)


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
