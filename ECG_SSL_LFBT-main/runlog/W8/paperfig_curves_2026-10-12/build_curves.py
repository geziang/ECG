# -*- coding: utf-8 -*-
"""build_curves.py — 论文配图母版第6/7页曲线数据打包 (2026-10-12).

第6页(预训练损失曲线): 四分支逐 step JSON loss 行 → fig6_pretrain_curves.csv
  beat_rr   : runlog/W6/logs/dv_pt_seed0.log + W7/logs/dv_pt_seed{2,4}.log   (三种子全)
  noise_iso : runlog/W8/logs/noiseiso_pt_seed{0,2,4}.log                      (三种子全)
  c1_lfbt   : runlog/W1/pt_c1_nfh_b0_seed0.log  (同协议替身, 仅seed0, 见README脚注)
  c2_trc    : runlog/W8/logs/c2pt_fig_seed0.log  (2026-10-12 补录, 仅seed0, 见README)
第7页(下游验证损失曲线, C2→CPSC): 逐 epoch Val loss → fig7_downstream_curves.csv
  lp/ft10/lpft × seeds{0,2,4} = W5/W6/W7 各三份日志。
终点指标(非曲线): fig7_final_metrics.csv 从 W5/W6/W7 台账抽取。

用法: python build_curves.py   (在任意目录可跑, 路径按仓库根自动推导)
"""
import csv
import json
import re
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parents[2]  # ECG_SSL_LFBT-main
sys.path.insert(0, str(ROOT))
from utils.pathguard import open_out_file

FIG6_SOURCES = [
    # (branch, seed, relpath, note)
    ("beat_rr", 0, "runlog/W6/logs/dv_pt_seed0.log", "W6官方链"),
    ("beat_rr", 2, "runlog/W7/logs/dv_pt_seed2.log", "W7官方链"),
    ("beat_rr", 4, "runlog/W7/logs/dv_pt_seed4.log", "W7官方链"),
    ("noise_iso", 0, "runlog/W8/logs/noiseiso_pt_seed0.log", "W8 E6官方链"),
    ("noise_iso", 2, "runlog/W8/logs/noiseiso_pt_seed2.log", "W8 E6官方链"),
    ("noise_iso", 4, "runlog/W8/logs/noiseiso_pt_seed4.log", "W8 E6官方链"),
    ("c1_lfbt", 0, "runlog/W1/pt_c1_nfh_b0_seed0.log", "W1同协议替身非confirm重跑"),
    ("c2_trc", 0, "runlog/W8/logs/c2pt_fig_seed0.log", "2026-10-12补录非confirm原跑"),
]

FIG7_SOURCES = [
    ("lp", 0, "runlog/W5/logs/c2_cpsc_seed0.log"),
    ("lp", 2, "runlog/W5/logs/c2_cpsc_seed2.log"),
    ("lp", 4, "runlog/W5/logs/c2_cpsc_seed4.log"),
    ("ft10", 0, "runlog/W6/logs/c2_cpsc_ft10_seed0.log"),
    ("ft10", 2, "runlog/W6/logs/c2_cpsc_ft10_seed2.log"),
    ("ft10", 4, "runlog/W6/logs/c2_cpsc_ft10_seed4.log"),
    ("lpft", 0, "runlog/W7/logs/e4a_c2_cpsc_lpft_seed0.log"),
    ("lpft", 2, "runlog/W7/logs/e4a_c2_cpsc_lpft_seed2.log"),
    ("lpft", 4, "runlog/W7/logs/e4a_c2_cpsc_lpft_seed4.log"),
]

FINAL_LEDGERS = [
    ("lp", "runlog/W5/lp_results.csv", "lp", "cpsc", "c2"),
    ("ft10", "runlog/W6/ft10grid_results_L1.csv", "ft10", "cpsc", "c2"),
    ("lpft", "runlog/W7/e4a_lpft.csv", "lpft10", "cpsc", "c2"),
]

PT_FIELDS = ["loss", "loss_r", "loss_t", "loss_dv", "loss_rr"]


def read_lines_mixed(path):
    """W1 日志整文件是带 BOM 的 UTF-16LE(PowerShell 生成), 其余日志为 python -u
    重定向的纯 ASCII: 带 BOM 按整文件 utf-16 解, 否则 utf-8(errors=replace)。"""
    raw = Path(path).read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") \
        else raw.decode("utf-8", errors="replace")
    for line in text.splitlines():
        s = line.strip("\x00\r\n ").strip()
        if s:
            yield s


def parse_pt_log(path):
    rows = []
    for s in read_lines_mixed(path):
        if not s.startswith('{"epoch"'):
            continue
        try:
            d = json.loads(s)
        except json.JSONDecodeError:
            continue
        if "loss" not in d:  # 跳过 ccm_stats 等非损失行
            continue
        rows.append(d)
    return rows


def build_fig6():
    out = PKG / "fig6_pretrain_curves.csv"
    summary = []
    with open_out_file(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["branch", "seed", "source", "note", "epoch", "step"] + PT_FIELDS)
        for branch, seed, rel, note in FIG6_SOURCES:
            p = ROOT / rel
            if not p.exists():
                summary.append(f"MISSING  {branch} s{seed} {rel}")
                continue
            rows = parse_pt_log(p)
            for d in rows:
                w.writerow([branch, seed, Path(rel).name, note, d.get("epoch"),
                            d.get("step")] + [d.get(k, "") for k in PT_FIELDS])
            if rows:
                eps = sorted({r["epoch"] for r in rows})
                summary.append(f"OK      {branch} s{seed}: {len(rows)} 行, epoch {eps[0]}-{eps[-1]}"
                               f", 最终 loss={rows[-1].get('loss'):.1f}")
            else:
                summary.append(f"EMPTY   {branch} s{seed} {rel}")
    return summary


VAL_RE = re.compile(r"Val_?\s?[ _]loss:\s*([\d.]+)", re.I)
EPOCH_RE = re.compile(r"^Epoch\s+(\d+)\s+-")


def build_fig7():
    out = PKG / "fig7_downstream_curves.csv"
    summary = []
    with open_out_file(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phase", "seed", "source", "epoch", "val_loss"])
        for phase, seed, rel in FIG7_SOURCES:
            p = ROOT / rel
            if not p.exists():
                summary.append(f"MISSING  {phase} s{seed} {rel}")
                continue
            n, cur, best, last_ep = 0, None, None, None
            for s in read_lines_mixed(p):
                m = EPOCH_RE.match(s)
                if m:
                    cur = int(m.group(1))
                    continue
                m = VAL_RE.search(s)
                if m and cur is not None:
                    w.writerow([phase, seed, Path(rel).name, cur, m.group(1)])
                    best = m.group(1) if best is None else min(best, m.group(1))
                    last_ep = cur
                    n += 1
            summary.append(f"OK      {phase} s{seed}: {n} 点, epoch 1-{last_ep}, min val_loss={best}")
    return summary


def build_final_metrics():
    out = PKG / "fig7_final_metrics.csv"
    n = 0
    with open_out_file(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phase", "seed", "auroc", "auprc", "ledger"])
        for phase, ledger, eval_name, ds, ckpt in FINAL_LEDGERS:
            p = ROOT / ledger
            if not p.exists():
                continue
            for r in csv.DictReader(open(p, encoding="utf-8")):
                if (r.get("ckpt") == ckpt and r.get("eval") == eval_name
                        and r.get("downstream") == ds):
                    w.writerow([phase, r["seed"], r["auroc"], r["auprc"], ledger])
                    n += 1
    return [f"final_metrics: {n} 行"]


if __name__ == "__main__":
    print("== fig6 预训练曲线 ==")
    for s in build_fig6():
        print(" ", s)
    print("== fig7 下游曲线 ==")
    for s in build_fig7():
        print(" ", s)
    print("== 终点指标 ==")
    for s in build_final_metrics():
        print(" ", s)
    print("DONE ->", PKG)
