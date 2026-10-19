# -*- coding: utf-8 -*-
"""W9 finalize: 完整性检查(§七 1-6) + summary.csv + table4_complete.md + W9_最终报告.md 草案。

用法(conda DL, 仓库根 cwd):
    python runlog/W9/run_w9_finalize.py           # 全部
    python runlog/W9/run_w9_finalize.py --checks  # 只跑完整性检查
只读 W4/W6/W7/W8 旧账本(引用不修改); W9 新产物全部在 runlog/W9/ 下。
"""
import argparse
import csv
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUTD = ROOT / "runlog/W9"
PROTO = "w9-submission-ft10-20261019"
NC = {"ptbxl": 5, "cpsc": 9, "chapman": 4}
SEEDS = [0, 2, 4]
METHOD_LABEL = {
    "simclr": "S1 SimCLR", "simph": "SimCLR+NSTDB(真实噪声视图)",
    "clocs": "S2 CLOCS", "tsr": "TSR 预任务",
    "m2": "M2 LFBT+Beat-RR", "m5": "M5 LFBT+NSTDB(时间隔离版)",
}
# 既有单元格出处(引用, 只读): (method, dataset) -> [(csv, ckpt 过滤), ...]
EXISTING_SOURCES = {
    ("simclr", "ptbxl"): [("runlog/W4/baseline_results.csv", "simclr")],
    ("simclr", "cpsc"): [("runlog/W8/w8_simph_results.csv", "simclr-w4frozen")],
    ("simph", "ptbxl"): [("runlog/W8/w8_simph_results.csv", "simph")],
    ("simph", "cpsc"): [("runlog/W8/w8_simph_results.csv", "simph")],
    ("clocs", "ptbxl"): [("runlog/W4/baseline_results.csv", "clocs")],
    ("tsr", "ptbxl"): [("runlog/W8/w8_tsr_results.csv", "tsr")],
    ("tsr", "cpsc"): [("runlog/W8/w8_tsr_results.csv", "tsr")],
    ("m2", "ptbxl"): [("runlog/W6/dv_results.csv", "dv"),
                      ("runlog/W7/e1_dv_results.csv", "dv")],
}
ALL_CELLS = [(m, ds) for m in ("simclr", "simph", "clocs", "tsr", "m2", "m5")
             for ds in ("ptbxl", "cpsc", "chapman")]


def load_ledger_rows():
    rows = []
    for f in sorted((OUTD / "ledger").glob("*.json")):
        rows.append(json.loads(f.read_text(encoding="utf-8")))
    return rows


def load_existing():
    """旧账本引用读取 -> {(method,ds): {seed: (auroc, auprc, src)}}。"""
    out = {}
    for (m, ds), srcs in EXISTING_SOURCES.items():
        for rel, ck in srcs:
            p = ROOT / rel
            if not p.exists():
                continue
            for r in csv.DictReader(open(p, encoding="utf-8")):
                if (r.get("eval") or "") != "ft10" or (r.get("downstream") or "") != ds:
                    continue
                if (r.get("ckpt") or "").strip() != ck:
                    continue
                try:
                    seed = int(r["seed"])
                except (KeyError, ValueError):
                    continue
                out.setdefault((m, ds), {})[seed] = (
                    float(r["auroc"]), float(r["auprc"]), rel)
    return out


def expected_test_labels(ds):
    """从 sample_order 的 [test] 段推导期望标签序列(类目录序=类索引)。"""
    lines = (OUTD / "sample_order" / f"{ds}_sample_order.txt").read_text(
        encoding="utf-8").splitlines()
    classes, seq, cur = [], [], None
    in_test = False
    for l in lines:
        if l.startswith("["):
            cur = l.strip("[]")
            in_test = (cur == "test")
            continue
        if not in_test or not l:
            continue
        cls = l.split("/")[0]
        if cls not in classes:
            classes.append(cls)
        seq.append(classes.index(cls))
    # 类目录序须与全库排序一致(sample_order 由 sorted() 生成, 此处仅核对计数)
    return np.array(seq, dtype=int), classes


def run_checks():
    gate = {}

    # 门1: 恰好 30 行, 唯一键, P0=15/P1+P1x=15
    rows = load_ledger_rows()
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    keys = [(r["method"], r["seed"], r["eval"], r["dataset"]) for r in ok_rows]
    gate["1_rows"] = dict(n_ok=len(ok_rows), n_ledger=len(rows), want=30,
                          unique=len(set(keys)) == len(keys),
                          ok=(len(ok_rows) == 30 and len(set(keys)) == 30))
    # 门2: 每单元格恰 {0,2,4}
    cells = {}
    for r in ok_rows:
        cells.setdefault((r["method"], r["dataset"]), set()).add(r["seed"])
    bad = {k: sorted(v) for k, v in cells.items() if v != set(SEEDS)}
    gate["2_cells"] = dict(n_cells=len(cells), want=10, bad=bad,
                           ok=(len(cells) == 10 and not bad))

    # 门3+4: 27+3 套 metrics/预测完整; y_true 与样本清单逐位一致; 概率维度/行和
    pre = json.loads((OUTD / "preflight.json").read_text(encoding="utf-8"))
    probs = {}
    for r in ok_rows:
        tag = f"{r['method']}_{r['dataset']}_s{r['seed']}"
        pdir = OUTD / "predictions" / tag
        mjson = OUTD / "ft" / tag / "metrics.json"
        need = [mjson, pdir / "y_true.npy", pdir / "y_pred.npy",
                pdir / "y_prob.npy", pdir / "metrics_ext.json"]
        if not all(p.exists() for p in need):
            gate.setdefault("3_artifacts", {"missing": []}).setdefault(
                "missing", []).append(tag)
            continue
        m = json.loads(mjson.read_text(encoding="utf-8"))
        yt = np.load(pdir / "y_true.npy")
        yp = np.load(pdir / "y_pred.npy")
        pr = np.load(pdir / "y_prob.npy")
        exp, _ = expected_test_labels(r["dataset"])
        checks = dict(
            ytrue_match=bool(yt.shape == exp.shape and (yt == exp).all()),
            prob_dim=bool(pr.shape == (exp.shape[0], NC[r["dataset"]])),
            prob_finite=bool(np.isfinite(pr).all()),
            rowsum=bool(np.allclose(pr.sum(axis=1), 1.0, atol=1e-3)),
            pred_argmax=bool((yp == pr.argmax(axis=1)).all()),
        )
        # 门5: 独立复算与 metrics.json / metrics_ext.json 一致
        y1h = np.eye(pr.shape[1])[yt]
        auroc = roc_auc_score(y1h, pr, average="macro")
        auprc = average_precision_score(y1h, pr, average="macro")
        me = json.loads((pdir / "metrics_ext.json").read_text(encoding="utf-8"))
        checks["recompute_match"] = bool(
            abs(auroc - float(m["auroc"])) < 1e-6 and abs(auprc - float(m["auprc"])) < 1e-6
            and abs(float(me["macro_ap"]) - float(m["auprc"])) < 1e-6)
        # 门6: SHA/协议字段与 preflight 一致
        checks["meta_match"] = bool(
            r["protocol_id"] == PROTO
            and r["split_sha"] == pre[f"split_sha.{r['dataset']}"]["got"]
            and r["git_sha"] == pre["git_sha"]
            and len(r["checkpoint_sha256"]) == 64)
        probs[tag] = (m, checks)
    bad_art = [t for t, (m, c) in probs.items() if not all(c.values())]
    gate["3_artifacts"] = dict(
        checked=len(probs), missing=gate.get("3_artifacts", {}).get("missing", []),
        ok=len(probs) == 30)
    gate["4_predictions"] = dict(bad=sorted(bad_art), ok=not bad_art)
    gate["5_recompute_incl"] = "见 gate4.bad 明细"
    gate["6_meta"] = "并入 gate4.bad 明细"

    # 门7: 六方法×三域全格 3-seed
    existing = load_existing()
    grid = {}
    for (m, ds), seeds in existing.items():
        grid[(m, ds)] = sorted(seeds)
    for (m, ds), seeds in cells.items():
        grid.setdefault((m, ds), [])
        grid[(m, ds)] = sorted(set(grid[(m, ds)]) | seeds)
    gaps = [k for k in ALL_CELLS if grid.get(k) != SEEDS]
    gate["7_grid"] = dict(n_cells=len(grid), want=18, gaps=gaps,
                          ok=(len(grid) == 18 and not gaps))
    return gate, probs, cells


def fmt(v, nd=4):
    return f"{v:.{nd}f}"


def write_summary():
    rows = [r for r in load_ledger_rows() if r.get("status") == "ok"]
    out = ["method,dataset,metric,seed0,seed2,seed4,mean,sd,min,max,n,source"]
    by = {}
    for r in rows:
        by.setdefault((r["method"], r["dataset"]), {})[r["seed"]] = r
    existing = load_existing()
    for (m, ds), seeds in sorted(by.items()):
        for metric in ("auroc", "auprc"):
            vals = [float(seeds[s][metric]) for s in SEEDS]
            out.append(",".join(str(x) for x in [
                m, ds, metric, *(f"{v:.10f}" for v in vals),
                f"{np.mean(vals):.10f}", f"{np.std(vals, ddof=1):.10f}",
                f"{min(vals):.10f}", f"{max(vals):.10f}", 3, "W9"]))
    for (m, ds), seeds in sorted(existing.items()):
        for metric, idx in (("auroc", 0), ("auprc", 1)):
            vals = [seeds[s][idx] for s in SEEDS]
            out.append(",".join(str(x) for x in [
                m, ds, metric, *(f"{v:.10f}" for v in vals),
                f"{np.mean(vals):.10f}", f"{np.std(vals, ddof=1):.10f}",
                f"{min(vals):.10f}", f"{max(vals):.10f}", 3,
                seeds[0][2].split("/")[-1]]))
    (OUTD / "summary.csv").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"summary.csv: {len(out) - 1} rows")
    return by, existing


def write_table4(by, existing):
    lines = ["# W9 table4_complete 草表(AUROC / AUPRC mean±SD, 3 seeds {0,2,4})",
             "",
             "来源: W9=本批 30 次; 其余引用旧账本(见 source 列, summary.csv 有明细)。",
             "数值 4 位小数(论文口径); 原值 10 位在 summary.csv。", "",
             "| 方法 | PTB-XL AUROC | PTB-XL AUPRC | CPSC AUROC | CPSC AUPRC | Chapman AUROC | Chapman AUPRC | 来源 |",
             "|---|---|---|---|---|---|---|---|"]

    def cell(m, ds):
        if (m, ds) in by:
            seeds = by[(m, ds)]
            a = [float(seeds[s]["auroc"]) for s in SEEDS]
            p = [float(seeds[s]["auprc"]) for s in SEEDS]
            return (f"{np.mean(a):.4f}±{np.std(a, ddof=1):.4f}",
                    f"{np.mean(p):.4f}±{np.std(p, ddof=1):.4f}", "W9")
        if (m, ds) in existing:
            s = existing[(m, ds)]
            a = [s[x][0] for x in SEEDS]
            p = [s[x][1] for x in SEEDS]
            src = ";".join(sorted({s[x][2].split("/")[-1] for x in SEEDS}))
            return (f"{np.mean(a):.4f}±{np.std(a, ddof=1):.4f}",
                    f"{np.mean(p):.4f}±{np.std(p, ddof=1):.4f}", src)
        return ("—", "—", "MISSING")

    for m in ("simclr", "simph", "clocs", "tsr", "m2", "m5"):
        cells_ = [cell(m, ds) for ds in ("ptbxl", "cpsc", "chapman")]
        lines.append("| " + METHOD_LABEL[m] + " | " + " | ".join(
            c for pair in cells_ for c in pair[:2]) + " | "
            + ";".join(c[2] for c in cells_) + " |")
    (OUTD / "table4_complete.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("table4_complete.md written")


def write_report(gate):
    rows = [r for r in load_ledger_rows() if r.get("status") == "ok"]
    ev = (OUTD / "events.md").read_text(encoding="utf-8") if (OUTD / "events.md").exists() else ""
    lines = ["# W9 最终报告(草稿, 主机A)", "",
             f"- 生成: {time.strftime('%Y-%m-%d %H:%M')}  protocol_id={PROTO}",
             "- 矩阵: 任务书§三 27 次 + 用户 10-19 裁决扩充 m5-ptbxl×3 = 30 次",
             "  (M5 PTB-XL FT10 全库为零, §七-7 要求三域闭合; 同冻结协议, checkpoint SHA 均在§4.3 表)",
             "", "## 一、收官验收门(§七)", ""]
    names = {"1_rows": "门1 30 行唯一键", "2_cells": "门2 单元格恰{0,2,4}",
             "3_artifacts": "门3 30 套产物完整", "4_predictions": "门4/5/6 预测与复算一致",
             "7_grid": "门7 六方法×三域全格"}
    for k, n in names.items():
        if k in gate:
            lines.append(f"- {n}: {'✅' if gate[k]['ok'] else '❌'} `{json.dumps(gate[k], ensure_ascii=False)}`")
    lines += ["", "## 二、执行记录", "",
              f"- ok {sum(1 for r in rows if r['status'] == 'ok')}/30;"
              f" 总 wall {sum(float(r.get('wall_s', 0)) for r in rows) / 3600:.2f} h",
              "", "### events.md 全文(失败/重试/偏差如实)", "", "```", ev.strip(), "```", "",
              "## 三、论文回填提示(办公机§八)", "",
              "- 表 4 五空格+M2/M5 行 -> table4_complete.md; 5.3 删旧限制;",
              "- 摘要/结论/加粗值随新表复核。", ""]
    (OUTD / "W9_最终报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("W9_最终报告.md written (draft)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", action="store_true")
    a = ap.parse_args()
    gate, probs, cells = run_checks()
    print(json.dumps(gate, ensure_ascii=False, indent=1))
    all_ok = all(v.get("ok", True) for v in gate.values() if isinstance(v, dict))
    print("VERDICT:", "ALL GATES PASS" if all_ok else "GATES FAILED")
    if not a.checks:
        by, existing = write_summary()
        write_table4(by, existing)
        write_report(gate)
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
