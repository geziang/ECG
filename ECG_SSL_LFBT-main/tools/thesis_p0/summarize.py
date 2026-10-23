"""thesis-p0-20261023 summarize: 重建 results.csv(固定 schema), 三张分阶段 summary,
工程选线门计算, 预算对账, resume_manifest。唯一事实源 = jobs/*/DONE.json。"""
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.thesis_p0 import common as C

FIELDS = ["job_id", "stage", "model", "mode", "pretrain_seed", "training_seed",
          "status", "macro_auprc", "macro_auroc", "val_bce", "best_epoch",
          "stop_reason", "feat_wall_s", "train_wall_s", "gpu_wall_s", "vram_peak_mib",
          "macro_auprc_single", "macro_auprc_multi", "macro_auprc_beat_valid",
          "trainable_params", "code_sha", "stop_note"]


def collect():
    rows = []
    for done in sorted(C.JOBS.glob("*/DONE.json")):
        r = json.loads(done.read_text(encoding="utf-8"))
        rows.append(r)
    return rows


def write_results(rows):
    with open(C.RESULTS / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def summary_local_lp(rows):
    lp = [r for r in rows if r["stage"] == "A2_local_lp"]
    out = C.RESULTS / "summary_local_lp.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "seed0_auprc", "seed2_auprc", "seed4_auprc",
                    "mean_auprc", "mean_auroc", "mean_val_bce"])
        for m in ("M0", "M4", "M6"):
            rs = {r["pretrain_seed"]: r for r in lp if r["model"] == m}
            if len(rs) < 3:
                continue
            auprcs = [rs[s]["macro_auprc"] for s in (0, 2, 4)]
            w.writerow([m] + [f"{a:.4f}" for a in auprcs] +
                       [f"{sum(auprcs)/3:.4f}",
                        f"{sum(rs[s]['macro_auroc'] for s in (0,2,4))/3:.4f}",
                        f"{sum(rs[s]['val_bce'] for s in (0,2,4))/3:.4f}"])
    return out


def summary_representation(rows, gate_out):
    rep = [r for r in rows if r["stage"] == "A3_representation"]
    out = C.RESULTS / "summary_representation.csv"
    modes = ["D0_matched_fixed", "D1_mean_template", "D2_median_template", "D3_beat_set"]
    means = {}
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "seed0", "seed2", "seed4", "mean_auprc", "mean_auroc",
                    "n_beat_valid(val)"])
        for m in modes:
            rs = {r["training_seed"]: r for r in rep if r["mode"] == m}
            if len(rs) < 3:
                continue
            a = [rs[s]["macro_auprc"] for s in (0, 2, 4)]
            means[m] = a
            w.writerow([m] + [f"{x:.4f}" for x in a] + [f"{sum(a)/3:.4f}",
                        f"{sum(rs[s]['macro_auroc'] for s in (0,2,4))/3:.4f}",
                        rs[0].get("n_beat_valid", "")])
    # 工程选线门: ref = D0/D1/D2 三种子均值最高; D3 配对增量
    if len(means) == 4:
        ref_mode = max(("D0_matched_fixed", "D1_mean_template", "D2_median_template"),
                       key=lambda m: sum(means[m]) / 3)
        ref = means[ref_mode]
        d3 = means["D3_beat_set"]
        deltas = [d - r for d, r in zip(d3, ref)]
        gate = C.PROTOCOL["engineering_selection_gate"]
        g = {
            "reference_mode": ref_mode,
            "reference_mean": sum(ref) / 3, "d3_mean": sum(d3) / 3,
            "paired_deltas": deltas, "d3_minus_ref_mean": sum(deltas) / 3,
            "mean_min_required": gate["D3_minus_reference_mean_min"],
            "positive_seed_count": sum(1 for d in deltas if d > 0),
            "positive_seed_required": gate["positive_seed_count_required"],
            "per_class_gate": "见 FINAL_REPORT(逐类 AP 均值降幅)",
            "passed": (sum(deltas) / 3 >= gate["D3_minus_reference_mean_min"]
                       and sum(1 for d in deltas if d > 0) >= gate["positive_seed_count_required"]),
        }
        gate_out.update(g)
    return out


def summary_heartlang(rows):
    hl = [r for r in rows if r["stage"] == "A4_heartlang"]
    out = C.RESULTS / "summary_heartlang.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["job_id", "macro_auprc", "macro_auroc", "val_bce", "best_epoch",
                    "trainable_params", "stop_reason"])
        for r in hl:
            w.writerow([r["job_id"], f"{r['macro_auprc']:.4f}", f"{r['macro_auroc']:.4f}",
                        f"{r['val_bce']:.4f}", r["best_epoch"], r.get("trainable_params", ""),
                        r["stop_reason"]])
    return out


def budget():
    ru = C.RESULTS / "resource_usage.csv"
    gpu_s = cpu_s = 0.0
    if ru.exists():
        with open(ru, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("counts_as_gpu_process") == "1":
                    gpu_s += float(row["wall_s"])
                else:
                    cpu_s += float(row["wall_s"])
    b = C.PROTOCOL["budgets"]
    return {
        "gpu_process_hours_used": round(gpu_s / 3600, 2),
        "gpu_budget_hours": b["max_gpu_process_hours"],
        "cpu_hours_used": round(cpu_s / 3600, 2),
        "cpu_budget_hours": b["max_cpu_compute_hours"],
        "primary_training_jobs_budget": b["max_primary_training_jobs"],
    }


def main():
    rows = collect()
    write_results(rows)
    gate = {}
    summary_local_lp(rows)
    summary_representation(rows, gate)
    summary_heartlang(rows)
    bd = budget()
    resume = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "jobs_done": [r["job_id"] for r in rows],
        "n_done": len(rows), "gate": gate, "budget": bd,
        "remaining_matrix_jobs": 22 - len(rows),
    }
    (C.RESULTS / "resume_manifest.json").write_text(
        json.dumps(resume, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"jobs={len(rows)}/22  gpu={bd['gpu_process_hours_used']}h/{bd['gpu_budget_hours']}h  "
          f"cpu={bd['cpu_hours_used']}h/{bd['cpu_budget_hours']}h")
    if gate:
        print("gate:", json.dumps(gate, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
