# -*- coding: utf-8 -*-
"""thesis-inc 主线门尾清理: 真实状态标注 + resume_manifest + 主线总结表。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.thesis_inc import common as C

GATE_REASON = {
    "R2": "not_entered_by_gate: R2 规则=I1/I2/I3 全非正时仅 combo_slot1(最佳形态项+RR)一次机会, slot2/3 不进",
    "R3": "not_entered_by_gate: R1+R2 无任何相对 I0 正向配方, 无权重档可扫",
    "R4": "not_entered_by_gate: 无正向配方, 种子确认阶段按规则结束",
    "R5": "not_entered_by_gate: 无三种子平均增量>0 配方(种子确认未进入), R5 不启动",
}


def main():
    res = C.RESULTS
    m = json.loads((res / "job_manifest.json").read_text(encoding="utf-8"))
    for j in m["jobs"]:
        if j["status"] == "planned" and j["stage"] in GATE_REASON:
            j["status"] = "not_entered_by_gate"
            j["note"] = GATE_REASON[j["stage"]]
    (res / "job_manifest.json").write_text(
        json.dumps({"protocol_id": C.PROTOCOL["protocol_id"], "ts": C.bj_str(),
                    "jobs": m["jobs"]}, ensure_ascii=False, indent=1), encoding="utf-8")

    # 主线结果汇总
    jobs = res / "jobs"
    def lp(job):
        return json.loads((jobs / job / "DONE.json").read_text(encoding="utf-8"))
    i0 = lp("r1_i0_s0_e20_lp")
    rows = []
    for r in ("I0", "I1", "I2", "I3", "I4", "I5"):
        d = lp(f"r1_{r.lower()}_s0_e20_lp")
        pt = json.loads((jobs / f"r1_{r.lower()}_s0_e20_pt" / "result.json").read_text(encoding="utf-8"))
        rows.append({"recipe": r, "macro_auprc": d["macro_auprc"],
                     "macro_auroc": d["macro_auroc"],
                     "delta_vs_I0": round(d["macro_auprc"] - i0["macro_auprc"], 6),
                     "val_bce": d["val_bce"], "best_epoch": d["best_epoch"],
                     "pt_wall_s": pt["wall_s"],
                     "encoder_sha256": pt.get("encoder_sha256", "")[:16],
                     "per_class_ap": d["per_class_ap"]})
    combo = lp("r2_combo_slot1_s0_e20_lp")
    i2 = lp("r1_i2_s0_e20_lp")
    summary = {
        "ts": C.bj_str(),
        "primary_metric": "val macro-AUPRC (CPSC train4809/val683, 冻结编码器 LP)",
        "r1_seed0_20ep": rows,
        "r2_single_combo": {
            "recipe": "I2+RR (median template 0.01 + RR 0.1)",
            "macro_auprc": combo["macro_auprc"],
            "delta_vs_I0": round(combo["macro_auprc"] - i0["macro_auprc"], 6),
            "delta_vs_parent_I2": round(combo["macro_auprc"] - i2["macro_auprc"], 6),
            "macro_auroc": combo["macro_auroc"], "best_epoch": combo["best_epoch"]},
        "conclusion": "R1 六配方全部非正(I2 最接近 -0.0003); R2 唯一允许的单次组合 I2+RR 亦非正"
                      "(-0.0007, 相对母配方 -0.0003 无协同); 按冻结门 R3/R4/R5 不进入。"
                      "结论=本批增量辅助目标在 20ep/seed0 口径下为有效阴性, 无需种子确认。",
    }
    (res / "summary_main_line.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                encoding="utf-8")

    # resume_manifest: 可恢复工件清单
    resumes = []
    for jd in sorted((res / "jobs").iterdir()):
        if not jd.is_dir():
            continue
        cfg = jd / "config.json"
        if not cfg.exists():
            continue
        ck = C.CKPT_ROOT / jd.name
        entry = {"job_id": jd.name, "config": str(cfg),
                 "result": str(jd / "result.json") if (jd / "result.json").exists() else None,
                 "done": (jd / "DONE.json").exists(),
                 "ckpt_dir": str(ck),
                 "backup_dir": str(C.BACKUP_ROOT / jd.name),
                 "backup_exists": (C.BACKUP_ROOT / jd.name / "encoder_group.pth").exists()
                 if "ref_" not in jd.name else None}
        if (ck / "encoder_group.pth").exists():
            entry["encoder_sha256"] = C.sha256_of(ck / "encoder_group.pth")
        resumes.append(entry)
    (res / "resume_manifest.json").write_text(
        json.dumps({"protocol_id": C.PROTOCOL["protocol_id"], "ts": C.bj_str(),
                    "state": "main_line_closed_negative; REF side-line running",
                    "budget": C.load_state(), "jobs": resumes},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("finalized: job_manifest annotated, summary_main_line.json, resume_manifest.json")


if __name__ == "__main__":
    main()
