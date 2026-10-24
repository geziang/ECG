# -*- coding: utf-8 -*-
"""thesis-inc runner: 阶段调度 / 门解析 / 预算与截止 / manifest / 回传。

用法:
  python -m tools.thesis_inc.runner --preflight      # A0 预检
  python -m tools.thesis_inc.runner --freeze         # 冻结 protocol_resolved
  python -m tools.thesis_inc.runner --stage R1       # 跑 R1(6 PT + 6 LP)
  python -m tools.thesis_inc.runner --stage R2|R3|R4|R5  (门解析后执行)
  python -m tools.thesis_inc.runner --auto           # 依门自动推进到预算/截止
状态机: planned/queued/running/trained/evaluated/complete/partial/failed。
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.thesis_inc import common as C
from tools.thesis_inc import pretrain as PT
from tools.thesis_inc import lp as LP

MATRIX = {}  # job_id -> row
for _row in (C.PKG / "task_matrix.csv").read_text(encoding="utf-8-sig").splitlines()[1:]:
    _f = _row.split(",")
    MATRIX[_f[0]] = {"job_id": _f[0], "stage": _f[1], "kind": _f[2],
                     "recipe_or_slot": _f[3], "seed": int(_f[4]),
                     "pt_epochs": int(_f[5]) if _f[5] else 0, "depends_on": _f[6],
                     "condition": _f[7], "priority": int(_f[8]), "status": _f[9]}


def done_of(job_id: str):
    d = C.JOBS / job_id / "DONE.json"
    return json.loads(d.read_text(encoding="utf-8")) if d.exists() else None


def lp_result_for(stage: str):
    """已完成 LP 结果: {(recipe_key, seed): result}。recipe_key 用 PT job 的 recipe。"""
    out = {}
    for job_id, row in MATRIX.items():
        if row["kind"] != "linear_probe" or row["stage"] != stage:
            continue
        r = done_of(job_id)
        if r:
            out[(r["recipe"], row["seed"])] = r
    return out


def pt_result_for(stage: str):
    out = {}
    for job_id, row in MATRIX.items():
        if row["kind"] != "pretrain" or row["stage"] != stage:
            continue
        d = C.JOBS / job_id / "result.json"
        if d.exists():
            r = json.loads(d.read_text(encoding="utf-8"))
            out[(r.get("recipe") or row["recipe_or_slot"], row["seed"])] = r
    return out


# ---------------- 门解析 ----------------
def _delta(lp_by, key, i0_key=("I0", 0)):
    a = lp_by.get(key)
    b = lp_by.get(i0_key)
    if a is None or b is None:
        return None
    return a["macro_auprc"] - b["macro_auprc"]


def resolve_r2(lp_by) -> list:
    """返回 combo 定义列表 [{slot, recipe_name, recipe, reason}]。"""
    deltas = {k: _delta(lp_by, (k, 0)) for k in ("I1", "I2", "I3")}
    known = {k: v for k, v in deltas.items() if v is not None}
    if not known:
        return []
    pos = sorted([k for k, v in known.items() if v > 0],
                 key=lambda k: (-known[k], k))
    combos = []
    if pos:
        for i, k in enumerate(pos[:2]):
            base = R1_RECIPES[k]
            combos.append({
                "slot": f"combo_slot{i+1}", "recipe_name": f"{k}+RR",
                "recipe": {**base, "rr_weight": 0.1},
                "reason": f"{k} 正向(delta={known[k]:+.4f}) 叠加 RR0.1"})
        rest = [k for k in ("I1", "I2", "I3") if k not in pos[:2]]
        nonpos = sorted([k for k in rest if known.get(k, 0) <= 0],
                        key=lambda k: (-(known[k] if known[k] is not None else -9), k))
        if nonpos:
            k = nonpos[0]
            combos.append({
                "slot": "combo_slot3", "recipe_name": f"{k}+RR",
                "recipe": {**R1_RECIPES[k], "rr_weight": 0.1},
                "reason": f"非正形态项 {k}(delta={known[k]:+.4f})+RR 互补机会(形态-节律互补理由)"})
    else:
        k = sorted(known, key=lambda k: (-known[k], k))[0]
        combos.append({
            "slot": "combo_slot1", "recipe_name": f"{k}+RR",
            "recipe": {**R1_RECIPES[k], "rr_weight": 0.1},
            "reason": f"I1/I2/I3 全非正: 最佳形态项 {k}(delta={known[k]:+.4f})+RR 单次组合机会"})
    return combos[:3]


def resolve_r3(lp_by, pt_by) -> list:
    """R3 候选池 = I1..I5 与 R2 组合中相对 I0 正向者; 选第一名做权重档。"""
    cands = []
    for k in ("I1", "I2", "I3", "I4", "I5"):
        d = _delta(lp_by, (k, 0))
        if d is not None:
            t = (pt_by.get((k, 0)) or {}).get("wall_s")
            cands.append({"recipe_name": k, "recipe": R1_RECIPES[k], "delta": d,
                          "train_s": t if t is not None else float("inf")})
    for (rn, seed), r in lp_by.items():
        if seed == 0 and rn not in ("I0", "I1", "I2", "I3", "I4", "I5"):
            d = _delta(lp_by, (rn, 0))
            if d is not None:
                rc = _recipe_from_configs(rn)
                if rc is not None:
                    cands.append({"recipe_name": rn, "recipe": rc, "delta": d,
                                  "train_s": float(r.get("pt_wall_s", float("inf")))})
    if not cands:
        return []
    best = sorted(cands, key=lambda c: (-c["delta"], c["train_s"], c["recipe_name"]))[0]
    if best["delta"] <= 0:
        return []
    r = best["recipe"]
    if r.get("rr_weight", 0) > 0 and not r.get("template_mode") and not r.get("local_mode"):
        slots = [("weight_slot1", "rr_weight", round(r["rr_weight"] * 0.3, 4)),
                 ("weight_slot2", "rr_weight", round(r["rr_weight"] * 3.0, 4))]
    else:
        slots = [("weight_slot1", "morph_weight", round(r["morph_weight"] * 0.3, 6)),
                 ("weight_slot2", "morph_weight", round(r["morph_weight"] * 3.0, 6))]
    out = []
    for slot, field, w in slots:
        rr = {**r, field: w}
        out.append({"slot": slot, "recipe_name": f"{best['recipe_name']}@{field}={w}",
                    "recipe": rr,
                    "reason": f"第一名 {best['recipe_name']}(delta={best['delta']:+.4f}) "
                              f"{field} 档 {w} (原 {r[field]})"})
    return out


def _pos_rank(lp_by):
    """R4: 全部有效短训结果(含组合/权重档)相对 I0 正向排名。"""
    i0 = lp_by.get(("I0", 0))
    if i0 is None:
        return []
    cands = {}
    for (rn, seed), r in lp_by.items():
        if seed != 0 or rn == "I0":
            continue
        d = r["macro_auprc"] - i0["macro_auprc"]
        if d > 0:
            cands[rn] = max(d, cands.get(rn, -9))
    return sorted(cands, key=lambda rn: (-cands[rn], rn))


def resolve_r4(lp_by) -> list:
    top = _pos_rank(lp_by)[:2]
    if not top:
        return []
    out = []
    for i, rn in enumerate(top):
        out.append({"slot": f"winner_slot{i+1}", "recipe_name": rn})
    return out


def resolve_r5(lp_by) -> dict | None:
    """三种子短训平均增量>0 的均值第一名; 返回 {winner_recipe_name, mean_delta}。"""
    agg = {}
    for (rn, seed), r in lp_by.items():
        if rn == "I0":
            continue
        if rn not in agg:
            agg[rn] = {}
        i0s = lp_by.get(("I0", seed))
        if i0s is None:
            continue
        agg[rn][seed] = r["macro_auprc"] - i0s["macro_auprc"]
    best = None
    for rn, d in agg.items():
        seeds_done = {s for s in (0, 2, 4) if s in d}
        if seeds_done == {0, 2, 4}:
            md = sum(d[s] for s in (0, 2, 4)) / 3
            if md > 0 and (best is None or md > best[1]):
                best = (rn, md)
    return {"recipe_name": best[0], "mean_delta": best[1]} if best else None


# ---------------- 执行 ----------------
R1_RECIPES = {
    "I0": {"template_mode": None, "local_mode": None, "morph_weight": 0.0, "rr_weight": 0.0},
    "I1": {"template_mode": "mean", "local_mode": None, "morph_weight": 0.01, "rr_weight": 0.0},
    "I2": {"template_mode": "median", "local_mode": None, "morph_weight": 0.01, "rr_weight": 0.0},
    "I3": {"template_mode": None, "local_mode": "peaks", "morph_weight": 0.01, "rr_weight": 0.0},
    "I4": {"template_mode": None, "local_mode": None, "morph_weight": 0.0, "rr_weight": 0.1},
    "I5": {"template_mode": None, "local_mode": "fixed", "morph_weight": 0.01, "rr_weight": 0.0},
}


def _pt_deadline_ok(kind: str) -> bool:
    now = C.now_bj()
    if kind == "pretrain" and now >= C.deadline("latest_start_pretraining"):
        return False
    return now < C.deadline("latest_start_gpu_job")


def _est_fit(est_s: float) -> bool:
    now = C.now_bj()
    stop = min(C.deadline("gpu_stop_and_save_by"),
               now.__class__.fromtimestamp(time.time() + C.budget_left_hours()["gpu"] * 3600,
                                           now.tzinfo))
    return now.timestamp() + est_s * 1.25 + 600 <= stop.timestamp()


def run_pair(pt_job_id: str, recipe_name: str, recipe: dict, seed: int, epochs: int,
             stage: str, per_epoch_est_s: float | None):
    """PT + LP 成对执行; 返回 (lp_result|None, status_str)。"""
    lp_job_id = pt_job_id.replace("_pt", "_lp")
    assert MATRIX[pt_job_id]["kind"] == "pretrain" and lp_job_id in MATRIX
    # --- PT ---
    if done_of(pt_job_id) is None:
        if not _pt_deadline_ok("pretrain"):
            _mark(pt_job_id, "queued_not_started", "PT 截止(10-26 10:00)已过")
            return None, "queued"
        if C.budget_left_hours()["jobs"] <= 0:
            _mark(pt_job_id, "queued_not_started", "主训练作业 50 上限已用尽")
            return None, "queued"
        est = (per_epoch_est_s or 565.0) * epochs
        if not _est_fit(est):
            _mark(pt_job_id, "queued_not_started",
                  f"预计 {est/3600:.1f}h*1.25 不适配剩余窗口/预算")
            return None, "queued"
        C.count_job_start(pt_job_id, stage)
        C.log_event(f"启动 PT {pt_job_id} ({recipe_name}, seed{seed}, {epochs}ep)")
        _freeze_recipe_note(pt_job_id, recipe_name, recipe)
        cfg = {"job_id": pt_job_id, "stage": stage, "recipe_name": recipe_name,
               "seed": seed, "epochs": epochs, "recipe": recipe}
        t0 = time.time()
        try:
            pt_res = PT.run_pt_job(
                cfg, stop_deadline=C.deadline("gpu_stop_and_save_by"),
                budget_guard=lambda: C.budget_left_hours()["gpu"] > 0.02)
            _mark(pt_job_id, pt_res["status"], pt_res["stop_reason"])
            if pt_res["status"] == "trained":
                C.log_decision({"job": pt_job_id, "backup": C.backup_job_artifacts(pt_job_id)})
            else:
                return None, "partial"
        except Exception as e:
            C.add_resource("gpu", time.time() - t0, pt_job_id)
            _mark(pt_job_id, "failed", repr(e)[:300])
            C.log_event(f"PT {pt_job_id} 失败: {e!r}")
            raise
    else:
        pt_res = done_of(pt_job_id)
    # --- LP ---
    if done_of(lp_job_id) is None:
        if not _pt_deadline_ok("gpu"):
            _mark(lp_job_id, "queued_not_started", "GPU 截止(10-26 11:00)已过")
            return None, "queued"
        if C.budget_left_hours()["jobs"] <= 0:
            _mark(lp_job_id, "queued_not_started", "主训练作业 50 上限已用尽")
            return None, "queued"
        if pt_res["status"] != "trained" or "encoder_path" not in pt_res:
            _mark(lp_job_id, "not_started", "PT 未完整产出编码器")
            return None, "no_encoder"
        C.count_job_start(lp_job_id, stage)
        lp_cfg = {"job_id": lp_job_id, "stage": stage, "pt_job_id": pt_job_id,
                  "recipe": recipe_name, "seed": seed,
                  "encoder_path": pt_res["encoder_path"],
                  "encoder_sha256": pt_res["encoder_sha256"], "pt_epochs": epochs}
        try:
            lp_res = LP.run_lp_job(lp_cfg)
            _mark(lp_job_id, "complete", lp_res["stop_reason"])
            return lp_res, "complete"
        except Exception as e:
            _mark(lp_job_id, "failed", repr(e)[:300])
            C.log_event(f"LP {lp_job_id} 失败: {e!r}")
            raise
    lp_res = done_of(lp_job_id)
    return lp_res, "complete"


def _mark(job_id: str, status: str, note: str = ""):
    C.log_decision({"job": job_id, "status": status, "note": note})
    write_job_manifest()


def write_job_manifest():
    rows = []
    for job_id in sorted(MATRIX):
        row = dict(MATRIX[job_id])
        done = done_of(job_id)
        rj = C.JOBS / job_id / "result.json"
        res = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else None
        if done is not None:
            row["status"] = "complete"
        elif res is not None:
            row["status"] = res["status"]
        else:
            # decisions.jsonl 的最后状态(无记录则维持 planned/conditional)
            row["status"] = row.get("initial_status", "planned")
            p = C.RESULTS / "decisions.jsonl"
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    try:
                        d = json.loads(line)
                        if d.get("job") == job_id and "status" in d:
                            row["status"] = d["status"]
                    except Exception:
                        pass
        row.pop("initial_status", None)
        rows.append(row)
    (C.RESULTS / "job_manifest.json").write_text(
        json.dumps({"protocol_id": C.PROTOCOL["protocol_id"], "ts": C.bj_str(),
                    "jobs": rows}, ensure_ascii=False, indent=1), encoding="utf-8")


def stage_R1():
    lp_by = {}
    order = ["I0", "I1", "I2", "I3", "I4", "I5"]
    for i, name in enumerate(order):
        pt_id = f"r1_{name.lower()}_s0_e20_pt"
        res, _st = run_pair(pt_id, name, R1_RECIPES[name], 0, 20, "R1", per_epoch_est_s=None)
        if res:
            lp_by[(name, 0)] = res
    return lp_by


def stage_R2():
    lp_by = dict(lp_result_for("R1"))
    combos = resolve_r2(lp_by)
    C.log_decision({"gate": "R2", "combos": [(c["slot"], c["recipe_name"], c["reason"])
                                             for c in combos]})
    for c in combos:
        pt_id = f"r2_{c['slot']}_s0_e20_pt"
        res, _ = run_pair(pt_id, c["recipe_name"], c["recipe"], 0, 20, "R2", None)
        if res:
            lp_by[(c["recipe_name"], 0)] = res
    return lp_by


def stage_R3():
    lp_by = dict(lp_result_for("R1"))
    lp_by.update(lp_result_for("R2"))
    slots = resolve_r3(lp_by, pt_result_for("R1"))
    C.log_decision({"gate": "R3", "slots": [(s["slot"], s["recipe_name"], s["reason"])
                                            for s in slots]})
    for s in slots:
        pt_id = f"r3_{s['slot']}_s0_e20_pt"
        run_pair(pt_id, s["recipe_name"], s["recipe"], 0, 20, "R3", None)


def stage_R4():
    lp_by = dict(lp_result_for("R1"))
    lp_by.update(lp_result_for("R2"))
    lp_by.update(lp_result_for("R3"))
    winners = resolve_r4(lp_by)
    C.log_decision({"gate": "R4", "winners": winners})
    if not winners:
        C.log_event("R4: 无正向配方, 种子确认阶段结束(有效阴性)")
        return
    # I0 对照种子先行(无论 winner 是谁都需要的配对底座)
    for seed in (2, 4):
        run_pair(f"r4_i0_s{seed}_e20_pt", "I0", R1_RECIPES["I0"], seed, 20, "R4", None)
    for wi, w in enumerate(winners):
        recipe = R1_RECIPES.get(w["recipe_name"]) \
            or _recipe_from_configs(w["recipe_name"]) \
            or _recipe_from_decisions(w["recipe_name"])
        if recipe is None:
            C.log_event(f"R4: 配方 {w['recipe_name']} 定义缺失, 跳过(如实记录)")
            continue
        for seed in (2, 4):
            run_pair(f"r4_winner_slot{wi+1}_s{seed}_e20_pt",
                     w["recipe_name"], recipe, seed, 20, "R4", None)


def _recipe_from_configs(recipe_name: str):
    """从已跑 PT 作业的 jobs/<id>/config.json 找回配方定义(动态组合/权重档)。"""
    if not C.JOBS.exists():
        return None
    for jd in sorted(C.JOBS.iterdir()):
        cp = jd / "config.json"
        if not cp.exists():
            continue
        try:
            cfg = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if cfg.get("recipe_name") == recipe_name:
            return cfg.get("recipe")
    return None


def _freeze_recipe_note(job_id: str, recipe_name: str, recipe: dict):
    C.log_decision({"job": job_id, "recipe_frozen":
                    {"recipe_name": recipe_name, "recipe": recipe}})


def _recipe_from_decisions(recipe_name: str):
    p = C.RESULTS / "decisions.jsonl"
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("recipe_frozen", {}).get("recipe_name") == recipe_name:
            return d["recipe_frozen"]["recipe"]
    return None


def stage_R5():
    lp_by = dict(lp_result_for("R1"))
    for st in ("R2", "R3", "R4"):
        lp_by.update(lp_result_for(st))
    w = resolve_r5(lp_by)
    if w is None:
        C.log_decision({"gate": "R5", "decision": "no_positive_three_seed_mean",
                        "note": "无三种子平均增量>0 的配方, R5 不启动"})
        return
    C.log_decision({"gate": "R5", "winner": w, "note": "需时间/预算适配后才真正开跑"})
    recipe = R1_RECIPES.get(w["recipe_name"]) or _recipe_from_decisions(w["recipe_name"])
    if recipe is None:
        C.log_event(f"R5: 配方 {w['recipe_name']} 定义缺失")
        return
    pe = _measured_per_epoch(recipe)
    for seed in (0, 2, 4):
        run_pair(f"r5_i0_s{seed}_e100_pt", "I0", R1_RECIPES["I0"], seed, 100, "R5", pe)
        run_pair(f"r5_full_winner_s{seed}_e100_pt", w["recipe_name"], recipe, seed, 100, "R5", pe)


def _measured_per_epoch(recipe: dict) -> float | None:
    """用已完成同类作业(局部型/主干型)实测每 epoch 秒: 取最近匹配作业的中位。"""
    is_local = recipe.get("local_mode") is not None
    vals = []
    for jd in sorted(C.JOBS.iterdir() if C.JOBS.exists() else []):
        cfg_p = jd / "config.json"
        log_p = jd / "train_log.jsonl"
        if not (cfg_p.exists() and log_p.exists()):
            continue
        try:
            cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        except Exception:
            continue
        rc = cfg.get("recipe", {})
        if (rc.get("local_mode") is not None) != is_local:
            continue
        ep_walls = []
        for line in log_p.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                if d.get("event") == "epoch_end":
                    ep_walls.append(float(d["wall_s"]))
            except Exception:
                pass
        if ep_walls:
            vals = ep_walls  # 取最近一个匹配作业的全部 epoch 墙钟
    if not vals:
        return None
    vals = sorted(vals)
    return vals[len(vals) // 2]


def run_stage(stage: str):
    write_job_manifest()
    {"R1": stage_R1, "R2": stage_R2, "R3": stage_R3,
     "R4": stage_R4, "R5": stage_R5}[stage]()
    write_job_manifest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--stage", choices=["R1", "R2", "R3", "R4", "R5"])
    ap.add_argument("--auto", action="store_true")
    args = ap.parse_args()
    if args.preflight:
        from tools.thesis_inc import preflight
        sys.exit(preflight.main())
    if args.freeze:
        from tools.thesis_inc import make_resolved
        make_resolved.freeze()
        return
    if args.stage:
        run_stage(args.stage)
        return
    if args.auto:
        for st in ("R1", "R2", "R3", "R4", "R5"):
            C.log_event(f"--auto 进入 {st}")
            run_stage(st)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
