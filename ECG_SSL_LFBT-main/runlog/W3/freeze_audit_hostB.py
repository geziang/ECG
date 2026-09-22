# -*- coding: utf-8 -*-
"""B-0 冻结材料审计(主机B, CPU-only) —— 任务书《下一阶段双机任务书-2026-09-22》B-0。

只读审计 W2 冻结材料与 W1/W3 佐证文件, 产出 runlog/W3/freeze_audit_hostB.json。
不修改任何 W2 文件; 任一硬性检查失败 -> exit 1 (任务书停止线)。

用法: python runlog/W3/freeze_audit_hostB.py   (在 ECG_SSL_LFBT-main/ 下运行)
"""
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # ECG_SSL_LFBT-main/
W2 = ROOT / "runlog" / "W2"
W3 = ROOT / "runlog" / "W3"
W1 = ROOT / "runlog" / "W1"
M = ROOT / "runlog" / "M"

FAILS = []


def check(name, ok, detail=""):
    if not ok:
        FAILS.append(f"{name}: {detail}")
    return {"ok": bool(ok), "detail": detail}


def sha256_of(p: Path):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_rows(p: Path):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- 1. 文件存在性与 SHA256
AUDIT_FILES = [
    # W2 冻结集 (只读)
    "runlog/W2/confirm_matrix.py",
    "runlog/W2/confirm_results.csv",
    "runlog/W2/freeze_manifest_v2.json",
    "runlog/W2/missing_lead_c2.csv",
    "runlog/W2/missing_lead_c2_sweep.py",
    "runlog/W2/nfh_manifest_reconciliation.md",
    "runlog/W2/stats/main_table.csv",
    "runlog/W2/stats/paired_delta.csv",
    "runlog/W2/paper/missing_lead_heatmap.png",
    "runlog/W2/paper/pipeline_c1c2.png",
    # W3 佐证 (主机A A-0)
    "runlog/W3/freeze_audit.json",
    # W1 佐证 (单seed参考值与三锚点 SHA 来源)
    "runlog/W1/c3_evals.csv",
    "runlog/W1/freeze_manifest.json",
    "runlog/W1/nfh_manifest.json",
    "runlog/W1/cpsc_manifest.json",
    "runlog/W1/chapman_manifest.json",
    "runlog/W1/nfh_sha256.txt",
    "runlog/W1/missing_lead_curves.csv",
    # 双机探针账本 (failed_directions_index 的原始证据)
    "runlog/M/matrix_results.csv",
    "runlog/M/matrix_results_hostB.csv",
]
files_report = {}
for rel in AUDIT_FILES:
    p = ROOT / rel
    entry = {"exists": p.exists()}
    if p.exists():
        entry["bytes"] = p.stat().st_size
        entry["sha256"] = sha256_of(p)
    else:
        FAILS.append(f"missing file: {rel}")
    files_report[rel] = entry

# git 冻结版本核对: 审计对象在 HEAD 的 blob SHA 与工作区一致(未被改动)
git_checks = {}
for rel in AUDIT_FILES:
    r = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", f"HEAD:{rel}"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        git_checks[rel] = {"tracked": False}
        continue
    blob = r.stdout.strip()
    g = subprocess.run(
        ["git", "-C", str(ROOT), "hash-object", str(ROOT / rel)],
        capture_output=True, text=True,
    )
    same = g.stdout.strip() == blob
    git_checks[rel] = {"tracked": True, "worktree_equals_head": same}
    if not same:
        FAILS.append(f"worktree differs from HEAD: {rel}")

head_sha = subprocess.run(
    ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
    capture_output=True, text=True,
).stdout.strip()

# ---------------------------------------------------------------- 2. 冻结清单解析
manifest = json.loads((W2 / "freeze_manifest_v2.json").read_text(encoding="utf-8"))
code_sha = manifest["code_sha_at_runs"]

# ---------------------------------------------------------------- 3. W2 账本完整性
ledger_rows = read_csv_rows(W2 / "confirm_results.csv")
keys = [(r["ckpt"], r["seed"], r["eval"], r["downstream"]) for r in ledger_rows]
dup_keys = sorted({k for k in keys if keys.count(k) > 1})
git_shas = sorted({r["git_sha"] for r in ledger_rows})
ts_list = [r["ts"] for r in ledger_rows]
ledger_report = {
    "path": "runlog/W2/confirm_results.csv",
    "rows": len(ledger_rows),
    "rows_expect_28": check("ledger_rows_28", len(ledger_rows) == 28, f"got {len(ledger_rows)}"),
    "dup_keys": dup_keys,
    "dup_keys_zero": check("ledger_dup_keys", not dup_keys, str(dup_keys)),
    "git_sha_values": git_shas,
    "git_sha_uniform_fb08bc5": check(
        "ledger_git_sha", git_shas == [code_sha], f"{git_shas} vs {code_sha}"),
    "ts_monotonic": check("ledger_ts", ts_list == sorted(ts_list), "ts not monotonic"),
}

# checkpoint_sha256 前缀 <-> manifest 全长 SHA
def manifest_key(ckpt, seed):
    m = {"b0": {2: "b0_seed2", 4: "b0_seed4"},
         "c1": {0: "c1_seed0", 2: "c1_seed2", 4: "c1_seed4"},
         "c2": {0: "c2_seed0", 2: "c2_seed2", 4: "c2_seed4"}}
    return m[ckpt][int(seed)]


prefix_report = {}
for r in ledger_rows:
    mk = manifest_key(r["ckpt"], r["seed"])
    full = manifest["checkpoints"][mk]["sha256"]
    ok = full.startswith(r["checkpoint_sha256"])
    prefix_report[f'{r["ckpt"]}_s{r["seed"]}_{r["eval"]}_{r["downstream"]}'] = {
        "ledger_prefix": r["checkpoint_sha256"], "manifest": mk, "match": ok}
    if not ok:
        FAILS.append(f"ckpt sha prefix mismatch {r}")
ledger_report["checkpoint_sha_prefix_all_match"] = check(
    "ckpt_prefix", all(v["match"] for v in prefix_report.values()))

# ---------------------------------------------------------------- 4. 统计复算
# 4a. 从账本 + W1 单seed参考值 重构 main_table
w1_ref = {}
for r in read_csv_rows(W1 / "c3_evals.csv"):
    w1_ref[(r["ckpt"], r["eval"], r["downstream"])] = float(r["auprc"])

W1_REF_MAP = {  # main_table 的 seed0 参考列来源
    ("lp", "ptbxl", "b0"): ("b0_anchor", "lp", "ptbxl"),
    ("ft10", "ptbxl", "b0"): ("b0_anchor", "ft10", "ptbxl"),
    ("lp", "cpsc", "b0"): ("b0_anchor", "lp", "cpsc"),      # 单seed参考, n_seeds=1
    ("lp", "chapman", "b0"): ("b0_anchor", "lp", "chapman"),
    ("lp", "ptbxl", "c1"): ("c1_nfh_b0", "lp", "ptbxl"),    # 逐位一致性佐证(仅记录)
    ("lp", "cpsc", "c1"): ("c1_nfh_b0", "lp", "cpsc"),
    ("lp", "chapman", "c1"): ("c1_nfh_b0", "lp", "chapman"),
    ("lp", "ptbxl", "c2"): ("c2_nfh_trc", "lp", "ptbxl"),
    ("lp", "cpsc", "c2"): ("c2_nfh_trc", "lp", "cpsc"),
    ("lp", "chapman", "c2"): ("c2_nfh_trc", "lp", "chapman"),
    ("ft10", "ptbxl", "c1"): ("c1_nfh_b0", "ft10", "ptbxl"),
}

def seeds_for(eval_, ds, ck):
    got = {}
    if (eval_, ds, ck) in W1_REF_MAP and eval_ in ("lp", "ft10") and (
            (eval_, ds, ck) in (("lp", "cpsc", "b0"), ("lp", "chapman", "b0"))):
        return {0: w1_ref[W1_REF_MAP[(eval_, ds, ck)]]}  # b0 外部域: W1 单seed参考
    for r in ledger_rows:
        if r["eval"] == eval_ and r["downstream"] == ds and r["ckpt"] == ck:
            got[int(r["seed"])] = float(r["auprc"])
    if (eval_, ds, ck) in W1_REF_MAP and 0 not in got:
        got[0] = w1_ref[W1_REF_MAP[(eval_, ds, ck)]]
    return got

main_rows = read_csv_rows(W2 / "stats" / "main_table.csv")
mt_report = []
for r in main_rows:
    ev, ds, ck = r["eval"], r["downstream"], r["ckpt"]
    sd_map = seeds_for(ev, ds, ck)
    vals = [sd_map[s] for s in sorted(sd_map)]
    mean = sum(vals) / len(vals)
    sd = (math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
          if len(vals) > 1 else 0.0)
    per_seed = "/".join(f"{sd_map[s]:.4f}" for s in sorted(sd_map))
    mt_report.append({
        "cell": f"{ev}/{ds}/{ck}",
        "recomputed": {"n": len(vals), "mean": round(mean, 4),
                       "sd": round(sd, 4), "per_seed": per_seed},
        "frozen": {"n": int(r["n_seeds"]), "mean": float(r["mean"]),
                   "sd": float(r["sd"]), "per_seed": r["per_seed"]},
        "match": (len(vals) == int(r["n_seeds"])
                  and abs(round(mean, 4) - float(r["mean"])) < 5e-5
                  and abs(round(sd, 4) - float(r["sd"])) < 5e-5
                  and per_seed == r["per_seed"]),
    })
mt_ok = all(c["match"] for c in mt_report)
if not mt_ok:
    FAILS.append("main_table recompute mismatch")

# 4b. paired_delta 复算 (seed 配对, pt; 精确符号翻转置换; t(2) CI)
def cell(ev, ds, ck, s):
    m = seeds_for(ev, ds, ck)
    return m[s]

def exact_perm_p_one_sided(deltas):
    """H1: sum>0 的符号翻转精确单侧 p (等于 abs-sum 集合上翻转)."""
    n = len(deltas)
    obs = sum(deltas)
    hit = tot = 0
    for mask in range(1 << n):
        s = sum(d if (mask >> i) & 1 else -d for i, d in enumerate(deltas))
        tot += 1
        if s >= obs - 1e-12:
            hit += 1
    return hit / tot

T2 = 4.302655  # t(0.975, df=2)
pd_rows = read_csv_rows(W2 / "stats" / "paired_delta.csv")
pd_report = []
for r in pd_rows:
    ev, ds, pair = r["eval"], r["downstream"], r["pair"]
    a, b = pair.split("-")  # C2-C1 / C1-B0
    seeds = [0, 2, 4]
    ds_pt = []
    for s in seeds:
        try:
            d = (cell(ev, ds, a.lower(), s) - cell(ev, ds, b.lower(), s)) * 100
        except KeyError:
            d = None
        ds_pt.append(d)
    if any(d is None for d in ds_pt):
        pd_report.append({"pair": f"{ev}/{ds}/{pair}", "match": False,
                          "note": "seed cell missing (C1-B0 仅 ptbxl 有 b0 三seed)"})
        continue
    mean_pt = sum(ds_pt) / 3
    sd_pt = math.sqrt(sum((d - mean_pt) ** 2 for d in ds_pt) / 2)
    ci_lo, ci_hi = mean_pt - T2 * sd_pt / math.sqrt(3), mean_pt + T2 * sd_pt / math.sqrt(3)
    sign = "".join("+" if d > 1e-9 else ("-" if d < -1e-9 else "0") for d in ds_pt)
    p = exact_perm_p_one_sided(ds_pt)
    exp = {
        "d_s0": f'{ds_pt[0]:+.2f}', "d_s2": f'{ds_pt[1]:+.2f}', "d_s4": f'{ds_pt[2]:+.2f}',
        "mean_pt": f"{mean_pt:+.2f}", "sign": sign,
        "perm_p": f"{p:.3f}", "ci_lo": f"{ci_lo:+.2f}", "ci_hi": f"{ci_hi:+.2f}",
    }
    frozen = {"d_s0": r["d_s0"], "d_s2": r["d_s2"], "d_s4": r["d_s4"],
              "mean_pt": r["mean_pt"], "sign": r["sign"],
              "perm_p": f'{float(r["perm_p_1sided"]):.3f}'}
    ci_ok = True
    if r["ci95_low"] not in ("NA", ""):
        ci_ok = (abs(float(r["ci95_low"]) - ci_lo) < 0.015
                 and abs(float(r["ci95_high"]) - ci_hi) < 0.015)
    match = all(exp[k] == frozen[k] for k in frozen) and ci_ok
    pd_report.append({"pair": f"{ev}/{ds}/{pair}", "recomputed": exp,
                      "frozen": frozen, "match": match})
    if not match:
        FAILS.append(f"paired_delta mismatch {ev}/{ds}/{pair}")
pd_ok = all(c["match"] for c in pd_report)

# 4c. headline 数字
c2c1_cpsc = next(c for c in pd_report if c["pair"] == "lp/cpsc/C2-C1")
c2c1_ft10 = next(c for c in pd_report if c["pair"] == "ft10/ptbxl/C2-C1")
headline = {
    "cpsc_lp_c2_minus_c1_mean_pt": c2c1_cpsc["recomputed"]["mean_pt"],
    "cpsc_lp_sign": c2c1_cpsc["recomputed"]["sign"],
    "ft10_ptbxl_c2_minus_c1_mean_pt": c2c1_ft10["recomputed"]["mean_pt"],
    "ft10_sign": c2c1_ft10["recomputed"]["sign"],
    "perm_p_method": "exact sign-flip permutation, one-sided, n=3 (2^-3=0.125 floor)",
    "ci_method": "t(0.975,df=2)=4.3027 * sd(ddof=1)/sqrt(3)",
}

# ---------------------------------------------------------------- 5. 清单内部一致性
ck = manifest["checkpoints"]
params = manifest["params"]
internal = {
    "c2_seed0_sha_equals_w1_c2_nfh_seed0": check(
        "c2_seed0==W1", ck["c2_seed0"]["sha256"] == ck["c2_nfh_seed0(W1)"]["sha256"]),
    "trc_1024_over_635304_pct": round(1024 / 635304 * 100, 4),
    "trc_share_pct_claim_0.161": check(
        "trc_share", abs(1024 / 635304 * 100 - 0.161) < 5e-4),
    "trc_param_count_128x8": check("trc_count", params["trc_total"] == 128 * 8),
    "w1_anchor_shas_in_v2": {
        "b0_anchor(W1)": check("b0_anchor",
            ck["b0_anchor(W1)"]["sha256"].startswith(
                json.loads((W1 / "freeze_manifest.json").read_text(encoding="utf-8"))
                ["checkpoints"]["b0_anchor"][:32])),
        "c1_nfh_seed0(W1)": check("c1_w1",
            ck["c1_nfh_seed0(W1)"]["sha256"] == json.loads(
                (W1 / "freeze_manifest.json").read_text(encoding="utf-8"))
            ["checkpoints"]["c1_nfh_b0_seed0"]),
        "c2_nfh_seed0(W1)": check("c2_w1",
            ck["c2_nfh_seed0(W1)"]["sha256"] == json.loads(
                (W1 / "freeze_manifest.json").read_text(encoding="utf-8"))
            ["checkpoints"]["c2_nfh_trc_seed0"]),
    },
    "note_checkpoints_params_total_zero": (
        "manifest checkpoints.*.params_total=0 为登记缺口(参数量在 params 块: "
        "c2_total=635304/c1_total=634280); 不影响 SHA 一致性判定"),
}

# W1 c3_evals 读数与 W2 账本 seed0 逐位一致性(audit_notes 主张)
bit_rows = [
    ("c1", 0, "lp", "ptbxl", "c1_nfh_b0", "lp", "ptbxl"),
    ("c1", 0, "lp", "cpsc", "c1_nfh_b0", "lp", "cpsc"),
    ("c1", 0, "lp", "chapman", "c1_nfh_b0", "lp", "chapman"),
    ("c2", 0, "lp", "ptbxl", "c2_nfh_trc", "lp", "ptbxl"),
    ("c2", 0, "lp", "cpsc", "c2_nfh_trc", "lp", "cpsc"),
    ("c2", 0, "lp", "chapman", "c2_nfh_trc", "lp", "chapman"),
    ("c1", 0, "ft10", "ptbxl", "c1_nfh_b0", "ft10", "ptbxl"),
]
w1_w2_bit = {}
for ckpt_b, s, ev, ds, ck_w1, ev_w1, ds_w1 in bit_rows:
    try:
        row = next(r for r in ledger_rows if r["ckpt"] == ckpt_b and int(r["seed"]) == s
                   and r["eval"] == ev and r["downstream"] == ds)
        v_b = row["auprc"]
    except StopIteration:
        v_b = None
    v_w1 = f'{w1_ref[(ck_w1, ev_w1, ds_w1)]:.4f}'
    w1_w2_bit[f"{ckpt_b}_{ev}_{ds}_s0"] = {"w2_ledger": v_b, "w1_c3_evals": v_w1,
                                            "bit_identical": v_b == v_w1}
    if v_b != v_w1:
        FAILS.append(f"W1/W2 not bit-identical {ckpt_b} {ev} {ds}")

# ---------------------------------------------------------------- 6. 与主机A freeze_audit.json 交叉核对
fa = json.loads((W3 / "freeze_audit.json").read_text(encoding="utf-8"))
cross_a = {
    "a_all_match": fa.get("all_match") is True,
    "a_ledger_rows": fa["ledger"]["rows"],
    "a_dup_keys": fa["ledger"]["dup_keys"],
    "prefix16_are_prefix_of_manifest": {},
    "signs_agree_with_B_recompute": check(
        "a_signs", fa["ledger"]["signs"]["c2-c1 lp/cpsc"]["sign"] == c2c1_cpsc["recomputed"]["sign"]
        and fa["ledger"]["signs"]["c2-c1 ft10/ptbxl"]["sign"] == c2c1_ft10["recomputed"]["sign"]),
}
for name, rec in fa["checkpoints"].items():
    full = manifest["checkpoints"].get(name, {}).get("sha256", "")
    cross_a["prefix16_are_prefix_of_manifest"][name] = full.startswith(rec["recomputed"])
if not all(cross_a["prefix16_are_prefix_of_manifest"].values()):
    FAILS.append("hostA 16-char prefixes not consistent with v2 manifest")

# ---------------------------------------------------------------- 7. 数据 manifest 与缺导表
nfh_m = json.loads((W1 / "nfh_manifest.json").read_text(encoding="utf-8"))
ml = read_csv_rows(W2 / "missing_lead_c2.csv")
ml_conds = [r for r in ml if r["cond"] == "full"]
data_reports = {
    "nfh_manifest_n_ok": nfh_m.get("n_ok", nfh_m.get("n_records")),
    "nfh_manifest_preprocess_version": nfh_m.get("preprocess_version"),
    "missing_lead_c2_rows": len(ml),
    "missing_lead_c2_rows_expect_111": check("ml_rows", len(ml) == 3 * 37, f"got {len(ml)}"),
    "missing_lead_full_matches_ledger_s0124": check(
        "ml_full", all(
            abs(float(r["auprc"]) - cell("lp", "ptbxl", "c2", int(r["seed"]))) < 5e-5
            for r in ml_conds)),
    "cpsc_manifest_exists": (W1 / "cpsc_manifest.json").exists(),
    "chapman_manifest_exists": (W1 / "chapman_manifest.json").exists(),
}

# ---------------------------------------------------------------- 汇总输出
out = {
    "task": "B-0 freeze material audit (git-side, CPU-only, read-only)",
    "host": "B(DESKTOP-0PBLCND)",
    "date": subprocess.run(["git", "-C", str(ROOT), "show", "-s", "--format=%cI", "HEAD"],
                           capture_output=True, text=True).stdout.strip()[:10],
    "audit_head_git_sha": head_sha,
    "runs_code_sha_declared": code_sha,
    "files": files_report,
    "git_frozen_track": git_checks,
    "ledger": ledger_report,
    "checkpoint_prefix_vs_manifest": prefix_report,
    "manifest_internal": internal,
    "w1_w2_bit_identical_reads": w1_w2_bit,
    "stats_recompute": {
        "main_table": {"rows": mt_report, "all_match": mt_ok},
        "paired_delta": {"rows": pd_report, "all_match": pd_ok},
        "headline": headline,
    },
    "cross_check_hostA_freeze_audit": cross_a,
    "data_manifests_and_missing_lead": data_reports,
    "all_pass": not FAILS,
    "failures": FAILS,
    "limitations": [
        "B 侧无 checkpoint 本体(在主机A), SHA 复核采用 git 冻结材料内一致性+前缀匹配;"
        "字节级独立重算以主机A freeze_audit.json(11/11)为准",
        "W2 未保存逐记录预测概率 -> 仅 seed-level 描述统计+精确置换, 不得声称 patient-level bootstrap",
        "b0 的 CPSC/Chapman 为 W1 单seed参考列(n_seeds=1, 无 SD)",
        "绝对 AUPRC 跨机不可比(历史差 0.8~1.0pt); 跨机只比方向与符号",
        "NFH 无患者ID(记录内容 SHA 查重替代); CPSC/Chapman record-level split",
    ],
}

out_path = W3 / "freeze_audit_hostB.json"
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"written {out_path}")
print("ALL PASS" if not FAILS else f"FAILURES ({len(FAILS)}): " + "; ".join(FAILS))
sys.exit(0 if not FAILS else 1)
