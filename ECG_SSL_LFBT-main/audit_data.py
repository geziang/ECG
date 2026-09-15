# -*- coding: utf-8 -*-
"""W0-3 数据审计：冻结协议数据完整性对账（stdlib-only，可在任何机器跑）。

检查项（对应《04-工程与数据集适配设计》W0-3 ①-⑤）：
  ① pt_pretrain=17,418 对账；pt_pretrain(17,418) vs pretrain(21,837) 甄别
  ②③ 两套下游划分体系（A: data/ptbxl 类子目录式 17,092 / B: data/downstream g-fold 式）计数与口径核对
  ④ checkpoint/ptxl_gamma08 SHA256 复核（对 E001 登记值）
  ⑤ 两套体系的患者隔离/泄漏核对（经 manifest + ptbxl_database.csv join）
输出：runlog/W0/audit_data_report.{json,md}（详细底账，作为 E008 第一份归档数据）
"""
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT_DIR = ROOT / "runlog" / "W0"
EXPECT_SHA256 = "026033d532d4e42a1b53ca9c4d2e95f9c09b2daaaab5c9041c9d474a1356cdad"
CLASSES = ["CD", "HYP", "MI", "NORM", "STTC"]


def count_files(d: Path, suffix=".npy") -> int:
    if not d.exists():
        return 0
    return sum(1 for _ in d.rglob(f"*{suffix}"))


def list_files(d: Path, suffix=".npy"):
    return [p for p in d.rglob(f"*{suffix}")] if d.exists() else []


def npy_header(path: Path):
    """最小 npy 头解析（不依赖 numpy）：返回 (descr, shape)。"""
    with open(path, "rb") as f:
        head = f.read(128)
    if head[:6] != b"\x93NUMPY":
        return None, None
    m = re.search(rb"\{'descr': '([^']+)', 'fortran_order': (True|False), 'shape': \(([^)]*)\)", head)
    if not m:
        return None, None
    descr = m.group(1).decode()
    shape = tuple(int(x) for x in m.group(3).decode().split(",") if x.strip())
    return descr, shape


def sha256_of(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"),
              "checks": {}, "pass": True}
    def fail(msg):
        report["pass"] = False
        print(f"  [FAIL] {msg}")

    # ---------- ① 预训练目录 ----------
    print("① 预训练目录对账 ...")
    pt_n = count_files(DATA / "pt_pretrain")
    pre_n = count_files(DATA / "pretrain")
    pt_names = {p.name for p in list_files(DATA / "pt_pretrain")[:50]}
    pre_names = {p.name for p in list_files(DATA / "pretrain")[:50]}
    pt_scheme = "sample_XXXXX" if any(n.startswith("sample_") for n in pt_names) else "HRxxxxx"
    pre_scheme = "HRxxxxx" if any(n.startswith("HR") for n in pre_names) else "sample_XXXXX"
    report["checks"]["pretrain_dirs"] = {
        "pt_pretrain_count": pt_n, "pt_pretrain_expect": 17418,
        "pt_pretrain_naming": pt_scheme,
        "pretrain_count": pre_n, "pretrain_expect_全量": 21837,
        "pretrain_naming": pre_scheme,
        "conclusion": "pt_pretrain=冻结协议(folds1-8, sample_索引命名)；pretrain=全集(HR命名, 未用)",
    }
    print(f"  pt_pretrain={pt_n} ({pt_scheme}) / pretrain={pre_n} ({pre_scheme})")
    if pt_n != 17418:
        fail(f"pt_pretrain 计数 {pt_n} != 17418")

    # ---------- ② 体系A: data/ptbxl ----------
    print("② 下游体系A（data/ptbxl，类子目录式）...")
    sysA = {}
    for split in ["train", "val", "test"]:
        per_cls = {c: count_files(DATA / "ptbxl" / split / c) for c in CLASSES}
        sysA[split] = {"per_class": per_cls, "total": sum(per_cls.values())}
    sysA["total"] = sum(v["total"] for v in sysA.values() if isinstance(v, dict))
    report["checks"]["downstream_A_ptbxl"] = sysA
    for s in ["train", "val", "test"]:
        print(f"  {s}: {sysA[s]['total']}  {sysA[s]['per_class']}")

    # ---------- ③ 体系B: data/downstream + manifest ----------
    print("③ 下游体系B（data/downstream，g-fold 式）+ manifest 对账 ...")
    sysB_files = {}
    for split in ["train", "val", "test"]:
        files = list_files(DATA / "downstream" / split)
        sysB_files[split] = files
    sysB_counts = {s: len(v) for s, v in sysB_files.items()}
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest.get("samples", manifest.get("records", []))
    man_split = Counter(e["split"] for e in entries)
    man_pretrain = Counter(e["in_pretrain"] for e in entries)
    report["checks"]["downstream_B_gfold"] = {
        "dir_counts": sysB_counts,
        "manifest_split_counts": dict(man_split),
        "manifest_in_pretrain": dict(man_pretrain),
        "manifest_naming": "HR%05d ↔ ecg_id（可 join 官方 csv）",
    }
    print(f"  目录: {sysB_counts}  manifest.split: {dict(man_split)}  in_pretrain: {dict(man_pretrain)}")
    # manifest 与体系A对账（A = 类子目录式，计数应逐 split 一致）
    a_match = all(sysA[s]["total"] == man_split.get(s, -1) for s in ["train", "val", "test"])
    report["checks"]["downstream_B_gfold"].update({
        "manifest_vs_A_match": a_match,
        "manifest_protocol_reading": (
            f"manifest 即体系A口径：downstream(train 13,639⊂pretrain 17,418) + 纯预训练 3,779 + "
            f"val/test 3,453(=in_pretrain False)；体系B为旧划分，与 manifest 偏离，待 W0 定版"),
    })
    print(f"  manifest vs 体系A 计数一致: {a_match}")
    if not a_match:
        fail("manifest 与体系A计数不一致（意外）")
    # 体系B内部重复检测（22,299 > 21,837 全集，需查明重复）
    all_hr = [p.name for files in sysB_files.values() for p in files]
    dup_names = {n: c for n, c in Counter(all_hr).items() if c > 1}
    name_splits = defaultdict(list)
    for split, files in sysB_files.items():
        for p in files:
            name_splits[p.name].append(split)
    dup_split_combos = Counter(tuple(sorted(name_splits[n])) for n in dup_names)
    report["checks"]["downstream_B_gfold"]["duplicate_files"] = {
        "total_files": len(all_hr), "unique": len(set(all_hr)),
        "dup_names_count": len(dup_names), "split_combos": {str(k): v for k, v in dup_split_combos.items()},
    }
    print(f"  体系B重复: {len(dup_names)} 个文件名重复, 组合 {dict(dup_split_combos)}")

    # ---------- ⑤ join 官方 csv：fold→split 映射、患者隔离、泄漏 ----------
    print("⑤ 患者隔离 / 泄漏核对（join ptbxl_database.csv）...")
    csv_path = ROOT / "ptb-xl" / "ptbxl_database.csv"
    ecg2patient, ecg2fold = {}, {}
    def to_int(x):
        return int(float(x))
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ecg2patient[to_int(row["ecg_id"])] = to_int(row["patient_id"])
            ecg2fold[to_int(row["ecg_id"])] = to_int(row["strat_fold"])
    fold_split_pairs = Counter()
    patient_splits = defaultdict(set)
    unmapped_ids = set()
    for split, files in sysB_files.items():
        for p in files:
            m = re.match(r"HR(\d{5})", p.name)
            if not m:
                continue
            ecg = int(m.group(1))
            if ecg not in ecg2patient:
                unmapped_ids.add(ecg)
                continue
            fold_split_pairs[(ecg2fold[ecg], split)] += 1
            patient_splits[ecg2patient[ecg]].add(split)
    iso_violations = sorted(pid for pid, ss in patient_splits.items() if len(ss) > 1)
    manifest_by_ecg = {}
    for e in entries:
        manifest_by_ecg[e["ecg_id"]] = e
    leakage = [e["ecg_id"] for e in entries
               if e["split"] in ("val", "test") and e.get("in_pretrain")]
    fold_table = {}
    for (fold, split), n in sorted(fold_split_pairs.items(), key=lambda x: (x[0][1], x[0][0] or 0)):
        fold_table.setdefault(split, {})[str(fold)] = n
    report["checks"]["isolation_leakage"] = {
        "fold_to_split": fold_table,
        "patient_isolation_violations": len(iso_violations),
        "violation_patient_ids_head": iso_violations[:10],
        "unmapped_ecg_ids": sorted(unmapped_ids)[:10],
        "unmapped_count": len(unmapped_ids),
        "in_pretrain_leakage_val_test": leakage[:10],
        "leakage_count": len(leakage),
    }
    print(f"  fold→split: {fold_table}")
    print(f"  患者跨 split 冲突: {len(iso_violations)}")
    if iso_violations:
        fail(f"体系B患者隔离被打破: {len(iso_violations)} 个患者跨 split")
    if leakage:
        fail(f"manifest 标记 val/test 且 in_pretrain 的记录 {len(leakage)} 条")

    # ---------- ④ checkpoint SHA256 ----------
    print("④ 基线权重 SHA256 复核 ...")
    ckpt = ROOT / "checkpoint" / "ptxl_gamma08" / "encoder_group.pth"
    got = sha256_of(ckpt)
    ok = got == EXPECT_SHA256
    report["checks"]["baseline_sha256"] = {"expect": EXPECT_SHA256, "got": got, "match": ok}
    print(f"  {'OK' if ok else 'MISMATCH'}: {got[:16]}...")
    if not ok:
        fail("基线权重 SHA256 与登记值不一致")

    # ---------- npy 抽查（形状/dtype）----------
    print("⑥ npy 抽查（各系统 3 条头信息）...")
    spot = []
    for tag, p in [("pt_pretrain", next(iter(list_files(DATA / 'pt_pretrain')))),
                   ("pretrain", next(iter(list_files(DATA / 'pretrain')))),
                   ("ptbxl", next(iter(list_files(DATA / 'ptbxl')))),
                   ("downstream", next(iter(list_files(DATA / 'downstream'))))]:
        d, s = npy_header(p)
        spot.append({"where": tag, "file": p.name, "descr": d, "shape": s,
                     "ok": s == (8, 2048) and d == "<f4"})
    report["checks"]["npy_spot_check"] = spot
    for x in spot:
        print(f"  {x['where']}: {x['file']} {x['descr']} {x['shape']} {'OK' if x['ok'] else 'BAD'}")
        if not x["ok"]:
            fail(f"npy 抽查不符: {x}")

    # ---------- 落盘 ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "audit_data_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    md = ["# W0-3 数据审计报告（自动生成）", "",
          f"- 生成时间：{report['generated_at']}",
          f"- 总判定：{'✅ PASS' if report['pass'] else '❌ 存在 FAIL 项'}", "",
          f"- 预训练：pt_pretrain={pt_n}（{pt_scheme}，冻结协议）/ pretrain={pre_n}（{pre_scheme}，全集未用）",
          f"- 体系A（ptbxl 类子目录）：train {sysA['train']['total']} / val {sysA['val']['total']} / test {sysA['test']['total']}",
          f"- 体系B（downstream g-fold）：{sysB_counts}；manifest.split={dict(man_split)}",
          f"- fold→split 映射：{json.dumps(fold_table, ensure_ascii=False)}",
          f"- 患者隔离冲突：{len(iso_violations)}；in_pretrain 泄漏：{len(leakage)}",
          f"- 基线 SHA256：{'一致' if ok else '不一致'}", ""]
    (OUT_DIR / "audit_data_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n判定: {'✅ PASS' if report['pass'] else '❌ FAIL'} → {OUT_DIR / 'audit_data_report.md'}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
