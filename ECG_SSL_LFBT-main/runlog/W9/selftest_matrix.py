# -*- coding: utf-8 -*-
"""W9 runner 矩阵自检(不跑任何实验): 30 任务唯一性/构成/路径存在性 + 既有 15 单元格三种子盘点。"""
import importlib.util
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "w9r", ROOT / "runlog/W9/run_w9_submission_ft10.py")
mod = importlib.util.module_from_spec(spec)
sys.argv = [spec.name]
spec.loader.exec_module(mod)

ts = mod.all_seed_tasks()
print("n_tasks =", len(ts))
print("parts:", dict(Counter(t["part"] for t in ts)))
cells = Counter((t["method"], t["dataset"]) for t in ts)
for k in sorted(cells):
    assert cells[k] == 3, (k, cells[k])
    print("cell", k, "x3")
for t in ts[:7]:
    print(t["idx"], t["tag"], t["ckpt"].name, "nc=%d" % t["nc"])
print("...")
for t in ts[-4:]:
    print(t["idx"], t["tag"], t["ckpt"].name)
missing = [str(t["ckpt"]) for t in ts if not (t["ckpt"] / "encoder_group.pth").exists()]
print("all ckpt exist:", not missing, missing or "")

# ---- 既有单元格三种子盘点(六方法×三域中不由 W9 补的部分) ----
import csv, glob
have = {}
for f in glob.glob(str(ROOT / "runlog/**/*.csv"), recursive=True):
    try:
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
    except Exception:
        continue
    for r in rows:
        if (r.get("eval") or "") != "ft10":
            continue
        ck = (r.get("ckpt") or "").strip()
        ds = (r.get("downstream") or "").strip()
        seed = (r.get("seed") or "").strip()
        key = None
        if ck == "simclr" and ds == "ptbxl":
            key = ("simclr", "ptbxl")
        elif ck == "simclr-w4frozen" and ds == "cpsc":
            key = ("simclr", "cpsc")
        elif ck == "simph":
            key = ("simph", ds)
        elif ck == "clocs" and ds == "ptbxl":
            key = ("clocs", "ptbxl")
        elif ck == "tsr":
            key = ("tsr", ds)
        elif ck == "dv" and ds == "ptbxl":
            key = ("m2", "ptbxl")
        if key:
            have.setdefault(key, set()).add(seed)
print("\nexisting cells (expect 3 seeds {0,2,4} each):")
for k in sorted(have):
    ok = have[k] == {"0", "2", "4"}
    print(" ", k, sorted(have[k]), "OK" if ok else "** MISSING SEEDS **")
expect_existing = {("simclr", "ptbxl"), ("simclr", "cpsc"), ("simph", "ptbxl"),
                   ("simph", "cpsc"), ("clocs", "ptbxl"), ("tsr", "ptbxl"),
                   ("tsr", "cpsc"), ("m2", "ptbxl")}
print("\ncoverage:", f"ALL {len(expect_existing)} EXISTING CELLS PRESENT (=24 runs)"
      if set(have) == expect_existing else f"DIFF: {expect_existing ^ set(have)}")
bad = [k for k in expect_existing if have.get(k) != {"0", "2", "4"}]
print("seed completeness:", "ALL 3-SEED" if not bad else f"BAD: {bad}")
