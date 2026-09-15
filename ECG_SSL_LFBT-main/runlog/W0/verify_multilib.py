# -*- coding: utf-8 -*-
"""W0-6 验收:泄漏单测 + 格式校验 + 每库 8 条波形抽查图。"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ML = ROOT / "data" / "multilib"
VIZ = ROOT / "runlog" / "W0" / "multilib_viz"
VIZ.mkdir(parents=True, exist_ok=True)
LEADS = ["II", "III", "V1", "V2", "V3", "V4", "V5", "V6"]

FAILS = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)

m = json.loads((ML / "manifest.json").read_text(encoding="utf-8"))
recs = m["records"]

# 1) 泄漏: 与 PTB-XL 下游 (data/ptbxl) 无交集; 文件名全局唯一
ml_files = {r["file"] for r in recs}
check("文件名唯一", len(ml_files) == len(recs) == 21264)
check("ptb-xl 未入库", not any(r["lib"] == "ptb-xl" for r in recs))
ptbxl_ids = {p.name for split in ("train", "val", "test")
             for cls in (ROOT / "data" / "ptbxl" / split).iterdir()
             for p in list(cls.glob("*.npy"))}
check("与 PTB-XL 下游零交集 (id 命名空间隔离)",
      not (ml_files & {f"ptbxl__{i}" for i in ptbxl_ids}))

# 2) 抽样 200 条格式校验: shape/dtype/finite/z-score 口径
rng = np.random.default_rng(0)
sample = rng.choice(recs, size=min(200, len(recs)), replace=False)
bad = 0
for r in sample:
    a = np.load(ML / "samples" / r["file"])
    ok = (a.shape == (8, 2048) and a.dtype == np.float32 and np.isfinite(a).all()
          and abs(a.std() - 1.0) < 0.05)
    bad += (not ok)
check(f"抽样 200 条 (8,2048) float32 有限值 std≈1 [bad={bad}]", bad == 0)

# 3) manifest 与磁盘一致 (抽查 sha256)
sha_bad = 0
import hashlib
for r in rng.choice(recs, size=20, replace=False):
    h = hashlib.sha256((ML / "samples" / r["file"]).read_bytes()).hexdigest()
    sha_bad += (h != r.get("sha256"))
check(f"抽查 20 条 SHA256 与 manifest 一致 [bad={sha_bad}]", sha_bad == 0)

# 4) 每库 8 条可视化: 波形 8 导联 (导联序+z-score 形态检查)
for lib in m["counts"]:
    lib_recs = [r for r in recs if r["lib"] == lib]
    picks = rng.choice(lib_recs, size=min(8, len(lib_recs)), replace=False)
    fig, axes = plt.subplots(8, 8, figsize=(16, 10), sharex=True)
    for col, r in enumerate(picks):
        a = np.load(ML / "samples" / r["file"])
        for row in range(8):
            ax = axes[row, col]
            ax.plot(a[row], lw=0.4, color="tab:blue")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(LEADS[row], rotation=0, ha="right", fontsize=7)
        axes[0, col].set_title(r["record_id"], fontsize=7)
    fig.suptitle(f"{lib} — 8 samples × 8 leads (z-score, 2048pts)", fontsize=10)
    fig.tight_layout()
    fig.savefig(VIZ / f"{lib}.png", dpi=110)
    plt.close(fig)
    print(f"viz saved: {lib}.png")

print("\nALL PASS" if not FAILS else f"\nFAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
