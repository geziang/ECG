# -*- coding: utf-8 -*-
"""A-6: S3 监督直训参照(TFS 随机初始化, PTB 五类全量), seeds {0,2,4} 串行."""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
ROWS = "runlog/W4/baseline_results.csv"
CODE_SHA = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()

for seed in (0, 2, 4):
    tag = f"s3_seed{seed}"
    log = ROOT / f"runlog/W4/logs/{tag}.log"
    cmd = [PY, "-u", "run_ft.py", "--data-dir", "data/ptbxl", "--num-classes", "5",
           "--fraction", "1.0", "--epochs", "100", "--batch-size", "128",
           "--learning-rate", "0.0001", "--seed", str(seed),
           "--model-dir", f"runlog/W4/ft/{tag}",
           "--extended-metrics", "1",
           "--save-predictions", f"runlog/W4/predictions/s3_ptbxl_seed{seed}",
           "--protocol-id", "w4-s3-supervised"]
    print(f"[{datetime.now():%H:%M:%S}] {tag} start", flush=True)
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    if rc != 0:
        print(f"[{datetime.now():%H:%M:%S}] {tag} FAIL rc={rc} -> {log}", flush=True)
        sys.exit(rc)
    m = json.loads((ROOT / f"runlog/W4/ft/{tag}/metrics.json").read_text(encoding="utf-8"))
    row = (f"{datetime.now():%Y-%m-%d %H:%M},s3,{seed},ft100,ptbxl,"
           f"{m['auroc']:.4f},{m['auprc']:.4f},{CODE_SHA},random-init-TFS,"
           f"supervised_direct,run_ft默认超参(任务书A-6: lr1e-4/100ep/bs128/全量)")
    with open(ROOT / ROWS, "a", encoding="utf-8") as f:
        f.write(row + "\n")
    print(f"[{datetime.now():%H:%M:%S}] {tag} auroc={m['auroc']:.4f} auprc={m['auprc']:.4f}", flush=True)
print("S3 ALL DONE", flush=True)
