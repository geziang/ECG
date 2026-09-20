# -*- coding: utf-8 -*-
"""smoke_w1.sh — W1 开关 GPU 冒烟(任务书 07 §一/§六: 5 epoch smoke 纪律; 此处 1 epoch 快验)

依次验证四条新路径端到端(预训练->checkpoint 保存->诊断量打印):
  1) d1l    : D1L-fix(τ 对角), 检查 target_matrix.npy/d1l_check.json 工件
  2) acl    : T2 区域目标 full 模式, 检查 last_extra(acl_intra/acl_inter) 打印
  3) multiseg: B2 多段 mask + masked MSE, 检查 loss_rec/grad_rec_bt 打印
  4) ccm    : B3 周期遮挡(需 data/pt_rpeaks.npz), 检查 ccm_stats 命中率
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
CK = ROOT / "checkpoint" / "W1_smoke"
LOGD = ROOT / "runlog" / "W1"

CASES = [
    ("d1l", ["--d1l", "0.5,0.2"]),
    ("acl", ["--acl-mode", "full", "--acl-partition", "random", "--acl-rand-seed", "101"]),
    ("multiseg", ["--rec-style", "multiseg", "--d7-weight", "0.1", "--rec-ratio", "0.2"]),
]
if (ROOT / "data" / "pt_rpeaks.npz").exists():
    CASES.append(("ccm", ["--rec-style", "ccm", "--d7-weight", "0.1", "--rec-ratio", "0.2"]))
else:
    print("[smoke] data/pt_rpeaks.npz 尚未生成, 跳过 ccm(队列启动前必须完成)")

LOGD.mkdir(parents=True, exist_ok=True)
fail = 0
for name, extra in CASES:
    ck = CK / name
    cmd = [PY, "run_pt.py", "--data-dir", "data/pt_pretrain", "--epochs", "1",
           "--batch-size", "128", "--workers", "4", "--seed", "0",
           "--checkpoint-dir", str(ck)] + extra
    logp = LOGD / f"smoke_{name}.log"
    print(f"[smoke] {name}: {' '.join(extra)} -> {logp}", flush=True)
    with logp.open("w", encoding="utf-8", errors="replace") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    text = logp.read_text(encoding="utf-8", errors="replace")
    ok = ("Checkpoint saved" in text) and ("SHA256" in text)
    checks = []
    if name == "d1l":
        checks = [(ck / "target_matrix.npy").exists(), (ck / "d1l_check.json").exists()]
    if name == "acl":
        checks = ["acl_intra" in text, "acl_inter" in text]
    if name == "multiseg":
        checks = ["loss_rec" in text, "grad_rec_bt" in text]
    if name == "ccm":
        checks = ["ccm_stats" in text, "loss_rec" in text]
    allok = ok and all(checks)
    print(f"[smoke] {name}: {'PASS' if allok else 'FAIL'} (rc={r.returncode}, "
          f"ckpt={ok}, diag={checks})", flush=True)
    if not allok:
        fail += 1
print(f"[smoke] done, fail={fail}")
sys.exit(1 if fail else 0)
