# -*- coding: utf-8 -*-
"""W6 Stage 1: FT20/FT40 标签效率曲线 — 双 GPU 车道驱动脚本。

用法(conda DL 解释器, 仓库根为 cwd, 分离进程启动):
    python runlog/W6/run_ft2040_lane.py L1   # 重车道: b0/c1 × {20,40} × seeds{0,2,4} = 12 跑
    python runlog/W6/run_ft2040_lane.py L2   # 机制车道: c2 × {20,40}(6) + 1b: simclr/clocs × 20(12) = 18 跑

车道纪律(任务书《车道纪律》, 红线级):
- 协议超参 = W2 FT10 冻结模板一字不动(epochs 100/batch 128/lr 1e-4/val 选点/test 只评一次),
  仅 --fraction 依格子取 0.2/0.4;
- 车道隔离: L1/L2 各写 ft2040_results_L{1,2}.csv 与各自 tag 目录, 不共写任何文件;
  阶段收口时由 CPU 车道纯拼接合并为 runlog/W6/ft2040_results.csv;
- workers 4/车道, 双车道合计 8 ≤ 12 上限;
- 幂等: 本车道 csv 已有 (ckpt,seed,eval,downstream) 行自动跳过, 断点重启只补缺;
- 单跑失败: 记 FAIL 后继续下一跑(跳行不空转), 不中止车道。

命令模板来源: runlog/W2/confirm_matrix.py::run_ft10 + W4 落盘三参数
(--extended-metrics 1 --save-predictions runlog/W6/predictions/{tag} --protocol-id w6-ft2040)。
checkpoint 映射同 W5A run_w5_cpscredo.py: (b0,0)->checkpoint/ptxl_gamma08,
其余 -> checkpoint/confirm/{kind}_seed{seed}; --trc 仅 c2=1。
"""
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.pathguard import open_out

PY = sys.executable
LOGD = ROOT / "runlog/W6/logs"
SEEDS = [0, 2, 4]
TRC = {"c2": "1", "c1": "0", "b0": "0", "simclr": "0", "clocs": "0"}
FAMILY = {"b0": "bt_ssl(baseline)", "c1": "bt_ssl", "c2": "bt_ssl",
          "simclr": "contrastive_ssl", "clocs": "contrastive_ssl"}
LANES = {
    "L1": [("b0", 0.2), ("c1", 0.2), ("b0", 0.4), ("c1", 0.4)],
    "L2": [("c2", 0.2), ("c2", 0.4), ("simclr", 0.2), ("clocs", 0.2)],
}


def ck_of(kind, seed):
    if (kind, seed) == ("b0", 0):
        return ROOT / "checkpoint/ptxl_gamma08"
    return ROOT / f"checkpoint/confirm/{kind}_seed{seed}"


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def git_sha():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def sha256_of(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def out_csv(lane):
    return ROOT / f"runlog/W6/ft2040_results_{lane}.csv"


def row_done(lane, ckpt, seed, ev):
    p = out_csv(lane)
    if not p.exists():
        return False
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["ckpt"], r["seed"], r["eval"], r["downstream"]) == (ckpt, str(seed), ev, "ptbxl"):
                return True
    return False


def record(lane, ckpt, seed, ev, auroc, auprc, cksha, frac):
    p = out_csv(lane)
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "ckpt", "seed", "eval", "downstream", "fraction",
                        "auroc", "auprc", "git_sha", "checkpoint_sha256",
                        "protocol_id", "method_family", "hparams_ref"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), ckpt, seed, ev, "ptbxl", f"{frac:g}",
                    f"{auroc:.4f}", f"{auprc:.4f}", git_sha(), cksha,
                    "w6-ft2040", FAMILY[ckpt],
                    "W6 Stage1: W2 FT10 冻结协议仅改标注比例, 见 runlog/W6/run_ft2040_lane.py"])
    log(f"  + L{lane} {ckpt} s{seed} {ev}/ptbxl: {auroc:.4f}/{auprc:.4f}")


def run_ft(lane, kind, frac, seed):
    ev = f"ft{int(round(frac * 100))}"
    tag = f"{kind}_ptbxl_{ev}_seed{seed}"
    if row_done(lane, kind, seed, ev):
        log(f"{tag}: 已有账, 跳过")
        return True
    ckdir = ck_of(kind, seed)
    ckpt = ckdir / "encoder_group.pth"
    if not ckpt.exists():
        log(f"{tag}: checkpoint 缺失 {ckpt}, 跳过(车道继续)")
        return False
    mdir = ROOT / f"runlog/W6/ft/{tag}"
    with open_out(LOGD, f"{tag}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_ft.py", "--data-dir", "data/ptbxl", "--num-classes", "5",
             "--fraction", str(frac), "--checkpoint", str(ckpt),
             "--model-dir", str(mdir), "--workers", "4", "--epochs", "100",
             "--batch-size", "128", "--learning-rate", "0.0001",
             "--seed", str(seed), "--trc", TRC[kind],
             "--extended-metrics", "1",
             "--save-predictions", f"runlog/W6/predictions/{tag}",
             "--protocol-id", "w6-ft2040"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((mdir / "metrics.json").read_text(encoding="utf-8"))
        record(lane, kind, seed, ev, m["auroc"], m["auprc"], sha256_of(ckpt), frac)
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e}; 见 runlog/W6/logs/{tag}.log 尾部)")
        return False


def main():
    lane = sys.argv[1] if len(sys.argv) > 1 else ""
    if lane not in LANES:
        sys.exit("用法: python runlog/W6/run_ft2040_lane.py {L1|L2}")
    LOGD.mkdir(parents=True, exist_ok=True)
    log(f"W6 Stage1 车道 {lane} 启动, 解释器={PY}, 作业表={LANES[lane]}")
    fails = 0
    for kind, frac in LANES[lane]:
        for seed in SEEDS:
            if not run_ft(lane, kind, frac, seed):
                fails += 1
    log(f"车道 {lane} 结束: {len(LANES[lane]) * len(SEEDS) - fails} 成功 / {fails} 失败")


if __name__ == "__main__":
    main()
