# -*- coding: utf-8 -*-
"""W6 Stage 3: 三域 FT10 网格补全(CPSC + Chapman) — GPU 车道 1 接力脚本。

用法(conda DL 解释器, 仓库根为 cwd, 分离进程启动):
    python runlog/W6/run_ft10grid_lane.py L1   # {b0,c1,c2} × seeds{0,2,4} × {cpsc,chapman} = 18 跑
                                               # (排程: 先 CPSC 9 跑[前半] 再 Chapman 9 跑[后半])

按任务书《多车道排程》: 本脚本由巡检在 L1 的 Stage 1(FT20/40 前半)排空后自动接力起跑。
协议 = W2 FT10 冻结模板(fraction 0.1), 仅数据集与落盘三参数按 W6 Stage 3 任务书:
  --extended-metrics 1 --save-predictions runlog/W6/predictions/{tag} --protocol-id w6-ft10grid
车道纪律同 run_ft2040_lane.py(车道隔离 csv: ft10grid_results_L1.csv; 幂等跳已记账行;
单跑失败记 FAIL 继续下一跑)。CPSC 用 W5A 修正分区(data/cpsc, 9 类, test=1385);
Chapman 同 W1 口径(data/chapman, 4 类)。checkpoint 映射与 --trc 同 W5A/W4 惯例。
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
TRC = {"c2": "1", "c1": "0", "b0": "0"}
FAMILY = {"b0": "bt_ssl(baseline)", "c1": "bt_ssl", "c2": "bt_ssl"}
DATASETS = [("cpsc", 9), ("chapman", 4)]  # 排程顺序: 前半 CPSC -> 后半 Chapman
LANES = {"L1": [(k, ds, nc) for (ds, nc) in DATASETS for k in ("b0", "c1", "c2")]}


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


def out_csv():
    return ROOT / "runlog/W6/ft10grid_results_L1.csv"


def row_done(ckpt, seed, ds):
    p = out_csv()
    if not p.exists():
        return False
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["ckpt"], r["seed"], r["eval"], r["downstream"]) == (ckpt, str(seed), "ft10", ds):
                return True
    return False


def record(ckpt, seed, ds, auroc, auprc, cksha):
    p = out_csv()
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "ckpt", "seed", "eval", "downstream", "fraction",
                        "auroc", "auprc", "git_sha", "checkpoint_sha256",
                        "protocol_id", "method_family", "hparams_ref"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), ckpt, seed, "ft10", ds, "0.1",
                    f"{auroc:.4f}", f"{auprc:.4f}", git_sha(), cksha,
                    "w6-ft10grid", FAMILY[ckpt],
                    "W6 Stage3: W2 FT10 冻结协议换数据集, 见 runlog/W6/run_ft10grid_lane.py"])
    log(f"  + L1 {ckpt} s{seed} ft10/{ds}: {auroc:.4f}/{auprc:.4f}")


def run_ft(kind, ds, nc, seed):
    tag = f"{kind}_{ds}_ft10_seed{seed}"
    if row_done(kind, seed, ds):
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
            [PY, "-u", "run_ft.py", "--data-dir", f"data/{ds}", "--num-classes", str(nc),
             "--fraction", "0.1", "--checkpoint", str(ckpt),
             "--model-dir", str(mdir), "--workers", "4", "--epochs", "100",
             "--batch-size", "128", "--learning-rate", "0.0001",
             "--seed", str(seed), "--trc", TRC[kind],
             "--extended-metrics", "1",
             "--save-predictions", f"runlog/W6/predictions/{tag}",
             "--protocol-id", "w6-ft10grid"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((mdir / "metrics.json").read_text(encoding="utf-8"))
        record(kind, seed, ds, m["auroc"], m["auprc"], sha256_of(ckpt))
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e}; 见 runlog/W6/logs/{tag}.log 尾部)")
        return False


def main():
    lane = sys.argv[1] if len(sys.argv) > 1 else ""
    if lane not in LANES:
        sys.exit("用法: python runlog/W6/run_ft10grid_lane.py {L1}")
    LOGD.mkdir(parents=True, exist_ok=True)
    log(f"W6 Stage3 车道 {lane} 启动, 解释器={PY}, 作业表={LANES[lane]}")
    fails = 0
    for kind, ds, nc in LANES[lane]:
        for seed in SEEDS:
            if not run_ft(kind, ds, nc, seed):
                fails += 1
    log(f"车道 {lane} 结束: {len(LANES[lane]) * len(SEEDS) - fails} 成功 / {fails} 失败")


if __name__ == "__main__":
    main()
