# -*- coding: utf-8 -*-
"""W8 E1 w8-rand: 随机初始化地板参照。

跑法(conda DL 解释器, 仓库根为 cwd, 分离进程启动):
    python runlog/W8/run_e1_rand.py

18 跑 = LP 三域×seeds{0,2,4}(随机初始化冻结特征+线性头, --checkpoint none)
      + FT10 三域×seeds{0,2,4}(随机初始化全参微调, 省略 --checkpoint, fraction 0.1)。
协议: LP=run_lp 默认超参(100ep/bs128/lr1e-3); FT=run_ft 默认超参(100ep/bs128/lr1e-4/f0.1),
与 W4 S3/W5/W6 模板同构, --trc 0(C1 等价架构)。先跑完 LP 9 跑(sanity 解锁)再跑 FT10。
幂等: w8_rand_results.csv 已有行且预测产物在 -> 跳过; 单跑失败记 FAIL 继续下一跑。
判定见 runlog/W8/events.md 10-10 预注册块(先写后跑)。
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.pathguard import open_out

PY = sys.executable
OUT = ROOT / "runlog/W8/w8_rand_results.csv"
LOGD = ROOT / "runlog/W8/logs"
PRED = "runlog/W8/predictions"
DOMAINS = [("ptbxl", 5), ("cpsc", 9), ("chapman", 4)]
SEEDS = [0, 2, 4]
HP_LP = ("W8 E1: run_lp 默认超参(100ep/bs128/lr1e-3), 随机初始化冻结特征+线性头"
         "(--checkpoint none), 见 runlog/W8/run_e1_rand.py")
HP_FT = ("W8 E1: run_ft 默认超参(100ep/bs128/lr1e-4/fraction0.1), 随机初始化全参微调"
         "(省略 --checkpoint), 见 runlog/W8/run_e1_rand.py")


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def git_sha():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def row_done(evalkind, ds, seed):
    if not OUT.exists():
        return False
    with open(OUT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["ckpt"], r["seed"], r["eval"], r["downstream"]) == ("rand", str(seed), evalkind, ds):
                return True
    return False


def record(evalkind, ds, seed, auroc, auprc):
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "ckpt", "seed", "eval", "downstream", "auroc", "auprc",
                        "git_sha", "checkpoint_sha256", "protocol_id", "hparams_ref"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), "rand", seed, evalkind, ds,
                    f"{auroc:.4f}", f"{auprc:.4f}", git_sha(), "random-init-TFS",
                    "w8-rand", HP_LP if evalkind == "lp" else HP_FT])
    log(f"  + rand s{seed} {evalkind}/{ds}: {auroc:.4f}/{auprc:.4f}")


def run_lp(ds, nc, seed):
    tag = f"rand_{ds}_lp_seed{seed}"
    if row_done("lp", ds, seed) and (ROOT / PRED / tag / "y_prob.npy").exists():
        log(f"{tag}: 已有账+预测, 跳过")
        return True
    feat = ROOT / f"runlog/W8/feat/{tag}"
    with open_out(LOGD, f"{tag}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_lp.py", "--data-dir", f"data/{ds}", "--num-classes", str(nc),
             "--checkpoint", "none", "--feat-dir", str(feat),
             "--seed", str(seed), "--workers", "6", "--trc", "0",
             "--extended-metrics", "1",
             "--save-predictions", f"{PRED}/{tag}",
             "--protocol-id", "w8-rand"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((feat / "metrics.json").read_text(encoding="utf-8"))
        record("lp", ds, seed, m["auroc"], m["auprc"])
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e}; 见 runlog/W8/logs/{tag}.log 尾部)")
        return False


def run_ft(ds, nc, seed):
    tag = f"rand_{ds}_ft10_seed{seed}"
    if row_done("ft10", ds, seed) and (ROOT / PRED / tag / "y_prob.npy").exists():
        log(f"{tag}: 已有账+预测, 跳过")
        return True
    mdir = ROOT / f"runlog/W8/ft/{tag}"
    with open_out(LOGD, f"{tag}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_ft.py", "--data-dir", f"data/{ds}", "--num-classes", str(nc),
             "--fraction", "0.1",
             "--model-dir", str(mdir), "--workers", "4", "--epochs", "100",
             "--batch-size", "128", "--learning-rate", "0.0001",
             "--seed", str(seed), "--trc", "0",
             "--extended-metrics", "1",
             "--save-predictions", f"{PRED}/{tag}",
             "--protocol-id", "w8-rand"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((mdir / "metrics.json").read_text(encoding="utf-8"))
        record("ft10", ds, seed, m["auroc"], m["auprc"])
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e}; 见 runlog/W8/logs/{tag}.log 尾部)")
        return False


def main():
    LOGD.mkdir(parents=True, exist_ok=True)
    (ROOT / "runlog/W8/feat").mkdir(parents=True, exist_ok=True)
    (ROOT / "runlog/W8/predictions").mkdir(parents=True, exist_ok=True)
    fails = 0
    log("E1 阶段1: LP 三域×3seeds (sanity 解锁)")
    for ds, nc in DOMAINS:
        for s in SEEDS:
            log(f"rand {ds} s{s} lp: 启动")
            if not run_lp(ds, nc, s):
                fails += 1
    log("E1 阶段2: FT10 三域×3seeds")
    for ds, nc in DOMAINS:
        for s in SEEDS:
            log(f"rand {ds} s{s} ft10: 启动")
            if not run_ft(ds, nc, s):
                fails += 1
    log(f"E1 全部跑完, 失败 {fails} 跑" + ("(见 logs)" if fails else ""))


if __name__ == "__main__":
    main()
