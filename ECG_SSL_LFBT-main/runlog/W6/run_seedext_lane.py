# -*- coding: utf-8 -*-
"""W6 Stage 5: 种子扩展 n=3→5 (seeds {1,3}) — 全链 PT+三域LP+PTB FT10。

用法(conda DL 解释器, 仓库根 cwd, 分离进程):
    python runlog/W6/run_seedext_lane.py L1   # 重车道: b0/c1 × seeds{1,3} = 4 条全链
    python runlog/W6/run_seedext_lane.py L2   # 机制车道: c2 × seeds{1,3} = 2 条全链(排程在 LCM/双视角之后)

任务书定位: 纯堆量回应"全部结论建立在 3 种子上"; 预注册 3-seed 门与 W2 冻结结论不动,
5-seed 只作稳定性扩展证据(符号一致性 5/5、置换地板 0.0625); 主表重算归办公机, 本脚本只出原始账本。

协议(与 W2 confirm_matrix.py 逐字同源, 禁改):
  PT: b0 -> data/pt_pretrain 200ep trc=0;  c1/c2 -> data/pt_pretrain_nfh 100ep, c2 trc=1
      (batch 128 / workers 4 / --resume)
  LP: 三域 {ptbxl(5类), cpsc(9类,修正分区), chapman(4类)} 冻结+线性, W2 模板
  FT10: PTB-XL fraction 0.1, W2 FT10 模板
  落盘三参数(全链所有评估): --extended-metrics 1 --save-predictions
      runlog/W6/predictions/seedext_{tag} --protocol-id w6-seedext
车道纪律同 run_ft2040_lane.py(车道隔离 csv: seedext_results_L{1,2}.csv; 幂等;
单跑失败记 FAIL 继续下一项; PT 以 encoder_group.pth+config.json 判完成)。
新 checkpoint 目录: checkpoint/w6_seedext/{kind}_seed{s}(不入 git)。
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
TRC = {"c2": "1", "c1": "0", "b0": "0"}
PT_SPEC = {"b0": ("data/pt_pretrain", "200"), "c1": ("data/pt_pretrain_nfh", "100"),
           "c2": ("data/pt_pretrain_nfh", "100")}
DOMAINS = [("ptbxl", 5), ("cpsc", 9), ("chapman", 4)]
LANES = {"L1": [("b0", 1), ("b0", 3), ("c1", 1), ("c1", 3)],
         "L2": [("c2", 1), ("c2", 3)]}


def ckdir(kind, seed):
    return ROOT / f"checkpoint/w6_seedext/{kind}_seed{seed}"


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
    return ROOT / f"runlog/W6/seedext_results_{lane}.csv"


def row_done(lane, ckpt, seed, ev, ds):
    p = out_csv(lane)
    if not p.exists():
        return False
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["ckpt"], r["seed"], r["eval"], r["downstream"]) == (ckpt, str(seed), ev, ds):
                return True
    return False


def record(lane, ckpt, seed, ev, ds, auroc, auprc, cksha):
    p = out_csv(lane)
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "ckpt", "seed", "eval", "downstream", "auroc", "auprc",
                        "git_sha", "checkpoint_sha256", "protocol_id", "hparams_ref"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), ckpt, seed, ev, ds,
                    f"{auroc:.4f}", f"{auprc:.4f}", git_sha(), cksha,
                    "w6-seedext", "W6 Stage5: W2 全链冻结协议 seed{1,3} 扩展, 见 runlog/W6/run_seedext_lane.py"])
    log(f"  + L{lane} {ckpt} s{seed} {ev}/{ds}: {auroc:.4f}/{auprc:.4f}")


def run_pt(kind, seed):
    ckdir_ = ckdir(kind, seed)
    tag = f"seedext PT {kind} s{seed}"
    if (ckdir_ / "encoder_group.pth").exists() and (ckdir_ / "config.json").exists():
        log(f"{tag}: 已完成, 跳过")
        return True
    ckdir_.mkdir(parents=True, exist_ok=True)
    data, epochs = PT_SPEC[kind]
    with open_out(LOGD, f"seedext_pt_{kind}_seed{seed}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_pt.py", "--data-dir", data, "--epochs", epochs,
             "--batch-size", "128", "--workers", "4", "--seed", str(seed),
             "--trc", TRC[kind], "--resume", "--checkpoint-dir", str(ckdir_)],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    ok = rc == 0 and (ckdir_ / "encoder_group.pth").exists()
    log(f"{tag}: exit={rc} {'OK' if ok else 'FAIL(见日志)'}")
    return ok


def run_lp(kind, seed, ds, nc):
    tag = f"seedext_{kind}_{ds}_lp_seed{seed}"
    if row_done(lane_holder[0], kind, seed, "lp", ds):
        log(f"{tag}: 已有账, 跳过")
        return True
    ckpt = ckdir(kind, seed) / "encoder_group.pth"
    if not ckpt.exists():
        log(f"{tag}: checkpoint 缺失, 跳过(车道继续)")
        return False
    feat = ROOT / f"runlog/W6/feat/seedext_{kind}_{ds}_seed{seed}"
    with open_out(LOGD, f"{tag}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_lp.py", "--data-dir", f"data/{ds}", "--num-classes", str(nc),
             "--checkpoint", str(ckpt), "--feat-dir", str(feat),
             "--seed", str(seed), "--workers", "6", "--trc", TRC[kind],
             "--extended-metrics", "1",
             "--save-predictions", f"runlog/W6/predictions/{tag}",
             "--protocol-id", "w6-seedext"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((feat / "metrics.json").read_text(encoding="utf-8"))
        record(lane_holder[0], kind, seed, "lp", ds, m["auroc"], m["auprc"], sha256_of(ckpt))
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e})")
        return False


def run_ft10(kind, seed):
    tag = f"seedext_{kind}_ptbxl_ft10_seed{seed}"
    if row_done(lane_holder[0], kind, seed, "ft10", "ptbxl"):
        log(f"{tag}: 已有账, 跳过")
        return True
    ckpt = ckdir(kind, seed) / "encoder_group.pth"
    if not ckpt.exists():
        log(f"{tag}: checkpoint 缺失, 跳过(车道继续)")
        return False
    mdir = ROOT / f"runlog/W6/ft/{tag}"
    with open_out(LOGD, f"{tag}.log", encoding="utf-8") as f:
        rc = subprocess.run(
            [PY, "-u", "run_ft.py", "--data-dir", "data/ptbxl", "--num-classes", "5",
             "--fraction", "0.1", "--checkpoint", str(ckpt),
             "--model-dir", str(mdir), "--workers", "4", "--epochs", "100",
             "--batch-size", "128", "--learning-rate", "0.0001",
             "--seed", str(seed), "--trc", TRC[kind],
             "--extended-metrics", "1",
             "--save-predictions", f"runlog/W6/predictions/{tag}",
             "--protocol-id", "w6-seedext"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    try:
        m = json.loads((mdir / "metrics.json").read_text(encoding="utf-8"))
        record(lane_holder[0], kind, seed, "ft10", "ptbxl", m["auroc"], m["auprc"], sha256_of(ckpt))
        return True
    except Exception as e:
        log(f"{tag}: FAIL rc={rc} ({type(e).__name__}: {e})")
        return False


lane_holder = [""]

def main():
    lane = sys.argv[1] if len(sys.argv) > 1 else ""
    if lane not in LANES:
        sys.exit("用法: python runlog/W6/run_seedext_lane.py {L1|L2}")
    lane_holder[0] = lane
    LOGD.mkdir(parents=True, exist_ok=True)
    log(f"W6 Stage5 车道 {lane} 启动, 解释器={PY}, 链表={LANES[lane]}")
    for kind, seed in LANES[lane]:
        if not run_pt(kind, seed):
            log(f"{kind} s{seed}: PT 失败, 跳过该链(车道继续)")
            continue
        for ds, nc in DOMAINS:
            run_lp(kind, seed, ds, nc)
        run_ft10(kind, seed)
    log(f"车道 {lane} 结束")


if __name__ == "__main__":
    main()
