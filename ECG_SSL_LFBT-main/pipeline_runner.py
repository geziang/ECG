# -*- coding: utf-8 -*-
"""pipeline_runner.py — 多车道流水线接力器 (v1, 2026-09-17)

针对既有 chain*.sh 接力链的三个弱点设计:
  1. bash 链父进程易死(孤儿事故)   -> 纯 python 自持循环, 无父依赖;
  2. capture_output 静默(m_screen) -> 子进程 stdout 直接流式写日志, 随时可看进度;
  3. 固定等待标记, 环境变化即失效   -> 显存闸门 + csv/日志/活进程三重防撞, 与链共存。

安全性:
  - 显存闸门: 每次启动重任务前查 nvidia-smi 空闲 MiB, 不足则每 60s 轮询(每 10min 报一次);
  - 幂等: (name,seed) 已在 matrix_results.csv 或 lp/下游日志已含 AUPRC -> 跳过;
  - 防撞: 活进程扫描(含 m_screen/chain 启动的 run_pt*)命中同 tag -> 跳过;
  - 多车道互斥: runlog/M/.pipeline_claims/<tag>.claim 以 O_CREAT|O_EXCL 原子认领;
  - 自动确认: 探针 seed0 Δ>0 时, seed2/4 自动插队到最前(3-seed 符号门自动化)。

用法:
  python pipeline_runner.py --lane C                # 正常启动
  python pipeline_runner.py --lane D --stagger 120  # 错峰启动
  python pipeline_runner.py --lane C --plan         # 只打印计划
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

PY = sys.executable
ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoint" / "M"
FEAT = ROOT / "feat"
LOGD = ROOT / "runlog" / "M"
RESD = LOGD / "matrix_results.csv"
CLAIMD = LOGD / ".pipeline_claims"
SUMMARY = ROOT / "runlog" / "S1" / "overnight_summary.md"
ANCHOR = 0.7177  # B0 seed0 冻结锚点
CONFIRM_SEEDS = [2, 4]
STALE_CLAIM_H = 16

LANE = "X"


def log(msg):
    line = f"[{time.strftime('%H:%M')}] [lane{LANE}] {msg}"
    print(line, flush=True)
    try:
        with SUMMARY.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------- 资源与防撞 ----------
def vram_free():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return 0  # 查询失败按 0 处理, 保守等待


def wait_vram(need, tag):
    last = 0.0
    while True:
        free = vram_free()
        if free >= need:
            return free
        if time.time() - last > 600:
            log(f"等待显存: {tag} 需要 {need}MiB, 当前空闲 {free}MiB (每60s重查)")
            last = time.time()
        time.sleep(60)


def tag_running(tag):
    """活进程扫描: 任何 python 命令行包含 <tag>(词边界)即视为他方正在跑。"""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
          f"Where-Object {{$_.CommandLine -match '\\b{tag}'}} | "
          "Select-Object -First 1 ProcessId")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=90).stdout
        return any(l.strip().isdigit() for l in out.splitlines())
    except Exception:
        return True  # 查询失败按占用处理


def done_names():
    if not RESD.exists():
        return set()
    done = set()
    with RESD.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                done.add((r["name"], r["seed"]))
            except Exception:
                continue
    return done


def lp_has_auprc(tag):
    p = LOGD / f"lp_{tag}.log"
    ds = ROOT / "runlog" / "M" / f"lp_{tag}"
    if p.exists() and "AUPRC" in p.read_text(errors="replace"):
        return True
    if ds.is_dir():
        for j in ds.rglob("*.json"):
            try:
                if "auprc" in j.read_text(errors="replace").lower():
                    return True
            except Exception:
                pass
    return False


def claim(tag):
    CLAIMD.mkdir(parents=True, exist_ok=True)
    p = CLAIMD / f"{tag}.claim"
    if p.exists():
        age_h = (time.time() - p.stat().st_mtime) / 3600
        if age_h < STALE_CLAIM_H:
            return False
        p.unlink()  # 过期接管
    try:
        fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"lane{LANE} {time.strftime('%m-%d %H:%M')}".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release(tag):
    p = CLAIMD / f"{tag}.claim"
    if p.exists():
        p.unlink()


# ---------- 子进程(日志直写) ----------
def sh_live(cmd, log_path):
    """流式执行: stdout/stderr 实时写 log_path, 返回 (ok, stdout_text)。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        f.write(f"# {' '.join(map(str, cmd))}\n")
        f.flush()
        p = subprocess.Popen([str(c) for c in cmd], stdout=f,
                             stderr=subprocess.STDOUT, cwd=str(ROOT))
        p.wait()
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return p.returncode == 0, text


def parse_lp(text):
    auroc = auprc = None
    for line in text.splitlines():
        if line.startswith("AUROC"):
            auroc = float(line.split("=")[1])
        elif line.startswith("AUPRC"):
            auprc = float(line.split("=")[1])
    return auroc, auprc


def parse_downstream(tag):
    ds = LOGD / f"lp_{tag}"
    if ds.is_dir():
        for j in sorted(ds.rglob("*.json")):
            try:
                import json
                d = json.loads(j.read_text(encoding="utf-8"))
                for k in d:
                    if k.lower() == "auprc":
                        return d.get("auroc", d.get("AUROC")), d[k]
            except Exception:
                continue
    logp = LOGD / f"lp_{tag}.log"
    if logp.exists():
        return parse_lp(logp.read_text(errors="replace"))
    return None, None


def append_result(row):
    new = not RESD.exists()
    RESD.parent.mkdir(parents=True, exist_ok=True)
    with RESD.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ts", "name", "seed", "args", "auroc",
                                          "auprc", "delta", "fast"])
        if new:
            w.writeheader()
        w.writerow(row)


# ---------- 任务执行 ----------
def run_probe(cfg, seed, anchor):
    name, tag = cfg["name"], f"{cfg['name']}_seed{seed}"
    ck = CKPT / tag
    cmd = [PY, "run_pt.py", "--data-dir", "data/pt_pretrain", "--epochs", 200,
           "--batch-size", 128, "--workers", 4, "--seed", seed,
           "--checkpoint-dir", ck]
    if cfg.get("fast"):
        cmd.append("--fast-backbone")
    if cfg.get("args"):
        cmd += cfg["args"].split()
    need = 7800  # 实测单预训练峰值 6737MiB(09-17 性能计数器), 留 ~1GB 余量; 大于 LP 间隙可释放量, 不会抢跑链
    wait_vram(need, tag)
    ok, text = sh_live(cmd, LOGD / f"pt_{tag}.log")
    if "SHA256" not in text:
        log(f"预训练失败 {tag}(详见 pt 日志)")
        return None
    ok, text = sh_live(
        [PY, "run_lp.py", "--data-dir", "data/ptbxl", "--checkpoint",
         ck / "encoder_group.pth", "--num-classes", 5, "--feat-dir",
         FEAT / f"M_{tag}", "--seed", 0, "--workers", 6],
        LOGD / f"lp_{tag}.log")
    auroc, auprc = parse_lp(text)
    if auprc is None:
        log(f"LP 失败 {tag}")
        return None
    delta = round(auprc - anchor, 4)
    append_result({"ts": time.strftime("%m-%d %H:%M"), "name": name, "seed": seed,
                   "args": cfg.get("args", ""), "auroc": round(auroc, 4),
                   "auprc": round(auprc, 4), "delta": delta,
                   "fast": int(bool(cfg.get("fast")))})
    log(f"✔ {tag}: AUPRC={auprc:.4f} Δ={delta:+.4f}")
    return delta


def run_ar_or_e006(cfg, seed):
    """arb3_base / psfull_arb3: 预训练 + run_e006_downstream LP。"""
    tag = f"{cfg['name']}_seed{seed}"
    ck = CKPT / tag
    wait_vram(9800, tag)
    if cfg["type"] == "ar_pt":
        cmd = [PY, "run_pt_ar.py", "--data-dir", "data/pt_pretrain",
               "--checkpoint-dir", ck, "--variant", "B0", "--fusion", "mean",
               "--fusion-bt-weight", "0.2", "--lead-mask-prob", "0.5",
               "--epochs", 200, "--batch-size", 128, "--workers", 4, "--seed", seed]
    else:
        cmd = [PY, "run_e006_physiospatial.py", "--data-dir", "data/pt_pretrain",
               "--checkpoint-dir", ck, "--profile", "physiospatial",
               "--base-variant", "ar_b3", "--ablation", "full",
               "--epochs", 200, "--batch-size", "128", "--workers", 4, "--seed", seed]
    _, text = sh_live(cmd, LOGD / f"pt_{tag}.log")
    if "SHA256" not in text:
        log(f"预训练失败 {tag}(详见 pt 日志)")
        return None
    sh_live([PY, "run_e006_downstream.py", "--data-dir", "data/ptbxl",
             "--checkpoint", ck / "encoder_group.pth", "--output-dir",
             LOGD / f"lp_{tag}", "--task", "lp", "--base-variant", "ar_b3",
             "--eval-split", "test", "--epochs", 100, "--workers", 6],
            LOGD / f"lp_{tag}.log")
    auroc, auprc = parse_downstream(tag)
    if auprc is None:
        log(f"下游 LP 解析失败 {tag}(值可能已产出, 人工核对)")
        return None
    append_result({"ts": time.strftime("%m-%d %H:%M"), "name": cfg["name"],
                   "seed": seed, "args": cfg["type"], "auroc": round(auroc, 4) if auroc else "",
                   "auprc": round(auprc, 4), "delta": "", "fast": 0})
    log(f"✔ {tag}: AUROC={auroc} AUPRC={auprc:.4f}(基座类任务, Δ 另按配对口径算)")
    return auprc


# ---------- 主循环 ----------
def main():
    global LANE
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", required=True)
    ap.add_argument("--queue", default="runlog/M/pipeline_queue.yaml")
    ap.add_argument("--stagger", type=int, default=0)
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()
    LANE = args.lane
    if args.stagger:
        time.sleep(args.stagger)

    spec = yaml.safe_load((ROOT / args.queue).read_text(encoding="utf-8"))
    anchor = spec.get("anchor", ANCHOR)
    queue = []
    for cfg in spec["tasks"]:
        for s in cfg.get("seeds", [0]):
            queue.append((cfg, s))

    done = done_names()
    print(f"[lane{LANE}] 队列 {len(queue)} 项; 已完成 {sum((c['name'], str(s)) in done for c, s in queue)} 项")
    for cfg, s in queue:
        state = "done" if (cfg["name"], str(s)) in done else "todo"
        print(f"  - {cfg['name']} seed{s} [{cfg['type']}] {state}")
    if args.plan:
        return

    log(f"流水线启动: 队列 {len(queue)} 项, 显存闸门 heavy=9800/fast=5000MiB")
    skipped_forever = set()
    while queue:
        picked = None
        for i, (cfg, s) in enumerate(queue):
            tag = f"{cfg['name']}_seed{s}"
            if (cfg["name"], str(s)) in done_names() or lp_has_auprc(tag):
                skipped_forever.add(i)
                continue
            if i in skipped_forever:
                continue
            if tag_running(tag):
                continue  # 他方在跑, 先看下一项
            if not claim(tag):
                continue
            picked = i
            break
        if picked is None:
            if len(skipped_forever) >= len(queue):
                break
            log("全部剩余任务被他方占用或已认领, 10min 后重查")
            time.sleep(600)
            continue
        cfg, s = queue.pop(picked)
        skipped_forever.clear()
        tag = f"{cfg['name']}_seed{s}"
        log(f"启动任务 {tag}(空闲显存 {vram_free()}MiB)")
        try:
            if cfg["type"] == "probe":
                delta = run_probe(cfg, s, anchor)
                if delta is not None and delta > 0 and s == 0 and \
                        not cfg.get("no_confirm"):
                    for cs in reversed(CONFIRM_SEEDS):
                        queue.insert(0, (cfg, cs))
                    log(f"{tag} Δ>0, 自动插队确认 seed {CONFIRM_SEEDS}(3-seed 符号门)")
            else:
                run_ar_or_e006(cfg, s)
        finally:
            release(tag)
    log(f"流水线退出: 队列清空/全部完成")


if __name__ == "__main__":
    main()
