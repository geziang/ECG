# -*- coding: utf-8 -*-
"""W8 车道接力器: E1 收官后串行跑 E2 链 -> E4 链 (IO 车道独占, 2026-10-10 USB 盘实测教训)。

背景: F: 为 USB 桥接盘, 多进程并发随机读 npy 时吞吐塌陷(实测 9.6 files/s vs 单车道
W7 时代 ~300 files/s, 缓存暖后恢复)。PT 类重 IO 任务必须独占车道串行执行。

行为: 轮询 e1_driver.log 出现收尾行("E1 全部跑完")或 E1 驱动进程退出 -> 依次
subprocess 串行执行 run_e2_simph_chain.py / run_e4_trcaff_chain.py (各自幂等)。
日志: runlog/W8/logs/relay.log。
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
LOG = ROOT / "runlog/W8/logs/relay.log"
E1_LOG = ROOT / "runlog/W8/logs/e1_driver.log"
E1_DONE_MARK = "E1 全部跑完"


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def e1_driver_alive():
    # 判活走独立 ps1(避免 bash/ps 双层引号打穿; 2026-10-10 实测可靠)
    return subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(ROOT / "runlog/W8/check_e1_alive.ps1")],
        capture_output=True).returncode == 0


def main():
    log("接力器启动: 等待 E1 收官")
    while True:
        txt = E1_LOG.read_text(encoding="utf-8", errors="replace") if E1_LOG.exists() else ""
        if E1_DONE_MARK in txt:
            log("E1 收尾标记出现")
            break
        if not e1_driver_alive():
            log("E1 驱动进程已退出(无收尾标记, 按进程退出处理)")
            time.sleep(60)  # 收尾记账窗口
            break
        time.sleep(120)
    for chain in ("run_e2_simph_chain.py", "run_e4_trcaff_chain.py",
                  "run_e6_noiseiso_chain.py", "run_e3_tsr_chain.py",
                  "run_e5_e7_chain.py"):
        log(f"启动 {chain}")
        with open(LOG, "a", encoding="utf-8") as f:
            rc = subprocess.run([PY, "-u", f"runlog/W8/{chain}"],
                                cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT).returncode
        log(f"{chain} exit={rc}")
        if rc != 0:
            log(f"{chain} 失败, 停止接力(人工介入)")
            return
    log("接力器全部完成 (E2+E4)")


if __name__ == "__main__":
    main()
