# -*- coding: utf-8 -*-
"""换挡看门狗 (2026-09-17 21:35): 老调度车道在跑 arb3_base/d7_rec01, 完成后必须换新调度
(修复 psfull 饿死)。本脚本等到两个任务的 csv 行出现, 即刻停旧 runner、清 claim、起新车道。
若竞态窗口内旧 runner 抢跑了一个新探针(分钟级损失), 一并停掉并清其 claim。幂等, 可重复启动。
"""
import csv
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESD = ROOT / "runlog" / "M" / "matrix_results.csv"
CLAIMD = ROOT / "runlog" / "M" / ".pipeline_claims"
PY = r"C:/Users/admin/.conda/envs/DL/python.exe"
WAIT_TAGS = {("arb3_base", "0"), ("d7_rec01", "0")}


def done_pairs():
    with RESD.open(encoding="utf-8") as f:
        return {(r["name"], r["seed"]) for r in csv.DictReader(f)}


def pids_matching(pat):
    ps = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
          f"Where-Object {{$_.CommandLine -match '{pat}'}} | "
          "Select-Object -ExpandProperty ProcessId")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                         capture_output=True, text=True, timeout=90).stdout
    return [int(l) for l in out.split() if l.strip().isdigit()]


def start_lane(lane):
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    f"Start-Process -FilePath '{PY}' -ArgumentList 'pipeline_runner.py',"
                    f"'--lane','{lane}' -WorkingDirectory '{ROOT}' -WindowStyle Hidden "
                    f"-RedirectStandardOutput '{ROOT}\\runlog\\M\\lane{lane}.out.log' "
                    f"-RedirectStandardError '{ROOT}\\runlog\\M\\lane{lane}.err.log'"],
                   timeout=60)


def main():
    print("[watchdog] 等待 arb3_base/d7_rec01 完成行…", flush=True)
    while not WAIT_TAGS <= done_pairs():
        time.sleep(10)
    print(f"[watchdog] {time.strftime('%H:%M')} 两任务完成, 换挡开始", flush=True)
    # 停旧 runner 与其可能抢跑的子任务(孤儿 arb3/d7 的 pt 已结束, 不受影响)
    for pid in pids_matching("pipeline_runner"):
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"Stop-Process -Id {pid} -Force"], timeout=30)
    time.sleep(2)
    for pid in pids_matching("run_pt"):
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"Stop-Process -Id {pid} -Force"], timeout=30)
    for c in CLAIMD.glob("*.claim"):
        c.unlink()
    print("[watchdog] 旧调度已停, 启动新调度车道 C2/D2", flush=True)
    start_lane("C2")
    time.sleep(30)
    start_lane("D2")
    print("[watchdog] 完成, 退出", flush=True)


if __name__ == "__main__":
    main()
