# -*- coding: utf-8 -*-
"""W7 种子补做接力（用户 10-07 指令: 单种子正向补多种子, 队列 Q8d）。

序列(单车道串行, 幂等):
  1. 等 E5 seed0 链出 e5_summary.md(最多 4h, 超时不阻塞 E2c, E5 排障归值班轮);
  2. E2c seed2 → E2c seed4 (无条件, b0trc 200ep PT + ptbxl/cpsc LP);
  3. 若 E5 seed0 判定含"✅过门" → E5 seed2 → E5 seed4 (条件补做);
     未过门/无判定文件 → 按 Q7 预注册关线跳过。
日志: logs/seedrelay_head.log(本脚本) / e2c_chain_s{2,4}.log / e5_chain_s{2,4}.log。
重启: 直接重跑本脚本(各链 row_done 幂等跳过已入账格)。
"""
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = "C:/Users/admin/.conda/envs/DL/python.exe"
OUTD = ROOT / "runlog/W7"
LOGD = OUTD / "logs"
E2C = "runlog/W7/run_e2c_chain.py"
E5 = "runlog/W7/run_e5_noiseaug_chain.py"
MAX_WAIT_S = 4 * 3600


def log(m):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}", flush=True)


def run(script, *args, logfile):
    cmd = [PY, "-u", script, *args]
    with open(LOGD / logfile, "a", encoding="utf-8") as f:
        f.write(f"\n==== {time.strftime('%F %T')} {' '.join(cmd)}\n")
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    log(f"{Path(script).name} {' '.join(args)}: exit={rc}")
    return rc


if __name__ == "__main__":
    log("seedrelay 启动: 等待 E5 seed0 判定(e5_summary.md)...")
    waited = 0
    while not (OUTD / "e5_summary.md").exists() and waited < MAX_WAIT_S:
        time.sleep(300)
        waited += 300
    log(f"E5 seed0 判定文件{'已出' if waited < MAX_WAIT_S else '等待超时(4h)'}, 开始 E2c 补种子")
    run(E2C, "--seed", "2", logfile="e2c_chain_s2.log")
    run(E2C, "--seed", "4", logfile="e2c_chain_s4.log")
    summ = OUTD / "e5_summary.md"
    txt = summ.read_text(encoding="utf-8") if summ.exists() else ""
    gate_pass = "✅过门" in txt
    weak_pos = any("= +" in l for l in txt.splitlines() if "ΔAUROC 均值" in l)
    if gate_pass or weak_pos:
        why = "过门" if gate_pass else "弱正(噪声格均值>0; 用户 10-07 指令: 有涨点即补种子, 微弱正收益也是工作量)"
        log(f"E5 seed0 {why} → 补 E5 seed2/4(参照同种子)")
        run(E5, "--seed", "2", logfile="e5_chain_s2.log")
        run(E5, "--seed", "4", logfile="e5_chain_s4.log")
    else:
        verdict = txt.strip().splitlines()[-1] if txt else "(无判定文件)"
        log(f"E5 seed0 无任何涨点方向({verdict}) → 不补种子(全负无从升格)")
    log("SEEDRELAY_END — E2c 三种子判定读数归值班轮/主会话(门见队列 Q8d)")
