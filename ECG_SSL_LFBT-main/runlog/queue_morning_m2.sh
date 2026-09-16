#!/bin/bash
# 早间接力: 等 M 第一批结束 -> 第二批(B0 超参邻域重调 gamma/lambd)
cd "/f/新实验/ECG_SSL_LFBT-main" || exit 1
SUM=runlog/S1/overnight_summary.md
waited=0
while ! grep -q "M 第一批结束" "$SUM" 2>/dev/null; do
  sleep 180; waited=$((waited + 3))
  [ "$waited" -ge 600 ] && { echo "[$(date +%H:%M)] M2 等待超时(10h)退出" >> "$SUM"; exit 1; }
done
echo "[$(date +%H:%M)] == M 第二批启动: B0 超参邻域重调(gamma 0.7/0.9, lambd 0.003/0.01) ==" >> "$SUM"
"C:/Users/admin/.conda/envs/DL/python.exe" m_screen.py --config runlog/M/matrix_batch2.yaml >> "$SUM" 2>&1
echo "[$(date +%H:%M)] M 第二批结束" >> "$SUM"
