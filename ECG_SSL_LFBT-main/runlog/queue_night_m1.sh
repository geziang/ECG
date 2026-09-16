#!/bin/bash
# 夜间接力: 等 D9 队列结束 -> M 矩阵第一批粗探 (D1L / D1L-NEG / N4 EMA)
cd "/f/新实验/ECG_SSL_LFBT-main" || exit 1
SUM=runlog/S1/overnight_summary.md
waited=0
while ! grep -q "下午队列结束" "$SUM" 2>/dev/null; do
  sleep 120; waited=$((waited + 2))
  [ "$waited" -ge 480 ] && { echo "[$(date +%H:%M)] M1 等待超时(8h)退出" >> "$SUM"; exit 1; }
done
echo "[$(date +%H:%M)] == M 第一批粗探启动 (D1L / D1L-NEG / N4) ==" >> "$SUM"
"C:/Users/admin/.conda/envs/DL/python.exe" m_screen.py --config runlog/M/matrix_batch1.yaml >> "$SUM" 2>&1
echo "[$(date +%H:%M)] M 第一批结束" >> "$SUM"
