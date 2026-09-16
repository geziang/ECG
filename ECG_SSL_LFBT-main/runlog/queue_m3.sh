#!/bin/bash
# 接力: 等 M 第二批结束 -> 第三批(增强扫描 + 投影维度)
cd "/f/新实验/ECG_SSL_LFBT-main" || exit 1
SUM=runlog/S1/overnight_summary.md
waited=0
while ! grep -q "M 第二批结束" "$SUM" 2>/dev/null; do
  sleep 180; waited=$((waited + 3))
  [ "$waited" -ge 720 ] && { echo "[$(date +%H:%M)] M3 等待超时(12h)退出" >> "$SUM"; exit 1; }
done
echo "[$(date +%H:%M)] == M 第三批启动: 增强扫描+投影维度 ==" >> "$SUM"
"C:/Users/admin/.conda/envs/DL/python.exe" m_screen.py --config runlog/M/matrix_batch3.yaml >> "$SUM" 2>&1
echo "[$(date +%H:%M)] M 第三批结束" >> "$SUM"
