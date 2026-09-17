#!/bin/bash
# 车道B(并行): crop_weak 3seed 确认(seed 2,4) — 首个正探针的入库判定
cd "/f/新实验/ECG_SSL_LFBT-main" || exit 1
SUM=runlog/S1/overnight_summary.md
PY="C:/Users/admin/.conda/envs/DL/python.exe"
export PYTHONUNBUFFERED=1
sleep 120  # 等车道A先占稳
echo "[$(date +%H:%M)] == 车道B启动: aug_crop_weak 3seed 确认 ==" >> "$SUM"
"$PY" m_screen.py --config runlog/M/matrix_batch3.yaml --confirm aug_crop_weak >> "$SUM" 2>&1
echo "[$(date +%H:%M)] 车道B: 确认轮结束" >> "$SUM"
