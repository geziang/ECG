#!/bin/bash
# 接力队列: 等夜间队列结束 -> B0 原版 seed2 复跑(为粗筛提供同 seed 配对基准)
cd "/f/新实验/ECG_SSL_LFBT-main" || exit 1
PY="C:/Users/admin/.conda/envs/DL/python.exe"
export PYTHONUNBUFFERED=1
SUM=runlog/S1/overnight_summary.md

# 等夜间队列写完结束标记(最多等到 09:00, 防呆)
while ! grep -q "夜间队列全部结束" "$SUM" 2>/dev/null; do
  h=$(date +%H%M)
  [ "$h" -ge 0900 ] && { echo "[$(date +%H:%M)] 超过 09:00 放弃接力" >> "$SUM"; exit 1; }
  sleep 120
done

echo "[$(date +%H:%M)] == 接力: B0 seed2 复跑开始 ==" >> "$SUM"
"$PY" run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 \
  --workers 6 --seed 2 --checkpoint-dir checkpoint/s1_b0_seed2 \
  > runlog/S1/pt_b0_seed2.log 2>&1 \
  && "$PY" run_lp.py --data-dir data/ptbxl \
       --checkpoint checkpoint/s1_b0_seed2/encoder_group.pth \
       --num-classes 5 --feat-dir feat/s1_b0_seed2 --seed 0 --workers 6 \
       > runlog/S1/lp_b0_seed2.log 2>&1 \
  && grep -E "AUROC|AUPRC" runlog/S1/lp_b0_seed2.log >> "$SUM"
echo "[$(date +%H:%M)] 接力队列结束(预计 ~08:30)" >> "$SUM"
