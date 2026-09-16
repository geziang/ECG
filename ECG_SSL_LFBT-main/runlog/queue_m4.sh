#!/bin/bash
# 接力: 等 M 第三批结束 -> B0-fast 校准 -> 第四批(跨域迁移四件套)
cd "/f/新实验/ECG_SSL_LFBT-main" || exit 1
SUM=runlog/S1/overnight_summary.md
PY="C:/Users/admin/.conda/envs/DL/python.exe"
waited=0
while ! grep -q "M 第三批结束" "$SUM" 2>/dev/null; do
  sleep 180
  waited=$((waited + 3))
  if [ "$waited" -ge 900 ]; then
    echo "[$(date +%H:%M)] M4 等待超时(15h)退出" >> "$SUM"
    exit 1
  fi
done
echo "[$(date +%H:%M)] == B0-fast 校准跑启动(fast 路径 LP 应与慢路径锚点差<0.3pt) ==" >> "$SUM"
"$PY" run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 \
  --workers 6 --seed 0 --fast-backbone --checkpoint-dir checkpoint/M/b0fast_seed0 \
  > runlog/M/pt_b0fast_seed0.log 2>&1 \
  && "$PY" run_lp.py --data-dir data/ptbxl --checkpoint checkpoint/M/b0fast_seed0/encoder_group.pth \
     --num-classes 5 --feat-dir feat/M_b0fast_seed0 --seed 0 --workers 6 \
     > runlog/M/lp_b0fast_seed0.log 2>&1 \
  && grep -E "AUROC|AUPRC" runlog/M/lp_b0fast_seed0.log >> "$SUM"
echo "[$(date +%H:%M)] B0-fast 校准完成" >> "$SUM"
echo "[$(date +%H:%M)] == M 第四批启动: 跨域迁移(speed/asym/swap/cautious) ==" >> "$SUM"
"$PY" m_screen.py --config runlog/M/matrix_batch4.yaml >> "$SUM" 2>&1
echo "[$(date +%H:%M)] M 第四批结束" >> "$SUM"
