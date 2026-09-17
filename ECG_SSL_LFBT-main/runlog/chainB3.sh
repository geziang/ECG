#!/bin/bash
# 车道B二段接力: 等 B2 结束 -> batch6 四模块 -> D1L×AR-B3(先验×鲁棒基座判决)
cd "/f/新实验/ECG_SSL_LFBT-main" || exit 1
SUM=runlog/S1/overnight_summary.md
PY="C:/Users/admin/.conda/envs/DL/python.exe"
export PYTHONUNBUFFERED=1
waited=0
while ! grep -q "车道B接力全部结束" "$SUM" 2>/dev/null; do
  sleep 120; waited=$((waited + 2))
  if [ "$waited" -ge 900 ]; then
    echo "[$(date +%H:%M)] B3 等待超时(15h)退出" >> "$SUM"; exit 1
  fi
done
echo "[$(date +%H:%M)] == 车道B二段: batch6(D7/N3/共模) ==" >> "$SUM"
"$PY" m_screen.py --config runlog/M/matrix_batch6.yaml >> "$SUM" 2>&1
echo "[$(date +%H:%M)] 车道B: batch6 结束" >> "$SUM"
# D1L × AR-B3(需要 AR-B3 基座 checkpoint 已由车道A产出)
echo "[$(date +%H:%M)] == D1L×AR-B3 预训练 ==" >> "$SUM"
"$PY" run_pt_ar.py --data-dir data/pt_pretrain --checkpoint-dir checkpoint/M/d1larb3_seed0 \
  --variant B0 --fusion mean --fusion-bt-weight 0.2 --lead-mask-prob 0.5 \
  --d1l 0.5,0.2 --epochs 200 --batch-size 128 --workers 4 --seed 0 \
  > runlog/M/pt_d1larb3_seed0.log 2>&1 \
  && echo "[$(date +%H:%M)] D1L×AR-B3 预训练完成" >> "$SUM" \
  || echo "[$(date +%H:%M)] D1L×AR-B3 预训练失败" >> "$SUM"
if [ -f checkpoint/M/d1larb3_seed0/encoder_group.pth ]; then
  "$PY" run_e006_downstream.py --data-dir data/ptbxl \
    --checkpoint checkpoint/M/d1larb3_seed0/encoder_group.pth \
    --output-dir runlog/M/lp_d1larb3_seed0 --task lp --base-variant ar_b3 \
    --eval-split test --epochs 100 --workers 6 > runlog/M/lp_d1larb3_seed0.log 2>&1
  grep -iE "auroc|auprc" runlog/M/lp_d1larb3_seed0/*.json runlog/M/lp_d1larb3_seed0.log 2>/dev/null | head -4 >> "$SUM"
fi
echo "[$(date +%H:%M)] 车道B二段全部结束" >> "$SUM"
