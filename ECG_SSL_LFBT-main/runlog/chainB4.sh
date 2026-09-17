#!/bin/bash
# 车道B重建: FT10重跑 -> batch6(m_screen断点续跑) -> D1L×AR-B3
cd "/f/新实验/ECG_SSL_LFBT-main" || exit 1
SUM=runlog/S1/overnight_summary.md
PY="C:/Users/admin/.conda/envs/DL/python.exe"
export PYTHONUNBUFFERED=1
sleep 90
echo "[$(date +%H:%M)] == 车道B4 重建启动: FT10 重跑 ==" >> "$SUM"
run_ft10() {
  [ -f "$2/encoder_group.pth" ] || return 0
  echo "[$(date +%H:%M)] FT10: $1" >> "$SUM"
  "$PY" run_ft.py --data-dir data/ptbxl --checkpoint "$2/encoder_group.pth" \
    --num-classes 5 --fraction 0.1 --model-dir "ft_models/M_$1_ft10" \
    --workers 6 --seed 0 > "runlog/M/ft10_$1.log" 2>&1
  grep -E "AUROC|AUPRC" "runlog/M/ft10_$1.log" | tail -2 >> "$SUM"
}
run_ft10 b0_anchor checkpoint/ptxl_gamma08
run_ft10 crop_weak_s0 checkpoint/M/aug_crop_weak_seed0
echo "[$(date +%H:%M)] 车道B4: FT10 结束 -> batch6 ==" >> "$SUM"
"$PY" m_screen.py --config runlog/M/matrix_batch6.yaml >> "$SUM" 2>&1
echo "[$(date +%H:%M)] 车道B4: batch6 结束" >> "$SUM"
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
echo "[$(date +%H:%M)] 车道B4 全部结束" >> "$SUM"
