#!/bin/bash
# 车道B接力: 等确认轮结束 -> crop_085 探针 -> FT10 评估(B0锚点+crop_weak) -> batch6(若就绪)
cd "/f/新实验/ECG_SSL_LFBT-main" || exit 1
SUM=runlog/S1/overnight_summary.md
PY="C:/Users/admin/.conda/envs/DL/python.exe"
export PYTHONUNBUFFERED=1
waited=0
while ! grep -q "车道B: 确认轮结束" "$SUM" 2>/dev/null; do
  sleep 120; waited=$((waited + 2))
  if [ "$waited" -ge 720 ]; then
    echo "[$(date +%H:%M)] B2 等待超时(12h)退出" >> "$SUM"; exit 1
  fi
done
echo "[$(date +%H:%M)] == 车道B接力: crop_085 探针 ==" >> "$SUM"
"$PY" m_screen.py --config runlog/M/matrix_batchB2.yaml >> "$SUM" 2>&1
echo "[$(date +%H:%M)] 车道B: crop_085 结束" >> "$SUM"
# FT10 评估(轻量): B0 锚点 + crop_weak seed0
run_ft10() {  # $1=name $2=ckpt
  [ -f "$2/encoder_group.pth" ] || return 0
  echo "[$(date +%H:%M)] FT10: $1" >> "$SUM"
  "$PY" run_ft.py --data-dir data/ptbxl --checkpoint "$2/encoder_group.pth" \
    --num-classes 5 --fraction 0.1 --model-dir "ft_models/M_$1_ft10" \
    --workers 6 --seed 0 > "runlog/M/ft10_$1.log" 2>&1
  grep -E "AUROC|AUPRC" "runlog/M/ft10_$1.log" | tail -2 >> "$SUM"
}
run_ft10 b0_anchor checkpoint/ptxl_gamma08
run_ft10 crop_weak_s0 checkpoint/M/aug_crop_weak_seed0
echo "[$(date +%H:%M)] 车道B: FT10 评估结束" >> "$SUM"
# batch6 若已就绪(下午写完 b6 模块后生成)
if [ -f runlog/M/matrix_batch6.yaml ]; then
  echo "[$(date +%H:%M)] == 车道B: batch6(新模块) ==" >> "$SUM"
  "$PY" m_screen.py --config runlog/M/matrix_batch6.yaml >> "$SUM" 2>&1
  echo "[$(date +%H:%M)] 车道B: batch6 结束" >> "$SUM"
fi
echo "[$(date +%H:%M)] 车道B接力全部结束" >> "$SUM"
