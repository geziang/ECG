#!/bin/bash
# 车道A重建(127风暴后): batch3c剩余(proj) -> B0fast -> b4 -> AR-B3 -> P1/S1
cd "/f/新实验/ECG_SSL_LFBT-main" || exit 1
SUM=runlog/S1/overnight_summary.md
PY="C:/Users/admin/.conda/envs/DL/python.exe"
export PYTHONUNBUFFERED=1
echo "[$(date +%H:%M)] == 车道A2 重建启动 ==" >> "$SUM"
"$PY" m_screen.py --config runlog/M/matrix_batch3c.yaml >> "$SUM" 2>&1
echo "[$(date +%H:%M)] 车道A2: batch3c 补完" >> "$SUM"
"$PY" run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 \
  --workers 4 --seed 0 --fast-backbone --checkpoint-dir checkpoint/M/b0fast_seed0 \
  > runlog/M/pt_b0fast_seed0.log 2>&1 \
  && "$PY" run_lp.py --data-dir data/ptbxl --checkpoint checkpoint/M/b0fast_seed0/encoder_group.pth \
     --num-classes 5 --feat-dir feat/M_b0fast_seed0 --seed 0 --workers 6 \
     > runlog/M/lp_b0fast_seed0.log 2>&1 \
  && grep -E "AUROC|AUPRC" runlog/M/lp_b0fast_seed0.log >> "$SUM"
echo "[$(date +%H:%M)] 车道A2: B0fast 校准完成" >> "$SUM"
"$PY" m_screen.py --config runlog/M/matrix_batch4.yaml >> "$SUM" 2>&1
echo "[$(date +%H:%M)] 车道A2: b4 跨域四件套结束" >> "$SUM"
"$PY" run_pt_ar.py --data-dir data/pt_pretrain --checkpoint-dir checkpoint/M/arb3_seed0 \
  --variant B0 --fusion mean --fusion-bt-weight 0.2 --lead-mask-prob 0.5 \
  --epochs 200 --batch-size 128 --workers 4 --seed 0 > runlog/M/pt_arb3_seed0.log 2>&1 \
  && echo "[$(date +%H:%M)] AR-B3 预训练完成" >> "$SUM" \
  || echo "[$(date +%H:%M)] AR-B3 预训练失败" >> "$SUM"
if [ -f checkpoint/M/arb3_seed0/encoder_group.pth ]; then
  "$PY" run_e006_downstream.py --data-dir data/ptbxl \
    --checkpoint checkpoint/M/arb3_seed0/encoder_group.pth \
    --output-dir runlog/M/lp_arb3_seed0 --task lp --base-variant ar_b3 \
    --eval-split test --epochs 100 --workers 6 > runlog/M/lp_arb3_seed0.log 2>&1
  grep -iE "auroc|auprc" runlog/M/lp_arb3_seed0/*.json runlog/M/lp_arb3_seed0.log 2>/dev/null | head -4 >> "$SUM"
fi
"$PY" run_e006_physiospatial.py --data-dir data/pt_pretrain \
  --checkpoint-dir checkpoint/M/psfull_arb3_seed0 \
  --profile physiospatial --base-variant ar_b3 --ablation full \
  --epochs 200 --batch-size 128 --workers 4 --seed 0 > runlog/M/pt_psfull_arb3_seed0.log 2>&1 \
  && echo "[$(date +%H:%M)] P1/S1 预训练完成" >> "$SUM" \
  || echo "[$(date +%H:%M)] P1/S1 预训练失败" >> "$SUM"
if [ -f checkpoint/M/psfull_arb3_seed0/encoder_group.pth ]; then
  "$PY" run_e006_downstream.py --data-dir data/ptbxl \
    --checkpoint checkpoint/M/psfull_arb3_seed0/encoder_group.pth \
    --output-dir runlog/M/lp_psfull_arb3_seed0 --task lp --base-variant ar_b3 \
    --eval-split test --epochs 100 --workers 6 > runlog/M/lp_psfull_arb3_seed0.log 2>&1
  grep -iE "auroc|auprc" runlog/M/lp_psfull_arb3_seed0/*.json runlog/M/lp_psfull_arb3_seed0.log 2>/dev/null | head -4 >> "$SUM"
fi
echo "[$(date +%H:%M)] 车道A2 全部结束" >> "$SUM"
