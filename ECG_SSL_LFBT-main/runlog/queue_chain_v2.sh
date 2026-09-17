#!/bin/bash
# v2 链条(裁剪趋势优先): crop追边界 -> b3剩余 -> crop_weak确认(seed2,4) -> B0fast -> b4 -> AR车道
cd "/f/新实验/ECG_SSL_LFBT-main" || exit 1
SUM=runlog/S1/overnight_summary.md
PY="C:/Users/admin/.conda/envs/DL/python.exe"
export PYTHONUNBUFFERED=1
echo "[$(date +%H:%M)] == v2 链条启动: 裁剪趋势优先(aug_crop_weak +0.27 首个正探针) ==" >> "$SUM"
"$PY" m_screen.py --config runlog/M/matrix_batch3c.yaml >> "$SUM" 2>&1
echo "[$(date +%H:%M)] 裁剪追边界+b3剩余结束" >> "$SUM"
echo "[$(date +%H:%M)] == aug_crop_weak 3seed 确认(seed 2,4) ==" >> "$SUM"
"$PY" m_screen.py --config runlog/M/matrix_batch3.yaml --confirm aug_crop_weak >> "$SUM" 2>&1
echo "[$(date +%H:%M)] aug_crop_weak 确认轮结束" >> "$SUM"
echo "[$(date +%H:%M)] == B0-fast 校准跑 ==" >> "$SUM"
"$PY" run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 \
  --workers 6 --seed 0 --fast-backbone --checkpoint-dir checkpoint/M/b0fast_seed0 \
  > runlog/M/pt_b0fast_seed0.log 2>&1 \
  && "$PY" run_lp.py --data-dir data/ptbxl --checkpoint checkpoint/M/b0fast_seed0/encoder_group.pth \
     --num-classes 5 --feat-dir feat/M_b0fast_seed0 --seed 0 --workers 6 \
     > runlog/M/lp_b0fast_seed0.log 2>&1 \
  && grep -E "AUROC|AUPRC" runlog/M/lp_b0fast_seed0.log >> "$SUM"
echo "[$(date +%H:%M)] B0-fast 校准完成" >> "$SUM"
echo "[$(date +%H:%M)] == M 第四批: 跨域四件套 ==" >> "$SUM"
"$PY" m_screen.py --config runlog/M/matrix_batch4.yaml >> "$SUM" 2>&1
echo "[$(date +%H:%M)] M 第四批结束" >> "$SUM"
if awk -F, '$2=="b0_gamma09" && $7+0>0 {found=1} END{exit !found}' runlog/M/matrix_results.csv; then
  "$PY" m_screen.py --config runlog/M/matrix_batch2b.yaml >> "$SUM" 2>&1
else
  echo "[$(date +%H:%M)] γ=0.9 非正 -> 跳过追加" >> "$SUM"
fi
echo "[$(date +%H:%M)] == AR-B3 基座预训练 ==" >> "$SUM"
"$PY" run_pt_ar.py --data-dir data/pt_pretrain --checkpoint-dir checkpoint/M/arb3_seed0 \
  --variant B0 --fusion mean --fusion-bt-weight 0.2 --lead-mask-prob 0.5 \
  --epochs 200 --batch-size 128 --workers 6 --seed 0 > runlog/M/pt_arb3_seed0.log 2>&1 \
  && echo "[$(date +%H:%M)] AR-B3 预训练完成" >> "$SUM" \
  || echo "[$(date +%H:%M)] AR-B3 预训练失败" >> "$SUM"
if [ -f checkpoint/M/arb3_seed0/encoder_group.pth ]; then
  "$PY" run_e006_downstream.py --data-dir data/ptbxl \
    --checkpoint checkpoint/M/arb3_seed0/encoder_group.pth \
    --output-dir runlog/M/lp_arb3_seed0 --task lp --base-variant ar_b3 \
    --eval-split test --epochs 100 --workers 6 > runlog/M/lp_arb3_seed0.log 2>&1
  grep -iE "auroc|auprc" runlog/M/lp_arb3_seed0/*.json runlog/M/lp_arb3_seed0.log 2>/dev/null | head -4 >> "$SUM"
fi
echo "[$(date +%H:%M)] == P1/S1 full on AR-B3 ==" >> "$SUM"
"$PY" run_e006_physiospatial.py --data-dir data/pt_pretrain \
  --checkpoint-dir checkpoint/M/psfull_arb3_seed0 \
  --profile physiospatial --base-variant ar_b3 --ablation full \
  --epochs 200 --batch-size 128 --workers 6 --seed 0 > runlog/M/pt_psfull_arb3_seed0.log 2>&1 \
  && echo "[$(date +%H:%M)] P1/S1 预训练完成" >> "$SUM" \
  || echo "[$(date +%H:%M)] P1/S1 预训练失败" >> "$SUM"
if [ -f checkpoint/M/psfull_arb3_seed0/encoder_group.pth ]; then
  "$PY" run_e006_downstream.py --data-dir data/ptbxl \
    --checkpoint checkpoint/M/psfull_arb3_seed0/encoder_group.pth \
    --output-dir runlog/M/lp_psfull_arb3_seed0 --task lp --base-variant ar_b3 \
    --eval-split test --epochs 100 --workers 6 > runlog/M/lp_psfull_arb3_seed0.log 2>&1
  grep -iE "auroc|auprc" runlog/M/lp_psfull_arb3_seed0/*.json runlog/M/lp_psfull_arb3_seed0.log 2>/dev/null | head -4 >> "$SUM"
fi
echo "[$(date +%H:%M)] v2 链条全部结束" >> "$SUM"
