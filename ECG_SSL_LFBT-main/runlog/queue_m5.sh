#!/bin/bash
# b5 接力: 等 b4 结束 -> 条件γ追加 -> AR-B3 车道(基座 + P1/S1 full)
cd "/f/新实验/ECG_SSL_LFBT-main" || exit 1
SUM=runlog/S1/overnight_summary.md
PY="C:/Users/admin/.conda/envs/DL/python.exe"
waited=0
while ! grep -q "M 第四批结束" "$SUM" 2>/dev/null; do
  sleep 180; waited=$((waited + 3))
  if [ "$waited" -ge 1080 ]; then
    echo "[$(date +%H:%M)] M5 等待超时(18h)退出" >> "$SUM"; exit 1
  fi
done
# 条件 γ 追加: gamma09 为正才探 0.95
if awk -F, '$2=="b0_gamma09" && $7+0>0 {found=1} END{exit !found}' runlog/M/matrix_results.csv; then
  echo "[$(date +%H:%M)] γ=0.9 为正 -> 追加 γ=0.95 探针" >> "$SUM"
  "$PY" m_screen.py --config runlog/M/matrix_batch2b.yaml >> "$SUM" 2>&1
else
  echo "[$(date +%H:%M)] γ=0.9 非正 -> 跳过追加" >> "$SUM"
fi
# AR-B3 基座
echo "[$(date +%H:%M)] == AR-B3 基座预训练开始 ==" >> "$SUM"
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
# P1/S1 full on AR-B3 (E006 弱正重确认)
echo "[$(date +%H:%M)] == P1/S1(physiospatial full) on AR-B3 开始 ==" >> "$SUM"
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
echo "[$(date +%H:%M)] M 第五批(AR 车道)结束" >> "$SUM"
