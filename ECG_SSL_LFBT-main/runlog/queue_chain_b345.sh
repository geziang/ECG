#!/bin/bash
# 统一链条: 等 b2 结束(24h 预算) -> b3 -> B0fast校准 -> b4 -> 条件γ -> AR车道
# 单等待器串联, 中间不依赖标记传递
cd "/f/新实验/ECG_SSL_LFBT-main" || exit 1
SUM=runlog/S1/overnight_summary.md
PY="C:/Users/admin/.conda/envs/DL/python.exe"
export PYTHONUNBUFFERED=1
waited=0
while ! grep -q "M 第二批结束" "$SUM" 2>/dev/null; do
  sleep 180; waited=$((waited + 3))
  if [ "$waited" -ge 1440 ]; then
    echo "[$(date +%H:%M)] 链条等待超时(24h)退出" >> "$SUM"; exit 1
  fi
done
echo "[$(date +%H:%M)] == M 第三批启动: 增强扫描+投影维度 ==" >> "$SUM"
"$PY" m_screen.py --config runlog/M/matrix_batch3.yaml >> "$SUM" 2>&1
echo "[$(date +%H:%M)] M 第三批结束" >> "$SUM"
echo "[$(date +%H:%M)] == B0-fast 校准跑启动 ==" >> "$SUM"
"$PY" run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 \
  --workers 6 --seed 0 --fast-backbone --checkpoint-dir checkpoint/M/b0fast_seed0 \
  > runlog/M/pt_b0fast_seed0.log 2>&1 \
  && "$PY" run_lp.py --data-dir data/ptbxl --checkpoint checkpoint/M/b0fast_seed0/encoder_group.pth \
     --num-classes 5 --feat-dir feat/M_b0fast_seed0 --seed 0 --workers 6 \
     > runlog/M/lp_b0fast_seed0.log 2>&1 \
  && grep -E "AUROC|AUPRC" runlog/M/lp_b0fast_seed0.log >> "$SUM"
echo "[$(date +%H:%M)] B0-fast 校准完成" >> "$SUM"
echo "[$(date +%H:%M)] == M 第四批启动: 跨域迁移四件套 ==" >> "$SUM"
"$PY" m_screen.py --config runlog/M/matrix_batch4.yaml >> "$SUM" 2>&1
echo "[$(date +%H:%M)] M 第四批结束" >> "$SUM"
if awk -F, '$2=="b0_gamma09" && $7+0>0 {found=1} END{exit !found}' runlog/M/matrix_results.csv; then
  echo "[$(date +%H:%M)] γ=0.9 为正 -> 追加 γ=0.95 探针" >> "$SUM"
  "$PY" m_screen.py --config runlog/M/matrix_batch2b.yaml >> "$SUM" 2>&1
else
  echo "[$(date +%H:%M)] γ=0.9 非正 -> 跳过追加" >> "$SUM"
fi
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
echo "[$(date +%H:%M)] == P1/S1 full on AR-B3 开始 ==" >> "$SUM"
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
echo "[$(date +%H:%M)] 统一链条(b3-校准-b4-γ-AR车道)全部结束" >> "$SUM"
