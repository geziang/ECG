#!/bin/bash
# 下午队列: 等 B0seed2 接力结束 -> D9 最后确认批次 (R1 重标定系数 / lite 最小防塌补丁)
cd "/f/新实验/ECG_SSL_LFBT-main" || exit 1
PY="C:/Users/admin/.conda/envs/DL/python.exe"
export PYTHONUNBUFFERED=1
SUM=runlog/S1/overnight_summary.md

waited=0
while ! grep -q "接力队列结束" "$SUM" 2>/dev/null; do
  sleep 120; waited=$((waited + 2))
  [ "$waited" -ge 480 ] && { echo "[$(date +%H:%M)] 下午队列等待超时(8h)退出" >> "$SUM"; exit 1; }
done

run_lp() {
  "$PY" run_lp.py --data-dir data/ptbxl --checkpoint "$1/encoder_group.pth" \
    --num-classes 5 --feat-dir "$2" --seed 0 --workers 6 > "$3" 2>&1
  grep -E "AUROC|AUPRC" "$3" | tee -a "$SUM"
}
run_pair() {  # $1=name $2=extra_args $3=seed
  echo "[$(date +%H:%M)] == D9 确认: $1 seed $3 开始 ==" >> "$SUM"
  "$PY" run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 \
    --workers 6 --seed "$3" $2 --checkpoint-dir "checkpoint/s1_$1_seed$3" \
    > "runlog/S1/pt_${1}_seed$3.log" 2>&1 \
    && echo "[$(date +%H:%M)] $1 seed$3 预训练完成" >> "$SUM" \
    || { echo "[$(date +%H:%M)] $1 seed$3 预训练失败" >> "$SUM"; return 1; }
  run_lp "checkpoint/s1_$1_seed$3" "feat/s1_${1}_seed$3" "runlog/S1/lp_${1}_seed$3.log"
}

# D9-R1: 系数重标定 (sim 降 25->1, 去掉 MSE 主导; var 10; cov 1)
run_pair d9r1  "--loss-mode vicreg --vicreg-sim 1 --vicreg-var 10 --vicreg-cov 1" 0
# D9-lite: B0 原样 + 仅 variance hinge(权重10)
run_pair d9lite "--bt-var-hinge 10" 0

echo "[$(date +%H:%M)] 下午队列结束(D9 最终判定材料齐)" >> "$SUM"
