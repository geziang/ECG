#!/bin/bash
# 夜间值守队列: N1 完成 -> LP 判定 -> S1/D9 粗筛 x4 (预训练+LP 成对)
# 纪律: 全部 PTB-XL 单库语料 (data/pt_pretrain), 与 B0 锚点直接可比
cd "/f/新实验/ECG_SSL_LFBT-main" || exit 1
PY="C:/Users/admin/.conda/envs/DL/python.exe"
export PYTHONUNBUFFERED=1
SUM=runlog/S1/overnight_summary.md
mkdir -p runlog/S1 runlog/N1
: > "$SUM"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUM"; }
log "夜间队列启动"

# ---- 1. 等 N1 预训练完成 (checkpoint + SHA256 行; 45 分钟无日志更新视为停滞则跳过) ----
while true; do
  if [ -f checkpoint/n1_multilib_seed0/encoder_group.pth ] && grep -q "SHA256" runlog/N1/pt_n1_seed0.log 2>/dev/null; then
    log "N1 预训练完成"; break
  fi
  now=$(date +%s); mtime=$(stat -c %Y runlog/N1/pt_n1_seed0.log 2>/dev/null || echo $now)
  if [ $((now - mtime)) -gt 2700 ]; then
    log "警告: N1 日志 45 分钟未更新, 判定停滞, 跳过 N1-LP 继续队列"; SKIP_N1=1; break
  fi
  sleep 60
done

run_lp() {  # $1=checkpoint dir $2=feat dir $3=log
  "$PY" run_lp.py --data-dir data/ptbxl --checkpoint "$1/encoder_group.pth" \
    --num-classes 5 --feat-dir "$2" --seed 0 --workers 6 > "$3" 2>&1
  grep -E "AUROC|AUPRC" "$3" | tee -a "$SUM"
}

# ---- 2. N1 LP 判定 ----
if [ -z "$SKIP_N1" ]; then
  log "== N1 LP 评估 =="
  run_lp checkpoint/n1_multilib_seed0 feat/n1_seed0 runlog/N1/lp_n1_seed0.log
fi

# ---- 3. S1/D9 粗筛: (变体名, run_pt 附加参数, seed) ----
run_pair() {  # $1=name $2=extra_args $3=seed
  log "== S1 粗筛: $1 seed $3 开始 =="
  "$PY" run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 \
    --workers 6 --seed "$3" $2 --checkpoint-dir "checkpoint/s1_$1_seed$3" \
    > "runlog/S1/pt_${1}_seed$3.log" 2>&1 \
    && log "$1 seed$3 预训练完成" \
    || { log "$1 seed$3 预训练失败, 跳过其 LP"; return 1; }
  log "$1 seed$3 LP 开始"
  run_lp "checkpoint/s1_$1_seed$3" "feat/s1_${1}_seed$3" "runlog/S1/lp_${1}_seed$3.log"
}

run_pair d9vicreg    "--loss-mode vicreg"                              0
run_pair d9vicreg_bn "--loss-mode vicreg --vicreg-keep-bn"             0
run_pair d9vicreg_ln "--loss-mode vicreg --projector-norm layernorm"   0
run_pair d9vicreg    "--loss-mode vicreg"                              2

log "夜间队列全部结束"
