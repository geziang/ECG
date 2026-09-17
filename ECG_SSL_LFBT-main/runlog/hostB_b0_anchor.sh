#!/usr/bin/env bash
# 主机B(RTX 3080 10G) B0 锚点: 预训练(200ep 全默认=冻结协议) + LP, 结果写 matrix_results_hostB.csv
# 幂等性: 预训练完成以日志含 SHA256 为准; LP 完成以能解析出 AUPRC 为准。
set -uo pipefail
cd "$(dirname "$0")/.."
PY="C:/Users/508/Anaconda3/envs/DL/python.exe"
LOGD=runlog/M
CKPT=checkpoint/ptxl_gamma08

echo "[$(date '+%m-%d %H:%M:%S')] 主机B B0 锚点启动: run_pt 200ep(默认参数)" >> $LOGD/b0_hostB.log

if ! grep -q SHA256 $LOGD/pt_b0_seed0.log 2>/dev/null; then
  "$PY" run_pt.py --data-dir data/pt_pretrain --epochs 200 --batch-size 128 \
      --workers 6 --seed 0 --checkpoint-dir $CKPT > $LOGD/pt_b0_seed0.log 2>&1
fi
if ! grep -q SHA256 $LOGD/pt_b0_seed0.log 2>/dev/null; then
  echo "[$(date '+%m-%d %H:%M:%S')] 预训练失败(见 pt_b0_seed0.log)" >> $LOGD/b0_hostB.log
  exit 1
fi
echo "[$(date '+%m-%d %H:%M:%S')] 预训练完成, LP 开始" >> $LOGD/b0_hostB.log

"$PY" run_lp.py --data-dir data/ptbxl --checkpoint $CKPT/encoder_group.pth \
    --num-classes 5 --feat-dir feat/hostB_b0 --seed 0 --workers 6 > $LOGD/lp_b0_seed0.log 2>&1
AUROC=$(grep -oE 'AUROC *= *[0-9.]+' $LOGD/lp_b0_seed0.log | tail -1 | grep -oE '[0-9.]+')
AUPRC=$(grep -oE 'AUPRC *= *[0-9.]+' $LOGD/lp_b0_seed0.log | tail -1 | grep -oE '[0-9.]+')
if [ -z "${AUPRC:-}" ]; then
  echo "[$(date '+%m-%d %H:%M:%S')] LP 失败(见 lp_b0_seed0.log)" >> $LOGD/b0_hostB.log
  exit 1
fi

CSV=$LOGD/matrix_results_hostB.csv
[ -f "$CSV" ] || echo "ts,name,seed,args,auroc,auprc,delta,fast" > "$CSV"
echo "$(date '+%m-%d %H:%M'),b0,0,--gamma 0.8 --lambd 0.0051(默认全协议),$AUROC,$AUPRC,0,0" >> "$CSV"
echo "[$(date '+%m-%d %H:%M:%S')] ✔ 主机B B0 锚点完成: AUROC=$AUROC AUPRC=$AUPRC" >> $LOGD/b0_hostB.log
