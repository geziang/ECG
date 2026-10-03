#!/bin/bash
# W6 Stage6 前置: MIT-BIH NSTDB 真实噪声库下载(PhysioNet 公开, 6 记录 bw/ma/em 各2)
# 产出 data/nstdb/{bw1,bw2,ma1,ma2,em1,em2}.{dat,hea}; 日志 runlog/W6/logs/nstdb_download.log
set -u
cd /f/新实验/ECG_SSL_LFBT-main
BASE="https://physionet.org/files/nstdb/1.0.0"
for r in bw1 bw2 ma1 ma2 em1 em2; do
  for ext in dat hea; do
    f="data/nstdb/$r.$ext"
    [ -s "$f" ] && { echo "[skip] $f 已存在"; continue; }
    echo "[get] $r.$ext ..."
    curl -fsSL --retry 5 --retry-delay 10 -o "$f" "$BASE/$r.$ext" && echo "[ok] $f $(stat -c%s "$f") bytes" || echo "[FAIL] $r.$ext"
  done
done
echo "[done] nstdb download $(date '+%F %T')"
ls -la data/nstdb/
