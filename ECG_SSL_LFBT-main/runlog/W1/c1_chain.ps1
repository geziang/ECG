# C1 NFH B0' 手动链 (detached 运行, 不依赖任何终端会话)
# PT(100ep, worktree 代码, NFH 数据) -> LP(PTB-XL 下游, 外部锚点读数) -> csv 记账(幂等)
$ErrorActionPreference = 'Continue'
Set-Location 'F:\新实验\ECG_SSL_LFBT-wt1\ECG_SSL_LFBT-main'
$PY  = 'C:\Users\admin\.conda\envs\DL\python.exe'
$MAIN = 'F:\新实验\ECG_SSL_LFBT-main'

& $PY run_pt.py --data-dir "$MAIN\data\pt_pretrain_nfh" --epochs 100 --batch-size 128 `
    --workers 4 --seed 0 --checkpoint-dir "$MAIN\checkpoint\M\c1_nfh_b0_seed0" `
    1> "$MAIN\runlog\W1\pt_c1_nfh_b0_seed0.log" 2>&1
"PT_EXIT=$LASTEXITCODE" | Out-File "$MAIN\runlog\W1\c1_chain_status.txt" -Encoding utf8
if ($LASTEXITCODE -ne 0) { exit 1 }

& $PY run_lp.py --data-dir "$MAIN\data\ptbxl" `
    --checkpoint "$MAIN\checkpoint\M\c1_nfh_b0_seed0\encoder_group.pth" `
    --num-classes 5 --feat-dir "$MAIN\feat\M_c1_nfh_b0_seed0" --seed 0 --workers 6 `
    1> "$MAIN\runlog\W1\lp_c1_nfh_b0_seed0.log" 2>&1
"LP_EXIT=$LASTEXITCODE" | Out-File "$MAIN\runlog\W1\c1_chain_status.txt" -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { exit 2 }

& $PY "$MAIN\runlog\W1\append_c1_csv.py" *>> "$MAIN\runlog\W1\c1_chain_status.txt"
"DONE $(Get-Date -Format 'MM-dd HH:mm')" | Out-File "$MAIN\runlog\W1\c1_chain_status.txt" -Append -Encoding utf8
