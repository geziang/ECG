# launch_w9.ps1 - W9 FT10 chain idempotent launcher (ASCII only)
# Repo root = two levels up from this script (runlog\W9 -> runlog -> ECG_SSL_LFBT-main)
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
$py = "C:\Users\admin\.conda\envs\DL\python.exe"
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# The runner itself guards against double start via runlog\W9\logs\run.pid
# (a live python chain makes a fresh --stages run exit immediately, so this
#  launcher is always safe to re-run).
$logdir = Join-Path $repo "runlog\W9\logs"
if (-not (Test-Path $logdir)) { New-Item -ItemType Directory -Path $logdir | Out-Null }
$out = Join-Path $logdir "launcher_out.log"
$err = Join-Path $logdir "launcher_err.log"

Write-Output "[$stamp] launching W9 chain from $repo"
$p = Start-Process -FilePath $py -ArgumentList @('-u', 'runlog\W9\run_w9_submission_ft10.py', '--stages', 'run') -WorkingDirectory $repo -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 5
if ($p.HasExited) {
    Write-Output "[$stamp] LAUNCH FAILED, exit code $($p.ExitCode); stderr tail:"
    if (Test-Path $err) { Get-Content $err -Tail 20 }
    exit 1
}
Write-Output "[$stamp] W9 chain started, pid=$($p.Id)"
