# JOINT_20261013 chain detached launcher (ASCII only per host-A ps1 rule)
# Restarts the chain process if not alive. Idempotent: chain itself skips done work.
# This script lives at <repo>\runlog\JOINT_20261013\launch_joint.ps1, so repo root
# is three levels up from $PSCommandPath (no hardcoded non-ASCII paths).
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $PSCommandPath        # ...\runlog\JOINT_20261013
$root = Split-Path -Parent (Split-Path -Parent $here)   # repo root
$py = "C:\Users\admin\.conda\envs\DL\python.exe"
$chainLog = Join-Path $here "logs\chain.out.log"

# alive check: any python whose command line references run_joint_chain.py
$alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*run_joint_chain.py*" }
if ($alive) {
    Write-Output ("chain already alive, pid=" + ($alive[0].ProcessId))
    exit 0
}
New-Item -ItemType Directory -Force -Path (Split-Path $chainLog) | Out-Null
$proc = Start-Process -FilePath $py `
    -ArgumentList "-u", "runlog\JOINT_20261013\run_joint_chain.py" `
    -WorkingDirectory $root `
    -RedirectStandardOutput $chainLog `
    -RedirectStandardError ($chainLog + ".err") `
    -WindowStyle Hidden -PassThru
Write-Output ("chain launched, pid=" + $proc.Id)
