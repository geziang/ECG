# ABC_20261015 chain detached launcher (ASCII only per host-A ps1 rule)
# Restarts the chain process if not alive. Idempotent: chain itself skips done work.
# Alive check is two-layered: CIM command-line scan OR pidfile + Get-Process direct
# (CIM is known to drop rows; pidfile guards against a duplicate chain launch).
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $PSCommandPath        # ...\runlog\ABC_20261015
$root = Split-Path -Parent (Split-Path -Parent $here)   # repo root
$py = "C:\Users\admin\.conda\envs\DL\python.exe"
$chainLog = Join-Path $here "logs\chain.out.log"
$pidFile = Join-Path $here "logs\chain.pid"

# layer 1: any python whose command line references run_abc_chain.py
try {
    $alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*run_abc_chain.py*" }
} catch { $alive = $null }
if ($alive) {
    Write-Output ("chain already alive (cim), pid=" + ($alive[0].ProcessId))
    exit 0
}
# layer 2: pidfile + direct Get-Process
if (Test-Path $pidFile) {
    $oldPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    $oldPid = "$oldPid".Trim()
    if ($oldPid -match '^\d+$') {
        $p = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
        if ($p -and $p.ProcessName -eq "python") {
            Write-Output ("chain already alive (pidfile), pid=" + $oldPid)
            exit 0
        }
    }
}
New-Item -ItemType Directory -Force -Path (Split-Path $chainLog) | Out-Null
$proc = Start-Process -FilePath $py `
    -ArgumentList "-u", "runlog\ABC_20261015\run_abc_chain.py" `
    -WorkingDirectory $root `
    -RedirectStandardOutput $chainLog `
    -RedirectStandardError ($chainLog + ".err") `
    -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 3
Write-Output ("chain launched, pid=" + $proc.Id + " exited=" + $proc.HasExited)
