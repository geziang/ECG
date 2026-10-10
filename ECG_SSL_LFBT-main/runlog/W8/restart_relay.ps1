# ASCII-only on purpose: this file must survive any codepage (2026-10-10 lesson).
$repo = (Get-Item $PSScriptRoot).Parent.Parent.FullName
$logs = Join-Path $PSScriptRoot 'logs'
$old = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'run_w8_relay' }
if ($old) { $old | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; "KILLED: $($old.ProcessId -join ',')" }
else { "NO_OLD_RELAY" }
Start-Sleep 2
Start-Process -FilePath 'C:/Users/admin/.conda/envs/DL/python.exe' `
    -ArgumentList '-u', 'runlog/W8/run_w8_relay.py' `
    -WorkingDirectory $repo -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logs 'relay_stdout.log') `
    -RedirectStandardError (Join-Path $logs 'relay_stderr.log')
"RELAY_RESTARTED(E2-E4-E6)"
