$old = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'run_w8_relay' }
if ($old) { $old | ForEach-Object { Stop-Process -Id $_.ProcessId -Force } ; "KILLED: $($old.ProcessId -join ',')" }
else { "NO_OLD_RELAY" }
Start-Sleep 2
Start-Process -FilePath 'C:/Users/admin/.conda/envs/DL/python.exe' `
    -ArgumentList '-u', 'runlog/W8/run_w8_relay.py' `
    -WorkingDirectory 'F:/新实验/ECG_SSL_LFBT-main' -WindowStyle Hidden `
    -RedirectStandardOutput 'F:/新实验/ECG_SSL_LFBT-main/runlog/W8/logs/relay_stdout.log' `
    -RedirectStandardError 'F:/新实验/ECG_SSL_LFBT-main/runlog/W8/logs/relay_stderr.log'
"RELAY_RESTARTED(E2->E4->E6)"
