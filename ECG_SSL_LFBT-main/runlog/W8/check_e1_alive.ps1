$hits = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'run_e1_rand' }
if ($hits) { exit 0 } else { exit 1 }
