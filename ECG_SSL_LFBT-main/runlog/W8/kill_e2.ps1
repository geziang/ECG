$victims = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
    $_.CommandLine -match 'run_e2_simph_chain' -or
    ($_.CommandLine -match 'run_pt\.py' -and $_.CommandLine -match 'w8_simph')
}
$victims | ForEach-Object { "KILL $($_.ProcessId): $($_.CommandLine.Substring(0, [Math]::Min(110, $_.CommandLine.Length)))" }
$victims | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
# PT 的 DataLoader worker 是 python 子进程但命令行同 run_pt, 上面的 CommandLine 匹配已覆盖;
# 兜底再清一遍孤儿(父进程已死且命令行含 w8_simph)
Start-Sleep 2
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
    $_.CommandLine -match 'w8_simph'
} | ForEach-Object { "LEFT $($_.ProcessId)" }
