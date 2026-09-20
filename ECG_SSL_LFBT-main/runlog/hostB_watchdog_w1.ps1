# hostB_watchdog_w1.ps1 - W1 pipeline watchdog (2026-09-20)
# Every 10 min: if pipeline_runner dead AND queue unfinished -> relaunch detached.
# Hard deadline 2026-09-21 23:00 (26h) - stop relaunching, log final state.
$Repo = "E:\GZA\ECG-main\ECG_SSL_LFBT-main"
$Py   = "C:\Users\508\anaconda3\envs\DL\python.exe"
$Csv  = Join-Path $Repo "runlog\M\matrix_results_hostB.csv"
$Log  = Join-Path $Repo "runlog\M\watchdog_w1.log"
$Deadline = Get-Date "2026-09-21 23:00"
$Done = @("a1_d1lfix_05_02", "b3_ccm_rec01", "a4_rand_p101", "b2_multiseg_rec01",
          "a1_d1lfix_neg", "b1_d7_arch01", "a4_rand_p102", "a4_rand_p103")

function Log($m) {
    Add-Content -Path $Log -Value ("[{0}] {1}" -f (Get-Date -Format "MM-dd HH:mm"), $m)
}

Log "W1 watchdog started (PID $PID)"
while ($true) {
    Start-Sleep -Seconds 600
    try {
        if ((Get-Date) -gt $Deadline) {
            Log "past deadline, watchdog exit (runner left as-is)"
            break
        }
        $runner = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
            Where-Object { $_.CommandLine -match 'pipeline_runner' }
        if ($runner) { continue }

        $csvText = ""
        if (Test-Path $Csv) { $csvText = Get-Content $Csv -Raw }
        $missing = @($Done | Where-Object { $csvText -notmatch ",$_," })
        if ($missing.Count -eq 0) { Log "W1 queue finished, watchdog exit"; break }

        Log ("runner dead, missing: " + ($missing -join ",") + " -> relaunch lane B")
        Start-Process -FilePath "cmd.exe" -WindowStyle Hidden -WorkingDirectory $Repo -ArgumentList (
            "/c echo [restart by w1-watchdog %date% %time%] >> runlog\M\laneB_w1.log & " +
            "`"$Py`" -X UTF8 pipeline_runner.py --lane B --queue runlog/M/pipeline_queue_hostB.yaml " +
            "--results matrix_results_hostB.csv --max-pts 1 --vram-need 7000 >> runlog\M\laneB_w1.log 2>&1")
        Log "relaunch dispatched (orphan run_pt -> new runner yields via live-scan)"
    } catch {
        Log ("check error: " + $_.Exception.Message)
    }
}
