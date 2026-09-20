# hostB_watchdog.ps1 - pipeline watchdog (detached from any session, 2026-09-18)
# Every 10 min: if pipeline_runner is dead AND queue unfinished -> relaunch it detached.
# Safe by design: runner has csv-done / lp-log / claim / live-process-scan guards; run_pt has 6500MiB gate.
$Repo = "E:\GZA\ECG-main\ECG_SSL_LFBT-main"
$Py   = "C:\Users\508\Anaconda3\envs\DL\python.exe"
$Csv  = Join-Path $Repo "runlog\M\matrix_results_hostB.csv"
$Log  = Join-Path $Repo "runlog\M\watchdog.log"
$Done = @("c3_align_mix02", "c3_plain_mix02", "h4_hrv01")

function Log($m) {
    Add-Content -Path $Log -Value ("[{0}] {1}" -f (Get-Date -Format "MM-dd HH:mm"), $m)
}

Log "watchdog started (PID $PID)"
while ($true) {
    Start-Sleep -Seconds 600
    try {
        $runner = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
            Where-Object { $_.CommandLine -match 'pipeline_runner' }
        if ($runner) { continue }

        $csvText = ""
        if (Test-Path $Csv) { $csvText = Get-Content $Csv -Raw }
        $missing = @($Done | Where-Object { $csvText -notmatch ",$_," })
        if ($missing.Count -eq 0) { Log "queue finished, watchdog exit"; break }

        Log ("runner dead, missing tasks: " + ($missing -join ",") + " -> relaunch lane B")
        Start-Process -FilePath "cmd.exe" -WindowStyle Hidden -WorkingDirectory $Repo -ArgumentList (
            "/c echo [restart by watchdog %date% %time%] >> runlog\M\laneB_watchdog.log & " +
            "`"$Py`" -X UTF8 pipeline_runner.py --lane B --queue runlog\M/pipeline_queue_hostB.yaml " +
            "--results matrix_results_hostB.csv --max-pts 1 --vram-need 7000 >> runlog\M\laneB_watchdog.log 2>&1")
        Log "relaunch dispatched (if an orphan run_pt still runs, new runner will yield via live-scan)"
    } catch {
        Log ("check error: " + $_.Exception.Message)
    }
}
