param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$taskWorkspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$taskPrefix = $taskWorkspace.TrimEnd('\') + '\'
$taskPlanPath = Join-Path $PSScriptRoot 'cleanup_recycle_2026-10-23/plan.json.gz'
$taskReader = [System.IO.StreamReader]::new([System.IO.Compression.GZipStream]::new([System.IO.File]::OpenRead($taskPlanPath), [System.IO.Compression.CompressionMode]::Decompress))
try { $taskPlan = $taskReader.ReadToEnd() | ConvertFrom-Json } finally { $taskReader.Dispose() }
if ($taskPlan.workspace -ne $taskWorkspace) { throw 'Plan workspace mismatch' }
if ([System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($taskWorkspace)).DriveType -ne [System.IO.DriveType]::Fixed) { throw 'Recycle operation requires a local fixed drive' }
$taskTracked = @(git -C (Join-Path $taskWorkspace 'ECG') -c core.quotePath=false ls-files)
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect tracked files' }
$taskTargets = [System.Collections.Generic.List[object]]::new()
foreach ($taskEntry in $taskPlan.candidates.PSObject.Properties) {
    $taskTarget = [System.IO.Path]::GetFullPath((Join-Path $taskWorkspace $taskEntry.Name))
    if (-not $taskTarget.StartsWith($taskPrefix, [System.StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $taskTarget" }
    if (($taskTarget.Substring($taskPrefix.Length) -split '\\') -contains '.git') { throw 'Refusing Git metadata' }
    if ($taskEntry.Name.StartsWith('ECG/') -and $taskTracked -contains $taskEntry.Name.Substring(4)) { throw "Tracked file: $taskTarget" }
    $taskFile = Get-Item -LiteralPath $taskTarget -Force
    if ($taskFile.PSIsContainer) { throw "Expected a regular file: $taskTarget" }
    $taskAncestor = $taskFile
    while ($taskAncestor.FullName -ne $taskWorkspace) {
        if ($taskAncestor.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw "Reparse point: $($taskAncestor.FullName)" }
        $taskAncestor = Get-Item -LiteralPath ([System.IO.Path]::GetDirectoryName($taskAncestor.FullName)) -Force
    }
    if ($taskFile.Length -ne $taskEntry.Value.bytes -or (Get-FileHash -LiteralPath $taskTarget -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskEntry.Value.sha256) { throw "Candidate changed: $taskTarget" }
    $taskTargets.Add([pscustomobject]@{Path=$taskTarget;RelativePath=$taskEntry.Name;Bytes=$taskEntry.Value.bytes;Sha256=$taskEntry.Value.sha256})
}
if (-not $Execute) {
    [pscustomobject]@{Mode='ValidatedOnly';Files=$taskTargets.Count;Bytes=($taskTargets.Bytes | Measure-Object -Sum).Sum} | ConvertTo-Json
    return
}
$taskLogPath = Join-Path $PSScriptRoot 'cleanup_recycle_2026-10-23/recycled.jsonl'
if (Test-Path -LiteralPath $taskLogPath) { throw 'Execution journal already exists; inspect it before another attempt' }
Add-Type -AssemblyName Microsoft.VisualBasic.Core
$taskLog = [System.IO.StreamWriter]::new($taskLogPath, $false, [System.Text.UTF8Encoding]::new($false))
$taskDone = 0
try {
    foreach ($taskTarget in $taskTargets) {
        [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile(
            $taskTarget.Path,
            [Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs,
            [Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin,
            [Microsoft.VisualBasic.FileIO.UICancelOption]::ThrowException)
        if (Test-Path -LiteralPath $taskTarget.Path) { throw "File remains: $($taskTarget.RelativePath)" }
        $taskLog.WriteLine((@{path=$taskTarget.RelativePath;bytes=$taskTarget.Bytes;sha256=$taskTarget.Sha256;operation='SendToRecycleBin'} | ConvertTo-Json -Compress))
        $taskLog.Flush()
        $taskDone++
    }
} finally { $taskLog.Dispose() }
[pscustomobject]@{Mode='SendToRecycleBin';Files=$taskDone;Bytes=($taskTargets.Bytes | Measure-Object -Sum).Sum;RecycleBinEmptied=$false} | ConvertTo-Json
