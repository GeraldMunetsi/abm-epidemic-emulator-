# progress.ps1
# Quick status snapshot for the orchestrator. Safe to run any time -- it only reads.
#
# Usage:
#   .\progress.ps1                # one-shot snapshot
#   .\progress.ps1 -Watch         # refresh every 10 seconds
#   .\progress.ps1 -Tail          # live-tail the active subprocess log

param(
    [switch]$Watch,
    [switch]$Tail
)

$root = Join-Path $PSScriptRoot "experiments\runs"

function Show-Snapshot {
    Clear-Host
    Write-Host "ORCHESTRATOR STATUS  ($(Get-Date -Format 'HH:mm:ss'))" -ForegroundColor Yellow
    Write-Host ("=" * 70)

    Write-Host "`nTRAINED MODELS  (expect 6)" -ForegroundColor Cyan
    if (Test-Path "$root\models") {
        $dirs = Get-ChildItem "$root\models" -Directory
        $done = 0
        $dirs | ForEach-Object {
            $n = (Get-ChildItem $_.FullName -Filter "*.pt" -ErrorAction SilentlyContinue).Count
            if ($n -gt 0) { $done++ }
            $marker = if ($n -gt 0) { "OK " } else { "-- " }
            "  $marker {0,-15} {1} checkpoints" -f $_.Name, $n
        }
        Write-Host "  -> $done / 6 trained" -ForegroundColor Green
    } else { Write-Host "  (models folder not yet created)" }

    Write-Host "`nVALIDATION  (expect 6 rows)" -ForegroundColor Cyan
    if (Test-Path "$root\validation\validation_results.csv") {
        $n = (Import-Csv "$root\validation\validation_results.csv" | Measure-Object).Count
        Write-Host "  -> $n / 6 rows in validation_results.csv" -ForegroundColor Green
    } else { Write-Host "  (validation_results.csv not yet created)" }

    Write-Host "`nTEST  (expect 18 rows)" -ForegroundColor Cyan
    if (Test-Path "$root\results\test_results.csv") {
        $n = (Import-Csv "$root\results\test_results.csv" | Measure-Object).Count
        Write-Host "  -> $n / 18 rows in test_results.csv" -ForegroundColor Green
    } else { Write-Host "  (test_results.csv not yet created)" }

    Write-Host "`nLAST 8 ORCHESTRATOR EVENTS" -ForegroundColor Cyan
    if (Test-Path "$root\logs\orchestrator.log") {
        Get-Content "$root\logs\orchestrator.log" -Tail 8 | ForEach-Object {
            Write-Host "  $_"
        }
    } else { Write-Host "  (orchestrator.log not yet created)" }

    Write-Host "`nACTIVE SUBPROCESS" -ForegroundColor Cyan
    $logs = Get-ChildItem "$root\logs\*.log" -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ne "orchestrator.log" } |
            Sort-Object LastWriteTime -Descending
    if ($logs) {
        $active = $logs[0]
        $age = [int]((Get-Date) - $active.LastWriteTime).TotalSeconds
        Write-Host "  $($active.Name)   (last write ${age}s ago)"
        Get-Content $active.FullName -Tail 5 | ForEach-Object { Write-Host "    | $_" }
    } else { Write-Host "  (no subprocess logs yet)" }

    Write-Host "`n" ("=" * 70)
}

if ($Tail) {
    $active = Get-ChildItem "$root\logs\*.log" |
              Where-Object { $_.Name -ne "orchestrator.log" } |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $active) { Write-Host "No subprocess log yet."; exit }
    Write-Host "Tailing $($active.Name) -- Ctrl-C to stop" -ForegroundColor Yellow
    Get-Content $active.FullName -Tail 40 -Wait
}
elseif ($Watch) {
    while ($true) { Show-Snapshot; Start-Sleep -Seconds 10 }
}
else {
    Show-Snapshot
}
