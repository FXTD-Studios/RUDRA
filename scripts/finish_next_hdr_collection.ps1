[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
$repoRoot=Split-Path -Parent $PSScriptRoot
$batchRoot='G:\datasets\sources\next_hdr_sources_20261019_v3'
$pythonExe=Join-Path $repoRoot '.venv\Scripts\python.exe'
function Write-State([string]$stage,[string]$detail) {
    $state=[ordered]@{stage=$stage;detail=$detail;updated_utc=[DateTime]::UtcNow.ToString('o');controller_pid=$PID;training_started=$false;release_authorized=$false}
    $path=Join-Path $batchRoot 'workflow_state.json'
    $state | ConvertTo-Json | Set-Content -LiteralPath ($path+'.tmp') -Encoding utf8
    Move-Item -LiteralPath ($path+'.tmp') -Destination $path -Force
}
Push-Location -LiteralPath $repoRoot
try {
    $watch=Get-Content -LiteralPath (Join-Path $batchRoot 'download_processes.json') -Raw | ConvertFrom-Json
    $deadline=[DateTime]::UtcNow.AddHours(4)
    do {
        $live=@($watch | Where-Object {
            $proc=Get-Process -Id $_.pid -ErrorAction SilentlyContinue
            $proc -and $proc.StartTime.ToUniversalTime().Ticks.ToString() -eq $_.start_ticks
        })
        if ($live.Count -gt 0) {
            if ([DateTime]::UtcNow -gt $deadline) { throw 'Download wait exceeded four hours; inspect retained partial files and live processes.' }
            Write-State 'waiting_for_downloads' "$($live.Count) recorded download processes still running."
            Start-Sleep -Seconds 15
        }
    } while ($live.Count -gt 0)
    foreach ($group in @('polyhaven','polyhaven_novel','sihdr','tears_of_steel','blender')) {
        if (!(Test-Path -LiteralPath (Join-Path (Join-Path $batchRoot $group) 'status.json'))) { throw "Missing completion record for $group" }
    }
    Write-State 'numeric_source_review' 'Full decodes, publisher archive checksums, hashes and split-history checks; no training.'
    foreach ($group in @('overlap','overlap_novel','sihdr','tos')) {
        & $pythonExe -u pipeline\review_next_hdr_sources.py $group
        if ($LASTEXITCODE -ne 0) { throw "Source review failed: $group" }
    }
    & $pythonExe -u pipeline\summarize_next_hdr_collection.py
    if ($LASTEXITCODE -ne 0) { throw 'Source summary failed' }
    Write-State 'source_collection_verified_merge_review_pending' 'Downloads and numeric source checks complete. Read collection_summary.json and NEXT_MERGE_PLAN.json. No training admission or launch.'
} catch {
    Write-State 'failed' $_.Exception.Message
    throw
} finally { Pop-Location }
