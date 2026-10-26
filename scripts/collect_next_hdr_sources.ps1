[CmdletBinding()]
param([ValidateSet('Status','Resume')][string]$Action='Status')
$ErrorActionPreference='Stop'
$repoRoot=Split-Path -Parent $PSScriptRoot
$batchRoot='G:\datasets\sources\next_hdr_sources_20261019_v3'
$pythonExe=Join-Path $repoRoot '.venv\Scripts\python.exe'
$active=@(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
    $_.CommandLine -match '(collect_next_hdr_sources|collect_tos_sequences|review_next_hdr_sources)\.py'
})
$controllers=@(Get-CimInstance Win32_Process -Filter "Name = 'pwsh.exe' OR Name = 'powershell.exe'" | Where-Object {
    $_.CommandLine -match 'finish_next_hdr_collection\.ps1'
})
if ($Action -eq 'Status') {
    $workflowPath=Join-Path $batchRoot 'workflow_state.json'
    if (Test-Path -LiteralPath $workflowPath) { Get-Content -LiteralPath $workflowPath }
    $summaryPath=Join-Path $batchRoot 'collection_summary.json'
    if (Test-Path -LiteralPath $summaryPath) {
        $summary=Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
        $summary |
            Select-Object stage,photographic_hdr_references,simulated_sdr_inputs,hdr_panoramas_8k,panorama_training_candidates,verified_tos_frames,continuous_exr_frames,sparse_tos_stills,excluded_tos_frames,blender_source_assets,training_pairs_added,training_enabled |
            ConvertTo-Json
        if ($summary.stage -eq 'source_pack_downloaded_and_numerically_verified' -and $active.Count -eq 0 -and $controllers.Count -eq 0) { return }
    }
    foreach ($group in @('polyhaven','polyhaven_novel','sihdr','tears_of_steel','blender')) {
        $statusPath=Join-Path (Join-Path $batchRoot $group) 'status.json'
        if (Test-Path -LiteralPath $statusPath) {
            $state=Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
            [pscustomobject]@{Group=$group;Stage=$state.stage;Count=$state.PSObject.Properties['count'].Value;Completed=$state.completed;Total=$state.total;Frames=$state.frames}
        } else { [pscustomobject]@{Group=$group;Stage='Pending or downloading; inspect files/logs'} }
    }
    $active | Select-Object ProcessId,CommandLine
    $controllers | Select-Object ProcessId,CommandLine
    return
}
if ($active.Count -gt 0 -or $controllers.Count -gt 0) { throw 'Collection/review/controller is already running. Use Status; do not launch duplicate writers.' }
Push-Location -LiteralPath $repoRoot
try {
    foreach ($group in @('environments','novel_environments','blender','sihdr')) {
        & $pythonExe -u pipeline\collect_next_hdr_sources.py $group
        if ($LASTEXITCODE -ne 0) { throw "Collection failed for $group; completed downloads retained." }
    }
    & $pythonExe -u pipeline\collect_tos_sequences.py
    if ($LASTEXITCODE -ne 0) { throw 'Sequence collection failed; completed frames retained.' }
    # Manual review inputs: no pipeline step writes these, and review/summary stop
    # with FileNotFoundError without them. Say so before launching the chain.
    $manual=@('licence_evidence\sihdr_reference_metadata.json','licence_evidence\sihdr_input_metadata.json',
              'initial_visual_review.json','sihdr\reference_admission_notes.json') |
        Where-Object { !(Test-Path -LiteralPath (Join-Path $batchRoot $_)) }
    if ($manual.Count -gt 0) { throw "Manual review prerequisites missing under ${batchRoot}: $($manual -join ', ')" }
    foreach ($group in @('overlap','overlap_novel','sihdr','tos')) {
        & $pythonExe -u pipeline\review_next_hdr_sources.py $group
        if ($LASTEXITCODE -ne 0) { throw "Review failed for $group; do not merge into training." }
    }
    & $pythonExe -u pipeline\summarize_next_hdr_collection.py
    if ($LASTEXITCODE -ne 0) { throw 'Source summary failed; inspect review artifacts.' }
    [ordered]@{stage='source_collection_verified_merge_review_pending';detail='Downloads and numeric source checks complete. Read collection_summary.json. No training admission or launch.';updated_utc=[DateTime]::UtcNow.ToString('o');training_started=$false;release_authorized=$false} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batchRoot 'workflow_state.json') -Encoding utf8
    Write-Output 'Source downloads and numeric reviews finished. Colour, split and visual review still govern training admission; no trainer launched.'
} finally { Pop-Location }
