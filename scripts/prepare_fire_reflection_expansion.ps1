param([ValidateSet('Status','Prepare','Start')][string]$Action='Status')
$ErrorActionPreference='Stop'
$repoRoot=Split-Path -Parent $PSScriptRoot
$batchRoot='G:\datasets\corpora\rudra_fire_reflection_expansion_20261018_v2'
$sourceRoot='G:\datasets\sources\fire_reflection_expansion_20261018_v2'
$pythonExe=Join-Path $repoRoot '.venv\Scripts\python.exe'
$blenderExe='C:\Program Files\Blender Foundation\Blender 5.0\blender.exe'
if ($Action -eq 'Status') {
    $workflow=Join-Path $batchRoot 'workflow_state.json'
    if (Test-Path -LiteralPath $workflow) { Get-Content -LiteralPath $workflow }
    $summary=Join-Path $batchRoot 'preparation_summary.json'
    if (Test-Path -LiteralPath $summary) { Get-Content -LiteralPath $summary }
    else {
        $renders=@(Get-ChildItem -LiteralPath (Join-Path $batchRoot 'render_records') -Filter '*.json' -ErrorAction SilentlyContinue).Count
        $receipts=@(Get-ChildItem -LiteralPath (Join-Path $sourceRoot 'receipts') -Filter '*.json' -ErrorAction SilentlyContinue).Count
        [ordered]@{stage='preparing';downloaded_source_packs=$receipts;target_source_packs=9;rendered_frames=$renders;target_frames=156;target_new_pairs=312;training_started=$false} | ConvertTo-Json
    }
    $review=Join-Path $batchRoot 'review_status.json'
    if (Test-Path -LiteralPath $review) {
        $r=Get-Content -LiteralPath $review -Raw | ConvertFrom-Json
        $manifest=Join-Path $batchRoot 'combined_candidate_manifest.jsonl'
        if ((Test-Path -LiteralPath $manifest) -and (Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash.ToLowerInvariant() -eq $r.manifest_sha256) { Get-Content -LiteralPath $review }
        else { Write-Output 'Saved visual review is stale.' }
    }
    foreach ($logName in @('prepare_errors.log','prepare_console.log')) {
        $logPath=Join-Path $batchRoot $logName
        if (Test-Path -LiteralPath $logPath) { Get-Content -LiteralPath $logPath -Tail 4 }
    }
    return
}
$workflowPath=Join-Path $batchRoot 'workflow_state.json'
if (Test-Path -LiteralPath $workflowPath) {
    $workflow=Get-Content -LiteralPath $workflowPath -Raw | ConvertFrom-Json
    if ($workflow.controller_pid -ne $PID -and $workflow.stage -notin @('failed','numeric_checks_complete_visual_review_pending')) {
        $controller=Get-CimInstance Win32_Process -Filter "ProcessId = $($workflow.controller_pid)" -ErrorAction SilentlyContinue
        if ($controller -and $controller.CommandLine -match 'finish_preparation.ps1') { throw 'The preparation controller is active. Use -Action Status.' }
    }
}
$running=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(python|blender)\.exe$' -and $_.CommandLine -match '(collect_fire_reflection_sources|prepare_fire_reflection_expansion|render_fire_reflection_scenes)\.py' })
if ($running.Count -gt 0) { throw 'This preparation is already running. Use -Action Status.' }
New-Item -ItemType Directory -Path $batchRoot -Force | Out-Null
if ($Action -eq 'Start') {
    $arguments=@('-NoProfile','-ExecutionPolicy','Bypass','-File',('"'+$PSCommandPath+'"'),'-Action','Prepare')
    $process=Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $batchRoot 'prepare_console.log') -RedirectStandardError (Join-Path $batchRoot 'prepare_errors.log')
    Write-Output "Preparation started: PID $($process.Id). No training will start."
    return
}
if (!(Test-Path -LiteralPath $pythonExe) -or !(Test-Path -LiteralPath $blenderExe)) { throw 'Required existing runtime missing.' }
Push-Location -LiteralPath $repoRoot
try {
    if (!(Test-Path -LiteralPath (Join-Path $sourceRoot 'summary.json'))) {
        & $pythonExe -u 'pipeline\collect_fire_reflection_sources.py'
        if ($LASTEXITCODE -ne 0) { throw 'Source collection failed; completed receipts retained.' }
    }
    & $pythonExe -u 'pipeline\prepare_fire_reflection_expansion.py' plan
    if ($LASTEXITCODE -ne 0) { throw 'Plan failed.' }
    $planPath=Join-Path $batchRoot 'plan.json'
    $plan=Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
    $uninspected=@($plan.cg_jobs | Where-Object { $_.category -eq 'fire_reflection' -and !$_.volume_scale })
    if ($uninspected.Count -gt 0) {
        & $blenderExe --background --factory-startup --disable-autoexec --threads 2 --python-exit-code 1 --python (Join-Path $repoRoot 'pipeline\render_fire_reflection_scenes.py') -- --plan $planPath --out $batchRoot --inspect
        if ($LASTEXITCODE -ne 0) { throw 'Volume inspection failed.' }
    }
    & $blenderExe --background --factory-startup --disable-autoexec --threads 2 --python-exit-code 1 --python (Join-Path $repoRoot 'pipeline\render_fire_reflection_scenes.py') -- --plan $planPath --out $batchRoot
    if ($LASTEXITCODE -ne 0) { throw 'Render failed; completed frames retained.' }
    & $pythonExe -u 'pipeline\prepare_fire_reflection_expansion.py' convert
    if ($LASTEXITCODE -ne 0) { throw 'Pair conversion failed.' }
    & $pythonExe -u 'pipeline\prepare_fire_reflection_expansion.py' verify
    if ($LASTEXITCODE -ne 0) { throw 'Verification failed.' }
    Write-Output 'Numerical checks complete. Inspect saved previews before the separate next run; no training started.'
} finally { Pop-Location }
