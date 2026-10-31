[CmdletBinding()]
param([ValidateSet('Status','Start','Prepare')][string]$Action='Status')
$ErrorActionPreference='Stop'
$repoRoot=Split-Path -Parent $PSScriptRoot
$batchRoot='G:\datasets\corpora\rudra_native1080_expansion_20261019_v3'
$pythonExe=Join-Path $repoRoot '.venv\Scripts\python.exe'
$blenderExe='C:\Program Files\Blender Foundation\Blender 5.0\blender.exe'
if ($Action -eq 'Status') {
    $readyPath=Join-Path $batchRoot 'ready.json'
    if (Test-Path -LiteralPath $readyPath) {
        Get-Content -LiteralPath $readyPath -Raw | ConvertFrom-Json | Select-Object stage,rows,by_split,release_authorized | ConvertTo-Json
        Write-Output 'Recorded readiness is complete. Use train_next_hdr_commercial.ps1 -Action Check to revalidate it.'
        return
    }
    foreach ($name in @('workflow_state.json','pair_progress.json','preparation_summary.json','ready.json')) {
        $path=Join-Path $batchRoot $name
        if (Test-Path -LiteralPath $path) {
            if ($name -eq 'ready.json') { Get-Content -LiteralPath $path -Raw | ConvertFrom-Json | Select-Object stage,rows,by_split,release_authorized | ConvertTo-Json }
            else { Get-Content -LiteralPath $path }
        }
    }
    [ordered]@{rendered_stills=@(Get-ChildItem -LiteralPath (Join-Path $batchRoot 'render_records') -Filter '*.json' -ErrorAction SilentlyContinue).Count;target_stills=8;training_started=$false} | ConvertTo-Json
    foreach ($name in @('prepare_errors.log','prepare_console.log','panorama_console.log','render_console.log')) {
        $path=Join-Path $batchRoot $name
        if (Test-Path -LiteralPath $path) { Get-Content -LiteralPath $path -Tail 3 }
    }
    return
}
if (Test-Path -LiteralPath (Join-Path $batchRoot 'ready.json')) { throw 'Candidate already finalized. Use Status or the training Check; do not overwrite reviewed media.' }
$active=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(python|blender)\.exe$' -and $_.CommandLine -match '(prepare_next_hdr_training|render_next_hdr_scenes)\.py' })
if ($active.Count -gt 0) { throw 'This preparation is active. Use -Action Status.' }
New-Item -ItemType Directory -Path $batchRoot -Force | Out-Null
if ($Action -eq 'Start') {
    $arguments=@('-NoProfile','-ExecutionPolicy','Bypass','-File',('"'+$PSCommandPath+'"'),'-Action','Prepare')
    $child=Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $batchRoot 'prepare_console.log') -RedirectStandardError (Join-Path $batchRoot 'prepare_errors.log')
    Write-Output "Preparation started: PID $($child.Id). No training will start."
    return
}
$lockPath=Join-Path $batchRoot 'prepare.lock'
$lockStream=[System.IO.File]::Open($lockPath,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None)
function Set-Stage([string]$stage) {
    [ordered]@{stage=$stage;controller_pid=$PID;updated_utc=[DateTime]::UtcNow.ToString('o');training_started=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batchRoot 'workflow_state.json') -Encoding utf8
}
Push-Location -LiteralPath $repoRoot
try {
    Set-Stage 'preparing_panorama_pairs'
    & $pythonExe -u pipeline\prepare_next_hdr_training.py plan
    if ($LASTEXITCODE -ne 0) { throw 'Plan failed' }
    & $pythonExe -u pipeline\prepare_next_hdr_training.py panoramas
    if ($LASTEXITCODE -ne 0) { throw 'Panorama preparation failed' }
    Set-Stage 'rendering_native1080_stills'
    & $blenderExe --background --factory-startup --disable-autoexec --threads 8 --python-exit-code 1 --python pipeline\render_next_hdr_scenes.py -- --plan (Join-Path $batchRoot 'plan.json') --out $batchRoot
    if ($LASTEXITCODE -ne 0) { throw 'Render failed; completed records retained' }
    Set-Stage 'converting_and_verifying'
    & $pythonExe -u pipeline\prepare_next_hdr_training.py convert
    if ($LASTEXITCODE -ne 0) { throw 'Render conversion failed' }
    & $pythonExe -u pipeline\prepare_next_hdr_training.py verify
    if ($LASTEXITCODE -ne 0) { throw 'Dataset checks failed' }
    Set-Stage 'numeric_checks_complete_visual_review_pending'
    Write-Output 'Numerical checks complete. Inspect actual images and record review before finalizing. No training started.'
} catch { Set-Stage 'failed'; throw }
finally { Pop-Location; $lockStream.Dispose(); Remove-Item -LiteralPath $lockPath }
