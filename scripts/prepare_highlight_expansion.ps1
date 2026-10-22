param(
    [ValidateSet('Status','Prepare')][string]$Action = 'Status'
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$batchRoot = 'G:\datasets\corpora\rudra_highlight_expansion_20261018_v1'
$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
$blenderExe = 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe'
if ($Action -eq 'Status') {
    $summaryPath = Join-Path $batchRoot 'preparation_summary.json'
    if (Test-Path -LiteralPath $summaryPath) {
        Get-Content -LiteralPath $summaryPath
    } else {
        $renderCount = @(Get-ChildItem -LiteralPath (Join-Path $batchRoot 'render_records') -Filter '*.json' -ErrorAction SilentlyContinue).Count
        [ordered]@{ stage='preparing'; rendered_cg_frames=$renderCount; target_cg_frames=32; training_started=$false } | ConvertTo-Json
    }
    $reviewPath = Join-Path $batchRoot 'review_status.json'
    if (Test-Path -LiteralPath $reviewPath) {
        $savedReview = Get-Content -LiteralPath $reviewPath -Raw | ConvertFrom-Json
        $manifestPath = Join-Path $batchRoot 'combined_candidate_manifest.jsonl'
        if ((Test-Path -LiteralPath $manifestPath) -and ((Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant() -eq $savedReview.manifest_sha256)) {
            Get-Content -LiteralPath $reviewPath
        } else { Write-Output 'Saved visual review is stale or the candidate manifest is missing.' }
    }
    return
}

# Prevent duplicate preparation jobs. Existing training may continue: this
# workflow uses a bounded CPU renderer and never launches or edits a trainer.
$other = @(Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and $_.Name -match '^(python|blender)\.exe$' -and
    $_.CommandLine -match '(prepare_highlight_expansion|render_highlight_scenes|verify_highlight_expansion)\.py'
})
if ($other.Count -gt 0) { throw 'Highlight preparation is already running. Use -Action Status.' }
if (!(Test-Path -LiteralPath $pythonExe) -or !(Test-Path -LiteralPath $blenderExe)) { throw 'Expected local Python or Blender executable is missing.' }
Push-Location -LiteralPath $repoRoot
try {
    & $pythonExe -u 'pipeline\prepare_highlight_expansion.py' plan
    if ($LASTEXITCODE -ne 0) { throw 'Plan failed.' }
    & $pythonExe -u 'pipeline\prepare_highlight_expansion.py' panoramas
    if ($LASTEXITCODE -ne 0) { throw 'Panorama preparation failed.' }
    & $blenderExe --background --factory-startup --disable-autoexec --threads 2 --python (Join-Path $repoRoot 'pipeline\render_highlight_scenes.py') -- --plan (Join-Path $batchRoot 'plan.json') --out $batchRoot
    if ($LASTEXITCODE -ne 0) { throw 'Synthetic render failed.' }
    & $pythonExe -u 'pipeline\prepare_highlight_expansion.py' convert
    if ($LASTEXITCODE -ne 0) { throw 'Pair conversion failed.' }
    & $pythonExe -u 'pipeline\prepare_highlight_expansion.py' finalize
    if ($LASTEXITCODE -ne 0) { throw 'Manifest assembly failed.' }
    & $pythonExe -u 'pipeline\verify_highlight_expansion.py'
    if ($LASTEXITCODE -ne 0) { throw 'Dataset verification failed.' }
    Write-Output 'Numeric verification complete. Review saved previews and limitations before a separate future training run. No training was started.'
} finally { Pop-Location }
