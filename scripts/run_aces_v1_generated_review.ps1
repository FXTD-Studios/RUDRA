# Registered generated-clip review of the v1 candidate (configs/experiments/aces_v1_generated_review_20261024.json).
# Runs the real export path on 18 generated clips (48 frames each), then summarises. Resumable per clip.
#   powershell -ExecutionPolicy Bypass -File scripts\run_aces_v1_generated_review.ps1
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot; Set-Location $repo
$py=Join-Path $repo '.venv\Scripts\python.exe'
$ckpt='G:/rudra_training/aces_v1_analytic_shadow_20261024/candidate.pt'
$root='G:/rudra_training/aces_v1_generated_review_20261024'
$log="$root.log"
$busy=Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match 'train_aces|resumable_training|evaluate_aces|rudra_aces_export' }
if ($busy) { throw "Another RUDRA process is running (PID $($busy.ProcessId -join ',')); not starting." }
function Run($name,[string[]]$arguments) {
    $stamp="== $name $(Get-Date -Format o)"; Write-Host $stamp; Add-Content -Path $log -Value $stamp
    $previous=$ErrorActionPreference; $ErrorActionPreference='Continue'
    $output=& $py @arguments 2>&1; $code=$LASTEXITCODE
    $ErrorActionPreference=$previous
    $output | ForEach-Object { "$_" } | Add-Content -Path $log
    if ($code -ne 0) { $output | Select-Object -Last 20 | Write-Host; throw "$name failed (exit $code); see $log" }
}
Run 'tests' @('-m','pytest','tests/test_rudra_aces_export.py','tests/test_aces_analytic_shadow.py','-q')
New-Item -ItemType Directory -Force -Path $root | Out-Null
$clips=Get-ChildItem 'G:/rudra_training/generative_review_20261022/inputs','G:/rudra_training/generative_review_20261022/inputs_b' -Filter *.mp4 | Sort-Object Name
foreach ($clip in $clips) {
    $out="$root/$($clip.BaseName)"
    if (Test-Path "$out/report.json") { continue }
    if (Test-Path $out) { throw "$out exists without report.json; inspect/rename it before rerunning" }
    Run "export $($clip.Name)" @('tools/rudra_aces_export.py',$clip.FullName,'--checkpoint',$ckpt,'--out',$out,'--video','--max-frames','48','--exr-every','12','--matrix','bt709')
}
$summary="$root/summary.json"
if (Test-Path $summary) { Remove-Item $summary }
Run 'summary' @('tools/summarize_aces_generated.py','--root',$root,'--out',$summary)
Write-Host "Done. Summary: $summary"
