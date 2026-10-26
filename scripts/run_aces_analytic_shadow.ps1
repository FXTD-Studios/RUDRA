# Registered v1 analytic-shadow candidate (configs/experiments/aces_analytic_shadow_20261024.json).
# No training: derives the candidate from coverage4800, evaluates it and the parent's motion, writes acceptance.
#   powershell -ExecutionPolicy Bypass -File scripts\run_aces_analytic_shadow.ps1
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot; Set-Location $repo
$py=Join-Path $repo '.venv\Scripts\python.exe'
$parent='G:/rudra_training/aces_coverage_restorer4800_20261023'
$out='G:/rudra_training/aces_v1_analytic_shadow_20261024'
$ckpt="$out/candidate.pt"
$natural='G:/datasets/corpora/run05_aces_spatial_dev24_v2_20261023/report.json'
$log='G:/rudra_training/aces_v1_analytic_shadow_20261024.log'
$busy=Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match 'train_aces|resumable_training|evaluate_aces' }
if ($busy) { throw "Another RUDRA training/evaluation process is running (PID $($busy.ProcessId -join ',')); not starting." }
function Run($name,[string[]]$arguments) {
    $stamp="== $name $(Get-Date -Format o)"; Write-Host $stamp; Add-Content -Path $log -Value $stamp
    $previous=$ErrorActionPreference; $ErrorActionPreference='Continue'
    $output=& $py @arguments 2>&1; $code=$LASTEXITCODE
    $ErrorActionPreference=$previous
    $output | ForEach-Object { "$_" } | Add-Content -Path $log
    if ($code -ne 0) { $output | Select-Object -Last 20 | Write-Host; throw "$name failed (exit $code); see $log" }
}
Run 'tests' @('-m','pytest','tests/test_aces_analytic_shadow.py','tests/test_aces_analytic_candidate.py','tests/test_aces_model.py','tests/test_aces_shadow.py','tests/test_aces_shadow_context.py','-q')
if (-not (Test-Path $ckpt)) { Run 'derive candidate' @('tools/derive_aces_analytic_shadow.py','--parent',"$parent/step4800.pt",'--out',$ckpt) }
if (-not (Test-Path "$out/development24/report.json")) { Run 'natural dev24' @('tools/evaluate_aces_smoke.py','--checkpoint',$ckpt,'--training-pilot',"$parent/training_ancestry.json",'--pilot',$natural,'--out',"$out/development24",'--device','cpu') }
if (-not (Test-Path "$out/original_development24/report.json")) { Run 'original shadow dev24' @('tools/evaluate_aces_smoke.py','--checkpoint',$ckpt,'--training-pilot',"$parent/training_ancestry.json",'--pilot','G:/datasets/corpora/aces_original_shadow_v2_20261023/val/report.json','--out',"$out/original_development24",'--device','cpu') }
if (-not (Test-Path "$out/object_motion384/report.json")) { Run 'candidate motion 384' @('tools/evaluate_aces_temporal.py','--manifest','G:/datasets/corpora/aces_object_motion_20261023/manifest.json','--checkpoint',$ckpt,'--out',"$out/object_motion384") }
if (-not (Test-Path "$parent/object_motion384/report.json")) { Run 'parent motion 384' @('tools/evaluate_aces_temporal.py','--manifest','G:/datasets/corpora/aces_object_motion_20261023/manifest.json','--checkpoint',"$parent/step4800.pt",'--out',"$parent/object_motion384") }
if (-not (Test-Path "$out/grade_ramps/report.json")) { Run 'grade ramps' @('tools/audit_aces_grade_ramp.py','--checkpoint',$ckpt,'--out',"$out/grade_ramps") }
Run 'acceptance' @('tools/compare_aces_analytic_shadow.py','--candidate',$out,'--parent',$parent,'--pilot',$natural,'--registration','configs/experiments/aces_analytic_shadow_20261024.json','--out',"$out/acceptance.json")
Write-Host "Done. Report: $out/acceptance.json"
