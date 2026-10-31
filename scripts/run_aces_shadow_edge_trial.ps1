# Registered paired shadow trial (configs/experiments/aces_shadow_edge_20261024.json).
# Trains the local shadow branch with image-edge supervision, evaluates it exactly like the
# local1200 control, and writes the paired acceptance report. Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\run_aces_shadow_edge_trial.ps1
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot; Set-Location $repo
$py=Join-Path $repo '.venv\Scripts\python.exe'
$out='G:/rudra_training/aces_shadow_local_edges1200_20261024'
$control='G:/rudra_training/aces_shadow_mixed_local1200_20261023'
$base='G:/rudra_training/aces_coverage_restorer4800_20261023'
$log='G:/rudra_training/aces_shadow_local_edges1200_20261024.log'
$busy=Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match 'train_aces|resumable_training|evaluate_aces' }
if ($busy) { throw "Another RUDRA training/evaluation process is running (PID $($busy.ProcessId -join ',')); not starting." }
foreach ($need in "$control/report.json","$control/development24/report.json","$control/original_development24/report.json","$control/object_motion384/report.json","$base/step4800.pt") {
    if (-not (Test-Path $need)) { throw "Missing $need" }
}
function Step($name,[string[]]$arguments) {
    Write-Host "== $name" ; "== $name $(Get-Date -Format o)" | Add-Content $log
    & $py @arguments 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "$name failed (exit $LASTEXITCODE); see $log" }
}
.\.venv\Scripts\python.exe -m pytest tests/test_aces_shadow_context.py tests/test_aces_shadow.py -q
if ($LASTEXITCODE -ne 0) { throw 'shadow tests failed' }
if (-not (Test-Path "$out/step1200.pt")) {
    Step 'train' @('tools/train_aces_shadow_context.py','--checkpoint',"$base/step4800.pt",'--base-pilot',"$base/training_ancestry.json",'--pilot','G:/datasets/corpora/aces_shadow_mixed144_20261023/report.json','--out',$out,'--steps','1200','--include-image-edges')
}
Step 'natural development' @('tools/evaluate_aces_smoke.py','--checkpoint',"$out/step1200.pt",'--training-pilot',"$out/training_ancestry.json",'--pilot','G:/datasets/corpora/run05_aces_spatial_dev24_v2_20261023/report.json','--out',"$out/development24",'--device','cpu')
Step 'original shadow development' @('tools/evaluate_aces_smoke.py','--checkpoint',"$out/step1200.pt",'--training-pilot',"$out/training_ancestry.json",'--pilot','G:/datasets/corpora/aces_original_shadow_v2_20261023/val/report.json','--out',"$out/original_development24",'--device','cpu')
Step 'object motion 384' @('tools/evaluate_aces_temporal.py','--manifest','G:/datasets/corpora/aces_object_motion_20261023/manifest.json','--checkpoint',"$out/step1200.pt",'--out',"$out/object_motion384")
Step 'paired acceptance' @('tools/compare_aces_shadow_edge.py','--candidate',$out,'--control',$control,'--baseline',$base,'--registration','configs/experiments/aces_shadow_edge_20261024.json','--out',"$out/edge_acceptance.json")
Write-Host "Done. Report: $out/edge_acceptance.json"
