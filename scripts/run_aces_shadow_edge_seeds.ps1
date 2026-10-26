# Registered multi-seed study (configs/experiments/aces_shadow_edge_seeds_20261024.json).
# Trains and evaluates control and edge arms for seeds 1-4, then summarises all five seed pairs.
# Resumable: finished runs/evaluations are skipped. About 7 minutes per run (8 runs).
#   powershell -ExecutionPolicy Bypass -File scripts\run_aces_shadow_edge_seeds.ps1
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot; Set-Location $repo
$py=Join-Path $repo '.venv\Scripts\python.exe'
$base='G:/rudra_training/aces_coverage_restorer4800_20261023'
$log='G:/rudra_training/aces_shadow_edge_seeds_20261024.log'
$busy=Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match 'train_aces|resumable_training|evaluate_aces' }
if ($busy) { throw "Another RUDRA training/evaluation process is running (PID $($busy.ProcessId -join ',')); not starting." }
function Run($name,[string[]]$arguments) {
    $stamp="== $name $(Get-Date -Format o)"; Write-Host $stamp; Add-Content -Path $log -Value $stamp
    $previous=$ErrorActionPreference; $ErrorActionPreference='Continue'   # stderr warnings must not abort
    $output=& $py @arguments 2>&1; $code=$LASTEXITCODE
    $ErrorActionPreference=$previous
    $output | ForEach-Object { "$_" } | Add-Content -Path $log
    if ($code -ne 0) { $output | Select-Object -Last 20 | Write-Host; throw "$name failed (exit $code); see $log" }
}
& $py -m pytest tests/test_aces_shadow_context.py tests/test_aces_shadow.py -q
if ($LASTEXITCODE -ne 0) { throw 'shadow tests failed' }
$controls=@('G:/rudra_training/aces_shadow_mixed_local1200_20261023'); $edges=@('G:/rudra_training/aces_shadow_local_edges1200_20261024')
foreach ($seed in 1,2,3,4) {
    foreach ($arm in 'control','edge') {
        $out="G:/rudra_training/aces_shadow_seed_${arm}_${seed}_20261024"
        if (-not (Test-Path "$out/step1200.pt")) {
            if (Test-Path $out) { throw "$out exists without step1200.pt; inspect/rename it before rerunning" }
            $train=@('tools/train_aces_shadow_context.py','--checkpoint',"$base/step4800.pt",'--base-pilot',"$base/training_ancestry.json",'--pilot','G:/datasets/corpora/aces_shadow_mixed144_20261023/report.json','--out',$out,'--steps','1200','--seed',"$seed")
            if ($arm -eq 'edge') { $train+='--include-image-edges' }
            Run "train $arm seed $seed" $train
        }
        if (-not (Test-Path "$out/development24/report.json")) { Run "natural dev $arm $seed" @('tools/evaluate_aces_smoke.py','--checkpoint',"$out/step1200.pt",'--training-pilot',"$out/training_ancestry.json",'--pilot','G:/datasets/corpora/run05_aces_spatial_dev24_v2_20261023/report.json','--out',"$out/development24",'--device','cpu') }
        if (-not (Test-Path "$out/original_development24/report.json")) { Run "original dev $arm $seed" @('tools/evaluate_aces_smoke.py','--checkpoint',"$out/step1200.pt",'--training-pilot',"$out/training_ancestry.json",'--pilot','G:/datasets/corpora/aces_original_shadow_v2_20261023/val/report.json','--out',"$out/original_development24",'--device','cpu') }
        if (-not (Test-Path "$out/object_motion384/report.json")) { Run "motion $arm $seed" @('tools/evaluate_aces_temporal.py','--manifest','G:/datasets/corpora/aces_object_motion_20261023/manifest.json','--checkpoint',"$out/step1200.pt",'--out',"$out/object_motion384") }
        if ($arm -eq 'control') { $controls+=$out } else { $edges+=$out }
    }
}
$summary='G:/rudra_training/aces_shadow_edge_seeds_20261024_summary.json'
if (Test-Path $summary) { throw "$summary already exists" }
Run 'summary' (@('tools/summarize_aces_shadow_seeds.py','--control')+$controls+@('--edge')+$edges+@('--baseline',$base,'--registration','configs/experiments/aces_shadow_edge_seeds_20261024.json','--out',$summary,'--min-wins','4'))
Write-Host "Done. Summary: $summary"
