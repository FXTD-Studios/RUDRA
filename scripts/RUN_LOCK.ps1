<#
.SYNOPSIS
  The 0 EV fix without new data: the per-channel lock (training/aces_partial_tighten.py
  lock_observed), registered 5 Nov 2026, on w4 (primary) and d6400 (the shipped preview).
  Per arm: paired development gates (CUDA) -> old24 grades -> Hunyuan48 -> fresh FXTD v2.
  No training. Resumable: a finished step is skipped. Then one decision table.
      powershell -ExecutionPolicy Bypass -File scripts\RUN_LOCK.ps1
#>
[CmdletBinding()]
param(
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$Fresh = "G:\rudra_training\fresh_fxtd_grade_v2_rebuild_20261102",
    [string]$Out = "G:\rudra_training\lock_20261105",
    [string[]]$Arms = @("w4", "d6400")
)
$ErrorActionPreference = "Stop"
function Say($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $Python)) { Fail "no $Python" }
$E = "configs\experiments"
$runs = @{ "w4" = @("G:\rudra_training\research_bin_hinge_w4_20261102", "step1600.pt"); "d6400" = @("G:\rudra_training\partial_pcgrad_duration6400_retry1_20261025", "step6400.pt") }
Say "Tests"
& $Python -m pytest -q tests/test_aces_partial_tighten.py
if ($LASTEXITCODE -ne 0) { Fail "tests" }
New-Item -ItemType Directory -Force $Out | Out-Null
foreach ($arm in $Arms) {
    $src = $runs[$arm][0]; $ckn = $runs[$arm][1]
    $ck = Join-Path $src $ckn; $anc = Join-Path $src "training_ancestry.json"; $rep = Join-Path $src "report.json"
    foreach ($p in $ck, $anc, $rep) { if (-not (Test-Path $p)) { Fail "missing $p" } }
    $root = Join-Path $Out "dev_$arm"
    if (-not (Test-Path "$root\paired_development\acceptance.json")) {
        Say "Development gates, $arm + lock (CUDA)"
        New-Item -ItemType Directory -Force $root | Out-Null
        Copy-Item $ck, $anc, $rep $root -Force
        if (Test-Path "$root\paired_development") { Remove-Item -Recurse -Force "$root\paired_development" }
        & $Python tools/evaluate_aces_context_motion.py --registration "$E\lock_${arm}_dev_20261105.json" --root $root
        if ($LASTEXITCODE -ne 0) { Fail "development gates ($arm)" }
    }
    if (-not (Test-Path "$Out\grade_$arm.json")) {
        Say "Old24 grades, $arm + lock"
        $g = Get-Content "$E\research_bin_hinge_grade_template_20261102.json" -Raw | ConvertFrom-Json
        $g.evaluation = ("$root\paired_development\old_coherent\candidate_report.json" -replace "\\", "/")
        $g.parent = ("$root\paired_development\old_coherent\parent_report.json" -replace "\\", "/")
        [IO.File]::WriteAllText("$Out\grade_registration_$arm.json", ($g | ConvertTo-Json -Depth 20), (New-Object Text.UTF8Encoding $false))
        & $Python tools/audit_aces_consistent_detail_grade.py --registration "$Out\grade_registration_$arm.json" --out "$Out\grade_$arm.json"
        if ($LASTEXITCODE -ne 0) { Fail "grades ($arm)" }
    }
    if (-not (Test-Path "$Out\video_$arm.json")) {
        Say "Hunyuan48, $arm + lock"
        $v = Get-Content "$E\lock_${arm}_video_20261105.json" -Raw | ConvertFrom-Json
        $v.candidate = ($ck -replace "\\", "/"); $v.checkpoint_sha256 = (Get-FileHash $ck -Algorithm SHA256).Hash.ToLower()
        [IO.File]::WriteAllText("$Out\video_registration_$arm.json", ($v | ConvertTo-Json -Depth 20), (New-Object Text.UTF8Encoding $false))
        & $Python tools/evaluate_aces_feather_video.py --registration "$Out\video_registration_$arm.json" --out "$Out\video_$arm.json"
        if ($LASTEXITCODE -ne 0) { Fail "video ($arm)" }
    }
    if (-not (Test-Path "$Out\fresh_$arm.json")) {
        Say "Fresh FXTD v2, $arm + lock (CPU)"
        & $Python tools/evaluate_route_arms.py --corpus $Fresh --registration "$E\lock_fresh_20261105.json" --candidate "${arm}=$ck" --out "$Out\fresh_$arm.json"
        if ($LASTEXITCODE -ne 0) { Fail "fresh ($arm)" }
    }
}
Say "Decision table"
& $Python tools/summarize_lock.py --out $Out --arms @Arms
