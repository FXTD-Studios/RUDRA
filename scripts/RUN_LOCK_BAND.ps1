<#
.SYNOPSIS
  The band lock on w4 (registered 6 Nov 2026): bands 0.15, 0.25, 0.35 code. Per band:
  development gates (CUDA), old24 grades, Hunyuan48; then the fresh set for all three
  at once; then the decision table with the registered selection. Resumable.
      powershell -ExecutionPolicy Bypass -File scripts\RUN_LOCK_BAND.ps1
#>
[CmdletBinding()]
param(
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$Fresh = "G:\rudra_training\fresh_fxtd_grade_v2_rebuild_20261102",
    [string]$Out = "G:\rudra_training\lockband_20261106",
    [string[]]$Bands = @("b015", "b025", "b035")
)
$ErrorActionPreference = "Stop"
function Say($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $Python)) { Fail "no $Python" }
$E = "configs\experiments"
$src = "G:\rudra_training\research_bin_hinge_w4_20261102"
$ck = Join-Path $src "step1600.pt"; $anc = Join-Path $src "training_ancestry.json"; $rep = Join-Path $src "report.json"
foreach ($p in $ck, $anc, $rep) { if (-not (Test-Path $p)) { Fail "missing $p" } }
Say "Tests"
& $Python -m pytest -q tests/test_aces_partial_tighten.py
if ($LASTEXITCODE -ne 0) { Fail "tests" }
New-Item -ItemType Directory -Force $Out | Out-Null
foreach ($b in $Bands) {
    $root = Join-Path $Out "dev_$b"
    if (-not (Test-Path "$root\paired_development\acceptance.json")) {
        Say "Development gates, w4 + lock band $b (CUDA)"
        New-Item -ItemType Directory -Force $root | Out-Null
        Copy-Item $ck, $anc, $rep $root -Force
        if (Test-Path "$root\paired_development") { Remove-Item -Recurse -Force "$root\paired_development" }
        & $Python tools/evaluate_aces_context_motion.py --registration "$E\lockband_w4_${b}_dev_20261106.json" --root $root
        if ($LASTEXITCODE -ne 0) { Fail "development gates ($b)" }
    }
    if (-not (Test-Path "$Out\grade_$b.json")) {
        Say "Old24 grades, band $b"
        $g = Get-Content "$E\research_bin_hinge_grade_template_20261102.json" -Raw | ConvertFrom-Json
        $g.evaluation = ("$root\paired_development\old_coherent\candidate_report.json" -replace "\\", "/")
        $g.parent = ("$root\paired_development\old_coherent\parent_report.json" -replace "\\", "/")
        [IO.File]::WriteAllText("$Out\grade_registration_$b.json", ($g | ConvertTo-Json -Depth 20), (New-Object Text.UTF8Encoding $false))
        & $Python tools/audit_aces_consistent_detail_grade.py --registration "$Out\grade_registration_$b.json" --out "$Out\grade_$b.json"
        if ($LASTEXITCODE -ne 0) { Fail "grades ($b)" }
    }
    if (-not (Test-Path "$Out\video_$b.json")) {
        Say "Hunyuan48, band $b"
        $v = Get-Content "$E\lockband_w4_${b}_video_20261106.json" -Raw | ConvertFrom-Json
        $v.candidate = ($ck -replace "\\", "/"); $v.checkpoint_sha256 = (Get-FileHash $ck -Algorithm SHA256).Hash.ToLower()
        [IO.File]::WriteAllText("$Out\video_registration_$b.json", ($v | ConvertTo-Json -Depth 20), (New-Object Text.UTF8Encoding $false))
        & $Python tools/evaluate_aces_feather_video.py --registration "$Out\video_registration_$b.json" --out "$Out\video_$b.json"
        if ($LASTEXITCODE -ne 0) { Fail "video ($b)" }
    }
}
if (-not (Test-Path "$Out\fresh_w4.json")) {
    Say "Fresh FXTD v2, all bands (CPU)"
    & $Python tools/evaluate_route_arms.py --corpus $Fresh --registration "$E\lockband_fresh_20261106.json" --candidate "w4=$ck" --out "$Out\fresh_w4.json"
    if ($LASTEXITCODE -ne 0) { Fail "fresh" }
}
Say "Decision table"
& $Python tools/summarize_lock.py --out $Out --bands @Bands
