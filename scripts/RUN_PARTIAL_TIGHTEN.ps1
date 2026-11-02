<#
.SYNOPSIS
  The 0 EV arm (training/aces_partial_tighten.py), registered 2 Nov 2026, on d6400 (step6400.pt):
    1. fresh FXTD v2 (rebuilt set), CPU:     tools/evaluate_route_arms.py
    2. paired development gates + grades:    tools/evaluate_aces_context_motion.py  (CUDA)
    3. Hunyuan48 video:                      tools/evaluate_aces_feather_video.py
  for the primary arm (bisect) and the control (halving), then one decision table.
  Inference only. Registrations: configs/experiments/claude_partial_tighten_*_20261102.json.
#>
[CmdletBinding()]
param(
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$RunRoot = "G:\rudra_training\partial_pcgrad_duration6400_retry1_20261025",
    [string]$Fresh = "G:\rudra_training\fresh_fxtd_grade_v2_rebuild_20261102",
    [string]$Out = "G:\rudra_training\claude_partial_tighten_20261102",
    [switch]$SkipFresh, [switch]$SkipDev, [switch]$SkipVideo
)
$ErrorActionPreference = "Stop"
function Say($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $Python)) { $Python = "python" }
$ckpt = Join-Path $RunRoot "step6400.pt"
$anc = Join-Path $RunRoot "training_ancestry.json"
$rep = Join-Path $RunRoot "report.json"
foreach ($p in $ckpt, $anc, $rep) { if (-not (Test-Path $p)) { Fail "missing $p" } }
if ((Get-FileHash $ckpt -Algorithm SHA256).Hash.ToLower() -ne "f73f14c7e74d5bb54d0cbf949b94365f2eeb12a3cc50ed3ea45d1ad1b163256e") { Fail "step6400.pt is not the registered checkpoint" }
New-Item -ItemType Directory -Force $Out | Out-Null

Say "Tests"
& $Python -m pytest -q tests/test_aces_partial_tighten.py tests/test_evaluate_route_arms.py
if ($LASTEXITCODE -ne 0) { Fail "tests" }

if (-not $SkipFresh) {
    Say "1. Fresh FXTD v2 (CPU)"
    if (-not (Test-Path "$Fresh\report.json")) { Fail "no rebuilt fresh set at $Fresh" }
    & $Python tools/evaluate_route_arms.py --corpus $Fresh --registration configs/experiments/claude_partial_tighten_fresh_20261102.json `
        --candidate "d6400=$ckpt" --out "$Out\fresh_arms.json"
    if ($LASTEXITCODE -ne 0) { Fail "fresh evaluation" }
}
foreach ($arm in "bisect", "halving") {
    if (-not $SkipDev) {
        Say "2. Development gates, $arm (CUDA)"
        $root = Join-Path $Out "dev_$arm"
        New-Item -ItemType Directory -Force $root | Out-Null
        Copy-Item $ckpt, $anc, $rep $root -Force
        $pd = Join-Path $root "paired_development"
        if (Test-Path $pd) { Remove-Item -Recurse -Force $pd }   # the evaluator reads them from its root and writes root\paired_development
        & $Python tools/evaluate_aces_context_motion.py --registration "configs/experiments/claude_partial_tighten_${arm}_dev_20261102.json" --root $root
        if ($LASTEXITCODE -ne 0) { Fail "development gates ($arm)" }
    }
    if (-not $SkipVideo) {
        Say "3. Hunyuan48 video, $arm"
        & $Python tools/evaluate_aces_feather_video.py --registration "configs/experiments/claude_partial_tighten_${arm}_video_20261102.json" --out "$Out\video_$arm.json"
        if ($LASTEXITCODE -ne 0) { Fail "video ($arm)" }
    }
}
Say "Decision table"
& $Python tools/summarize_partial_tighten.py --out $Out
