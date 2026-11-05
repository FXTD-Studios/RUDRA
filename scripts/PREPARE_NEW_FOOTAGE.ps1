<#
.SYNOPSIS
  New footage for the 0 EV fix (5 Nov 2026, option 1: keep the gate, fix the model).
    TRAINING  F:\RND\2017\concave  (ARRI ALEXA, LogC3 in EXR)  -> G:\datasets\corpora\rudra_new_concave_20261105
              D:\updown            (ARRI ALEXA Mini LF plates)  -> the same folder
    HOLDOUT   F:\CINEMA\The Crew   (ARRI ALEXA LogC3 plates)    -> G:\holdout\rudra_holdout_crew_20261105
  The holdout lives OUTSIDE every training root, so nothing can train on it and the
  holdout kit's leak check works. Stages, in order (each can be run alone):
    ingest    decode both to scene-linear plates (pipeline/ingest_fxtd_footage.py; long)
    register  seal The Crew's holdout (tools/holdout_kit.py) BEFORE any case is built; commit it
    dryrun    count the training pairs concave gives (tools/build_real_partial_corpus.py --dry-run)
      powershell -ExecutionPolicy Bypass -File scripts\PREPARE_NEW_FOOTAGE.ps1                 # all three
      powershell -ExecutionPolicy Bypass -File scripts\PREPARE_NEW_FOOTAGE.ps1 -Stage register
#>
[CmdletBinding()]
param(
    [ValidateSet("all", "ingest", "register", "dryrun")][string]$Stage = "all",
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$TrainOut = "G:\datasets\corpora\rudra_new_concave_20261105",
    [string]$HoldOut = "G:\holdout\rudra_holdout_crew_20261105",
    [string]$Fresh = "G:\rudra_training\fresh_fxtd_grade_v2_rebuild_20261102\report.json",
    [string]$Registration = "configs\experiments\holdout_crew_20261105.json",
    [int]$HoldoutCases = 48
)
$ErrorActionPreference = "Stop"
function Say($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $Python)) { Fail "no $Python" }
foreach ($p in "F:\RND\2017\concave", "D:\updown", "F:\CINEMA\The Crew") { if (-not (Test-Path $p)) { Fail "footage not found: $p" } }
if ($HoldOut -like "G:\datasets\corpora*" -or $HoldOut -like "G:\rudra_training*") { Fail "the holdout must live outside the training roots" }

if ($Stage -in @("all", "ingest")) {
    Say "Ingest concave and updown (training) -> $TrainOut"
    & $Python pipeline/ingest_fxtd_footage.py --out $TrainOut --sources concave updown
    if ($LASTEXITCODE -ne 0) { Fail "ingest concave and updown" }
    & $Python pipeline/ingest_fxtd_footage.py --out $TrainOut --sources concave updown --finalize --path-root $TrainOut
    if ($LASTEXITCODE -ne 0) { Fail "finalize concave" }
    Say "Ingest The Crew (holdout) -> $HoldOut"
    New-Item -ItemType Directory -Force (Split-Path $HoldOut) | Out-Null
    & $Python pipeline/ingest_fxtd_footage.py --out $HoldOut --sources thecrew
    if ($LASTEXITCODE -ne 0) { Fail "ingest The Crew" }
    & $Python pipeline/ingest_fxtd_footage.py --out $HoldOut --sources thecrew --finalize --path-root $HoldOut
    if ($LASTEXITCODE -ne 0) { Fail "finalize The Crew" }
}
if ($Stage -in @("all", "register") -and (Test-Path $Registration)) {
    Say "Holdout already registered: verifying the seal (a holdout is registered once)"
    & $Python tools/holdout_kit.py verify $Registration --exclude-roots G:\datasets\corpora G:\rudra_training
    if ($LASTEXITCODE -ne 0) { Fail "$Registration does not verify: stop and tell Claude, do not re-register" }
} elseif ($Stage -in @("all", "register")) {
    Say "The Crew's candidates (references not sensor-clipped)"
    $cand = Join-Path $HoldOut "candidates.jsonl"
    & $Python tools/new_footage_rows.py --ingest $HoldOut --for holdout --camera "ARRI ALEXA LogC3" --out $cand
    if ($LASTEXITCODE -ne 0) { Fail "no usable holdout frames" }
    Say "Register and seal the holdout ($HoldoutCases cases) before anything is built from it"
    & $Python tools/holdout_kit.py register --candidates $cand --exclude-roots G:\datasets\corpora G:\rudra_training `
        --n $HoldoutCases --strata scene_id --out $Registration
    if ($LASTEXITCODE -ne 0) { Fail "holdout registration" }
    git add $Registration
    git commit -qm "Holdout registered and sealed: The Crew, $HoldoutCases cases (before any case is built)"
    Write-Host "  committed ${Registration}: push it so the seal is on record" -ForegroundColor Green
}
if ($Stage -in @("all", "dryrun")) {
    Say "Count concave's training pairs (nothing written)"
    $rows = Join-Path $TrainOut "rows_for_corpus.jsonl"
    & $Python tools/new_footage_rows.py --ingest $TrainOut --for corpus --out $rows
    if ($LASTEXITCODE -ne 0) { Fail "no usable training frames" }
    $ex = @($Fresh)
    if (Test-Path $Registration) { $ex += $Registration }
    & $Python tools/build_real_partial_corpus.py --sources $rows --plates (Join-Path $TrainOut "hdr") --exclude @ex `
        --out G:\datasets\corpora\real_partial_v1 --dry-run
    Write-Host "`nAim for 300+ pairs from 20+ shots. Paste this whole output to Claude." -ForegroundColor Cyan
}
