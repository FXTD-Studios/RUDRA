<#
.SYNOPSIS
  The Crew confirmatory holdout, once (registered 7 Nov 2026): w4 + lock_dqs050g, the arm the
  lockguard_20261107 decision table chose. Verifies the seal and every file, builds the grading
  set exactly as fresh v2 was built, evaluates the arm once, prints the verdict under the
  registered rule. Run it only after the two registrations are committed.

      powershell -ExecutionPolicy Bypass -File scripts\RUN_CREW_HOLDOUT.ps1

  Resumable up to the evaluation; the evaluation itself runs once and is never repeated.
#>
[CmdletBinding()]
param(
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$Plates = "G:\holdout\rudra_holdout_crew_20261105\hdr",
    [string]$Out = "G:\rudra_training\holdout_crew_20261107"
)
$ErrorActionPreference = "Stop"
function Say($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $Python)) { Fail "no $Python" }
$E = "configs\experiments"
$Hold = "$E\holdout_crew_20261105.json"
$Grade = "$E\holdout_crew_grade_set_20261107.json"
$Eval = "$E\holdout_crew_eval_20261107.json"
$Ck = "G:\rudra_training\research_bin_hinge_w4_20261102\step1600.pt"
foreach ($p in $Hold, $Grade, $Eval, $Ck) { if (-not (Test-Path $p)) { Fail "missing $p" } }
if (Test-Path "$Out\eval.json") { Fail "$Out\eval.json exists: the holdout runs once. Read it with: $Python tools/holdout_verdict.py $Out\eval.json" }

Say "Registered before the run (committed, unchanged)"
foreach ($p in $Hold, $Grade, $Eval) {
    git ls-files --error-unmatch $p *> $null
    if ($LASTEXITCODE -ne 0) { Fail "$p is not committed: commit the registrations before running" }
    git diff --quiet HEAD -- $p
    if ($LASTEXITCODE -ne 0) { Fail "$p differs from the committed one" }
}
$pin = (Get-Content $Eval -Raw | ConvertFrom-Json).grade_set_registration_sha256
$have = (Get-FileHash $Grade -Algorithm SHA256).Hash.ToLower()
if ($pin -ne $have) { Fail "$Grade is not the one the evaluation registration pins ($pin, have $have)" }

Say "Tests"
& $Python -m pytest -q tests/test_aces_partial_tighten.py tests/test_aces_lock_guard.py tests/test_holdout_tools.py
if ($LASTEXITCODE -ne 0) { Fail "tests" }

Say "Holdout seal, files, and no leak into training"
New-Item -ItemType Directory -Force $Out | Out-Null
$roots = (Get-Content $Hold -Raw | ConvertFrom-Json).exclude_roots
& $Python tools/holdout_kit.py verify $Hold --exclude-roots @roots | Tee-Object -FilePath "$Out\verify.txt"
if ($LASTEXITCODE -ne 0) { Fail "holdout verify (see $Out\verify.txt)" }

if (-not (Test-Path "$Out\selection.json")) {
    Say "Selection (the 48 registered frames)"
    & $Python tools/holdout_selection.py $Hold --plate-primaries rec2020 --out "$Out\selection.json"
    if ($LASTEXITCODE -ne 0) { Fail "selection" }
}
if (-not (Test-Path "$Out\grade_set\report.json")) {
    Say "Grading set, as fresh v2 (CPU)"
    if (Test-Path "$Out\grade_set") { Fail "$Out\grade_set exists without report.json (a stopped build): move it aside and run again" }
    & $Python tools/prepare_fresh_fxtd_grade_v2.py --selection "$Out\selection.json" --plates $Plates --registration $Grade --out "$Out\grade_set"
    if ($LASTEXITCODE -ne 0) { Fail "grading set" }
}

Say "The evaluation, once: w4 + lock_dqs050g (CPU)"
& $Python tools/evaluate_route_arms.py --corpus "$Out\grade_set" --registration $Eval --candidate "w4=$Ck" --out "$Out\eval.json"
if ($LASTEXITCODE -ne 0) { Fail "evaluation" }

Say "Verdict (registered rule)"
& $Python tools/holdout_verdict.py "$Out\eval.json" --out "$Out\verdict.json"
$code = $LASTEXITCODE
Write-Host "`nEvidence: $Out (verify.txt, selection.json, grade_set\, eval.json, verdict.json). release_authorized stays false." -ForegroundColor Yellow
exit $code
