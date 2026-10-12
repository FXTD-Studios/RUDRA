<#
  v10 preparation (docs/MODEL_PLAN_V10.md): the residual on the Source picker's
  curve, the crushed hold, live-action validation, bench-like selection, the
  v10 gate and its run script. Commits and pushes `native`. Then:

    powershell -ExecutionPolicy Bypass -File scripts\run_v10_2026_10_13.ps1 -Stage smoke

    .\scripts\release_v10_prep_2026-10-13.ps1
#>
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
function Invoke-Git { & git @args; if ($LASTEXITCODE -ne 0) { Fail "git $args" } }
if (Test-Path .git\index.lock) { Fail ".git\index.lock exists: move it to .git\_stale_locks\ first (AGENTS.md)" }
if ((git rev-parse --abbrev-ref HEAD) -ne "native") { Fail "not on the native branch" }

$coauthor = "Co-authored-by: FXTD Studios Research <research@fxtdstudios.com>"
$commits = @(
    @{ msg = "v10: the residual on the Source picker's curve (--baseline-source; rudra/source_baseline.py, the app's knots, held to native to 1e-5), each sample's curve from effective_baseline_curve (BT.1886 for real and degraded, a rendered sample's own, half on BT.1886 anyway); --crushed-hold charges changes to crushed blacks; bench-like selection (--best-metric bench_gain: full-frame PU21 on real and rendered val, clipped error at +1/+2 EV, crushed error, against the picker's baseline); live-action real validation (pipeline/build_v10_manifest.py); export_bench_pairs --baseline-curve / --picker-baseline; the v10 gate (training/cp10_verdicts.py: real above BT.1886, clip rows above the clip fill, crushed not worse) and scripts/run_v10_2026_10_13.ps1; 9 tests";
       paths = @("rudra/source_baseline.py", "rudra/sdr2hdr.py", "rudra/inference.py", "training/sdr2hdr_dataset.py",
                 "training/train_sdr2hdr.py", "training/bench_val.py", "training/cp10_verdicts.py", "training/export_bench_pairs.py",
                 "pipeline/build_v10_manifest.py", "tests/test_v10_2026_10_13.py", "tests/fixtures/source_curve_knots_native.json",
                 "scripts/run_v10_2026_10_13.ps1") },
    @{ msg = "Notes: the model plan after 1.0 (docs/MODEL_PLAN_V10.md): why v4 to v9 failed, what v10 changes, its gate, what the app needs if it passes, and what follows";
       paths = @("docs/MODEL_PLAN_V10.md", "STATUS.md", "scripts/release_v10_prep_2026-10-13.ps1") }
)
foreach ($c in $commits) {
    if (@(git status --porcelain -- $c.paths).Count -eq 0) { Write-Host "nothing to commit for: $($c.msg.Substring(0, 40))..."; continue }
    Invoke-Git add -- $c.paths
    Invoke-Git commit -m $c.msg -m $coauthor -- $c.paths
}
$left = @(git status --porcelain --untracked-files=no)
if ($left.Count) { Write-Host "Still modified, not committed (left for you):" -ForegroundColor Yellow; $left | Write-Host }
$credits = (git log origin/native..HEAD --format=%B) -join "`n"
if ($credits -match '(?i)anthropic\.com|co-authored-by:\s*claude|claude-session') { Fail "a commit to push credits Claude; amend it first (AGENTS.md)" }
Invoke-Git push origin native
Write-Host "native pushed. Next: powershell -ExecutionPolicy Bypass -File scripts\run_v10_2026_10_13.ps1 -Stage smoke" -ForegroundColor Green
