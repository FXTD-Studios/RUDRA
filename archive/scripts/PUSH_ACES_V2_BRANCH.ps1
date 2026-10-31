<#
.SYNOPSIS
  Put all uncommitted ACES 2.0 model, native v2 and tooling work on a NEW branch
  `aces-v2` and push it to GitHub. Never touches main, never tags, never force-pushes.

    powershell -ExecutionPolicy Bypass -File scripts\PUSH_ACES_V2_BRANCH.ps1
    powershell -ExecutionPolicy Bypass -File scripts\PUSH_ACES_V2_BRANCH.ps1 -DryRun

  Left out on purpose: result.json (a code-review report, 417 KB; it quotes a
  third-party web page's embedded Google key). Everything git ignores stays ignored.
#>
param([switch]$DryRun, [string]$Branch = "aces-v2")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)   # the repo this script sits in
function Fail([string]$m) { Write-Host "`nFAILED: $m" -ForegroundColor Red; exit 1 }

if (git branch --list $Branch) { Fail "branch $Branch already exists locally" }
if (git ls-remote --heads origin $Branch) { Fail "branch $Branch already exists on GitHub" }
$from = git rev-parse --abbrev-ref HEAD
Write-Host "from $from ($(git rev-parse --short HEAD)) -> new branch $Branch"

# Anything over 20 MB must not go in by accident.
$big = git status --porcelain --untracked-files=all | ForEach-Object { $_.Substring(3).Trim('"') } |
       Where-Object { (Test-Path $_ -PathType Leaf) -and ((Get-Item $_).Length -gt 20MB) }
if ($big) { $big | ForEach-Object { Write-Host "  large: $_" }; Fail "large files would be committed; add them to .gitignore first" }

git switch -c $Branch
if ($LASTEXITCODE -ne 0) { Fail "git switch -c" }
git add -A
git reset -q -- result.json
$staged = (git diff --cached --name-only | Measure-Object).Count
Write-Host "$staged files staged"
git diff --cached --stat | Select-Object -Last 1

if ($DryRun) {
    git reset -q
    git switch $from
    git branch -D $Branch | Out-Null
    Write-Host "`nDry run: nothing committed, back on $from." -ForegroundColor Yellow
    exit 0
}

$msg = @"
ACES 2.0 model, native v2 staged pipeline and tooling (work since 21 Oct)

- training/aces_*: ACES 2.0 contract model (v7 analytic shadow, v8 partial highlights,
  v9 fully clipped), clipped-region smoothing (hybrid64), colour constraint, display projection (rejected)
- tools/: exporters (rudra_aces_export --clipped-smoothing, export_native_v2 TorchScript+ONNX),
  train_partial_display, check_full_support_unchanged, provenance_archive, audits and evaluators
- native/: core/aces_v2 (+manifest major 2), infer OCIO 2.5.2 view, LibTorch and ONNX Runtime G1,
  engine/aces_v2_job, CLI aces-v2, tests/test_aces_v2 (8 tests), tools/aces_v2_parity
- configs/experiments: registrations incl. claude_partial_tolerance_20261025 (0 EV RMS +0.1 code)
- scripts/NATIVE_ACES_V2.ps1, docs/THIRD_PARTY_LICENSES.md (draft), docs/RUDRA_MODEL_STATUS_2026-10-25.md

Research branch: no default changed, no model released, release_authorized false everywhere.

"@
$tmp = New-TemporaryFile
Set-Content -Path $tmp -Value $msg -Encoding UTF8
git commit -q -F $tmp
Remove-Item $tmp
if ($LASTEXITCODE -ne 0) { Fail "git commit" }
git push -u origin $Branch
if ($LASTEXITCODE -ne 0) { Fail "git push (check your GitHub sign-in)" }
Write-Host "`nPushed $Branch ($(git rev-parse --short HEAD)). main is unchanged." -ForegroundColor Green
Write-Host "Open a pull request only when you want review: https://github.com/FXTD-Studios/RUDRA/compare/$Branch"
