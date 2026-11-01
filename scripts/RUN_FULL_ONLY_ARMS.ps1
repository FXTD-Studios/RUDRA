<#
.SYNOPSIS
  Run the pre-registered full-only routing arms (configs/experiments/claude_full_only_route_20261101.json)
  on the fresh FXTD grading set v2. Inference only: no training, CPU is enough (about the time
  the 26 Oct fresh-grade run took, times the number of arms).

  The registration names the primary arm (full_only_f8) and the gates BEFORE this runs. Do not
  edit the registration after running; register a new one instead.
#>
[CmdletBinding()]
param(
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$Corpus = "G:\rudra_training\claude_release_candidate_20261026\fresh_fxtd_grade_v2_20261026",
    [string]$Checkpoint = "G:\rudra_training\partial_pcgrad_duration6400_retry1_20261025\step6400.pt",
    [string]$OutDir = "G:\rudra_training\claude_full_only_route_20261101"
)
$ErrorActionPreference = "Stop"
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
if (-not (Test-Path "$Corpus\report.json")) { Fail "no grading set at $Corpus (report.json)" }
if (-not (Test-Path $Checkpoint)) { Fail "no checkpoint at $Checkpoint" }
$want = "f73f14c7e74d5bb54d0cbf949b94365f2eeb12a3cc50ed3ea45d1ad1b163256e"
$got = (Get-FileHash $Checkpoint -Algorithm SHA256).Hash.ToLower()
if ($got -ne $want) { Fail "checkpoint is not the registered candidate ($got)" }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$out = Join-Path $OutDir ("route_arms_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
& $Python -m pytest -q tests/test_aces_full_only.py tests/test_evaluate_route_arms.py
if ($LASTEXITCODE -ne 0) { Fail "the arm's tests" }
& $Python tools/evaluate_route_arms.py --corpus $Corpus `
    --registration configs/experiments/claude_full_only_route_20261101.json `
    --candidate "d6400=$Checkpoint" --out $out
if ($LASTEXITCODE -ne 0) { Fail "evaluation" }
& $Python -c @"
import json,sys
r=json.load(open(sys.argv[1]))
print('\nprimary', r['primary'], 'numerical pass:', r['numerical_pass'])
for k,v in r['gates'].items(): print(f'  {k:18s} {\"pass\" if v else \"FAIL\"}')
print('\n arm                    grade-3     grade0     grade+3   H        ')
for n,s in r['summary'].items(): print(f'  {n:22s} {s[\"grade_-3\"]:9.3f} {s[\"grade_0\"]:9.4f} {s[\"grade_3\"]:9.4f} {s[\"H\"]:.4f}')
"@ $out
Write-Host "`nReport: $out"
Write-Host "Before any release use: the paired development gates and the seam review the registration lists, then the new holdout."
