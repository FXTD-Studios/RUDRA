<#
.SYNOPSIS
  The 0 EV training kit (registered 2 Nov 2026): two arms from d6400, 1600 steps each on the
  4080 (the 25 Oct continuation recipe + the bin hinge, weights 4 and 16), then for each arm:
    paired development gates -> old coherent grades -> Hunyuan48 video -> fresh FXTD v2,
  then one decision table (tools/summarize_bin_hinge.py).
  Training needs CUDA and is the long part. Re-running skips what already finished.
#>
[CmdletBinding()]
param(
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$ParentRoot = "G:\rudra_training\partial_pcgrad_duration6400_retry1_20261025",
    [string]$DataRoot = "G:/",
    [string]$Fresh = "G:\rudra_training\fresh_fxtd_grade_v2_rebuild_20261102",
    [string[]]$Arms = @("w4", "w16"),
    [switch]$SkipTrain, [switch]$SkipEval
)
$ErrorActionPreference = "Stop"
function Say($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $Python)) { $Python = "python" }
$parent = Join-Path $ParentRoot "step6400.pt"
$anc = Join-Path $ParentRoot "training_ancestry.json"
foreach ($p in $parent, $anc) { if (-not (Test-Path $p)) { Fail "missing $p" } }
if ((Get-FileHash $parent -Algorithm SHA256).Hash.ToLower() -ne "f73f14c7e74d5bb54d0cbf949b94365f2eeb12a3cc50ed3ea45d1ad1b163256e") { Fail "the parent is not d6400" }
$E = "configs\experiments"

Say "Tests"
& $Python -m pytest -q tests/test_aces_bin_hinge.py tests/test_evaluate_route_arms.py
if ($LASTEXITCODE -ne 0) { Fail "tests" }

foreach ($arm in $Arms) {
    $reg = Get-Content "$E\research_bin_hinge_${arm}_20261102.json" -Raw | ConvertFrom-Json
    $out = $reg.out -replace "/", "\"
    $ck = Join-Path $out "step1600.pt"
    if (-not $SkipTrain -and -not (Test-Path $ck)) {
        if (Test-Path $out) { Fail "$out exists without step1600.pt (a stopped run): move it aside, then run again" }
        Say "Train $arm (bin weight $($reg.bin_weight), 1600 steps, CUDA)"
        & $Python tools/train_partial_display.py --checkpoint $parent --out $out --device cuda --data-root $DataRoot `
            --steps 1600 --batch 4 --crop 192 --border 64 --aligned-crops --paired-motion --temporal-weight 1 `
            --display-weight 0.5 --display-evs -3 0 3 --display-ev-weights 1 1 1 --relative --pcgrad --deterministic `
            --lr 0.0001 --seed $reg.seed --bin-weight $reg.bin_weight --bin-tol-codes $reg.bin_tol_codes
        if ($LASTEXITCODE -ne 0) { Fail "training $arm" }
    }
    if (-not (Test-Path $ck)) { Fail "no $ck" }
    Copy-Item $anc $out -Force   # the parent's ancestry, as complete_partial_duration1600.py does
    if ($SkipEval) { continue }
    $sha = (Get-FileHash $ck -Algorithm SHA256).Hash.ToLower()

    if (-not (Test-Path "$out\paired_development\acceptance.json")) {
        if (Test-Path "$out\paired_development") { Remove-Item -Recurse -Force "$out\paired_development" }
        Say "Paired development gates, $arm (CUDA)"
        & $Python tools/evaluate_aces_context_motion.py --registration "$E\research_bin_hinge_dev_20261102.json" --root $out
        if ($LASTEXITCODE -ne 0) { Fail "development gates ($arm)" }
    }
    if (-not (Test-Path "$out\grade.json")) {
        Say "Old coherent grades, $arm"
        $g = Get-Content "$E\research_bin_hinge_grade_template_20261102.json" -Raw | ConvertFrom-Json
        $g.evaluation = ("$out\paired_development\old_coherent\candidate_report.json" -replace "\\", "/")
        $g.parent = ("$out\paired_development\old_coherent\parent_report.json" -replace "\\", "/")
        [IO.File]::WriteAllText("$out\grade_registration.json", ($g | ConvertTo-Json -Depth 20), (New-Object Text.UTF8Encoding $false))
        & $Python tools/audit_aces_consistent_detail_grade.py --registration "$out\grade_registration.json" --out "$out\grade.json"
        if ($LASTEXITCODE -ne 0) { Fail "grades ($arm)" }
    }
    if (-not (Test-Path "$out\video.json")) {
        Say "Hunyuan48 video, $arm"
        $v = Get-Content "$E\research_bin_hinge_video_template_20261102.json" -Raw | ConvertFrom-Json
        $v.candidate = ($ck -replace "\\", "/"); $v.checkpoint_sha256 = $sha
        [IO.File]::WriteAllText("$out\video_registration.json", ($v | ConvertTo-Json -Depth 20), (New-Object Text.UTF8Encoding $false))
        & $Python tools/evaluate_aces_feather_video.py --registration "$out\video_registration.json" --out "$out\video.json"
        if ($LASTEXITCODE -ne 0) { Fail "video ($arm)" }
    }
    if (-not (Test-Path "$out\fresh.json")) {
        Say "Fresh FXTD v2, $arm (CPU)"
        & $Python tools/evaluate_route_arms.py --corpus $Fresh --registration "$E\research_bin_hinge_fresh_20261102.json" `
            --candidate "${arm}=$ck" --out "$out\fresh.json"
        if ($LASTEXITCODE -ne 0) { Fail "fresh ($arm)" }
    }
}
Say "Decision table"
& $Python tools/summarize_bin_hinge.py --arms @Arms
