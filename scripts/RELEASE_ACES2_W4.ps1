<#
.SYNOPSIS
  RUDRA ACES 2 release candidate: w4 + lock_dqs050g, the arm the Crew holdout passed (7 Nov 2026).
  Does every step that needs no person, then stops for the visual review:

    1. the holdout verdict passes, for this arm (G:\rudra_training\holdout_crew_20261107\verdict.json)
    2. export w4 with the lock and its provenance (contract major 3: apps without the lock refuse it),
       build, unit tests, C++ == Python stage by stage with the lock on the real ACES 2.0 view,
       the master's round trip, LibTorch against ONNX Runtime DirectML and CPU
       (scripts\NATIVE_ACES_V2.ps1), and the A/B masters for the review
    3. the provenance record (tools\model_provenance.py): files, and the reports naming the checkpoint
    4. the lock's time per frame, from the master's provenance.json
    5. the review page and its sign-off template (tools\aces2_review_kit.py)

      powershell -ExecutionPolicy Bypass -File scripts\RELEASE_ACES2_W4.ps1
      powershell -ExecutionPolicy Bypass -File scripts\RELEASE_ACES2_W4.ps1 -Frames <real stills folder>

  Then, by hand: the review (review.json), Core ML on a Mac, tools\promote_aces2_release.py,
  scripts\PUBLISH_HF_MODELS.ps1 -AcesChannel release. It prints the exact commands.
#>
param(
    [string]$Checkpoint = "G:\rudra_training\research_bin_hinge_w4_20261102\step1600.pt",
    [string]$Frames = "G:\datasets\corpora\aces_consistent_exposures_v2_20261024\val",
    [string]$Holdout = "G:\rudra_training\holdout_crew_20261107",
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$OcioPrefix = "",
    [string]$PackageName = "RUDRA_ACES_2_w4",
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $Repo
function Say([string]$m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail([string]$m) { Write-Host "`nFAILED: $m" -ForegroundColor Red; exit 1 }
$Reg = "configs\experiments\holdout_crew_eval_20261107.json"
$Arm = "lock_dqs050g"
foreach ($p in $Checkpoint, $Python, $Reg, "$Holdout\verdict.json", "$Holdout\eval.json") { if (-not (Test-Path $p)) { Fail "missing $p" } }
git ls-files --error-unmatch $Reg *> $null
if ($LASTEXITCODE -ne 0) { Fail "$Reg is not committed" }

Say "1. The holdout verdict"
$v = Get-Content "$Holdout\verdict.json" -Raw | ConvertFrom-Json
if ($v.passes -ne $true -or $v.primary -ne "w4:$Arm") { Fail "the holdout verdict is $($v.primary) passes=$($v.passes); need w4:$Arm passing" }
Write-Host "w4:$Arm passed the Crew holdout ($($v.cases) cases, $($v.scenes) scenes)"

Say "2. Export, build, parity, backends, review masters"
$stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
New-Item -ItemType Directory -Force "tmp" | Out-Null
$prov = Join-Path $Repo "tmp\aces2_w4_provenance_$stamp.json"
$ckSha = (Get-FileHash $Checkpoint -Algorithm SHA256).Hash.ToLower()
[ordered]@{
    model = "RUDRA ACES 2"; version = "2.0"
    checkpoint = [ordered]@{ run = "research_bin_hinge_w4_20261102"; step = 1600; sha256 = $ckSha }
    lock = $Arm
    holdout = [ordered]@{
        registration = $Reg; registration_sha256 = (Get-FileHash $Reg -Algorithm SHA256).Hash.ToLower()
        verdict_sha256 = (Get-FileHash "$Holdout\verdict.json" -Algorithm SHA256).Hash.ToLower()
        result = "docs/competitive/RESULT_CREW_HOLDOUT_2026_11_07.md"
        cases = $v.cases; scenes = $v.scenes
    }
    limits = "One production (The Crew), one camera (ARRI ALEXA LogC3), single frames."
} | ConvertTo-Json -Depth 5 | ForEach-Object { [IO.File]::WriteAllText($prov, $_, (New-Object Text.UTF8Encoding $false)) }   # no BOM: Python json refuses it
$before = Get-ChildItem reports -Directory -Filter "aces_v2_*" -ErrorAction SilentlyContinue | ForEach-Object FullName
$nativeArgs = @{ Checkpoint = $Checkpoint; Frames = $Frames; Python = $Python; PackageName = $PackageName
                 LockRegistration = $Reg; LockArm = $Arm; Provenance = $prov; ReviewMasters = $true }
if ($OcioPrefix) { $nativeArgs.OcioPrefix = $OcioPrefix }
if ($SkipBuild) { $nativeArgs.SkipBuild = $true }
& (Join-Path $PSScriptRoot "NATIVE_ACES_V2.ps1") @nativeArgs
if ($LASTEXITCODE -ne 0) { Fail "scripts\NATIVE_ACES_V2.ps1 (its report folder says why)" }
$Report = Get-ChildItem reports -Directory -Filter "aces_v2_*" | Where-Object { $before -notcontains $_.FullName } |
          Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $Report) { Fail "no new reports\aces_v2_* folder" }
$Package = "dist\models\$PackageName"
$m = Get-Content "$Package\manifest.json" -Raw | ConvertFrom-Json
if ($m.contract_major -ne 3 -or $m.lock.name -ne $Arm) { Fail "the package is not major 3 with $Arm" }
Write-Host "package $Package, report $($Report.FullName)"

Say "3. Provenance record"
New-Item -ItemType Directory -Force "docs\models" | Out-Null
& $Python tools\model_provenance.py $Package --reports $Holdout "G:\rudra_training\research_bin_hinge_w4_20261102" reports `
    --out "docs\models\RUDRA_ACES_2.provenance.json"
if ($LASTEXITCODE -ne 0) { Fail "model_provenance.py exit $LASTEXITCODE (1: a file does not match; 2: no report names the checkpoint)" }

Say "4. The lock's time per frame (the native master)"
$pj = Get-Content (Join-Path $Report.FullName "master\provenance.json") -Raw | ConvertFrom-Json
$lockMs = @($pj.frames | ForEach-Object { [double]$_.lock_ms } | Sort-Object)
$allMs = @($pj.frames | ForEach-Object { [double]$_.ms } | Sort-Object)
if ($lockMs.Count) {
    "{0} frames: lock median {1:N0} ms, max {2:N0} ms; whole frame median {3:N0} ms, max {4:N0} ms" -f $lockMs.Count,
        $lockMs[[int]($lockMs.Count / 2)], $lockMs[-1], $allMs[[int]($allMs.Count / 2)], $allMs[-1] |
        Tee-Object -FilePath (Join-Path $Report.FullName "lock_timing.txt")
}

Say "5. Review page and sign-off template"
$Review = "reports\aces2_review_$stamp"
& $Python tools\aces2_review_kit.py --source $Frames --out $Review --package $Package `
    --master "analytic=$($Report.FullName)\review_analytic" --master "no_lock=$($Report.FullName)\review_nolock" `
    --master "candidate=$($Report.FullName)\master"
if ($LASTEXITCODE -ne 0) { Fail "review kit" }

Write-Host "`nEvery automatic step passed. Left for you:" -ForegroundColor Green
Write-Host @"
  a. Review: open $Review\index.html, and the EXRs in $($Report.FullName)\master on the HDR monitor
     (ACEScg; compare review_analytic and review_nolock). Fill in $Review\review.json.
  b. Core ML: on a Mac, masters with --device cpu and --device coreml, then
     python tools/check_aces_masters.py --source <frames> --master <cpu> --other <coreml> --package <package> --report coreml.json
     (or decide to release with Core ML not measured, and say so).
  c. Promote:
     $Python tools\promote_aces2_release.py --package $Package --holdout-verdict $Holdout\verdict.json ``
         --holdout-registration $Reg --native-report $($Report.FullName) ``
         --provenance docs\models\RUDRA_ACES_2.provenance.json --review $Review\review.json ``
         --coreml-report <coreml.json> --stage dist\hf\rudra-models\models\RUDRA_ACES_2
  d. Ship the app with the lock (1.2.0), then: scripts\PUBLISH_HF_MODELS.ps1 -AcesChannel release
"@
