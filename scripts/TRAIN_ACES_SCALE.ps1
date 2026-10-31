<#
.SYNOPSIS
  Build the commercial-clean ACES 2.0 training crops from every train plate we own or that is
  CC0 (FXTD footage, Poly Haven moves, highlight/fire renders), then train the learned ACES
  model's partial-highlight route on them overnight (GPU).

    powershell -ExecutionPolicy Bypass -File scripts\TRAIN_ACES_SCALE.ps1
    powershell -ExecutionPolicy Bypass -File scripts\TRAIN_ACES_SCALE.ps1 -BuildOnly
    powershell -ExecutionPolicy Bypass -File scripts\TRAIN_ACES_SCALE.ps1 -SmokeTest   # 20 plates, 20 updates

  Registrations (written before anything ran):
    configs\experiments\claude_scale_crops_20261027.json     the data
    configs\experiments\claude_scale_partial_20261027.json   the training run and its gates
  Nothing here changes a default or releases a model. Keep the PC awake (no sleep) overnight.
#>
param(
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$Crops = "G:\datasets\corpora\claude_aces_scale_crops_20261027",
    [string]$Out = "G:\rudra_training\claude_scale_partial_20261027",
    [string]$Parent = "G:\rudra_training\partial_pcgrad_duration6400_retry1_20261025\step6400.pt",
    [int]$Workers = 8,
    [switch]$BuildOnly,
    [switch]$SmokeTest
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $Repo
function Say([string]$m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail([string]$m) { Write-Host "`nFAILED: $m" -ForegroundColor Red; exit 1 }

if (-not (Test-Path $Python)) { Fail "no $Python (the repo's venv)" }
if (-not (Test-Path $Parent)) { Fail "no parent checkpoint $Parent" }
$ParentSha = (Get-FileHash $Parent -Algorithm SHA256).Hash.ToLower()
if ($ParentSha -ne "f73f14c7e74d5bb54d0cbf949b94365f2eeb12a3cc50ed3ea45d1ad1b163256e") { Fail "parent checkpoint hash differs from the registration" }
& $Python -c "import torch, PyOpenColorIO as o; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'OCIO', o.__version__)"
if ($LASTEXITCODE -ne 0) { Fail "the venv needs torch and OpenColorIO" }
$Cuda = (& $Python -c "import torch; print(torch.cuda.is_available())").Trim()

if ($SmokeTest) {
    $Crops = "$Crops`_smoke"; $Out = "$Out`_smoke"
    foreach ($p in @($Crops, $Out)) { if (Test-Path $p) { Remove-Item -Recurse -Force $p } }
}

# ---- 1. the data -----------------------------------------------------------
if (Test-Path (Join-Path $Crops "summary.json")) {
    Write-Host "crops already built: $Crops (delete the folder to rebuild)"
} else {
    if (Test-Path $Crops) { Fail "$Crops exists without summary.json (an interrupted build): delete it, then run again" }
    Say "Build the training crops ($Workers workers) -> $Crops"
    $bargs = @("tools\build_aces_scale_crops.py", "--out", $Crops, "--registration", "configs\experiments\claude_scale_crops_20261027.json", "--workers", $Workers)
    if ($SmokeTest) { $bargs += @("--limit", "20") }
    $ErrorActionPreference = "Continue"   # Python warnings on stderr must not stop the run
    & $Python @bargs 2>&1 | Tee-Object -FilePath "$Crops.build.log"
    $rc = $LASTEXITCODE; $ErrorActionPreference = "Stop"
    if ($rc -ne 0) { Fail "crop build (see $Crops.build.log)" }
}
$sum = Get-Content (Join-Path $Crops "summary.json") | ConvertFrom-Json
Write-Host ("crops {0} from {1} plates, {2} scenes; commercial-clean {3}" -f $sum.crops, $sum.plates_used, $sum.scenes_used, $sum.commercial_clean)
if (-not $sum.commercial_clean) { Fail "a crop without a commercial-clean licence" }
if ($BuildOnly) { exit 0 }

# ---- 2. the training run ---------------------------------------------------
if (-not $SmokeTest -and $Cuda -ne "True") { Fail "no CUDA GPU visible to torch: the overnight run needs it" }
if (Test-Path $Out) { Fail "$Out exists: the registered run writes a new folder" }
$steps = if ($SmokeTest) { 20 } else { 60000 }
$every = if ($SmokeTest) { 10 } else { 2500 }
Say "Train (registered: claude_scale_partial_20261027) -> $Out"
Write-Host "stops after 9.5 hours at the next 2,500-update save, or at 60,000 updates"
$targs = @("tools\train_partial_display.py", "--checkpoint", $Parent, "--out", $Out, "--device", $(if ($Cuda -eq "True") { "cuda" } else { "cpu" }),
           "--data-root", "G:/", "--crop-store", $Crops.Replace('\', '/'),
           "--steps", $steps, "--save-every", $every, "--max-hours", "9.5",
           "--batch", "4", "--crop", "192", "--border", "64", "--aligned-crops", "--paired-motion", "--temporal-weight", "1",
           "--display-weight", "0.5", "--display-evs", "-3", "0", "3", "--relative", "--pcgrad", "--deterministic",
           "--lr", "0.0001", "--seed", "20261028")
$ErrorActionPreference = "Continue"
& $Python @targs 2>&1 | Tee-Object -FilePath "$Out.train.log"
$rc = $LASTEXITCODE; $ErrorActionPreference = "Stop"
if ($rc -ne 0) { Fail "training (see $Out.train.log)" }
$rep = Get-Content (Join-Path $Out "report.json") | ConvertFrom-Json
Write-Host ("`nDone: {0} updates, frozen parts bit-exact {1}, checkpoint {2}" -f $rep.steps_done, $rep.frozen_parent_bit_exact, $rep.checkpoint_sha256) -ForegroundColor Green
Write-Host "Tell Claude it finished: the gates and the fresh FXTD set run next."
