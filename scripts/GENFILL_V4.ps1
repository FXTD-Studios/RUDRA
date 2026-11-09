<#
.SYNOPSIS
  AI fill v4, region re-render (9 Nov 2026): build the v4 clips (half of them a bright region parked
  under white by a display-referred curve, the condition the BT.1886 inverse), then train the LoRA.
  Registration: configs\experiments\research_genfill_v4_20261109.json. Unattended: build then train.

    powershell -ExecutionPolicy Bypass -File scripts\GENFILL_V4.ps1              # build (resumes an interrupted one), then train
    powershell -ExecutionPolicy Bypass -File scripts\GENFILL_V4.ps1 -BuildOnly
    powershell -ExecutionPolicy Bypass -File scripts\GENFILL_V4.ps1 -SmokeTest   # 6 sources, 20 updates

  The GPU must be free. Nothing here changes an app default or releases a model.
#>
param(
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$Base = "G:\datasets\models\Wan2.1-VACE-1.3B-diffusers",
    [string]$Clips = "G:\datasets\corpora\research_genfill_clips_v4_20261109",
    [string]$Out = "G:\rudra_training\research_genfill_v4_20261109",
    [string]$InitLora = "G:\rudra_training\research_genfill_v1_20261027\rudra_genfill.safetensors",
    [int]$Workers = 8,
    [switch]$BuildOnly,
    [switch]$SmokeTest
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $Repo
$Reg = "configs\experiments\research_genfill_v4_20261109.json"
function Say([string]$m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail([string]$m) { Write-Host "`nFAILED: $m" -ForegroundColor Red; exit 1 }
function Py([string[]]$a, [string]$logFile) {
    $ErrorActionPreference = "Continue"
    if ($logFile) { & $Python @a 2>&1 | Tee-Object -FilePath $logFile | Out-Host } else { & $Python @a 2>&1 | Out-Host }
    $rc = $LASTEXITCODE; $ErrorActionPreference = "Stop"
    return $rc
}
if (-not (Test-Path $Python)) { Fail "no $Python (the repo's venv)" }
$env:HF_HOME = "G:\datasets\models\_hf_home"
if (-not (Test-Path (Join-Path $Base "transformer\config.json"))) { Fail "no base model at $Base (scripts\GENFILL.ps1 -Setup)" }
$Cuda = (& $Python -c "import torch; print(torch.cuda.is_available())").Trim()
if ($Cuda -ne "True") { Fail "no CUDA GPU visible to torch" }
if ($SmokeTest) {
    $Clips = "$Clips`_smoke"; $Out = "$Out`_smoke"
    foreach ($p in @($Clips, $Out)) { if (Test-Path $p) { Remove-Item -Recurse -Force $p } }
}

# ---- 1. the clips (recipe v4) ---------------------------------------------
if (Test-Path (Join-Path $Clips "summary.json")) {
    Write-Host "clips already built: $Clips (delete the folder to rebuild)"
} else {
    $resume = Test-Path (Join-Path $Clips "clips.jsonl")
    Say "Plan"
    $rc = Py @("tools\build_genfill_clips.py", "--out", $Clips, "--registration", $Reg, "--recipe", "v4", "--seed", "20261109", "--dry-run") ""
    if ($rc -ne 0) { Fail "plan" }
    Say "Build and encode the v4 clips ($Workers workers, VAE on cuda) -> $Clips"
    $bargs = @("tools\build_genfill_clips.py", "--out", $Clips, "--registration", $Reg, "--recipe", "v4", "--seed", "20261109",
               "--base", $Base, "--workers", $Workers, "--device", "cuda")
    if ($SmokeTest) { $bargs += @("--limit", "6", "--dev-fraction", "0.34") }
    if ($resume) { $bargs += "--resume"; Write-Host "continuing the interrupted build in $Clips" }
    $rc = Py $bargs $(if ($resume) { "$Clips.resume.log" } else { "$Clips.build.log" })
    if ($rc -ne 0) { Fail "clip build (see $Clips.build.log)" }
}
$sum = Get-Content (Join-Path $Clips "summary.json") | ConvertFrom-Json
Write-Host ("clips {0} from {1} sources, {2} scenes; commercial-clean {3}; by kind {4}" -f $sum.clips, $sum.sources_used, $sum.scenes_used, $sum.commercial_clean, ($sum.by_kind | ConvertTo-Json -Compress))
if (-not $sum.commercial_clean) { Fail "a clip without a commercial-clean licence" }
if ($BuildOnly) { exit 0 }

# ---- 2. the LoRA -----------------------------------------------------------
if (Test-Path $Out) { Fail "$Out exists: the registered run writes a new folder" }
if (-not (Test-Path $InitLora)) { Fail "no v1 LoRA to start from: $InitLora" }
$steps = if ($SmokeTest) { 20 } else { 6000 }
$every = if ($SmokeTest) { 10 } else { 1000 }
Say "Train the v4 LoRA (registered: research_genfill_v4_20261109; weight on the mask) -> $Out"
$targs = @("training\genfill_train.py", "--clips", $Clips, "--base", $Base, "--out", $Out, "--device", "cuda",
           "--steps", $steps, "--save-every", $every, "--max-hours", "14", "--registration", $Reg,
           "--weight-on", "mask", "--region-weight", "4", "--calib-weight", "0.5", "--calib-sigma", "0.6", "--over-weight", "4",
           "--init-lora", $InitLora)
if ($SmokeTest) { $targs += @("--dev-clips", "4") }
$rc = Py $targs "$Out.train.log"
if ($rc -ne 0) { Fail "training (see $Out.train.log)" }
$rep = Get-Content (Join-Path $Out "report.json") | ConvertFrom-Json
Write-Host ("`nDone: {0} updates in {1} h; LoRA {2} ({3})" -f $rep.steps_done, $rep.hours, $rep.result, $rep.result_sha256) -ForegroundColor Green
