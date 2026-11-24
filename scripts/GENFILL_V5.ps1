<#
.SYNOPSIS
  AI fill v5 (22 Nov 2026): the sampling file, the v5 LoRA, then the registered checks.
  Registration: configs\experiments\research_genfill_v5_20261122.json (DRAFT until Ahmed approves it).
  The clips are already built (scripts\GENFILL_V5_DATA.ps1) and reviewed (tmp\v5_review\REVIEW_2026-11-22.md).

    powershell -ExecutionPolicy Bypass -File scripts\GENFILL_V5.ps1              # all three (each skips what is done)
    powershell -ExecutionPolicy Bypass -File scripts\GENFILL_V5.ps1 -SmokeTest   # 20 updates into *_smoke, no checks
    powershell -ExecutionPolicy Bypass -File scripts\GENFILL_V5.ps1 -ChecksOnly  # the checks on a finished LoRA

  The GPU must be free. About 14 hours of training, then about 3 hours of checks. Nothing here changes an app
  default or releases a model.
#>
param(
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$Base = "G:\datasets\models\Wan2.1-VACE-1.3B-diffusers",
    [string]$Clips = "G:\datasets\corpora\research_genfill_clips_v5_20261122",
    [string]$Scenes = "G:\datasets\sources\genfill_v5\blender\manifest.jsonl",
    [string]$Out = "G:\rudra_training\research_genfill_v5_20261122",
    [string]$InitLora = "G:\rudra_training\research_genfill_v1_20261027\rudra_genfill.safetensors",
    [switch]$SmokeTest,
    [switch]$ChecksOnly
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $Repo
$Reg = "configs\experiments\research_genfill_v5_20261122.json"
function Say([string]$m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail([string]$m) { Write-Host "`nFAILED: $m" -ForegroundColor Red; exit 1 }
function Py([string[]]$a, [string]$logFile) {
    $ErrorActionPreference = "Continue"
    if ($logFile) { & $Python @a 2>&1 | Tee-Object -FilePath $logFile | Out-Host } else { & $Python @a 2>&1 | Out-Host }
    $rc = $LASTEXITCODE; $ErrorActionPreference = "Stop"
    return $rc
}
if (-not (Test-Path $Python)) { Fail "no $Python (the repo's venv)" }
if (-not (Test-Path $Reg)) { Fail "no registration $Reg" }
$env:HF_HOME = "G:\datasets\models\_hf_home"
if (-not (Test-Path (Join-Path $Base "transformer\config.json"))) { Fail "no base model at $Base" }
$Cuda = (& $Python -c "import torch; print(torch.cuda.is_available())").Trim()
if ($Cuda -ne "True") { Fail "no CUDA GPU visible to torch" }
if ($SmokeTest) { $Out = "$Out`_smoke"; if (Test-Path $Out) { Remove-Item -Recurse -Force $Out } }

if (-not $ChecksOnly) {
    # ---- 1. the store and its sampling file -----------------------------------
    $sumFile = Join-Path $Clips "summary.json"
    if (-not (Test-Path $sumFile)) { Fail "no built v5 store at $Clips (scripts\GENFILL_V5_DATA.ps1)" }
    $sum = Get-Content $sumFile | ConvertFrom-Json
    if ($sum.recipe -ne "v5" -or -not $sum.commercial_clean -or -not $sum.encoded) { Fail "$Clips is not an encoded, commercial-clean v5 store" }
    $Sampling = Join-Path $Clips "sampling_v5.json"
    if (Test-Path $Sampling) {
        Write-Host "sampling file already written: $Sampling"
    } else {
        Say "Sampling file (tools\genfill_v5_weights.py; a few minutes) -> $Sampling"
        $rc = Py @("tools\genfill_v5_weights.py", "--clips", $Clips, "--scenes", $Scenes) (Join-Path $Clips "sampling_v5.log")
        if ($rc -ne 0) { Fail "sampling file (see $Clips\sampling_v5.log)" }
    }

    # ---- 2. the LoRA ---------------------------------------------------------
    if (Test-Path (Join-Path $Out "report.json")) {
        Write-Host "already trained: $Out"
    } else {
        if (Test-Path $Out) { Fail "$Out exists without a report (an interrupted run): move it aside; the registered run writes a new folder" }
        if (-not (Test-Path $InitLora)) { Fail "no v1 LoRA to start from: $InitLora" }
        $steps = if ($SmokeTest) { 20 } else { 6000 }
        $every = if ($SmokeTest) { 10 } else { 1000 }
        $devn = if ($SmokeTest) { 4 } else { 32 }
        Say "Train the v5 LoRA (registered: research_genfill_v5_20261122) -> $Out"
        $targs = @("training\genfill_train.py", "--clips", $Clips, "--base", $Base, "--out", $Out, "--device", "cuda",
                   "--steps", $steps, "--save-every", $every, "--max-hours", "14", "--registration", $Reg,
                   "--weight-on", "mask", "--region-weight", "4", "--calib-weight", "0",
                   "--structure-boost", "3", "--structure-ref", "0.10", "--temporal-weight", "1.0", "--temporal-sigma", "0.6",
                   "--clip-weights", $Sampling, "--dev-pick", "spread", "--dev-clips", $devn,
                   "--seed", "20261122", "--init-lora", $InitLora)
        $rc = Py $targs "$Out.train.log"
        if ($rc -ne 0) { Fail "training (see $Out.train.log)" }
    }
    $rep = Get-Content (Join-Path $Out "report.json") | ConvertFrom-Json
    Write-Host ("`nTrained: {0} updates in {1} h; LoRA {2} ({3}); sampling {4}" -f $rep.steps_done, $rep.hours, $rep.result,
        $rep.result_sha256, ($rep.sampling | ConvertTo-Json -Compress)) -ForegroundColor Green
    if ($SmokeTest) { Write-Host "Smoke test done (no checks)."; exit 0 }
}

# ---- 3. the registered checks ----------------------------------------------
$Lora = Join-Path $Out "rudra_genfill.safetensors"
if (-not (Test-Path $Lora)) { Fail "no trained LoRA at $Lora" }
Say "Registered checks: 09V gates and lamp, testwindow, Porch, water, held-out H"
$rc = Py @("tmp\09V_test\v4_gates\run_v5_gates.py", "--lora", $Lora) ""
if ($rc -ne 0) { Fail "checks (see tmp\09V_test\v4_gates\run_v5.log)" }
Get-Content "tmp\09V_test\v4_gates\v5_verdict.json" | Out-Host
Write-Host "`nDone. Tell Claude: it reads v5_verdict.json and makes the 09V crops against v4.2." -ForegroundColor Green
