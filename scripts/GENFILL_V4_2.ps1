<#
.SYNOPSIS
  AI fill v4.2 (14 Nov 2026): build the v4.2 clips, train the LoRA, then run the registered checks.
  Registration: configs\experiments\research_genfill_v4_2_20261114.json. Unattended: build, train, check.

    powershell -ExecutionPolicy Bypass -File scripts\GENFILL_V4_2.ps1              # all three (each resumes / skips what is done)
    powershell -ExecutionPolicy Bypass -File scripts\GENFILL_V4_2.ps1 -SmokeTest   # 6 sources, 20 updates, no checks
    powershell -ExecutionPolicy Bypass -File scripts\GENFILL_V4_2.ps1 -ChecksOnly  # the checks on a finished LoRA

  The GPU must be free. Nothing here changes an app default or releases a model.
#>
param(
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$Base = "G:\datasets\models\Wan2.1-VACE-1.3B-diffusers",
    [string]$Clips = "G:\datasets\corpora\research_genfill_clips_v4_2_20261114",
    [string]$Out = "G:\rudra_training\research_genfill_v4_2_20261114",
    [string]$InitLora = "G:\rudra_training\research_genfill_v1_20261027\rudra_genfill.safetensors",
    [int]$Workers = 8,
    [switch]$SmokeTest,
    [switch]$ChecksOnly
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $Repo
$Reg = "configs\experiments\research_genfill_v4_2_20261114.json"
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
if (-not (Test-Path (Join-Path $Base "transformer\config.json"))) { Fail "no base model at $Base (scripts\GENFILL.ps1 -Setup)" }
$Cuda = (& $Python -c "import torch; print(torch.cuda.is_available())").Trim()
if ($Cuda -ne "True") { Fail "no CUDA GPU visible to torch" }
if ($SmokeTest) {
    $Clips = "$Clips`_smoke"; $Out = "$Out`_smoke"
    foreach ($p in @($Clips, $Out)) { if (Test-Path $p) { Remove-Item -Recurse -Force $p } }
}

if (-not $ChecksOnly) {
    # ---- 1. the clips (recipe v4_2) -----------------------------------------
    if (Test-Path (Join-Path $Clips "summary.json")) {
        Write-Host "clips already built: $Clips (delete the folder to rebuild)"
    } else {
        $resume = Test-Path (Join-Path $Clips "clips.jsonl")
        Say "Plan"
        $rc = Py @("tools\build_genfill_clips.py", "--out", $Clips, "--registration", $Reg, "--recipe", "v4_2", "--seed", "20261114", "--dry-run") ""
        if ($rc -ne 0) { Fail "plan" }
        Say "Build and encode the v4.2 clips ($Workers workers, VAE on cuda) -> $Clips"
        $bargs = @("tools\build_genfill_clips.py", "--out", $Clips, "--registration", $Reg, "--recipe", "v4_2", "--seed", "20261114",
                   "--base", $Base, "--workers", $Workers, "--device", "cuda")
        if ($SmokeTest) { $bargs += @("--limit", "6", "--dev-fraction", "0.34") }
        if ($resume) { $bargs += "--resume"; Write-Host "continuing the interrupted build in $Clips" }
        $rc = Py $bargs $(if ($resume) { "$Clips.resume.log" } else { "$Clips.build.log" })
        if ($rc -ne 0) { Fail "clip build (see $Clips.build.log)" }
    }
    $sum = Get-Content (Join-Path $Clips "summary.json") | ConvertFrom-Json
    Write-Host ("clips {0} from {1} sources, {2} scenes; commercial-clean {3}; by kind {4}" -f $sum.clips, $sum.sources_used, $sum.scenes_used, $sum.commercial_clean, ($sum.by_kind | ConvertTo-Json -Compress))
    if (-not $sum.commercial_clean) { Fail "a clip without a commercial-clean licence" }
    if ($sum.recipe -ne "v4_2") { Fail "$Clips is not a v4_2 store" }

    # ---- 2. the LoRA ---------------------------------------------------------
    if (Test-Path (Join-Path $Out "report.json")) {
        Write-Host "already trained: $Out"
    } else {
        if (Test-Path $Out) { Fail "$Out exists without a report (an interrupted run): move it aside; the registered run writes a new folder" }
        if (-not (Test-Path $InitLora)) { Fail "no v1 LoRA to start from: $InitLora" }
        $steps = if ($SmokeTest) { 20 } else { 6000 }
        $every = if ($SmokeTest) { 10 } else { 1000 }
        Say "Train the v4.2 LoRA (registered: research_genfill_v4_2_20261114) -> $Out"
        $targs = @("training\genfill_train.py", "--clips", $Clips, "--base", $Base, "--out", $Out, "--device", "cuda",
                   "--steps", $steps, "--save-every", $every, "--max-hours", "14", "--registration", $Reg,
                   "--weight-on", "mask", "--region-weight", "4", "--calib-weight", "0",
                   "--structure-boost", "3", "--structure-ref", "0.10", "--temporal-weight", "1.0", "--temporal-sigma", "0.6",
                   "--seed", "20261114", "--init-lora", $InitLora)
        if ($SmokeTest) { $targs += @("--dev-clips", "4") }
        $rc = Py $targs "$Out.train.log"
        if ($rc -ne 0) { Fail "training (see $Out.train.log)" }
    }
    $rep = Get-Content (Join-Path $Out "report.json") | ConvertFrom-Json
    Write-Host ("`nTrained: {0} updates in {1} h; LoRA {2} ({3})" -f $rep.steps_done, $rep.hours, $rep.result, $rep.result_sha256) -ForegroundColor Green
    if ($SmokeTest) { Write-Host "Smoke test done (no checks)."; exit 0 }
}

# ---- 3. the registered checks ----------------------------------------------
$Lora = Join-Path $Out "rudra_genfill.safetensors"
if (-not (Test-Path $Lora)) { Fail "no trained LoRA at $Lora" }
Say "Registered checks: 09V and testwindow door gates, water and Porch reported, held-out H (gated and blind)"
$rc = Py @("tmp\09V_test\v4_gates\run_v4_2_gates.py", "--lora", $Lora) ""
if ($rc -ne 0) { Fail "checks (see tmp\09V_test\v4_gates\run_v4_2.log)" }
Get-Content "tmp\09V_test\v4_gates\v4_2_verdict.json" | Out-Host
