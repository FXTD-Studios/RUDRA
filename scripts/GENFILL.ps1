<#
.SYNOPSIS
  RUDRA layer 3 (generative fill): set up Wan2.1-VACE-1.3B, build the training clips, train the
  LoRA, and fill a master. Registration: configs\experiments\claude_genfill_v1_20261027.json

    powershell -ExecutionPolicy Bypass -File scripts\GENFILL.ps1 -Setup        # packages + the base model (~25 GB to G:)
    powershell -ExecutionPolicy Bypass -File scripts\GENFILL.ps1 -SmokeTest    # 6 sources, 20 updates, fill a test master
    powershell -ExecutionPolicy Bypass -File scripts\GENFILL.ps1               # build every clip, then train (<= 9.5 h)
    powershell -ExecutionPolicy Bypass -File scripts\GENFILL.ps1 -BuildOnly
    powershell -ExecutionPolicy Bypass -File scripts\GENFILL.ps1 -Fill -Master <EXR folder> [-Lora <file>]
    powershell -ExecutionPolicy Bypass -File scripts\GENFILL.ps1 -Brightness    # v3: build the brightness set, train the estimator (CPU)
    powershell -ExecutionPolicy Bypass -File scripts\GENFILL.ps1 -Evaluate      # the registered gates (v3, held-out scenes)

  The GPU must be free: do not run the build or the training while another training run uses it.
  Nothing here changes an app default or releases a model.
#>
param(
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$Base = "G:\datasets\models\Wan2.1-VACE-1.3B-diffusers",
    [string]$Clips = "G:\datasets\corpora\claude_genfill_clips_v2_20261029",
    [string]$Out = "G:\rudra_training\claude_genfill_v2_20261029",
    [string]$InitLora = "G:\rudra_training\claude_genfill_v1_20261027\rudra_genfill.safetensors",
    [string]$V3 = "G:\rudra_training\claude_genfill_v3_20261030",
    [string]$BSet = "G:\datasets\corpora\claude_brightness_set_20261030",
    [string]$Master = "",
    [string]$Lora = "",
    [int]$Workers = 8,
    [switch]$Setup,
    [switch]$SmokeTest,
    [switch]$BuildOnly,
    [switch]$Fill,
    [switch]$Evaluate,
    [switch]$Brightness
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $Repo
$Reg = "configs\experiments\claude_genfill_v2_20261029.json"
function Say([string]$m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail([string]$m) { Write-Host "`nFAILED: $m" -ForegroundColor Red; exit 1 }
function Py([string[]]$a, [string]$logFile) {
    $ErrorActionPreference = "Continue"   # Python warnings on stderr must not stop the run
    if ($logFile) { & $Python @a 2>&1 | Tee-Object -FilePath $logFile | Out-Host } else { & $Python @a 2>&1 | Out-Host }
    $rc = $LASTEXITCODE; $ErrorActionPreference = "Stop"
    return $rc
}
if (-not (Test-Path $Python)) { Fail "no $Python (the repo's venv)" }
$env:HF_HOME = "G:\datasets\models\_hf_home"          # never fill C: or D: with model caches

# ---- setup -----------------------------------------------------------------
if ($Setup) {
    Say "Packages (diffusers, transformers, peft, accelerate, OpenEXR, safetensors, sentencepiece)"
    & $Python -m pip install "diffusers>=0.35" "transformers>=4.45" "peft>=0.13" "accelerate>=1.0" "OpenEXR>=3.2" "safetensors>=0.4" "sentencepiece" "huggingface-hub>=0.25" "ftfy"
    if ($LASTEXITCODE -ne 0) { Fail "pip install" }
    Say "Base model Wan-AI/Wan2.1-VACE-1.3B-diffusers (Apache-2.0) -> $Base"
    & $Python -c "from huggingface_hub import snapshot_download as d; d('Wan-AI/Wan2.1-VACE-1.3B-diffusers', local_dir=r'$Base')"
    if ($LASTEXITCODE -ne 0) { Fail "model download" }
    & $Python -c "import json,sys; c=json.load(open(r'$Base\transformer\config.json')); print('transformer', c.get('_class_name'), 'layers', c.get('num_layers'), 'vace', c.get('vace_layers'))"
    exit 0
}

& $Python -c "import torch, diffusers, peft, OpenEXR, PyOpenColorIO as o; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'diffusers', diffusers.__version__, 'peft', peft.__version__, 'OCIO', o.__version__)"
if ($LASTEXITCODE -ne 0) { Fail "run with -Setup first (the venv needs diffusers, peft, OpenEXR)" }
if (-not (Test-Path (Join-Path $Base "transformer\config.json"))) { Fail "no base model at $($Base); run with -Setup" }
$Cuda = (& $Python -c "import torch; print(torch.cuda.is_available())").Trim()
$Dev = if ($Cuda -eq "True") { "cuda" } else { "cpu" }

# ---- fill a master ---------------------------------------------------------
if ($Fill) {
    if (-not $Master) { Fail "-Fill needs -Master <EXR folder or file>" }
    if (-not $Lora) { $Lora = Join-Path $V3 "rudra_genfill.safetensors" }
    if (-not (Test-Path $Lora)) { Fail "no LoRA $Lora" }
    $dest = "$Master`_aifill"; $k = 2
    while (Test-Path $dest) { $dest = "$Master`_aifill_$k"; $k++ }
    Say "Fill $Master -> $dest"
    $rc = Py @("-m", "rudra.genfill.run", "--master", $Master, "--out", $dest, "--base", $Base, "--lora", $Lora, "--device", $Dev) ""
    if ($rc -ne 0) { Fail "fill" }
    exit 0
}

# ---- v3: the brightness estimator (shape from the v1 LoRA, brightness from here) ----
$Reg3 = "configs\experiments\claude_genfill_v3_20261030.json"
if ($Brightness) {
    if (Test-Path (Join-Path $V3 "brightness.json")) { Fail "$V3\brightness.json exists: the registered run writes it once" }
    if (-not (Test-Path (Join-Path $BSet "summary.json"))) {
        if (Test-Path $BSet) { Fail "$BSet exists without summary.json (interrupted): delete it, then run again" }
        Say "Build the brightness set ($Workers workers, CPU) -> $BSet"
        $rc = Py @("tools\build_brightness_set.py", "--out", $BSet, "--registration", $Reg3, "--workers", $Workers) "$BSet.build.log"
        if ($rc -ne 0) { Fail "brightness set (see $BSet.build.log)" }
    }
    New-Item -ItemType Directory -Force -Path $V3 | Out-Null
    Say "Train the estimator -> $V3\brightness.json"
    $rc = Py @("tools\train_brightness.py", "--set", $BSet, "--out", (Join-Path $V3 "brightness.json"), "--over-weight", "4", "--epochs", "300", "--seed", "20261030") "$V3\brightness.train.log"
    if ($rc -ne 0) { Fail "estimator training" }
    Copy-Item $InitLora (Join-Path $V3 "rudra_genfill.safetensors")
    $pc = Join-Path (Split-Path $InitLora -Parent) "prompt_cache"
    if (Test-Path $pc) { Copy-Item -Recurse $pc (Join-Path $V3 "prompt_cache") }
    Write-Host "v3 ready in $V3 (the v1 LoRA + brightness.json). Next: -Evaluate" -ForegroundColor Green
    exit 0
}

# ---- the registered gates -------------------------------------------------
if ($Evaluate) {
    if (-not $Lora) { $Lora = Join-Path $V3 "rudra_genfill.safetensors" }
    if (-not (Test-Path $Lora)) { Fail "no LoRA $Lora (run -Brightness first)" }
    $bm = Join-Path (Split-Path $Lora -Parent) "brightness.json"
    if (-not (Test-Path $bm)) { Fail "no brightness estimator $bm (run -Brightness first)" }
    $ev = "$V3`_eval"; $k = 2
    while (Test-Path $ev) { $ev = "$V3`_eval_$k"; $k++ }
    Say "Evaluate (claude_genfill_v3_20261030 gates) -> $ev"
    # v3 (30 Oct): v1 shape + estimated brightness, blend at code 255 only, cap 3 stops, Poly Haven 72-95 gated
    $rc = Py @("tools\evaluate_genfill.py", "--out", $ev, "--base", $Base, "--lora", $Lora, "--brightness-model", $bm, "--device", $Dev, "--registration", $Reg3) "$ev.log"
    if ($rc -ne 0) { Fail "evaluation (see $ev.log)" }
    $rep = Get-Content (Join-Path $ev "report.json") | ConvertFrom-Json
    Write-Host ("`nVerdict: {0} ({1} clips)" -f $rep.verdict, $rep.clips) -ForegroundColor $(if ($rep.verdict -eq "PASS") { "Green" } else { "Yellow" })
    exit 0
}

if ($SmokeTest) {
    $Clips = "$Clips`_smoke"; $Out = "$Out`_smoke"
    foreach ($p in @($Clips, $Out)) { if (Test-Path $p) { Remove-Item -Recurse -Force $p } }
}

# ---- 1. the clips ----------------------------------------------------------
if (Test-Path (Join-Path $Clips "summary.json")) {
    Write-Host "clips already built: $Clips (delete the folder to rebuild)"
} else {
    $resume = Test-Path (Join-Path $Clips "clips.jsonl")   # an interrupted build: continue it
    Say "Plan"
    $rc = Py @("tools\build_genfill_clips.py", "--out", $Clips, "--registration", $Reg, "--dry-run") ""
    if ($rc -ne 0) { Fail "plan" }
    Say "Build and encode the clips ($Workers workers, VAE on $Dev) -> $Clips"
    $bargs = @("tools\build_genfill_clips.py", "--out", $Clips, "--registration", $Reg, "--base", $Base, "--workers", $Workers, "--device", $Dev)
    if ($SmokeTest) { $bargs += @("--limit", "6", "--dev-fraction", "0.34") }
    if ($resume) { $bargs += "--resume"; Write-Host "continuing the interrupted build in $Clips (finished sources are kept)" }
    $rc = Py $bargs $(if ($resume) { "$Clips.resume.log" } else { "$Clips.build.log" })
    if ($rc -ne 0) { Fail "clip build (see $Clips.build.log)" }
}
$sum = Get-Content (Join-Path $Clips "summary.json") | ConvertFrom-Json
Write-Host ("clips {0} from {1} sources, {2} scenes; commercial-clean {3}" -f $sum.clips, $sum.sources_used, $sum.scenes_used, $sum.commercial_clean)
if (-not $sum.commercial_clean) { Fail "a clip without a commercial-clean licence" }
if ($BuildOnly) { exit 0 }

# ---- 2. the LoRA -----------------------------------------------------------
if (-not $SmokeTest -and $Cuda -ne "True") { Fail "no CUDA GPU visible to torch" }
if (Test-Path $Out) { Fail "$Out exists: the registered run writes a new folder" }
$steps = if ($SmokeTest) { 20 } else { 4000 }
$every = if ($SmokeTest) { 10 } else { 1000 }
Say "Train the LoRA (registered: claude_genfill_v1_20261027) -> $Out"
$targs = @("training\genfill_train.py", "--clips", $Clips, "--base", $Base, "--out", $Out, "--device", $Dev,
           "--steps", $steps, "--save-every", $every, "--max-hours", "9.5", "--registration", $Reg,
           "--calib-weight", "0.5", "--calib-sigma", "0.6", "--over-weight", "4")
if (Test-Path $InitLora) { $targs += @("--init-lora", $InitLora) } elseif (-not $SmokeTest) { Fail "no v1 LoRA to start from: $InitLora" }
if ($SmokeTest) { $targs += @("--dev-clips", "4") }
$rc = Py $targs "$Out.train.log"
if ($rc -ne 0) { Fail "training (see $Out.train.log)" }
$rep = Get-Content (Join-Path $Out "report.json") | ConvertFrom-Json
Write-Host ("`nDone: {0} updates in {1} h; LoRA {2} ({3})" -f $rep.steps_done, $rep.hours, $rep.result, $rep.result_sha256) -ForegroundColor Green

if ($SmokeTest) {
    Say "Smoke fill: a 9-frame synthetic master"
    $tm = Join-Path $Out "smoke_master"
    $rc = Py @("-c", "from pathlib import Path; from rudra.genfill.synthetic import write_master; write_master(Path(r'$tm'), n=9, h=272, w=480)") ""
    if ($rc -ne 0) { Fail "smoke master" }
    $rc = Py @("-m", "rudra.genfill.run", "--master", $tm, "--out", "$tm`_aifill", "--base", $Base, "--lora", (Join-Path $Out "rudra_genfill.safetensors"),
               "--device", $Dev, "--steps", "4", "--window", "9", "--overlap", "4", "--json") ""
    if ($rc -ne 0) { Fail "smoke fill" }
    Write-Host "Smoke test passed." -ForegroundColor Green
}
