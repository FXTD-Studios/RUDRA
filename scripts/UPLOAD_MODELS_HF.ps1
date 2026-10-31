<#
.SYNOPSIS
  Publish the RUDRA 1.0 models to Hugging Face and write the models.json the app
  downloads from (31 Oct 2026).

    powershell -ExecutionPolicy Bypass -File scripts\UPLOAD_MODELS_HF.ps1 -DryRun
    powershell -ExecutionPolicy Bypass -File scripts\UPLOAD_MODELS_HF.ps1

  The upload is the prepared folder dist\hf\rudra-models, and only it:
       README.md, LICENSE, NOTICE           the model card (docs\HF_MODEL_CARD_1.0.md), Apache 2.0
       models\RUDRA_ACES_2\                 the final learned ACES 2.0 model (release_authorized true)
       models\rudra_genfill_v3\             the AI fill v3: rudra_genfill.safetensors, brightness.json,
                                            prompt_cache\, registration_v3.json, report_v3.json
  1. Checks the folder: every file the app needs, RUDRA_ACES_2's TorchScript and ONNX the same as
     dist\models\RUDRA_ACES_2's, and nothing that is not a model file (.part, download markers).
  2. Uploads it to -Repo in one commit, tags it -Revision (tools\hf_publish_models.py), and checks
     every file is on the hub at its size.
  3. Writes models.json (tools\make_models_json.py): each file's size, SHA-256 and URL at that tag,
     into dist\hf\ and beside the user guide (-Docs), which serves it at
     https://www.fxtdstudios.com/rudra-docs/models.json (the app's default list).

  Sign in to Hugging Face first: .venv\Scripts\hf.exe auth login (a write token for the
  fxtdstudios organisation), or set HF_TOKEN. -DryRun checks and writes models.json, uploads nothing.
#>
param(
    [string]$Repo = "fxtdstudios/rudra-models",
    [string]$Revision = "v1.0.0",
    [string]$Stage = "dist\hf\rudra-models",
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$Docs = "..\website\rudra-docs",
    [switch]$Private,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
function Fail([string]$m) { Write-Host "`nFAILED: $m" -ForegroundColor Red; exit 1 }
function Say([string]$m) { Write-Host "`n== $m" -ForegroundColor Cyan }

Say "The upload: $Stage"
if (-not (Test-Path $Python)) { Fail "no $Python (the repo's .venv)" }
$aces = Join-Path $Stage "models\RUDRA_ACES_2"
$fill = Join-Path $Stage "models\rudra_genfill_v3"
foreach ($f in @("README.md", "LICENSE", "NOTICE", "models\RUDRA_ACES_2\manifest.json", "models\RUDRA_ACES_2\parity.json",
                 "models\RUDRA_ACES_2\rudra_g1.ts", "models\RUDRA_ACES_2\rudra_g1.onnx",
                 "models\rudra_genfill_v3\rudra_genfill.safetensors", "models\rudra_genfill_v3\brightness.json")) {
    if (-not (Test-Path (Join-Path $Stage $f))) { Fail "$Stage has no $f" }
}
$m = Get-Content (Join-Path $aces "manifest.json") -Raw | ConvertFrom-Json
if (-not $m.parity_pass) { Fail "RUDRA_ACES_2 failed its export parity check" }
if (-not $m.release_authorized) { Fail "RUDRA_ACES_2's manifest is not release_authorized" }
foreach ($f in @("rudra_g1.ts", "rudra_g1.onnx")) {
    $src = Join-Path "dist\models\RUDRA_ACES_2" $f
    if ((Test-Path $src) -and ((Get-FileHash $src).Hash -ne (Get-FileHash (Join-Path $aces $f)).Hash)) {
        Fail "$f in the upload differs from dist\models\RUDRA_ACES_2: not the final model"
    }
}
$stray = Get-ChildItem $Stage -Recurse -File | Where-Object { $_.Name -match '\.part$|^\.rudra-download\.json$|^Thumbs\.db$|^\.DS_Store$' }
if ($stray) { $stray | ForEach-Object { Write-Host "  stray: $($_.FullName)" }; Fail "remove the stray files above first" }
Get-ChildItem $Stage -Recurse -File | ForEach-Object { "{0,13:N0}  {1}" -f $_.Length, $_.FullName.Substring((Resolve-Path $Stage).Path.Length + 1) }

Say "models.json"
$index = Join-Path (Split-Path $Stage) "models.json"
$card = "https://huggingface.co/$Repo/blob/$Revision/README.md"
& $Python tools\make_models_json.py --repo $Repo --revision $Revision --prefix models --out $index `
    --model $aces --id rudra_aces_2 --title "RUDRA ACES 2" --version 1.0 --channel release --kind aces_v2 `
        --summary "the learned ACES 2.0 reconstruction: rebuilds the highlights the SDR clipped" `
        --licence "Apache-2.0" --licence-url $card --min-app 1.0.0 `
    --model $fill --id rudra_genfill_v3 --title "AI fill v3 (experimental)" --version 3 --channel preview --kind genfill `
        --summary "draws the shape inside blown highlights; needs Wan2.1-VACE-1.3B and a 16 GB GPU" `
        --licence "Apache-2.0" --licence-url $card --min-app 1.0.0
if ($LASTEXITCODE -ne 0) { Fail "make_models_json" }

Say "Hugging Face"
& $Python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('huggingface_hub') else 1)"
if ($LASTEXITCODE -ne 0) { & $Python -m pip install -U huggingface_hub; if ($LASTEXITCODE -ne 0) { Fail "pip install huggingface_hub" } }
$pubArgs = @("tools\hf_publish_models.py", "--repo", $Repo, "--stage", $Stage, "--revision", $Revision)
if ($Private) { $pubArgs += "--private" }
if ($DryRun) { $pubArgs += "--dry-run" }
& $Python @pubArgs
if ($LASTEXITCODE -ne 0) { Fail "publishing (exit $LASTEXITCODE)" }
if ($DryRun) {
    Write-Host "`nDry run: checked, and models.json written to $index; nothing uploaded." -ForegroundColor Yellow
    exit 0
}

Say "models.json beside the user guide"
if (Test-Path $Docs) {
    Copy-Item $index (Join-Path $Docs "models.json") -Force
    Write-Host "$Docs\models.json: deploy the website to publish it at https://www.fxtdstudios.com/rudra-docs/models.json"
} else {
    Write-Host "no ${Docs}: copy $index to the website so it is served at https://www.fxtdstudios.com/rudra-docs/models.json" -ForegroundColor Yellow
}
Write-Host "`nDone: https://huggingface.co/$Repo/tree/$Revision" -ForegroundColor Green
exit 0
