<#
.SYNOPSIS
  Publish RUDRA ACES 2 and the AI fill v3 to Hugging Face (Apache 2.0), then
  write the models.json the app's Model packages window downloads from.

    powershell -ExecutionPolicy Bypass -File scripts\PUBLISH_HF_MODELS.ps1 -DryRun
    powershell -ExecutionPolicy Bypass -File scripts\PUBLISH_HF_MODELS.ps1
    powershell -ExecutionPolicy Bypass -File scripts\PUBLISH_HF_MODELS.ps1 -Private   # a private repo first

  Uploads dist\hf\rudra-models\ (README model card, LICENSE, NOTICE,
  models\RUDRA_ACES_2\, models\rudra_genfill_v3\) to https://huggingface.co/<Repo>.
  Before the upload it checks every file against the expected SHA-256 values.
  Writes dist\hf\models.json, pinned to the uploaded commit; put that file at
  https://www.fxtdstudios.com/rudra/models.json (the app's default index).

  Needs a Hugging Face login with write access to the org, done once:  .venv\Scripts\hf.exe auth login
#>
param([string]$Repo = "fxtdstudios/rudra-models", [string]$Python = ".venv\Scripts\python.exe",
      [switch]$Private, [switch]$DryRun)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
function Fail([string]$m) { Write-Host "`nFAILED: $m" -ForegroundColor Red; exit 1 }
function Say([string]$m) { Write-Host "`n== $m" -ForegroundColor Cyan }
$Stage = "dist\hf\rudra-models"
if (-not (Test-Path "$Stage\README.md")) { Fail "no $Stage (the staged upload folder)" }
if (-not (Test-Path $Python)) { Fail "no Python at $Python" }

Say "Check the staged files"
$expect = @{
  "models\rudra_genfill_v3\rudra_genfill.safetensors" = "9fff7eb8492ae62619a126d302d9c940403b2e4aa03e02daf79d98142008238a"
  "models\rudra_genfill_v3\brightness.json"           = "34892fb840de138fb3ddd2e2c8e6e3b13c232973ed59ea23338444fcaff2a95a"
  "models\RUDRA_ACES_2\rudra_g1.ts"                   = "fb30e3df748f4fa9ae17f7db189c8fe49a58d9863af6c1c45a408a72f3b7a59b"
  "models\RUDRA_ACES_2\rudra_g1.onnx"                 = "8ba97cb0ed36f5033958e752afad95d585d50bcefefc3bc66a61b27a6618f237"
}
foreach ($k in $expect.Keys) {
    $h = (Get-FileHash (Join-Path $Stage $k) -Algorithm SHA256).Hash.ToLower()
    if ($h -ne $expect[$k]) { Fail "$k has SHA-256 $h, expected $($expect[$k])" }
    Write-Host "  ok  $k"
}
$weights = Get-ChildItem $Stage -Recurse -File | Where-Object { $_.FullName -match 'sdr2hdr_|shadow_v1|\.ckpt$' }
if ($weights) { Fail "non-commercial Beta 2 weights are in the stage: $($weights.Name -join ', ')" }

Say "Hugging Face tools"
& $Python -m pip install --quiet --upgrade "huggingface_hub>=0.34"
if ($LASTEXITCODE -ne 0) { Fail "pip install huggingface_hub" }
$who = & $Python -c "from huggingface_hub import whoami; print(whoami()['name'])" 2>$null
if ($LASTEXITCODE -ne 0 -or -not $who) { Fail "not logged in to Hugging Face: run  $(Split-Path $Python)\hf.exe auth login  (a write token for the org)" }
Write-Host "logged in as $who"

if ($DryRun) {
    Get-ChildItem $Stage -Recurse -File | ForEach-Object { "{0,12:N0}  {1}" -f $_.Length, $_.FullName.Substring((Resolve-Path $Stage).Path.Length + 1) }
    Write-Host "`nDry run: would upload the files above to https://huggingface.co/$Repo$(if ($Private) {' (private)'})." -ForegroundColor Yellow
    exit 0
}

Say "Upload to $Repo"
$py = @"
import sys
from huggingface_hub import HfApi
repo, stage, private = sys.argv[1], sys.argv[2], sys.argv[3] == '1'
api = HfApi()
api.create_repo(repo, repo_type='model', private=private, exist_ok=True)
info = api.upload_folder(repo_id=repo, repo_type='model', folder_path=stage,
                         commit_message='RUDRA ACES 2 and AI fill v3 (Apache 2.0)')
print(info.oid)
"@
$sha = & $Python -c $py $Repo $Stage $(if ($Private) { "1" } else { "0" })
if ($LASTEXITCODE -ne 0 -or -not $sha) { Fail "upload (check write access to $Repo)" }
$sha = ($sha | Select-Object -Last 1).Trim()
Write-Host "uploaded: https://huggingface.co/$Repo/tree/$sha" -ForegroundColor Green

Say "models.json for the app"
& $Python tools\make_models_json.py --repo $Repo --revision $sha --prefix models --out dist\hf\models.json `
    --model "$Stage\models\RUDRA_ACES_2" --id RUDRA_ACES_2 --title "RUDRA ACES 2" --version 2.0 --channel preview --kind aces_v2 `
        --summary "Learned ACES 2.0: rebuilds clipped highlights through the ACES 2.0 view; the SDR codes are kept" `
        --licence "Apache-2.0" --licence-url "https://huggingface.co/$Repo/blob/main/LICENSE" `
    --model "$Stage\models\rudra_genfill_v3" --id rudra_genfill_v3 --title "RUDRA AI fill v3 (experimental)" --version 3.0 --channel preview --kind genfill `
        --summary "LoRA for Wan2.1-VACE-1.3B + brightness estimator: fills fully clipped highlights; needs the base model, Python and a 16 GB GPU" `
        --licence "Apache-2.0" --licence-url "https://huggingface.co/$Repo/blob/main/LICENSE"
if ($LASTEXITCODE -ne 0) { Fail "make_models_json.py" }
Write-Host "`nDone. Publish dist\hf\models.json at https://www.fxtdstudios.com/rudra/models.json" -ForegroundColor Green
if ($Private) { Write-Host "The repo is PRIVATE: the app cannot download from it until you make it public on huggingface.co." -ForegroundColor Yellow }
