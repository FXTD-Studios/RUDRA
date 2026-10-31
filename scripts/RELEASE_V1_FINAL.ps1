<#
.SYNOPSIS
  RUDRA 1.0.0: publish the models to Hugging Face, set the version, commit and push
  `aces-v2`, tag v1.0.0 (the release workflow builds and publishes the release), and
  rebuild the user guide for 1.0 (31 Oct 2026).

    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_V1_FINAL.ps1 -DryRun
    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_V1_FINAL.ps1
    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_V1_FINAL.ps1 -SkipModels   # models already published

  In order:
    1. checks: on `aces-v2`, nothing to pull, no v1.0.0 tag yet, signed in to GitHub
    2. scripts\UPLOAD_MODELS_HF.ps1: the ACES 2.0 model and the AI fill v3 to
       huggingface.co/fxtdstudios/rudra-models at tag v1.0.0, and models.json beside the guide
    3. native/CMakeLists.txt: RUDRA_RELEASE "" (the app says 1.0.0)
    4. the user guide for 1.0 (tools\build_user_guide.py --release --tag v1.0.0) into ..\website\rudra-docs
    5. commit everything (no weights, nothing over 20 MB, result.json left out), push `aces-v2`
    6. tag v1.0.0 and push it: .github/workflows/release.yml builds the Windows installer and ZIP and
       the macOS DMG and publishes them as a release with docs/RELEASE_NOTES_1.0.0.md

  Then deploy the website (the guide and models.json). Builds are unsigned unless the
  signing secrets are set in GitHub. Never touches main, never force-pushes.
#>
param([switch]$DryRun, [switch]$NoTag, [switch]$SkipModels, [string]$Branch = "aces-v2", [string]$Tag = "v1.0.0",
      [string]$Docs = "..\website\rudra-docs", [string]$Python = ".venv\Scripts\python.exe")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
function Fail([string]$m) { Write-Host "`nFAILED: $m" -ForegroundColor Red; exit 1 }
function Say([string]$m) { Write-Host "`n== $m" -ForegroundColor Cyan }

Say "Checks"
if ((Test-Path ".git\rebase-merge") -or (Test-Path ".git\rebase-apply")) { Fail "a rebase is in progress: git rebase --quit first" }
if (Test-Path ".git\index.lock") { Fail ".git\index.lock exists: close other git tools, then move it aside" }
$here = git rev-parse --abbrev-ref HEAD
if ($here -ne $Branch) { Fail "on $here, not $Branch (git switch $Branch first)" }
git fetch -q origin $Branch --tags
if ($LASTEXITCODE -ne 0) { Fail "git fetch (check your GitHub sign-in)" }
$behind = [int](git rev-list --count "HEAD..origin/$Branch")
if ($behind -gt 0) { Fail "origin/$Branch has $behind commit(s) you do not have: git pull --ff-only first" }
if (-not $NoTag -and (git tag --list $Tag)) { Fail "tag $Tag already exists" }
foreach ($f in @("docs\RELEASE_NOTES_1.0.0.md", "docs\HF_MODEL_CARD_1.0.md", "tools\make_models_json.py", "tools\hf_publish_models.py",
                 "tools\build_user_guide.py", "scripts\UPLOAD_MODELS_HF.ps1")) {
    if (-not (Test-Path $f)) { Fail "$f is missing" }
}
if (-not (Select-String -Path ".github\workflows\release.yml" -Pattern 'body_path: docs/RELEASE_NOTES_1\.0\.0\.md' -Quiet)) {
    Fail ".github/workflows/release.yml does not use docs/RELEASE_NOTES_1.0.0.md"
}

if (-not $SkipModels) {
    Say "Models to Hugging Face"
    $up = @{ Revision = $Tag; Docs = $Docs; Python = $Python }
    if ($DryRun) { $up.DryRun = $true }
    & (Join-Path $PSScriptRoot "UPLOAD_MODELS_HF.ps1") @up
    if ($LASTEXITCODE -ne 0) { Fail "UPLOAD_MODELS_HF.ps1" }
}

Say "Version 1.0.0"
$cm = "native\CMakeLists.txt"
$text = [IO.File]::ReadAllText((Resolve-Path $cm))
$new = $text -replace 'set\(RUDRA_RELEASE "[^"]*"', 'set(RUDRA_RELEASE ""'
if ($new -notmatch 'project\(RudraNative VERSION 1\.0\.0 ') { Fail "$cm is not at project VERSION 1.0.0" }
if ($DryRun) { Write-Host "would set RUDRA_RELEASE to """" in $cm" }
elseif ($new -ne $text) { [IO.File]::WriteAllText((Resolve-Path $cm), $new, (New-Object System.Text.UTF8Encoding($false))); Write-Host "RUDRA_RELEASE set to """"" }

Say "User guide for 1.0"
if (Test-Path $Docs) {
    if ($DryRun) { Write-Host "would rebuild $Docs\index.html for 1.0.0" }
    else {
        & $Python tools\build_user_guide.py --version 1.0.0 --label 1.0.0 --tag $Tag --release --out $Docs
        if ($LASTEXITCODE -ne 0) { Fail "build_user_guide.py" }
    }
} else { Write-Host "no ${Docs}: run tools\build_user_guide.py --version 1.0.0 --tag $Tag --release --out <rudra-docs> later" -ForegroundColor Yellow }

Say "Stage"
$big = git status --porcelain --untracked-files=all | ForEach-Object { $_.Substring(3).Trim('"') } |
       Where-Object { (Test-Path $_ -PathType Leaf) -and ((Get-Item $_).Length -gt 20MB) }
if ($big) { $big | ForEach-Object { Write-Host "  large: $_" }; Fail "large files would be committed; add them to .gitignore first" }
git add -A
git reset -q -- result.json dist
$files = git diff --cached --name-only
if ($files -match '\.safetensors$|\.pt$|\.onnx$|\.ts$|brightness\.json$') {
    $files -match '\.safetensors$|\.pt$|\.onnx$|\.ts$|brightness\.json$' | ForEach-Object { Write-Host "  model file: $_" }
    git reset -q; Fail "model weights are staged: they live on Hugging Face, not in git"
}
if ($DryRun) {
    if ($files) { git diff --cached --name-status; git reset -q }
    Write-Host "`nDry run: nothing committed, pushed or tagged." -ForegroundColor Yellow
    exit 0
}
if ($files) {
    Say "Commit"
    $msg = @"
RUDRA 1.0.0

- The learned ACES 2.0 model and the AI fill v3 are published under Apache 2.0 at
  huggingface.co/fxtdstudios/rudra-models ($Tag) and download inside the app; models.json
  (tools/make_models_json.py) is served beside the user guide.
- app: the downloaded AI fill LoRA becomes Deliver > AI fill's; the model list is read from
  www.fxtdstudios.com/rudra-docs/models.json
- release: version 1.0.0; release notes docs/RELEASE_NOTES_1.0.0.md; the user guide for 1.0
  (tools/build_user_guide.py); scripts/UPLOAD_MODELS_HF.ps1, scripts/RELEASE_V1_FINAL.ps1

Co-authored-by: FXTD Studios Research <research@fxtdstudios.com>
"@
    $tmp = New-TemporaryFile
    [IO.File]::WriteAllText($tmp.FullName, $msg, (New-Object System.Text.UTF8Encoding($false)))
    git commit -q -F $tmp.FullName
    $code = $LASTEXITCODE
    Remove-Item $tmp
    if ($code -ne 0) { Fail "git commit" }
}
$credits = (git log "origin/$Branch..HEAD" --format=%B) -join "`n"
if ($credits -match '(?i)anthropic\.com|co-authored-by:\s*claude|claude-session') { Fail "a commit to push credits an AI assistant; amend it first (AGENTS.md)" }

Say "Push $Branch"
git push origin $Branch
if ($LASTEXITCODE -ne 0) { Fail "git push (check your GitHub sign-in)" }
Write-Host "Pushed $Branch ($(git rev-parse --short HEAD)). main is unchanged." -ForegroundColor Green
if ($NoTag) { exit 0 }

Say "Tag $Tag"
git tag -a $Tag -m "RUDRA 1.0.0"
if ($LASTEXITCODE -ne 0) { Fail "git tag" }
git push origin $Tag
if ($LASTEXITCODE -ne 0) { Fail "git push $Tag" }
Write-Host "`nTag $Tag pushed. GitHub builds Windows and macOS (about 30-60 min) and publishes the release:" -ForegroundColor Green
Write-Host "  https://github.com/FXTD-Studios/RUDRA/actions/workflows/release.yml"
Write-Host "  https://github.com/FXTD-Studios/RUDRA/releases/tag/$Tag"
Write-Host "Then deploy the website: $Docs (the 1.0 guide and models.json)."
