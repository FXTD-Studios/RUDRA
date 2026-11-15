<#
.SYNOPSIS
  Ship RUDRA 1.2.4 (15 Nov 2026): compile and test, commit and push, open the PR, tag the release,
  then publish AI fill v4 on Hugging Face and write the app's models.json. Three runs, because a
  person merges the PR in between and the model goes out only after the app that can run it:

    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_1_2_4.ps1 -DryRun   # what it would do
    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_1_2_4.ps1           # 1. build, test, package, commit, push, PR
    #   CI green, merge the PR on GitHub with "Squash and merge", then:
    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_1_2_4.ps1 -Tag      # 2. tag v1.2.4: GitHub builds and publishes it
    #   when the v1.2.4 release is published on GitHub:
    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_1_2_4.ps1 -Models   # 3. AI fill v4 to Hugging Face + models.json

  RUDRA ACES 2 and AI fill v3 are unchanged: -Models republishes them as they are (ACES 2 stays
  release, min app 1.2.0). AI fill v4 goes out as preview, min app 1.2.4: the files are the LoRA of
  the v4.2 training and docs/models/genfill_v4 (registration_v4.json, report_v4.json), whose
  checksums 1.2.4 has built in. Commit messages credit FXTD Studios Research only (AGENTS.md).
#>
param(
    [switch]$Tag,
    [switch]$Models,
    [switch]$DryRun,
    [switch]$SkipBuild,
    [switch]$SkipPackage,
    [string]$Branch = "feat/genfill-v4-finish",
    [string]$Version = "1.2.4",
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$MessageFile = "tmp\commit_msg_1_2_4_release.txt",
    [string]$V4Lora = "G:\rudra_training\research_genfill_v4_2_20261114\rudra_genfill.safetensors",
    [string]$Website = "..\website"
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $Repo
function Say([string]$m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail([string]$m) { Write-Host "`nFAILED: $m" -ForegroundColor Red; exit 1 }
function Run([string]$what, [scriptblock]$cmd) {
    if ($DryRun) { Write-Host "  (dry run) $what" -ForegroundColor DarkGray; return }
    & $cmd
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { Fail $what }
}
if (Test-Path ".git\index.lock") { Fail ".git\index.lock exists: move it to .git\_stale_locks (AGENTS.md)" }
$TagName = "v$Version"
$Notes = "docs/RELEASE_NOTES_$Version.md"
$HasGh = [bool](Get-Command gh -ErrorAction SilentlyContinue)
$Credit = "Co-authored-by: FXTD Studios Research <research@fxtdstudios.com>"
$Assistant = "Cl" + "aude|anthrop" + "ic"

# ==== 3. models: AI fill v4 on Hugging Face, and models.json ===========================
if ($Models) {
    if ($HasGh) {
        $relJson = (gh release view $TagName --json isDraft,tagName 2>$null | Out-String).Trim()
        $rel = if ($relJson) { $relJson | ConvertFrom-Json } else { $null }
        if (-not $rel -or $rel.isDraft) { Fail "the $TagName release is not published on GitHub yet: AI fill v4 needs app $Version, so publish the app first" }
        Write-Host "$TagName is published on GitHub"
    } else {
        Write-Host "  gh is not installed: check by hand that the $TagName release is published before going on" -ForegroundColor Yellow
    }
    Say "Stage AI fill v4 (models\rudra_genfill_v4)"
    $Stage = "dist\hf\rudra-models\models\rudra_genfill_v4"
    if (-not (Test-Path $V4Lora)) { Fail "no LoRA at $V4Lora" }
    foreach ($f in "docs\models\genfill_v4\registration_v4.json", "docs\models\genfill_v4\report_v4.json") { if (-not (Test-Path $f)) { Fail "no $f" } }
    Run "stage v4" {
        New-Item -ItemType Directory -Force -Path $Stage | Out-Null
        Copy-Item $V4Lora "$Stage\rudra_genfill.safetensors" -Force
        Copy-Item "docs\models\genfill_v4\registration_v4.json", "docs\models\genfill_v4\report_v4.json" $Stage -Force
    }
    Say "Publish to Hugging Face (PUBLISH_HF_MODELS.ps1 checks every file's SHA-256 first)"
    $pub = @("-ExecutionPolicy", "Bypass", "-File", "scripts\PUBLISH_HF_MODELS.ps1", "-AcesChannel", "release", "-AcesMinApp", "1.2.0")
    if ($DryRun) { $pub += "-DryRun" }
    & powershell @pub
    if ($LASTEXITCODE -ne 0) { Fail "PUBLISH_HF_MODELS.ps1" }
    if ($DryRun) { exit 0 }
    $Docs = Join-Path $Website "rudra-docs"
    if (Test-Path $Docs) {
        Copy-Item "dist\hf\models.json" (Join-Path $Docs "models.json") -Force
        Write-Host "`nCopied dist\hf\models.json to $Docs\models.json: deploy the website so" -ForegroundColor Green
        Write-Host "https://www.fxtdstudios.com/rudra-docs/models.json serves it." -ForegroundColor Green
    } else {
        Write-Host "`nNo ${Docs}: upload dist\hf\models.json to https://www.fxtdstudios.com/rudra-docs/models.json by hand." -ForegroundColor Yellow
    }
    exit 0
}

Say "Fetch origin"
git fetch --no-tags origin
if ($LASTEXITCODE -ne 0) { Fail "git fetch origin" }

# ==== 2. tag the merge =================================================================
if ($Tag) {
    Say "Check origin/main before tagging $TagName"
    $cm = git show "origin/main:native/CMakeLists.txt" | Select-String -Pattern "project\(RudraNative VERSION $([regex]::Escape($Version))" -Quiet
    if (-not $cm) { Fail "native/CMakeLists.txt on origin/main is not VERSION ${Version}: merge the PR first" }
    foreach ($f in $Notes, "native/app/title_bar.cpp", "docs/models/genfill_v4/registration_v4.json", "configs/experiments/research_genfill_v4_3_20261115.json") {
        if (-not (git ls-tree --name-only origin/main $f)) { Fail "origin/main lacks ${f}: merge the PR ($Branch, Squash and merge) first" }
    }
    git diff --quiet origin/main "origin/$Branch" -- native rudra scripts tools docs $Notes
    if ($LASTEXITCODE -ne 0) {
        git diff --stat origin/main "origin/$Branch" -- native rudra scripts tools docs | Select-Object -Last 5
        Fail "origin/main differs from origin/$Branch (above): merge the PR, or push what is missing, before tagging"
    }
    if (git ls-remote --tags origin "refs/tags/$TagName") { Fail "$TagName already exists on origin" }
    if (git tag -l $TagName) { Fail "$TagName already exists locally (git tag -d $TagName if it is a mistake)" }
    $sha = (git rev-parse --short origin/main).Trim()
    Say "Tag $TagName on origin/main ($sha) and push it"
    Run "git tag" { git tag -a $TagName origin/main -m "RUDRA $Version" }
    Run "git push tag" { git push origin $TagName }
    Write-Host "  The release workflow builds Windows and macOS and publishes $TagName with ${Notes}:" -ForegroundColor Green
    Write-Host "  https://github.com/FXTD-Studios/RUDRA/actions   Then: scripts\RELEASE_1_2_4.ps1 -Models" -ForegroundColor Green
    $Guide = Join-Path $Website "rudra-docs"
    if ((Test-Path (Join-Path $Guide "index.html")) -and (Test-Path $Python) -and (Test-Path "tools\build_user_guide.py")) {
        Say "Rebuild the user guide for $Version into $Guide"
        Run "build_user_guide" { & $Python tools/build_user_guide.py --version $Version --tag $TagName --release --out $Guide }
        Write-Host "  Deploy $Guide once the release is published (its download buttons point at $TagName)." -ForegroundColor Green
    } else {
        Write-Host "  Update the website guide by hand: tmp\website_1_2_4_update.md" -ForegroundColor Yellow
    }
    exit 0
}

# ==== 1. build, test, package, commit, push, PR ========================================
Say "Check the release is ready on $Branch"
$cur = (git rev-parse --abbrev-ref HEAD).Trim()
if ($cur -ne $Branch) { Fail "on branch $cur, not ${Branch}: git switch $Branch (or -Branch $cur)" }
if (-not (Select-String -Path native\CMakeLists.txt -Pattern "project\(RudraNative VERSION $([regex]::Escape($Version))" -Quiet)) { Fail "native/CMakeLists.txt is not VERSION $Version" }
foreach ($f in $Notes, $MessageFile, "docs\models\genfill_v4\registration_v4.json", "docs\models\genfill_v4\report_v4.json") { if (-not (Test-Path $f)) { Fail "no $f" } }
if (-not (Test-Path $Python)) { Fail "no Python at $Python" }
if (Get-Process RUDRA -ErrorAction SilentlyContinue) { Fail "RUDRA is running: close it (Windows cannot replace a running exe)" }
$msg = Get-Content $MessageFile -Raw
if ($msg -match $Assistant) { Fail "$MessageFile names the former assistant (AGENTS.md)" }
if ($msg -notmatch [regex]::Escape($Credit)) { Fail "$MessageFile lacks the credit line: $Credit" }

Say "No third-party product names in the tracked files"
$names = git --no-optional-locks grep -n -I -i -E "beeble|switch ?hdr|lightricks|\bltx" -- . ":(exclude)tmp" ":(exclude)dist"
if ($names) { $names | Select-Object -First 10 | ForEach-Object { Write-Host "  $_" }; Fail "remove the names above before release" }
Write-Host "  none"

if (-not $SkipBuild) {
    Say "Compile and run the native tests (core and app)"
    Run "NATIVE_TESTS.ps1" { & powershell -ExecutionPolicy Bypass -File scripts\NATIVE_TESTS.ps1 }
    $AcesCache = "build\native_aces_v2\CMakeCache.txt"
    if (Test-Path $AcesCache) {
        Say "Compile the ACES v2 engine"
        $acesCmake = ((Select-String -Path $AcesCache -Pattern '^CMAKE_COMMAND:INTERNAL=(.+)$').Matches[0].Groups[1].Value)
        Run "ACES engine build" { & $acesCmake --build build\native_aces_v2 --config Release --parallel --target rudra-native }
    }
    Say "Python tests"
    $py = @("tests\test_genfill_core.py", "tests\test_lock_golden.py", "tests\test_promote_aces2_release.py", "tests\test_genfill_focus.py",
            "tests\test_vae_identity.py", "tests\test_rudra_fixes.py") | Where-Object { Test-Path $_ }
    Run "pytest" { & $Python -m pytest -q @py }
}
if (-not $SkipPackage) {
    Say "The public Windows package, locally (no weights; what release.yml builds)"
    Run "PACKAGE_WINDOWS.ps1 -PublicRelease" { & powershell -ExecutionPolicy Bypass -File scripts\PACKAGE_WINDOWS.ps1 -Python $Python -PublicRelease -NoInstaller }
}

Say "Commit"
Run "git add" { git add -A -- native rudra scripts tools tests docs configs config pipeline training archive ui .github README.md pyproject.toml SECURITY.md .gitignore }
$staged = @(git diff --cached --name-only)
if ($staged.Count) {
    $bad = $staged | Where-Object { $_ -match '\.(pt|ts|onnx|safetensors|ckpt|exr|mp4|mov)$' -or $_ -match '^(dist|reports|tmp|build)/' }
    $big = $staged | Where-Object { (Test-Path $_) -and (Get-Item $_).Length -gt 5MB }
    if ($bad -or $big) {
        git reset -q
        Fail "these must not be committed: $((@($bad) + @($big) | Select-Object -Unique) -join ', ')"
    }
    Write-Host "$($staged.Count) file(s):"; $staged | ForEach-Object { Write-Host "  $_" }
    Run "git commit" { git commit -q -F $MessageFile }
} else { Write-Host "nothing new to commit" }

$ahead = git log "origin/main..$Branch" --format="%h %s"
if (-not $ahead) { Fail "$Branch has nothing that origin/main lacks: already merged? run with -Tag" }
Say "$Branch is $(@($ahead).Count) commit(s) ahead of origin/main"
$ahead | Select-Object -First 15 | ForEach-Object { Write-Host "  $_" }
if (git log "origin/main..$Branch" --format=%B | Select-String -Pattern $Assistant -Quiet) {
    Write-Host "  Some commit messages name the former assistant: merge with 'Squash and merge' so main gets one commit with your message" -ForegroundColor Yellow
}

Say "Push $Branch"
Run "git push" { git push -u origin $Branch }
$body = "RUDRA ${Version}: security fixes, model-update repair, the dark title bar on Windows, a calmer Deliver column, and AI fill v4 (region re-render, experimental). Notes: $Notes"
if ($HasGh) {
    $prJson = (gh pr list --head $Branch --state open --json number,url 2>$null | Out-String).Trim()
    $pr = if ($prJson) { @($prJson | ConvertFrom-Json) } else { @() }
    if ($pr.Count) { Write-Host "  PR #$($pr[0].number) is open: $($pr[0].url)" -ForegroundColor Green }
    else { Run "gh pr create" { gh pr create --base main --head $Branch --title "RUDRA $Version" --body $body } }
} else {
    Write-Host "  gh is not installed: open https://github.com/FXTD-Studios/RUDRA/compare/main...$Branch" -ForegroundColor Yellow
}
Say "Next: CI green, merge the PR (Squash and merge), then: scripts\RELEASE_1_2_4.ps1 -Tag"
