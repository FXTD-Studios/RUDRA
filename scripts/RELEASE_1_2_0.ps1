<#
.SYNOPSIS
  Ship RUDRA 1.2.0 (8 Nov 2026): compile and test, push and open the PR, tag the release on
  GitHub, then publish the models on Hugging Face and the app's models.json. Three runs, because a
  person merges the PR in between and the model goes out only after the app that can run it:

    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_1_2_0.ps1 -DryRun     # what it would do
    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_1_2_0.ps1             # 1. build, test, package, commit, push, PR
    #   merge the PR on GitHub with "Squash and merge" (CI green), then:
    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_1_2_0.ps1 -Tag        # 2. tag v1.2.0: GitHub builds and publishes it
    #   when the v1.2.0 release is published on GitHub:
    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_1_2_0.ps1 -Models     # 3. Hugging Face + models.json (model as preview)
    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_1_2_0.ps1 -Models -Review reports\aces2_review_<stamp>\review.json `
        -CoremlNotMeasured "No Mac; Core ML ships as preview"                       #    or as release, with the signed review

  Step 1 builds the app and the ACES engine, runs the native tests (core and app) and the Python
  tests of the lock, the release gate and the AI fill, builds the public Windows package locally
  (no weights; the same check release.yml makes), then commits everything on the branch with your
  message and pushes it. Nothing is merged for you. Commit messages never name the assistant (AGENTS.md).
#>
param(
    [switch]$Tag,
    [switch]$Models,
    [switch]$DryRun,
    [switch]$SkipBuild,
    [switch]$SkipPackage,
    [string]$Branch = "feat/native-lock",
    [string]$Version = "1.2.0",
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$Message = "",
    # -Models: the model package and its release evidence
    [string]$Package = "dist\models\RUDRA_ACES_2_w4",
    [string]$Review = "",                 # a signed review.json: publish RUDRA ACES 2 as release
    [string]$CoremlReport = "",
    [string]$CoremlNotMeasured = "",
    [string]$NativeReport = "",           # default: the newest reports\aces_v2_* folder
    [string]$Holdout = "G:\rudra_training\holdout_crew_20261107",
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

# ==== 3. models: Hugging Face and models.json ========================================
if ($Models) {
    if ($HasGh) {
        $relJson = (gh release view $TagName --json isDraft,tagName 2>$null | Out-String).Trim()
        $rel = if ($relJson) { $relJson | ConvertFrom-Json } else { $null }
        if (-not $rel -or $rel.isDraft) { Fail "the $TagName release is not published on GitHub yet: the locked model needs app $Version, so publish the app first" }
        Write-Host "$TagName is published on GitHub"
    } else {
        Write-Host "  gh is not installed: check by hand that the $TagName release is published before going on" -ForegroundColor Yellow
    }
    if (-not (Test-Path "$Package\manifest.json")) { Fail "no $Package" }
    $Stage = "dist\hf\rudra-models\models\RUDRA_ACES_2"
    $src = Get-Content "$Package\manifest.json" -Raw | ConvertFrom-Json
    if ($src.contract_major -ne 3 -or $src.lock.name -ne "lock_dqs050g") { Fail "$Package is not the w4 + lock_dqs050g package (major 3)" }
    if ($Review) {
        Say "Promote RUDRA ACES 2 to release (tools\promote_aces2_release.py checks every piece of evidence)"
        if (-not $NativeReport) {
            $r = Get-ChildItem reports -Directory -Filter "aces_v2_*" | Sort-Object LastWriteTime | Select-Object -Last 1
            if (-not $r) { Fail "no reports\aces_v2_* (scripts\RELEASE_ACES2_W4.ps1 makes it); or -NativeReport" }
            $NativeReport = $r.FullName
        }
        $prom = @("tools\promote_aces2_release.py", "--package", $Package, "--holdout-verdict", "$Holdout\verdict.json",
                  "--holdout-registration", "configs\experiments\holdout_crew_eval_20261107.json", "--native-report", $NativeReport,
                  "--provenance", "docs\models\RUDRA_ACES_2.provenance.json", "--review", $Review, "--stage", $Stage)
        if ($CoremlReport) { $prom += @("--coreml-report", $CoremlReport) }
        elseif ($CoremlNotMeasured) { $prom += @("--coreml-not-measured", $CoremlNotMeasured) }
        else { Fail "give -CoremlReport <coreml.json> or -CoremlNotMeasured ""<reason>""" }
        Run "promote_aces2_release.py" { & $Python @prom }
        $Channel = "release"
    } else {
        Say "Stage RUDRA ACES 2 (w4 + lock) as preview: no -Review given"
        $same = $false
        if (Test-Path "$Stage\manifest.json") {
            $st = Get-Content "$Stage\manifest.json" -Raw | ConvertFrom-Json
            $same = ($st.g1_sha256 -eq $src.g1_sha256 -and $st.lock.name -eq $src.lock.name)
        }
        if ($same) { Write-Host "the staged package is already w4 + lock" }
        else { Run "robocopy $Package $Stage" { robocopy $Package $Stage /MIR /NFL /NDL /NJH /NJS /NP | Out-Null; if ($LASTEXITCODE -lt 8) { $global:LASTEXITCODE = 0 } } }
        $Channel = "preview"
    }
    Say "Publish to Hugging Face (channel $Channel)"
    $pub = @("-ExecutionPolicy", "Bypass", "-File", "scripts\PUBLISH_HF_MODELS.ps1", "-AcesChannel", $Channel, "-AcesMinApp", $Version)
    if ($DryRun) { $pub += "-DryRun" }
    & powershell @pub
    if ($LASTEXITCODE -ne 0) { Fail "PUBLISH_HF_MODELS.ps1" }
    if ($DryRun) { exit 0 }
    $Docs = Join-Path $Website "rudra-docs"
    if (Test-Path $Docs) {
        Copy-Item "dist\hf\models.json" (Join-Path $Docs "models.json") -Force
        Write-Host "`nCopied dist\hf\models.json to $Docs\models.json: deploy the website (DEPLOYMENT.md) so" -ForegroundColor Green
        Write-Host "https://www.fxtdstudios.com/rudra-docs/models.json serves it." -ForegroundColor Green
    } else {
        Write-Host "`nNo ${Docs}: upload dist\hf\models.json to https://www.fxtdstudios.com/rudra-docs/models.json by hand." -ForegroundColor Yellow
    }
    exit 0
}

Say "Fetch origin"
git fetch --no-tags origin
if ($LASTEXITCODE -ne 0) { Fail "git fetch origin" }

# ==== 2. tag the merge ================================================================
if ($Tag) {
    Say "Check origin/main before tagging $TagName"
    $cm = git show "origin/main:native/CMakeLists.txt" | Select-String -Pattern "project\(RudraNative VERSION $([regex]::Escape($Version))" -Quiet
    if (-not $cm) { Fail "native/CMakeLists.txt on origin/main is not VERSION ${Version}: merge the PR first" }
    foreach ($f in $Notes, "native/core/src/aces_lock.cpp", "tools/promote_aces2_release.py", "native/app/flow_layout.cpp") {
        if (-not (git ls-tree --name-only origin/main $f)) { Fail "origin/main lacks ${f}: merge the PR ($Branch, Squash and merge) first" }
    }
    git diff --quiet origin/main "origin/$Branch" -- native rudra scripts tools $Notes
    if ($LASTEXITCODE -ne 0) {
        git diff --stat origin/main "origin/$Branch" -- native rudra scripts tools $Notes | Select-Object -Last 5
        Fail "origin/main differs from origin/$Branch (above): merge the PR, or push what is missing, before tagging"
    }
    if (git ls-remote --tags origin "refs/tags/$TagName") { Fail "$TagName already exists on origin" }
    if (git tag -l $TagName) { Fail "$TagName already exists locally (git tag -d $TagName if it is a mistake)" }
    $sha = (git rev-parse --short origin/main).Trim()
    Say "Tag $TagName on origin/main ($sha) and push it"
    Run "git tag" { git tag -a $TagName origin/main -m "RUDRA $Version" }
    Run "git push tag" { git push origin $TagName }
    Write-Host "  The release workflow builds Windows and macOS and publishes $TagName with ${Notes}:" -ForegroundColor Green
    Write-Host "  https://github.com/FXTD-Studios/RUDRA/actions   Then: scripts\RELEASE_1_2_0.ps1 -Models" -ForegroundColor Green
    $Guide = Join-Path $Website "rudra-docs"
    if ((Test-Path (Join-Path $Guide "index.html")) -and (Test-Path $Python)) {
        Say "Rebuild the user guide for $Version into $Guide"
        Run "build_user_guide" { & $Python tools/build_user_guide.py --version $Version --tag $TagName --release --out $Guide }
        Write-Host "  Deploy $Guide once the release is published (its download buttons point at $TagName)." -ForegroundColor Green
    }
    exit 0
}

# ==== 1. build, test, package, commit, push, PR ======================================
Say "Check the release is ready on $Branch"
$cur = (git rev-parse --abbrev-ref HEAD).Trim()
if ($cur -ne $Branch) { Fail "on branch $cur, not ${Branch}: git switch $Branch (or -Branch $cur)" }
if (-not (Select-String -Path native\CMakeLists.txt -Pattern "project\(RudraNative VERSION $([regex]::Escape($Version))" -Quiet)) { Fail "native/CMakeLists.txt is not VERSION $Version" }
if (-not (Test-Path $Notes)) { Fail "no $Notes" }
if (-not (Test-Path $Python)) { Fail "no Python at $Python" }
if (Get-Process RUDRA -ErrorAction SilentlyContinue) { Fail "RUDRA is running: close it (Windows cannot replace a running exe)" }

if (-not $SkipBuild) {
    Say "Compile and run the native tests (core and app)"
    Run "NATIVE_TESTS.ps1" { & powershell -ExecutionPolicy Bypass -File scripts\NATIVE_TESTS.ps1 }
    $AcesCache = "build\native_aces_v2\CMakeCache.txt"
    if (Test-Path $AcesCache) {
        Say "Compile the ACES v2 engine (rudra-native with the lock)"
        $acesCmake = ((Select-String -Path $AcesCache -Pattern '^CMAKE_COMMAND:INTERNAL=(.+)$').Matches[0].Groups[1].Value)
        Run "ACES engine build" { & $acesCmake --build build\native_aces_v2 --config Release --parallel --target rudra-native }
    }
    Say "Python tests: the lock golden, the release gate, the AI fill focus, the head-to-head"
    $py = @("tests\test_lock_golden.py", "tests\test_promote_aces2_release.py", "tests\test_genfill_focus.py") | Where-Object { Test-Path $_ }
    Run "pytest" { & $Python -m pytest -q @py }
}
if (-not $SkipPackage) {
    Say "The public Windows package, locally (no weights; what release.yml builds)"
    Run "PACKAGE_WINDOWS.ps1 -PublicRelease" { & powershell -ExecutionPolicy Bypass -File scripts\PACKAGE_WINDOWS.ps1 -Python $Python -PublicRelease -NoInstaller }
}

Say "Commit"
Run "git add" { git add -A -- native rudra scripts tools tests docs configs .github }
$staged = @(git diff --cached --name-only)
if ($staged.Count) {
    $bad = $staged | Where-Object { $_ -match '\.(pt|ts|onnx|safetensors|ckpt|exr|mp4|mov)$' -or $_ -match '^(dist|reports|tmp|build)/' }
    $big = $staged | Where-Object { (Test-Path $_) -and (Get-Item $_).Length -gt 5MB }
    if ($bad -or $big) {
        git reset -q
        $list = (@($bad) + @($big) | Select-Object -Unique) -join ", "
        Fail "these must not be committed: $list"
    }
    Write-Host "$($staged.Count) file(s):"; $staged | ForEach-Object { Write-Host "  $_" }
    if (-not $Message) {
        $Message = "RUDRA ${Version}: RUDRA ACES 2 with the native lock (contract 3), real-time playback, one-frame AI fill preview, " +
                   "multi-area AI fill, warm/blue region tint, the window fits the screen. Notes: $Notes"
    }
    if ($Message -match ("Cl" + "aude|anthrop" + "ic")) { Fail "the commit message names the former assistant (AGENTS.md)" }
    Run "git commit" { git commit -q -m $Message }
} else { Write-Host "nothing new to commit" }

$ahead = git log "origin/main..$Branch" --format="%h %s"
if (-not $ahead) { Fail "$Branch has nothing that origin/main lacks: already merged? run with -Tag" }
Say "$Branch is $(@($ahead).Count) commit(s) ahead of origin/main"
$ahead | Select-Object -First 15 | ForEach-Object { Write-Host "  $_" }
if (git log "origin/main..$Branch" --format=%B | Select-String -Pattern ("Cl" + "aude|anthrop" + "ic\.com") -Quiet) {
    Write-Host "  Some commit messages name the former assistant: merge with 'Squash and merge' so main gets one commit with your message" -ForegroundColor Yellow
}

Say "Push $Branch"
Run "git push" { git push -u origin $Branch }
$body = "RUDRA ${Version}: RUDRA ACES 2 with the native lock (packages of contract 3), real-time playback, the AI fill previewed on one frame and filled in several areas, the region tint warm for highlights and blue for shadows, and a window that fits the screen. Notes: $Notes`n`nMerge with Squash and merge, then: scripts\RELEASE_1_2_0.ps1 -Tag"
if ($HasGh) {
    $prJson = (gh pr list --head $Branch --state open --json number,url 2>$null | Out-String).Trim()
    $pr = if ($prJson) { @($prJson | ConvertFrom-Json) } else { @() }
    if ($pr.Count) { Write-Host "  PR #$($pr[0].number) is open: $($pr[0].url)" -ForegroundColor Green }
    else { Run "gh pr create" { gh pr create --base main --head $Branch --title "RUDRA $Version" --body $body } }
} else {
    Write-Host "  gh is not installed: open https://github.com/FXTD-Studios/RUDRA/compare/main...$Branch" -ForegroundColor Yellow
}
Say "Next: CI green, merge the PR (Squash and merge), then: scripts\RELEASE_1_2_0.ps1 -Tag"
