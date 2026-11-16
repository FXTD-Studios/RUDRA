<#
.SYNOPSIS
  Ship RUDRA 1.2.5 (16 Nov 2026): compile and test, commit and push, open the PR, then tag the
  release and rebuild the website guide. Two runs, because a person merges the PR in between:

    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_1_2_5.ps1 -DryRun   # what it would do
    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_1_2_5.ps1           # 1. build, test, package, commit, push, PR
    #   CI green, merge the PR on GitHub with "Squash and merge", then:
    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_1_2_5.ps1 -Tag      # 2. tag v1.2.5: GitHub builds and publishes it

  1.2.5: the 1.2.4 feedback fixes (Use the model, drag and drop, Model packages), the privacy page
  and update-check choice, and Windows code signing through SignPath (docs/SIGNPATH_SETUP.md).
  No new model: every model stays as 1.2.4 published it. -Tag refuses until the team roles of
  docs/CODE_SIGNING_POLICY.md are filled in, and reminds you to approve the two signing requests.
  Commit messages credit FXTD Studios Research only (AGENTS.md).
#>
param(
    [switch]$Tag,
    [switch]$DryRun,
    [switch]$SkipBuild,
    [switch]$SkipPackage,
    [string]$Branch = "release/1.2.5",
    [string]$Version = "1.2.5",
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$MessageFile = "tmp\commit_msg_1_2_5_release.txt",
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

# The signing policy names its team before anything is tagged (SignPath, and the website guide, ask for it).
function Assert-SigningRoles {
    if (Select-String -Path docs\CODE_SIGNING_POLICY.md -Pattern "to fill in" -SimpleMatch -Quiet) {
        Fail "docs/CODE_SIGNING_POLICY.md: fill in the team roles (GitHub accounts) first"
    }
}

Say "Fetch origin"
git fetch --no-tags origin
if ($LASTEXITCODE -ne 0) { Fail "git fetch origin" }

# ==== 2. tag the merge =================================================================
if ($Tag) {
    Say "Check origin/main before tagging $TagName"
    $cm = git show "origin/main:native/CMakeLists.txt" | Select-String -Pattern "project\(RudraNative VERSION $([regex]::Escape($Version))" -Quiet
    if (-not $cm) { Fail "native/CMakeLists.txt on origin/main is not VERSION ${Version}: merge the PR first" }
    Assert-SigningRoles
    foreach ($f in $Notes, "docs/CODE_SIGNING_POLICY.md", "docs/PRIVACY.md", "scripts/PACKAGE_WINDOWS_ARCHIVE.ps1", ".signpath/artifact-configurations/app.xml") {
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
    Write-Host "  https://github.com/FXTD-Studios/RUDRA/actions" -ForegroundColor Green
    Write-Host "  Without SignPath's secret (before its approval) the Windows job publishes unsigned, with a warning." -ForegroundColor Yellow
    Write-Host "  Once SignPath is set up it waits for TWO signing requests (the programs, then the installer):" -ForegroundColor Yellow
    Write-Host "  approve each within two hours, or the job fails and nothing is published (docs/SIGNPATH_SETUP.md)." -ForegroundColor Yellow
    $Guide = Join-Path $Website "rudra-docs"
    if ((Test-Path (Join-Path $Guide "index.html")) -and (Test-Path $Python) -and (Test-Path "tools\build_user_guide.py")) {
        Say "Rebuild the user guide for $Version into $Guide"
        Run "build_user_guide" { & $Python tools/build_user_guide.py --version $Version --tag $TagName --release --out $Guide }
        Write-Host "  Deploy $Guide once the release is published (its download buttons point at $TagName)." -ForegroundColor Green
    } else {
        Write-Host "  No guide at $Guide (the website repository beside this one): run tools/build_user_guide.py --version $Version --tag $TagName --release --out <website>\rudra-docs" -ForegroundColor Yellow
    }
    exit 0
}

# ==== 1. build, test, package, commit, push, PR ========================================
Say "Check the release is ready on $Branch"
$cur = (git rev-parse --abbrev-ref HEAD).Trim()
if ($cur -ne $Branch) { Fail "on branch $cur, not ${Branch}: git switch $Branch (or -Branch $cur)" }
if (-not (Select-String -Path native\CMakeLists.txt -Pattern "project\(RudraNative VERSION $([regex]::Escape($Version))" -Quiet)) { Fail "native/CMakeLists.txt is not VERSION $Version" }
foreach ($f in $Notes, $MessageFile, "docs\CODE_SIGNING_POLICY.md", "docs\PRIVACY.md") { if (-not (Test-Path $f)) { Fail "no $f" } }
if (-not (Test-Path $Python)) { Fail "no Python at $Python" }
if (Get-Process RUDRA -ErrorAction SilentlyContinue) { Fail "RUDRA is running: close it (Windows cannot replace a running exe)" }
$msg = Get-Content $MessageFile -Raw
if ($msg -match $Assistant) { Fail "$MessageFile names the former assistant (AGENTS.md)" }
if ($msg -notmatch [regex]::Escape($Credit)) { Fail "$MessageFile lacks the credit line: $Credit" }

Say "No third-party product names in the tracked files"
$Banned = "bee" + "ble|swi" + "tch ?hdr|light" + "ricks|\bl" + "tx"   # in pieces, as $Assistant: this check must not find itself
$names = git --no-optional-locks grep -n -I -i -E $Banned -- . ":(exclude)tmp" ":(exclude)dist"
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
    Run "PACKAGE_WINDOWS.ps1 -PublicRelease -NoArchive" { & powershell -ExecutionPolicy Bypass -File scripts\PACKAGE_WINDOWS.ps1 -Python $Python -PublicRelease -NoArchive }
    # The ZIP and the installer as release.yml makes them after signing (here unsigned): the installer's
    # Privacy page and update-check choice are compiled before the tag.
    $PkgDir = "dist\beta\RUDRA-$Version-windows-x64"
    Run "PACKAGE_WINDOWS_ARCHIVE.ps1" { & powershell -ExecutionPolicy Bypass -File scripts\PACKAGE_WINDOWS_ARCHIVE.ps1 -Package $PkgDir }
}

Say "Commit"
Run "git add" { git add -A -- native rudra scripts tools tests docs configs config pipeline training archive ui .github .signpath README.md pyproject.toml SECURITY.md .gitignore }
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
$body = "RUDRA ${Version}: shots open again after an update (Use the model), drag and drop says what it cannot open, Model packages wording, a privacy page and update-check choice, and Windows code signing through SignPath. Notes: $Notes"
if ($HasGh) {
    $prJson = (gh pr list --head $Branch --state open --json number,url 2>$null | Out-String).Trim()
    $pr = if ($prJson) { @($prJson | ConvertFrom-Json) } else { @() }
    if ($pr.Count) { Write-Host "  PR #$($pr[0].number) is open: $($pr[0].url)" -ForegroundColor Green }
    else { Run "gh pr create" { gh pr create --base main --head $Branch --title "RUDRA $Version" --body $body } }
} else {
    Write-Host "  gh is not installed: open https://github.com/FXTD-Studios/RUDRA/compare/main...$Branch" -ForegroundColor Yellow
}
Say "Next: CI green, merge the PR (Squash and merge), then: scripts\RELEASE_1_2_5.ps1 -Tag"
