<#
.SYNOPSIS
  Ship RUDRA 1.1.0 (5 Nov 2026) from feat/gap-kits. It carries the 1.0.3 licence
  fix, which the v1.0.3 tag on GitHub does NOT contain (that tag is main at
  ce51d86, whose release workflow still exports the non-commercial
  sdr2hdr_image_v8). Merge the PR with "Squash and merge": main then gets one
  commit, yours, with the message in docs/RELEASE_NOTES_1.1.0.md.

    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_1_1_0.ps1 -DryRun
    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_1_1_0.ps1          # push the branch, open the PR
    # review and merge the PR on GitHub, then:
    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_1_1_0.ps1 -Tag     # tag v1.0.3 on main, push it, rebuild the user guide

  Pushing the tag runs .github/workflows/release.yml, which builds both
  packages, refuses any that carry weights, and publishes them with
  docs/RELEASE_NOTES_1.0.3.md. Nothing is merged for you.
#>
param(
    [switch]$Tag,
    [switch]$DryRun,
    [string]$Branch = "feat/gap-kits",
    [string]$Version = "1.1.0"
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

Say "Fetch origin"
git fetch --no-tags origin
if ($LASTEXITCODE -ne 0) { Fail "git fetch origin" }

if (-not $Tag) {
    # ---- 1. push the branch and open the PR -----------------------------------
    git rev-parse --verify --quiet "refs/heads/$Branch" | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail "no local branch $Branch" }
    $ahead = git log "origin/main..$Branch" --format="%h %s"
    if (-not $ahead) { Fail "$Branch has nothing that origin/main lacks: already merged? run with -Tag" }
    Say "$Branch is $(@($ahead).Count) commit(s) ahead of origin/main"
    $ahead | ForEach-Object { Write-Host "  $_" }
    if (git log "origin/main..$Branch" --format=%B | Select-String -Pattern ("Cl" + "aude|anthrop" + "ic\.com") -Quiet) {
        Write-Host "  Some commit messages mention the former assistant or its domain (AGENTS.md): merge the PR with 'Squash and merge' so main gets one commit with your message" -ForegroundColor Yellow
    }
    $cm = git show "${Branch}:native/CMakeLists.txt" | Select-String -Pattern "project\(RudraNative VERSION $([regex]::Escape($Version))" -Quiet
    if (-not $cm) { Fail "native/CMakeLists.txt on $Branch is not VERSION $Version" }
    if (-not (git ls-tree --name-only $Branch "docs/RELEASE_NOTES_$Version.md")) { Fail "docs/RELEASE_NOTES_$Version.md is not on $Branch" }

    Say "Push $Branch"
    Run "git push" { git push -u origin $Branch }

    $body = "RUDRA ${Version}: the viewer bar, Clean shadows, AI fill in place, AMF and alpha, real-time playback; carries the 1.0.3 licence fix (no model weights in the packages). Notes: docs/RELEASE_NOTES_$Version.md.`n`nMerge with Squash and merge, then: scripts\RELEASE_1_1_0.ps1 -Tag"
    if (Get-Command gh -ErrorAction SilentlyContinue) {
        Say "Open the PR with gh"
        Run "gh pr create" { gh pr create --base main --head $Branch --title "RUDRA $Version" --body $body }
    } else {
        Write-Host "  gh is not installed: open https://github.com/FXTD-Studios/RUDRA/compare/main...$Branch" -ForegroundColor Yellow
    }
    Say "Next: review and merge the PR (CI green), then run this script with -Tag"
    exit 0
}

# ---- 2. tag the merge -----------------------------------------------------------
Say "Check origin/main before tagging $TagName"
# A squash merge has no shared commit to test: check main has the release's content instead.
if (-not (git ls-tree --name-only origin/main "tools/check_package_weights.py")) { Fail "origin/main lacks tools/check_package_weights.py (the licence fix): merge the PR first" }
if (git show "origin/main:.github/workflows/release.yml" | Select-String -Pattern "export_model.py checkpoints/sdr2hdr_image_v8" -Quiet) { Fail "origin/main's release workflow still exports sdr2hdr_image_v8: merge the PR first" }
$cm = git show "origin/main:native/CMakeLists.txt" | Select-String -Pattern "project\(RudraNative VERSION $([regex]::Escape($Version))" -Quiet
if (-not $cm) { Fail "native/CMakeLists.txt on origin/main is not VERSION $Version" }
# 7 Nov 2026: twice the tag went on main before the PR was merged (the checks above passed: main
# already had the licence fix and the version). The release's own work must be on main, and main
# must hold what the branch holds in the code and the notes.
foreach ($f in "scripts/genfill_bootstrap.ps1", "tools/update_models_index.py") {
    if (-not (git ls-tree --name-only origin/main $f)) { Fail "origin/main lacks ${f}: merge PR $Branch (Squash and merge) first" }
}
git diff --quiet origin/main "origin/$Branch" -- native rudra scripts tools "docs/RELEASE_NOTES_$Version.md"
if ($LASTEXITCODE -ne 0) {
    git diff --stat origin/main "origin/$Branch" -- native rudra scripts tools "docs/RELEASE_NOTES_$Version.md" | Select-Object -Last 5
    Fail "origin/main differs from origin/$Branch (above): merge the PR, or push what is missing, before tagging"
}
if (git ls-remote --tags origin "refs/tags/$TagName") { Fail "$TagName already exists on origin" }
if (git tag -l $TagName) { Fail "$TagName already exists locally (git tag -d $TagName if it is a mistake)" }
$sha = (git rev-parse --short origin/main).Trim()
Say "Tag $TagName on origin/main ($sha) and push it"
Run "git tag" { git tag -a $TagName origin/main -m "RUDRA $Version" }
Run "git push tag" { git push origin $TagName }
Write-Host "  The release workflow is running: https://github.com/FXTD-Studios/RUDRA/actions" -ForegroundColor Green
Write-Host "  When it is published, check the Latest release is $TagName and its zip has no models\ folder." -ForegroundColor Green

# ---- 3. the user guide for this release -----------------------------------------
$Guide = Join-Path (Split-Path $Repo -Parent) "website\rudra-docs"
$Py = ".venv\Scripts\python.exe"
if ((Test-Path (Join-Path $Guide "index.html")) -and (Test-Path $Py)) {
    Say "Rebuild the user guide for $Version into $Guide"
    Run "build_user_guide" { & $Py tools/build_user_guide.py --version $Version --tag $TagName --release --out $Guide }
    Write-Host "  Deploy $Guide once the release is published (its download buttons point at $TagName)." -ForegroundColor Green
} else {
    Write-Host "  No $Guide\index.html or no ${Py}: rebuild the guide by hand (tools/build_user_guide.py)" -ForegroundColor Yellow
}
