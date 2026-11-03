<#
.SYNOPSIS
  Ship RUDRA 1.0.3 on its own (3 Nov 2026). GitHub's main is still at 1.0.1:
  its release workflow puts the non-commercial sdr2hdr_image_v8 in the packages.
  The fix is the local branch fix/1.0.3-licence-and-ci (0046ee7 1.0.2, 3858348 1.0.3).

    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_1_0_3.ps1 -DryRun
    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_1_0_3.ps1          # push the branch, open the PR
    # review and merge the PR on GitHub, then:
    powershell -ExecutionPolicy Bypass -File scripts\RELEASE_1_0_3.ps1 -Tag     # tag v1.0.3 on main, push it, rebuild the user guide

  Pushing the tag runs .github/workflows/release.yml, which builds both
  packages, refuses any that carry weights, and publishes them with
  docs/RELEASE_NOTES_1.0.3.md. Nothing is merged for you.
#>
param(
    [switch]$Tag,
    [switch]$DryRun,
    [string]$Branch = "fix/1.0.3-licence-and-ci",
    [string]$Version = "1.0.3"
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
    if (git log "origin/main..$Branch" --format=%B | Select-String -Pattern "Claude|anthropic\.com" -Quiet) {
        Fail "a commit above mentions Claude or anthropic.com (AGENTS.md): reword it first"
    }
    $cm = git show "${Branch}:native/CMakeLists.txt" | Select-String -Pattern "project\(RudraNative VERSION $([regex]::Escape($Version))" -Quiet
    if (-not $cm) { Fail "native/CMakeLists.txt on $Branch is not VERSION $Version" }
    if (-not (git ls-tree --name-only $Branch "docs/RELEASE_NOTES_$Version.md")) { Fail "docs/RELEASE_NOTES_$Version.md is not on $Branch" }

    Say "Push $Branch"
    Run "git push" { git push -u origin $Branch }

    $body = "RUDRA $Version. Public packages carry no model weights (1.0.0 to 1.0.2 bundled the non-commercial sdr2hdr_image_v8); RUDRA ACES 2 labelled preview; CI tests the shipped model; release notes picked from the tag. Notes: docs/RELEASE_NOTES_$Version.md.`n`nAfter merging: scripts\RELEASE_1_0_3.ps1 -Tag"
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
git merge-base --is-ancestor 3858348 origin/main
if ($LASTEXITCODE -ne 0) { Fail "origin/main does not contain the 1.0.3 commit 3858348: merge the PR first" }
$cm = git show "origin/main:native/CMakeLists.txt" | Select-String -Pattern "project\(RudraNative VERSION $([regex]::Escape($Version))" -Quiet
if (-not $cm) { Fail "native/CMakeLists.txt on origin/main is not VERSION $Version" }
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
    Write-Host "  No $Guide\index.html or no $Py: rebuild the guide by hand (tools/build_user_guide.py)" -ForegroundColor Yellow
}
