<#
.SYNOPSIS
  RUDRA 1.0 release candidates, and 1.0 itself (docs/RELEASE_V1.md, step 10).

.DESCRIPTION
  Run from PowerShell on Windows (AGENTS.md: never commit from a mounted
  shell), on `native`, after the v1 checklist in docs/RELEASE_V1.md.

    .\scripts\release_v1_rc.ps1                  # version 1.0.0-rc.1: commit and push native
    .\scripts\release_v1_rc.ps1 -Tag             # CI green: tag v1.0.0-rc.1 (a pre-release)
    .\scripts\release_v1_rc.ps1 -Rc 2            # the next candidate after fixes
    .\scripts\release_v1_rc.ps1 -Final           # version 1.0.0 (no suffix): commit and push
    .\scripts\release_v1_rc.ps1 -Final -Tag      # tag v1.0.0: the release workflow publishes a release

  The version lives in native/CMakeLists.txt (project VERSION and RUDRA_RELEASE);
  the app, the installers and the update check read it from there.
#>
[CmdletBinding()]
param([int]$Rc = 1, [switch]$Final, [switch]$Tag)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
function Invoke-Git { & git @args; if ($LASTEXITCODE -ne 0) { Fail "git $args" } }

if (Test-Path .git\index.lock) { Fail ".git\index.lock exists: move it to .git\_stale_locks\ first (AGENTS.md)" }
if ((git rev-parse --abbrev-ref HEAD) -ne "native") { Fail "not on the native branch" }

$label = if ($Final) { "" } else { "rc.$Rc" }
$version = if ($Final) { "v1.0.0" } else { "v1.0.0-rc.$Rc" }

if ($Tag) {
    if (git tag --list $version) { Fail "$version exists already" }
    $cmake = [System.IO.File]::ReadAllText((Join-Path (Get-Location) "native\CMakeLists.txt"))
    if ($cmake -notmatch 'project\(RudraNative VERSION 1\.0\.0 ' -or $cmake -notmatch ('set\(RUDRA_RELEASE "' + [regex]::Escape($label) + '"')) {
        Fail "native/CMakeLists.txt is not at ${version}: run this script without -Tag first"
    }
    if ((git status --porcelain --untracked-files=no -- native/CMakeLists.txt)) { Fail "the version change is not committed" }
    Invoke-Git fetch origin native
    if ((git rev-parse HEAD) -ne (git rev-parse origin/native)) { Fail "HEAD is not origin/native: push first, and wait for the native checks" }
    Invoke-Git tag -a $version -m "RUDRA $version (docs/BETA.md, docs/RELEASE_V1.md)"
    Invoke-Git push origin $version
    Write-Host "`n$version pushed: the release workflow builds Windows and macOS and publishes it" -ForegroundColor Green
    Write-Host ("(" + $(if ($Final) { "a release" } else { "a pre-release" }) + "). https://github.com/FXTD-Studios/RUDRA/actions/workflows/release.yml")
    exit 0
}

# The version: project VERSION 1.0.0 and the label, keeping the file's line endings.
$path = Join-Path (Get-Location) "native\CMakeLists.txt"
$text = [System.IO.File]::ReadAllText($path)
$text = $text -replace 'project\(RudraNative VERSION [0-9.]+ ', 'project(RudraNative VERSION 1.0.0 '
$text = $text -replace 'set\(RUDRA_RELEASE "[^"]*"', ('set(RUDRA_RELEASE "' + $label + '"')
[System.IO.File]::WriteAllText($path, $text, (New-Object System.Text.UTF8Encoding $false))

$coauthor = "Co-authored-by: FXTD Studios Research <research@fxtdstudios.com>"
$paths = @("native/CMakeLists.txt", "scripts/release_v1_rc.ps1")
$changed = @(git status --porcelain -- $paths)
if ($changed.Count -eq 0) { Write-Host "already at $version, nothing to commit" }
else {
    Invoke-Git add -- $paths
    Invoke-Git commit -m ("RUDRA " + $version.Substring(1) + ": version") -m $coauthor -- $paths
}

# AGENTS.md: no AI assistant credited on GitHub. Refuse to push if a commit names one.
$credits = (git log origin/native..HEAD --format=%B) -join "`n"
if ($credits -match '(?i)anthropic\.com|co-authored-by:\s*claude|claude-session') { Fail "a commit to push credits an AI assistant; amend it first (AGENTS.md)" }

Invoke-Git push origin native
Write-Host "`nnative pushed at $version. When the native checks are green:" -ForegroundColor Green
Write-Host ("  .\scripts\release_v1_rc.ps1 " + $(if ($Final) { "-Final " } else { if ($Rc -ne 1) { "-Rc $Rc " } else { "" } }) + "-Tag")
