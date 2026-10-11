<#
.SYNOPSIS
  Commit beta 10 (version and notes; the static hold is in 0dbad8f), push
  `native`, and, with -Tag once the native checks are green, tag
  v0.9.0-beta.10.

.DESCRIPTION
  Run from PowerShell on Windows (AGENTS.md: never commit from a mounted
  shell). Each commit names its own paths (git commit -- <paths>), so the
  order of staging does not matter and nothing else is swept in.

    .\scripts\release_beta10_2026-10-11.ps1          # commit and push native
    .\scripts\release_beta10_2026-10-11.ps1 -Tag     # then, CI green: tag and push the tag
#>
[CmdletBinding()]
param([switch]$Tag, [string]$Version = "v0.9.0-beta.10")

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
# Not named "Git": PowerShell names are case-insensitive, so a function Git
# would call itself instead of git.exe.
function Invoke-Git { & git @args; if ($LASTEXITCODE -ne 0) { Fail "git $args" } }

if (Test-Path .git\index.lock) { Fail ".git\index.lock exists: move it to .git\_stale_locks\ first (AGENTS.md)" }
if ((git rev-parse --abbrev-ref HEAD) -ne "native") { Fail "not on the native branch" }

if ($Tag) {
    if (git tag --list $Version) { Fail "$Version exists already" }
    Invoke-Git tag -a $Version -m "RUDRA $Version (docs/BETA.md)"
    Invoke-Git push origin $Version
    Write-Host "`n$Version pushed: the release workflow builds Windows and macOS and publishes the pre-release." -ForegroundColor Green
    Write-Host "Watch it: https://github.com/FXTD-Studios/RUDRA/actions/workflows/release.yml"
    exit 0
}

$commits = @(
    @{ msg = "Beta 10: version beta.10, release notes (the static hold; beta 9 was tagged before its notes), status";
       paths = @("native/CMakeLists.txt", "docs/BETA.md", "STATUS.md", "scripts/release_beta10_2026-10-11.ps1") }
)

# Credit FXTD Studios Research on every commit (no other co-author).
$coauthor = "Co-authored-by: FXTD Studios Research <research@fxtdstudios.com>"

foreach ($c in $commits) {
    $changed = @(git status --porcelain -- $c.paths)
    if ($changed.Count -eq 0) { Write-Host "nothing to commit for: $($c.msg.Substring(0, 50))..."; continue }
    Invoke-Git add -- $c.paths
    Invoke-Git commit -m $c.msg -m $coauthor -- $c.paths
}

$left = @(git status --porcelain --untracked-files=no)
if ($left.Count) { Write-Host "Still modified, not committed (left for you):" -ForegroundColor Yellow; $left | Write-Host }

# AGENTS.md: no AI assistant credited on GitHub. Refuse to push if a commit names one.
$credits = (git log origin/native..HEAD --format=%B) -join "`n"
if ($credits -match '(?i)anthropic\.com|co-authored-by:\s*claude|claude-session') { Fail "a commit to push credits Claude; amend it first (AGENTS.md)" }

Invoke-Git push origin native
Write-Host "`nnative pushed. When the native checks are green:" -ForegroundColor Green
Write-Host "  .\scripts\release_beta10_2026-10-11.ps1 -Tag"
