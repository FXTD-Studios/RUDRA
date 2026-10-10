<#
.SYNOPSIS
  Commit beta 8 (the analytic reconstruction as the default, after the 10 Oct
  external review; Node 24 actions) in three commits, push `native`, and,
  with -Tag once the native checks are green, tag v0.9.0-beta.8.

.DESCRIPTION
  Run from PowerShell on Windows (AGENTS.md: never commit from a mounted
  shell). Each commit names its own paths (git commit -- <paths>), so the
  order of staging does not matter and nothing else is swept in.

    .\scripts\release_beta8_2026-10-10.ps1          # commit and push native
    .\scripts\release_beta8_2026-10-10.ps1 -Tag     # then, CI green: tag and push the tag
#>
[CmdletBinding()]
param([switch]$Tag, [string]$Version = "v0.9.0-beta.8")

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
    @{ msg = "Analytic reconstruction by default: no model runs for the view or a master unless Reconstruct > Use the model is on (analytic_fields, kAnalyticConstants); masters record it (rudra:reconstruction, sidecar reconstruction); the CLI master honours analytic; projects keep the choice; the status bar says which is in use";
       paths = @("native/core/include/rudra/core/fields.hpp", "native/core/include/rudra/core/composite.hpp",
                 "native/deliver/include/rudra/deliver/master.hpp", "native/deliver/src/master.cpp", "native/cli/master.cpp",
                 "native/engine/src/actions.cpp", "native/engine/include/rudra/engine/project.hpp", "native/engine/src/project.cpp",
                 "native/app/main_window.hpp", "native/app/main_window.cpp", "native/app/workflow_check.cpp",
                 "native/app/tests/test_app.cpp", "native/tests/test_composite.cpp", "native/tests/test_project.cpp") },
    @{ msg = "CI: actions on Node 24 (cache v5, upload-artifact v6, download-artifact v7, action-gh-release v3)";
       paths = @(".github/workflows/native.yml", ".github/workflows/release.yml") },
    @{ msg = "Beta 8: version beta.8, release notes (the analytic default, what movie exports still do), user guide";
       paths = @("native/CMakeLists.txt", "docs/BETA.md", "docs/USER_GUIDE.md", "STATUS.md", "scripts/release_beta8_2026-10-10.ps1") }
)

foreach ($c in $commits) {
    $changed = @(git status --porcelain -- $c.paths)
    if ($changed.Count -eq 0) { Write-Host "nothing to commit for: $($c.msg.Substring(0, 50))..."; continue }
    Invoke-Git add -- $c.paths
    Invoke-Git commit -m $c.msg -- $c.paths
}

$left = @(git status --porcelain --untracked-files=no)
if ($left.Count) { Write-Host "Still modified, not committed (left for you):" -ForegroundColor Yellow; $left | Write-Host }

Invoke-Git push origin native
Write-Host "`nnative pushed. When the native checks are green:" -ForegroundColor Green
Write-Host "  .\scripts\release_beta8_2026-10-10.ps1 -Tag"
