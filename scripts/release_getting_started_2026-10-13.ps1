<#
  Getting started for 1.0: the welcome says the model is optional, the five
  steps cover the Source picker, exposure, the clip fill, the HDR metadata and
  Hold steady, and show once more after updating; README quick start, the user
  guide and the beta notes match. Commits and pushes `native`.

    .\scripts\release_getting_started_2026-10-13.ps1
#>
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
function Invoke-Git { & git @args; if ($LASTEXITCODE -ne 0) { Fail "git $args" } }
if (Test-Path .git\index.lock) { Fail ".git\index.lock exists: move it to .git\_stale_locks\ first (AGENTS.md)" }
if ((git rev-parse --abbrev-ref HEAD) -ne "native") { Fail "not on the native branch" }

$coauthor = "Co-authored-by: FXTD Studios Research <research@fxtdstudios.com>"
$commits = @(
    @{ msg = "Getting started for 1.0: the welcome says the model is optional; the five steps cover the Source picker (BT.1886 by default), exposure, Fill clipped highlights, the HDR10+ and Dolby Vision metadata and Hold steady, and show once per edition (QSettings help/gettingStarted 2), so beta users see them once more";
       paths = @("native/app/main_window.cpp", "native/app/main_window.hpp", "native/app/model_dialogs.cpp", "native/app/tests/test_app.cpp") },
    @{ msg = "Docs: the desktop quick start in the README (the first start and the five steps), the user guide's first start and the clip fill's limits, the beta notes";
       paths = @("README.md", "docs/USER_GUIDE.md", "docs/BETA.md", "scripts/release_getting_started_2026-10-13.ps1") }
)
foreach ($c in $commits) {
    if (@(git status --porcelain -- $c.paths).Count -eq 0) { Write-Host "nothing to commit for: $($c.msg.Substring(0, 40))..."; continue }
    Invoke-Git add -- $c.paths
    Invoke-Git commit -m $c.msg -m $coauthor -- $c.paths
}
$left = @(git status --porcelain --untracked-files=no)
if ($left.Count) { Write-Host "Still modified, not committed (left for you):" -ForegroundColor Yellow; $left | Write-Host }
$credits = (git log origin/native..HEAD --format=%B) -join "`n"
if ($credits -match '(?i)anthropic\.com|co-authored-by:\s*claude|claude-session') { Fail "a commit to push credits Claude; amend it first (AGENTS.md)" }
Invoke-Git push origin native
Write-Host "native pushed. With the checks green, tag the candidate: .\scripts\release_v1_rc.ps1 -Rc 2 -Tag" -ForegroundColor Green
