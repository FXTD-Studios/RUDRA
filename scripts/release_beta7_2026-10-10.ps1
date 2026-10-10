<#
.SYNOPSIS
  Commit the native port of the 9 Oct ship-path fixes (beta 7) in three
  commits, push `native`, and, with -Tag once the native checks are green,
  tag v0.9.0-beta.7 so the release workflow publishes the pre-release.

.DESCRIPTION
  Run from PowerShell on Windows (AGENTS.md: never commit from a mounted
  shell). Each commit names its own paths (git commit -- <paths>), so the
  order of staging does not matter and nothing else is swept in.

    .\scripts\release_beta7_2026-10-10.ps1          # commit and push native
    .\scripts\release_beta7_2026-10-10.ps1 -Tag     # then, CI green: tag and push the tag
#>
[CmdletBinding()]
param([switch]$Tag, [string]$Version = "v0.9.0-beta.7")

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
# Not named "Git": PowerShell names are case-insensitive, so a function Git
# would call itself instead of git.exe.
function Invoke-Git { & git @args; if ($LASTEXITCODE -ne 0) { Fail "git $args" } }

if (Test-Path .git\index.lock) { Fail ".git\index.lock exists: move it to .git\_stale_locks\ first (AGENTS.md)" }
if ((git rev-parse --abbrev-ref HEAD) -ne "native") { Fail "not on the native branch" }

$trailer = "`n`nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`nClaude-Session: https://claude.ai/code/session_01NjxFqr14DzbjUjMD8BL4G7"

if ($Tag) {
    if (git tag --list $Version) { Fail "$Version exists already" }
    Invoke-Git tag -a $Version -m "RUDRA $Version (docs/BETA.md)"
    Invoke-Git push origin $Version
    Write-Host "`n$Version pushed: the release workflow builds Windows and macOS and publishes the pre-release." -ForegroundColor Green
    Write-Host "Watch it: https://github.com/FXTD-Studios/RUDRA/actions/workflows/release.yml"
    exit 0
}

$commits = @(
    @{ msg = "Native follows the 9 Oct Python fixes: deliver HLG with the log10 system gamma and hue-preserving scale, Rec.709/P3 into Rec.2020 before the shoulder, a partial file renamed into place; masters grade after the anchor, chroma and grain, and the linear container is converted to Rec.2020 and tagged; goldens re-emitted";
       paths = @("native/deliver/include/rudra/deliver/sequence_encode.hpp", "native/deliver/src/sequence_encode.cpp",
                 "native/cli/deliver.cpp", "native/core/src/master.cpp", "native/deliver/src/master.cpp",
                 "native/tests/test_sequence_encode.cpp", "native/tests/test_composite.cpp",
                 "native/tests/golden/sequence_encode", "native/tests/golden/composite", "native/tests/golden/master",
                 "tools/emit_sequence_encode_golden.py", "tools/emit_composite_golden.py") },
    @{ msg = "Viewer: a Region EV push survives the live anchor (the composite carries the grade gain to the anchor's target; the measured hold is the ungraded one); Qt shell build: the project tests no longer need still decode";
       paths = @("native/core/include/rudra/core/composite.hpp", "native/core/src/composite.cpp",
                 "native/core/include/rudra/core/view.hpp", "native/core/src/view.cpp", "native/engine/src/measure.cpp",
                 "native/render/shaders/composite.frag", "native/tests/test_anchor_view.cpp", "native/app/tests/test_app.cpp") },
    @{ msg = "Beta 7: version beta.7, release notes since beta 6";
       paths = @("native/CMakeLists.txt", "docs/BETA.md", "STATUS.md", "scripts/release_beta7_2026-10-10.ps1") }
)

foreach ($c in $commits) {
    $changed = @(git status --porcelain -- $c.paths)
    if ($changed.Count -eq 0) { Write-Host "nothing to commit for: $($c.msg.Substring(0, 50))..."; continue }
    Invoke-Git add -- $c.paths
    Invoke-Git commit -m ($c.msg + $trailer) -- $c.paths
}

$left = @(git status --porcelain --untracked-files=no)
if ($left.Count) { Write-Host "Still modified, not committed (left for you):" -ForegroundColor Yellow; $left | Write-Host }

Invoke-Git push origin native
Write-Host "`nnative pushed. When the native checks are green:" -ForegroundColor Green
Write-Host "  .\scripts\release_beta7_2026-10-10.ps1 -Tag"
