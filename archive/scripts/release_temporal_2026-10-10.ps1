<#
.SYNOPSIS
  Commit temporal consistency for video (beta 9) and push
  `native`. Beta 9 is tagged afterwards with release_beta9_2026-10-10.ps1 -Tag
  once the native checks are green (it tags HEAD).

.DESCRIPTION
  Run from PowerShell on Windows (AGENTS.md: never commit from a mounted
  shell). Each commit names its own paths (git commit -- <paths>), so nothing
  else is swept in.

    .\scripts\release_temporal_2026-10-10.ps1
    # native checks green:
    .\scripts\release_beta9_2026-10-10.ps1 -Tag
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
function Invoke-Git { & git @args; if ($LASTEXITCODE -ne 0) { Fail "git $args" } }

if (Test-Path .git\index.lock) { Fail ".git\index.lock exists: move it to .git\_stale_locks\ first (AGENTS.md)" }
if ((git rev-parse --abbrev-ref HEAD) -ne "native") { Fail "not on the native branch" }

$commits = @(
    # Committed already (bbf6726): the shot hold and the flicker measure.
    @{ msg = "Static hold for video: where the source pixel is still, the HDR's gain over it held with the shot's retention (ShadowSmoother.hold_pixels; never where clipped, crushed or moved; --static-hold on|off, default on); local_flicker_ev in every movie report; goldens re-emitted";
       paths = @("rudra/video.py", "tests/test_temporal_hold_2026_10_10.py", "tools/emit_video_convert_golden.py",
                 "native/core/include/rudra/core/video_predict.hpp", "native/core/src/video_predict.cpp",
                 "native/video/include/rudra/video/convert.hpp", "native/video/src/convert.cpp",
                 "native/video/src/queue_runner.cpp", "native/cli/video.cpp", "native/deliver/src/queue.cpp",
                 "native/tests/test_video_predict.cpp", "native/tests/test_video_graded.cpp",
                 "native/tests/test_video_convert_real.cpp",
                 "native/tests/golden/video_convert", "native/tests/golden/video_queue") },
    # AGENTS.md and scripts/archive/ are ignored by .gitignore: local only, not committed.
    @{ msg = "Beta 9 notes: steady within a shot, the static hold, the flicker measures; user guide, architecture 6.5, status";
       paths = @("docs/BETA.md", "docs/USER_GUIDE.md", "docs/NATIVE_ARCHITECTURE.md", "STATUS.md",
                 "scripts/release_temporal_2026-10-10.ps1") }
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
if ($credits -match '(?i)anthropic\.com|co-authored-by:\s*claude|claude-session') { Fail "a commit to push credits an AI assistant; amend it first (AGENTS.md)" }

Invoke-Git push origin native
Write-Host "`nnative pushed. When the native checks are green:" -ForegroundColor Green
Write-Host "  .\scripts\release_beta9_2026-10-10.ps1 -Tag"
