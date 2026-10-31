<#
.SYNOPSIS
  Commit the temporal follow-ups and push `native`.

.DESCRIPTION
  The static hold inside clipped highlights (the HDR's own change decides),
  playback no longer carrying a late anchor measure across a loop or a cut,
  tools/measure_temporal.py for footage; goldens re-emitted. Notes in a second
  commit; docs/NEXT_TASKS.md is local (.gitignore NEXT_*.md).

    .\scripts\release_temporal_followups_2026-10-11.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
# Not named "Git": PowerShell names are case-insensitive, so a function Git
# would call itself instead of git.exe.
function Invoke-Git { & git @args; if ($LASTEXITCODE -ne 0) { Fail "git $args" } }

if (Test-Path .git\index.lock) { Fail ".git\index.lock exists: move it to .git\_stale_locks\ first (AGENTS.md)" }
if ((git rev-parse --abbrev-ref HEAD) -ne "native") { Fail "not on the native branch" }

$commits = @(
    @{ msg = "Temporal follow-ups: inside highlights clipped in both frames the static hold follows the HDR's own change (held under 0.05 EV, its own over 0.25); playback's anchor hold ignores a late measure from another shot or pass (PlaybackHold::shot); tools/measure_temporal.py and make_temporal_clips.py; rudra-native video --grade; goldens re-emitted";
       paths = @("rudra/video.py", "tests/test_temporal_hold_2026_10_10.py", "tools/measure_temporal.py",
                 "native/core/src/video_predict.cpp", "native/core/include/rudra/core/video_predict.hpp",
                 "native/engine/src/playback_hold.cpp", "native/engine/include/rudra/engine/playback_hold.hpp",
                 "native/app/main_window.cpp", "native/app/main_window.hpp",
                 "native/tests/test_engine.cpp", "native/tests/test_video_predict.cpp", "native/cli/video.cpp",
                 "tools/make_temporal_clips.py",
                 "native/tests/golden/video_convert", "native/tests/golden/video_queue") },
    @{ msg = "Notes: steadier highlights, the loop fix, the first flicker measurement on real footage (docs/TEMPORAL_RESULTS.md, release notes for the next beta, status)";
       paths = @("docs/BETA.md", "STATUS.md", "docs/TEMPORAL_RESULTS.md", "scripts/release_temporal_followups_2026-10-11.ps1") }
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
Write-Host "`nnative pushed. Watch the native checks: https://github.com/FXTD-Studios/RUDRA/actions" -ForegroundColor Green
