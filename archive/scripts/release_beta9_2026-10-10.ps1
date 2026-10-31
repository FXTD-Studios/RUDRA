<#
.SYNOPSIS
  Commit beta 9 (movie exports with the app's settings; playback with the
  model) in three commits, push `native`, and, with -Tag once the native
  checks are green, tag v0.9.0-beta.9.

.DESCRIPTION
  Run from PowerShell on Windows (AGENTS.md: never commit from a mounted
  shell). Each commit names its own paths (git commit -- <paths>), so the
  order of staging does not matter and nothing else is swept in.

    .\scripts\release_beta9_2026-10-10.ps1          # commit and push native
    .\scripts\release_beta9_2026-10-10.ps1 -Tag     # then, CI green: tag and push the tag
#>
[CmdletBinding()]
param([switch]$Tag, [string]$Version = "v0.9.0-beta.9")

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
    @{ msg = "Movie exports carry the app's settings: grade_master (the master's chain) per frame in video/convert when the job names a grade file; analytic exports need no package; queue option grade";
       paths = @("native/infer/include/rudra/infer/tiler.hpp", "native/infer/src/tiler.cpp",
                 "native/deliver/include/rudra/deliver/master.hpp", "native/deliver/src/master.cpp", "native/deliver/src/queue.cpp",
                 "native/video/include/rudra/video/predictor.hpp", "native/video/src/predictor.cpp",
                 "native/video/include/rudra/video/convert.hpp", "native/video/src/convert.cpp", "native/video/src/queue_runner.cpp",
                 "native/app/video_queues.hpp", "native/app/video_queues.cpp",
                 "native/tests/test_video_graded.cpp", "native/tests/test_video_convert_real.cpp", "native/tests/CMakeLists.txt") },
    @{ msg = "Playback with the model: inference ahead of the playhead (FrameEngine::set_playing), a proxy at 960 px refined where it stops, a movie decoded in order (MovieFeeder), keep time by skipping; Clip > While playing";
       paths = @("native/core/include/rudra/core/fields.hpp", "native/core/src/fields.cpp", "native/core/CMakeLists.txt",
                 "native/engine/include/rudra/engine/frame_engine.hpp", "native/engine/src/frame_engine.cpp", "native/engine/src/actions.cpp",
                 "native/media/include/rudra/media/video_decode.hpp", "native/media/src/video_decode.cpp",
                 "native/app/main_window.hpp", "native/app/main_window.cpp", "native/app/playback.hpp", "native/app/playback.cpp",
                 "native/app/CMakeLists.txt", "native/app/tests/test_app.cpp", "native/tests/test_engine.cpp") },
    @{ msg = "Beta 9: version beta.9, release notes, user guide (Play), status";
       paths = @("native/CMakeLists.txt", "docs/BETA.md", "docs/USER_GUIDE.md", "STATUS.md", "scripts/release_beta9_2026-10-10.ps1") }
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

Invoke-Git push origin native
Write-Host "`nnative pushed. When the native checks are green:" -ForegroundColor Green
Write-Host "  .\scripts\release_beta9_2026-10-10.ps1 -Tag"
