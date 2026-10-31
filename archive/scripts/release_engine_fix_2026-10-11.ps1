<#
.SYNOPSIS
  Commit the frame engine's delivery fix and push `native`.

.DESCRIPTION
  Native CI on 235138f failed FrameEngine.ScrubbingAFolderNeverShowsAStaleFrame
  on Windows only (1 of 220). The engine checked that a frame was still the
  one asked for and then handed it to the callback outside its lock, so a
  show() in between could let the previous request's frame through. A
  delivery mutex now covers the check and the hand-over, and show() takes it
  first, so after show(i) returns no older frame can arrive. The test now
  names the request in flight as well as the last one. Reproduced on Linux
  with a delay injected between the check and the callback: about 100 stale
  frames per run before, none after. The app re-checks on the UI thread, so
  it never showed a stale frame; this makes the engine's own promise hold.

    .\scripts\release_engine_fix_2026-10-11.ps1
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
    @{ msg = "Frame engine: a delivery and show() no longer interleave (delivery mutex), so no frame of an older request arrives after show() returns; the scrub test names the request in flight";
       paths = @("native/engine/src/frame_engine.cpp", "native/engine/include/rudra/engine/frame_engine.hpp",
                 "native/tests/test_engine.cpp", "scripts/release_engine_fix_2026-10-11.ps1") }
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
