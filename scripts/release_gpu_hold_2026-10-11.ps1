<#
.SYNOPSIS
  Commit the static hold in playback and push `native`.

.DESCRIPTION
  shaders/hold.frag runs ShadowSmoother::hold_pixels in the viewer after the
  composite while playing with steady shots on; PlaybackHold::pixel_step
  drives it; rudra-gpu-parity holds it to the C++ reference. It is built and
  checked by Gate B (Qt 6.8): watch "GPU parity (GL)" and "GPU parity
  (Vulkan)". Notes in a second commit; docs/NEXT_TASKS.md is local.

    .\scripts\release_gpu_hold_2026-10-11.ps1
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
    @{ msg = "Static hold in playback: shaders/hold.frag holds still pixels' gain in the viewer after the composite (state ping-pong, MRT), driven by PlaybackHold::pixel_step; GpuCompositor::hold_sequence and rudra-gpu-parity hold it to ShadowSmoother::hold_pixels";
       paths = @("native/render/shaders/hold.frag", "native/render/probe/hold_sequence.hpp",
                 "native/render/probe/gpu_parity.cpp", "native/render/CMakeLists.txt",
                 "native/render/src/viewer_window.cpp", "native/render/src/gpu_composite.cpp", "native/render/src/passes.hpp",
                 "native/render/include/rudra/render/viewer_window.hpp", "native/render/include/rudra/render/gpu_composite.hpp",
                 "native/engine/src/playback_hold.cpp", "native/engine/include/rudra/engine/playback_hold.hpp",
                 "native/app/main_window.cpp", "native/app/main_window.hpp", "native/tests/test_engine.cpp") },
    @{ msg = "Notes: still areas play steady too (release notes for the next beta, user guide, architecture, status)";
       paths = @("docs/BETA.md", "docs/USER_GUIDE.md", "docs/NATIVE_ARCHITECTURE.md", "STATUS.md",
                 "scripts/release_gpu_hold_2026-10-11.ps1") }
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
Write-Host "`nnative pushed. Watch the native checks: https://github.com/FXTD-Studios/RUDRA/actions" -ForegroundColor Green
