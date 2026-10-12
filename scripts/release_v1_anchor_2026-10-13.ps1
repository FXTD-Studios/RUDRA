<#
.SYNOPSIS
  Commit the anchor fix and the 1.0 decisions, and push `native`.

.DESCRIPTION
  The master's anchor follows the Source picker, its calibration and
  reference (native: core, deliver, engine, render, a parity case, 5 tests),
  the clip fill decision and the v9 result (notes), and the v9 bench options
  (-Checkpoint, -ClipOnly). Then the release candidate:
  scripts/release_v1_rc.ps1.

    .\scripts\release_v1_anchor_2026-10-13.ps1
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
    @{ msg = "Anchor: the master's anchor follows the Source picker, its calibration and its reference. Below the knee it targeted the sRGB reading at 203 nits whatever the picker said, so an anchored BT.1886 master was the sRGB one there and calibration was undone; now anchor_target_knots (the picked curve over the sRGB reading, per code) in the master, the view, the measure and composite.frag. Unknown and aces with neither are unchanged bit for bit. 5 tests, a parity view case";
       paths = @("native/core/include/rudra/core/composite.hpp", "native/core/src/composite.cpp",
                 "native/core/include/rudra/core/master.hpp", "native/core/src/master.cpp",
                 "native/core/include/rudra/core/view.hpp", "native/core/src/view.cpp",
                 "native/deliver/src/master.cpp", "native/engine/src/measure.cpp",
                 "native/render/src/passes.hpp", "native/render/src/gpu_composite.cpp",
                 "native/render/shaders/composite.frag", "native/render/probe/gpu_parity.cpp",
                 "native/tests/test_anchor_target.cpp", "native/tests/CMakeLists.txt",
                 "docs/composite.spec.md", "docs/USER_GUIDE.md") },
    @{ msg = "v9 bench: -Checkpoint benches a given step (best.pt stayed at step 0) and -ClipOnly runs the clip benches only";
       paths = @("scripts/run_v9_2026_10_12.ps1") },
    @{ msg = "Notes: the anchor fix, the clip fill ships on (glow 0.5, the held-out optimum; its +0 EV and fireworks losses are notes and roadmap 4.2), v9 at step 20,000 fails the clip gate (beats the fill at +0 EV, loses at +1 and +2); the roadmap keeps only what is open";
       paths = @("README.md", "STATUS.md", "docs/BETA.md", "docs/RELEASE_V1.md", "scripts/release_v1_anchor_2026-10-13.ps1") }
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
Write-Host "`nnative pushed. Watch the native checks (Gate B builds render/ and runs the parity on GL and Vulkan):" -ForegroundColor Green
Write-Host "  https://github.com/FXTD-Studios/RUDRA/actions"
Write-Host "Then docs/RELEASE_V1.md: R2 on the RTX machine, the clean PC, .\scripts\release_v1_rc.ps1"
