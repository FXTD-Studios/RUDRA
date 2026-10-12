<#
.SYNOPSIS
  Commit the v1 preparation and push `native`.

.DESCRIPTION
  R3 tooling (training/compare_market.py) and its first table, rudra batch
  running the app's queues (grade), clip-mode CVVDP in the temporal tools,
  the release workflow publishing v1.0.0 as a release, the release-candidate
  script, BT.1886 as the default source and the master exposure (native), and the notes (docs/RELEASE_V1.md, results, status). The version is
  not changed here: scripts/release_v1_rc.ps1 does that for each candidate.

    .\scripts\release_v1_prep_2026-10-12.ps1
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
    @{ msg = "rudra batch runs the app's queues: a job with grade (the app's export settings) goes to rudra-native video (PATH or RUDRA_NATIVE), the package or '-' for an analytic export; 4 tests";
       paths = @("rudra/batch.py", "tests/test_batch_grade_2026_10_12.py", "README.md") },
    @{ msg = "R3 tooling: training/compare_market.py puts the frozen comparison set through RUDRA, BT.2408, HDRTVDM and any tool run by hand (import), and scores them as R1 was scored (PU21, CVVDP, clipped stops, paired bootstrap against the analytic inverse)";
       paths = @("training/compare_market.py") },
    @{ msg = "Temporal: make_temporal_clips --reference writes each clip's HDR; measure_temporal --reference scores every master with CVVDP as a clip and frame by frame (STATUS row 5)";
       paths = @("tools/make_temporal_clips.py", "tools/measure_temporal.py") },
    @{ msg = "Release: a tag with a suffix is a pre-release and v1.0.0 a release; scripts/release_v1_rc.ps1 versions, pushes and tags each 1.0 candidate and 1.0";
       paths = @(".github/workflows/release.yml", "scripts/release_v1_rc.ps1") },
    @{ msg = "Source: BT.1886, the default for new sessions (release row R3: the SDR read as a display shows it, each BT.709 code to the power 2.4, white at 203 nits; projects and requests that name no curve stay unknown, the ACES inverse); Exposure: a master exposure in stops (-3 to +3), part of the grade (after the anchor, before the Region EV bands), live in the viewer and in masters, movie exports and projects; 0 is bit for bit. Tests, parity cases; the analytic-default app test turns the clip fill off";
       paths = @("native/core/include/rudra/core/source_curve.hpp", "native/core/src/source_curve.cpp", "native/engine/include/rudra/engine/session.hpp",
                 "native/engine/src/actions.cpp", "native/app/main_window.cpp", "native/app/tests/test_app.cpp",
                 "native/tests/test_source_curve.cpp", "native/tests/test_session.cpp", "native/deliver/src/master.cpp",
                 "native/render/probe/gpu_parity.cpp", "docs/USER_GUIDE.md", "docs/composite.spec.md",
                 "native/core/include/rudra/core/composite.hpp", "native/core/src/composite.cpp", "native/core/include/rudra/core/master.hpp",
                 "native/core/src/master.cpp", "native/deliver/include/rudra/deliver/master.hpp", "native/engine/src/session.cpp",
                 "native/engine/src/project.cpp", "native/app/layout.cpp", "native/app/main_window.hpp",
                 "native/render/shaders/composite.frag", "native/render/src/passes.hpp", "native/tests/test_exposure.cpp",
                 "native/tests/CMakeLists.txt") },
    @{ msg = "Notes: what is left for 1.0 and how to ship it (docs/RELEASE_V1.md), the R3 tables (the frozen set and HDRTV1K; BT.1886 the default because of them), v8 and CVVDP on the real clips, HDR10+ in the shipped encoders; status rows R2, R3, R5 and 5";
       paths = @("docs/RELEASE_V1.md", "docs/RESULTS.md", "docs/TEMPORAL_RESULTS.md", "docs/BETA.md", "STATUS.md",
                 "scripts/release_v1_prep_2026-10-12.ps1") }
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
Write-Host "Then docs/RELEASE_V1.md, from step 2."
