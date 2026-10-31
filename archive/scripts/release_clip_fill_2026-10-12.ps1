<#
.SYNOPSIS
  Commit the clip fill (clipped highlights in the analytic reconstruction) and
  the v9 preparation, then push `native`.

.DESCRIPTION
  core/clip_fill: a clipped channel rebuilt from the ones that survived, a
  smooth fill and a dome for fully clipped regions, the artist's peak in nits;
  wired as fields into every analytic path (view, master, movie export), the
  Reconstruct menu, projects and the request JSON; rudra-native
  bench-clip-fill. v9: --exposure-push in training, scripts\run_v9_2026_10_12.ps1,
  the gate's clip rows against the clip fill. Notes in a second commit.

  Before it pushes it builds and runs the native tests if a build folder is
  given (-Build), so a red CI is seen here first.

    .\scripts\release_clip_fill_2026-10-12.ps1
    .\scripts\release_clip_fill_2026-10-12.ps1 -Build build\native_phase4
#>
[CmdletBinding()]
param([string]$Build = "")

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
# Not named "Git": PowerShell names are case-insensitive, so a function Git
# would call itself instead of git.exe.
function Invoke-Git { & git @args; if ($LASTEXITCODE -ne 0) { Fail "git $args" } }

if (Test-Path .git\index.lock) { Fail ".git\index.lock exists: move it to .git\_stale_locks\ first (AGENTS.md)" }
if ((git rev-parse --abbrev-ref HEAD) -ne "native") { Fail "not on the native branch" }

if ($Build) {
    if (-not (Test-Path $Build)) { Fail "no build folder $Build" }
    & cmake --build $Build --config Release --parallel
    if ($LASTEXITCODE -ne 0) { Fail "native build" }
    # rudra_tests.exe needs OpenCV (and ONNX Runtime) DLLs, which the phase scripts copy
    # beside rudra-native.exe only; without them test discovery fails and ctest reports
    # "Errors while running CTest" before any test runs.
    $env:PATH = "$(Resolve-Path (Join-Path $Build 'cli\Release'));$env:PATH"
    & ctest --test-dir $Build -C Release -R "ClipFill|Session|Project|Actions|Composite" --output-on-failure
    if ($LASTEXITCODE -ne 0) { Fail "native tests" }
}
$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
& $py -m pytest -q tests\test_v9_clip_2026_10_12.py tests\test_v8_bench_2026_10_08.py
if ($LASTEXITCODE -ne 0) { Fail "python tests" }

$commits = @(
    @{ msg = "Clip fill: where the SDR clipped the analytic reconstruction fills instead of leaving the inverse's flat ceiling (core/clip_fill: a clipped channel from the surviving ones and the chroma around it; fully clipped regions a smooth log2-luminance fill from the edge plus a dome from the glow, capped; the artist's peak in nits per region). Fields into every analytic path (deliver analytic_fields_for: view, master, movie export), Reconstruct > Fill clipped highlights / Set the peak of a clipped region / Clear clipped peaks, undone with the grade, kept in projects and the request JSON, recorded in the EXR and sidecar; rudra-native bench-clip-fill. v9 preparation: --exposure-push (training re-renders at +0..+2 EV), scripts/run_v9_2026_10_12.ps1, cp8_verdicts --clip-against clipfill";
       paths = @("native/core/include/rudra/core/clip_fill.hpp", "native/core/src/clip_fill.cpp", "native/core/CMakeLists.txt",
                 "native/deliver/include/rudra/deliver/master.hpp", "native/deliver/src/master.cpp",
                 "native/engine/include/rudra/engine/session.hpp", "native/engine/src/session.cpp",
                 "native/engine/src/project.cpp", "native/engine/src/actions.cpp",
                 "native/video/include/rudra/video/predictor.hpp", "native/video/src/predictor.cpp",
                 "native/app/main_window.cpp", "native/app/main_window.hpp",
                 "native/cli/main.cpp", "native/cli/master.cpp", "native/cli/CMakeLists.txt",
                 "native/cli/bench_clip_fill.cpp", "native/cli/bench_clip_fill.hpp",
                 "native/tests/test_clip_fill.cpp", "native/tests/CMakeLists.txt",
                 "training/sdr2hdr_dataset.py", "training/train_sdr2hdr.py", "training/cp8_verdicts.py",
                 "tests/test_v9_clip_2026_10_12.py", "scripts/run_v9_2026_10_12.ps1") },
    @{ msg = "Notes: the clip fill, measured inside clipped pixels at +0/+1/+2 EV, and the v9 plan (status, user guide, architecture)";
       paths = @("STATUS.md", "docs/USER_GUIDE.md", "docs/NATIVE_ARCHITECTURE.md", "scripts/release_clip_fill_2026-10-12.ps1") }
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
