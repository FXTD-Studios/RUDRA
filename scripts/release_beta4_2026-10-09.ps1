<#
.SYNOPSIS
  Commit the 9 Oct 2026 work in five commits, push `native` (PR #29), and,
  with -Tag once PR #29's checks are green, tag v0.9.0-beta.4 so the release
  workflow builds the Windows installer/ZIP and the macOS DMG and publishes
  the pre-release.

.DESCRIPTION
  Run from PowerShell on Windows (AGENTS.md: never commit from a mounted
  shell). Each commit names its own paths (git commit -- <paths>), so the
  order of staging does not matter and nothing else is swept in.

    .\scripts\release_beta4_2026-10-09.ps1          # commit and push native
    .\scripts\release_beta4_2026-10-09.ps1 -Tag     # then, CI green: tag and push the tag
#>
[CmdletBinding()]
param([switch]$Tag, [string]$Version = "v0.9.0-beta.4")

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
    @{ msg = "R1 in: fp16 tile graph (model.tile.fp16.onnx, 1e-2 tolerance), --precision fp16 on diff and bench, a TensorRT device; Phase R and 5.x on the roadmap";
       paths = @("README.md", "docs/NATIVE_ARCHITECTURE.md", "native/README.md", "native/cli/batch.cpp", "native/cli/main.cpp",
                 "native/cli/video.cpp", "native/core/include/rudra/core/model_manifest.hpp", "native/core/src/model_manifest.cpp",
                 "native/engine/src/model_catalog.cpp", "native/infer/include/rudra/infer/backend.hpp",
                 "native/infer/src/libtorch_backend.cpp", "native/infer/src/onnxruntime_backend.cpp", "native/infer/src/self_test.cpp",
                 "native/infer/src/tiler.cpp", "native/tests/test_core.cpp", "native/tests/test_model_catalog.cpp",
                 "scripts/NATIVE_GATE_A.ps1", "scripts/native_gate_a.sh", "tests/test_export_model_2026_09_23.py", "tools/export_model.py") },
    @{ msg = "Ship-path review fixes: HLG gamma log10 in deliver, Region EV after the anchor in Studio masters, linear masters really Rec.2020, bounded tag reads, encode cleanup, Studio Host check, hard-link fallback, batch save retry, atomic EXR; tiler moved to rudra/inference.py";
       paths = @("rudra/batch.py", "rudra/delivery/cli.py", "rudra/delivery/exr.py", "rudra/delivery/profiles.py",
                 "rudra/delivery/video.py", "rudra/inference.py", "rudra/video.py", "tests/test_review_fixes_2026_10_09.py",
                 "tests/test_inference_module_2026_10_09.py", "training/infer_sdr2hdr.py", "training/quality_benchmark.py",
                 "ui/sequence.py", "ui/server.py") },
    @{ msg = "Hub: sdr2hdr_image_v8 in the export and the model card (optional model, its real-SDR numbers); Netflix Open Content attribution for v8's training titles";
       paths = @("checkpoints/LICENSE", "docs/HUB_MODEL_CARD.md", "training/export_for_hub.py") },
    @{ msg = "App: a bare start follows the default when it moves (an older beta's package gives way to v8); sequence-encode tests accept ffmpeg 6.1's complete colr atom (macOS CI); phase 3 exit runs the app tests in their own console";
       paths = @("native/app/main_window.cpp", "native/app/tests/test_app.cpp", "native/tests/test_sequence_encode.cpp",
                 "scripts/NATIVE_PHASE3_EXIT.ps1") },
    @{ msg = "Beta 4: version beta.4, the release workflow exports v8 (it still shipped shadow_v1), release notes since the beta 3 tag";
       paths = @("native/CMakeLists.txt", ".github/workflows/release.yml", "docs/BETA.md", "STATUS.md", "scripts/release_beta4_2026-10-09.ps1") }
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
Write-Host "`nnative pushed. When PR #29's checks are green:" -ForegroundColor Green
Write-Host "  .\scripts\release_beta4_2026-10-09.ps1 -Tag"
