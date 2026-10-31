<#
.SYNOPSIS
  Commit Dolby Vision and HDR10+ metadata for movie exports and push `native`.

.DESCRIPTION
  PQ exports (HDR10, ProRes PQ) write their dynamic HDR metadata per shot
  beside the master (<file>_hdr10plus.json, _dovi_generate.json,
  _hdr_analysis.json, _hdr10plus_scenes.json); HDR10 embeds HDR10+ when the
  ffmpeg's x265 can. The dovi_generate config now uses dovi_tool's block
  format. Python and native in parity; goldens re-emitted. Docs (release
  notes for the next beta, user guide, status) in a second commit;
  docs/NEXT_TASKS.md is local (.gitignore NEXT_*.md).

    .\scripts\release_dynamic_metadata_2026-10-11.ps1
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
    @{ msg = "Dolby Vision and HDR10+ metadata in movie exports: per shot of the export's cuts beside PQ masters (x265/hdr10plus_tool JSON, dovi_tool generate config, analysis, scenes); HDR10 embeds HDR10+ when x265 can, checked by QC; --dynamic-metadata on|off; dovi_generate blocks in dovi_tool's format";
       paths = @("rudra/video.py", "rudra/delivery/metadata.py", "tests/test_dynamic_metadata_2026_10_11.py",
                 "tests/test_delivery_2026_08_22.py", "tools/emit_video_master_golden.py", "tools/emit_video_qc_golden.py",
                 "native/core/src/metadata.cpp", "native/core/include/rudra/core/metadata.hpp",
                 "native/deliver/src/sidecars.cpp", "native/deliver/include/rudra/deliver/sidecars.hpp",
                 "native/deliver/src/video_master.cpp", "native/deliver/include/rudra/deliver/video_master.hpp",
                 "native/deliver/src/video_encode.cpp", "native/deliver/include/rudra/deliver/video_encode.hpp",
                 "native/deliver/src/queue.cpp", "native/video/src/convert.cpp", "native/video/include/rudra/video/convert.hpp",
                 "native/video/src/queue_runner.cpp", "native/cli/video.cpp",
                 "native/tests/test_video_convert_real.cpp", "native/tests/test_video_graded.cpp",
                 "native/tests/golden/delivery/seq_dovi_generate.json",
                 "native/tests/golden/video_convert", "native/tests/golden/video_queue") },
    @{ msg = "Notes: Dolby Vision and HDR10+ metadata (release notes for the next beta, user guide, status)";
       paths = @("docs/BETA.md", "docs/USER_GUIDE.md", "STATUS.md",
                 "scripts/release_dynamic_metadata_2026-10-11.ps1") }
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
