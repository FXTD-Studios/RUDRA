<#
.SYNOPSIS
  Commit the folder-of-frames movie export (HDR10, HLG, ProRes from the
  export sheet for a folder shot) on `native`; push only with -Push.

    .\scripts\release_frames_movie_2026-10-16.ps1
    .\scripts\release_frames_movie_2026-10-16.ps1 -Push
#>
[CmdletBinding()]
param([switch]$Push)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
function Invoke-Git { & git @args; if ($LASTEXITCODE -ne 0) { Fail "git $args" } }

if (Test-Path .git\index.lock) { Fail ".git\index.lock exists: move it to .git\_stale_locks\ first (AGENTS.md)" }
if ((git rev-parse --abbrev-ref HEAD) -ne "native") { Fail "not on the native branch" }

$commits = @(
    @{ msg = "Export: a folder of frames goes to HDR10, HLG, ProRes 422 HQ or 4444 from the export sheet (they were offered for movies only). The frames are mastered as a linear Rec.2020 EXR sequence into <name>_frames beside the movie, then encoded as rudra-native deliver encodes them (measured MaxCLL/MaxFALL, tags verified, PQ only, no audio, a frame-rate pick on the sheet); the Master EXR button stops it. deliver_linear_frames is shared by the CLI and the app. 2 tests";
       paths = @("native/deliver/include/rudra/deliver/sequence_encode.hpp", "native/deliver/src/sequence_encode.cpp",
                 "native/cli/deliver.cpp", "native/app/main_window.hpp", "native/app/main_window.cpp",
                 "native/app/export_sheet.hpp", "native/app/export_sheet.cpp", "native/tests/test_sequence_encode.cpp",
                 "native/app/tests/test_app.cpp", "scripts/release_frames_movie_2026-10-16.ps1") }
)

$coauthor = "Co-authored-by: FXTD Studios Research <research@fxtdstudios.com>"
foreach ($c in $commits) {
    $missing = @($c.paths | Where-Object { -not (Test-Path $_) })
    if ($missing.Count) { Fail "missing: $($missing -join ', ')" }
    $changed = @(git status --porcelain -- $c.paths)
    if ($changed.Count -eq 0) { Write-Host "nothing to commit for: $($c.msg.Substring(0, 50))..."; continue }
    Invoke-Git add -- $c.paths
    Invoke-Git commit -m $c.msg -m $coauthor -- $c.paths
}
$left = @(git status --porcelain --untracked-files=no)
if ($left.Count) { Write-Host "Still modified (left for you):" -ForegroundColor Yellow; $left | Write-Host }
# Every commit to be pushed carries the FXTD Studios Research credit and no
# other co-author (docs/RELEASE_V1.md); a git failure stops the push.
$shas = @(git rev-list origin/native..HEAD)
if ($LASTEXITCODE -ne 0) { Fail "git rev-list origin/native..HEAD (fetch origin first)" }
foreach ($c in $shas) {
    $body = (git log -1 --format=%B $c) -join "`n"
    if ($LASTEXITCODE -ne 0) { Fail "git log $c" }
    if ($body -match '(?i)anthropic\.com|co-authored-by:\s*claude|claude-session') { Fail "$c credits an AI assistant; amend it first (AGENTS.md)" }
    if (-not $body.Contains($coauthor)) { Fail "$c lacks '$coauthor'" }
    $others = @([regex]::Matches($body, '(?im)^co-authored-by:.*$') | ForEach-Object { $_.Value.Trim() } | Where-Object { $_ -ne $coauthor })
    if ($others.Count) { Fail "$c has another co-author: $($others -join '; ')" }
}
if ($Push) { Invoke-Git push origin native; Write-Host "native pushed: https://github.com/FXTD-Studios/RUDRA/actions" -ForegroundColor Green }
else { Write-Host "Committed locally, not pushed. Review: git log --stat origin/native..HEAD" -ForegroundColor Green }
