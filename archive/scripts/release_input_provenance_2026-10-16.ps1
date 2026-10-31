<#
.SYNOPSIS
  Commit the 1.0 audit fixes (stills' colour and orientation, master
  provenance, NOTICE attribution, the super-white note and counsel questions)
  on `native`; push only with -Push.

    .\scripts\release_input_provenance_2026-10-16.ps1
    .\scripts\release_input_provenance_2026-10-16.ps1 -Push
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
    @{ msg = "Stills keep their colour and orientation (audit P1-1, roadmap 1.1): media/still_metadata reads the ICC profile (PNG iCCP with a bounded inflate, JPEG APP2 in parts, TIFF 34675, WebP ICCP) and PNG cICP / sRGB / cHRM, matches the colorants to sRGB / Rec.709, Display P3, Rec.2020 or Adobe RGB (1998) (new Primaries::AdobeRgb, appended), and applies the EXIF orientation (JPEG APP1, PNG eXIf, TIFF, WebP). The master converts from the still's primaries unless the request names one; untagged files are unchanged; an unknown profile is reported and read as Rec.709. 8 tests";
       paths = @("native/media/include/rudra/media/still_metadata.hpp", "native/media/src/still_metadata.cpp",
                 "native/media/include/rudra/media/still.hpp", "native/media/src/still.cpp", "native/media/CMakeLists.txt",
                 "native/core/include/rudra/core/color.hpp", "native/core/src/gamut.cpp", "native/deliver/src/exr.cpp",
                 "native/tests/test_still_metadata.cpp", "native/tests/test_decode.cpp") },
    @{ msg = "Master provenance (audit P2-4 / P3-1, roadmap 1.3): every master's EXR and sidecar record the app version and the configure-time build commit, the source file's SHA-256, the checkpoint's SHA-256 when a model ran, and whether the network tiled (rudra:tiled was hard-coded False); the CLI and the app's master job fill them, and the still's own primaries reach the master. master-check leaves build provenance out when the Studio golden has none. 5 tests";
       paths = @("native/CMakeLists.txt", "native/deliver/CMakeLists.txt", "native/deliver/include/rudra/deliver/master.hpp",
                 "native/deliver/src/master.cpp", "native/engine/include/rudra/engine/master_job.hpp",
                 "native/engine/src/master_job.cpp", "native/cli/master.cpp", "native/cli/main.cpp",
                 "native/app/main_window.cpp", "native/tests/test_master_provenance.cpp", "native/tests/CMakeLists.txt") },
    @{ msg = "Notes: NOTICE credits the Netflix Open Content titles behind v8 and the BT.1886 shadows curve (Nocturne, Sparks, Meridian, Sol Levante, Cosmos Laundromat / Blender Foundation); the super-white design (docs/SUPER_WHITES_DESIGN.md) and its 1.0 limit; the questions for counsel (docs/LEGAL_QUESTIONS.md); USER_GUIDE and BETA";
       paths = @("NOTICE", "docs/SUPER_WHITES_DESIGN.md", "docs/LEGAL_QUESTIONS.md", "docs/USER_GUIDE.md", "docs/BETA.md",
                 "scripts/release_input_provenance_2026-10-16.ps1") }
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
