<#
.SYNOPSIS
  Commit the 10 Oct 2026 work (product items 2 to 5: bundled ffmpeg,
  projects and autosave, the update check and signing hooks, help) in five
  commits, push `native`, and, with -Tag once the checks are green, tag
  v0.9.0-beta.6 so the release workflow builds the Windows installer/ZIP and
  the macOS DMG and publishes the pre-release.

.DESCRIPTION
  Run from PowerShell on Windows (AGENTS.md: never commit from a mounted
  shell). Each commit names its own paths (git commit -- <paths>), so the
  order of staging does not matter and nothing else is swept in.

    .\scripts\release_beta6_2026-10-10.ps1          # commit and push native
    .\scripts\release_beta6_2026-10-10.ps1 -Tag     # then, CI green: tag and push the tag
#>
[CmdletBinding()]
param([switch]$Tag, [string]$Version = "v0.9.0-beta.6")

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
    @{ msg = "ffmpeg beside the app: platform/tools puts a package's own ffmpeg first on PATH (RUDRA_FFMPEG_DIR picks another, =path the PATH's); every Windows executable runs in the UTF-8 code page (cmake/utf8.manifest), so paths with any letters reach ffmpeg and OpenCV";
       paths = @("native/platform/include/rudra/platform/tools.hpp", "native/platform/src/tools.cpp", "native/platform/CMakeLists.txt",
                 "native/cli/main.cpp", "native/tests/test_platform.cpp", "native/cmake/RudraWarnings.cmake", "native/cmake/utf8.manifest") },
    @{ msg = "Projects: engine/project (.rudra JSON, versioned, atomic; masks beside as <name>.rudra.masks.png; paths relative to the project when under it; hand-edited values validated and clamped) and Session::load_state";
       paths = @("native/engine/include/rudra/engine/project.hpp", "native/engine/src/project.cpp", "native/engine/CMakeLists.txt",
                 "native/engine/include/rudra/engine/session.hpp", "native/engine/src/session.cpp",
                 "native/tests/test_project.cpp", "native/tests/CMakeLists.txt") },
    @{ msg = "App: File > Open/Save project, autosave and Reopen last session, the project's model queued behind a load, save prompt; Help > Getting started, User guide (F1), Check for updates, Open log folder; tooltips; Quit through close(); .rudra association in the installer";
       paths = @("native/app/main.cpp", "native/app/main_window.cpp", "native/app/main_window.hpp", "native/app/updates.cpp",
                 "native/app/updates.hpp", "native/app/model_dialogs.cpp", "native/app/startup.cpp", "native/app/CMakeLists.txt",
                 "native/app/tests/test_app.cpp", "native/app/windows/rudra.iss", "native/engine/src/actions.cpp") },
    @{ msg = "Packaging: the Windows package carries the gyan.dev full ffmpeg 7.1.1 (SHA-256 pinned, GPL licence and source note beside it); MAC_FFMPEG_DIR bundles a static one in the DMG; signtool signing when RUDRA_SIGN_PFX is set, WINDOWS_CERT_PFX in the release workflow";
       paths = @("scripts/PACKAGE_WINDOWS.ps1", "scripts/package_mac.sh", ".github/workflows/release.yml", "NOTICE") },
    @{ msg = "Beta 6: version beta.6, docs/USER_GUIDE.md, release notes since beta 5, repository links to FXTD-Studios/RUDRA";
       paths = @("native/CMakeLists.txt", "docs/BETA.md", "docs/USER_GUIDE.md", "README.md", "STATUS.md",
                 "scripts/release_beta6_2026-10-10.ps1") }
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
Write-Host "`nnative pushed. When the native checks are green:" -ForegroundColor Green
Write-Host "  .\scripts\release_beta6_2026-10-10.ps1 -Tag"
