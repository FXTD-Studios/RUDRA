<#
.SYNOPSIS
  Commit the CI fix for the static hold's goldens and push `native`.

.DESCRIPTION
  Native CI on a0e5099 failed VideoMaster.SpoolFramesAreThePythons on Linux:
  CI re-emits every golden from the Python, and the video_master and video_qc
  emitters run convert_video with a stand-in predictor, so the static hold
  (on by default) changed the HDR they record. They now pass
  --static-hold off; re-emitted, both goldens are byte-identical to the
  committed ones. Test tooling only: the beta 10 build is unaffected.

    .\scripts\release_ci_fix_2026-10-11.ps1
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
    @{ msg = "CI: the video_master and video_qc emitters record the stand-in HDR without the static hold (--static-hold off), so their goldens stay the mastering's and the QC's";
       paths = @("tools/emit_video_master_golden.py", "tools/emit_video_qc_golden.py", "scripts/release_ci_fix_2026-10-11.ps1") }
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
