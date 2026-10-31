<#
  CI fix (tests workflow, flake8 F821): the exposure push's ceiling scaling was
  pasted into SDRHDRVideoDataset too, where `pushed` and `ceiling_scale` do not
  exist. Removes those two lines, checks, commits and pushes `native`.

    .\scripts\fix_ci_2026-10-13.ps1
#>
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
function Invoke-Git { & git @args; if ($LASTEXITCODE -ne 0) { Fail "git $args" } }
if ((git rev-parse --abbrev-ref HEAD) -ne "native") { Fail "not on the native branch" }

$path = Join-Path (Get-Location) "training\sdr2hdr_dataset.py"
$lines = [System.Collections.Generic.List[string]]([System.IO.File]::ReadAllText($path) -split "`n")
# The video dataset's copy: the second "if ceiling and pushed:" in the file.
$hits = @(for ($i = 0; $i -lt $lines.Count; $i++) { if ($lines[$i] -match '^\s+if ceiling and pushed:') { $i } })
if ($hits.Count -eq 2) {
    $at = $hits[1]
    if ($lines[$at + 1] -notmatch 'ceiling_scale') { Fail "unexpected line after $($at + 1)" }
    $lines.RemoveRange($at, 2)
    [System.IO.File]::WriteAllText($path, ($lines -join "`n"), (New-Object System.Text.UTF8Encoding $false))
    Write-Host "removed the two lines at $($at + 1)"
} elseif ($hits.Count -ne 1) { Fail "found $($hits.Count) copies; expected 2 (or 1 if already fixed)" }

python -c "import pyflakes" 2>$null
if ($LASTEXITCODE -eq 0) {
    python -m pyflakes training\sdr2hdr_dataset.py
    if ($LASTEXITCODE -ne 0) { Fail "pyflakes still reports problems" }
} else { Write-Host "pyflakes not installed: skipping the local check (CI runs flake8)" -ForegroundColor Yellow }

$coauthor = "Co-authored-by: FXTD Studios Research <research@fxtdstudios.com>"
$paths = @("training/sdr2hdr_dataset.py", "scripts/fix_ci_2026-10-13.ps1")
if (@(git status --porcelain -- $paths).Count) {
    Invoke-Git add -- $paths
    Invoke-Git commit -m "Training: the video dataset had the exposure push's ceiling scaling pasted in (pushed / ceiling_scale undefined there; flake8 F821 failed the tests workflow); removed" -m $coauthor -- $paths
}
$credits = (git log origin/native..HEAD --format=%B) -join "`n"
if ($credits -match '(?i)anthropic\.com|co-authored-by:\s*claude|claude-session') { Fail "a commit to push credits an AI assistant; amend it first (AGENTS.md)" }
Invoke-Git push origin native
Write-Host "pushed; the tests workflow re-runs on it" -ForegroundColor Green
