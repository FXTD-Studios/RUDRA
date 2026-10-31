<#
  Puts v10's training\train_sdr2hdr.py in place (it is read-only on this disk, so
  the release script could not write it; the new version waits in tmp\), checks
  it, commits it and pushes `native`. Then re-run the smoke stage.

    .\scripts\fix_v10_train_2026-10-13.ps1
#>
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
function Invoke-Git { & git @args; if ($LASTEXITCODE -ne 0) { Fail "git $args" } }
if (Test-Path .git\index.lock) { Fail ".git\index.lock exists: move it to .git\_stale_locks\ first (AGENTS.md)" }
if ((git rev-parse --abbrev-ref HEAD) -ne "native") { Fail "not on the native branch" }

$target = "training\train_sdr2hdr.py"
$staged = "tmp\v10_train_sdr2hdr.py"
$want = "0f6b42eb8a28f74af6bcfb985466348e"
$have = (Get-FileHash $target -Algorithm MD5).Hash.ToLower()
if ($have -ne $want) {
    if (-not (Test-Path $staged)) { Fail "$staged is missing and $target is not the v10 version" }
    attrib -R $target
    try { [System.IO.File]::WriteAllBytes((Resolve-Path $target).Path, [System.IO.File]::ReadAllBytes((Resolve-Path $staged).Path)) }
    catch {
        Write-Host "Windows refuses to write $target. Check: attrib $target  (R = read-only)," -ForegroundColor Yellow
        Write-Host "and that no python process has it open: Get-Process python* | Select Id, StartTime, Path" -ForegroundColor Yellow
        Fail $_.Exception.Message
    }
    $have = (Get-FileHash $target -Algorithm MD5).Hash.ToLower()
    if ($have -ne $want) { Fail "$target did not take the new version ($have)" }
    Write-Host "$target updated" -ForegroundColor Green
}
python training\train_sdr2hdr.py --help | Select-String -SimpleMatch "bench_gain" | Out-Null
if (-not $?) { Fail "the new $target does not offer --best-metric bench_gain" }
if (Test-Path $staged) { Remove-Item $staged }

$coauthor = "Co-authored-by: FXTD Studios Research <research@fxtdstudios.com>"
$paths = @($target, "scripts/fix_v10_train_2026-10-13.ps1")
if (@(git status --porcelain -- $paths).Count) {
    Invoke-Git add -- $paths
    Invoke-Git commit -m "v10: train_sdr2hdr.py with --baseline-source, --baseline-dropout, --crushed-hold, the bench validation (--bench-val-frames / -max-side / -push) and --best-metric bench_gain (left out of the v10 commit: the file was read-only on the disk)" -m $coauthor -- $paths
}
$credits = (git log origin/native..HEAD --format=%B) -join "`n"
if ($credits -match '(?i)anthropic\.com|co-authored-by:\s*claude|claude-session') { Fail "a commit to push credits an AI assistant; amend it first (AGENTS.md)" }
Invoke-Git push origin native
Write-Host "Done. Next: powershell -ExecutionPolicy Bypass -File scripts\run_v10_2026_10_13.ps1 -Stage smoke" -ForegroundColor Green
