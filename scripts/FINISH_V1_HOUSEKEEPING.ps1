<#
.SYNOPSIS
  The v1 housekeeping that has to run on Windows (3 Nov 2026): the items of the
  "RUDRA v1 completion checklist" doc that an agent shell cannot do.

    powershell -ExecutionPolicy Bypass -File scripts\FINISH_V1_HOUSEKEEPING.ps1 -DryRun
    powershell -ExecutionPolicy Bypass -File scripts\FINISH_V1_HOUSEKEEPING.ps1
    powershell -ExecutionPolicy Bypass -File scripts\FINISH_V1_HOUSEKEEPING.ps1 -SkipBuild -PruneBranches

  In order:
    1. git fetch --tags; say whether v1.0.3 is tagged on origin
    2. commit the provenance record, tools/inspect_highlights.py, the .gitignore
       entry for result.json and the corrected LEGAL_QUESTIONS.md
    3. refuse to push if any unpushed commit mentions Claude or anthropic.com
       (AGENTS.md), then push feat/gap-kits and print the PR link
    4. delete the transfer leftovers in tmp\ (never tmp\native_deps)
    5. move the stale remote ref lock into .git\_stale_locks
    6. pytest tests/test_genfill_run.py tests/test_make_models_json.py in .venv
    7. scripts\RUN_APP_ACES.ps1: rebuild RUDRA and rudra-native, app tests, start
    8. with -PruneBranches: delete local branches already merged into origin/main
  Nothing is signed, tagged or released here.
#>
param(
    [switch]$DryRun,
    [switch]$SkipBuild,
    [switch]$SkipPush,
    [switch]$PruneBranches,
    [string]$Branch = "feat/gap-kits",
    [string]$Python = ".venv\Scripts\python.exe"
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $Repo
function Say([string]$m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail([string]$m) { Write-Host "`nFAILED: $m" -ForegroundColor Red; exit 1 }
function Run([string]$what, [scriptblock]$cmd) {
    if ($DryRun) { Write-Host "  (dry run) $what" -ForegroundColor DarkGray; return }
    & $cmd
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { Fail $what }
}
if (Test-Path ".git\index.lock") { Fail ".git\index.lock exists: move it to .git\_stale_locks (AGENTS.md), then run this again" }

# 1 ---------------------------------------------------------------------------
# Only v1.0.3 is fetched: a plain "git fetch --tags" stops on tags that differ
# locally (v1.0.0-rc.3 was re-made on origin), and those are reported, never clobbered.
Say "Is v1.0.3 on origin? Do any local tags differ from origin's?"
$remote = @{}
git ls-remote --tags origin | ForEach-Object {
    $sha, $ref = $_ -split "\s+"
    if ($ref -notmatch '\^\{\}$') { $remote[$ref -replace '^refs/tags/', ''] = $sha }
}
if ($LASTEXITCODE -ne 0) { Fail "git ls-remote origin" }
if ($remote.ContainsKey("v1.0.3")) {
    Write-Host "  v1.0.3 is tagged on origin: $($remote['v1.0.3'].Substring(0,7))" -ForegroundColor Green
    if (-not (git tag -l "v1.0.3")) { Run "git fetch v1.0.3" { git fetch --no-tags origin "refs/tags/v1.0.3:refs/tags/v1.0.3" } }
} else { Write-Host "  v1.0.3 is NOT tagged on origin: the README links a release that may not exist" -ForegroundColor Yellow }
foreach ($tag in (git tag -l)) {
    if ($remote.ContainsKey($tag)) {
        $local = (git rev-parse $tag).Trim()
        if ($local -ne $remote[$tag]) {
            Write-Host "  $tag differs: local $($local.Substring(0,7)), origin $($remote[$tag].Substring(0,7)) (kept as is; see below)" -ForegroundColor Yellow
        }
    }
}

# 2 ---------------------------------------------------------------------------
$cur = (git rev-parse --abbrev-ref HEAD).Trim()
if ($cur -ne $Branch) { Fail "on '$cur', expected '$Branch' (pass -Branch)" }
$files = @("docs/models/RUDRA_ACES_2.provenance.json", "tools/inspect_highlights.py", ".gitignore", "docs/LEGAL_QUESTIONS.md", "scripts/FINISH_V1_HOUSEKEEPING.ps1", "scripts/RELEASE_1_0_3.ps1")
$changed = git status --porcelain -- $files
if ($changed) {
    Say "Commit: provenance record, inspect_highlights, gitignore, legal question 10"
    $changed | ForEach-Object { Write-Host "  $_" }
    $msg = @"
v1 housekeeping: RUDRA ACES 2 provenance record, inspect_highlights, legal question 10

- docs/models/RUDRA_ACES_2.provenance.json: tools/model_provenance.py on the
  training reports (1 Nov); reports name checkpoint f73f14c7. The model stays
  labelled preview: it still fails the 0 EV grading check (0.121 vs 0.0604).
- tools/inspect_highlights.py: where a master's extreme highlights and its
  failing pixels come from.
- .gitignore: /result.json.
- docs/LEGAL_QUESTIONS.md: question 10 states the Apache 2.0 history; the questions no longer gate a release (free and open source).
- scripts/FINISH_V1_HOUSEKEEPING.ps1, scripts/RELEASE_1_0_3.ps1.

Co-authored-by: FXTD Studios Research <research@fxtdstudios.com>
"@
    Run "git add" { git add -- $files }
    if (-not $DryRun) {
        $tmpMsg = [IO.Path]::GetTempFileName()
        [IO.File]::WriteAllText($tmpMsg, $msg, (New-Object Text.UTF8Encoding $false))
        git commit -F $tmpMsg
        $rc = $LASTEXITCODE; Remove-Item $tmpMsg
        if ($rc -ne 0) { Fail "git commit" }
    }
} else { Say "Nothing to commit for the housekeeping files" }

# 3 ---------------------------------------------------------------------------
if (-not $SkipPush) {
    git merge-base --is-ancestor 3858348 origin/main
    if ($LASTEXITCODE -ne 0) { Fail "ship 1.0.3 first (scripts\RELEASE_1_0_3.ps1): origin/main lacks it, and this push would bundle it with $Branch. Or pass -SkipPush" }
    Say "Check unpushed commits for AI attribution, then push $Branch"
    $bodies = git log "origin/main..HEAD" --format=%B
    if ($bodies | Select-String -Pattern "Claude|anthropic\.com" -Quiet) {
        git log "origin/main..HEAD" --format="%h %s" | Write-Host
        Fail "a commit above mentions Claude or anthropic.com (AGENTS.md): reword it before pushing"
    }
    Write-Host "  $((git rev-list --count origin/main..HEAD).Trim()) commits ahead of origin/main, none credit an AI assistant" -ForegroundColor Green
    Run "git push" { git push -u origin $Branch }
    Write-Host "  Open the PR: https://github.com/FXTD-Studios/RUDRA/compare/main...$Branch" -ForegroundColor Green
}

# 4 ---------------------------------------------------------------------------
Say "Delete the transfer leftovers in tmp\ (tmp\native_deps is kept)"
$leftovers = @("agent_transfer_rudra_py_20261021.tgz", "claude_code_snapshot.tgz", "claude_xfer",
               "phase3_masters_2026-10-21_0715", "phase3_masters_2026-10-21_0803",
               "_cc_check.tgz", "_cc_native.tgz", "_cc_golden.tgz", "_cc_app.tgz", "_cc_ui.tgz", "_cc_py.tgz")
# tmp\torch_header_probe and tmp\native_export_python are left alone: nothing references
# them, but they may hold build probes; delete them by hand if they are not needed.
foreach ($l in $leftovers) {
    $p = Join-Path "tmp" $l
    if (Test-Path -LiteralPath $p) { Write-Host "  $p"; Run "delete $p" { Remove-Item -LiteralPath $p -Recurse -Force } }
}

# 5 ---------------------------------------------------------------------------
$lock = ".git\refs\remotes\origin\backup-before-rewrite.lock"
if (Test-Path -LiteralPath $lock) {
    Say "Move the stale ref lock $lock into .git\_stale_locks"
    Run "move lock" { New-Item -ItemType Directory -Force ".git\_stale_locks" | Out-Null; Move-Item -LiteralPath $lock ".git\_stale_locks\origin-backup-before-rewrite.lock" -Force }
}

# 6 ---------------------------------------------------------------------------
Say "Python tests (step 5 of the 29 Oct plan)"
if (-not (Test-Path $Python)) { Fail "no $Python" }
Run "pytest" { & $Python -m pytest tests/test_genfill_run.py tests/test_make_models_json.py -ra }

# 7 ---------------------------------------------------------------------------
if (-not $SkipBuild) {
    Say "Rebuild, test and start the app: scripts\RUN_APP_ACES.ps1"
    Run "RUN_APP_ACES.ps1" { & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "RUN_APP_ACES.ps1") }
}

# 8 ---------------------------------------------------------------------------
$merged = git branch --merged origin/main --format="%(refname:short)" | Where-Object { $_ -notin @("main", $Branch) }
if ($merged) {
    Say "Local branches already merged into origin/main: $($merged -join ', ')"
    if ($PruneBranches) { foreach ($b in $merged) { Run "git branch -d $b" { git branch -d $b } } }
    else { Write-Host "  (pass -PruneBranches to delete them; unmerged backup/* and codex/* are never touched)" }
}
Say "Done. Next by hand: the HDR pass on the rebuilt app, then NATIVE_GATE_B.ps1 -Machine rtx4080s-win"
