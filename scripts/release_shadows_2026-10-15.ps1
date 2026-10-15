<#
.SYNOPSIS
  Commit the Getting Started redesign and the opt-in "BT.1886 shadows" Source
  entry on `native`; optionally the 13-14 Oct research line; push only with -Push.

.DESCRIPTION
  Commit 1, Getting started: the native five-step guide (app/getting_started.*),
    the window wiring, its theme rules, the app test and the GTest include fix
    for rudra_app_tests, and the design preview in docs/design/.
  Commit 2, BT.1886 shadows (opt-in; BT.1886 stays the default): the measured
    real-SDR blacks below SDR code 40 in native core, the picker action, the
    master's message and tests; the Python twin (rudra/source_baseline.py),
    the native knots fixture and tests; the read-only tools that measured and
    scored it; the GTest include fix for rudra_tests; USER_GUIDE and BETA.
  Commit 3, only with -IncludeResearch: the 13-14 Oct internal research line
    (commercial candidate preflight, bounded scratch pilots, routing /
    preservation / region-focus / repeatability / push-baseline A/B runners,
    their verifiers, diagnostics and tests, STATUS and the handoff docs).
    These files were already uncommitted before 14 Oct; nothing in them ships
    in the app. Review before including.

  docs/ai/ and reports/ stay local (gitignored audits and evidence).
  Every commit credits FXTD Studios Research only. Nothing is pushed unless
  -Push is given, and a push is refused if any commit credits Claude.

    .\scripts\release_shadows_2026-10-15.ps1
    .\scripts\release_shadows_2026-10-15.ps1 -IncludeResearch -Push
#>
[CmdletBinding()]
param([switch]$IncludeResearch, [switch]$Push)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
function Invoke-Git { & git @args; if ($LASTEXITCODE -ne 0) { Fail "git $args" } }

if (Test-Path .git\index.lock) { Fail ".git\index.lock exists: move it to .git\_stale_locks\ first (AGENTS.md)" }
if ((git rev-parse --abbrev-ref HEAD) -ne "native") { Fail "not on the native branch" }

$commits = @(
    @{ msg = "Getting started: a five-step guide window (Source, Calibrate, Reconstruct, Check, Deliver) with a step list, Back / Next and the user guide, replacing the one-page sheet; Esc or the close button dismisses it, clicking the text no longer does; the Deliver step keeps the HDR10+ / Dolby Vision, autosave and F1 lines. The app tests prefer the linked GTest's headers over a Conda prefix's";
       paths = @("native/app/getting_started.cpp", "native/app/getting_started.hpp", "native/app/main_window.cpp",
                 "native/app/CMakeLists.txt", "native/app/tests/test_app.cpp", "native/app/theme/studio.qss.in",
                 "docs/design/getting-started-preview.html", "docs/design/getting-started-preview.png",
                 "docs/design/getting-started-native.png") },
    @{ msg = "Source: BT.1886 shadows (opt-in; BT.1886 stays the default). BT.1886 from SDR code 40 up and, below it, the median HDR the real Netflix Open Content trims hold per code (training titles only, equal weight, monotone, 0.02 stop per code at least): real masters keep blacks 2-7 stops above code^2.4. Scored through the app's knot path against BT.1886 on held-out frames: Sparks +0.040 dB (45/48) and +0.077 dB (191/200), Meridian +1.302 dB (287/287), PU21. Appended last (enum 7, Python index 7) so stored ids keep their meaning; Python matches native knots within 1e-6. tools/dark_tone_targets.py and tools/shadow_correction_eval.py (read-only); rudra_tests prefer the linked GTest's headers over a Conda prefix's";
       paths = @("native/core/include/rudra/core/source_curve.hpp", "native/core/src/source_curve.cpp",
                 "native/engine/src/actions.cpp", "native/deliver/src/master.cpp",
                 "native/tests/test_source_curve.cpp", "native/tests/CMakeLists.txt",
                 "rudra/source_baseline.py", "tests/fixtures/source_curve_knots_native.json",
                 "tests/test_v10_2026_10_13.py", "tests/test_shadow_correction.py",
                 "tools/dark_tone_targets.py", "tools/shadow_correction_eval.py",
                 "docs/USER_GUIDE.md", "docs/BETA.md", "scripts/release_shadows_2026-10-15.ps1") }
)

if ($IncludeResearch) {
    $commits += @{ msg = "Research (internal, not shipped): the commercial candidate preflight and audits, bounded scratch pilots, the paired A/B runner (routing, preservation, region focus, strict repeatability, push baseline) with replay guards, watchdogs and post-run verifiers, the frozen-checkpoint diagnostics, opt-in trainer options (pre-clamp hold, strict determinism, pushed PU21 reporting, --push-baseline), their tests, and the handoff notes. No default, model or app behaviour changes";
        paths = @("STATUS.md", "docs/MODEL_PLAN_V10.md", "docs/CURRENT_HANDOFF.md", "docs/PRODUCTION_PROGRESS.md",
                  "docs/RUDRA_PRODUCTION_ROADMAP.md",
                  "pipeline/licences.py", "pipeline/audit_candidate_duplicates.py", "pipeline/audit_candidate_quality.py",
                  "pipeline/commercial_preflight.py", "pipeline/prepare_commercial_candidate.py",
                  "rudra/sdr2hdr.py",
                  "training/bench_val.py", "training/cp10_verdicts.py", "training/paired_gate.py",
                  "training/sdr2hdr_dataset.py", "training/train_sdr2hdr.py", "training/bounded_pilot.py",
                  "training/preservation.py", "training/routing_ab.py", "training/routing_replay.py",
                  "tools/diagnose_pilot.py", "tools/diagnose_preservation.py", "tools/diagnose_preservation_tradeoff.py",
                  "tools/frame_luminance_breakdown.py", "tools/probe_preservation_batches.py", "tools/verify_preservation_run.py",
                  "tests/test_bounded_pilot.py", "tests/test_candidate_duplicates.py", "tests/test_candidate_quality.py",
                  "tests/test_commercial_candidate.py", "tests/test_pilot_diagnosis.py", "tests/test_pilot_routing_reporting.py",
                  "tests/test_preclamp_hold.py", "tests/test_preservation_diagnosis.py", "tests/test_preservation_tradeoff.py",
                  "tests/test_preservation_verification.py", "tests/test_production_safeguards.py",
                  "tests/test_push_baseline_ab.py", "tests/test_region_focus_ab.py", "tests/test_repeatability_ab.py",
                  "tests/test_routing_ab.py", "tests/test_training_determinism.py") }
}

# Credit FXTD Studios Research on every commit (no other co-author).
$coauthor = "Co-authored-by: FXTD Studios Research <research@fxtdstudios.com>"

foreach ($c in $commits) {
    $missing = @($c.paths | Where-Object { -not (Test-Path $_) })
    if ($missing.Count) { Fail "missing: $($missing -join ', ')" }
    $changed = @(git status --porcelain -- $c.paths)
    if ($changed.Count -eq 0) { Write-Host "nothing to commit for: $($c.msg.Substring(0, 50))..."; continue }
    Invoke-Git add -- $c.paths
    Invoke-Git commit -m $c.msg -m $coauthor -- $c.paths
}

$left = @(git status --porcelain)
if ($left.Count) { Write-Host "Not committed (left for you):" -ForegroundColor Yellow; $left | Write-Host }

# AGENTS.md: no AI assistant credited on GitHub.
$credits = (git log origin/native..HEAD --format=%B) -join "`n"
if ($credits -match '(?i)anthropic\.com|co-authored-by:\s*claude|claude-session') { Fail "a commit credits Claude; amend it first (AGENTS.md)" }

if ($Push) {
    Invoke-Git push origin native
    Write-Host "`nnative pushed. Watch the checks: https://github.com/FXTD-Studios/RUDRA/actions" -ForegroundColor Green
} else {
    Write-Host "`nCommitted locally, not pushed. Review with: git log --stat origin/native..HEAD" -ForegroundColor Green
    Write-Host "Then push with: .\scripts\release_shadows_2026-10-15.ps1 -Push   (or git push origin native)"
}
