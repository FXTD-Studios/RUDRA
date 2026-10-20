# RUDRA — current development handoff

Saved 14 October 2026 (evening), after the frozen-checkpoint diagnosis and the push-baseline packet.
Read this first when switching models or starting a new conversation. This is
the current continuation point; older approval requests and next actions in
the progress history are historical, not pending instructions.

## Resume here

**Getting Started widget: done (14 Oct).** The native five-step guide is in
`native/app/getting_started.cpp/.hpp`, wired through `MainWindow::show_getting_started`
and the `studio.qss.in` styles. `AppHelp.GettingStartedTheGuideAndTips` and the
10-test help/startup regression passed on Windows (`reports/getting_started_*_20261014.json`).
The native capture is `docs/design/getting-started-native.png`. After that run,
the Deliver step regained the HDR10+ / Dolby Vision, autosave and F1 lines (text
only, no test asserts it). Rebuild before the next native release.

**Latest completed milestone:** read-only frozen-checkpoint diagnosis:
[PUSH_BASELINE_DIAGNOSIS](ai/PUSH_BASELINE_DIAGNOSIS.md). The real-PU21 loss
is three dark Sparks frames (< 3 nits, −22 to −30 dB in every trained arm).
Without them the deterministic model is +0.44 dB. The likely cause is that
~41% of training draws are ACES exposure pushes read through the BT.1886
baseline, which teaches a 5–13 stop dark lift.

**Push-baseline A/B: completed 14 Oct, hypothesis refuted.**
[Result](ai/PUSH_BASELINE_RESULT.md).

- The control reproduced the verified scores exactly.
- The treatment made real PU21 −2.53 dB worse, the dark frames −4.21 dB
  worse, and crushed error −2.18 / −3.45 stops worse. Only clipped error
  improved slightly.
- The BT.1886-labelled ACES pushes are not the cause of the dark lift.
- No promotion and no retry. There is no running experiment.

**Read-only dark-tone check: done.** [DARK_TONE_TARGETS](ai/DARK_TONE_TARGETS.md).

- Every title's real HDR sits 2.4–7.5 stops above the BT.1886 baseline below
  SDR code 20. That includes held-out Meridian, with a tight spread.
- The training titles ask for about 0.1–0.8 stops more lift than Sparks
  between codes 10 and 159.

**Plan 4 check:** [PLAN4_STATUS_CHECK](ai/PLAN4_STATUS_CHECK.md).

**Analytic shadow correction scored (read-only).**
[SHADOW_CORRECTION_RESULT](ai/SHADOW_CORRECTION_RESULT.md). At the
pre-declared knee K = 40, PU21 against plain BT.1886:

| Set | PU21 gain | Frames improved | Scene 95% CI |
|---|---:|---:|---|
| Sparks bench 48 | +0.038 dB | 44/48 | [+0.017, +0.066] |
| Sparks 200 | +0.076 dB | 189/200 | [+0.017, +0.163] |
| Meridian 287 | +1.30 dB | 287/287 | [+0.68, +1.88] |

Knees 20 and 80 are not robust. On the dark Sparks frames the shadow error
gets slightly worse.

**Opt-in "BT.1886 shadows" picker entry: implemented 15 Oct, uncommitted.**
The default stays BT.1886. See the update section of
[SHADOW_CORRECTION_RESULT](ai/SHADOW_CORRECTION_RESULT.md).

- **Shipped table:** v3, with a minimum 0.02 stop per code. Through the app's
  knot path it scores +0.040 / +0.077 / +1.302 dB.
- **Rejected:** the black-level variant and the censored variant both fail on
  Sparks.
- **Changed files:**
  - native: `core` source_curve (hpp and cpp), `engine/actions.cpp`,
    `app/main_window.cpp` (tab map), `deliver/master.cpp` (message),
    `tests/test_source_curve.cpp`;
  - Python: `rudra/source_baseline.py`, the knots fixture,
    `tests/test_shadow_correction.py`, `tests/test_v10_2026_10_13.py`;
  - docs: USER_GUIDE, BETA.
- **Windows, 15 Oct:**
  - `rudra_tests` **237/237 passed** after the GTest include fix in
    `native/tests/CMakeLists.txt`. Conda's nlohmann_json prefix had shadowed
    the fetched GTest headers; the same fix as `rudra_app_tests`.
  - The focused Python run passed 45/45.
  - `rudra_app_tests` **41/41 passed**. The offscreen Qt plugin was missing
    from the build folder; `qoffscreen.dll` was copied into
    `build\native_gate_b\app\Release\platforms\`, a build-folder change
    only.
- **Next (Windows):**
  1. Done: all native and focused Python tests pass.
  - **Commit script:** `scripts/release_shadows_2026-10-15.ps1`.
    - It always makes commit 1 (Getting started) and commit 2 (BT.1886
      shadows).
    - `-IncludeResearch` adds commit 3, the 13–14 Oct research line, which was
      already uncommitted before.
    - `-Push` pushes; without it nothing is pushed.
    - It credits FXTD Studios Research only and refuses to push a Claude
      credit. `docs/ai/` and `reports/` stay local.
  2. Run the Python suite.
  3. Review dark scenes on a calibrated HDR display.
  4. Commit with the FXTD credit only.

**Earlier status, now superseded: owner approval of the push-baseline A/B**
([packet](ai/PUSH_BASELINE_APPROVAL.md)). It needs two decisions:

1. The updated internal evidence receipt: only the `sdr2hdr_dataset.py` pin
   changes.
2. The compute: two 200-step arms, 15 min each, 30 min combined, about 228 GB
   of read-only hashing.

Re-run the focused tests on Windows first. On approval, follow "On approval"
in the packet. Do not launch from the draft, and do not reuse earlier
receipts.

There is no running experiment. New expensive training or downloads need a
fresh estimate and approval. Safe read-only diagnostics, small tests and
focused reversible engineering remain authorized by the master request.

## Roadmap stage ledger

| Phase | Status | Evidence / remaining work |
| --- | --- | --- |
| 1. Repository, dataset, license, architecture audit | Initial audit completed; blockers recorded | [Execution audit](ai/RUDRA_EXECUTION_AUDIT.md), [architecture](ai/RUDRA_ARCHITECTURE_MAP.md), [dataset/license audit](ai/RUDRA_DATASET_LICENSE_AUDIT.md). Whole-library duplicate and legal clearance coverage are not complete. |
| 2. Validate existing v10 | Failure established; no promotion | v10 completed 20k steps and fails all clip gates. Other full-frame conditions were deliberately not rerun after the clip-first failure. Historical validation is contaminated by ancestor training. |
| 3. Commercial preparation and clean-training strategy | Internal-only candidate and scratch safeguards implemented; release clearance blocked | [Candidate validation](ai/COMMERCIAL_CANDIDATE_VALIDATION.md), [license decision](ai/LICENSE_DECISION_20261013.md), [training contract](ai/COMMERCIAL_TRAINING_CONTRACT.md). Owner accepted internal pilot interpretation, not distribution rights. |
| 4. Lightweight sky/fire/highlight prototypes | Not completed; gated by current quality failure | Existing routing, preservation and region-focus experiments tested prerequisites. No production-ready scene-aware enhancement established. |
| 5. Training, optimization, evaluation | Bounded scratch diagnostics performed; production training not achieved | Repeatability verified; every trained candidate still fails quality. The diagnosis isolates a dark-frame lift with a push-baseline mechanism; a single-factor A/B is prepared and awaits approval. No long training, convergence or superiority claim. |
| 6. Native integration and readiness | Not started for a qualifying neural model | Existing analytic/native defaults preserved. Competitors, temporal validation, display review, platform/package/signing/dependency checks remain. |

## Completed experiment sequence — do not redo

1. Audited v10 ancestry: v10 → v9 → v8 includes non-commercial HdM material.
   All 728 real v10 validation rows were seen by ancestors. These weights are
   research-only; fine-tuning does not establish commercial clearance.
2. Prepared a separate 5,944-pair internal candidate manifest, 36 shots,
   11,888 media files, approximately 114 GB. Train: 3,297 pairs; validation:
   Sparks 2,360; regression test: Meridian 287, not pristine. Source datasets
   were never modified. Full-file duplicate and sampled-quality checks passed
   within the documented scope; this is not exhaustive HDR/legal clearance.
3. Initial scratch smoke pilot failed. [Result](ai/PILOT_SMOKE_RESULT.md).
4. Fixed source-baseline routing and reporting; paired routing trial failed
   to produce a promotable checkpoint. [Result](ai/ROUTING_AB_RESULT.md).
5. Identified a post-clamp crushed-hold gradient dead zone. Implemented optional
   pre-clamp hold with tests; paired trial improved some crushed-region scores
   but damaged real/clipped quality. [Result](ai/PRESERVATION_AB_RESULT.md).
6. Frozen-checkpoint region/gradient diagnosis showed broader ordinary-region
   damage. Do not remove chroma loss merely because its gradient is large:
   measured initial alignment with PU improvement was positive.
   [Review](ai/PRESERVATION_TRADEOFF_REVIEW.md).
7. Region-focus 0 vs 0.5 trial improved some highlight scores but worsened real
   PU and crushed-region metrics. No promotion. A historical matched control
   diverged at step 45. [Result](ai/REGION_FOCUS_RESULT.md).
8. Added opt-in strict deterministic mode and runtime provenance. Synthetic
   fresh-process CUDA tests matched. First full-size pair timed out at repeat
   step 197; common records matched, but its verdict remains incomplete.
   [Partial result](ai/REPEATABILITY_RESULT.md).
9. Added separate wall-clock milestones and unbuffered worker output. A newly
   approved fresh pair with 900-second per-arm limits completed and passed
   exact repeatability. [Latest verified result](ai/REPEATABILITY_MARGIN_RESULT.md).
10. Read-only diagnosis from saved per-frame benches of all ten arms: the gain
    tracks frame brightness (Spearman 0.71–0.86), and three < 3-nit frames
    dominate. The training-target conflict was found and quantified.
    [Diagnosis](ai/PUSH_BASELINE_DIAGNOSIS.md). The single-factor runner and
    the updated receipt are prepared, not run. [Packet](ai/PUSH_BASELINE_APPROVAL.md).
11. Approved push-baseline A/B: completed and verified, hypothesis refuted.
    [Result](ai/PUSH_BASELINE_RESULT.md).

## Latest authoritative artifacts

Repository: `D:\A.I\Devlopments\rudra` (Windows PowerShell).
All paths below are relative to this repository.

- Master roadmap: `docs/RUDRA_PRODUCTION_ROADMAP.md` (copy of the supplied plan).
- Chronological progress: `docs/PRODUCTION_PROGRESS.md`.
- Latest report: `docs/ai/REPEATABILITY_MARGIN_RESULT.md`.
- Evidence copy: `docs/ai/repeatability_margin_result_20261014.json`.
- Approval/pinned code: `reports/repeatability_margin_review_20261014/`.
- Actual completed outputs: `reports/repeatability_margin_ab_20261014/`.
- Frozen trained checkpoints: `control/step_0000200.pt` and
  `repeat/step_0000200.pt` under that output directory.
- Frozen initial checkpoints: each arm's `step_0000000.pt`.
- **Each `best.pt` is step zero. Use the explicit step-200 checkpoint when
  diagnosing the trained model; do not mistake best.pt for trained weights.**
- Source snapshots: review packet `source_snapshot/`; verifier snapshot under
  output `verification_source_snapshot/`.
- Manifest: `reports/approved_internal_pilot_20261013/manifest.jsonl`.
- Internal rights receipt: `reports/routing_ab_review_20261013/review.approved.json`.
- `verified_summary.json`, `comparison.json`, per-arm status/config/train/bench
  journals, sample replay and timing journals remain in the output directory.

Approved recipe SHA256:
`f08727353d57a09053c893beba3eeaaaffddf5888b4baf5db50ad938641efd3e`.
Shared final model tensor fingerprint:
`affd6e411a9c721166331ce34f63ad09bbbc2caeee18a8da1a8f7338fe047223`.
Control checkpoint file SHA256:
`e760bb7871560e29708c886835458d7ce888386bfe8d098add05e63761f36a77`.
Repeat checkpoint file SHA256:
`4816228e314eecd4c38f0cbf87079f6273dccf7e47dee7a61922b66b0042fbfc`.
Whole-file hashes differ because metadata contains arm/output identities.

## What was actually established

Both 200-step runs match all 800 input draws, initial tensors, runtime/settings
apart from identity, every training/evaluation record, final model tensors and
full benchmark reports. Independent verification passed. Durations were
631.04 and 460.53 seconds after preflight (18.19 minutes combined). Both
11,888-file preflights passed. Artifacts approximately 49.1 MB.

Quality remains below the analytic baseline, identically in both runs:

| Metric | Value |
| --- | ---: |
| Real PU21 gain | -1.29023 dB |
| Pooled clipped gain | -0.09660 stops |
| Clipped +1 / +2 EV | -0.15131 / -0.04310 stops |
| Crushed +1 / +2 EV | -2.60237 / -1.22459 stops |
| Pushed PU21 +1 / +2 EV | -15.64336 / -7.73851 dB |

Repeatability is demonstrated only for the fixed recipe/environment. The
historical drift's specific cause and observed throughput variation are not
established. No checkpoint is promoted or commercially cleared.

## Push-baseline A/B acceptance criteria (historical: completed, do not rerun)

These were the criteria for the push-baseline A/B. It ran once, was verified
and refuted its hypothesis; no promotion, continuation or retry. They are kept
as a record only. The current next step is at the top of this file.

1. Re-run the focused tests on Windows: `tests/test_push_baseline_ab.py` plus
   the 97-test suite. 131 passed on CPU Linux.
2. Get the owner's explicit decisions on the draft receipt and on the compute.
   Write `review.approved.json`, then `recipe.approved.json` pinned to it,
   then `approval.json`. Never edit the drafts.
3. Run `training/routing_ab.py` once, then `tools/verify_preservation_run.py`
   once.
4. Report the pre-registered verdict, the dark-frame split, and the clipped and
   pushed tradeoffs. Do not promote, continue or retry.

## Implementation and verification map

- `training/train_sdr2hdr.py`: existing trainer, optional pre-clamp diagnostics,
  strict deterministic mode, runtime record, optional pushed PU reporting.
- `training/routing_ab.py`: approved scratch pair runner, pins, replay guards,
  watchdogs, exact repeatability comparison, timing journals.
- `training/routing_replay.py`: deterministic input replay/fingerprints.
- `training/preservation.py`, `rudra/sdr2hdr.py`: optional hold validation and
  pre-clamp outputs; working native/color-science behavior preserved.
- `pipeline/commercial_preflight.py`: internal/commercial candidate evidence,
  source hashes, scratch ancestry and separated outputs.
- `--push-baseline {picker,render}` (trainer and dataset; the default is
  unchanged), scope `internal-push-baseline-ab` in `training/routing_ab.py`,
  `allowed_baseline_change` in the replay.
- `tools/frame_luminance_breakdown.py`: read-only grouping of saved benches by
  reference brightness, and `--baseline-mismatch`.
- `tools/verify_preservation_run.py`: actual-artifact validation. It refuses to
  overwrite an existing verified summary. Read the saved result, do not rerun
  it into the same completed output just to get a second copy.

Latest focused suite: **97 passed in 5.61 s** before the approved run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_repeatability_ab.py tests/test_region_focus_ab.py tests/test_routing_ab.py tests/test_preclamp_hold.py tests/test_preservation_verification.py -q
```

Run relevant tests after new changes; no need to repeat expensive completed
experiments. Runtime used PyTorch 2.13.0+cu130, CUDA 13.0, RTX 4080 SUPER,
BF16, strict deterministic algorithms, TF32 disabled, zero loader workers.
Strict training must start in a fresh process before CUDA initialization.

## Workspace and permission boundaries

- This work is saved locally but **uncommitted**. Many modified and untracked
  files predate this handoff. Preserve them; no blanket reset/clean/removal.
  `reports/` holds essential local artifacts and may be ignored by Git.
- Read `AGENTS.md`. Use `git --no-optional-locks` for git reads; writes only in
  Windows host shell. No commit/push was requested. Follow FXTD credit rules.
- Treat all `G:\` source data as read-only. Keep derived outputs in separate
  repository report directories; never overwrite completed experiment evidence.
- Owner accepted internal pilot rights interpretation only. License conflicts
  and external review remain; attribution drafts are not legal clearance.
- Keep research-only ancestry separate from scratch candidates. Preserve
  native color science, HDR formats, artist controls and analytic default.
- No large downloads, expensive training or destructive/irreversible action
  without the user's required approval. Do not interpret historical “approved”
  receipts as permission for another run.
- Routine inspection/testing/documentation can proceed with a lighter model.
  Tell the user before consequential architecture/color-science/loss-strategy
  work needs stronger review; do not claim a particular model was consulted
  unless it actually was.
