# RUDRA production progress

**New model/session: start with [CURRENT_HANDOFF.md](CURRENT_HANDOFF.md).**
It records phase completion, authoritative artifacts, permissions and the exact
next task. Previous increments below are historical, including approval requests.

Updated 14 October 2026. Baseline: `native` / `428bd66`.
Master: [supplied roadmap](RUDRA_PRODUCTION_ROADMAP.md), copied verbatim.
Execution follows the user's authorization for safe incremental changes; the
embedded roadmap's audit-only pause does not override that authorization.

## Latest increment: opt-in BT.1886 shadows picker entry implemented

- Re-scored through the app's knot path; two further variants were tried and
  both fail on Sparks. The shipped table (v3, monotone) scores:
  - Sparks bench 48: +0.040 dB, 45/48 improved;
  - Sparks 200: +0.077 dB, 191/200 improved;
  - Meridian 287: +1.302 dB, 287/287 improved.
- Native and Python implementations, appended last so earlier ids keep their
  meaning; Python matches native knots within 1e-6.
- The core checks pass (g++), and 94 Python tests pass.
- Needed before release: the Windows native and app test runs, and a
  calibrated-display review.
- BT.1886 stays the default.

## Previous increment: analytic shadow correction scored

- A per-code BT.1886 shadow correction below code 40 was fitted on the
  training titles only and scored read-only on 535 held-out frames:
  - Sparks bench 48: +0.038 dB, 44/48 improved;
  - Sparks 200: +0.076 dB, 189/200 improved;
  - Meridian 287: +1.30 dB, 287/287 improved.
- Every scene confidence interval is above zero.
- Knee 40 was declared before scoring; 20 and 80 fail on Sparks.
  [Result](ai/SHADOW_CORRECTION_RESULT.md).
- No training and no default change. Next: an opt-in picker entry, on GO.

## Previous increment: read-only dark-tone check and plan 4 status

- 300 real pairs (60 per title) read on CPU in 69 s. No writes to G:.
- Below SDR code 20, every title's HDR sits 2.4–7.5 stops above the BT.1886
  baseline, held-out Meridian included.
- The training titles want about 0.1–0.8 stops more lift than Sparks between
  codes 10 and 159; Nocturne varies widely frame to frame.
  [Report](ai/DARK_TONE_TARGETS.md).
- The plan file matches `docs/RUDRA_PRODUCTION_ROADMAP.md`. Phase status and
  the five next actions are in [PLAN4_STATUS_CHECK](ai/PLAN4_STATUS_CHECK.md).
- Next (awaiting GO): score an analytic shadow correction. No training.

## Previous increment: push-baseline A/B refuted

- The owner approved the updated receipt and the compute. 131 tests passed on
  Windows first.
- Both arms completed and were verified:
  - 800 matched draws, with only the 342 declared baseline changes;
  - 1,277.5 s combined, 49.1 MB;
  - the control reproduced the verified scores exactly.
- Render vs control:
  - real PU21 **−2.531 dB**, dark frames −4.21 dB;
  - crushed −2.18 / −3.45 stops;
  - pushed PU21 −4.05 / −2.80 dB;
  - clipped +0.128 / +0.085 stops.
- The pre-registered test fails. This mechanism does not cause the dark lift.
  [Result](ai/PUSH_BASELINE_RESULT.md).
- Next: a read-only check of the training titles' dark-tone targets against
  BT.1886. No promotion and no retry.

## Previous increment: dark-frame diagnosis and push-baseline A/B prepared

- **Getting Started widget finished:** native five-step guide, Windows
  tests passed. The Deliver step regained the HDR10+ / Dolby Vision, autosave
  and F1 lines.
- **Read-only diagnosis** from the saved per-frame benches of all ten arms. No
  inference and no media reads:
  - Real PU21 −1.290 dB is three Sparks frames below 3 nits (−22 to −30 dB);
    without them the model is **+0.436 dB**.
  - Gain tracks brightness in every arm (Spearman 0.71–0.86).
  - [Diagnosis](ai/PUSH_BASELINE_DIAGNOSIS.md).
- **Mechanism (likely, not yet causal):**
  - ~41% of draws are ACES exposure pushes on the BT.1886 baseline, which reads
    them 5–13 stops too dark below 10 nits.
  - The network cannot tell those from real SDR, so it learns a dark lift.
- **Prepared, not run:** a `--push-baseline render` single-factor A/B on the
  deterministic recipe, with:
  - matched draws, and only BT.1886 → ACES id changes allowed;
  - a pre-registered verdict;
  - verifier support.
- **Updated receipt draft:** only the dataset pin changes, and no decode line
  changed. [Approval packet](ai/PUSH_BASELINE_APPROVAL.md).
- **131 focused tests passed** (CPU Linux) and 224 wider tests passed. Re-run
  them on Windows before approval.
- **Next:** the owner's receipt and compute decisions. No training started.

## Previous increment: full-size repeatability verified

UI priority added afterward: user requested a professional Getting Started
redesign, preview first. Proposal saved in `docs/design/getting-started-preview.html`
and `.png`; native widget unchanged pending design review. Quality diagnosis
remains the next research task after this UI work.

- Both fresh runs completed 200 steps. All **800 draws**, **200 training
  records**, final model tensors, evaluation logs and benchmark reports match
  exactly. Independent post-run verification passed; all approved pins match.
- **631.04 / 460.53 s**, combined **18.19 min** after preflight, within budget.
  Both 11,888-file hash checks passed; artifacts about 49.1 MB.
- Repeatability passes for this recipe/environment. Quality remains below
  analytic baseline: real PU21 **-1.29023 dB**, pooled clip **-0.09660 stops**;
  both best checkpoints are step zero. No promotion, retry or continuation.
- **97 pre-launch tests passed**; approved training code unchanged during run.
  [Verified result and timing limitations](ai/REPEATABILITY_MARGIN_RESULT.md).
- Next: frozen-checkpoint quality diagnosis using this deterministic baseline
  before another recipe change. Release licensing remains unresolved.

## Previous increment: approved revised repeatability pair launched

- Owner approved the revised pair. All 11 source pins/snapshots, resolved
  arguments and manifest/receipt hashes matched before launch.
- Approved recipe SHA256:
  `f08727353d57a09053c893beba3eeaaaffddf5888b4baf5db50ad938641efd3e`.
- Read-only preflight started. Each fresh run is limited to 900 seconds after
  preflight; shared artifact stop remains 900 MB. No automatic retry,
  continuation or promotion. Completed result is recorded above.

## Previous increment: timing review and revised repeat prepared

- Rounded cumulative throughput implies the repeat was already about 54 s
  slower at step 100. Logs cannot identify the component responsible.
- Added unbuffered worker output and separate coarse timing journals. Training
  settings are unchanged; only repeatability may request a 900 s ceiling.
- **97 tests passed in 5.61 s**. Eleven source files pinned/snapshotted;
  resolved training settings match the prior packet except output directory.
- [Review and pending approval](ai/REPEATABILITY_MARGIN_REVIEW.md): fresh
  200-step pair, 15 min/arm after preflight, 30 min combined; ~228 GB read-only
  verification, 1 GB storage budget. No new training started.
- Next: approval of this revised bounded compute. No numerical improvement or
  root cause is claimed; native defaults and internal-only licensing unchanged.

## Previous increment: repeatability pair stopped at approved cap

- Control completed 200 steps and evaluation in **524.91 s**. Repeat hit the
  **600 s cap** at 197 logged steps; no final repeat checkpoint/evaluation.
  No retry or continuation. Full repeatability verdict remains incomplete.
- All **197 common training records match exactly**, as do 791 recorded
  input draws, initial weights/evaluation and runtime/configs except identity.
- Both 11,888-file preflights passed. All approved code pins still match.
  Control remains below baseline: real PU21 **-1.29023 dB**, best step zero.
- **26 verifier/repeatability tests passed**; artifact size about 34.6 MB.
  [Result and limitations](ai/REPEATABILITY_RESULT.md).
- Next: inspect throughput evidence and prepare a revised bounded repeat with
  evaluation margin. No further training authorized by this completed receipt.

## Previous increment: approved repeatability pair launched

- Owner explicitly approved the bounded pair. Separate approval receipt and
  approved recipe preserve the pending draft; all 11 source pins/snapshots,
  resolved arguments and manifest/receipt hashes verified before launch.
- Read-only preflight started; outcome is recorded above.
- Approved recipe SHA256:
  `bff683152a0ffe23a272f9b1b96684b96b2aa6734795076dbd81421f29a91ab3`.
- No retry, continuation or promotion. Per-arm 600-second post-preflight cap
  and shared 900 MB stop remain enforced by the existing watchdogs.

## Previous increment: full-size repeatability pair prepared

- Added a same-recipe repeat mode to the existing bounded runner. Both arms
  use strict determinism, pre-clamp hold 1, region focus 0 and full-size inputs.
- Exact comparison covers every-step logs, final model tensors and benchmark
  reports; initialization/input/runtime mismatches invalidate the experiment.
- **76 tests passed in 4.82 s**. Eleven source files are pinned and snapshotted;
  manifest/receipt hashes match. Pending approval correctly blocks launch.
- [Review packet and resource estimate](ai/REPEATABILITY_APPROVAL.md): two
  200-step runs, 20-minute combined post-preflight cap, about 228 GB read-only
  media verification, 1 GB storage budget. No training started.
- Next: owner approval of this bounded compute, then execute without retries.

## Previous increment: deterministic training safeguard validated

- Added opt-in strict deterministic training and runtime/precision provenance
  in configs and checkpoints. Defaults and native inference remain unchanged.
- **81 tests passed in 33.66 s**. Two fresh-process synthetic CUDA BF16 runs
  matched all 60 losses/gradient norms and final weights exactly.
- Another **55 reporting, bounded-pilot and production-safeguard tests passed**
  in 3.49 s; whitespace checks passed (136 tests across both invocations).
- Historical step-45 drift remains unexplained; this small test does not prove
  full-size corpus repeatability or improve the failed quality gates.
- No dataset training or source-data changes. Next: prepare a newly pinned,
  bounded same-recipe repeatability comparison before changing the objective.
  [Evidence and limitations](ai/TRAINING_REPEATABILITY.md).

## Previous increment: approved region-focus comparison completed

- Owner approved the frozen two-arm budget. Separate
  [approved recipe](ai/region_focus_recipe.approved.json) and
  [compute receipt](ai/region_focus_approval.json) preserve the pending draft.
  All 11 implementation pins and snapshots matched before launch.
- Both arms completed **200 steps**, each passing 11,888-file preflight.
  Combined time after preflight **16.44 min** (9.14 / 7.31), within budget;
  artifacts about 49 MB. All 800 draws and initial tensors match.
- The first execution was interrupted during preflight (last logged milestone:
  5,000 files; zero training steps). On the owner's explicit request to continue,
  preserved its two partial log/status files under
  `reports/region_focus_ab_20261014_interrupted_preflight/` and restarted the
  same approved pair. All pins revalidated; fresh hashing adds disk traffic.
- Region focus improved clipped and pushed full-frame scores but worsened
  real PU21 **-0.80385 -> -2.07993 dB** and both pushed crushed scores. The
  planned hypothesis failed. Both best checkpoints remain step zero; no promotion.
- A reproducibility limitation emerged: control
  matches prior pre-clamp initialization and all 800 input draws, but training
  logs first diverge at step 45 and final weights differ. Real PU21 is -0.80385
  dB versus the prior -3.78189 dB. Cause is not established; interpretation must
  use the current paired control and retain this substantial variation caveat.
- **162 pre-launch tests** apply to unchanged pinned code; **11 verifier tests
  passed** separately. Actual post-run verification passed. [Full result](ai/REGION_FOCUS_RESULT.md).
  Next: audit/harden numerical repeatability before further recipe changes or
  longer training. Internal-only licensing and native defaults remain unchanged.

## Previous increment: region-focus comparison prepared for approval

- Extended the existing paired runner for region focus **0 vs 0.5**, with
  pre-clamp hold fixed at 1 in both arms. Resolved configurations differ only
  in region focus, arm name and output path. Defaults and native code unchanged.
- Added optional pushed full-frame PU21 reporting and required coverage checks.
  All nine historical 48-frame aggregate scores reproduce exactly; pushed
  PU21 matches the independent diagnosis exactly. No checkpoint-selection change.
- **162 relevant tests passed**, 8.06 s, including a real-trainer synthetic
  two-arm integration. Inference compatibility took 53.27 s; no corpus training.
- [Frozen approval packet](ai/REGION_FOCUS_APPROVAL.md): two 200-step runs,
  <=20 minutes combined after preflight, <=1 GB outputs, approximately 228 GB
  read-only integrity traffic. Pending receipt is rejected; training output
  is absent. Existing internal-only rights receipt is reused unchanged.
- Next: explicit approval of this new compute budget, then run the frozen pair.

## Previous increment: preservation tradeoff reviewed without training

- [Frozen-weight review](ai/PRESERVATION_TRADEOFF_REVIEW.md) reproduced every
  historical frame score exactly. **99.21% of the net real PU-MSE increase**
  is outside raw clipped/crushed masks; below-1-nit and above-100-nit error
  worsens while 1-10 nit error improves. Existing inference controls still fail.
- Verified 16 crops / 48 parameter-gradient probes, then six complete BF16
  batches / 18 probes. Original step-one loss and gradient norm reproduce
  exactly. Chroma strongly influences the initial gradient but agrees with
  local PU-MSE; removing it or merely increasing hold weight is not justified.
- **50 focused tests passed**, 3.52 s. Diagnostic execution: 47.68 + 10.54 s;
  zero optimizer steps. Checkpoints, source data and production code unchanged.
- Next: prepare a separately bounded region-focus 0 vs 0.5 comparison with
  pre-clamp hold fixed in both arms and matched replay. Include pushed full-frame
  PU21 alongside existing gates. New compute approval is required only after
  that concrete packet is prepared; current models remain internal-only.

## Previous increment: approved preservation comparison completed

- Owner approved the prepared two-arm, 200-step comparison. Finalized separate
  [approved recipe](ai/preservation_ab_recipe.approved.json) and
  [compute approval](ai/preservation_ab_approval.json), preserving the draft.
- Both arms stopped at **200 steps**: 8.85 / 7.66 minutes, **16.51 minutes
  combined after preflight**, within the 20-minute cap; artifacts about 49 MB.
  Each 11,888-file preflight passed. All 800 draws/baseline IDs and starting
  tensors matched; all 11 code pins and source snapshots remain unchanged.
- Pre-clamp hold improved pushed crushed gains **-4.03873 -> -2.87403** and
  **-2.39912 -> -2.06695 stops**, but worsened real PU21 **-1.79725 ->
  -3.78189 dB** and both clipped means. Training held-channel negative
  saturation fell **93.20% -> 61.15%**; the overall recipe still fails.
- The control reproduced all prior corrected-routing model tensors and
  benchmark fields exactly. Both best checkpoints remain step zero; no
  promotion, continuation or release. Internal-only rights limits remain.
- **210 pre-launch tests** apply to the unchanged implementation; **6 new
  verifier tests passed** separately. Actual post-run verification passed.
  [Result and limits](ai/PRESERVATION_AB_RESULT.md). Next: frozen-weight
  error/gradient review before selecting another bounded experiment.

## Previous increment: optional pre-clamp hold and bounded comparison ready

- Implemented opt-in pre-clamp crushed hold; historical post-clamp behavior
  remains the default. Same mask and weight, restoring gradients at black
  saturation. Added actual applied-residual capture and gradient diagnostics.
- Reused the paired runner with matching baseline IDs in both arms. Frozen
  draft changes only the hold domain; scratch initialization, replay, budget,
  finite-value checks and required per-EV scores remain enforced.
- **210 tests passed**, 7.30 s CPU, one existing warning. Synthetic paired
  integration passed; GPU bfloat16 restoring-gradient check passed without an
  optimizer update. Eight pre-change comparisons match outputs/loss/gradients
  exactly; all nine historical 48-frame benchmark fields also match exactly.
- [Concrete approval packet](ai/PRECLAMP_HOLD_APPROVAL.md): two 200-step arms,
  <=20 min after preflight, <=1 GB outputs, about 228 GB read-only preflight
  traffic. Compute approval was pending at preparation; the approved run is
  now completed above. No native changes, model promotion or new rights granted.

## Previous increment: preservation failure diagnosed

- [Research review](ai/PRESERVATION_REVIEW.md): post-clamp crushed hold has
  zero gradient on saturated black predictions. Over 99.99% of corrected-arm
  pushed crushed pixels become RGB zero. All historical scores reproduced exactly.
- Existing highlight-only inference raises corrected real PU21 **-1.79725 ->
  +1.64932 dB** and removes the pushed crushed regression, but pushed full-frame
  PU21 and +1 EV clipping still fail. This is diagnosis, not model promotion.
- Verified 16 recorded training crops, 48 frozen-weight gradient probes.
  Region focus excludes most saturated crushed pixels; a pre-clamp hold has
  restoring gradients. Broader dark-tone preservation remains a separate issue.
- **173 tests passed** (171 + 2); one existing warning. Diagnosis took 45.44 s.
  Added diagnostic tooling/tests only; no training, recipe/model/native changes.
- Next: prepare a default-preserving, opt-in pre-clamp-hold experiment, then
  obtain approval for its separately frozen two-arm 200-step comparison. Keep
  region focus and other loss weights fixed to isolate the change.

## Previous increment: approved routing A/B completed

- Owner explicitly approved the two bounded runs. Finalized separate
  `routing_ab_review_20261013` receipt/recipe/approval, preserving drafts and
  the previous pilot. All frozen code pins verified before launch.
- Both arms completed 200 steps; all 800 paired draws and initial weight hashes
  matched. Baseline selection changed on 67 draws (BT.1886 -> ACES). Both
  11,888-file preflights passed; source media and prior runs remain unchanged.
- Corrected routing improved pooled clipped gain **-0.21664 -> -0.02007 stops**
  but worsened real PU21 **-0.11233 -> -1.79725 dB**, including every sampled
  shot. Pushed crushed gains remain about -4.04 / -2.40 stops at +1/+2 EV.
  Neither checkpoint is promoted; both `best.pt` files remain step zero.
- Control 8.16 min, corrected 6.73 min: **14.89 min combined after preflight**,
  within the 20-minute cap. Artifacts approximately 49 MB, below the 1 GB cap.
  All final parameters finite; code pins and checkpoint/replay evidence verified.
- [Full result and limitations](ai/ROUTING_AB_RESULT.md). The 169 pre-launch
  tests still correspond to the unchanged code. No automatic retry, longer run,
  native integration or distribution. Next: research review of preservation
  objectives/region focus before specifying another experiment.

## Previous increment: controlled routing A/B runner ready for approval

- Added isolated deterministic augmentation replay and a sequential two-arm
  runner. The corrected arm verifies control input/target fingerprints before
  optimization; both arms verify identical scratch initialization. Normal
  training/model/native behavior is unchanged by the experiment wrapper.
- Enforced 200-step stops with the original 1,000-step scheduler horizon,
  zero workers, finite checks, region coverage, 10 minutes per arm after
  preflight, and a shared 900 MB stop threshold below the requested 1 GB cap.
  Parent/worker watchdogs and fail-closed code/evidence/approval pins are in place.
- **169 tests passed**, 6.70 s CPU, one existing warning. Includes a two-step
  per-arm synthetic CPU trainer integration test and approval/replay/budget
  rejection tests. No source-corpus training or new quality claim.
- Prepared [the concrete approval packet](ai/ROUTING_AB_APPROVAL.md): frozen
  draft recipe, updated evidence template, expanded arguments and explicitly
  unapproved receipt. Previous owner approval was not copied into a new grant.
- Next: owner approval for two 200-step scratch arms, <=20 minutes combined
  after preflight, <=1 GB artifacts. Expect approximately 228 GB total source
  reads for independent preflights; no downloads/source writes. The actual
  experiment output directory has not been created.

## Previous increment: routing correction and diagnostic reporting

- Fixed pushed-real baseline selection: undegraded exposure pushes now select
  ACES, subject to existing dropout; degraded and unpushed real inputs retain
  BT.1886. Training objective, region focus and native inference are unchanged.
- Added frame-weighted crop PU21, consumed training baseline counts, and
  `bench_val.jsonl` with frame/scene/EV results and explicit missing coverage.
  Existing aggregate scores and checkpoint selection remain unchanged.
- **148 tests passed**, 5.85 s CPU, one existing warning. The existing two-step
  synthetic CPU training test verifies saved reports and counts; no source-corpus
  training started. `git diff --check` passed.
- Inference-only rerun of the failed step-200 checkpoint on the original
  48 Sparks frames took 27.38 s and reproduced all nine original benchmark
  fields **exactly**. Newly exposed pushed crushed gains are -3.44502 stops
  at +1 EV (33 frames) and -1.88934 stops at +2 EV (13 frames).
  These diagnose the old checkpoint, not the effect of the training fix.
- [Changes, measurements and limits](ai/PILOT_ROUTING_REPORTING.md).
  Next: prepare a deterministic paired-input routing A/B runner and separate
  receipt/recipe. New source-corpus training remains approval-gated. The old
  receipt's pinned reader hash no longer matches; it was preserved unchanged.

## Previous increment: diagnostic review completed

- [Review and bounded experiment proposal](ai/PILOT_DIAGNOSTIC_REVIEW.md),
  [measured evidence](ai/pilot_diagnosis_20261013.json). Re-evaluated the same
  48 Sparks frames using the retained step-200 checkpoint; no new training.
- On identical full-frame predictions, log PSNR improves **+1.71492 dB** while
  PU21 loses **-2.03128 dB**. Dark-channel errors grow despite brighter-channel
  improvements. The sign disagreement persists on training-sized crops;
  half-precision input caching changes the mean by only 0.0000358 dB.
- Confirmed pushed real samples always select BT.1886 in training, while the
  pushed benchmark selects ACES. All pilot rows are real, so no training
  baseline selection exercised ACES. Also confirmed the scratch recipe uses
  `region_focus=0`, unlike v10's 0.5. These are separate experiment factors.
- The 200-step stop remains valid under its frozen rule; it does not establish
  architecture failure. Context sensitivity and limited live-action sampling
  remain concerns, not isolated causes. No new architecture is justified yet.
- Added an inference-only diagnostic and metric/serialization checks:
  **131 tests passed**, 5.82 s CPU, one existing warning. All 48 frame records
  were retained after a final summary-serialization error; fixed serialization
  and recovered the report without rerunning inference.
- Next: tested routing correction and matching perceptual/per-EV reporting,
  keeping the loss unchanged. A separate two-arm 200-step routing comparison
  is proposed at <=20 minutes total after preflight and <=1 GB artifacts;
  **new training approval is pending**. Source data, weights, frozen recipe,
  native inference and release restrictions remain unchanged.

## Previous increment: approved scratch pilot stopped at smoke gate

- User accepted the documented license interpretation and bounded internal
  scratch pilot. [Approval](ai/pilot_owner_acceptance_20261013.json).
  Release remains prohibited; no blanket commercial clearance is asserted.
- Created a separate reviewed manifest, evidence receipt and frozen recipe in
  `reports/approved_internal_pilot_20261013/`. Source datasets and original
  pending candidate remain unchanged. All 11,888 media files passed independent
  hash verification before training.
- Added explicit internal-pilot receipt scope and a budgeted runner: 30-minute
  smoke, conditional <=60-minute continuation, 1,000 steps maximum, storage
  safety threshold 900 MB below the approved 1 GB cap. Failed numerical or
  sampled-validation smoke checks stop before step 201.
- **126 tests passed**, 5.93 s CPU, one existing warning. Includes time-budget,
  smoke rejection, scope isolation and runner checkpoint/stop integration tests.
- **Stopped at step 200**, no continuation: sampled real PU21 **-2.0313 dB**,
  clipped-region gain **-0.07608 stops**, crushed-region gain +0.54686 stops.
  Coverage: 48 real frames; clip aggregates 89 frame/EV observations; crushed
  only 4 frames. Crop gains do not override failed bench screening.
- Elapsed after preflight **5.71 min**; peak PyTorch allocation **3.16 GiB**;
  all run artifacts **36.10 MB**. `best.pt` remains step 0; step 200 is retained.
  [Results](ai/PILOT_SMOKE_RESULT.md). No promotion or weight distribution.
- At that point: stronger-model diagnosis before any objective change or retry;
  that review is now complete above.
  No architecture, objective, native inference or source media changes.

## Previous increment: licensing interpretation and attribution for approval

- Rechecked the affirmative current Netflix/AOM CC BY 4.0 grant and saved
  the current page. Three old NC-ND text links return 404; that alone does not
  prove supersession. The remaining conflict is documented, not concealed.
- Prepared [a concrete license decision](ai/LICENSE_DECISION_20261013.md):
  recommend current CC BY 4.0 as the offered AOM license, preserving original
  Cosmos CC BY 3.0 attribution and historical notices. Owner acceptance or
  issuer clarification remains necessary before an approval receipt.
- [Attribution draft](ai/COMMERCIAL_ATTRIBUTION_DRAFT.md) and hash-bound inventory
  cover five titles, 36 shots and 72 archive URLs. Inventory totals and review
  evidence hashes verified. No code changed in this increment; last tests: 115.
- Pilot proposal is ready for the owner's licensing/budget decision: internal
  scratch only, 200-step smoke then <=1,000 total steps if checks pass, <=90
  GPU-minutes, <=1 GB checkpoints/logs. No model release, downloads or training
  started. Reviewed receipt and enforced budget remain prerequisites to launch.
- Stronger-model architecture review is separate from this licensing task;
  it cannot establish issuer rights and is not a substitute for evidence.

## Previous increment: reject invalid manifests before media reads

- Commercial preflight now checks every record and split before hashing media,
  avoiding a 114 GB reread before discovering a late invalid row. It rejects
  malformed digests and conflicting hashes for one path; valid runs still
  independently hash all unique files.
- **115 tests passed**, 5.65 s CPU, one existing warning. Eight added tests
  verify seven early-rejection cases and complete valid-path hashing.
- No source dataset writes, production training or native changes.
- **Stronger-model review is now needed before the next research decision**:
  interpreting v10 failures and accepting/changing the pilot objective/recipe.
  [Review handoff](ai/RESEARCH_REVIEW_HANDOFF.md). No model switch was made.
  Rights/attribution and compute approval remain unresolved separate gates.

## Previous increment: complete candidate screening and decoder safeguards

- Prepared a **review-only** real-pair candidate: 5,944 pairs, 36 shots,
  title-disjoint splits of 3,297 train / 2,360 validation / 287 regression test.
  [Candidate report, exact rights evidence and reproduction](ai/COMMERCIAL_CANDIDATE_V1.md).
- Full hash coverage: **11,888 files / 114.01 GB**, zero exact duplicates.
  All **5,944 SDR frames** decoded; zero cross-split perceptual flags. Thirteen
  low-information frames in one training transition remain outside structural
  matching. [Completed validation and limitations](ai/COMMERCIAL_CANDIDATE_VALIDATION.md).
- First/middle/last sampling across all 36 shots: **108 pairs**, no decode,
  finite-value, size or alignment-proxy flags. Source scaling agrees with
  recorded peaks within 0.0145% on the samples; this is not mastering validation.
- Saved conflicting exact-source license notices: a newer CC BY 4.0 license
  alongside an older README listing NC-ND/BY 3.0 terms. No approval receipt was
  issued; every candidate row stays `commercial_ok: false` pending review.
- Added three title-root license receipts; the conflicting older notice remains
  explicitly unresolved. All six saved evidence documents match their hashes.
- Commercial preflight schema 2 pins decoder code, storage sentinels and metadata
  sidecars, rejecting changed/missing evidence before expensive media reads.
  Added reusable quality and duplicate screening tools.
- **107 tests passed**, 5.43 s CPU, one existing warning, including a two-step
  synthetic training integration check. No production training, promotion,
  native changes or source dataset writes. [Proposed bounded scratch pilot](ai/CLEAN_TRAINING_PILOT.md)
  awaits exact rights/attribution review and compute approval.

## Completed

- Phase 1 initial repository, local dataset and active checkpoint audit:
  [findings and resource estimates](ai/RUDRA_EXECUTION_AUDIT.md),
  [machine-readable evidence](ai/production_audit_evidence_20261013.json),
  [14 sampled HDR/SDR pairs](ai/production_quality_samples_20261013.json).
  Source media remained read-only. Whole-corpus content/near-duplicate and
  legal certification remain incomplete; metadata counts are not clearance.
- Phase 2: existing v10 best checkpoint is step 20,000, not an unrun experiment.
  Recomputed existing clip CSVs; historical scores/statuses match exactly.
  [Frame verdict](ai/v10_revalidated_frame_20261013.json): **FAIL**.
  [Supplemental scene verdict](ai/v10_revalidated_scene_20261013.json): **FAIL**.
  Stop rule retained. No retraining, full inference rerun or model promotion.
- Evaluation safeguards: strict mode now rejects incomplete verdicts; v10
  rejects mismatched frame sets/finite coverage and duplicate/missing IDs.
  Optional scene bootstrap uses explicit matching scene IDs and equal scene
  weight; requires at least two scenes. Historical frame thresholds unchanged.
- Phase 3 foundation: commercial training preflight checks reviewed source
  evidence, manifest/media hashes, split isolation and separate empty output.
  Only scratch image initialization is supported. Research runs are labeled
  research; passing preflight means commercial **candidate**, not clearance.
  [Schema and workflow](ai/COMMERCIAL_TRAINING_CONTRACT.md).
- Corrected stale v10 status in STATUS and MODEL_PLAN_V10. No native/color
  science/artist control/output changes. No claimed quality or runtime gain.

## Validation

Initial audit: **62 tests passed** in 4.74 s on CPU; one pre-existing PyTorch warning. Suites:
production safeguards, v10, licences, CP7 gates, v8 bench, SDR2HDR, source-curve
training and source-curve model. `git diff --check` passed.

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='2'
.venv\Scripts\python.exe -m pytest tests/test_production_safeguards.py tests/test_v10_2026_10_13.py tests/test_licences_2026_09_23.py tests/test_cp7_gates_2026_09_24.py tests/test_v8_bench_2026_10_08.py tests/test_sdr2hdr.py tests/test_source_curve_training_2026_10_01.py tests/test_source_curve_model_2026_10_01.py -q
.venv\Scripts\python.exe training/cp10_verdicts.py --candidate v10 --strict --out docs/ai/v10_revalidated_frame_20261013.json
.venv\Scripts\python.exe training/cp10_verdicts.py --candidate v10 --scene-bootstrap --strict --out docs/ai/v10_revalidated_scene_20261013.json
```

Both verdict commands exit **1 as expected**. Re-scoring took 0.0485 s frame /
0.0156 s scene on this machine (single warm measurements, not inference
performance). Clip scene counts: 5/6/7 at +0/+1/+2 EV; crushed: 7 each, all
from one title. These intervals do not establish cross-title generalization.

Latest regression run adds `test_candidate_duplicates.py`,
`test_candidate_quality.py`, `test_commercial_candidate.py`,
`test_temporal_corpus_ev_2026_09_22.py` and `test_render_exposure_2026_09_05.py`
to the test command above: **107 passed**. `git diff --check` passed.

## Blockers and next action

1. v10 fails all three clipped-highlight gates. All 728 real validation rows
   were previously trained on by v8/v9; freeze new splits before scratch
   training. Remaining real/aces/oog/mix full-frame v10 metrics are unrun,
   deliberately not spent on after the predefined clip-first failure.
2. v10 -> v9 -> v8 ancestry includes HdM: 10,275/25,373 rows overall (40.50%),
   9,966/23,473 v10 training rows (42.46%). No commercial weights are cleared.
   Unknown/academic material remains logically excluded, never moved/deleted.
3. User accepted the documented rights/attribution interpretation for the
   internal pilot; a restricted receipt and separate derived manifest were
   created. The pilot failed its smoke gate; matched-frame diagnosis is now
   complete and routing/evaluation fixes are implemented. The approved paired
   rerun completed but neither checkpoint qualifies for promotion. Preservation
   review isolated a lost crushed-hold gradient and broader dark-tone damage;
   the optional pre-clamp fix and its frozen single-factor comparison then ran
   to completion within its original bounds (both arms 200 steps, verified):
   it improved pushed crushed gains but worsened real PU21, so the recipe still
   fails and nothing was promoted. Release rights remain separate;
   whole-library/crop/HDR duplicate coverage is incomplete. No third-party grant
   or weight-distribution approval was invented.
4. Phase 4 scene-aware prototypes and phases 5–6 training/native integration
   remain gated. Preserve analytic default and existing clip-fill controls.
   Full media hashing entails ~2.77 TB of reads; training/bench costs and
   storage estimates are in the audit. Only the approved smoke and routing pair
   have run; subsequent preservation diagnosis did not train weights.
5. Licensed competitors, calibrated-display blind review, macOS, signing,
   clean-PC checks and dependency/codec rights remain release requirements.

Rollback: changes are uncommitted and confined to the listed Python safeguards
and documentation. Reverting those focused diffs restores prior research
behavior; no source data, checkpoint or native artifact needs recovery. Do not
bulk-remove `docs/ai`, which also contains pre-existing audit documents.
