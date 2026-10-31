# RUDRA unfinished work

Updated 18 October 2026. This checklist contains only unfinished work, including
the remaining portions of partially implemented tasks. Remove tasks here only
after recording completion evidence in [PRODUCTION_PROGRESS.md](PRODUCTION_PROGRESS.md).
Completed work is preserved in the [increment report](PRODUCTION_CONTINUATION_2026-10-17.md).
Read [CURRENT_HANDOFF.md](CURRENT_HANDOFF.md) before continuing; historical requests
in older reports are not new approvals.

## Scope

The agreed v1 scope is analytic reconstruction with artist controls. A qualifying
commercial neural model is a separate roadmap milestone, not a prerequisite to
an explicitly model-free release. Commercial clearance still applies to that
release. No release readiness or quality superiority is implied by passing tests.

## Engineering sequence

- [ ] **E1 Application presentation integration.** Implemented 17–18 Oct and
  tested on Linux with a fake renderer
  ([HEADROOM_APP_INTEGRATION_2026-10-17.md](HEADROOM_APP_INTEGRATION_2026-10-17.md)).
  The routing covers paused frames, seeks, playback, grade changes, undo/redo,
  project restore, replace, failure and close, behind a capability gate.
  **Remaining:** Windows acceptance with the real decoder and the GPU
  upload/present lifetime (`scripts/validate_headroom_app_2026-10-18.ps1`),
  with its evidence recorded.
- [ ] **E2 Complete the headroom workflow.** Done: headroom scopes and readouts,
  and a transactional HDR10/HLG/ProRes export with provenance (decode receipt,
  then encode, then QC, then publish), which has not run on Windows yet.
  **Remaining:**
  - Windows export evidence.
  - EXR still masters and log (ACEScct/LogC4) exports, which still refuse the
    policy.
  - Dynamic metadata.
  - Keep public selection and default reader guards closed until all of these
    support the policy.
- [ ] **E3 Performance and durability.** Reduce initial loading cost without
  weakening source/timing checks; measure long clips, rapid seeks, simultaneous
  consumers and slow storage. Account for native, FFmpeg and GPU memory separately.
  Include rendered CPU images and GPU display buffers in memory accounting. The
  full CPU UHD preview currently measures ~0.56 fps before display; efficient
  preview remains unresolved, with no real-time or process-wide memory guarantee.
  Done 18 Oct: rendered-memory accounting in the preview driver, and a fused,
  threaded view conversion (~30% less per image on 2 cores). Windows timings
  come from the headroom check.
- [ ] **E4 Source controls.** Done 18 Oct: per-frame super-white detail is
  told apart from code-ceiling clipping and from values at white, and shown in
  the headroom status line. **Remaining:** validated input overrides.
- [ ] **E5 Delivery usability.** Add useful still/sequence delivery options and
  short trial exports; make QC results readable and export failures recoverable.
  Extend the existing export sheet rather than duplicate it.
- [ ] **E6 Repeatable application validation.** Extend the existing standalone
  workflow checks to larger mixed-media and production jobs. Run applicable
  application/native/Python regressions and GPU/backend parity/performance checks
  against the final candidate. Validate neural runtime acceptance when a qualifying
  model is available.

## Release gates

- [ ] **R1 Calibrated HDR acceptance.** Record display/preset/calibration, player
  HDR presentation and owner assessment of black levels, shadows, highlights,
  colour and motion, including native app and exported-file comparisons. Requires
  owner review and calibration evidence.
- [ ] **R2 Comparative evaluation.** Complete licensed commercial-tool comparisons
  on identical inputs with settings/provenance, colour error, temporal stability,
  latency/VRAM and blind HDR review. Publish results regardless of outcome. Tool
  access and human review are external dependencies.
- [ ] **R3 Commercial clearance.** Resolve source licence conflicts, attribution
  for footage-derived shadow values, checkpoint ancestry, code/dependency terms,
  HEVC and HDR format naming/metadata obligations. See LEGAL_QUESTIONS.md. Internal
  pilot approval is not distribution clearance; no AI-generated legal sign-off.
- [ ] **R4 Distribution refresh.** Rebuild isolated model-free packages from the
  accepted revision, verify contents and restricted-PATH workflows, and configure
  signing. Preserve existing artifacts. Certificate access is not configured here.
- [ ] **R5 Clean Windows acceptance.** Install, launch, load/grade/save/export,
  inspect outputs, test coexistence and uninstall on a clean VM/PC. Existing local
  tests/portable kit do not replace this; a clean environment is still needed.
- [ ] **R6 macOS acceptance.** Build/package on Apple silicon; validate HDR/EDR,
  backends, available inference, delivery and signing/notarization. Requires Mac
  hardware and credentials, or an explicit change to supported release scope.
- [ ] **R7 Release candidate.** Reconcile version/notes/evidence, run the candidate
  for normal production workflows, fix discovered regressions, then seek approval
  for irreversible publication. Do not treat this checklist as permission to publish.

## Full AI roadmap (separate from analytic v1)

- [ ] **A1 Dataset completion.** Extend duplicate/HDR-quality/licensing coverage
  beyond the scoped audit; prepare additional rights-cleared real SDR/HDR titles
  and owned footage with independent evaluation splits. Keep G: sources read-only.
- [ ] **A2 Quality diagnosis.** Use frozen checkpoints to resolve dark-frame and
  highlight regressions before changing the training recipe. Build on the existing
  v10 and push-baseline A/B evidence; do not repeat resolved hypotheses or present
  older approval requests as pending.
- [ ] **A3 Scene-aware prototypes.** Test deterministic sky/fire/specular/practical
  light confidence masks first, then compact learned gating only if justified.
  Preserve pixels outside selected regions, artist control, skin and black levels;
  measure flicker and uncertainty. Separate faithful and optional creative modes.
- [ ] **A4 Clean model training.** Train only from cleared lineage with a fresh
  compute/storage/benefit estimate and approval. Capacity ablations follow a sound
  quality hypothesis; no production candidate currently passes promotion gates.
- [ ] **A5 Neural production integration.** Promote only statistically supported
  improvements over the analytic baseline without clean-scene/temporal regressions;
  optimize inference, validate native parity/fallback and publish model card,
  licensing manifest, measured performance and limitations.

## Execution boundaries

Safe, reversible engineering and bounded verification are authorized. Source
datasets must not be edited. New expensive training/large downloads need the
estimated cost and approval requested by the master roadmap. No destructive
operations, signing credentials invention, legal clearance claims, external
messages or public release are authorized by this checklist.
