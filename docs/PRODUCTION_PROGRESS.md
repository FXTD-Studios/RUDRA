# RUDRA production progress

## 19 October 2026 — parallel production preparation while native1080 trains

- Live run state read from status/log files (not PowerShell): step 5,600,
  5.09 h, no errors. Saved steps 1000-5000 fail both-positive PU screen;
  best.pt (~800) has negative PU. No process or run file touched.
- Actual review of 25 validation scenes and all 251 decoded targets: real
  footage, pipeline SDR, max 1,903 nits, <0.01% pixels above 1000 nits; no
  fire/sparks/sun/water/neon. Recorded in G: category_review/.
- SI-HDR and Tears of Steel preparation verified (geometry, alignment, colour
  interpretation, scale anchors, plateau masks, title overlap, exclusions);
  conversion and scoring not done. Temporal test plan written.
- New `training/temporal_stability.py` and `training/production_candidate_compare.py`;
  20 new tests pass in the agent VM. Existing torch-dependent tests not re-run.
- Commercial provenance/lineage review saved; open owner rights questions
  listed. [Report](ai/PRODUCTION_PARALLEL_PREP_2026-10-19.md).

## 19 October 2026 — prepared native-resolution validation workflow

- Frozen validation kit: 251 pairs, 25 scenes, all 502 hashes checked; no scene
  ID overlap with other splits. Test set not used for checkpoint selection.
- New guarded exporter/scorer requires exact frame coverage, finite metrics,
  pinned inputs/code, unchanged checkpoint and fresh outputs. Native tiled
  fp32 inference; correct bt1886 picker baseline and 203-nit conversion.
- Equal-scene PU confidence interval gate prepared; it is not a temporal,
  CVVDP, independent real-SDR or product release pass. 19 focused tests pass;
  real full pipeline execution deferred until active trainer exits.
- Indexed 31 source evidence files, registered outstanding independent source
  checks and created category review queue. No new commercial clearance,
  no source admission, no training changes or GPU interference.

## 19 October 2026 — production checkpoint selection diagnosis

- User requests production quality and commercial use. Preserved owner-started
  native1080 run (worker present, initial validation at inspection).
- Audited all 15 saved previous-highlight weights against completed validation
  records. best.pt is step 600, hard PU gain -0.153 dB. Steps 7000/11000/14000/
  13000 improve both PU point estimates; step 7000 leads the min-gain ranking.
- Added CPU-only hash-bound read-only audit and PowerShell wrapper; eight
  focused tests pass. No checkpoint rewritten or automatically promoted.
- Added production sequence/criteria to next-run guide: scene-level validation,
  independent full-resolution/motion tests, targeted training, resumable future
  trainer, commercial provenance and delivery verification. These remain work,
  not claimed passes; active training code and corpus remain unchanged.

## 19 October 2026 — next native-1080p candidate preparation

- Owner requested preparation and next-run script. Completed separate v3 corpus:
  unchanged 3,079 base pairs plus 136 panorama and 16 native CG pairs, 3,231
  total. All additions train-only; source exclusions retained.
- Native panorama conversion and eight CPU Cycles renders completed. Existing
  trainer/corpus/code unchanged; no next training started.
- Added separate preparation, readiness and launcher tools, plus
  [next-run instructions](ai/NEXT_NATIVE1080_TRAINING_2026-10-19.md). Launcher
  checks readiness and other trainers; scratch-only, 24-hour default cap.
- Ten focused tests, Python compilation and PowerShell parsing pass.
- Full numeric verification passed: all 6,462 media files, unchanged base rows,
  zero whole-frame cross-split flags, all 152 new pairs through the real loader
  with correct HDR unit bounds. Largest HDR storage round-trip error 0.000211
  stops. All seven batch sheets and eleven paired/native-crop diagnostics
  visually inspected; limitations recorded. Readiness finalized and real
  launcher Check passes; no training launched.

## 19 October 2026 — expanded commercial source acquisition

- Owner requested research and downloads for the next merged run. Saved
  [collection report](ai/NEXT_HDR_COLLECTION_2026-10-19.md), rights/attribution
  record and G: NEXT_MERGE_PLAN.json; current trainer/corpus untouched.
- Downloaded and fully decoded 23 8K Poly Haven panoramas (5.302 GB), publisher
  MD5 verified. Inspected both complete panorama contact sheets. Historical
  grouping excludes six from next training; nine older train captures and
  eight new screened captures remain candidates, pending pixel matching/pairs.
- Downloaded SI-HDR reference.zip (1,491,596,813 bytes), verified publisher MD5,
  decoded all 181 EXRs and inspected seven exposure-normalized contact sheets.
  Actual geometry 1888x1280, not the nominal 1920 width. Three larger half-float
  ceiling plateaus flagged; numeric magnitude is not accepted as absolute nits.
  All 362 SDR inputs downloaded/decoded at 1888x1280; input publisher MD5 also
  passed. Eleven representative paired comparisons inspected, including all
  three ceiling flags; no target colour/exposure calibration inferred.
- Two Blender CC0 source assets downloaded; safe archive extraction and Blender
  5.0.1 inspection with auto-execution disabled found no missing dependencies.
  No rendering. Tears of Steel: 144 consecutive frames across three sections
  plus 22 sparse stills, all finite/decoded/hash-verified; selection overviews
  inspected. Mixed CG/live action, CC BY 3.0, not camera RAW/HDR master.
- Strict QC rejected three infinite-value frames in initial ToS 00_1a section;
  a fresh download reproduced publisher bytes for frame 66. Original section
  retained/excluded, replaced by fully checked 03_2a. Sparse 04_4k records are
  stills, not a continuous sequence. Initial controller failure retained as
  audit history; final source review and summary completed after replacement.
- Status/Resume wrapper added with duplicate-writer/controller guards. No
  automatic training/merge; future colour, split, full-resolution/motion review
  and pair conversion remain. Source-only collection has not closed all
  real-camera gaps. Active highlight trainer unchanged, last observed step 580.

## 19 October 2026 — missing highlight readiness review completed

- Diagnosed Check rejection: numeric preparation had completed; actual visual
  review and ready.json were still absent. No readiness guard was weakened.
- Inspected both full contact sheets and representative SDR, decoded-HDR and
  original-EXR diagnostic views for all 13 setups. Checked 39 first/middle/last
  EXRs and targets: finite values, matching source hashes, sampled luminance
  peaks 460–994 nits, at most 0.0078125% of pixels above 990 nits.
- [Review notes](ai/HIGHLIGHT_VISUAL_REVIEW_2026-10-19.md) accept a controlled
  pilot with synthetic-resolution, radiometric, motion and display-review
  limits. Evidence at reports/highlight_visual_review_20261019/ and corpus
  visual_diagnostics/ on G:. No human/calibrated-display approval implied.
- Existing finalizer successfully reverified all 6,158 media files, unchanged
  prior rows and pinned dependencies; issued ready.json for 3,079 pairs.
  Actual PowerShell Check exits 0. No dataset, renderer or trainer code change.
- Old Z trainer exited at eight hours, status step 12,537, last numbered saved
  checkpoint 12,000. Last logged validation (12,400) has clean/hard PU gains
  +0.825/+0.021 dB on 251 rendered pairs, zero independent real graded pairs;
  no final model-quality decision. No new trainer, promotion or release.

## 18 October 2026 — continuation handoff and next-training scripts

- Saved [Claude Opus 5.5 next-run handoff](ai/CLAUDE_OPUS_5_5_NEXT_RUN_HANDOFF_2026-10-18.md)
  and a copy under G:/rudra_training. Includes live-status commands, source
  scope, remaining preparation/visual review, baseline assessment, matched-run
  budget caveats, next launcher commands and later evaluation/integration work.
- Added isolated `fire_reflection_readiness.py`, `run_highlight_commercial.py`
  and `train_highlight_commercial.ps1`. Readiness binds preview review and
  artifacts, verifies prior rows/counts and runs full media preflight. Run
  snapshots resolve relative source-evidence paths correctly. Existing trainer
  recipe is reused unchanged, scratch only, eight-hour default cap.
- Eight focused guard/snapshot tests pass; PowerShell parsed. Read-only Status
  works; real candidate Check fails closed as expected because v2 is unfinished.
  No ready.json or visual acceptance fabricated; no next run directory/training.
- Snapshot: current Z step 6,520, v2 127/156 renders, controller finishing fire
  rendering. Step-6,400 validation has positive log gains but slightly negative
  PU gains; no overall quality-success claim. All 251 val pairs are rendered.

## 18 October 2026 — v2 fire/reflection preparation started

- [Report](ai/FIRE_REFLECTION_EXPANSION_2026-10-18.md). Owner authorized next-run
  preparation. Six ambientCG metal packs and three larger JangaFX fire/explosion
  packs selected under their published CC0 grants. Receipts/extracted licences
  retained on G:. Downloads still in progress at this entry.
- Isolated collector, CPU renderer, pair preparation/review tool and resumable
  PowerShell wrapper added. First metal render visually/numerically checked;
  retained float EXR contains bright highlight values. Bulk rendering underway.
- Target 312 pairs from 13 half-second authored snippets; prior 2,767 rows and
  held-outs unchanged. No readiness, measured quality gain or release claimed.
  Combined verification and visual review pending. Current trainer unchanged.
- A hidden one-shot continuation controller (PID 62416) now waits for existing
  collection/material renders, then performs the remaining inspection, fire
  rendering, pair conversion and numeric verification. Status/logs on G:.
  Initial controller array parsing repaired; replacement observed healthy.
  First ground-fire sample rendered in 253 s; a fixed -2 EV fire conversion
  was chosen after preview/HDR-range review. Four real-loader samples pass.
  These sampled checks do not establish readiness of the full candidate.
- Source collection subsequently completed: 7,574,031,336 archive bytes,
  581 VDB frames, 42 fully decoded/rehashed material/preview PNGs. All selected
  36 volume headers pass; fixed bounds frozen in final plan. First 68 completed
  render records match final jobs. Controller now waits only for material renders.
  Full-candidate verification/visual review still pending; no training launch.

## 18 October 2026 — supplied F: compositing/VFX library intake

- [Intake report](ai/F_VFX_LIBRARY_INTAKE_2026-10-18.md): 8,245 files / 62.12 GB
  across two deduplicated roots. Nested Footage/EFFECTS paths counted once.
- Header audit: 1,051 non-ActionVFX videos, one unreadable; successful formats
  are 8-bit with no PQ/HLG tags. 3,818 PNGs: 3,808 8-bit, ten 16-bit. No verified
  HDR ground truth or camera RAW/EXR/HDR files found in these roots.
- ActionVFX excluded pending applicable written AI permission under its explicit
  training/validation restriction; inventory only. Local Action Essentials PDF
  read and visually checked; commercial productions grant does not establish
  this training use. Pexels-labelled files and unidentified packs remain held.
- Inventory, per-file decisions, metadata, licence receipts and report saved to
  `G:/datasets/research/f_vfx_intake_20261018/`. Zero new training pairs. Owner
  asked to identify authored sources and provide special grants, if any.
  Existing candidate, active training, checkpoints and environment unchanged.

## 18 October 2026 — separate next-run highlight pilot prepared

- [Expansion report](ai/HIGHLIGHT_EXPANSION_2026-10-18.md): 328 new matched
  HDR/SDR pairs on G:, 2,767 combined candidate rows (2,256 train / 251 val /
  260 test). Existing held-out rows unchanged; all additions train-only.
- 22 approved-for-this-split HDR panoramas and eight authored setups cover
  skies, reflections, sparks, two JangaFX fire volumes, neon and bright windows.
  CC0 sources and authored geometry; Charge legacy Eevee assets remain unused.
  Twelve HDR captures excluded conservatively for historical split overlap.
- CPU-only Cycles renders (two threads, 640x360, 32 samples) plus ACES 1.3
  1000-nit pair conversion; clean and SDR-only clipped/JPEG variants share HDR.
  Source/plan hashes and licence evidence bound to internal-pilot review.
- All 5,534 paired media files rehashed; new pairs fully decoded, zero exact
  or screened perceptual cross-split flags. 164 variant groups, 52 short
  sequences and 28 actual-loader category/variant checks passed. Preview
  inspection accepts only a bounded pilot: some volume sampling noise,
  synthetic appearance and limited temporal/resolution/radiometric coverage.
- Source preparation scripts and status/resume wrapper added. No training,
  active-code/data change, package installation or model release. Current Z
  run observed at step 4,400, PID 58624. Real highlights/video evaluation,
  higher-quality renders and commercial release review remain outstanding.

## 18 October 2026 — free Blender open-movie effects and JangaFX volumes

- [Source audit](ai/BLENDER_HDR_GAP_SEARCH_2026-10-18.md): owner confirmed no
  subscription, prioritize free downloads. Saved five source packs on G:,
  323,298,196 bytes: Charge smoke/fire, muzzle flash with sparks, and industrial
  material library (CC BY 4.0); two JangaFX fire simulations (CC0), 301 VDB frames.
- Download receipts, local SHA-256, licence evidence and attribution saved.
  ZIP CRCs pass; RAR extraction succeeds. All three Blender projects open with
  no missing external images; first/middle/last volume headers load in each set.
  Older Eevee render compatibility and disabled camera drivers remain unverified.
- Synthetic source coverage improved; no measured model improvement, rendered
  training pairs or commercial model release clearance. Real fire/spark camera
  footage and independent evaluation still needed. Cosmos HDR held for archive
  CC BY-SA versus older CC BY grant reconciliation; CGHEVEN AI prohibition found.
- Added isolated source acquisition and inspection scripts, no environment
  changes or GPU rendering. Current trainer untouched; observed Z step 2,100,
  PID 58624, errors log empty. No new training, restart, promotion or deployment.

## 18 October 2026 — commercial highlight sources downloaded and verified

- [Collection report](ai/COMMERCIAL_HIGHLIGHT_COLLECTION_2026-10-18.md): separate
  Poly Haven CC0 pack on G:, 34 8K HDR panoramas plus 12 metal materials / 52 maps;
  86 media files, 7.265 GB. All publisher sizes/MD5 and full image decodes pass;
  local SHA-256 manifest and visual-review contact sheet saved.
- Saved current commercial licence, explicit AI permission, API terms, source
  metadata and authors. Restricted/uncertain vendors excluded; no paid purchases.
  RAW fire/spark or moving reflective-metal video was not obtained.
- 33 HDR capture groups, 28 selected assets overlap the existing auxiliary corpus.
  Old autumn_field variants span train/val; flagged and blocked for future split
  reconciliation. Current Z training excludes that auxiliary corpus.
- Added isolated download/review scripts; no new paired training samples, trainer
  changes, environment installation, GPU job, deployment or release. Active Z run
  observed at step 1,700, PID 58624, errors log empty. Re-read live status.

## 18 October 2026 — expanded corpus verified, training active, Claude handoff

- Verified completed corpus: 1,230 new pairs plus 1,209 old = 2,439 total;
  1,928 train / 251 validation / 260 test. All pair paths are on G:. Originals
  remain on Z: and were not fully copied; this run does not require Z: online.
- Combined pair screening found no exact or perceptual cross-split flags;
  commercial preflight passed and the trainer reverified all 4,878 media files.
  Receipt remains internal-pilot, release_authorized false.
- Active `fxtd_z_hdr_run01` observed at step 20, PID 58624, errors log empty;
  step-200 gate pending. Scratch initialization, original eight-hour cap retained.
- Previous run02 stopped at 28,800 seconds / status step 15,000. Actual checkpoint
  inventory ends at step_0014000.pt; best.pt metadata is step 400. New best.pt is
  step 0 at inspection. No checkpoint quality or model promotion inferred.
- Created [Claude Opus 5.5 process handoff](ai/CLAUDE_OPUS_5_5_HDR_TRAINING_HANDOFF.md)
  at the owner's request and updated the current handoff pointer. Read-only live
  verification plus documentation only; no training/data/code mutations.

## 18 October 2026 — Z: preparation workflow launched after source confirmation

- Owner requested “all next” and confirmed LogC4 for seven previously ambiguous
  TAHA files and Mataha A_0020C010. Updated exact-answer receipts and classifications:
  172 candidates, 23 exclusions. No unresolved source curves in this inventory.
- Completed 516 sampled frame decodes across 172 candidates, including repair
  of reserved metadata handling in 56 files. Seven same-split similarity flags;
  zero cross-split flags. Contact sheets reviewed; Fouad includes substantial
  greenscreen material. This does not prove whole-clip or calibrated HDR quality.
- Froze 123 selected take/version groups, 952.616 GB, targeting up to 1,230 new
  pairs; 40 alternatives omitted and nine PQ sources kept separate. All Mataha
  remains test-only; FASHONISTA/Sudan are validation productions.
- Added isolated CPU preparation, pair verification, preflight receipt and
  scratch-run adapters. **17 focused tests pass**; launcher syntax checks pass.
  Active trainer/decoder code remains unchanged. New ACES 1.3 rendering is pinned
  to installed OCIO; no training-environment package changes.
- Launched `prepare_z_commercial.ps1 -Action Start -TrainWhenReady`, hidden
  launcher PID 45880. Hashing/rendering/verification proceeds independently, then
  starts `fxtd_z_hdr_run01` only after successful preflight and existing HDR trainer
  exit. Same 20,000-step/eight-hour limits; no model promotion or release.
- First ten new pairs rendered and independently reopened/hashed/checked:
  matching uint16 geometry, finite HDR, sampled peaks 186–246 nits. Full selected
  source hashing and rendering continue; this is not final corpus acceptance.
- [Workflow report and status commands](ai/Z_FOOTAGE_PREPARATION_2026-10-18.md).
  Final corpus counts and preflight/training outcome remain pending; use live
  artifacts rather than treating the target pair count as completed output.

## 18 October 2026 — additional Z: footage intake and colour confirmation

- Inventoried all ten supplied roots: 260 files, 1.107 TB logical size; probed
  all 195 video headers successfully. Source media remained read-only.
- Recorded the owner's exact "alexa logc 4" reply for Sudan, FASHONISTA and
  FOUAD HAMZA (84 files). Kept AMIN camera LogC3 EI800 and tagged PQ separate.
- Created review-only per-file intake records: 164 identified-source candidates,
  eight ambiguous, 22 reference/proxy exclusions and one Rec.709 target exclusion.
  All records have training disabled. No new pairs or trainer code changes.
- 37 identified Mataha candidates remain test-only; filename matching flags
  20 files sharing prior take IDs, pending content comparison. Production groups
  prevent splitting FASHONISTA/Fouad folder variants as independent titles.
- [Report](ai/Z_FOOTAGE_INTAKE_2026-10-18.md) and reproducible artifacts in
  `reports/z_footage_intake_20261018/`. Pixel QC, source hashes, deduplication,
  source gamut/range/exposure validation and paired rendering remain pending.
- Corrected active run pointer: owner stopped run1 and launched run02. Latest
  observed status was training at 12,000/20,000 steps, 24,203 seconds elapsed.
  Eight-hour cap retained; no completion, release or model quality claim.

## 18 October 2026 — FXTD scratch training launcher

- Owner explicitly requested preparing and starting training plus runnable
  scripts. Added `scripts/train_fxtd_commercial.ps1` (Start/Foreground/Prepare/
  Status/Stop) and `training/run_fxtd_commercial.py`. Background job launched at
  `G:/rudra_training/fxtd_hdr_20261018_v1/`; consult its status/logs for current state.
- First job uses only 1,209 FXTD pipeline pairs; 1,098/51/60 production-separated
  train/validation/test. No inherited weights or unresolved-source additions.
- Added explicit pipeline-only baseline convention with invalid/mixed-target
  rejection. Metadata stays rendered/null exposure. Legacy non-pipeline behavior
  unchanged. **50 focused tests pass**, one existing warning; launcher syntax and
  runner compilation pass. Run-local code snapshots and hashes saved.
- Preparation performs full pair-file hashing/decoding and exact/SDR-thumbnail
  split screening. Existing commercial preflight independently hashes media and
  decoder dependencies. No broad data-licence flag bypass or release receipt.
- Recipe: 20k steps maximum, 8-hour training limit, scratch 32-channel model,
  batch 4/crop 256, BT.1886 baseline, zero-EV explicit model convention,
  mild degradation, pre-clamp dark preservation, no exposure-push rerendering.
  All 51 validation rows evaluated every 200 steps; smoke at 200 stops if either
  clean/hard gain falls more than 0.5 dB from initialization. No automatic release.
- [Run guide](ai/FXTD_TRAINING_RUN_2026-10-18.md). Completion, actual throughput
  and quality are not claimed by this launch entry; use live run artifacts.
- Preparation completed: 1,209 pairs / 2,418 files / 16.29 GB fully hashed and
  decoded; zero exact or SDR-thumbnail cross-split flags, no low-information
  exclusions. Independent trainer preflight verified all 2,418 files and entered
  initial validation. This is integrity/screening evidence, not a quality pass.
- GPU training confirmed through step 20 (loss 0.10704), no logged errors.
  Process continues independently; step-200 smoke and final quality pending.

## 18 October 2026 — supplied-dataset commercial selection

- Owner requested checking all supplied datasets and using commercially usable
  sources for the new model, broadening the earlier FXTD-only preference.
- Inventoried all six supplied F: roots and 16 G: source/legacy folders;
  inspected 20 corpus manifests. No inventory/parse errors. Found duplicate
  October 8/8b mixed manifests; did not count them as new independent footage.
- Prepared isolated `rudra_commercial_review_20261018_v1` on G:. Primary:
  1,209 FXTD pipeline pairs. Auxiliary: 4,098 FXTD camera/Poly Haven rows,
  separate from mastered-HDR targets and disabled in the primary plan.
- Netflix: 5,944 pending real pairs and a 7,153-row combined pending candidate.
  Fresh downloads confirm both older NC-ND README and newer CC BY licence remain
  available. No new release clearance inferred from owner task approval.
  Restricted and unresolved datasets remain excluded from active selection.
- All referenced selected/pending media exists; no scene/path cross-split reuse
  within prepared manifests. Four outputs reopened and hashes/checks repeated;
  deliberate leakage and missing-path cases rejected. 42 sampled pairs pass
  decode, shape, uint16-HDR and finite-value checks with zero flags.
- Full media hashing, perceptual duplicates and calibrated quality are not
  certified. Pipeline null exposure can trigger legacy fallback; resolve explicit
  baseline/target convention before training. Small dark test-set limits persist.
- [Report](ai/COMMERCIAL_DATA_SELECTION_2026-10-18.md), source/manifest audit,
  model-data policy and licence retrieval receipts saved. Planning artifacts do
  not bypass existing commercial preflight. No source edits, training, old-weight
  reuse, commit, push or release.

## 18 October 2026 — Windows headroom acceptance, first evidence

- Windows MSVC build of every new source passes (`build\native_phase3`, Qt 6.8.3,
  viewer on). Headroom native tests: **71/71 pass on Windows**. That is 48
  HeadroomFrames, 4 HeadroomSession, 14 driver and 5 export, including the
  real export publishing with provenance and cancellation leaving nothing.
- `RUDRA --headroom-check` on a 1080p, 48-frame clip that fits the v1 contract
  (limited range Rec.709, 10-bit 4:4:4 FFV1): **8 of 9 steps pass** with the
  real decoder.

  | Step | Result |
  |---|---|
  | First paused frame | 2.76 s |
  | Seeks (1, 47, 0) | 0.27 / 0.51 / 0.28 s; frame 0 bit-identical on return |
  | Grade change | 0.28 s; undo restores the pixels bit-identically |
  | Playback to the end | 47 frames in order, ~9.6 fps at 1080p (not real time) |
  | Transactional HDR10 export | published in 38.8 s; sidecar 48 frames, MaxCLL 385, policy recorded |
  | Return to the legacy engine | passes |

- **Viewer step failed.** The viewer presented 0 frames in the whole run, the
  legacy picture included, and the readback never returned. The window looks
  never to have been exposed, for example a locked screen or a hidden window.
  The check now records exposure and states that reason; the run needs
  repeating with the window visible.
- An earlier run correctly refused a clip with no transfer tag ("Missing/unsupported
  colour transfer") and showed nothing. The script now makes clips with the tags
  set on the frames themselves.
- **Still open:**
  - the viewer evidence;
  - why the app suite stalled on `AppModels.SwitchingModelsMidSessionKeepsTheSession`.
    That test runs with the headroom code off.
- Reports: `reports\headroom_app_20261018\` (`headroom_check_083651.json`,
  `headroom_native.log`).

## 18 October 2026 — headroom preview, scopes, export and source whites in the app (E1/E2, part E3/E4)

- Added `HeadroomPreviewDriver` (`engine/headroom_preview`). It runs one render
  worker and hands opaque tokens to the UI thread, where the bridge revalidates
  them before display. It has back-pressure, failure reporting, and no fallback
  to legacy pixels.
- MainWindow routes the following through it, behind a capability gate that
  stays closed by default: paused frames, seeks, playback, grade changes,
  undo/redo, project restore, model and hold changes, replace and close.
- Added headroom scopes and readouts (`measure_headroom`) and the
  transactional `export_headroom_movie`. The export runs decode receipt, then
  encode, then QC, then publish, with a provenance sidecar. The app's movie
  export uses it when the policy is on.
- Added per-frame super-white detail versus ceiling clipping
  (`super_white_stats`), rendered-memory accounting, and a fused, threaded
  view conversion.
- Fixed a defect: a headroom project could not be saved from the window,
  because the stored grain default failed validation.
- Added the developer check `RUDRA --headroom-check` and the script
  `scripts/validate_headroom_app_2026-10-18.ps1`.
- **Evidence, all from a Linux workspace** (Qt 6.4, no viewer, Windows-only
  decoder skipped):

  | Check | Result |
  |---|---|
  | Native tests | 330 pass, 48 skip (Windows-only) |
  | Driver tests, shuffled | 60 repeats pass |
  | Driver tests, ThreadSanitizer | 5 shuffled runs clean |
  | App tests | 71 of 73 pass |
  | `AppHeadroom` tests, shuffled | 10 repeats pass |

  The 2 app failures also fail on the untouched baseline in that environment.
  The real-port check refuses correctly at source leases, and nothing is
  published. Details: [HEADROOM_APP_INTEGRATION_2026-10-17.md](HEADROOM_APP_INTEGRATION_2026-10-17.md).
- **Not yet:** Windows acceptance (decoder, viewer, export, timings). No ledger
  task is closed until that evidence is recorded here. No training, downloads,
  dataset writes, commits, pushes or releases.

## 17 October 2026 — unfinished-only checklist

- Updated [remaining work](REMAINING_WORK.md) to keep only open tasks and the
  unfinished portions of E1/E4/E6; completed evidence remains in the entries below
  and the [increment report](PRODUCTION_CONTINUATION_2026-10-17.md).
- Retained all 18 open task IDs and external release gates. Updated the preview
  performance blocker to the measured full CPU UHD workload (~0.56 fps).
- Documentation only; no application changes or new runtime results. Next:
  guarded MainWindow presentation integration and efficient preview.

## 17 October 2026 — completion ledger, session preview bridge and app checks

- Wrote [remaining work](REMAINING_WORK.md) before implementation, as requested.
  Full roadmap remains open; completed subtasks and external gates are explicit.
- Added policy identity to session undo/restore and project grade, preserving
  legacy JSON/default rejection. Added isolated session-to-preview rendering with
  stale/foreign/out-of-order/late-failure rejection and serialized presentation.
  MainWindow headroom routing, scopes and transactional exports remain pending.
- Source card now displays actual still profiles/orientation/fallback. Added
  analytic-only standalone workflow checking, bounded dialog/wait failures and
  delivered-frame identity checks. Stage configured Windows ONNX DLLs so tests/CLI
  no longer find the older System32 runtime. Fixed the layer checker's bench error.
- **358/358 native, 65/65 app, 10/10 Python and 40/40 shuffled presentation checks
  pass.** Rebuilt Windows app standalone analytic workflow passes (three EXRs and
  six-frame HDR10 export); initial test compile errors fixed and logs retained.
- **8 HD/UHD preview benchmark runs** match direct CPU master/baseline pixels.
  UHD full CPU feed/render/hash: **14.18 s / 8 frames (~0.56 fps)** before display;
  this is a newly measured performance blocker, not the old decode-only benchmark.
  No real-time, calibrated visual or commercial release acceptance claimed.
- [Report](PRODUCTION_CONTINUATION_2026-10-17.md), evidence and handoff saved.
  Next: guarded MainWindow integration and preview optimization, then scopes/
  export. Packages/GPU performance not refreshed; external release gates persist.
  No training, downloads, dataset changes, model promotion, commit or release.

## 17 October 2026 — asynchronous playback and shared frame memory

- Added background open/seek/cancel with latest-generation publication, explicit
  old-decoder reaping and owner/generation-specific previews/receipts. One pixel
  allowance includes staging and retained/promoted frames across retired owners;
  decoder workers wait for space. Late failures cannot promote provisional data.
- **350/350 native tests**, **60/60 shuffled concurrency executions** and **15/15
  benchmark cases pass**. All 360 frame hashes/content identities match the prior
  path; fixtures unchanged. Build, whitespace and source/binary checks pass.
- API submission: 0.0027–0.0104 ms. Short UHD feed improves 24.0%; HD is effectively
  unchanged. Native pixel peak is 189.84 MiB within the 256 MiB allowance. UHD is
  still 10.52 fps, and slow-consumer total memory ~1.56 GiB. Small synthetic samples
  do not close real-time, long-footage or whole-process memory gates.
- [Report](HEADROOM_ASYNC_PLAYBACK_2026-10-17.md) and handoff saved. Next: session/
  source-policy and preview/paused-frame wiring, undo/restore and app lifecycle
  tests. Headroom UI remains disabled until all consumers/export support it.
- Pre-existing ORT warning/release blockers persist. App/Python/GPU not rerun;
  no app/package refresh, training, datasets, downloads, model promotion or release.

## 17 October 2026 — reusable, cancellable protected source preparation

- Full probe/hash preparation is shared by stream/batch owners; source/decoder
  leases last through the final owner. Stop tokens cancel probe pipe/exit waits
  and chunked hashing. Identity failures latch across owners and require fresh
  preparation; independent range receipts/cancellation and budgets are preserved.
- **336/336 native and 36/36 focused tests pass.** **12 benchmark cases pass**;
  18 seek frames and three sequential controls match previous frozen pixels.
  Source hashes and content identities remain unchanged. Build/whitespace pass.
- Reused preparation/binding: **0.050 ms HD / 0.138 ms UHD**; later seek frames
  including decode: **0.342 / 1.183 s**. Initial preparation is still slow
  (10.43/92.54 s). HD has two pairs, UHD one; varying load limits generalization.
  Real preparation cancellation: 1.86–4.34 ms. No real-time/global memory claim.
- One initial UHD case timed out at 180 s; retained separately. A bounded 300 s
  retry completed without relaxing validation. One test compile error corrected.
- [Report](HEADROOM_PREPARATION_2026-10-16.md), evidence and handoff saved. Next:
  asynchronous generation changes with a combined retained-frame budget, then
  application/export integration. Headroom UI stays disabled; reader guards stay.
- Existing ORT warning, commercial/model quality, calibrated HDR, signing,
  clean-machine and macOS blockers persist. App/Python/GPU not rerun; no app/package
  refresh, training, source dataset edits, downloads, model promotion or release.

## 16 October 2026 — FFV1 memory reduction and longer sequence validation

- Adopted a measured FFV1-only profile: two decoder workers, one raw-output
  worker and one output-queue packet. Other codecs/colour/delivery stay unchanged;
  command-dependent cache identity updates automatically.
- **328/328 native tests pass**, including real blocked-output cancellation and
  input/output option scoping. **88 benchmark runs pass**: 34 diagnostic sweeps
  and 54 short/240-frame/reference/slow-consumer/batch checks, matching all float
  and legacy control hashes. Build and whitespace checks pass.
- Sampled total memory falls **42.3% HD / 44.9% UHD** on short clips and
  **34.8% / 35.6%** on 240-frame clips. Long UHD still uses ~1.86 GiB and feeds
  at 11.63 fps; buffering grows during the run. No hard cap or real-time claim.
- [Report](HEADROOM_DECODER_MEMORY_2026-10-16.md) and handoff saved. Next: reusable
  protected source and cancellable preparation (about 35 s to open the 10 s UHD
  fixture), then asynchronous generation handling. Headroom UI remains disabled;
  reader guards and export integration requirements remain in place.
- Pre-existing ORT warning/release blockers persist. No app/Python/GPU rerun,
  app/package refresh, source dataset edits, training, download, commit or release.

## 16 October 2026 — continuous headroom decoder design and prototype

- Added isolated single-process range decoding with provisional preview frames,
  owner-specific completion receipts and explicit verified-frame promotion.
  Exact EOF/exit, source identity, cancellation and retained-frame budget remain
  enforced; shared lease helper preserves the existing verified batch path.
- **326/326 native tests pass**, including nine new lifecycle/real-pixel/grade
  tests; **42 paired benchmark runs** retain all float and legacy control hashes.
  Corrected a test compilation error; final build and whitespace checks pass.
- Median synthetic feed improvement over the previous frozen optimized feeder:
  **5.99x HD / 10.55x UHD**, one decoder start. With preparation included:
  **2.46x / 4.26x**. These are decode-only diagnostics, not app performance.
- **UHD remains 11.14 fps and sampled total memory rises 760.6→1234.7 MiB.**
  Owned pixels remain bounded; FFmpeg memory needs profiling before integration.
- [Report and next task](HEADROOM_CONTINUOUS_STREAM_2026-10-16.md): external
  decoder memory/longer sequences, then shared source preparation and asynchronous
  generation handling. Headroom UI stays disabled and reader guards stay in place;
  export transactions,
  session, preview, scopes and provenance still need integration.
- Handoff and frozen evidence saved. Pre-existing ORT warning/release blockers
  persist. App/Python/GPU not rerun; no app/package refresh, training, downloads,
  dataset changes, commit or release.

## 16 October 2026 — measured headroom playback optimization

- Added paired HD/UHD benchmark tooling and froze before/after executables.
  Four-thread float decode, guarded FFV1/Matroska seeking and batch reclamation
  preserve exact frame selection, source identity and completed-batch validation.
- **317/317 native tests pass**. Offset/keyframe/rate tests include 18 fixture
  combinations and video beginning after audio. Initial seek/tagging failures
  were retained and corrected without relaxing source/output requirements.
- **36 paired runs**, three repeats: median synthetic feeder improvement
  **1.69x HD / 1.17x UHD**. All float and unchanged legacy-control frame hashes
  match. Sampled tree-memory medians: 545→417 MiB HD, 1348→763 MiB UHD.
- Still not real time: **5.97 HD / 0.98 UHD fps**. UHD still requires 23 decoder
  starts for 24 frames; source probing/hashing remains expensive. No production
  quality/speed, competitor, app, neural or GPU acceptance claim.
- [Report and next action](HEADROOM_PLAYBACK_PERFORMANCE_2026-10-16.md): stronger
  review of continuous decoding and explicit frame/completion semantics, then a
  bounded prototype. App integration and headroom UI remain gated. Pre-existing
  ORT test-loader warning and existing release blockers recorded; handoff saved.
- No training, datasets, downloads, package changes, commit or release.

## 16 October 2026 — bounded headroom frame cache and playback feeder

- Added isolated asynchronous float source/cache with exact indexed requests,
  Windows read leases, source/decoder identity checks and stale-request rejection.
- Byte budget includes staging, cached frames and retained consumer handles.
  Verified batches publish only after successful EOF/exit; malformed output fails.
  Cancellation now also covers decoder finish/EOF and memory backpressure.
- **314/314 native tests pass**, including 12 new tests. Real indexed playback,
  paused/cache reads and +1 EV HDR masters match sequential decoding exactly.
  Legacy goldens pass; build/whitespace checks pass. App/Python/GPU not rerun.
- [Report and limits](HEADROOM_FRAME_CACHE_2026-10-16.md): next benchmark/optimize
  sustained HD/UHD decoding. Current batches restart decoding from the beginning;
  this is not yet efficient long-video playback. Startup cancellation and whole
  process peak-memory measurements are also pending.
- Session/preview/scopes/export/provenance wiring and UI enablement remain gated.
  Progress/handoff saved; no training, source dataset writes, download or release.
  Existing commercial, signing, visual and clean-machine gates remain open.

## 16 October 2026 — headroom saved-policy compatibility and indexed decoding

- Headroom saves use project version 2; legacy saves remain version 1. Default
  native/app readers and Python legacy render paths reject unsupported policy.
  Missing masks and incompatible settings fail without silently changing grade.
- Shared float decoder now supports exact indexed ranges for paused/playback/
  export callers. Random single-frame and sequential reads agree bit-for-bit on
  a 24000/1001-fps fixture with a 5-second timestamp offset.
- Added dedicated cache identity including full source/decoder content hashes,
  policy, contract and decoding settings. No cache payload store/eviction yet.
- **302/302 native, 60/60 app and 20/20 Python tests pass.** Initial missing-tag
  test fixture corrected; production validation retained. Build/whitespace pass.
- [Report and next criteria](HEADROOM_STORAGE_2026-10-16.md): bounded float cache
  and playback feeder, then actual session/preview/scopes/export integration and
  provenance. Mode remains unavailable in the app; no quality/speed/GPU claim.
- Progress/handoff saved. No training, datasets, downloads or release actions;
  existing commercial, visual and clean-machine gates remain open.

## 16 October 2026 — headroom grade and delivery boundary validated

- Added shared `render_headroom_frame`: exposure, existing Region EV and
  painted masks on preserved linear Rec.2020; explicit source/policy retained.
- Unsupported SDR anchoring/chroma/grain and nonneutral recovery controls fail
  clearly. Invalid settings/masks/overflow rejected; no silent setting changes.
- Added explicit 203-to-10,000-nit unit bridge to existing movie frame delivery.
  All nine profiles exercised; peak/knee leaves the linear master intact.
- **293/293 native tests pass**, including **29/29 focused**. Real FFV1 ramps
  retain 19/79 above-white steps through +1 EV, PQ frame preparation and actual
  linear/AP0 half-EXR roundtrips. Legacy integer decoder control still collapses
  those steps. Builds and whitespace checks pass.
- [Report](HEADROOM_RENDER_2026-10-16.md). App/CLI selection, project/cache,
  preview/scopes and export-job integration remain; no end-to-end movie, GPU,
  temporal-quality, throughput or visual claim. Existing defaults unchanged.
- Next: versioned saved policy/compatibility and shared float source/cache
  wiring before exposing the option. No training, dataset or package changes.

## 16 October 2026 — versioned headroom baseline complete

- Added isolated, default-disabled `bt1886-headroom-v1` CPU baseline. Source
  power curve precedes linear gamut conversion; output is Rec.2020, 203 nits/unit.
- Enforces analytic BT.1886 and supported metadata, rejects clip fill/peaks,
  calibration/reference, malformed/nonfinite inputs and numerical overflow.
  This function is before grading; it does not apply or replace artist controls.
- **283/283 native tests pass** (19 focused decoder/baseline checks). Real FFV1
  fixtures preserve all tested above-white levels after reconstruction. Code
  1019 gives 249.743557 nits; nominal white uses explicit new 203-nit policy.
- Legacy formulas/goldens unchanged. Initial test member-name build typo fixed;
  final build and whitespace checks pass. App/GPU/visual tests not rerun.
- [Report and next action](HEADROOM_BASELINE_2026-10-16.md): consistent
  preview/playback/master/movie integration, cache/project/provenance, then
  grading/GPU/quality/performance gates. No UI enablement or release action.

## 16 October 2026 — isolated floating-point decoder complete

- Added separate FloatVideoDecoder / ExtendedVideoFrame; legacy decoder and
  app defaults unchanged. Direct GBR-plane reads avoid an extra RGB copy.
- Tagged 8/10-bit limited YUV444 only; unsupported sources and alpha refused.
  Finite values, buffer overflow, EOF/truncation/extra output, process failures
  and cancellation tested. No silent success after failed/partial reads.
- **274/274 native tests pass**, including 10 focused tests. Four real FFV1
  fixtures preserve 19/79 headroom levels and match independent color matrices
  within 2e-6; legacy decoder collapses these levels. No quality/speed claim.
- First fixture lacked tags: retained failure, fixed fixture tags, kept input
  validation strict. Release build and whitespace check pass.
- [Evidence and next step](FLOAT_DECODER_2026-10-16.md): opt-in BT.1886 baseline
  next; preview/export/cache integration after that. App/GPU checks not rerun
  for this isolated API. No dataset/training/release/package actions.

## 16 October 2026 — measured super-white preservation review

- Audited paused/movie/master/CPU/GPU/Python paths and saved the
  [review, policy and gates](SUPER_WHITES_REVIEW_2026-10-16.md).
- New reproducible tiny synthetic decoder experiment: **20/20 diagnostic
  checks pass**, 8/10-bit YUV444, BT.709/BT.2020 matrices. Float retains all
  79 tested 10-bit above-white levels; current export retains 1. Matrix error
  <=1.25e-7. This is decoder evidence, not end-to-end quality/performance.
- Rejected old limited-integer proposal and clamped endpoint extrapolation.
  Initial PNG assertion corrected after 10-bit data showed 4 remaining levels;
  initial failure retained, final report matches script hash.
- Next: isolated float decoder API/tests, then opt-in analytic BT.1886 policy.
  Preserve legacy defaults; no unvalidated AI or automatic clip-fill interaction.
- No production code changes, training, datasets, downloads or package refresh.
  Prior 60 app / 264 native pass not rerun or presented as new headroom coverage.

## 16 October 2026 — source inspection, first usability increment

- Source card displays reported movie tags, missing-tag state and limited-range
  / HDR-input warnings. Movie-to-folder switching clears stale warnings.
- **60/60 app and 264/264 native tests pass** with pinned FFmpeg; build and
  whitespace check pass. Updated app built and launched (PID 15128).
- Evidence: `reports/source_inspection_{app,native}_20261016.{json,log}`;
  build logs `reports/source_inspection*_build_20261016.log`.
- [Scope and next review](SOURCE_INSPECTION_2026-10-16.md). No pixel behavior
  changed; super-white preservation, overrides, expanded export and recovery
  features remain. Stronger reasoning recommended for the headroom design.
- Existing export sheet already supplies format guidance; avoid duplicating it.
  Installer/ZIP still stale. No new training or release clearance.

## 16 October 2026 — Getting Started, F-to-fit and export matrix

- Added F-to-fit with typing regression; corrected guide metadata wording.
- Found/fixed actual ProRes ACEScct/LogC4 QC failure by clearing frame tags
  for log output only; grading and QC thresholds unchanged.
- **59 app / 264 native tests pass**, including all eight movie encodings
  and four folder movie formats. Updated app built and launched.
- [Workflow review and improvements](WORKFLOW_REVIEW_2026-10-16.md).
  Prior ZIP/installer stale; manual release gates remain pending.

## 16 October 2026 — portable clean-machine kit ready

- Owner confirmed no clean Windows test environment is currently available.
- Bundled installer/checksum, fixture, standalone test and manual checklist.
  Actual Windows PowerShell 5.1 smoke checks pass on staged local package.
- [Kit and evidence](CLEAN_MACHINE_KIT_2026-10-16.md). No clean-machine or
  install/uninstall pass claimed. Await a separate PC/VM; no host installation.

## 16 October 2026 — model-free evaluation installer staged

- Inno compile passes; unsigned 144 MB installer and SHA-256 saved separately.
- Evaluation identity/install folder/shortcuts separated; `.rudra` association
  registration excluded. App preferences still shared, documented.
- [Installer evidence](MODEL_FREE_INSTALLER_2026-10-16.md) and
  [clean-machine checklist](CLEAN_MACHINE_ACCEPTANCE.md) saved. No install
  performed; Sandbox unavailable here. Next requires a clean Windows VM/PC.
- Signing, owner visual assessment and licensing remain open.

## 16 October 2026 — isolated model-free package created

- Added `-NoModels` and explicit build-directory selection; research beta path
  preserved and analytic outputs isolated by unique staging directory.
- Unsigned evaluation ZIP created without weights/catalogue. Restricted-PATH
  app startup and analytic HDR10/ProRes exports pass; contaminated package
  rejection also passes. Smoke checks integrated before model-free ZIP creation.
- [Artifact and verification](MODEL_FREE_PACKAGE_2026-10-16.md). Next: installer
  staging/review, clean-machine test and signing. Visual review pending by
  owner choice; no commercial clearance or deployment.

## 16 October 2026 — visual review and packaging preflight

- Verified 10-bit PQ/BT.2020 review movie and opened it for owner assessment;
  calibration, black-level judgment and native-app comparison remain pending.
- Inno Setup found; signing certificate not configured in this session.
- Existing packager mandates/copies research-only v8 and replaces versioned
  outputs. Not executed. Next: isolated, explicitly model-free packaging.
- [Details and pending review](RELEASE_REVIEW_2026-10-16.md). No clearance,
  deployment or visual-quality pass claimed.

## 16 October 2026 — four-backend parity and latency passed

- Workload removed: all four backend tools return success against existing
  budgets. 4K composite + display wall medians: D3D12 1.41 ms, D3D11 1.24 ms,
  Vulkan 1.48 ms, OpenGL 1.16 ms. Numerical parity passes throughout.
- Evidence: `reports/gpu_idle_20261016_153330/`. GPU memory ~1.6 GB;
  background utilization 32–33%, documented rather than described as zero.
- Primary HDR output and pinned-FFmpeg suites already pass. Next: calibrated
  display review and remaining release/package/licensing gates. No training,
  publishing or budget changes.

## 16 October 2026 — HDR enabled, output probes pass

- PA279CRV HDR output passes D3D12 scRGB/HDR10 and D3D11 scRGB probes;
  viewer cards pass D3D12/D3D11/Vulkan. LG still reports HDR off.
- Four-backend numerical parity passes again. Latency remains unqualified:
  GPU utilization is still 100% before/after testing, with another compute
  process present. No unrelated process stopped.
- Fresh evidence: `reports/gpu_ready_20261016_153101/`. Next: idle-GPU
  timing rerun; physical display review remains necessary.

## 16 October 2026 — GPU parity and pinned FFmpeg validation

- [Report](GPU_RELEASE_VALIDATION_2026-10-16.md): all four backends pass
  composite/view and real-viewer SDR parity.
- Performance gate not passed: concurrent GPU utilization stays 100%; HDR
  probes report both displays in SDR. Clean rerun pending the user's changes.
- Approved 178 MB FFmpeg 7.1.1 archive downloaded and hash-verified. With the
  pinned dependency, **264 native and 57 app tests pass**.
- Next: clean timing/HDR rerun; physical review, licensing and package checks
  remain. No deployment or training.

## 16 October 2026 — four Windows failures resolved

- [Diagnosis and results](FFMPEG_VALIDATION_FIX_2026-10-16.md): synthetic
  frame metadata now survives encoding; self-test and temporal checks pass.
- Python and native alpha checks agree at 15 codes on the installed FFmpeg;
  cross-build golden assertion corrected, original 128-code bound retained.
- Rebuild passed. Focused **9/9**, full native **264/264**, full app **57/57**.
- Next: GPU parity/latency and packaged-dependency validation. No release
  clearance, training, push or deployment claimed.

## 16 October 2026 — Windows build and export validation

- [Validation report](WINDOWS_VALIDATION_2026-10-16.md): minimal build passes
  249 native and 41 app tests; full OpenCV/ONNX build passes 260/264 native
  tests and all 57 app tests.
- Fixed the new export test's missing image-decoder build guard; added actual
  movie frame-count, frame-rate and HDR color-tag assertions, all passing.
- Four full-native failures remain: alpha golden difference, FFmpeg self-test,
  and two temporal tests whose generated input lacks transfer tags.
- Next: resolve those dependency/input-generation failures before release
  validation. No training, deployment, commit or push performed.

## 16 October 2026 — repository status reconciled

- [Completed work and remaining tasks](STATUS_REPORT_2026-10-16.md), reviewed
  against local HEAD `5efc693`, source files and saved evidence.
- Getting Started, opt-in shadows, still metadata/orientation, master provenance,
  attribution notes and sequence movie export are committed locally.
- Push-baseline A/B is complete and failed; no model promoted. Commercial
  licensing and calibrated shadow review remain open.
- Next: validate the latest Windows build and sequence export, then release
  gates. Historical passing tests are not proof of the latest revision.
- This increment inspected and documented only; no training or new test run.

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
