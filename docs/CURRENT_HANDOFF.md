# RUDRA — current development handoff

**Latest — parallel production preparation, no GPU use (19 October, evening):**
[Details, commands, evidence](ai/PRODUCTION_PARALLEL_PREP_2026-10-19.md). Native1080
run read from files (PowerShell Status not run from agent VM): training, step
5,600/20,000 at 5.09 h, errors empty; untouched. Saved 1000-5000 all fail the
both-positive PU screen; best.pt is ~step 800 with negative PU; old step 7000
stays the reference. Validation review: all 25 scenes real footage + pipeline
SDR; targets almost never exceed 1000 nits (max 1,903) and contain no
fire/sparks/sun/water/neon, so the PU gate cannot test >1000-nit highlights.
SI-HDR/ToS colour, scale, alignment, plateaus, overlap and exclusions verified
and recorded; not converted or scored. Added training/temporal_stability.py
and training/production_candidate_compare.py with 20 passing tests; pinned
evaluation files unchanged. Provenance review: both runs scratch; 2,439/3,231
rows rely only on the owner's footage statement; all sources weight
distribution false. Next: after exit audit/evaluate step 7000 + shortlist,
then candidate compare. Owner rights confirmation still needed.

**Latest — production evaluation kit prepared (19 October):** Owner authorized
parallel production preparation during training. Added training/production_eval_kit.py
and scripts/evaluate_hdr_production.ps1. Frozen 251 validation pairs /25 scenes
on G:/rudra_training/production_eval_20261019/validation_kit; verified 502 media
hashes, no scene-ID split overlap. Native tiled clean/hard evaluation against
bt1886 baseline, strict frame coverage and equal-scene PU confidence intervals.
19 focused tests pass; actual full export/score deferred while trainer active.
No GPU work, training mutation or release pass. Indexed 31 existing source
evidence files and 25-scene category review queue; independent SI-HDR/ToS
colour/overlap and temporal checks remain unresolved. Full commands/evidence
in [next-run guide](ai/NEXT_NATIVE1080_TRAINING_2026-10-19.md).

**Latest — production checkpoint audit (19 October):** Owner requests production
quality and commercial use. New native1080 worker 51548 observed present at
initial validation; do not restart or mutate it. Old highlight run stopped at
14,400/8h; last completed evaluation 14,200. Added read-only CPU checkpoint
audit and scripts/audit_hdr_production.ps1; eight tests pass. Fifteen saved
weights audited: 7000/11000/14000/13000 have positive clean AND hard PU gain.
best.pt is step 600 and has negative hard PU gain; do not blindly promote it.
Evidence: G:/rudra_training/production_audit_20261019/highlight_checkpoints.json.
Production work sequence and proposed acceptance criteria are in
[next-run guide](ai/NEXT_NATIVE1080_TRAINING_2026-10-19.md#production-continuation--19-october).
Next: compare actual saved shortlist on per-scene validation/deployment path,
then independently prepared full-resolution and temporal evidence. No new
training, source admission, release authorization or final quality pass here.

**Latest — native-1080p next-run preparation (19 October):**
Owner asked to prepare the collected sources and provide the next training
script. [Next-run instructions](ai/NEXT_NATIVE1080_TRAINING_2026-10-19.md).
Separate candidate: `G:/datasets/corpora/rudra_native1080_expansion_20261019_v3/`.
Completed 152 new pairs (136 panorama + 16 native CG), 3,231 total, unchanged
251 validation / 260 test. All 6,462 media hashes verified, all 152 new pairs
read by the actual loader with finite 256px crops and correct HDR unit bounds.
All 152 SDR previews and eleven paired/full-resolution diagnostic categories
inspected; [review](ai/NATIVE1080_VISUAL_REVIEW_2026-10-19.md). Ready receipt
issued and actual launcher Check passes. Read its live workflow and
`scripts/prepare_next_hdr_training.ps1 -Action Status`. New guarded launcher:
`scripts/train_next_hdr_commercial.ps1`; defaults to Status, scratch only,
20,000 steps / 24-hour cap. Ten focused tests pass. Current highlight training
continues unchanged (last observed step 1,200). No next trainer started.
SI-HDR/ToS remain evaluation-source candidates; no unresolved source conversion
is admitted. New images are 1080p but the unchanged trainer uses 256px crops.
The next run is `fxtd_native1080_hdr_run01`; its frozen data snapshot is already
prepared. Start only after the active trainer exits. No resume support.

**Latest — next high-resolution source collection (19 October):**
[Collection report](ai/NEXT_HDR_COLLECTION_2026-10-19.md) and
[source rights](ai/NEXT_HDR_SOURCE_RIGHTS_2026-10-19.md). Separate G: source
pack `datasets/sources/next_hdr_sources_20261019_v3/`. All 23 8K HDRIs and
two Blender source assets downloaded/inspected; 181 photographic SI-HDR EXRs
and 362 simulated SDR inputs fully decoded, both archive publisher checksums
verified. All reference overviews and 11 paired comparisons inspected. ToS:
144 continuous frames in three sections plus 22 sparse stills, all finite and
hash-verified; overview reviewed. Original 00_1a section excluded after three
infinite-value frames; replacement 03_2a passed. Controller 33184's failure log
is historical and resolved by replacement. Read collection_summary.json,
live workflow_state.json and
`scripts/collect_next_hdr_sources.ps1 -Action Status`; no blind duplicate launch.
Six panoramas overlap historical held-outs and stay excluded from next train;
eight additional capture/location-screened panoramas are new to the auxiliary
manifest. Actual SI-HDR references are 1888x1280 with uncalibrated scale; three
larger ceiling plateaus need review. Keep SI-HDR and ToS as source/title-held-out
evaluation candidates until colour/objective/splits are settled. No merge or
readiness receipt exists for this pack. Native 1080p re-render plan saved; no
upscaling of 360p pairs as new detail. Current highlight trainer is active,
last observed step 580; unchanged. New source evidence is CC0 / CC BY with
attribution; release remains false. Completed/remaining details are in report.

**Latest — highlight candidate readiness completed (19 October):**
The user's failed Check was caused by the missing visual review, not failed
preparation. [Actual review](ai/HIGHLIGHT_VISUAL_REVIEW_2026-10-19.md) records
complete contact-sheet inspection and representative SDR/HDR/source-EXR views
for all 13 setups, plus 39 sampled EXR/target checks. Full finalization verified
6,158 media files and issued hash-bound ready.json for 3,079 pairs
(2,568 train / 251 val / 260 test). The real
`scripts/train_highlight_commercial.ps1 -Action Check` now exits 0.
No training launched. Old Z run stopped at its eight-hour cap at step 12,537;
last numbered checkpoint is step_0012000.pt. Keep its outputs. Readiness is
for a controlled internal pilot, not quality success or release authorization.
G: contains visual notes and visual_diagnostics/ as well as the frozen corpus.
Earlier pending-review snapshots below are superseded by this entry.

**Latest — next-run handoff and guarded launcher saved (18 October):**
[Claude Opus 5.5 next-run handoff](ai/CLAUDE_OPUS_5_5_NEXT_RUN_HANDOFF_2026-10-18.md)
is the primary continuation document. Also copied to
`G:/rudra_training/CLAUDE_OPUS_5_5_NEXT_RUN_HANDOFF.md`. Current Z run observed at
step 6,520, eight-hour cap. v2 preparation controller is now finishing fire
renders/conversion/verification; 127/156 rendered frames at snapshot. Re-read
live status. New `scripts/train_highlight_commercial.ps1` supports Check,
Prepare, Start, Foreground, Status, Stop; defaults to Status. New readiness
finalizer requires actual preview review, notes, hashes and full preflight.
Readiness is NOT issued now; Check correctly rejects the unfinished candidate.
Eight focused readiness/snapshot tests pass. No next training launched, no
current-training/preparation-code changes, and no promotion/release. Use the
new handoff for remaining tasks, commands, storage dependencies and budget caveats.

**Current — v2 fire/reflection preparation RUNNING (18 October):**
[Preparation report](ai/FIRE_REFLECTION_EXPANSION_2026-10-18.md). Owner requested
preparation for the next run. Sources on G: at
`datasets/sources/fire_reflection_expansion_20261018_v2/`; separate candidate at
`datasets/corpora/rudra_fire_reflection_expansion_20261018_v2/`. Six ambientCG
CC0 metals and three JangaFX free CC0 VDB sets selected. Source collection is
complete; two-thread CPU rendering continues; no new training launched. Target 13 x 12 frames,
312 clean/hard pairs, 3,079 combined including the earlier 2,767 unchanged.
All additions train-only. Source collection is complete (nine archives, 7.57 GB,
581 VDB frames, 42 fully decoded/rehashed PNGs). Final plan frozen; 36 selected
VDB headers loaded, fields/bounds verified; first 68 completed render records
matched final jobs. Remaining renders, paired-data verification and visual
review must complete before readiness. Read live
`scripts/prepare_fire_reflection_expansion.ps1 -Action Status` and files; do
not treat this target count as completed. Current Z trainer last observed at
step 5,000, PID 58624; unchanged. Material/reflection render subplan and log
are in `G:/datasets/research/fire_reflection_smoke_20261018/`; final fire
plan is now frozen in the candidate. The renderer rejects executing downloaded code.
Hidden one-shot controller PID 62416 is now waiting only for the existing
material renderer, then invokes the full resumable preparation. Inspect
`workflow_state.json`, `prepare_console.log`, and `prepare_errors.log` in the
candidate. Controller source/prerequisite PID-start records are in the smoke
audit folder. An initial controller array-parsing error was fixed and its log
retained as controller_attempt1_error.log; the replacement was observed waiting
normally with no errors. Metal + ground-fire previews inspected, four actual
loader samples pass. Ground-fire frame took 253 seconds CPU; full fire renders
may take hours. Fixed fire -2 EV pair exposure reduces near-ceiling target
values. Full-candidate checks and visual review are still pending.

**Current — supplied F: VFX library audited; zero additions (18 October):**
[Intake report](ai/F_VFX_LIBRARY_INTAKE_2026-10-18.md). Four owner-supplied paths
reduce to two non-overlapping roots under `F:/Assets/VFX Library/`: Comp and
Effects/FX/ActionVFX. 8,245 files / 62.12 GB inventoried; metadata/evidence on
`G:/datasets/research/f_vfx_intake_20261018/`. All 1,051 non-ActionVFX video
headers checked (one unreadable); decoded formats are 8-bit, none PQ/HLG-tagged.
3,818 PNG headers: 3,808 8-bit and ten 16-bit. No verified HDR ground truth.
ActionVFX explicitly restricts AI training/fine-tuning/validation without an
applicable separate grant. Action Essentials, Pexels-labelled clips and other
stock/tutorial packs remain on rights/provenance hold. No ActionVFX media
probe or model evaluation. Owner asked to identify FXTD-authored footage and
supply any special AI grant; reply pending. No files admitted/copied into
training. Existing 2,767-pair pilot and active trainer unchanged. Next: review
original camera/renders or applicable rights evidence when supplied; these
folders have not closed the real-highlight gap.

**Current — next-run highlight pilot prepared (18 October):**
[Expansion report](ai/HIGHLIGHT_EXPANSION_2026-10-18.md). Separate G: corpus
`datasets/corpora/rudra_highlight_expansion_20261018_v1/`: 328 new synthetic/
panorama pairs, combined candidate 2,767 (2,256 train / 251 val / 260 test).
All new rows train-only; original held-out rows unchanged. 22 HDR environments
plus eight authored CG setups (metal, sparks, fire, neon, windows); clean and
degraded SDR variants. Twelve panorama sources excluded for prior held-out or
mixed capture/location/family overlap. Charge legacy projects remain unused.
5,534 media hashes, full new-pair decode, split/similarity checks and 28 loader
checks passed. Sampled SDR previews reviewed; volume sampling noise, 640x360
resolution, very short snippets and uncalibrated source radiometry limit this
to a controlled pilot. See GAP_COVERAGE.md and review_status.json in the corpus.
Internal-pilot source evidence only; no release/distribution approval. No
automatic training. Next: assess existing run, then a separate matched scratch
comparison; real-camera highlights and independent temporal evaluation still
needed. Active Z run observed at step 4,400, PID 58624; re-read live status.
Status: `scripts/prepare_highlight_expansion.ps1 -Action Status`.
Owner subsequently supplied VFX-library paths; see the newer intake above.
Inspect rights/encoding/coverage before a separately versioned addition. Do not
infer that the remaining real-camera gaps are already filled.

**Earlier — free Blender/JangaFX gap sources downloaded (18 October):**
[Source audit](ai/BLENDER_HDR_GAP_SEARCH_2026-10-18.md). Owner has no Blender
subscription and prioritizes free downloads. Five free packs on G: at
`datasets/sources/blender_jangafx_gap_sources_20261018_v1/`: three Charge Blender
projects (fire, muzzle-flash/sparks, industrial materials) under CC BY 4.0;
two JangaFX CC0 fire simulations, 301 VDB frames. 323.3 MB downloaded with hashes,
licence evidence and credits. Archive extraction and Blender source inspection
passed; no missing external images. Sampled VDB headers load. Older Eevee
compatibility and blocked camera drivers still require render review.
Zero paired renders, no active training/data changes. Latest observed Z step
2,100, PID 58624, errors log empty; re-read live status. Cosmos HDR held for exact
CC BY-SA grant reconciliation; CGHEVEN excluded for explicit AI restrictions.
Next source work is compatible scene rendering, split freezing and quality
checks before any separate future training run. Real camera fire/sparks remain
a gap. No release authorization inferred.

**Earlier — commercial highlight sources downloaded; existing training continues (18 October):**
[Collection report](ai/COMMERCIAL_HIGHLIGHT_COLLECTION_2026-10-18.md). Separate source
pack on G: contains 34 8K HDRIs and 12 metal materials / 86 files / 7.265 GB.
All publisher checksums and full image decodes passed. Poly Haven CC0 commercial
and explicit AI permission evidence saved. Zero new paired samples; nothing added
to the active trainer. RAW fire/spark and moving-metal video remain absent.
Capture overlap audit flags old auxiliary autumn_field train/val variants; that
auxiliary corpus is excluded from the current Z run. Future splits and compatible
pair rendering are required before these new sources enter training. Current
Z run last observed at step 1,700, PID 58624, errors log empty; re-read live status.
No release/promotion or changes to the active code, data or environment.

**Earlier snapshot — expanded training ACTIVE; preparation complete (18 October):**
[Claude Opus 5.5 HDR handoff](ai/CLAUDE_OPUS_5_5_HDR_TRAINING_HANDOFF.md) contains
the verified state, source interpretations, paths, commands and next checks.
Run `G:/rudra_training/fxtd_z_hdr_run01/` observed training at step 20, worker
PID 58624; re-read live status before acting. 2,439 pairs prepared: 1,928 train /
251 validation / 260 test, including 1,230 Z-derived pairs. All 4,878 media files
passed trainer preflight, no cross-split flags, errors log empty. All active
pair paths are on G:; Z: is unnecessary for this run, but retain its originals.
Run02 stopped at the eight-hour cap, status step 15,000; highest numbered saved
checkpoint is 14,000 and best.pt is step 400. Current new best.pt is still step 0
at inspection. Step-200 smoke and new model quality remain pending. No restart,
extra trainer, mutable dataset/code changes, promotion or release. This current
entry supersedes queued/in-progress snapshots below.

**Earlier snapshot — expanded Z: preparation and subsequent scratch run queued (18 October):**
Owner requested “all next” and confirmed LogC4 for the remaining seven TAHA UHD
files and Mataha A_0020C010. [Workflow report](ai/Z_FOOTAGE_PREPARATION_2026-10-18.md).
172 candidates / 516 sampled frames decoded successfully; seven same-split
similarities, zero cross-split flags. Frozen plan selects 123 take/version groups,
targets 1,230 new pairs, hashes 952.616 GB. Nine PQ display sources remain separate;
40 take/version alternatives omitted. All Mataha stays test-only.
`scripts/prepare_z_commercial.ps1 -Action Start -TrainWhenReady` launched a hidden
background workflow (launcher PID 45880; read live process/status, do not assume
PID persists). Output `G:/datasets/corpora/rudra_z_pipeline_20261018_v1/`.
Hash/render/combined-pair QC/preflight must finish before ready.json exists. Then
it waits for existing HDR trainers to exit and starts `fxtd_z_hdr_run01`, scratch,
20k steps / eight-hour cap with the existing numerical and step-200 gates. Current
run02 is untouched. Do not start another job or change preparation code in flight.
17 focused tests pass. This is launched work, not completed pairs or model quality.

**Earlier snapshot — FXTD scratch training requested and launched (18 October):**
Owner explicitly requested preparation, starting training and runnable scripts.
Launcher: `scripts/train_fxtd_commercial.ps1`; [run guide](ai/FXTD_TRAINING_RUN_2026-10-18.md).
Run: `G:/rudra_training/fxtd_hdr_run02/`; read `status.json`, `console.log`
and `errors.log` for live state before starting another job. Pipeline hashes and
decodes 1,209 pairs, screens split duplicates, then commercial preflight rehashes
all media before scratch GPU training. Target 20,000 steps, 8 hours after
preflight, step-200 non-regression gate. Internal-pilot receipt records this
training authorization, not release approval. FXTD pipeline targets only; no
Netflix, Poly Haven auxiliary data or old checkpoints. Fifty focused tests pass.
Explicit `--pipeline-corpus-ev 0` replaces silent legacy fallback for these
targets; source tonemap metadata remains null/rendered. No guarded resume yet.
Use Status/Stop commands; do not launch duplicate jobs or alter pinned decoder
files while preparation/training is active. No product model promotion.
Preparation and independent rehash passed for all 2,418 files; zero split
duplicate flags. GPU optimization confirmed through step 20 with finite loss
and no logged errors in the original run. The owner stopped that run and started
run02. Latest observed run02 status: training, step 12,000/20,000, elapsed
24,203 seconds; final quality remains unassessed. Preserve the eight-hour cap.

**Latest intake — ten additional Z: footage folders (18 October):**
[Intake report](ai/Z_FOOTAGE_INTAKE_2026-10-18.md). All accessible; 260 files,
1.107 TB, 195 successfully probed video headers. Owner confirmed "alexa logc 4"
specifically for Sudan, FASHONISTA and FOUAD HAMZA: 84 files. AMIN retains separate
ALEXA Mini LogC3 EI800 and PQ/BT.2020 interpretations. 164 source-identified
candidates, including 37 test-only Mataha; eight ambiguous and 23 excluded.
Review artifacts under `reports/z_footage_intake_20261018/`; no new training
pairs or active-run input changes. Next: pixel/colour QC, take/version and
cross-corpus deduplication, production splits, then isolated pair preparation.
All Mataha remains held out; group FASHONISTA folders and Fouad folders by title.

**Latest data request — commercial source selection (18 October):**
Owner broadened the prior FXTD-only preference to commercially usable supplied
datasets. [Selection report](ai/COMMERCIAL_DATA_SELECTION_2026-10-18.md).
New manifests: `G:/datasets/corpora/rudra_commercial_review_20261018_v1/`.
Primary FXTD pipeline: 1,209 rows; separate scene-linear FXTD/Poly Haven
auxiliary: 4,098 rows, disabled until its objective is specified. Netflix's
5,944 real pairs and the 7,153-row combined candidate remain pending and disabled:
fresh exact-source retrieval still finds conflicting AOM notices. No fabricated
commercial distribution receipt. Restricted/unresolved sources excluded.
16 source/legacy folders and 20 manifests inventoried; 42 sampled pairs pass
decode/shape/finite checks. Full hashes, perceptual duplicates, colour/scale and
target exposure convention remain open. Pipeline `tonemap_ev: null` must not
silently trigger the legacy -1 EV fallback. Scratch-only plan, no training,
checkpoint reuse, promotion, commit or release. Existing source datasets untouched.
The user's request authorized isolated selection artifacts and licence retrieval;
historical no-dataset-write notes below are not a new block on that completed work.

**Latest — headroom preview wired into the app; export, scopes and source whites (18 October):**
[Integration report](HEADROOM_APP_INTEGRATION_2026-10-17.md).

**What was added**
- `HeadroomPreviewDriver` (engine) renders on its own worker and posts opaque
  tokens. `present()` revalidates them on the UI thread. One token is in
  flight, and failures never fall back to legacy pixels.
- MainWindow routes the following through it when the window is capable, the
  policy is on and the shot is a movie: paused frames, seeks, play and stop,
  grade changes, undo/redo, project restore, model and hold changes, replace
  and close.
- **The capability is closed everywhere except tests and
  `RUDRA --headroom-check`.**
- `measure_headroom` provides the scopes and readouts. It borrows no SDR masks,
  and the clip lanes are hidden on this route.
- `export_headroom_movie` stages float EXRs. Only after the decoder receipt and
  the source identity validate does it encode via `deliver_linear_frames`. It
  then runs QC (probe and source re-hash), writes the provenance sidecar, and
  publishes without replacing anything. Deliver's HDR10/HLG/ProRes export uses
  it when the policy is on. EXR stills and log exports still refuse.
- `super_white_stats` reports super-white detail versus code-ceiling clipping.
- The driver also reports rendered-memory accounting, and the view conversion
  is now fused and threaded.
- A headroom project now saves `settle_grain = false`. Before this it could not
  be saved.

**Evidence: Linux only**

| Check | Result |
|---|---|
| Native tests | 330 pass, 48 skip (Windows-only) |
| App tests | 71 of 73 pass |
| Driver tests | 60 shuffled repeats and 5 ThreadSanitizer runs, all clean |

The 2 app failures are environmental; the baseline fails them too.

**Next**
1. Run `.\scripts\validate_headroom_app_2026-10-18.ps1` on Windows. It
   covers the build, tests, real decoder, viewer readback, export and timings.
2. Record its reports in PRODUCTION_PROGRESS.
3. Then E2's remaining consumers, E3's CPU preview rate, and E4's input
   overrides.

Keep public selection and default readers closed. No training, downloads,
dataset writes, commit, push or release.

**Latest — remaining-work ledger, session preview foundation and app workflow fixes (17 October):**
The owner requested the full remaining checklist before implementation; it is
saved in [REMAINING_WORK.md](REMAINING_WORK.md). It now lists only unfinished work;
record completion evidence in the progress report before removing a task. Partial
work leaves the unfinished portion listed. [Increment report](PRODUCTION_CONTINUATION_2026-10-17.md).
Policy now lives in `GradeSnapshot` (`project.grade.headroom_policy`), including
undo/restore, with default sessions/readers still rejecting it. Added the isolated
`HeadroomPresentation` bridge: session snapshots, source/seek/grade revisions,
out-of-order/foreign/late-failure rejection and serialized final display callbacks.
It is **not wired into MainWindow**. Render off UI; snapshot on the session owner
thread; the final callback must not re-enter/queue naked pixels. Rendered images
are additional memory outside the shared decoder-pixel allowance. A paused grade
change needs another seek. See the report/header contracts before integrating.
The Source card now shows actual still-profile/orientation/fallback details.
Added `--workflow-check --analytic`, bounded waits, safe dialog failure/reporting
and delivered-frame checks. CMake stages matching ONNX DLLs beside Windows app,
CLI/tests, fixing System32 1.17 vs SDK 1.22 loading; the warning is absent in the
full native run. The layer checker now handles standalone diagnostic targets.
**358 native / 65 app / 10 Python tests pass; 40 shuffled presentation executions
pass.** Rebuilt real Windows app standalone analytic workflow passes, including
three EXRs and a six-frame HDR10 movie. **8 HD/UHD benchmark runs** preserve all
master/baseline hashes; full CPU UHD preview is only **~0.56 fps** (8 frames in
14.18 s median), before GPU display. This differs from the earlier decode-only
10.52 fps and existing GPU viewer measurements. No real-time/quality gain claim.
Evidence: `reports/production_continuation_20261017/`, native/app JSON+logs with
prefix `production_continuation_*_20261017`, stress `headroom_presentation_stress_20261017.log`.
Initial test compile errors were corrected; final builds/layers/whitespace pass.
**Next:** MainWindow source/paused/playback/undo lifecycle and efficient preview
integration, then scopes/provenance/transactional exports and rendered-buffer
memory accounting. Keep public headroom selection/default reader guards closed.
Input overrides, delivery improvements, long-footage/GPU/platform acceptance,
package refresh and the full AI research line remain in the ledger. Commercial
clearance, calibrated HDR review, signing, clean machine and Mac are still open.
No training, downloads, source dataset writes, model promotion, package refresh,
commit, push or release. Historical approvals are not new pending experiments.

**Latest — asynchronous headroom playback and shared pixel budget complete (17 October):**
[Design contract, validation and measurements](HEADROOM_ASYNC_PLAYBACK_2026-10-17.md).
`HeadroomPlayback` provides generation-based open/seek/cancel, one decoding worker
and a cancellation worker. Old decoders are explicitly closed/reaped before new
ones start. Preview/receipt tokens reject old/foreign generations; late failures
clear queued output and cannot issue completion. `HeadroomPixelBudget` charges
staging and borrowed/promoted frames through final pixel destruction, including
frames held after their original stream owner retires. Default allowance: 256 MiB
per adapter, not a process/FFmpeg/GPU cap. The separate batch feeder is not part
of this shared allowance; account for it before mixing both paths in the app.
**350/350 native tests pass** (14 new); **60/60 shuffled race executions pass**;
**15/15 bounded benchmark cases pass**, all 360 frame hashes and source identities
matching the previous frozen path. No build failure or benchmark timeout this
increment. Current synchronous feed differs only +1.1% HD / +0.6% UHD in this
small sample. Background UHD feed is 24.0% faster, with 189.84 MiB native pixels;
request dispatch measures 0.0027–0.0104 ms. First UHD frame remains ~420 ms after
preparation; UHD is 10.52 fps before rendering. Slow consumers still take ~1.56 GiB
sampled total memory. No real-time, whole-process cap or production-footage claim.
Evidence/frozen executable/fingerprints: `reports/headroom_playback_20261017/`;
full native results: `reports/headroom_playback_native_20261017.{json,log}`.
Build/whitespace/fingerprint checks pass. Pre-existing ORT API 22/17 warning persists.
**Next:** integrate source-policy/session state and preview/paused-frame consumers
with undo/restore, stale-presentation checks and application lifecycle tests.
Read the report's contracts first: default reads poll; destruction blocks and
joins workers; callers must validate before presentation and serialize the final
presentation decision with their seek handling. Decode receipts are not export
publication receipts. Preserve full source checks, colour/delivery and artist
control gates. Keep headroom UI disabled/default reader guards active until scopes,
provenance and transactional export support the policy. Long-footage/multi-consumer
validation and existing commercial/model, calibrated HDR, signing, clean-machine
and macOS gates persist. App/Python/GPU not rerun; app/packages not rebuilt. No
training, download, source dataset writes, model promotion, commit or release.

**Latest — reusable protected source and cancellable preparation complete (17 October):**
[Ownership contract, measurements and limits](HEADROOM_PREPARATION_2026-10-16.md).
`PreparedHeadroomSource` shares full probe/hash results and source/decoder leases
across stream/batch owners; stop tokens cancel probing and chunked hashing.
Observed identity changes latch across all sharing owners, even if timestamps
are restored: **fresh preparation is required**, not another batch request.
Owner-specific preview/receipt validation, cancellation and pixel budgets remain.
**336/336 native and 36/36 focused tests pass**. **12 completed benchmark cases
pass**, including 18 seek frames and three full sequential controls matching the
previous frozen increment; content IDs/source hashes also match. Later
preparation+binding is **0.050 ms HD / 0.138 ms UHD**; first seek frame including
decode is **0.342 s / 1.183 s**. Full initial preparation remains expensive
(10.43/92.54 s in reused runs); UHD decoding remains below real time. HD has two
pairs; UHD has one, under varying load. No general speed or global memory claim.
One initial UHD case timed out at 180 s; retained in `initial_timeout.json`.
The bounded 300 s retry completed. Evidence/frozen binary/source fingerprints:
`reports/headroom_preparation_20261016/`; native test prefix:
`reports/headroom_preparation_native_20261016`. Build/whitespace checks pass;
one test compile error was corrected. Pre-existing ORT API 22/17 warning persists.
**Next:** asynchronous generation adapter with cancellation/join ordering,
stale-result rejection and one retained-frame budget spanning old/new owners.
Read the report's acceptance criteria. Current budgets are per owner, not global;
shared-source invalidation is checked at owner validation boundaries, not broadcast
as an immediate interrupt. Then session/preview/scopes/export/provenance integration.
Keep headroom UI disabled and default reader guards active until those consumers are ready.
App/Python/GPU not rerun; app/packages not rebuilt. No training, downloads, source
dataset edits, model promotion, commit or release. Existing commercial/model,
calibrated HDR, signing, clean-machine and macOS release gates persist.

**Latest — FFV1 decoder memory reduction and longer validation complete:**
[Measured profile, evidence and limits](HEADROOM_DECODER_MEMORY_2026-10-16.md).
Float FFV1 uses two input workers, one raw-output worker and one output-queue
packet. Other codecs retain existing settings. Command identity automatically
invalidates old cache keys; colour transforms, EOF/exit, leases, cancellation,
provisional/verified frame semantics and native pixel budgets remain unchanged.
**328/328 native tests pass**; **88 benchmark runs pass** (34 sweeps + 54 final
checks), with matching float and legacy control hashes. Short synthetic tree
memory drops **42.3% HD / 44.9% UHD**. On 240-frame clips, reduction is
**34.8% / 35.6%**; UHD total is still about **1.86 GiB**, and buffering grows
during the run. Slow consumers and existing batch controls pass. No hard global
memory cap, sustained-production or real-time claim: UHD long feed is 11.63 fps.
Next safe step: shared protected-source ownership and cancellable preparation,
then asynchronous generation changes with a combined retained-frame budget.
Opening the 10-second UHD fixture still costs about **35 seconds**; retain full
timing/identity checks while making preparation reusable/cancellable. Keep app
headroom disabled and default reader guards active until all consumers/export
transactions support it. Long production footage remains a validation gate.
Evidence/frozen binaries: `reports/headroom_memory_20261016/`; native results:
`reports/headroom_memory_native_20261016.json`. Build/whitespace checks pass.
Pre-existing ORT API 22/17 warning persists. App/Python/GPU not rerun; app/packages
not rebuilt. No training, source dataset changes, downloads, commit or release.
Existing commercial/model, calibrated HDR, signing and clean-machine gates persist.

**Latest — continuous headroom decoder prototype validated:**
[Design contract, measurements and next task](HEADROOM_CONTINUOUS_STREAM_2026-10-16.md).
Added isolated `HeadroomStream`: one decoder per immutable range, provisional
preview tokens, owner-specific decode completion receipts and explicit verified
frame promotion. Retained frames count against the byte budget. Late failures,
identity changes and cancellation invalidate trust; EOF/exit checks remain.
Shared private read leases preserve the existing verified batch feeder.
**326/326 native tests pass** (nine new); **42 paired benchmark runs** preserve
all float and legacy control hashes. Median synthetic feed gain versus the
previous frozen optimized batch feeder: **5.99x HD / 10.55x UHD**, one decoder
start per range. Source preparation remains costly; preparation+feed gains are
2.46x/4.26x. These are decode-only results, not full application performance.
**Tradeoff:** UHD sampled process-tree memory rises from 760.6 to 1234.7 MiB;
owned pixels remain within 256 MiB. UHD is still only 11.14 fps before rendering.
Next safe task: profile/bound external decoder buffering and validate longer
sequences. Then shared protected-source ownership, cancellable preparation and
an asynchronous adapter with a budget spanning old/new generations. Read the
report's receipt/export boundaries; a decode receipt is not export completion.
Keep app headroom selection disabled and default project-reader guards in place
until all consumers and export transactions support the policy. Evidence and frozen
binaries: `reports/headroom_stream_20261016/`; full native report:
`reports/headroom_stream_native_20261016.json`. Initial test compile error fixed;
corrected build/whitespace checks pass. Pre-existing ORT API 22/17 warning persists.
App/Python/GPU not rerun; app/packages not rebuilt. No training, dataset changes,
downloads, model promotion, commit or release. Existing release gates persist.

**Latest — headroom HD/UHD performance measured and incrementally improved:**
[Measured comparison and next design criteria](HEADROOM_PLAYBACK_PERFORMANCE_2026-10-16.md).
Added reproducible paired benchmarks/frozen binaries, four-thread float decode,
guarded FFV1/Matroska clock-based seeking, and better batch-space reclamation.
**317/317 native tests pass**; 18 seek fixture combinations plus differing audio/
video starts match exact samples. **36 paired benchmark runs** retain all float
frame hashes and legacy control hashes. Median synthetic feeder improvement:
**1.69x HD / 1.17x UHD**; sampled process-tree memory also decreased. These are
decode-only local diagnostics, not production throughput or quality guarantees.
The feeder is still **not real time** (5.97 HD / 0.98 UHD fps); UHD still starts
23 decoders for 24 frames. Next: stronger-model continuous-decoder design review
and an isolated prototype with explicit preview/cache/export completion semantics.
Preserve late-failure, EOF, source identity, cancellation and memory guarantees.
Do not enable app mode or remove default project-reader guards yet.
Benchmark evidence: `reports/headroom_perf_20261016/paired/summary.json`,
`comparison.json`, frozen `baseline.exe`/`candidate.exe`; see report for reproduction.
Pre-existing test ORT API 22/17 warning remains; no neural/GPU acceptance claimed.
App/Python/GPU not rerun. No dataset edits, training, downloads, app rebuild,
package refresh, commit or release. Existing commercial/release gates persist.

**Latest — bounded headroom float cache and asynchronous feeder complete:**
[Implementation, evidence and remaining performance work](HEADROOM_FRAME_CACHE_2026-10-16.md).
`HeadroomFrameSource` owns Windows read leases, source/decoder content identity,
full-resolution immutable frames and cancellable indexed requests. Budget covers
staging/cache plus retained consumer frames; invalid batches never publish.
Source changes invalidate cached data; stale request generations cannot publish.
Float decoder interruption now supports finish/EOF/child-exit waits safely.
**314/314 native tests pass**, including 12 new tests and unchanged legacy goldens.
Real fractional-fps, offset-PTS reads and +1 EV masters match full sequential
decode exactly. Prior 60 app/20 Python passes were not rerun this increment.
Next: benchmark and optimize sustained HD/UHD decoding; current verified batches
restart FFmpeg and traverse preceding frames, so long-video efficiency is not
ready. Source-open hash/probe cancellation and OS peak-memory measurements remain.
Then wire session/preview/scopes/export/provenance before UI enablement. Keep
default reader guard; headroom is still unavailable in app/CLI selections.
Read leases are Windows-only; other platforms fail closed. No GPU/visual/speed
claim, training, dataset changes, app rebuild or package refresh. Release gates persist.

**Latest — headroom storage compatibility and indexed decode complete:**
[Evidence and next integration criteria](HEADROOM_STORAGE_2026-10-16.md).
Headroom projects use version 2; legacy writes stay version 1. Current app/default
readers and Python legacy render paths explicitly refuse unsupported policies.
Unsupported grades/missing masks fail without changing saved/current work.
FloatVideoDecoder ranges give exact paused/playback/full-decode sample parity on
the fractional-fps, offset-PTS fixture. Separate content/decoder/policy cache
identity implemented; the cache owner/payload store is not implemented yet.
**302/302 native, 60/60 app, 20/20 Python tests pass.** Initial fixture-tagging
failure corrected with no relaxed source validation. Math/defaults unchanged.
Next: bounded float cache/source owner and playback feeder, then session/preview/
scopes/export job wiring and provenance before UI enablement. Preserve the default
project-reader guard until those consumers really support the new path. Indexed
trim traverses preceding frames; full hash costs and performance need measuring.
No headroom app/CLI selection, GPU/visual/speed pass, training or package refresh.

**Latest — shared headroom grading / delivery boundary validated:**
[Implementation, restrictions and next wiring](HEADROOM_RENDER_2026-10-16.md).
`render_headroom_frame` applies exposure, Region EV and masks in linear Rec.2020;
`master_headroom_video_frame` explicitly adapts units to existing delivery code.
**293/293 native tests pass** (29 focused). Real 8/10-bit FFV1 fixtures retain
19/79 tested above-white steps through +1 EV, PQ spool and half EXR roundtrips.
Legacy paths unchanged. This is not app/CLI enablement or encoded-movie acceptance.
SDR anchor/chroma/grain must be off; unsupported requests fail, never silently
ignore controls. No new temporal/GPU/performance/visual pass is claimed.
Next: versioned project policy and fail-closed compatibility, shared paused/playback
float source/cache identity, then preview/scopes/master/movie caller wiring and
provenance before UI exposure. Read the report's integration limits first.
Current eligibility remains tagged 8/10-bit YUV444. Release gates persist.

**Latest — opt-in BT.1886 headroom baseline complete:**
[Implementation, evidence and integration limits](HEADROOM_BASELINE_2026-10-16.md).
Versioned `bt1886-headroom-v1`, disabled by default, consumes ExtendedVideoFrame
and returns linear Rec.2020 at 203 nits/unit before grade. Requires analytic
BT.1886 with clip fill/peaks, calibration and reference off; checks metadata.
**283/283 native tests pass**, including 19 focused decoder/baseline tests.
Real lossless decoder -> baseline fixtures retain all tested headroom levels;
10-bit code 1019 yields 249.743557 nits before grade. Legacy paths unchanged.
Next: shared paused/playback/master/movie-export integration, cache namespace,
project policy and provenance; verify grading controls before UI exposure.
No app/GPU/performance/visual pass claimed for this new path. Do not simply feed
its output through old SDR clamps or network-unit conversion. Release gates persist.

**Latest — isolated float decoder implemented and tested:**
[Implementation and boundaries](FLOAT_DECODER_2026-10-16.md).
FloatVideoDecoder / ExtendedVideoFrame preserve signed encoded RGB and source
contract, without entering the SdrImage/model pipeline. Eligibility is limited
to tagged 8/10-bit YUV444; alpha/unsupported sources are refused.
**274/274 native tests pass**, including 10 new decoder tests and real lossless
fixtures compared to an independent matrix and the legacy decoder. No app UI
or default delivery behavior changed; no app/GPU rerun or throughput claim.
Next: versioned opt-in analytic BT.1886 baseline with unit tests, following
SUPER_WHITES_REVIEW_2026-10-16.md; shared viewer/export/cache integration follows.
No new training authorization inferred. Packaging/release gates remain open.

**Latest — super-white preservation review complete:**
[Measured review and implementation gates](SUPER_WHITES_REVIEW_2026-10-16.md).
20/20 bounded decoder checks pass; float preserves 79/79 tested 10-bit
above-white levels, existing export and proposed limited-integer RGB keep 1.
Do not implement the old design's limited-RGB expansion or endpoint-slope
extension. New recommendation: separate float decoder, then versioned opt-in
analytic BT.1886 mode, with shared preview/export and explicit restrictions.
Next safe task: isolated float decoder API/tests with production default intact.
No production pixels changed or new native/GPU/quality pass claimed this turn.
Evidence: `reports/superwhite_review_20261016_final/review.json`; earlier failed
assumption run retained. Existing release gates and stale-package status persist.

**Latest — source inspection first increment complete:**
[Source inspection review](SOURCE_INSPECTION_2026-10-16.md).
Movie stream tags and limited-range/HDR warnings are shown in the Source card;
**60 app / 264 native tests pass**. Updated executable built and opened.
No headroom preservation or input override implemented. User approved the
proposed product improvements; remaining work is listed in the review.
Recommend stronger reasoning for the next super-white design review: preview,
playback and export decode differently, so decoder-only changes are inadequate.
Routine UI/tests can continue on Light. No new training approval inferred.
Packages remain stale and all outstanding release gates remain in force.

**Latest — user workflow review and F shortcut complete:**
[WORKFLOW_REVIEW_2026-10-16.md](WORKFLOW_REVIEW_2026-10-16.md).
F now invokes Fit without stealing text input; Getting Started copy updated.
Expanded export matrix found/fixed ProRes log frame-tagging QC failure.
**59 app / 264 native tests pass**, all eight movie encodings and four folder
movie formats exercised. Updated build launched for user; prior ZIP/installer
are stale and must be rebuilt before distribution. Next safe engineering:
analytic mode for standalone workflow-check and fresh evaluation packaging.
Manual visual/clean-machine/signing/licensing gates remain pending.

**Latest — portable acceptance kit ready; no clean environment available:**
Owner confirmed this limitation. See [CLEAN_MACHINE_KIT_2026-10-16.md](CLEAN_MACHINE_KIT_2026-10-16.md).
Installer, fixture, standalone checks and checklist are bundled. Copied script
passed on Windows PowerShell 5.1 against the staged package; this is local
portability evidence only. Clean install/uninstall remains pending until a
separate Windows VM/PC is available. Do not repeatedly substitute local tests.
Visual review, signing and commercial clearance also remain open.

**Latest — evaluation installer staged:**
[MODEL_FREE_INSTALLER_2026-10-16.md](MODEL_FREE_INSTALLER_2026-10-16.md).
Unsigned 144 MB installer compiled from the tested model-free folder; hash
recorded. Separate AppId, install folder/shortcuts, no `.rudra` associations.
Application preferences still shared: use a fresh user/VM. No installation
performed. Windows Sandbox absent here. Next: clean-machine acceptance using
`docs/CLEAN_MACHINE_ACCEPTANCE.md`; signing, visual review and legal clearance
remain pending. Do not report installer runtime acceptance as passed.

**Latest — model-free package implemented and tested:**
[MODEL_FREE_PACKAGE_2026-10-16.md](MODEL_FREE_PACKAGE_2026-10-16.md).
`PACKAGE_WINDOWS.ps1 -NoModels` stages in a fresh isolated directory and
excludes model weights/catalogue. Local unsigned ZIP created; restricted-PATH
startup plus analytic HDR10/ProRes delivery passed, model-contamination guard
rejection passed. Existing research packaging remains available. Next:
installer staging/review and clean-machine/signing checks. Owner is not ready
for visual HDR assessment; leave it pending. Commercial clearance unresolved.

**Latest — visual review opened / packaging preflight:** see
[RELEASE_REVIEW_2026-10-16.md](RELEASE_REVIEW_2026-10-16.md).
HDR review movie verified and opened; owner assessment and calibration details
pending, no visual pass claimed. Inno Setup exists, signing is not configured.
Packaging currently requires/copies non-commercial v8 and removes prior
versioned outputs, so it was not run. Next safe engineering task: explicit
model-free packaging with isolated output and restricted-PATH smoke checks.
Commercial clearance remains unresolved even without weights.

**Latest — four-backend latency gate PASSED, 16 October:** after the user
removed the competing workload, D3D12/D3D11/Vulkan/OpenGL all pass parity
and unchanged latency budgets. 4K composite + display: **1.16–1.48 ms**;
this is not neural inference latency. GPU memory ~1.6 GB, background activity
32–33%. Evidence: `reports/gpu_idle_20261016_153330/`. Primary-display HDR
probes already pass and pinned FFmpeg tests pass (264 native / 57 app).
Next: calibrated HDR/black-level review and remaining release workflow,
packaging/signing, clean-machine/platform and licensing checks. Historical
requests to free the GPU below are resolved; no more timing reruns needed
without a new change or failure.

**HDR rerun completed:** primary PA279CRV now has HDR on; D3D12 scRGB/HDR10
and D3D11 scRGB probes pass. Viewer HDR cards pass on D3D12/D3D11/Vulkan;
OpenGL reports no HDR. Secondary LG remains SDR. All GPU numerical parity
checks pass, but latency still fails under 100% GPU load (Python compute PID
45712 remains present). Next: idle-GPU latency rerun. Evidence:
`reports/gpu_ready_20261016_153101/`; see the GPU validation report.

**Latest — GPU parity and pinned dependency, 16 October:**
[GPU_RELEASE_VALIDATION_2026-10-16.md](GPU_RELEASE_VALIDATION_2026-10-16.md).
All four backends pass 28 composite + 30 view cases each and real-viewer SDR
parity. Timing gates fail under concurrent 100% GPU load; both displays report
HDR off. User intends to make GPU idle and enable HDR: clean rerun pending.
Approved FFmpeg 7.1.1 download is complete, pinned SHA-256 verified; full-build
**264/264 native and 57/57 app tests pass with that exact dependency**.
Next: rerun timing/HDR into new reports when the machine is ready. No changes
to display settings, unrelated workloads, budgets or system PATH were made.

**Latest result — all four Windows failures resolved:**
[FFMPEG_VALIDATION_FIX_2026-10-16.md](FFMPEG_VALIDATION_FIX_2026-10-16.md).
Full Release build: **264/264 native and 57/57 app tests pass**. Synthetic
input frame tags fixed the self-test/temporal failures; Python independently
confirmed the decoder-dependent alpha difference, and its test preserves the
existing tolerance. Earlier failure entries below are historical. Next:
release-candidate GPU parity/latency gates and packaged-dependency validation.
Installed FFmpeg is January 2025, not the package's pinned 7.1.1. Calibrated
display review and commercial licensing remain open. Changes are local.

**Latest Windows validation, 16 October:** see
[WINDOWS_VALIDATION_2026-10-16.md](WINDOWS_VALIDATION_2026-10-16.md).
Both configurations build. Minimal: 249 native / 41 app tests passed. Full
OpenCV/ONNX: 260/264 native and 57/57 app tests passed, including actual
folder-export HDR tags, fps and frame count. Four native failures remain
(alpha golden difference, FFmpeg self-test and two temporal generated-input
tagging failures). Next: diagnose those failures against the intended release
FFmpeg build; do not mark the release green. Changes this turn are test-only.

**16 October reconciliation:** read [the current status report](STATUS_REPORT_2026-10-16.md)
first. Local HEAD is `5efc693`: input metadata, provenance, attribution and
sequence export are committed. Older uncommitted/pending-A/B statements below
are superseded. Next is latest-revision Windows validation and HDR display
review; the push-baseline A/B is already complete. No new training is authorized
by this status update.

Saved 14 October 2026 (evening), after the frozen-checkpoint diagnosis and the push-baseline packet.
Read this first when switching models or starting a new conversation. This is
the current continuation point; older approval requests and next actions in
the progress history are historical, not pending instructions.

## Resume here

**16 Oct, 1.0 audit fixes (uncommitted, awaiting the Windows build).**
`scripts/release_input_provenance_2026-10-16.ps1` makes 3 commits:

1. **Stills (P1-1):** ICC / cICP / sRGB / cHRM primaries and EXIF
   orientation (`media/still_metadata`).
2. **Master provenance (P2-4, P3-1):** app version, build commit, source and
   checkpoint SHA-256, the real `rudra:tiled`.
3. **Notes:** NOTICE credits the Netflix titles; `docs/SUPER_WHITES_DESIGN.md`
   (P1-2, design only, documented as a 1.0 limit); `docs/LEGAL_QUESTIONS.md`.

Linux build without Qt: `rudra_tests` **263/263** with OpenCV. The Qt app
target was not built here, so rebuild `build\native_gate_b` and run
`rudra_tests` and `rudra_app_tests` before committing. The earlier shadows
and research commits were pushed (`1ab6a5a`).


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
