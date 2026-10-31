# RUDRA — current development handoff

**Current - native/research video decode mismatch measured (25 Oct):**

CPU audit tools/audit_aces_video_decoder_difference.py compares existing native
zscale and research swscale filters, without inference. New original 10-bit YUV444
colour fixture is byte-exact through FFV1. Difference max/RMS in255-code units:
colour0.9883/0.5038; prior HEVC neutral ramp0.9805/0.6432. Both retain877 ramp
levels. Evidence reports/aces_video_decoder_difference_retry1_20261025/report.json.
Do not silently connect existing native decoder to ACES learned export and claim
Python parity: match research decoder or separately qualify changed input codes.
No model/training/decoder behavior changed. Initial fixture failed roundtrip and
is preserved in reports/aces_video_decoder_difference_20261025; corrected generator
declares input AND output colour tags, one encoder thread, verifies exact bytes.
Training/verifier live; previous turn was schedule-audit progress. Goal incomplete.

**Current - 6400 schedule coverage measured (25 Oct):**

New CPU-only tools/audit_partial_duration_schedule.py reproduces all six common
counts in the completed1600 coverage audit exactly. Registered6400 schedule has
1013 distinct asset/origin crops versus962 by1600; later19200 frame exposures
include428 on51 newly encountered crops and18772 on prefix crops (97.77%).
Evidence reports/duration_schedule_coverage_20261025, hash-bound to all pools and
schedule. These are scheduled counts, not proof of executed updates or pixel
coverage. Fixed622 TRAIN-fit audit remains a common-supervision comparison.
Training/verifier remain live; no GPU work added. Goal incomplete, original gates.

**Current - native EXR collision guard; training 2225/6400 (25 Oct):**

Training PID41812 and verifier PID46372 confirmed live; verifier waiting for
launcher28076. GPU remains dedicated to training. Prior turn was a verified wait.
Native ACES-v2 input enumeration now precedes model loading and rejects duplicate
EXR stems (including ASCII case aliases), missing/unsupported inputs and empty
folders before creating output. Release CLI rebuilt successfully; five executable
refusal checks pass in reports/native_aces_v2_setup_20261025/input_preflight.json.
Reproducer: tools/audit_native_aces_preflight.py. This is input safety evidence,
not inference parity. Native ACES-v2 still only accepts stills/folders; direct
learned video/app integration remains pending. No candidate promoted; goal active.

**Current - training1600/6400;printed prefix matches;13 evaluator tests pass (25 Oct):**

79502live last1600/6400;64rounded25-step mean loss summaries match1600control.
Fullhistory check stillpending;wrong single-step audit oracle preserved/corrected.
13synthetic metric/qualification tests pass incl empty-support/per-scene safety.
No prospective evaluation/freeze;fullruntime not yetverified. See
production/AMBIENTCG_HOLDOUT_CANDIDATES_2026-10-25.md finalsection.
Verifier57516waiting;no concurrentGPUexport. Goal incomplete.


**Current - prospective evaluator guarded;training1325/6400 (25 Oct):**

OriginalDEVnumeric/hash/requiredcheck guard implemented;7tests pass. Actual1600
preflight rejected before model/newdata evaluation. Evaluatorprepared with48case
completeness,one-checkpoint freeze,rawreference metrics/FLOAT masters;fullruntime
unrun. No freeze marker yet. See production/AMBIENTCG_HOLDOUT_CANDIDATES_2026-10-25.md
latestsection. Training79502last1325/6400,verifier57516waiting;goal incomplete.


**Current -48prospective SDR cases prepared before model evaluation (25 Oct):**

2screened ambientCG scenes x4views x3EV x8/10bit=48sources,24trueHDRreferences.
Pre-registered fixedrecipe;CPUbuild and independenthash/shape/dtype checks pass.
No model evaluation until originalDEVpass and checkpointfreeze. Not fullholdout
qualification;thirdscene excluded. See production/AMBIENTCG_HOLDOUT_CANDIDATES_2026-10-25.md
finalsection; corpus ambientcg_prospective_stills_20261025. Training79502last775/6400,
verifier57516waiting. Goal incomplete;no promotion.


**Current - panorama audit complete; one source-pair identity unresolved (25 Oct):**

6667terminal exit0:993x3scores,zeroerrors,3positivecontrols pass.3skyflags
retained;visualfields/clouds differ. All3nearest5sheets inspected. DaySky071A
vs hilly_terrain_01 terrain similarity unresolved;not cleared. Other2sources
can proceed protocol preparation within audit scope,not certifiedholdout.
See production/AMBIENTCG_HOLDOUT_CANDIDATES_2026-10-25.md latestsection.
Training79502 last525/6400,verifier57516waiting;goal incomplete.


**Current - training advancing;panorama audit finishing and partial flags reviewed (25 Oct):**

Training79502 last325/6400,verifier57516waiting. CPUaudit6667 last950/993,
PID42836 live. First2skyflags visually different terrain/clouds;originalflags
retained,plains_sunset still unreviewed. No certification. Finalreport pending;
after completion run tools/render_holdout_similarity_review.py and inspect3
sheets. See production/AMBIENTCG_HOLDOUT_CANDIDATES_2026-10-25.md.
No GPUexport concurrency;goal incomplete.


**Current -6400 retry LIVE after user GPU-priority decision (25 Oct):**

User explicitly authorized stopping separate videoPID11592; identity verified,
only that process stopped. Existing79502 retry now trainingPID41812,launcher28076;
verifier57516 waits. Last25/6400. Pools/source snapshots/first1600schedule exact.
Full history check remains after completion;do not restart. CPUaudit6667 last
500/993 with skyflags needing review. No parallel GPUexport. See
production/PARTIAL_DURATION6400_RETRY_2026-10-25.md latestsection. Goal incomplete.


**Current -993-panorama CPU similarity screen LIVE (25 Oct):**

3tests pass; audit6667 compares3new sources against993known panoramas, fixed
.98correlation flag, yaw/exposure controls and explicit errors. Last50/993;
no GPU/model use. Poll6667; inspect finalreport before eligibility. Still
quarantine,not independent-holdout proof. See production/AMBIENTCG_HOLDOUT_CANDIDATES_2026-10-25.md.
Training79502/verifier57516 still queued behind videoPID11592;goal incomplete.


**Current - ambientCG older-storage overlap screens complete (25 Oct):**

Expanded sources+G:/rudra_v02 screen79,083media files,zero exact/path matches
or errors. Older-storage metadata scan complete/no references. Not a perceptual
near-duplicate proof;3candidates remain quarantine,not holdout-certified.
See production/AMBIENTCG_HOLDOUT_CANDIDATES_2026-10-25.md. Bothscreen jobs
terminal. Retry79502/57516 still queued behind live videoPID11592. No newmodel
results;goal incomplete.


**Current - ambientCG source basis verified;3ACEScg references converted (25 Oct):**

Provider Blender files explicitly mark3EXRs LinearRec709; inspected autoexecoff.
OfficialOCIO conversion toACEScg unitexposure, no clamp,3FLOAT/AP1 masters with
exactreadback and roundtrip checks. Still quarantine,not model outputs/holdout.
Evidence G:/datasets/corpora/ambientcg_quarantine_acescg_20261025/report.json.
See production/AMBIENTCG_HOLDOUT_CANDIDATES_2026-10-25.md finalsection.
Retry79502/57516 remain queued;goal incomplete,no promotion.


**Current - three ambientCG HDR candidates acquired after history screen (25 Oct):**

All10remaining PolyHaven candidates have prior source references. NewambientCG
3x2k candidates: selected families absent from historical text scan; commercial
CC0 receipt, ZIP sizes/CRC/SHA256 and finiteRGB checked. Not certified holdout.
EXRs lack chromaticities;confirm source basis before conversion;no model run.
See production/AMBIENTCG_HOLDOUT_CANDIDATES_2026-10-25.md. Retry79502/57516
still queued behind live videoPID11592. Goal incomplete;no promotion.


**Current - all4 candidate scenes REJECTED for fresh holdout (25 Oct):**

Reference scan85742 complete:all4already in run05/PeakNet corpora,3train/1val
in source manifest. Explicit holdout_eligibility=false saved; downloads preserved.
Name/content screen missed converted G:/rudra_v02/pairs_moves sources. Future
selection must screen historical scene IDs before download. No new holdout
coverage claimed. See production/HOLDOUT_CANDIDATES_2026-10-25.md final section.
Retry79502/57516 still queued behind videoPID11592;goal incomplete.


**Current - candidate exact-duplicate/location screens complete (25 Oct):**

28,969EXR/HDR paths screened,zero exact-byte duplicates or missing files.
Coordinate screen no known sourcewithin1km;265knownIDs lack coordinates,
road shares broader region. No independence claim;all4remain quarantine.
Historical reference scan85742 still live. Retry79502/57516 still queued behind
confirmed videoPID11592. See production/HOLDOUT_CANDIDATES_2026-10-25.md.
No GPU concurrency,threshold change or promotion;goal incomplete.


**Current - four CC0 holdout candidates acquired; not certified yet (25 Oct):**

4x2k PolyHaven HDRIs,56.49MB, providerchecksums and finiteRGB verified; official
license/API receipts saved. Deterministic metadata-only selection; no model
evaluation. Remain quarantine pending ancestry/content/scene overlap audit.
Reference scan85742 LIVE; empty output so far is not no-match proof. See
production/HOLDOUT_CANDIDATES_2026-10-25.md. GPUretry79502/57516 still queued
behind videoPID11592;do not duplicate. Goal incomplete; no promotion.


**Current - HEVC Main10 transport verified; retry still queued (25 Oct):**

Two original10bit ramp frames preserve877 levels vslegacy256; sourceHEVC Main10
YUV codes exact. CPUguarded600 export FLOAT masters/readback and PQ16 video
decode exact;all10checks pass. Scope is procedural/lossless, not real footage
or model qualification. See production/ACES_HEVC10_TRANSPORT_2026-10-25.md.
Retry79502/57516 still live waiting on videoPID11592;do not duplicate or run
GPU tests. No quality thresholds changed;goal incomplete.


**Current - retry comparison verifier ready; GPU queue still waiting (25 Oct):**

Summary helper now supports6400 retry paths and stronger checkpoint/identity/
TRAIN-audit checks. Completed1600 replay reproduces all saved metrics/failures
exactly in reports/duration_summary_verifier_20261025. No new quality claim.
Sessions79502/57516 confirmed live waiting; videoPID11592 still active.
See production/PARTIAL_DURATION6400_RETRY_2026-10-25.md for final summary command.
No duplicate launch; no concurrent native GPU test. Goal incomplete.


**Current - controlled6400 retry QUEUED, not training yet (25 Oct):**

Retry training session79502/launcherPID28076 waits for videoPID11592 and GPU
memory<6000MiB. Coordinator57516 waits for retry folder, then launcher lifetime.
Original failed6400 artifacts preserved. New root partial_pcgrad_duration6400_retry1_20261025
under G:/rudra_training. Identical trainer verified and registration saved.
Do not restart waiting jobs or run simultaneous native GPU checks. See
production/PARTIAL_DURATION6400_RETRY_2026-10-25.md for paths/verification.
Original31352/92141 terminal failed.1600 still latest completed;goal incomplete.


**Current - CUDA loader fixed;6400 FAILED, retry pending GPU availability (25 Oct):**

All4native targets build,394tests pass. CUDA export85883 explicitly stopped;
no GPU parity result. Training31352/PID35572 exit1 illegal CUDA memory access
after last375; coordinator92141 exit1 missing report. BOTH terminal, not live.
Fresh CUDA health passes; unrelated video generationPID11592 confirmed live,
left untouched. Defer retry until it ends; preserve failed6400 artifacts.
Next identical controlled6400 retry, with GPU workloads serialized. Exact cause
not established. See production/NATIVE_ACES_CUDA_LOADER_2026-10-25.md.
1600 latest completed; all3grading gates fail. Goal incomplete; no promotion.


**Current —1600 complete,6400 controlled run LIVE (25 Oct):**
1600 fixedTRAIN0EV1.705618->1.167375 vsanalytic.080820; aggregatephoto/allEV
improve, motion-3 regresses. DEVall3grades still fail;video/preservation pass.
Aggregate partial_duration1600_comparison_20261025/comparison.json saved.
New6400 training session31352/PID35572, coordinator92141 waiting. Pools/first1600
schedule exact; fullhistory required at end. No restart on timeout. Old17987 done.
See production/PARTIAL_DURATION1600_FINAL_2026-10-25.md for settings/next action.
Goal incomplete; no promotion/default/threshold change.

**Current — duration1600 video/preservation PASS; TRAIN fit LIVE (25 Oct):**
Hunyuan48 all8checks pass,flicker .01378024EV. Independent24frame raw/smoothed/
constrained full/outside-highlight preservation and frozen weights exact.
Coordinator17987 now train_fit; do not restart. All3grading failures remain.
1600schedule962unique crops versus622first400;934later exposures on340newcrops.
Next finish fit, run tools/summarize_partial_duration.py, then choose next model
experiment from evidence. See production/PARTIAL_DURATION1600_INTERIM_2026-10-25.md.
Goal incomplete; no promotion/default/threshold change.

**Current — duration1600 paired/grade complete; video evaluation LIVE (25 Oct):**
NewH .251418,oldH .381858,motion .008860 pass. Legacy .778691 worse thananalytic.
Grades-3/0/+3=18.763108/.073843/.004582 improve400 but ALL original grade gates
fail. Correct grading input hashes verified. Coordinator17987 live in video,
then preservation/TRAINfit; no restart. Run tools/summarize_partial_duration.py
once complete. See production/PARTIAL_DURATION1600_INTERIM_2026-10-25.md.
Goal incomplete; no promotion/threshold change.

**Current — native masters verified;1600 training DONE, evaluation LIVE (25 Oct):**
Training97117 exit0; six identity checks pass including complete400history.
Coordinator17987 live in paired evaluation; queued grade paths corrected before
grading (Windows separator bug), prior registration retained; verify input hashes.
NativeCPU/DirectML2frame EXRs pass FLOAT/AP1/mask checks; CPU Python exact,
DML within tolerance. Derived FFV1 PQ16 decodes exactly. Research weights only.
Native/audit jobs terminal; see production/NATIVE_ACES_MASTER_VERIFY_2026-10-25.md.
Next poll17987 and inspect full gates/TRAINfit. Goal incomplete; no promotion.

**Current — Windows native build/tests/parity PASS; training LIVE (25 Oct):**
Removed obsolete /permissive override; original19.51 builds all4targets.
394/394native tests pass;2real720p frames C++/Python bit-exact all stages. ONNX
CPU export passes existing tolerances; guarded600 research package only, no
release. Native jobs terminal; evidence reports/native_aces_v2_setup_20261025.
Training97117 last1475/1600, coordinator17987 waiting PID29648. Poll existing
handles. Next duration full evaluation, native master/DirectML checks. See
production/NATIVE_ACES_WINDOWS_PARITY_2026-10-25.md. Goal incomplete.

**Current — native compiler fallback LIVE; training LIVE (25 Oct):**
Build63214 failed C2139 in Torch headers with MSVC19.51. Minimal header compile
passes installed19.44; no dependency edits. Ninja configure succeeds using
scripts/BUILD_NATIVE_ACES_V2_MSVC1444.cmd; build39003 live,last15/169 in separate
build/native_aces_v2_msvc1444 (no Release subfolder). Training97117 last725/1600;
coordinator17987 waits PID29648. Poll existing handles. Runtime tests pending.
See production/NATIVE_ACES_COMPILER_2026-10-25.md. No promotion; goal incomplete.

**Current — native build configured and LIVE; training LIVE (25 Oct):**
Found existing pinned OCIO2.5.2 under C:/vcpkg/installed/x64-windows. Configuration
with LibTorch/OpenCV/ORT succeeds. Build session63214 compiling Release tests,
CLI and parity tools,parallel2. No runtime pass yet. Training97117 last525/1600;
coordinator17987 still waiting on PID29648. Poll existing handles, do not restart.
See production/NATIVE_ACES_BUILD_2026-10-25.md and duration operational record.
Goal incomplete; no promotion. Earlier missing-OCIO concern now resolved.

**Current — native preflight repaired; duration1600 still LIVE (25 Oct):**
Training session97117 last325/1600; coordinator17987 waiting on process29648.
Do not restart. Fixed Windows Torch root resolution (three parent levels->two),
verified real headers/DLL; full native suite failures now stop gate. PowerShell
syntax passes. Full native qualification still pending pinned OCIO C++ setup.
See production/NATIVE_ACES_PREFLIGHT_2026-10-25.md; duration operational details
remain in production/PARTIAL_DURATION1600_2026-10-25.md. Goal incomplete.

**Current — duration1600 training and verification coordinator LIVE (25 Oct):**
Training session97117/PID29648, last150/1600; coordinator session17987 waits on
actual process handle. Pools/first400schedule exact; full history check pending.
Coordinator then requires identity and runs all DEV/video/preservation/TRAINfit
checks. Inspect run verification_status.json and live handles; do not restart on
timeout. Root G:/rudra_training/partial_pcgrad_duration1600_20261025.
Six focused tests pass. No promotion; goal incomplete. Operational details in
production/PARTIAL_DURATION1600_2026-10-25.md. Earlier next actions historical.

**Current — hard TRAIN crop fitting diagnostic complete (25 Oct):**
Eight derivative probes all signs agree,max relative discrepancy .73615%.
Fixed top4TRAIN crops,100updates each: combined0EV25.666979->.875311 but
photo .499571->.602405 and-3EV576.609985->620.931152 regress.0EV-only worse.
Frozen weights exact;6tests pass; both jobs terminal; diagnostic checkpoints only.
Next deterministic uncappedPCGrad1600 from guarded600, unchanged originalrecipe;
require first400schedule/history reproduce saved400control, then full original
DEV/video/preservation and TRAIN fit evaluation. No promotion/threshold change.
See production/PARTIAL_HARD_CROP_FIT_2026-10-25.md. Goal incomplete.

**Current — full scheduled TRAIN fit audited (25 Oct):**
All622unique crops/1600frame/982607pixel exposures, four frozen checkpoints.
Raw pixel-weighted0EV analytic .080820, initial13.891455,uncapped1.705618,
capped12.755371,direct12.996763. Both domains remain worse than analytic.
Top10crops account66.40% uncapped/72.05%direct error; ferndale motion dominates.
Weights unchanged, analytic calibration reproduced, aggregation test passes;
audit terminal. Raw TRAIN values are not constrained DEV gate values.
Next hard-TRAIN-crop finite-gradient check and bounded fixed-batch fitting test
before longer training; see production/PARTIAL_TRAINING_FIT_2026-10-25.md.
No promotion/default/threshold change. Goal incomplete; earlier next actions historical.

**Current — direct projected400 complete;0EV still fails (25 Oct):**
No Adam moments; normalizedPCGrad with capped-control saved TRAIN step lengths.
Same pools/schedule/EMA; actual movement99.262% requested due rounding;nozero
directions. NewH .289495,oldH .381424,motion .008835 pass. Grades-3/0/+3=
11.256049/.103474/.002295:only0EV grade fails original. Video .015691EV/all8
and independent24frame preservation pass;15tests pass;alljobs terminal.
Next fixed TRAIN fit audit initial/deterministicuncapped/capped/direct on all
unique scheduled crops with explicit exposure weighting before longer training.
No promotion/default/threshold change. Evidence: production/PARTIAL_PCGRAD_DIRECT_2026-10-25.md;
G:/rudra_training/partial_pcgrad_direct_comparison_20261025/comparison.json.
Goal incomplete; earlier next actions historical.

**Current — deterministic Adam update-cap400 complete (25 Oct):**
All400actual steps<=1e-4; identical pools/schedule/EMA versus deterministic control.
Both freshly evaluated. Cap newH .287594,oldH .381257,motion .008796 pass.
Cap grades-3/0/+3=12.274305/.103647/.002594:0/+3fail,0worse than control.
Video .015544EV/all8 and independent24frame preservation pass.13combined tests;
alljobs terminal; no promotion/default/gate change. Next direct normalizedPCGrad
400 using saved capped actual step-length schedule, same deterministic recipe;
compare direction versus cappedAdam, full quality evaluation. No more weight
or step-count changes in this comparison. Evidence: production/PARTIAL_PCGRAD_CAP_2026-10-25.md;
G:/rudra_training/partial_pcgrad_cap_comparison_20261025/comparison.json.
Goal incomplete; earlier next actions historical.

**Current — deterministic PCGrad step replay verified (25 Oct):**
Initial default-mode replay drifted(92trainable tensors,max.003287); not exact.
Added opt-in deterministic algorithms;2step then400step control/probe pairs match
final tensors,complete histories,pools/schedule exactly. Seven tests pass.
8TRAIN updates: total-loss increases actualAdam2/8,matched-length projected4/8,
small1e-4 Adam0/8,smallprojected0/8. Local evidence only; alljobs terminal.
Next deterministic PCGrad400 with1e-4 L2 cap on actual Adam update, preserve
direction/moments/recipe; evaluate BOTH capped and saved deterministic control
through all original quality/preservation/video gates. Control not yet evaluated.
Evidence: production/PARTIAL_PCGRAD_STEP_REPLAY_2026-10-25.md;
G:/rudra_training/partial_pcgrad_deterministic_probe400_20261025/step_probe_summary.json.
Goal incomplete; no defaults/gates/promotion; earlier actions historical.

**Current — PCGrad400 complete; grading regression prevents promotion (25 Oct):**
Same pools/schedule/EMA and982607pixel exposures as temporal1. NewH .274009 and
oldH .380173 pass; grades-3/0/+3=18.927318/.078361/.010324 all fail. Video
.013881EV/all8checks and independent24frame preservation pass.16tests pass.
1747projections; actual Adam directions frequently adverse despite projected
direction favorable. StepL2 median.005702,min.002945,max.027561,all>.001.
Next exact replay with opt-in before/after and counterfactual-step diagnostics
on frozen preregistered TRAIN updates; verify final model tensors unchanged.
Measure actual losses, not just local predictions; preserve optimizer/RNG and
save selected states. All current jobs terminal; no default/gate/promotion.
Evidence: production/PARTIAL_PCGRAD_2026-10-25.md and
G:/rudra_training/partial_pcgrad_comparison_20261025/comparison.json.
Goal incomplete; earlier next actions historical.

**Current — parameter directions validated by finite steps (25 Oct):**
Same32TRAIN cases x2normalizations x4steps. At1e-4,288/288derivative signs agree;
median relative discrepancy.000125,p95.022912. Large1e-2 has22sign mismatches.
Local adverse directions confirmed, not causal explanation of0EV DEV failure.
Exact weight restoration; baseline losses match prior audit;5tests pass; terminal.
Next bounded opt-in PCGrad400 versus temporal1 original schedule/recipe/gates.
Read published algorithm, test vectors, log actual Adam step directional changes;
no guarantee every term improves. No production/default/threshold change.
Evidence: production/PARTIAL_PARAMETER_PROBES_2026-10-25.md and
G:/rudra_training/partial_parameter_probes_20261025/report.json.
Goal incomplete; earlier next actions historical.

**Current — parameter-objective conflict audit complete (25 Oct):**
Eight fixed TRAIN control batches x4weights,32cases; weights unchanged.
Common-controlEMA sum locally adverse to photo10/32,0EV1/32; temporal6/16.
Fixed normalization0EV adverse0/32; its weighted gradient median much larger
than photo. Conflicts present but insufficient explanation of0EV DEV failure.
Three analytical tests pass, audit terminal; no optimizer/default change.
Next symmetric parameter-step finite-difference validation across step sizes on
same frozen batches, exact restoration, before adopting gradient mitigation.
Evidence: production/PARTIAL_OBJECTIVE_CONFLICT_2026-10-25.md;
G:/rudra_training/partial_objective_conflict_20261025/report.json.
Goal incomplete; earlier next actions historical.

**Current — group-preserving coverage400 complete; rejected (25 Oct):**
All original motion scene/exposure slots and still batches retained; identical
pools. NewH .284811 passes; oldH .383616,0EV .094277,+3EV .002576 fail.
Motion/detail/video pass(.013259EV), independent24frame preservation exact.
14tests pass; alljobs terminal. No promotion/tolerance/default change.
Next frozen TRAIN parameter-gradient conflict audit across photo/eachEV/temporal
terms at initial and saved weights, before more schedule/weight changes.
PCGrad/CAGrad primary abstracts reviewed as diagnostic motivation, not adopted.
Evidence: production/PARTIAL_GROUP_COVERAGE_2026-10-25.md and
G:/rudra_training/partial_group_coverage_comparison_20261025/comparison.json.
Goal incomplete; earlier next actions historical.

**Current — sampling balance audited; group-preserving schedule prepared (25 Oct):**
Still79crop pool exactly reproduced;8 excluded reachable rows contain only111
pixels, all rejected by64candidate minimum. No threshold change. Coverage400
shifted .05-clipped motion frames338->392, so scene/exposure confound confirmed.
New schedule keeps every control motion slot group and all still batches, raises
motion union294071->412679(87.20% of pool),381unique pairs. All400batches validate;
14tests pass. Audits terminal; next train group-preserving400 with original EMA
temporal1 recipe/gates, then full evaluation. No new training launched yet.
Evidence: production/PARTIAL_SAMPLING_BALANCE_2026-10-25.md and
G:/rudra_training/partial_group_coverage_schedule_20261025.
Goal incomplete; earlier next actions historical.

**Current — coverage schedule400 complete; still unqualified (25 Oct):**
Motion pool union coverage62.14%->100%; identical pools/still batches/losses.
0EV improves .087391->.068653 but fails .060382; +3EV .002155 passes.
NewH .292812 narrowly fails .292351; oldH .379488, aligned .009776 pass.
Video .013260EV and all8checks pass; independent24-frame preservation exact.
13tests pass; alljobs terminal. No promotion/default/tolerance change.
Next TRAIN scene/exposure schedule-distribution audit and still-pool exclusion
audit before another controlled sampling correction. Coverage alone is not an
isolated causal explanation. Evidence: production/PARTIAL_COVERAGE_SCHEDULE_2026-10-25.md;
G:/rudra_training/partial_coverage_comparison_20261025/comparison.json.
Goal incomplete; earlier next actions historical.

**Current — actual TRAIN coverage audited (25 Oct):**
351 rows,321 with partial support;223 actually scheduled versus228 pool-eligible.
Unique supervision320158/2498368 partial pixels(12.81%),27.88% of reachable
support. Motion covers62.14% of available pool; still100%. Border excludes
1350052 pixels. Repeated982607pixel/1600frame exposures independently match
calibration. No causal claim or model change. Next prepare coverage-oriented
motion schedule using existing pools, fixed400updates/domain balance/still
schedule/loss/initialization; quantify gain before controlled training.
Evidence: production/PARTIAL_TRAINING_COVERAGE_2026-10-25.md and
G:/rudra_training/partial_training_coverage_20261025/report.json.
Audit terminal; goal incomplete. Earlier next actions are historical.

**Current — raw/final constraint gap audited (25 Oct):**
Eight TRAIN crop interiors: EMA0/1168 and fixed1/1168 partial pixels corrected.
Old DEV:14/24 nonempty partial images,207494pixels;35.19%/35.42% corrected.
Correction improves mean0/+3EV errors; both already worse than analytic raw.
Raw/final EMA-vs-fixed rankings differ. Not evidence to remove constraint.
Next quantify actual scheduled TRAIN crop-interior union coverage per scene/
exposure and never-supervised partial pixels before sampling/loss changes.
Four smoothing tests pass, audits terminal; no parameter/default change.
Evidence: docs/production/PARTIAL_CONSTRAINT_GAP_2026-10-25.md; roots
G:/rudra_training/partial_constraint_gap_20261025 and partial_constraint_dev_gap_20261025.
Goal incomplete; earlier entries historical.

**Current — fixed-scale temporal comparison complete; rejected (25 Oct):**
Same pools/schedule/400updates verified. Fixed0EV .085134 and+3EV .003077 remain
above original limits; oldH .387133 regresses/fails. NewH .290077, aligned .010177
and video .013332EV pass. Full/outside-highlight preservation exact on24frames.
All jobs terminal; no adoption/promotion/tolerance change. Next TRAIN raw-vs-final
appearance-constrained error and activation audit, before another loss weight.
Evidence: docs/production/PARTIAL_FIXED_SCALE_2026-10-25.md and
G:/rudra_training/partial_fixed_comparison_20261025/comparison.json.
Goal incomplete; earlier entries below are historical.

**Current — display gradient/scaling audit and fixed calibration complete (25 Oct):**
Eight TRAIN scenes x initial/final x3EV,96directions: no sign disagreement at
.01/.001 probes;4/7 at smaller steps. Moderate medians .28%/.19%; tails larger.
+3EV effective EMA coefficient varies .279->27.828 (~100x), floor never hit.
Fixed denominators from all1600scheduled TRAIN frames/982607pixels computed:
-3=149.4128279345,0=.08082007430,+3=.41492909093. No DEV/model-based selection.
Opt-in --fixed-display-scale implemented; nine tests pass; audits terminal.
Next matched400-update fixed-scale temporal1 versus existing EMA control, verify
identical pools/schedule and every original quality gate. No weights/promotions.
Evidence: docs/production/PARTIAL_DISPLAY_GRADIENT_2026-10-25.md; roots
G:/rudra_training/partial_display_gradient_20261025 and partial_display_fixed_scale_20261025.
Goal incomplete; historical entries below are not pending work.

**Current — partial temporal pair complete (25 Oct):**
Both400-update runs complete, same538 motion pairs/79still crops,228 frames and
batch schedules. Temporal1 passes newH .285187, oldH .381566, aligned .009988 and
video .013195EV. Still fails original0EV .087391 and+3EV .003335; no promotion.
Temporal0 aligned .009853 is slightly better, so no claim temporal term improved
that metric. Both preservation exact; nine tests pass; all jobs terminal.
Next frozen TRAIN finite-difference display-gradient/relative-scaling audit,
directional derivatives across step sizes and EV terms, before another weight.
Evidence: docs/production/PARTIAL_TEMPORAL_PAIR_2026-10-25.md and
G:/rudra_training/partial_temporal_comparison_20261025/comparison.json.
Goal incomplete; historical entries below are not pending instructions.

**Current — matched partial crop pair complete (25 Oct):**
400updates each,832 matching centres/212TRAINframes, exact batch schedules.
Aligned oldH .382479 passes;0EV .084570 improves control .108137 but still fails
.060382. Both fail newH(.301169 aligned) and aligned residual(.010595).
Both Hunyuan48 pass(.013158/.013328EV); preservation exact on24frames.
Eleven tests pass; all jobs terminal; no promotion/tolerance change. Next paired
motion training on aligned crops with active-support temporal objective, separate
still replay and bounded matched comparison. Do not blindly extend this recipe.
Evidence: docs/production/PARTIAL_PHASE_PAIR_2026-10-25.md and
G:/rudra_training/partial_phase_comparison_20261025/comparison.json.
Goal incomplete. Earlier entries below are historical, not pending instructions.

**Current — partial crop phase mismatch verified and option fixed (25 Oct):**
Eight fixed TRAIN scenes: aligned crop/full-frame max asinh difference9.54e-7;
one-pixel-offset crops .182714, unchanged by48->64px border. This demonstrates
phase mismatch, not cause of DEV failures. Added explicit --aligned-crops with
border>=64 and divisible4 dimensions; original behavior retained as control.
Seven tests pass; audit terminal; no retraining/default change. Next preregister
legacy/aligned partial comparison, fix local data portability, retain original
quality gates plus verified full-support preservation. CPU finite-difference
display loss needs explicit runtime planning. Evidence:
docs/production/PARTIAL_CROP_GEOMETRY_2026-10-25.md and
G:/rudra_training/partial_crop_geometry_20261025/report.json.
Goal incomplete; historical entries below are not new pending instructions.

**Current — independent partial preservation verified (25 Oct, Codex):**
rel3_w05 versus guarded_projected600: fully clipped and outside-highlight pixels
bit exact on all24 old DEV frames, raw/smoothed/constrained stages; nontrained
weights exact. Audit terminal; no default/weight/gate change. Original old H and
0EV grade still fail. No tolerance approval inferred from the assistant's attached report.
Saved trainer report uses229 eligible frames/916 crops from351-row pool; arbitrary
crop phase and48px border merit full-frame equivalence measurement before more
partial-route training. Next fixed TRAIN crop/phase/border audit, then targeted
correction if mismatch confirmed. Evidence:
docs/production/PARTIAL_PRESERVATION_2026-10-25.md and
G:/rudra_training/claude_partial_finetune_20261025/ft_rel3_w05/preservation_independent_20261025.json.
Earlier entries below are historical; goal remains incomplete.

**Current — partial-highlight fine-tune arms + native ACES v2 staged pipeline (25 Oct):**
MODEL: 4 arms fine-tuned only highlight_context (frozen parent bit-exact), 400 steps on 351 CC0 TRAIN
rows (tools/train_partial_display.py; now snapshots itself). With hybrid64 + 1-code constraint, NONE
passes all gates. Best 0EV: display_w1 .0732 / rel3_w05 .0833 vs analytic .0604 (baseline .1210);
rel3_w05 passes +3/-3EV but old H .38539 > cap .38252. Partially clipped 0EV error .186 -> .133 code^2
(analytic .086): RMS .43 -> .36 vs .29 code, all sub-half-code. outside_full_exact is false by design
for partial-route training (outside_highlight_exact holds): gate needs restating. DECISION NEEDED:
keep "no worse than analytic" exact grading gates (then more GPU research), or pre-declare a display
tolerance (e.g. RMS within 0.1 code of analytic) BEFORE testing on a fresh holdout. Not holdout evidence.
Evidence: G:/rudra_training/claude_partial_finetune_20261025/README.md.
NATIVE (no default changed): core/aces_v2 (compose, hybrid64 smoothing, 1-code constraint, run_frame),
core/aces_v2_manifest (contract rudra_native_aces_v2_staged major 2), infer/aces_ocio_view.cpp
(RUDRA_WITH_OCIO, OCIO 2.5.2 exact), infer/aces_g1_libtorch.cpp, ORT G1 in onnxruntime_backend.cpp,
engine/aces_v2_job (background master job, EXR float32 + rudra.reconstructed mask), CLI
`rudra-native aces-v2`. tools/export_native_v2.py writes rudra_g1.ts + rudra_g1.onnx + manifest.
Parity: C++ vs Python bit-exact at every stage on 8 real frames (constraint also on perturbed inputs);
TorchScript bit-exact incl. odd sizes; ONNX within .002 code on real frames, mask identical.
1080p CPU (2 cores): OCIO inverse .78s x2, smoothing .13s, constraint .08s; G1 needs GPU timing.
tests/test_aces_v2.cpp: 8 tests pass (cloud gtest; and on this machine's Linux sandbox via a shim).
RUN ON WINDOWS: scripts/NATIVE_ACES_V2.ps1 (-OcioPrefix <OCIO 2.5.2> or VCPKG_ROOT) = export, build,
unit tests, stage parity, LibTorch + DirectML masters, Python reference export.
Also: tools/rudra_aces_export.py --clipped-smoothing hybrid64; docs/THIRD_PARTY_LICENSES.md (draft);
provenance archive G:/rudra_training/provenance_guarded_projected600_20261025 (7-checkpoint chain, all
10 recorded code hashes resolved; root run05_aces_lower_lr300 records no parent/trainer). Signing: last.

**Current — clipped-region smoothing: first candidate passing video + all 11 paired gates (24 Oct):**
Spatial600 evaluation finished and is REJECTED (new H .29592, old H .38930 > cap, full detail .07442,
Hunyuan .02230EV). Diagnosis: inside fully clipped support, predicted log-luminance gradients are
nearly uncorrelated with the reference (cosine .02 spatial600/.09 parent); optimal texture scale ~0,
so the detail gate rewards not inventing texture. New opt-in post-process
training/aces_clipped_smoothing.py smooth_clipped(mode='hybrid',sigma=64,ramp=1,start=0): Gaussian
log-luminance field in fully clipped support, exactly flat RGB in each 2-px-eroded interior,
outside-support bit-exact. Apply to raw prediction BEFORE constrain_colour(include_fully_clipped=True).
4 tests in tests/test_aces_clipped_smoothing.py (passed in cloud; rerun on this machine).
guarded_projected600 + hybrid64 (cloud CPU reproduction; raw metrics match stored CUDA to ~1e-7,
Hunyuan exactly): ALL 11 paired gates pass - new H .28844 (<= .29235), old H .38117, full detail
.0591864 (exact tie with analytic), aligned .00865. Hunyuan48 .01539EV and all 8 video checks pass.
Old24 grade: detail and -3/+3EV pass; 0EV FAILS .1210 vs analytic .0604. Failure is entirely in
PARTIALLY clipped highlights (.186 vs .086 code^2); fully clipped already beats analytic. Targets equal
the reference on old24, so the gate is consistent. Half-code discrete backtrack (0EV .0694, breaks
aligned residual) and continuous display projection (0EV .0615-.081 but new H .315-.358) both rejected.
Next (GPU): fine-tune partial-highlight route with a 0EV display-space loss vs target, keep hybrid64
at inference, rerun all gates + generated-suite review. Development data reused; not holdout evidence.
Evidence: G:/rudra_training/claude_clipped_smoothing_20261024/README.md. No default changed.

**Current — learned-context TRAIN audit complete (24 October):**
All four certified-blind images/43 crops audited;110,771 unique/701,385 repeated
pixels. Parent/local/coarse blind MSE7.25548/7.30995/7.36265; both residuals worsen
all four images. Coarse remote sensitivity is real but output std only3.38e-8 to
1.83e-6 versus target .12-.59; detail essentially unchanged. Seven tests pass.
Float32 variance artifact corrected; authoritative rerun is
G:/rudra_training/aces_learned_context_stable_20261024/report.json.
All jobs terminal; no weight/default change; goal incomplete. Next spatially
varying coarse encoder/decoder bottleneck, verify position-dependent distant
influence/crop consistency, then matched bounded comparison with unchanged gates.
Evidence: docs/production/ACES_LEARNED_CONTEXT_2026-10-24.md.
Earlier entries below are historical, not pending instructions.

**Current — local/wider-context comparison complete (24 October):**
Both600-update runs and all evaluations terminal. Same3,251 parameters, initial
weights/schedule/first loss; frozen v9 exact. Eight focused tests pass. New H
.290632 local/.288728 coarse both pass, but full detail,0EV grading and video
fail; coarse also fails old H control. Flicker .023176/.022415 exceeds .02.
No promotion/default change; goal incomplete. Evidence:
docs/production/ACES_WIDE_CONTEXT_2026-10-24.md and
G:/rudra_training/aces_wide_comparison_20261024/comparison.json.
Next audit trained residual use on four certified-blind TRAIN replay images:
regional photo/detail, distant-context sensitivity, inherited versus added error.
Do not blindly extend this recipe. Historical entries below are not pending work.

**Current — receptive-context audit complete (24 October):**
Conservative54px full-branch reach verified structurally and numerically. Four
replay images contain110,771 certified-blind white pixels with varying HDR targets;
19.753% of scheduled replay full supervision is in these regions. None certified
in288 motion images; this does not explain all failures. Five tests pass; all jobs
terminal. No parameter change/promotion; goal remains incomplete.
Next original zero-initialized coarse full-frame context residual, existing branches
frozen and outside-full exact. Compute full-image conditioning before replay crops;
test identity/reach/crop consistency, then bounded comparison and unchanged gates.
Evidence: docs/production/ACES_RECEPTIVE_CONTEXT_2026-10-24.md.
Root: G:/rudra_training/aces_receptive_context_20261024.
Earlier entries below are historical, not new pending instructions.

**Current — balanced-domain gradient audit complete (24 October):**
All100 frozen TRAIN windows at three states audited;1800 sample-state/300 window
rows. Replay/motion norm balance reverses1.146 initial ->1.509 sequential ->.740
accumulated. Better TRAIN fitting still accompanies6/8 new DEV scene regressions.
No parameter changes/promotion; all jobs terminal. Eleven helper tests pass and
initial training loss reproduces. Goal remains incomplete.
Next verify full-branch spatial receptive support and audit fully clipped TRAIN
input ambiguity/target variation before a wider-context intervention. Do not infer
causality from gradient snapshots or choose a replay weight blindly.
Evidence: docs/production/ACES_BALANCED_DOMAIN_AUDIT_2026-10-24.md.
Root: G:/rudra_training/aces_balanced_domains_20261024.
Earlier entries below are historical, not new pending instructions.

**Current — balanced accumulation comparison complete (24 October):**
All jobs terminal. Identical600 sample exposures, sequential600 versus accumulated
100 optimizer updates. Both fail new highlight/full detail/old detail/0EV grade;
Hunyuan48 .044291 versus .054989EV fails .02. Accumulation improves old H but
worsens new/legacy H and video. No promotion or default change. Ten tests pass;
parent/outside-full and sample schedule equality/counters verified. Goal incomplete.
Next audit motion/replay gradient magnitude/direction plus scene/exposure strata
on frozen schedule at initialization and final checkpoints before a targeted
loss/representation change. Do not simply extend this failed recipe.
Evidence: docs/production/ACES_BALANCED_ACCUMULATION_2026-10-24.md.
Aggregate: G:/rudra_training/aces_balanced_comparison_20261024/comparison.json.
Roots aces_balanced_sequential600_20261024 and aces_balanced_accum6600_20261024.
Earlier entries below are historical, not new pending instructions.

**Current — cyclic panel rotations complete (24 October):**
All jobs terminal. Across three folds current-only27/36 accepted, panel-aware7/36
(6,1,0). Monitor photo mean+2.178% versus+0.372%; no robust reconstruction gain.
Small fixed-panel safeguard is rejected as main training path. No checkpoint saved
or promoted. Frozen parent, original branch, rollback, hashes/rotation coverage
verified; new aggregator compiles. Goal remains incomplete.
Next preregister balanced multi-scene gradient accumulation over full eligible
TRAIN, preserving objectives/geometry/frozen region. Explicitly distinguish update
and sample-exposure budgets. Train research checkpoints and run unchanged full
DEV/detail/grade/video gates; no favourable-panel or best-step selection.
Evidence: docs/production/ACES_ANCHOR_ROTATION_2026-10-24.md.
Aggregate: G:/rudra_training/aces_anchor_rotation_20261024/report.json.
Earlier entries below are historical, not new pending instructions.

**Current — panel-aware direction12 complete (24 October):**
Three arms terminal: current-only12/12, rejection-only0/12, panel-direction6/12.
Panel aggregate metrics preserved but monitor photo mean+1.43%,4/6 members worse;
one anchor/monitor near-region detail regresses. No broad-training adoption,
checkpoint/default promotion or saved inference weights. Original branch restored.
Eleven tests pass; exact rollback and frozen-parent checks pass. Controls reproduce
acceptance with max measurement drift3.82e-7. Goal remains incomplete.
Next preregister cyclic role rotations of the same18 TRAIN scenes, reset per fold,
compare current-only versus panel-direction with matched12 attempts/scales and
unchanged gates. Do not tune panel selection to observed monitor failures.
Evidence: docs/production/ACES_ANCHOR_DIRECTION12_2026-10-24.md.
Root: G:/rudra_training/aces_anchor_direction12_20261024.
Earlier entries below are historical, not new pending instructions.

**Current — anchor retention12 pilot complete (24 October):**
All jobs terminal. Current-only12/12 accepted; strict anchor gate0/12 with48 scale
rejections despite local acceptance. Original branch restored; no inference weights
saved or promoted. Panel gate is not adopted. Current-only averages improve but
photo regresses on1 anchor and2 monitor scenes; no short-run aggregate detail or
motion regression. Do not claim this reproduces the long-run detail failure.
Eight tests pass; exact rollback/counters/frozen parent verified. Goal incomplete.
Next identify rejecting anchor objectives for six distinct initial proposals and
pilot panel-aware directions with unchanged panel/scales/finite-step gates.
Evidence: docs/production/ACES_ANCHOR_RETENTION12_2026-10-24.md.
Root: G:/rudra_training/aces_anchor_retention12_20261024.
Earlier entries below are historical, not new pending instructions.

**Current — guarded600 diagnosis complete (24 October):**
All jobs terminal; no training or default change. Detail regresses in all eight
new scenes; medium/heavy clipping account for96.3% of net detail regression.
New deep-region RGB-asinh gradient error +107.2%; selected TRAIN motion also
regresses. Hunyuan worst transition19->20 has both support and learned-field
variation; appearance constraints reduce but do not remove flicker. Four tests
pass; scene aggregates reproduce acceptance to1e-12. Goal remains incomplete.
Next freeze and run a bounded TRAIN-only anchor-panel retention comparison,
with matched proposals/budgets and unchanged DEV gates. Do not replace acceptance
metrics or infer cross-scene preservation from current-batch acceptance.
Evidence: docs/production/ACES_GUARDED_DIAGNOSIS_2026-10-24.md.
The entries below describe earlier work and are not new pending instructions.

**Current — guarded600 and input-precision work complete (24 October):**
Both candidates fail qualification; no default promotion. Projected new-highlight
error .292532550 misses the registered 5% target; full detail and old 0 EV grading
fail. Opt-in inverse-boundary repair passes 48-frame appearance at 16-bit decode,
but flicker .021655090 EV exceeds .02. All launched jobs terminal; 28 tests pass.
CLI decode now defaults to 16-bit; historical API/benchmarks retain 8-bit. Both
four-frame actual exports independently verify exact EXR-to-video codes and tags.
Current parent's 527 declared data rows audited; full historical source and release
closure remain open. Native integration, real input breadth, calibrated review,
Mac/clean-machine validation and signing remain. Goal is not complete.
Authoritative current checklist: docs/production/GOAL_REMAINING_2026-10-24.md.
Evidence: docs/production/ACES_GUARDED_MIXED600_2026-10-24.md and
ACES_INPUT_PRECISION_2026-10-24.md. Next diagnose per-scene detail/flicker regressions
before a targeted registered intervention; do not weaken gates or run blindly.
Entries below are historical, not pending instructions or current run statuses.

**Latest — persistent-Adam five-step TRAIN pilot complete (24 Oct):**
Same24 pair starts, both methods, max5 updates and four-scale search; all jobs terminal.
Protected81 accepted/94 attempted vsAdam45/64; full budget11/24 vs5/24.
Final trained-pair pass21/24 vs13/24; neighbouring overlapping pair18/24 vs10/24.
Median temporal loss change -17.26% vs-1.18%, including zero-update starts.
13 protected early stops,3 without updates. Each method has neighbouring detail
regression on1 pair and temporal regression on2 pairs. Not generalization evidence.
32 exact rejection rollbacks; optimizer counters verified; parent exact and original
branch restored. Initial losses/first-step outcomes reproduce previous pilot.
5 tests pass. No inference checkpoint/default change. Goal incomplete.
Next registered bounded mixed-scene/replay comparison with equal attempt budgets,
transactional rejection and scene/clipping acceptance logs, then unchanged DEV gates.
Define replay photo/detail-only acceptance before running; no temporal on stills.
Root G:/rudra_training/aces_protected_trajectory5_20261024.
Evidence: docs/production/ACES_PROTECTED_TRAJECTORY5_2026-10-24.md.

**Latest — protected-step TRAIN pilot complete (24 Oct):**
24 independent probes, four salted-hash groups per clipping stratum; fresh Adam1e-4.
Matched four-scale search: projected21/24 pass vsAdam13/24 (full Adam10/24).
8 projection-only passes, none Adam-only; all gains in medium/heavy clipping.
Neighbour pair passes18/21 projected and10/13 searched-Adam accepted updates;
these overlapping same-scene pairs are not independent holdout.
3 light-clipping probes stall with slight temporal increase despite first-order
feasibility. Actual finite-step gate is necessary; not proof no useful step exists.
Original branch restored and parent exact; no inference checkpoint/default change.
Initial comparison had search-budget confound, resolved by disclosed matched control.
Next bounded multi-step TRAIN pilot with persistent moments, equal search budgets,
explicit weight+optimizer rollback on rejection, including all stalled cases.
5 focused tests pass;3 rerun after matched-control addition. All jobs terminal.
Evidence: docs/production/ACES_PROTECTED_STEP_2026-10-24.md. Goal incomplete.
Roots G:/rudra_training/aces_protected_step_20261024 and aces_protected_step_matched_20261024.

**Latest — gradient alignment/scene audit complete (24 Oct):**
192 TRAIN pairs at both final checkpoints;384 pair/checkpoint evaluations, no updates.
Active checkpoint temporal/detail conflict73/192 vs32 prior; photo/detail92 vs61.
Combined plain-gradient direction is detail-uphill87 vs55, photo-uphill0 in both.
These are local final-checkpoint first-order diagnostics, NOT actual Adam or causality.
New DEV7/8 scenes worsen;5%-clipping stratum contributes92.3% of net H regression.
Old14 highlight-bearing images improve mainly in2 scenes;10 no-highlight excluded.
Next bounded TRAIN pilot: test all-three-objective detail-protecting update directions
and actual finite-step changes, including stalled/conflicting cases; no blind reweight.
8 tests pass; all jobs terminal. No optimization/default change; goal incomplete.
Evidence: docs/production/ACES_GRADIENT_ALIGNMENT_2026-10-24.md.
Root G:/rudra_training/aces_gradient_alignment_20261024.

**Latest — active-union600 complete, rejected (24 Oct):**
Matched600 optimization/evaluation complete; same sampling/exclusions, frozen parent.
Hunyuan48 improves35.12% from0.066582700 to0.043198491EV but fails0.02; other7 pass.
New H worsens0.310286980->0.342798894; clipped detail0.081728859->0.088140910 fails.
Old H improves0.348914478->0.326939908; old overall/clipped detail and0EV grade fail.
Aligned0.008791357 passes. All120 outside-full exact;96-frame routing audit passes.
302 finite component-gradient logs: temporal/(photo norm+detail norm) median0.02871,
max0.94730. Norms do not reveal vector alignment/cancellation. Shared route fails.
Next diagnose temporal-versus-photo/detail gradient alignment on fixed TRAIN pairs
and scene/clipping-geometry regressions before further optimization; no weight guess.
12 tests pass. All jobs terminal; no promotion/default or actual export change.
Root G:/rudra_training/aces_full_active600_20261024. Goal incomplete.
Evidence: docs/production/ACES_FULL_ACTIVE600_2026-10-24.md.

**Latest — temporal support audit complete (24 Oct):**
All192 TRAIN pairs/96 groups audited; eight actual parameter-gradient comparisons.
Active/valid fraction median0.63937, mean0.59403, minimum0.00573; pooled0.30924.
Frozen-only residuals dilute temporal gradients unevenly; measured scaling identity
holds within1.71e-6. Most diluted selected gradient grows174.5x with active normalization.
Entry/exit footprints18.12% of active residuals; two pairs lack aligned transitions.
Fixed-coordinate mask changes occur in all192 pairs and are larger on average than
Hunyuan48, so simple transition scarcity is unsupported; real-video coverage unproven.
Next register matched active-union normalization, preserving transition footprints
and all quality gates; log gradient balance given potentially large scale changes.
No optimization/default change. All jobs terminal,10 tests pass. Goal incomplete.
Evidence: docs/production/ACES_TEMPORAL_SUPPORT_2026-10-24.md.
Root G:/rudra_training/aces_temporal_support_20261024.

**Latest — full-temporal600 complete, rejected (24 Oct):**
Attribution found substantial mask and field contributions on eligible fixed cells;
a regression exposed newly clipped regions missed by full-only temporal support.
Matched600 experiment expanded temporal support on composed isolated output.
Same parent/seed/sampling; parent frozen and all120 outside-full predictions exact.
New H0.310286980, fully-clipped detail0.081728859 still fail; aligned0.008990677 passes.
Hunyuan48 flicker worsens0.061841547->0.066582700EV (limit0.02), other7checks pass.
Old clipped detail and0EV grade fail; -3/+3EV pass. Shared routing also fails.
96-frame routing audit verifies identical full predictions and parent isolation.
All jobs terminal,12 focused tests pass. No promotion/default or actual export change.
Next audit temporal TRAIN support/gradient scale and clipping-entry/exit coverage
before another registered objective/data change. All-highlight normalization includes
frozen partial pixels, so support coverage and gradient dilution are not separated.
Evidence: docs/production/ACES_FLICKER_ATTRIBUTION_2026-10-24.md and
 docs/production/ACES_FULL_TEMPORAL600_2026-10-24.md.
Root G:/rudra_training/aces_full_temporal600_20261024. Goal incomplete.

**Latest — multi-scene full-context600 complete, rejected (24 Oct):**
All training/evaluation/audit jobs terminal. Root G:/rudra_training/aces_full_context600_20261024.
Isolated checkpoint8f4a788d... and shared_control7a7da720... use identical learned
branch weights from one600-update optimization; not independent runs. Parent frozen.
Isolated all120 raw outside-full exact; coherent96 full predictions match shared exactly.
New H0.313047858 worsens parent constrained0.308351923; old H0.343326694 improves.
New fully-clipped detail0.085965048 fails analytic0.059186345 and regresses parent.
Hunyuan48 temporal FAIL0.061841547EV versus0.02 limit(parent0.014813336); other7checks pass.
Old fully-clipped/0EV grade fail; -3/+3EV pass. Shared route is worse broadly.
Fixed48 TRAIN full fit improves9.8% motion/14.4% replay, so fitting improvement does
not transfer reliably to new scenes/video. No promotion/default change or actual export.
Next attribute flicker to full-support transitions versus learned-field variation on
fixed Hunyuan48/known-motion TRAIN before a boundary/temporal change; no blind longer run.
16 focused tests pass; tools compile. Main goal incomplete.
Evidence: docs/production/ACES_FULL_CONTEXT600_2026-10-24.md.

**Latest — isolated full-branch TRAIN pilot complete (24 Oct):**
Experimental v9 full_context branch implemented; initially bit-exact to v8, parent
frozen and outside-full output exact after training. Three600-step arms complete
at G:/rudra_training/aces_full_branch_pilot_20261024. No saved inference checkpoint,
no promotion/default change. Same TRAIN urban_courtyard group; not generalization.
Full fit0.184954->0.00266472 (98.56% improvement); phase probe0.00659787.
Shared-full and isolated-full predictions identical inside full support. Shared-full
severely damages partial output; isolation preserves it exactly. Isolated boundary
0.103958 and high temporal0.009066 improve versus initial0.161025/0.011589, but
shared-balanced is better on overall/partial/boundary metrics. No universal win claim.
Next bounded multi-scene TRAIN isolated-full comparison with shared-full control,
then unchanged coherent DEV/detail/grade/video and source-appearance gates.
Hard support introduces boundary/temporal risk; no quality acceptance yet.
21 targeted tests pass (9 new/related,12 existing); modules compile. All jobs terminal.
Evidence: docs/production/ACES_FULL_BRANCH_PILOT_2026-10-24.md.
Main goal incomplete; remaining checklist unchanged.

**Latest — highlight-region diagnosis complete (24 Oct):**
All jobs terminal; no training/default changes. Fixed48 TRAIN +96 coherent DEV
strata audit completed. New full-deep error improves0.394456->0.319450 but still
fails analytic0.205912; partial error worsens0.142379->0.162722. Old full-deep
worsens1.51415->1.67243. New/old deep residual bias has opposite signs; no blanket
brightness reduction justified. Fully clipped raw/constrained nearly identical.
TRAIN-only gradient audit: opposing partial/full photo+detail gradients in16/32
motion and5/8 eligible replay crops. Local diagnostic, not causal proof.
Next bounded change: separate fully-clipped branch TRAIN fitting pilot, initially
identical output, frozen ordinary/shadow/partial route, matched shared control.
Require boundary/temporal gates; do not infer quality or promotion from isolation.
No separate branch implemented/trained yet.2 diagnostic tests pass; tools compile.
Evidence: docs/production/ACES_HIGHLIGHT_STRATA_2026-10-24.md.
Reports: G:/rudra_training/aces_highlight_strata_20261024/report.json and
G:/rudra_training/aces_region_gradients_20261024/report.json. Main goal incomplete.

**Latest — controlled sampler-v2 cycle complete, rejected for adoption (24 Oct):**
All jobs terminal. Registered6000-step sampler-only comparison completed; parent
weights exact and recorded visits match the zero-empty-update audit. Checkpoint:
G:/rudra_training/aces_context_sampler_v2_6000_20261024/step6000.pt,
SHA256 ac63918fd97fa94ab9f7febcb6219935f13391c084ae73a85c0bf40823e277ec.
New72 H0.308351923 improves2.29% over prior1e-4 but fails analytic0.294311011.
Old24 H0.362807529 passes fixed cap but worsens4.59% versus matched control.
New fully-clipped detail0.065115801 still fails analytic0.059186345; old detail and
0EV grade also fail. Aligned residual0.010005866 passes its fixed cap; Hunyuan48
all eight checks pass at0.014813336EV. No model/default promotion or new export.
Next isolate fully-clipped interior/boundary error and old-scene regressions on
fixed TRAIN/development strata before another registered model/objective experiment.
Keep sampler-v2 opt-in; it fixes empty supervision but does not qualify the model.
Evidence: docs/production/ACES_SAMPLER_V2_2026-10-24.md.
Remaining requirements: docs/production/GOAL_REMAINING_2026-10-24.md.
Concrete input gap: research video reader uses8-bit RGB;16-bit output is not proof
of10-bit source precision preservation. Main goal remains incomplete.

**Latest — matched rate trial rejected; replay sampler fixed (24 Oct):**
All jobs from this cycle are terminal. Single-pair fitting succeeds at both rates;
1e-4 fits faster. Broad matched6000 at1e-4 improves average error and motion, but
fails new-scene analytic accuracy and fully-clipped detail. No model/default promotion.
Checkpoint G:/rudra_training/aces_context_lr1e4_6000_20261024/step6000.pt,
SHA256 0cf6220c2cba5d12a6fdb2e29f3637cefab17742ee45da4f8fb21cc7fd7882c2.
New72 H0.315573809 versus analytic0.294311011; old24 H0.346886636 passes.
Aligned residual0.009710400 passes; new fully-clipped gradient0.077681369
versus analytic0.059186345 fails. Hunyuan48 passes at0.015537215EV.
Old coherent fully-clipped detail and0/+3EV grade also fail; legacy retained.

Exact legacy sampler replay found61 empty updates. Opt-in nonempty_phase_aligned_v2
fixes interior support and odd-size image-edge handling; its6000-decision audit
has zero empty updates, excludes one border-only replay row. Sampling trajectory
changes; this is not a matched model result. Crop-first preparation preserves values.
15 focused tests pass. Next register a controlled sampler-v2 training comparison
with fixed parent/rate/budget and the existing quality gates; v2 has NOT been trained.
Do not infer model improvement from sampler coverage or promote the higher-rate model.
Details: docs/production/ACES_FIT_SANITY_2026-10-24.md. Main goal incomplete.

**Latest — balanced context and de-banding cycle complete (24 Oct):**
All training, evaluation, export and audit jobs launched in this cycle are terminal.
No model/export default promoted. Main goal remains incomplete. Saved goal status
is usageLimited; automatic continuation is not active. No status override was made.

Balanced1800 and further6000-step v8 runs complete/rejected. Latest checkpoint:
G:/rudra_training/aces_context_balanced6000_20261024/step6000.pt,
SHA256 2a2a9d7ca3ad28489193458e443fe4d21349b41ba88bd1c7af91c336ae0286c8.
New72 H0.322709585 fails analytic0.294311011; old coherent24 H0.366255403 passes
fixed0.382524861 control. Aligned motion residual and fully-clipped detail fail;
old-3EV grade improves, but0/+3EV fail. Legacy H0.851267627 retained as failure.
Frozen weights/outside-highlight output exact; all-highlight colour and sign pass.
Hunyuan48 passes all checks, max0.019034568EV versus0.02limit. Four-frame real
float32EXR/16-bitPQ smoke independently verified exact, including tags/frame count.
Delivery verifier now accepts registered clip sets/counts instead of assuming18.
Earlier equal-head18-clip suite:17/18 quality passes,18/18 delivery passes,826frames;
70 exact EXR/PQ samples. Sampled81,799,936pixels have zero >1-code appearance errors.

Separate de-banding research: unrestricted bounded curvature fails authored
texture/edge and real48-crop checks. Source-coherence guard passes authored48-frame
checks but still slightly worsens real-image-3EV and old-set code errors. Rejected;
no integration. Solver/guard/motion/real reports and snapshots preserved.
38 combined regression tests pass; changed tools compile and whitespace check passes.
Declared239-row receipt audit passed; new40-scene CC0 corpus built; full historical
weight-origin/distribution audit still open. No external weights/code imported.

Next technical work: fixed TRAIN-subset/scene fitting sanity check and per-scene
residual/target analysis to distinguish capacity, optimization and ambiguity before
more broad training. Longer same-recipe runs have not solved new-scene recovery.
For de-banding, require real reference-aware grade/detail and motion preservation;
do not choose a threshold solely to pass the studio example or reused patterns.
Evidence: docs/production/ACES_CONTEXT_MOTION_2026-10-24.md,
docs/production/ACES_CURVATURE_DEQUANT_2026-10-24.md, and
ACES_REAL_EXPORT_DETAIL_FOLLOWUP_2026-10-24.md. Earlier entries below are historical.

**In progress — real export and detail-loss follow-up (24 Oct):**
Constrained equal-image real export suite and independent delivery verifier running
at G:/rudra_training/aces_constrained_generated_review_20261024 (18 registered clips).
Coherent detail/grade audit complete: highlight gradient RMSE improves9.93%,
but fully-clipped detail0.55% worse and0/+3EV view-error checks fail. -3EV improves.
Evidence: aces_equal_head_colour_backtrack_v2_20261024/detail_grade.json.
Neutral ramps retain small steps/ripples; restorer unchanged. No release promotion.
Registered gradient-loss head600 trained on same corrected CC0 TRAIN and fixed
seed/sampler; evaluation running in aces_consistent_gradient_head600_v2_20261024.
Do not launch overlapping runs or overwrite these output roots. Main goal incomplete.

**Latest — constrained equal-image head passes coherent/motion diagnostics (24 Oct):**
Registered explicit existing colour backtracking on corrected equal-image head
4904f8c3...; no model/export default promoted. Coherent development24 all checks
pass: highlight MSE0.3693457005 versus parent0.3825248607 and analytic0.4204336752;
overall7.308% better than analytic. Partial-highlight appearance <=1code everywhere,
ordinary/shadow/fully-clipped regions unchanged. Legacy development still FAILS:
highlight0.9106586410, overall17.906% worse than analytic. Retain that limitation.
384 procedural motion frames all registered checks pass; worst transition mean
excess0.00527555065 EV versus0.02 limit, zero new negativeY/shadow lift/colour errors.
Pure-blue stress contact inspected; centre max error0.6221codes, no former cyan drift.
Evidence and exact paths: docs/production/ACES_CONSTRAINED_HEAD_2026-10-24.md.
All jobs terminal.13 model/shadow/constraint tests plus5 temporal/development/
constraint tests pass (overlap); changed tools compile. No release authorization.
Next: real generated-clip colour/motion and grading/detail evaluation of constrained
candidate; discrete alpha/chromatic flicker and legacy target conflict remain open.
Main goal incomplete. Previous summary below remains historical context.

**Latest — generated review complete; corrected highlight pair rejected (24 Oct):**
All work launched in this session is terminal; no training/export/audit jobs remain.
V7 generated review:18/18 registered passes,826 frames. Independent18-video
verification passes;70 EXR samples exactly match decoded16-bit PQ codes.
Worst temporal transition0.01988959 EV (Hunyuan demo, near0.02 limit).
Separate sampled colour audit:793978/1078418 partial-highlight pixels exceed
one code (73.62% of that support, about0.97% of all sampled pixels); max105codes.
Thus export transport passes but highlight appearance still fails product intent.
Evidence: G:/rudra_training/aces_v1_generated_review_20261024/{summary,
delivery_verification,colour_audit}.json; selected real crop in cogvideox_res2/.

Only corrected corpus aces_consistent_exposures_v2_20261024 is valid. Earlier
non-v2 coherent corpora/trials remain explicitly INVALID (storage encoding).
Corrected pooled/equal-image head600 trials complete: equal-image highlight MSE
0.38252486 vs v7 0.45382048 and analytic0.42043368 on coherent development
(15.71%/9.02% improvements), but only26.21% partial highlights meet1code;
legacy natural error regresses. Both trials FAIL; no model/default promoted.
Frozen weights and outside-highlight predictions bit-exact; matched comparison
integrity verified. Equal candidate4904f8c3... is research-only, not a replacement.
Full evidence: docs/production/ACES_CONSISTENT_HEAD_2026-10-24.md and
G:/rudra_training/aces_consistent_head_equal600_v2_20261024/paired_comparison.json.

Retained fixes: capped decoder/empty-output handling, exact suite/input/artifact
verification, cached inverse reuse and bit-exact threaded OCIO export.41 focused
regression tests pass. Export guide now discloses colour and temporal limits.
Next: register an explicit highlight colour constraint and resolve legacy target
appearance conflict before further training; preserve ordinary/shadow behaviour.
Then grading/detail, temporal/platform breadth, native integration and independent
release/licensing checks. Main goal remains incomplete; do not mark complete.

**Correction — first coherent corpora and derived head trials INVALID (24 Oct):**
The new builder mistakenly saved Rec709 RGB directly while loaders expect
canonical Rec2020 storage. Both aces_consistent_pairs_20261024 and
aces_consistent_exposures_20261024 have BUILD_INVALID.json and build_pass=false.
Both head600/equal600 trials and backtrack_consistent diagnostic carry
EXPERIMENT_INVALID.json; their quality comparisons MUST NOT be interpreted.
The original v7 generated review, natural target-colour audit, original natural
backtracking comparison and export parity checks are unaffected.
Builder fixed to canonical(source)->uint16 PNG->actual training-loader read-back
before target creation; saturated-colour storage regression test passes.
Corrected exposure corpus aces_consistent_exposures_v2_20261024 rebuilding
(session41709). Future head trials use new v2 roots and verify source-support
identity, saved target/source display agreement and initial cached-head parity.
Generated review runner30948 and per-clip independent verifier51089 remain live.
No model promoted. This correction supersedes all coherent-corpus/trial claims below.

**Latest — coherent highlight-head trial evaluating; 9/18 generated clips pass so far (24 Oct):**
Generated export runner session30948 remains live. Root:
G:/rudra_training/aces_v1_generated_review_20261024. No failed completed clips.
Backtracking diagnostic complete and rejected: colour drift fixed but old natural
highlight MSE +0.84% versus v7. Target audit finds96.46% of partial-highlight
pixels drift >1code (max125.06); docs/production/ACES_GENERATED_COLOUR_REVIEW_2026-10-24.md.
Separate exact-view CC0 corpus built72 rows; exposure-expanded corpus87 rows
(TRAIN63/dev24) passes7 construction checks,798017/570312 highlight pixels.
Only exposure-expanded TRAIN used for registered final-head600 trial;16 eligible
scenes, all non-highlight-output weights bit-exact. Checkpoint:
G:/rudra_training/aces_consistent_head600_20261024/step600.pt
sha9a6630d3023335c16848108fb9ce4f3575b65c952a294fb0cdb0b836b24e8aef.
Paired original/new24-frame evaluations now running(session89804).
No new model promoted. Candidate source ancestry merged and hashed.
Export performance: reuse restored inverse (bit-exact inference test), opt-in
CPU OCIO threads (export defaults4, reference default1); odd-size signed/HDR
parity tests pass.19 model/export tests and20 view/video tests passed. New export
reports record implementation hashes. No native or release claim.

**Latest — generated review running; partial-channel highlight colour defect measured (24 Oct, Codex):**
Actual v7 checkpoint/hash and seven acceptance evidence hashes verified; 10 focused
local tests pass. Registered 18-input review is running via PowerShell runner
(exec session30948). Four CogVideoX clips complete and pass; remaining clips pending.
Review now requires exact input inventory/hashes, correct capped frame counts,
checkpoint identity and output hashes. Two FastVideo inputs are only29 frames;
all others use48. Capped decoder and empty-output handling corrected.
Independent first-clip decode:48 frames, gbrp16le/full-range/PQ tags and four EXR
sample transforms bit-exact. Tool: tools/verify_aces_generated_delivery.py.
New colour stress diagnostic: pure blue acquires ~75 green code values at0EV,
which ordinary-only R hides because the whole patch is marked reconstructed.
Evidence G:/rudra_training/aces_v1_colour_extremes_20261024/report.json (+contacts).
A separately registered partial-highlight backtracking diagnostic is running
(session71571; configs/experiments/aces_colour_backtrack_20261024.json).
No model/default promoted. Product grading/detail/native/release goals remain open.

**Latest — v1 export workflow built; generated-clip review ready (24 Oct):**
tools/rudra_aces_export.py: still/folder/video -> per-frame float32 ACEScg EXR (AP1
chromaticities, rudra.reconstructed = highlight reconstruction weight, provenance,
bit-exact read-back) + optional ACES2 HDR1000 PQ FFV1 gbrp16le .mkv + report with
appearance/shadow/negativeY/temporal checks and first-frame contact sheet. Guide:
docs/RUDRA_V1_ACES_EXPORT.md. Tested on cloud CPU (OCIO 2.6 copy; contract pins
OCIO 2.5.2 so the real checkpoint is refused under a different OCIO, by design).
Registered generated-clip review configs/experiments/aces_v1_generated_review_
20261024.json; run scripts/run_aces_v1_generated_review.ps1 (18 clips x48 frames)
-> G:/rudra_training/aces_v1_generated_review_20261024/summary.json.
Known: fully saturated primaries at code 255 enter highlight reconstruction.

**Latest — v1 analytic-shadow candidate PASSES registered acceptance 12/12 (24 Oct):**
G:/rudra_training/aces_v1_analytic_shadow_20261024/candidate.pt (sha 63e76ee2...).
Removes dark-motion negative luminance (coverage4800 66,646/74,211 px -> 0/0),
lowest dark flicker so far (.00208/.00132 EV), zero shadow lift, outside-shadow
bit-exact to coverage4800, highlights unchanged; natural shadow MSE +1.0% vs
coverage4800 (disclosed, within seed spread). Development candidate, not release.
Next: export workflow for this model (EXR32+mask, RGB16 video), then native app.
Details: docs/production/ACES_V1_ANALYTIC_SHADOW_2026-10-24.md.

**Latest — owner decision: v1 uses analytic shadows; candidate ready to evaluate (24 Oct):**
training/aces_model.py gains shadow='analytic' (contract v7): in source crushed-
shadow support, w*exact source inverse + (1-w)*inverse of bounded restored source;
bit-exact analytic where w==1, outside support unchanged. No training: candidate =
coverage4800 weights + new contract (tools/derive_aces_analytic_shadow.py).
Registered configs/experiments/aces_analytic_shadow_20261024.json; acceptance
tools/compare_aces_analytic_shadow.py. 21 tests pass (cloud CPU, torch 2.14).
Run scripts/run_aces_analytic_shadow.ps1 -> G:/rudra_training/
aces_v1_analytic_shadow_20261024/acceptance.json. Expected cost: natural shadow MSE
= analytic (~2% above single-seed coverage4800, inside +-4% seed spread).

**Latest — 5-seed study: edge shadow supervision has no effect; learned shadows unreliable (24 Oct):**
Natural shadow MSE mean control 6.572e-7 (CV .6%) vs edge 6.571e-7 (CV3.1%);
edge wins 3/5. Seed-20261023 gain was luck. 0/10 runs pass all candidate gates
(sporadic lift, motion lift, one negativeY). Edge spread +-4% means earlier
few-percent single-seed shadow verdicts were noise. No trained shadow branch is
reliably better than the analytic inverse. Recommendation awaiting owner: v1 uses
analytic inverse in crushed shadows (marked not reconstructed) with learned
restorer/highlights; validate as registered candidate. Details:
docs/production/ACES_SHADOW_CONTEXT_2026-10-23.md.

**Latest — edge shadow trial FAIL (narrow); 5-seed study registered (24 Oct):**
aces_shadow_local_edges1200_20261024: natural shadow MSE 6.274e-7, first below
coverage4800 (6.447e-7), but only 4.81% below control (need 5%), 3 lifted px
(max1.33 code), original-dev +0.09%, dark motion means .00447/.00376 vs control
.00389/.00326 EV (all <<.02). Registered FAIL, no promotion. Natural shadow metric
= 7 frames/3 scenes, so seed noise is unmeasured. Next: run
scripts/run_aces_shadow_edge_seeds.ps1 (seeds 1-4 both arms, ~1h, resumable) ->
G:/rudra_training/aces_shadow_edge_seeds_20261024_summary.json. Registration
configs/experiments/aces_shadow_edge_seeds_20261024.json. Details:
docs/production/ACES_SHADOW_CONTEXT_2026-10-23.md.

**Latest — shadow context trial FAIL; edge-supervision trial prepared (24 Oct):**
Codex ran matched local(radius8) vs wide(radius40) shadow branches on mixed144
(natural48+original96), frozen coverage4800, 1200 steps; then hit its usage limit
(locked until 29 Oct) before recording. Wide FAIL: worse original-dev shadow and
dark compressed motion .03054EV. Local most stable so far (.00800EV) but natural
shadow MSE worse than coverage4800. No promotion. Exact numbers:
G:/rudra_training/aces_shadow_mixed_wide1200_20261023/context_acceptance.json.
Found 4/21 shadow-eligible natural TRAIN images supervised nowhere (shadow only
inside the 40px border); --include-image-edges fix tested, not yet trained.
Registered next trial configs/experiments/aces_shadow_edge_20261024.json; run
scripts/run_aces_shadow_edge_trial.ps1 (train+3 evals+paired report). the assistant
session has no GPU/Windows access: owner launches. Best restoration baseline
remains coverage4800. Details: docs/production/ACES_SHADOW_CONTEXT_2026-10-23.md.

**Latest — quantized consistency FAIL; shadow data and local-context limit found:**
Added opt-in --consistency-quantize, defaultfalse. Same600-step/weight10
experiment with translated inputs rounded8bit; six consistency/shadow testsPASS.
Candidate G:/rudra_training/aces_shadow_quantized600_20261023/step600.pt SHA
d8b78e8fd4f1e597b5da19f29fcf83557d3fba75f2962c53968eb3b2a1dcacdf.
Natural shadowMSE6.59080648e-7 worse than both matchedcontinuation and prior
consistency andcoverage4800. Dark raw/codec meanEV.012427/.012825, worst
.022480/.025350; both exceed.02. Full384 zero lift, one negativeY observation
-3.12e-12 from float cancellation atdarkraw frame3(96,138). StrictFAIL retained,
no tolerance/clamp/promotion. Quantization alone is not supported as a fix.
Built original known-linear shadow96TRAIN24dev v2, not optimized on:
G:/datasets/corpora/aces_original_shadow_v2_20261023/train/report.json SHA
0ce2d28b6d4e5a3f237aba6760ac0c5abdc88d9daed8436f2d22ceea621131f8.
4,896,600 TRAINshadow pixels/2,664,200 blacksource; maxR.504120code, splits
scene/source/target-disjoint, noexternalmedia. Two generation tests pass.
First aces_original_shadow_20261023 build is incomplete/unusable, marked
BUILD_FAILED.json after scene49 lacked shadow support; v2 fixes field range.
TRAIN observabilityaudit:2,050,322 allblack17x17 cores. Current shadowradius8
has identical local inputs there despite varying targets; constantprediction
scaledRGBasinhMSE lowerbound.000373669. Wider context required to use surviving
surrounding cues; fully lost information remains ambiguous. Contact inspected.
Next wider-context shadow branch plus balanced original/natural TRAIN; keep
all shadow/temporal/visual gates and fullproductscope. All jobs94220/74137/
31817/68345 terminal, no livejobs. Details:
docs/production/ACES_SHADOW_QUANTIZATION_2026-10-23.md. Goal unfinished.

**Latest — shadow TRAIN audit and paired consistency fine-tuning FAIL:**
audit_aces_shadow_fit.py fixed168 TRAIN crops (21images x8) shows v5 improves
TRAIN brightness/chroma but not development: scaledRGB .016153 vs v4.023598,
luma .006005 vs.011222, chroma .022122 vs.028341. Generalization investigation.
Read primary EilertsenCVPR2019 temporal regularization. Added signed-asinh
error-consistency under subpixel CPU translations, common shadow support.
Matched fine-tune600 fromsqrt1200, freshAdam.0001, consistency0 vs10; same
sampling visits, frozencoverage4800 weights. Five focused tests pass.
Control G:/rudra_training/aces_shadow_continue600_20261023 SHAstep600
823dab31e71240496f870913fdf894d1387779504602ea586535a29dfd2727d1;
treatment aces_shadow_consistency600_20261023 SHAstep600
c11b637e072a59ed74a5370cd79b0cfb87f525f7dac599dabfeacf127c4b56f0.
Natural shadowMSE6.5522753e-7 control/6.5361635e-7 treatment, both beat
analytic but not coverage4800(6.4474825e-7). Five gatesPASS, stricterFAIL.
Full384 negativeY/lift0 each; treatment dark means .010898/.011226 worse
than control.010026/.010288. Worst darkcodec .021460 vscontrol.020221,
both exceed.02. paired_acceptance.json FAIL; no promotion. Graded bands persist.
Quantizationprobe one TRAINcrop: fraction>.1code frominteger0% source vs
67.472% bilinearshift. Input distribution mismatch verified, causal link NOT.
Next quantization-aware TRAIN motion simulation/regularization; preserve all
shadow/temporal/visual requirements. No external code/weights/data imported.
All jobs18553/2199/41423/46189/91418/46395/22011 terminal, none live. Details:
docs/production/ACES_SHADOW_CONSISTENCY_2026-10-23.md. Full goal unfinished.

**Latest — isolated v5 shadow representation removes negativeY but trial FAIL:**
New training/aces_shadow.py and opt-in ACESModel shadow='sqrt_luma':
square-root luminance residual plus signed R-Y/B-Y, source shadow support only.
No RGB clamp/outputfloor/radianceceiling; zero derivative at exactblack/zerofield
is a known learnability limitation. Default models/contracts remain unchanged.
Fresh1200 shadow-only Adam.0001, natural TRAIN48 (21shadow-eligible), targeted
crops, normalized signedasinh loss. coverage4800 predictor/restorer frozenbitexact.
Candidate G:/rudra_training/aces_shadow_sqrt1200_20261023/step1200.pt SHA
d7afafe193af66c395e2e8b5f00c3b1d40f55f84ce82fcb4b19d92eb01091073.
Nine tests pass. Natural24 shadowMSE6.69620947e-7 vscontrol6.44748247e-7
(3.858%worse), vsanalytic6.59086168e-7 (1.598%worse): trialFAIL.
Other naturalgatesPASS,zero lift; hashverified full24 outside-shadow outputs
bitexact tocoverage4800. All384motion frames zero negativeY/zero lift/R100%.
BUT worsttransitionmeanexcessEV bright raw.000676622,bright codec.004862556,
dark raw.018966367,dark codec.020932958 exceeds.02 target. Darkexcludedcells
1235/1474; nonnegative model does not eliminate common-positive support gaps.
Inspected0/+3/+6EV shadow contact retains bands vs smooth truth. No promotion.
shadow_trial_acceptance.json FAIL includes full-motion adverse evidence.
Retain coverage4800 as restoration baseline; its shared-shadow defect stillopen.
Next shadow supervision/generalization/temporal work; not threshold relaxation.
All jobs38394/13540/64044/11838/88753 terminal, no livejobs. Details:
docs/production/ACES_SHADOW_SQRT_2026-10-23.md. Full goal remains unfinished.

**Latest — combined coverage4800 passes registered restoration development:**
TRAIN176 natural48+legacy procedural32+balanced96, explicit batch quotas2/1/1;
fresh4800 Adam.001 gradientweight1, seed20261023, frozen lower_lr300 HDR.
All176 visited,9600/4800/4800 crops. Six tests pass, frozen predictor bitexact.
Candidate G:/rudra_training/aces_coverage_restorer4800_20261023/step4800.pt SHA
10ea7e9c7e715284f11063098254572402e005f8d00d62f19f4de2f4cccba79e.
Natural24 ordinarycodeMSE .13402769 (13.059%betteranalytic), legacy8 .00934152
(88.640%better), both beat gradient1200. Balanced24 supplemental .02026255
(76.160%better). Original five natural gatesPASS,zero lift,H/S unchanged.
All9 ramp gradient reductions56.2–81.1% versus mixed1200. Registered
coverage_trial_acceptance.json PASS, not visual/release. Plot retains steps
and ripples. Correction-support probe maxima .428/.386/.367code, no sampled
corrections above.45code; errors not explained simply by hitting.49 bound.
Data/sampling/duration changed together; not an isolated causal experiment.
Use this as restoration research candidate, keep production defaults unchanged.
Next separately address negative shadow luminance and HDR detail while retaining
restoration checks; grading/platform/temporal/native/independent release remain.
Details docs/production/ACES_COVERAGE_RESTORER_2026-10-23.md.
Temporal18 completed: allR100%,zero lift/newnegativeY/excludedcells.
Worsttransitionmean cell excessEV .001846724/.000410060, higher than v3
but below.02 in this limited fixed-cell diagnostic; not releaseT.
All jobs56577/39834/27554/56378/28899 terminal; no live jobs.

**Latest — balanced data fixes ramp metric gap but regresses code fidelity:**
Fresh1200 frozen-HDR local restorer, gradient weight1, explicit two natural
and two procedural crops per batch. New TRAIN144 manifest natural48+balanced96,
all visited,2400 crops each domain. Five tests pass, frozen predictor bitexact.
Candidate G:/rudra_training/aces_balanced_restorer1200_20261023/step1200.pt SHA
2a7960c08d8c887d58f62c7f85deed04472e5875989ee4b335ff9508a84901c0.
Ramp gradient reductions versus mixed1200: broad81.546/80.297/78.947%,
dark68.188/68.018/67.265%, bright68.827/68.854/69.247% at-3/0/+3EV.
BUT natural codeMSE .14393650 and old procedural8 .04064778 both worse than
gradient1200 (.13533504/.01602599). Registered balanced trial FAIL, no promotion.
Natural five gates pass, zero lift; frozen H/S unchanged. Supplemental balanced24
codeMSE .02253464,73.486% betteranalytic. Plot still has dark steps/bright ripples.
New compare_aces_balanced_trial.py records stricter result with source hashes.
Next retain older coloured procedural coverage explicitly alongside new groups;
current experiment changes data and sampling together, not causal isolation.
Separate shadow luminance/HDR-detail fixes and platform/native/release remain.
All jobs52787/31432/90348/11322 terminal. Details:
docs/production/ACES_BALANCED_RESTORER_2026-10-23.md. Goal unfinished.

**Latest — gradient supervision improves both domains; balanced data gap fixed:**
train_aces_restorer.py optional --gradient-weight default0, ordinary-endpoint
target derivativeMSE in8bit units. Fresh1200 weight1 on same mixed80, frozenHDR.
Candidate G:/rudra_training/aces_restorer_gradient1200_20261023/step1200.pt SHA
b69a82a38cf96a24d58a43187d35d259e34562fc02c044b0442de68b366471a3.
Natural24 codeMSE.13533504 (12.211%betteranalytic), procedural8 .01602599
(80.511%better). Both improve over mixed1200. Five natural/base restoration
checksPASS; H/S unchanged,zero shadowlift. Stricter registered gradient trial
FAIL: broad reductions35.85/48.42/50.08%, dark29.48/29.02/27.52%, bright
17.78/21.15/22.74% at-3/0/+3EV misses required25%each. No relaxation/promotion.
gradient_trial_acceptance.json separate from base restoration_acceptance.json.
New plot_aces_grade_ramps.py hashes inputs/plot; inspectedplot stillhassteps.
Original proceduralTRAIN coverage audit finds0neutral,0all-dark,0all-bright
among2,097,152pixels. Added opt-in balanced generator mode (legacydefaultsame).
New G:/datasets/corpora/aces_procedural_balanced_20261023:96TRAIN24dev, seeds
20261027/28, original media only, scene/contentdisjoint, maxR.497786code.
TRAIN reportSHAb1c1c5c365835c5c9c9abd2a55f693140dc7327a6eed3dc720fe24cdc23c4359.
Counts3,493,408neutral/1,359,336darkneutral/1,133,305brightneutral of6,291,456.
Not used for optimization yet; originaldata intact, syntheticdev not release.
Natural TRAIN shadow-target audit:35658pixels,35607positiveY/51zero/0negative.
Report mixed_restorer1200/train_shadow_target_luminance.json. Supports separate
nonnegative-luminance shadow learning, NOT signed-RGB clamping.
Four relevant tests pass; jobs93271/56278/49153/21643/80299 allterminal.
Next balanced-data restoration with controlled domain sampling, plus separately
fix shadow luminance. HDR-detail/platform/native/independent release stillopen.
Details docs/production/ACES_RESTORATION_GRADIENT_2026-10-23.md.

**Latest — mixed1200 restoration passes development; new motion reveals shadow defect:**
Mixed80 TRAIN manifest48natural+32procedural under G:/datasets/corpora/
aces_mixed_restoration80_20261023/report.json. Trainer masks ordinary pixels,
skips empty support, safely deduplicates matching ancestry, rejects conflicts.
--steps now configurable default300; HDR predictor frozen bitexact throughout.
Mixed300 naturalcodeMSE4.273%better but procedural3.120%worse -> combinedFAIL.
Registered fresh1200 same recipe otherwise, finalonly. Candidate:
G:/rudra_training/aces_mixed_restorer1200_20261023/step1200.pt SHA
e9e970fe307daf6224c3982f2abbcb9420d4e59545f2a6c1564574defd9c39e2.
Natural24 ordinarycodeMSE .13972373 vsanalytic.15415941 (9.364%better);
procedural8 .02765991 vs.08223222 (66.364%better). Natural overall7.872%better,
five gatesPASS,H/S same frozenbase,Rpass,zero shadowlift. Registered combined
restoration_acceptance.json PASS via new compare_aces_restoration.py.
BUT inspected rampplot retains steps/ripples; tiny plateau fractions .894%/
1.564%/0% can reflect waviness, NOT98%visual debanding or Gpass. No release.
New original analytic object-motion dataset: aces_object_motion_20261023,
two96frame4s384x216 scenes + actual H264CRF28 variants, float32AP1 references;
576 unique frame/reference hashes verified, contact inspected. No external
media, no training use, not real/generator/independent-release coverage.
Lower_lr300 evaluated all384frames in sibling object_motion384: R100%,zero
lift; worsttransitionmean excessEV .000095/.000396/.002549/.004485 across
bright/dark raw/codec. Dark variants exclude1625/1521 nonpositive cell samples.
Dark model negativeY observations66646/74211; frame0 model753/849 vsanalytic0
andtruth0, minimumY~-4.5e-5 AP1. Do not hide this by clamping signed RGB.
Evaluator now includes variant/sourcecoverage and future baseline/newnegativeY.
Seven relevant tests pass. Jobs25308/29235/2742/53802/44448/46199/15690/42911
all terminal; no live jobs. Next improve ramp gradients/steps and shadow
luminance behavior; earlier HDR-detail failure and full release scope remain.
Docs: ACES_MIXED_RESTORER_2026-10-23.md and ACES_OBJECT_MOTION_2026-10-23.md
in docs/production. All candidates remain experimental, not production defaults.

**Latest — isolated local restorer learns ramps but regresses natural fidelity:**
New training/aces_restorer.py: source+5/17/65px local code differences, no norm,
small dilated CNN. ACESModel opt-in restoration=local_codes contractv4 replaces
first3 fields only at source weight0; reconstruction regions retain v3 fields.
Default v2/v3 unchanged. model_from_checkpoint validates full contract/state;
image/ramp/temporal evaluators supportv4. New train_aces_restorer.py freezes
lower_lr300 predictor bitexact, trains procedural32 only,300 Adam.001, loss
centre48 of128 crops (40px border). Training ancestryunion80 ensures both
frozen-base and new-data scene separation in evaluation. No verified resume.
Checkpoint G:/rudra_training/aces_local_restorer300_20261023/step300.pt SHA
66a525088b6083fce045c07ed81879a61b4287bb9b62942316dc2d2bd8c574e0.
Full8 procedural codeMSE70.918%betteranalytic. Three0EV ramp plateau fractions
1.788%/38.994%/.894% versus76.648%/97.542%/94.302%; gradient errors improve
alltestedEV. Plot inspected: dark steps/bright ripples remain, no Gpass claim.
Natural24 numerical5gatesPASS, overall7.534%betteranalytic,H/S unchanged,
ordinaryRpass,zero shadowlift. BUT natural ordinary codeMSE .269854 vsanalytic
.154159 (75.049%WORSE). Do NOT promote. restoration_acceptance.json retains
registered numerical probePASS plus additional adverse quality evidence/reject.
Nine branch/model/split tests pass. Jobs68328/73699 terminal; alljobs finished.
Next mixed natural+procedural restoration training with ordinary-only loss,
frozen HDR; require natural code fidelity alongside ramp gains. Frozen HDR
detail failure remains. Full video/platform/native/release scope still open.
See docs/production/ACES_LOCAL_RESTORER_2026-10-23.md for results/paths.

**Latest — direct restoration objective and procedural300 tests fail:**
Added optional restoration_code_loss: ordinary(source weight0) mean RGB MSE
in8bit codes against exact OCIO view(target), fixed target/no target gradient.
Trainer --restoration-weight defaults0. Tests verify units/gradient/mask/empty.
Registered fresh300 Adam.0001 weight1 on TRAIN48, fullCPUdev24:
G:/rudra_training/run05_aces_restoration300_20261023, step300 SHA
68c5b159760027a94c3b9e270b18343e031fadaa1f72490ac151b0c003e7e9d3.
Overall error .009486734 (1.6602%betteranalytic, fails5%gate). H/S/ordinaryR
and zero shadowlift pass. Ordinary codeMSE .1541594->.1490426 (3.319%better),
but all3 ramp0EV plateaus unchanged; restoration_acceptance.json FAIL.
Created original procedural dataset32train/8dev, separate seeds/scenes/hashes,
no external media or public license dedication; bounded target max.492925code.
G:/datasets/corpora/aces_procedural_restoration_20261023/{train,val}/report.json.
Isolated fresh300 Adam.001 weight1 procedural run (not HDR training) completed:
G:/rudra_training/aces_procedural_learnability300_20261023, step300 SHA
08361f875437d4b16834e62e71f1fc6ce4f30905fc0278b486ac2176b6171520.
Full8 codeMSE .08223222analytic->.08227135model (.0476%worse); all3 ramp
plateaus unchanged. Registered learnability diagnostic FAIL. No adoption.
Fixed65px bounded box reference on proceduralTRAIN worsensMSE65.01%; notused.
New tools prepare_aces_restoration_data.py and score_aces_restoration.py.
Four loss/sampler tests pass. Jobs87678/95727/18338/46114/9094 terminal; no jobs.
Next isolate a source-restoration branch with explicit local code-difference
features; current300-step results do not prove fundamental network incapacity.
See docs/production/ACES_RESTORATION_TRIAL_2026-10-23.md. Goal remains unfinished.

**Latest — stronger detail/headroom audit fails the virtual-RAW quality promise:**
New tools/audit_aces_learned_detail.py compares lower_lr300 to original CC0 HDR
on identical positive-luminance support (two-pixel-eroded interiors). Fulldev24
gradient RMSE analytic/target/model: H6eligible .294416/.033120/.297591;
S2eligible .410849/.085208/.529692; fullyclipped6 .106101/.027418/.106172EV.
746 shared shadow pixels excluded,0elsewhere; model does not recover improved
structure despite its five numerical development checks passing. Do NOT promote.
Report lower_lr300/development300/detail_truth.json. Tools hash-verify sources,
targets and predictions; -3/0/+3EV view errors also saved but use experimental
targets, not physical exposure truth. New tools/audit_aces_grade_ramp.py tests
three known smooth neutral ramps. At0EV model/analytic plateau fractions equal:
76.648% broad,97.542% dark,94.302% bright; no meaningful dequantization gain.
Use lower_lr300/grade_ramps_v2/report.json (relative1% reversal/plateau threshold,
max negative code amplitude). Initial grade_ramps report had overly sensitive
absolute reversal statistic; do not call its tiny derivatives visible artifacts.
Two focused support/reference tests pass. Job97762 exited0; no live jobs.
Next controlled experiment: explicit ordinary-pixel restoration supervision
in display-code units, then reconstruction gradient supervision separately.
Keep original appearance/shadow gates and release scope; no threshold relaxation.
Details docs/production/ACES_LEARNED_HEADROOM_2026-10-23.md.

**Latest — ACES temporal18 smoke and learned RGB16 HDR delivery verified:**
tools/evaluate_aces_temporal.py runs lower_lr300 CPU full-frame on existing
run05_temporal_probe_v2_20261023 (2CC0 camera moves,9frames each). AP1 luminance
36x64 cell log2 changes relative to analytic; nonpositive cells excluded/count,
no log floor, no motion alignment. Mean/worst-transition mean excessEV:
abandoned_construction .001506754/.001558774; tank_farm .000003706/.000003895.
All18 ordinaryR100%, zero shadowlift, zero negativeY, zero excluded cells.
This is NOT release T: short synthetic24fps camera moves, no object motion,
codec/VAE/platform/long-sequence coverage. No temporal smoothing used.
Reports/predictions: G:/rudra_training/run05_aces_lower_lr300_20261023/temporal18.
New rudra/delivery/aces_video.py uses official ACES2 HDR1000 Rec2020/PQ view
on signed AP1 directly. video.encode_sequence adds opt-in input_encoding=pq_codes
(PQ target,Rec2020,shoulderFalse required; finite[0,1] validated). Legacy default
linear_nits unchanged. Avoids double tone mapping and PQ decode/re-encode.
tools/verify_aces_video.py exports both learned9frame clips, decodes and checks
against independently quantized OCIO output: max16bit code error0 both. FFV1
gbrp16le,16bits,fullrange,24fps,1280x720,bt2020/smpte2084/gbr tags verified.
Files/report in sibling video16. Derived display HDR, NOT scene-linear master
or native/Resolve/player compatibility proof. Keep float32EXR grading master.
45 tests pass,2 existing delivery skips. First signed-value test expectation
corrected: official HDR view itself may map negative/clipped AP1 identically;
wrapper passthrough verified separately, direct official parity still exact.
Jobs1465/1839/29744 exited0. Next full-resolution grade/detail tests, longer
object-motion/compressed/generated data and native integration; goal unfinished.

**Latest — smaller learning rate passes numerical development gates:**
Fixed TRAIN144-crop audit shows control300 worse than analytic (.038667/.031026)
and worse than control200. Added tools/audit_aces_training_curve.py; same crops
and hashes across snapshots, no optimizer or checkpoint selection. Registered
configs/experiments/aces_lower_lr_20261023.json: fresh300 Adam.0001, otherwise
same deterministic settings, shadow weight0. Trainer now exposes validated
--learning-rate (default.001 unchanged). RNG/contract/recipe match previous
control except lr and trainer argument code hash (comparison_identity.json).
G:/rudra_training/run05_aces_lower_lr300_20261023/step300.pt SHA
2fbcf10fa1cc71e6a4008aba1a4159ae32ecc606b14181a4bee87d2c0817118d.
CPU full24 development300/acceptance.json numericalPASS: overall .008894431
vs analytic .009646895 (7.8001% better), H .696091884 vs .758664747,
S6.447482e-7 vs6.590862e-7. Ordinary R passes, zero shadow lift>1code.
TRAIN fixed-crop overall .03057016 vs analytic .03102557 (modest improvement).
Contact00/06/08/09 reviewed; no large magenta streaks apparent, case9 window
detail remains close to analytic and behind target. visual_review.json partial;
NO release/visual/headroom claim. Five focused tests pass. Jobs23640/67591/
55079/80450 exited0; no live jobs. Next assess full-resolution detail/headroom
and temporal behavior of this candidate, expand commercial coverage/independent
holdout, then signed-AP1 native/video integration. Do not relax release scope.
Details/research: docs/production/ACES_OPTIMIZATION_DIAGNOSIS_2026-10-23.md.

**Latest — deterministic300 comparison completed; both candidates FAIL:**
Registered configs/experiments/aces_deterministic_shadow_pair_20261023.json.
Fresh weight0/1 runs and CPU full24 development evaluations completed; matched
recipe/code/data/RNG mechanics pass. Reports under G:/rudra_training/
run05_aces_deterministic_{control,scaled}300_20261023; scaled/comparison.json.
Overall target error analytic .009646895, control .010594832, treatment
.019774891. Shadow error control6.716783e-7 -> treatment1.518636e-6
(126.10% worse). Ordinary R passes both, but shadow lift>1code counts5/13359.
Neither checkpoint qualifies; shadow term stays OFF, thresholds unchanged.
Control SHA7aece903e30a77ac57614e1fed60b58621fa1492a2d03baa228e759ba4c564ea;
treatment SHAa24fa0e56a2ea470a69f0af7d431b97d75319614518f8c620ed2090ceb44beea.
Only contact00 of each visually reviewed; full visual review remains incomplete.
Jobs87879/19587/54770 exited0. No live jobs from this comparison.
The pasted session reconciled: virtual RAW remains the approved mission,
not demonstrated quality. Existing 16-bit video transport is not proof of
new signed AP1 model integration; explicit ACES display conversion still needed.
Next diagnose failed deterministic baseline/target and shadow behavior before
broader training; independent holdout, temporal/platform data and release remain.

**Latest — shadow scale trial FAIL; deterministic training now verified/default:**
training/aces_losses.py adds optional view-scaled signed-asinh shadow loss,
scale=.0036889303 AP1 (official inverse neutral5/255, not physical exposure).
TRAIN8hi/8shadow parameter probe train_loss_balance.json in spatial300:
median absolute gradients13.9157hi vs5.9792e-6shadow; scaled shadow.31054,
max.44772. No optimizer updates in probe. Research reference in new trial doc.
Registered matched fresh300 pair weight0/1. Mechanics verifies same recipe/code,
data/RNG/contract, finite weights. Both evaluated CPU full24, comparison under
G:/rudra_training/run05_aces_shadow_scaled300_20261023/comparison.json FAIL:
S control6.396912e-7->6.944492e-7 (8.5601%worse), all .01017949->.01224194,
H .8271072->1.1268851; treatment1shadowpixel>1code lift, both ordinaryRpass.
Do NOT adopt weight1. ControlSHA153a493e780253a0837c1ced0ff0ac9a3b8fbd27c00b54e6e604cb4b74e29185;
treatment7422084761d0261dbf63d03f547efd49368587bb750c33725600d15ac8ba5b37.
Repeated control varies substantially vs earlier .007358 overall error, so
stochastic numerical variation confounds single-run objective comparisons.
Added deterministic algorithms, cuDNN deterministic/no benchmark/noTF32,
CUBLAS_WORKSPACE_CONFIG=:4096:8. Two fresh60-step GPU repeats bitexact across
model,optimizer,RNG,recipe; reports run05_aces_determinism_probe_{a,b}_20261023,
b/repeatability.json. Same-machine proof only. --deterministic now defaultsON
(BooleanOptionalAction; --no-deterministic explicitly opts out). Changing this
CLI default after proof did not change enabled math; old recipe hashes preserved.
Five loss/sampler/gate/split tests pass. Only treatment contact00 reviewed this
turn; full visual acceptance remains incomplete and numericalFAIL sufficient.
All60432/95184/21039/41262/42453/81792 jobs exited0. Next use deterministic
baseline/objective comparisons; do not promote failed shadow term or prior
single-run gains as reproducible quality. Whole release goal still unfinished.

**Latest — spatial TRAIN48/dev24 trial completed; overall improves23.73%, shadow gate narrowly fails:**
Builder now supports explicit split/per-look, rejects source-scene overlap,
stores scene IDs and builder hashes. Initial train48 stopped after40 on17 exact
black HDR pixels in venice_sunset. Fixed smooth-shadow solver to preserve exact
black without log/radiance floor; rejects negative luminance. Test covers this.
Use ONLY G:/datasets/corpora/run05_aces_spatial_train48_v2_20261023 (48images,
21scenes) and run05_aces_spatial_dev24_v2_20261023 (24images,8disjointscenes).
Both builder code hashes match; metadataCC0/sourceHDR/SDR hashes verified;
allordinaryR100%, maxtrain.632509/dev.58505code. First train48 directory partial,
old dev24 superseded. Original dev has prior research use, NOT release holdout.
tools/train_aces_spatial.py fresh v3 channels8,300steps,Adam.001,batch4,crop128,
25%hi/25%shadow/50%uniform location modes, uniform image sampling. All48visited
16..44times,1196uniquecrops,95.57s. Snapshots/RNG saved but no verified resume
command. Recipe preregistered configs/experiments/aces_spatial_development_20261023.json.
Checkpoint G:/rudra_training/run05_aces_spatial300_20261023/step300.pt SHA
77dd6075af4fa97fcb581dc8897c4c8b2dbeb9704435ff2a5cf42e6aaf8f53ad.
Evaluator now verifies separate training-report identity and disjoint scenes/
source hashes for dev. Full24 final300 completed; acceptance.json numericalFAIL:
equal-image mean targetasinhMSE .009646895->.007358012 (23.7266%better), H
.75866475->.62021936, S6.5908617e-7->6.5937369e-7 (~.0436%worse). Rallpass,
zero shadowlift>1code. Do NOT relax preregistered S guard or promote checkpoint.
Partial contacts00/06/08 reviewed; visual_review.json marks incomplete. Six
split/sampling/black-reference/gate tests pass. All build/train93437/eval51967
jobs terminal. Next diagnose TRAIN highlight/shadow loss-scale and gradients
before next objective trial; current absolute asinh objective may underweight
dark reconstruction. Broader data/video/native/independent release work remains.

**Latest — full-frame failure found; pixel-local normalization candidate improves11/12:**
tools/evaluate_aces_smoke.py evaluates full1280x720 untiled FP32 TRAIN frames,
saves predictions/contacts and learnedFLOAT AP1 EXR. GroupNorm smoke fails:
magenta streaks in sky/opening, target all-image error worse than analytic10/12.
Ordinary R100% still; signed AP1 can be <-100 in bad reconstructions. Do not
clamp gamut to hide failure. IndependentOpenEXR read verifies FLOAT RGB+mask,
bitexactRGB for group smoke export; mask100% change, not source R support.
Same-pixel crop/full context audit: stonehighlight inner64 targetMSE .001979
->.018444, contextRMSE.120579. Includes convolution and normalization context.
New opt-in ACESModel(normalization='pixel_rms') contractv3 replaces spatial
GroupNorm with per-pixel channel RMS affine norm; v2/group default unchanged.
Locality test with distant changed input passes exact equality;17 focused
model/view/gradient tests pass. Same4crops/seed/Adam.001/60steps candidate done.
G:/rudra_training/run05_aces_pixel_rms_smoke_20261023 checkpoint
SHA6c9889b1f2dda98b0ef757b5aad8021490d0fcbdc07bf2743f15a978abe51421.
Crop final loss.005542 (worse than group.001606), but full12 mean targetasinhMSE
.040001->.011842 (70.40%lower),11/12improve vsgroup; only6/12beat analytic.
OrdinaryR100%,worst.108609code,0shadowlift>1code. Stonehighlight inner64
contextRMSE.000132 vs.120579; crop/full targetMSE .003297/.003298.
Reviewed contacts00/06/08 bothmodels: large magenta streaks gone, some highlight
edge errors remain. Not quality release, independent holdout or video proof.
Both fullframe reports/context_audit/comparison saved. All16737/90898/75648
jobs exited0. Next use v3 explicitly for broader spatial TRAIN sampling and
separate development evaluation; don't repeat4crop overfit or promote smoke.

**Latest — direct/smooth shadow target probes; first ACEScg learnability run complete:**
Direct-shadow-reference opt-in bypasses noisy baseline mixing/Poisson in shadows,
mandatory region bound. New run05_aces_direct_shadow_pilot_20261023 full12:
ordinary R100%,0lift>1code,max.49code; shadow interior gradient RMSE ~5–7e-7
cases6/7/8 (actual HDR detail retained), but +3EV boundaries cases6/8 worsen
vs previous region-bound7.212->8.582 and6.358->7.596codes. Not adopted.
pipeline/aces_shadow_smooth.py opt-in constrained log-exposure field, fixed
outside luma anchors, per-pixel verified OCIO upper bounds, L-BFGS-B. New
run05_aces_smooth_shadow_pilot_20261023 full12 ordinaryR100%,0lift,max.49code.
Shadow RMSE6/7/8=.08195/.03786/.14379; mixed vs previous region target
.18664/.14796/.11690. +3EV boundary6/8=7.6555/5.9719. No universal adoption.
Reviewed direct shadow crops05/06/08 and smooth06/08 at0/+3EV. Black seams
remain visible under push;26 focused tests pass. Full quality is unfinished.
Changed approach from target-only iteration to a clearly isolated learnability
probe (NOT production training): tools/train_aces_smoke.py, four fixed TRAIN
128 crops, channels8, Adam.001,60steps. CUDA run finished20.81s, loss.0298612
->.00160594; ordinary37055pixels allwithin1code,worst.219385. No validation.
Checkpoint G:/rudra_training/run05_aces_learnability_smoke_20261023/smoke.pt
SHA57c4763702729c5695e21dbc8300e09cfc80744ff3ab6e84b3483d5b048b0a7d.
Load ACESModel(8), experimental v2 only. ExactCPU-backed view/autograd, not
native/exportable or generalization proof. Pilot hash in checkpoint/report.
No live jobs: direct42477/12606/84031,smooth3215/35334/49916,smoke75770 exited0.
Next inspect learned reconstruction on full TRAIN frames and improve target
transition/headroom criteria before broad training; keep commercial data,
video/platform/native delivery and independent validation scope intact.

**Latest — coherent shadow scaling preserves interior detail; edge trade-off remains:**
pipeline/aces_shadow_regions.py applies one verified feasible AP1 exposure scale
per4-connected shadow component.22 bisection iterations, exact OCIO check;
doesn't claim optimality for nonmonotonic gamut. Opt-in region_shadows=True,
mutually exclusive with pixel bound. Source support/defaults unchanged.
New full12 pilot G:/datasets/corpora/run05_aces_region_shadow_pilot_20261023:
ordinary R100%;0shadow pixels lift>1code, max.4900001code. Shadow interior
gradient scores unchanged to FP precision from unbounded boundary target
(cases6/7/8 RMSE .186641/.147956/.116898), unlike rejected pixel projection.
No zeroed components; minscale .032843 (~4.93stops dimmer) in grass case5.
Comparison boundary metrics improve grass +3EV26.965->2.610codes, but worsen
stone cases6/8 +3EV3.847->7.212 and3.595->6.358. Mixed candidate, not release.
Reviewed shadow_review.png cases5/6 at+3EV: broadly similar at contact scale;
this does not override numeric edge regression.24 focused tests pass.
All builder85737/detail79381/comparison14824 exited0; no training live.
Next separate true-reference shadow gradients from Poisson baseline noise:
current coherent bound preserves whatever interior errors were already there.
Assess direct HDR-guided shadow target with coherent exposure and transition
constraints; preserve actual detail and no-lift jointly, not just interior scores.

**Latest — real HDR detail measured; per-pixel shadow bound fixes lift but damages gradients:**
tools/audit_aces_reference_detail.py hash-verifies TRAIN sourceHDR/SDR/targets,
measures scale-invariant log2-luminance gradient RMSE/gain/cosine on eroded
highlight/shadow/fullyclipped interiors. Test proves exposure invariance and
detects flattened detail. Counts/valid-positive support recorded (differs for
analytic black pixels; do not compare RMSE without checking support).
boundary pilot reference_detail.json: fullyclipped stone cases6/7/8 retain
gradient gain .9749/.9969/.9770, cosine .9637/.9988/.9665; analytic gain0.
Thus smoother target retains real luminance detail, not just analytic white.
Shadow lift>1code: scalar8683, boundary6223 out of12478 supported pixels.
New opt-in bounded_shadows clips viewed shadow channels to source+.49code,
then exact inverse; G:/datasets/corpora/run05_aces_bounded_shadow_pilot_20261023.
Full12 audit:0lift>1code, max.490038code; ordinary R100%, highlight metrics
unchanged. BUT shadow gradient RMSE cases6/7/8 .39756/.17391/.29190, worse
than unbounded boundary .18664/.14796/.11690 (case6 valid support differs).
Per-pixel projection is NOT adopted: satisfies lift at expense of detail.
Reviewed shadow_review.png cases5/6 at+3EV; contact alone doesn't expose all
gradient damage.23 focused tests pass. Builder57918/audit41327 exited0.
Next solve shadow constraints jointly with gradient preservation (e.g. coherent
region scaling versus constrained gradient optimization); don't accept clipped
detail merely because lift passes. No new training or release claims.

**Latest — gradient-domain target candidate reduces sky seams; mixed results elsewhere:**
pipeline/aces_boundary.py implements original Dirichlet gradient-domain join in
signed asinh AP1 (Perez et al2003 idea, no external code/weights). Opt-in
build_target boundary_method=poisson_asinh; scalar remains default. Source
support/bounds unchanged.10 focused boundary/target/model tests pass.
New full12 TRAIN pilot G:/datasets/corpora/run05_aces_boundary_pilot_20261023:
all ordinary R100%, worst.493214code. comparison/report.json from
tools/compare_aces_boundaries.py verifies hashes and compares boundary gradient
discrepancy to analytic baseline at -3/0/+3EV (NOT groundtruth quality).
Evening-field clip mean discrepancy at0EV31.613->1.776codes, -3EV47.970->4.607.
Reviewed comparisons00,02,07,09 plus regressions.png (05/11 at+3EV).
Sky patch seams greatly reduced. Other banding and +3EV foreground artifacts
also exist in analytic baseline; not solely caused by scalar target fitting.
Case05 AgX safari shadow boundaries worsen at+3EV14.536->26.965codes;
case11 small regression. No universal adoption/training approval. Need assess
shadow boundary objective and true reference detail retention (stone opening
candidate looks closer to analytic and may lose visible recovery). No jobs live:
builder93622 and comparison85619 exited0. No new training.
Full notes and primary research link in production/ACES_TARGET_PILOT_REVIEW_2026-10-23.md.

**Latest — ACEScg prototype v2 representability passes; pilot targets rejected visually:**
training/aces_model.py now explicit experimental AP1 contract v2, signed linear
RGB residual with shared max-abs RGB scale+.001; bounded source correction and
source-defined reconstruction support unchanged. Old exponential four-stop
residual failed three TRAIN targets (179 pixels >1 viewed code); twelve stops
still failed131. No checkpoint migration or production/trainer integration.
range_audit_v2.json under G:/datasets/corpora/run05_aces_target_pilot_20261023:
actual compose oracle covers12 targets at -3/0/+3EV, zero pixels >1code,
worst .001808code; maxAP1 error3.052e-5, max raw residual6.536. Oracle only.
Random-field full12 TRAIN stress aces2_composition_train_v2.json under
shadow_repair_comparison_20261023: ordinary R100%, worst.538467code, allfinite.
18 focused tests pass. All three pilot contact sheets individually inspected:
evening-field clip sky has coloured patch seams, ACES row +3 foreground patches,
veld-fire sky still banded. DO NOT TRAIN ON OR EXPAND CURRENT AP1 TARGET PILOT.
Next diagnose/fix reference-to-inverse boundary colour/slope mismatch, then
repeat full-resolution target/headroom review. See production/
ACES_TARGET_PILOT_REVIEW_2026-10-23.md. Both audits exited0; no training launched.

**Latest — exact ACES reference path passes real sources; gamut constraint found:**
rudra/aces_view.py wraps official OCIO2.5.2 ACES2 SDR100 Rec1886/Rec709
inverse/forward toACEScg, CPU NumPy only, no assumed physical exposure scale.
tools/audit_aces2_sources.py full24 savedSDR audit under
shadow_repair_comparison_20261023/aces2_real_sources: ALLpixels within1code,
worst.688964code; ordinaryworst.56919. Analytic only, no model/dequant/detail.
First real source exportedFLOAT AP1 EXR, bitexact readback; independentOpenEXR
RGB/types verified. Maskzero means analyticinverse, not learnedreconstruction.
training/aces_reference.py experimental exactforward/finite-difference backward
bridge:11 tests pass withview tests; directionalgradient and endpointfinite.
Batch4x3x256x256 CPUforward.133s/backward.695s, CUDAinput .109/.731s because
OCIO executesCPU; finite. Not analytic derivative, higherorder/Script/native
unsupported. Not wired into model/trainer; no newtraining.
Critical synthetic65536gamut probe:4.747% validAP1 inverse samples have negative
Rec2020 channels. Clamping them (current model nonnegative convention) drops
roundtrip to95.468%, max70.44codes. Keep working outputAP1 or preserve signed
values with consistent model/nativecontract; don't simply swap oldbaseline.
Next design/test explicit ACEScg model output/composition and matching targets,
retaining source-basedR/gradeheadroom tests. Current BT1886 weights not promoted.


**Latest — signed repair FAIL; ACES LUT and temporal smoke completed:**
All recovery200 evaluations complete. comparison.json: signed gradient2.081745
vsrepaircontrol2.066213 (0.7517%worse), clean2.495742vs2.462099, also worse
earlier goodcontrol. R24pass, S0/3both; banding~.2894FAIL. Do NOT adopt signed.1.
Four fixed visual reviews inspected: blueblotches gone in BOTH arms, no clear
signed benefit. Shadow diagnostics nearlyequal; removing spatial loss sufficed.
No newtraining proposed merely to tune another auxiliary scalar.
temporal_smoke_v2.json:2x9frame camera probes using same FP32 tiled512/64 Runner.
Global mean excess EV construction~.00116both, local~.0396/.0398; tankfarm
global~.0000177, local~.00072both. Fixedcells, noflowalignment; camera/content
changes confound local values, not release Tpass. v1 superseded for wrongfloorunits.
tools/audit_aces2_lut.py official OCIO2.5.2 ACES2 inverse differentiable LUT
probe:65cube~96% onecode,129~99%,2568bit100% but continuous99.6017% (<99.9).
Exact official inverse100% for SAME randomsets; prior boundaryprobe exceptions
still apply. LUT256 p9992.345code/max9.888 on continuous; not usable as declared
productview yet. Trilinearaxis/gradient test passes. No model/native migration.
Next prioritize exact official ACES2 baseline/target feasibility on actual
images and a consistent training/export contract before further BT1886-only
auxiliary trials. ApproximateLUTs are not approved. No known live jobs now.


**Latest — repair pair completed200; CPU evaluations and temporal fixtures:**
Signed training exited completed200 (run25258df4222244d6a57b422c8a238ef4).
mechanics200.json allchecks pass, same RNG/code/data except signedweight.1.
ControlSHA9ba6d9363d49994200dc275d45980f5f57e9a643681915de4c770259411d87a0;
signedSHAa2018a602de174e66bc83f23131fc74967cc0bcb7605f65ae02aed2e463cb51f.
Control CPU audit still live; signed CPU audit launcher37224 just launched,
output shadow_repair_comparison_20261023/auxiliary200. Recheck actualworkers.
New compare_shadow_repair.py takes --control --treatment --original --out,
enforces registered earlier-good-control no-regression and exact zero visible
shadow lift, leaves visual/release evidence pending.7 related tests pass.
Original = reconstruction_comparison_20261022/control400/report.json. After
both new audits finish use this comparator, audit_saved_shadows.py each report,
and review_run05_dequant.py plus actual image inspection. No quality result yet.
Prepared G:/datasets/corpora/run05_temporal_probe_v2_20261023/manifest.json:
2CC0 development camera-move snippets,9frames each, fixedlook/fixedexposure,
no codec/VAE.18hash/decode checks pass; first/last contacts visually inspected.
Original fixturefolder is empty after refusing32frames (sources have9-frame
snippets, two moves spaced by1000 in filenames). v2 only. Synthetic24fps,
not originalcapturetiming. No temporal model evaluation yet; longer/objectmotion
and independent platform coverage still required. Builder tool added separately.


**Latest — repair control200 done; signed treatment and control audit launched:**
Control run ID4bfc713dd84b44f38b6e0a8576e79fb1 completed200, actual worker50712
exited before treatment launch. Signed treatment launcher34788 under
G:/rudra_training/run05_shadow_repair_signed_20261023. Control CPU full24 audit
launcher50712 (PID REUSED, now evaluation) under
G:/rudra_training/run05_shadow_repair_comparison_20261023/control200.
Keep training dependencies fixed. After signed completes, verify pair final200
--treatment signed_shadow_weight --weight .1, then same CPU full24 treatment
audit; compare plus original nonspatial control400, render/inspect reviews.
New tools/audit_saved_shadows.py supplements saved full24 outputs with source
max<=5/16code signed-reference, code-error, >10code-change and gamut diagnostics.
Two tests pass. Run on both new reports once ready. Earlier control vsfailed
reports saved under reconstruction_comparison as control_shadow_audit.json and
failed_shadow_audit.json. At<=5code mean signed-referenceRMSE .006852vs.614756,
source>10code increase fraction0vs.113601 across8 measurable pairs. Reference
codeRMSE1.6518vs9.2696; tiny-support equal-pair means need their counts. Negative
Rec709 alone is NOT invalidHDR: targets contain some too. No gate substitution.


**Latest — signed-shadow recovery trial control launched (23 October):**
Standalone signed source-Rec709 asinh radiance loss implemented and integrated
opt-in --signed-shadow-weight, default0. Keeps negative channels in supervision;
target anchored, source max<=16code support, .1nit asinh scale.15 focused tests
pass including exact resume/model/optimizer, sampler/RNG match and invalidweight.
New TRAIN12 diagnostics under reconstruction_comparison_20261022:
signed_shadow_train.json, signed_shadow_fullcontext_train.json,
signed_shadow_parent_parameter_train.json, signed_shadow_failed_parameter_train.json.
Spatial-only constant wrong-colour loss is essentially0 (synthetic test).
Fullcontext does not broadly worsen TRAIN12; one bad crop improves, so GroupNorm
is not established as root cause. Candidate/primary parameter norm original
parent median.00938/max.06899, rejected parent median.01128/max11.159.
Registered configs/experiments/run05_shadow_repair_control.json and _signed.json:
200steps from rejected gradient400, spatial/dequant OFF both, signed .1 only
treatment. Trial must beat repaircontrol>=5% gradient and no clean regression,
AND match/beat earlier nonspatial control400, zero shadow lift/no colour blobs.
Control launched37884 under G:/rudra_training/run05_shadow_repair_control_20261023.
Recheck actual worker; keep trainer/math dependencies fixed. After control exits,
launch signed recipe same way. Final200 same-device full24 evaluation, pair
verifier now supports --treatment signed_shadow_weight --weight .1. No promotion.


**Latest — reconstruction-gradient trial FAIL; dark blue artefact diagnosed:**
Both CPU full24 audits finished; comparison.json under G:/rudra_training/
run05_reconstruction_comparison_20261022: gradient2.070497->2.549559
(23.1375% worse), clean2.463664->3.392623. Rallpass and relative H/S guards
pass, but original banding FAIL and nonzero shadow lift; do NOT adopt .01 loss.
Three fixed visual reviews inspected. Acoustical-shell clean image shows blue
blotches on dark roof/foliage. dark_colour_regression.json: source max<=5code
23875pixels; treatment11244 gain>10codes, control0; treatment50.785% have
negative linearRec709 channels vscontrol0. Max channel increase90.85codes.
Next diagnose loss/display-clamp gradient behaviour and shadow chroma on TRAIN
crops before any new trial. Do not change release thresholds to accept this.
Synthetic compression-bound audit also confirms .49code preservation cannot
fully restore clean codec inputs; details/research links in continuation audit.
No training/evaluation live expected; recheck processes before new launches.


**Latest — reconstruction pair complete; full-image audits live; direct graph parity:**
Treatment completed400 and training processes exited. mechanics400.json under
G:/rudra_training/run05_reconstruction_comparison_20261022 passes all matched
identity/config/RNG checks; CUDA nondeterminism caveat retained. Treatment SHA
e688ea8b383e124b82070071165b900ccf5ec33436d8170c77f2e9f83e259aa7.
CPU control audit worker29100 live20/24; treatment audit launcher47484,
worker51456 live6/24. Recheck before restarting anything. After both finish,
compare control400/report.json and auxiliary400/report.json with existing
compare_run05_dequant.py, render review_run05_dequant.py and inspect images.
No quality result or promotion yet.
tools/export_run05_direct.py exports complete default-control BT1886 output
including bounded restoration as research TorchScript; native v1 unchanged.
direct_graph_probe contains graph/metadata and real_crop_parity.json: exact
CPU eager/reloaded parity on5 synthetic shapes and6 TRAIN codec crops.
Batch1 only; no frame-conditioned heads, ACES-view/native integration or
quality claim. Six focused export tests pass (13 expected tracing warnings).


**Latest — reconstruction control finished; treatment and CPU audit live (23 Oct Cairo):**
Control38568 exited completed400. Treatment run05_reconstruction_gradient_
20261022 started launcher52116,80/400 observed, errors empty. Keep pinned code
fixed. Control final400 full24 CPU audit launcher45392 under
G:/rudra_training/run05_reconstruction_comparison_20261022/control400.
Use CPU for treatment full24 too (comparison requires same device). Old/new
control tensors differ94 entries, so no result reuse despite same data/seed.
After treatment completes: verify_run05_pair --treatment
reconstruction_gradient_weight --weight .01, evaluate final400 CPU, compare,
render/inspect selected reviews. No quality result yet.
Native v1 exporter now explicitly refuses baseline_source/bounded_source models
because it drops picker conditioning/restoration head;3 tests pass. Extended
native contract/inference/composite including model primaries remains required.
Research-only export_run05_master.py writes cached audited predictions as
float32 AP1 with conservative change mask. One real saved master decoded Rpass
and independent OpenEXR types verified; not an ACES-view or production pass.
Mask marks100% of example because learned subcode corrections affect allpixels;
do not hide them by changing exported-mask thresholds or the source-based R test.

**Latest — reconstruction-gradient matched control launched:**
New multi-exposure reconstruction-gradient term implemented opt-in in resumable
trainer, default0; rejected dequant term stays disabled. TRAIN12 input-boundary
probe median candidate/primary norm24.721, tested weight.01 (median.247).
24 focused loss/resume/dequant tests pass, exact CPU resume and unchanged
sampler/RNG, invalid weights rejected. Code hashes include new loss module.
Registered recipes run05_reconstruction_control.json and _gradient.json:
400 steps, same strict step1000 parent/data/seed, gradient weight.01 only in
treatment. Control launched56092 under G:/rudra_training/
run05_reconstruction_control_20261022; inspect actual worker/logs. Keep training
dependencies fixed, run gradient arm only after control exit. Then full24
evaluate_run05_dequant.py on both final400 and compare; same >=5% and clean
guard plus R/H/S; no gate changes. verify_run05_pair.py now supports
--treatment reconstruction_gradient_weight --weight .01.
Channel audit24: known channels in high pixels105152, model changes>1code96427,
target90604. Current pixel mask permits this; mechanism evidence, not a reason
to silently change policy/rebuild data. Details in continuation audit.

**Latest — branch diagnosis and bounded-diffusion pilot completed:**
No training/evaluation active. audit_dequant_branch.py FP32 CPU TRAIN12:
raw abs never>3, derivatives healthy; .003 auxiliary gradient comparable to
primary, median cosine.952 (mostly same direction). Spatial component norm
only.001-.023 of pixel component BEFORE .5weight: nominal gradient supervision
is weak. Clean2 crops target mean correction~0 but model+.268/+.286codes;
codeMAE worsens. Target bound overshoot max.0467codes, not a large infeasible
target problem; don't rebuild corpus merely from >.49 fractions.
control_regions.json zeroEV diagnostic: ordinary pooled gradient1.232615vs
baseline1.232575; reconstruction/boundary30.6386vs11.7523. Only.24% measured
edge-channel samples are in latter group; they explain almost all extra error.
Primary registered trial remains FAIL; no gate changed.
Experimental rudra/interval_dequant.py (hard-projected edge-limited diffusion)
passes2 synthetic tests but TRAIN12 pilot worsens11/12 real crops (including
one clean). NOT integrated/adopted; retain negative evidence. No new weights.
Next separate spatial-loss learning from pixel bias and reconstruction-boundary
quality; inspect branch-specific gradient scaling before a registered trial.

**Latest — paired dequant trial FAIL; mask writers verified:**
All training/evaluation/build sessions finished (18247,17198,30286,78102,
33931 terminal). No known live run. Full24 comparison.json: auxiliary gradient
2.070899vscontrol2.070258 (-0.03095% improvement), clean2.465394vs2.463009;
fails both learning criteria. R/highlight/shadow guards pass; legacy banding
.289924FAIL. Do NOT adopt .003 loss or promote checkpoint. Four selected
visual comparisons inspected, no obvious improvement; subcode verdict from
metrics, not these overview images. Next investigate dequant-head saturation,
actual branch gradients and target attainability on TRAIN crops before more
training, plus ordinary/reconstructed error decomposition for diagnosis.
Python/native EXR writers/readers now support separate rudra.reconstructed
plane, no alpha overwrite, mask unchanged by exposure/primary transform.
53 Python tests pass/2existing skips;14 native export tests pass; native/Python
float32 mask EXRs byte-identical and independently read by OpenEXR. Release
RUDRA build passes in native_gate_b. Still NOT wired from model to app/master
requests; mask provenance semantics must be explicit, do not hide edits in R.
Full evidence in continuation audit and run05_dequant_comparison_20261022.

**Latest — both400 runs complete; control full24 audit running:**
Auxiliary worker52996 exited completed400, SHA41cf77e10af6db00891c8c54992bce004609016d0f2ace787b41eb7a44732eac.
verify_run05_pair.py checks all pass at200 AND400 (mechanics200/400.json):
code/data/parent/config except treatment, sampler/RNG, schedule/AMP, finite.
CUDA nondeterministic kernels mean this is not bitwise repeatability proof.
Parent full24 CPU report complete: zeroEV gradient1.9175vsbaseline1.4844,
clean2.4190vs1.6851. All Rpass, but parent reference gradients worsebaseline.
Control400 GPU full24 audit running session18247, worker11988/launcher44916,
under run05_dequant_comparison_20261022/control400. Poll same session/process;
do not restart on timeout. After exit run auxiliary400 same evaluator/device,
then compare reports, render review_run05_dequant.py and inspect visuals.
Separate official ACES2 inverse probe passed sampled99.9% R but not allpixels;
see audit and aces2_inverse_probe.json. No model/native ACES migration yet.

**Latest — control400 complete, auxiliary training and parent audit live:**
Control worker7124 exited, status completed400; explicit step_0000400.pt SHA
b3d1787d1425146f61680fe99f28631f9feac88e3f0dc723ef5877fad41ee6d1.
Auxiliary run05_observed_dequant_20261022 launched34964, same recipe except
.003 loss weight/output/run ID (config diff verified). Keep pinned code fixed.
tools/evaluate_run05_dequant.py audits all24 val images, saves predictions,
contract and reference code/gradient/curvature at7 pushes, clean/codec strata.
Parent CPU audit live launcher49988 under run05_dequant_comparison_20261022;
inspect process and parent_console/errors/report. New aggregator/comparison
tests pass; comparison requires full24 and matching baseline/reference masks.
After auxiliary finishes, evaluate explicit control400 and auxiliary400 with
that tool (GPU available then), compare_run05_dequant.py and inspect visuals.
Aggregation fixed before auxiliary results: mean of7 per-push equal-pair
gradient means; >=5% overall improvement, no clean mean regression, guards.
Missing exposure evidence cannot pass. Training identity/mechanics still need
checkpoint-level verification. Parent audit alone cannot judge trained pair.

**Latest — corrected corpus audited; matched control launched (22 October):**
Corrected624 corpus complete:1872 media decode/hash checks pass,208scenes,
600train/24val, saved target R624/624; same624 source SDR hashes as old corpus.
Opt-in dequant code/reference-gradient loss integrated with view/augmentation
guards; exact CPU resume and unchanged paired crop RNG tested. Shared sRGB
zero-channel backward NaN fixed without changing forward values. Latest23
dequant/resume tests plus builder dependency-pin test pass. Builder now pins
target-math dependencies; do not resume completed corpus with changed builder.
Control400 launched under G:/rudra_training/run05_observed_control_20261022
(launcher42500); inspect logs/worker before further action. Recipes observed
control/dequant use same parent and corrected data; auxiliary weight .003.
Keep pinned code fixed. After control finishes, run auxiliary sequentially and
evaluate explicit final400 checkpoints on full24 with reference-gradient
metrics and clean/codec strata. No promotion; banding/ACES/temporal gaps remain.

**Latest — observed clipping policy validated; full rebuild running (22 October):**
Expansion2000 completed, worker38808 exited; no training now. Final old-policy
fullres24: Rpass, highlights.818vs1.920baseline, shadow0/3pairs, banding.2876FAIL.
Added opt-in observed_clip_v1 across builder/head/checkpoint/trainer/evaluator;
legacy defaults retained.42 focused tests pass including actual resume trainer
policy mismatch refusal and config reload. Corrected24 validation targets have
same input hashes,72 media audit pass,R24/24; >1code bright-region changes fall
52,226->21. Visually confirmed sky patches gone in target AND policy-only model.
Policy-only step1000 checkpoint (unchanged weights, not retrained) fullres24:
Rallpass, highlight1.073vs2.618baseline on6pairs, shadow0/3, banding.2889FAIL.
Full624 corrected corpus building(session22197,worker11408),495rows observed.
Do not edit builder/policy/contract dependencies while building. After completion
audit all hashes/source identity/targets before training. No model promotion.

**Latest — step1000 review exposes target/metric defects (22 October):**
Fullres24 complete: R24/24, highlight.714vs1.920 baseline, visible shadow lift0
(3pairs), fidelity.02133vs.03120; banding.2885 FAIL. Four visual comparisons
inspected; sky patches exist in TARGET and model. New24-target audit finds
45,066 newly clipped bright-but-unclipped input pixels;11pairs change >1code.
Old mask starts reconstruction at.94 and exempts near-white detail from R.
Do NOT promote or start dequant trial before correcting/auditing this policy.
New reference banding diagnostics show noise can fool old flat-step gate;
all4 reviewed model outputs worsen zero-push reference code/gradient errors.
Training38808 last verified live1820/2000; let finish, preserve pinned code.
Review/export/metric evidence paths in continuation audit. No release pass.

**Latest — dequant loss prototype; fullres1000 running (22 October):**
Standalone code/reference-gradient objective,5 tests pass; NOT enabled in trainer.
Real12-crop train diagnostic finite after fixing new loss zero-channel gradient;
step800 improves9/12 but both uncompressed crops regress. New loss gradient
median360x legacy, needs small-weight matched trial, not automatic adoption.
CPU fullres24 ofstep1000 running(session6187,worker17032), output under
run05_expansion_eval_20261022/step1000_fullres. Training38808 live1060/2000.
Keep trainer fixed; no promotion. Full evidence in continuation audit.

**Latest — desktop RGB16 integrated (22 October):**
FFV1 RGB16 now wired through movie CLI/queue, desktop movie and folder export,
and headroom movie export; full RGB/PQ tags and MKV container. Release builds
pass;8 native encode/QC and5 distinct focused desktop tests pass, including
exact16-bit decode and real queue/folder/headroom exports. EXR labelsfloat32.
Resolve/Nuke and visual UI review remain unverified; PQ video is not a
scene-linear master. Training worker38808 verified live step680/2000; keep
pinned trainer unchanged. No model quality pass. See continuation audit.

**Latest — expansion2000 launched (22 October 22:20 Cairo):**
All624 saved targets now pass R across every codec group. Complete data hash,
decode and input-sheet checks passed. Started run05_expansion_20261022 with
configs/experiments/run05_expansion.json,2000 scratch steps,2-hour cap, resume
save100. Check actual process/status; do not edit pinned trainer code while
active. No quality or release pass; no VAE rows in this run. Older pending
target-check notes below are superseded.

**Latest — expanded data audited; target check pending; Wan VAE verified small (22 October):**
Expansion624 complete, all1872 media hashes/decode pass, all26 input sheets
visually reviewed; visual_review.json records limitations. Target-contract check
still live worker48072/session20160; inspect target_contract.json before launch.
Prepared expansion2000 NOT started. Wan VAE downloaded official Apache2.0 pinned
revision, hashes at G:/datasets/models/wan21_vae_run05_20261022.144x256 GPU probe
and visual roundtrip pass; no fullres/temporal claim. Weight identity+resume
protection and loader cache added;22 tests pass. No VAE rows in existing624.

**Latest — shadow loss improves but fails; expansion QA begun (22 October):**
Shadowloss fullres24 complete: R24/24, highlight1.263stops vs baseline1.920,
shadow lift .303 vs nativecrop .631, fidelity .03020vs.03120; banding .283
still fails. Expansion builder still live(session5624),473/624 pairs observed.
Partial24-pair audit/visual review passed (72 hashes; masks now decoded too).
At snapshot63 pairs have >.1% true shadow target region. Complete full audit
and all contact sheets before training. Prepared2000-step expansion recipe,
NOT launched. See continuation audit for evidence and unfinished tasks.

**Latest — expansion conversion and shadow-loss evaluation running (22 October):**
Shadowloss200 completed: clean/hard PU -1.57627/-.416666dB. Fullres24 evaluation
running at run05_bounded_eval_20261022/shadowloss200_fullres (session70374).
CC0 Poly Haven expansion selected/hash-verified200train+8val scenes;58 train
names indicate low-light candidates. Conversion624 pairs running at
rudra_run05_ph_expansion_v2_20261022/pairs (builder37760,session5624).
Check live state before relaunching. Initial expansion folder marked REJECTED;
use v2 only. Needs visual/hash/target audit and broader location-overlap review.
Detailed evidence in continuation audit; model and goal remain incomplete.

**Latest — target-relative shadow loss experiment (22 October 22:04 Cairo):**
Nativecrop fullres complete: R24/24, highlights1.288vs1.920 baseline; shadow
lift .631 and fidelity .03525vs.03120 fail. Train-only dark target audit11/72
pairs shows all crushed target pixels <.1nit. Existing near_black_log_loss
enabled (weight.05,code16) in matched200-step run05_shadowloss_smoke_20261022.
10 dark/objective tests pass including target-recovery gradients and resume.
Check live worker/status; do not edit pinned trainer modules while active.
No production pass. See continuation audit for all reports and next evaluation.

**Latest — native RGB16 and crop-smoke result (22 October):**
Native FFV1 RGB16 sequence export implemented;7 SequenceEncode tests pass,
including actual lossless16-bit decode and HDR tags. Desktop UI/queue unwired.
Native-crop smoke completed200; clean/hard PU -.890269/-.532466dB, no promotion.
Full-res same24-frame evaluation running at run05_bounded_eval_20261022/
nativecrop200_fullres. Check process/report before citing results. Next inspect
dark target distributions to address visible shadow lift; do not resume old runs.

**Latest — native crop correction and matched run (22 October 21:58 Cairo):**
Bounded full-res24 evaluation complete: R24/24 pass, highlight error1.92->1.27
stops, fidelity .0312->.0280; shadow lift .284 and banding excess .287 still fail.
Found dataset resizing invalidates source-preserving pairs (clean R3/24).
Native crops now used for these rows; real crop audit R24/24,19 tests pass.
Legacy hard augmentation changes appearance without target adjustment: R0/24
targets. Keep stress scores, but do not claim they test the matching contract.
run05_nativecrop_smoke_20261022 launched for200 steps. Check actual worker/status;
do not change pinned training files while active. Full audit has evidence paths.

**Latest — bounded smoke completed; full-frame evaluation running (22 October):**
run05_bounded_smoke_20261022 completed200/200 and worker exited. Clean PU
+0.076043dB, hard -0.490607dB; no promotion. Full-resolution24-frame evaluation
started at G:/rudra_training/run05_bounded_eval_20261022/step200_fullres;
inspect report/process for completion. Tiled source-preservation test passed.
Future source collection now rejects missing/malformed commercial_ok flags;
20 corpus tests pass. Existing corpora unchanged. Main goal incomplete.

**Latest — bounded output diagnostic running (22 October, 21:51 Cairo):**
Corrected96-pair preservation smoke completed200 steps but regressed clean/hard
PU -7.62154/-0.34067 dB. Added experimental --bounded-source output head;
21 focused integration/resume/objective tests passed. Matched200-step run
run05_bounded_smoke_20261022 launched; worker27140 verified live after preflight.
Check actual process and logs before further work; do not edit pinned trainer
while active. Recipe configs/experiments/run05_bounded_smoke.json. No quality
pass; needs full-resolution evaluation. See continuation audit for details.
Python FFV1 RGB16 HDR delivery now tested end-to-end; native UI and grading
software interoperability remain incomplete. Contract view remains BT.1886,
not standard ACES display transform. Main goal remains incomplete.

**Latest — run05 development smoke failed; storage defect found (22 October evening):**
Read [continuation audit](production/RUN05_AUDIT_2026-10-22.md). Run05 scratch
200-step smoke completed, final crop clean/hard PU -4.714/-0.296 dB; full-res
24-pair diagnostic also fails preservation/shadows. No promotion. Explicit
restoration/source-preserving target modes added; first saved preservation
targets exposed a 0.005-nit storage floor lifting blacks. New output floor1e-6
nits; 19 corpus/coverage tests pass. preserve_validation_v2 completed and all
24 saved targets pass R (1.0 within-one-code fraction). Full96-pair
preserve_corpus_v2 rebuild is in progress for a matched200-step smoke using
configs/experiments/run05_preserve_smoke.json; check completion before launch. Original
corpora/checkpoints retained. Native float32 ACES/AP0 and linear master request
integration builds; 12 export tests pass. FFV1 RGB16 transport byte-exact, but
HDR video integration remains incomplete. Main goal and release are incomplete.

**Latest — SI-HDR independent check PASSES; dark guard not confirmed (22 October):** frozen run02
best.pt passes SI-HDR A1-A3 at both clip levels (full +0.75/+1.12, highlights +0.71/+0.83 dB, CIs > 0;
simulated SDR, independent captures), but fails the Mataha reserved test from codec-crushed blacks.
The luminance dark guard fixes Mataha (diagnostic) but costs SI-HDR highlights (C3 fail) -> not
confirmed. Next: training fix for crushed/noisy near-black (run03 draft), checked on validation,
SI-HDR and ToS. No unused footage exists, so no fresh held-out production test is possible.
release_authorized=false. Details: PRODUCTION_PROGRESS.md 22 Oct (later).

**Latest — frozen candidate FAILED the reserved test (22 October):** run02 best.pt (16,200) passed
full-res validation (clean +0.88 / hard +0.21 dB) but the Mataha test gives clean -0.35 [-0.99,+0.18]
and hard -0.11 [-0.24,+0.02] dB: both gates fail; CVVDP also lower. Cause: crushed, codec-noisy blacks
in dark interiors are lifted into coloured blobs with tile seams (A_0027C021, -6.3 dB). ToS: flicker
pass, moving-highlight T2 fail 3/4. Not production-ready; release_authorized=false. Test kit spent;
do not tune on it. See PRODUCTION_PROGRESS.md 22 Oct and G:/rudra_training/production_eval_20261019/
test_findings_run02_best_step16200/.

**Latest — run02 completed; audit, shortlist and production tooling ready, full-res eval NOT run (21 October):**
[Report, commands, evidence](ai/PRODUCTION_CONTINUATION_2026-10-21.md). Run02 (`fxtd_native1080_hdr_run02`)
status completed 20,000/20,000 in 14.3 h, errors.log empty, 0 nonfinite records; 20 numbered
checkpoints + best.pt (= step 16,200, weights only) hashed and untouched. Read from files: process exit
not observable from the agent VM. Real audit code (torch, byte-identical copies): 12/21 pass the PU
screen; run02 late phase clean +0.97 / hard +0.087 dB (crop-level) vs old run late +0.83 / -0.002; old
step 7000's hard +0.14 is a lucky point (neighbourhood -0.06). Frozen shortlist: run02 best.pt(16200),
16000, 19000; refs old 7000/11000 (`G:/rudra_training/production_audits/20261021_run02_agent/`).
Not run: GPU full-resolution comparison (agent has no GPU; G:\datasets access declined). Next:
`.\scripts\evaluate_production_shortlist.ps1 -DryRun` then without -DryRun. SI-HDR protocol v1 and
ToS temporal recipe v1 frozen before any score; new reserved-test, freeze, overlap and resumable-trainer
tools; 169 tests pass (CPU, cloud). Rights: 1,928/2,720 train rows rest only on the owner statement;
weight distribution false for all sources; see docs/production/. Implementation complete only:
no validation pass, no production approval, release_authorized=false.

**Latest — native1080 run lost, restart failed (20 October):** Run01 folder was
recreated 21:23 UTC 19 Oct; earlier checkpoints 1000-5000 are gone. The new
attempt failed before step 1 with `CUDA error: unknown error` at initial
validation. No resume path exists. Recover GPU (reboot, nvidia-smi, check
nvlddmkm/TDR events), then `train_next_hdr_commercial.ps1 -Action Check` and
`-Action Start -RunName fxtd_native1080_hdr_run02`. Old highlight step 7000
remains the evaluation reference. Details in PRODUCTION_PROGRESS.md.

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
next-run handoff
is the primary continuation document. Also copied to
the next-run handoff on G:. Current Z run observed at
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
HDR handoff contains
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
    - It credits FXTD Studios Research only and refuses to push a the assistant
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
