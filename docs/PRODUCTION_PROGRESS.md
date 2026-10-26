# RUDRA production progress

## 25 October 2026 - quantify video decoder mismatch before learned native integration

New CPU-only audit compares exact current native zscale filter against research
swscale filter using two synthetic 10-bit inputs. Original coloured YUV444 FFV1
fixture roundtrips source bytes exactly; neutral fixture is prior HEVC10 ramp.
Native/research maximum differences0.9883 and0.9805 display codes; RMS0.5038 and
0.6432 respectively. Both retain877 neutral ramp levels. This is a systematic
input difference, not evidence of a model regression or one decoder being wrong.
Future native ACES video must match research decoding or receive separate input
and final-output qualification. Existing still parity does not cover this.
Evidence reports/aces_video_decoder_difference_retry1_20261025/report.json includes
filter text, source/code hashes and FFmpeg version. No inference/GPU used.
Initial fixture failed byte roundtrip and remains preserved; corrected generator
tags both raw input and output and uses one encoder thread. No claim is made from
the failed fixture. Training/verifier remain live; model gates unchanged.

## 25 October 2026 - measure longer-run supervision before interpreting results

Added tools/audit_partial_duration_schedule.py, a standard-library-only auditor
for the registered alternating four-frame still/motion recipe. Completed1600
replay matches all six previously recorded crop/frame-exposure counts. Planned
6400 has1013 unique asset/origin crops, versus962 at1600; only428 of19200 later
frame exposures concern51 new crops. Remaining18772 reuse prefix crops.
Evidence reports/duration_schedule_coverage_20261025 includes input hashes and
explicit planned-only scope. No model evaluation, training changes, pixel-count
claim or quality conclusion. Training and verifier confirmed live. Prior turn
was progress (native EXR overwrite prevention); this turn adds duration evidence.

## 25 October 2026 - prevent colliding native EXR outputs

Native ACES-v2 previously mapped shot.png and shot.tif to the same shot.exr.
The complete input/output mapping is now checked before loading weights or
creating output; ASCII case aliases are also rejected for portability. Missing
files, unsupported single-file inputs and empty folders fail at that boundary.
Release CLI build passed. Five actual executable refusal cases passed without
creating output; evidence and binary SHA in
reports/native_aces_v2_setup_20261025/input_preflight.json. Reproducer:
tools/audit_native_aces_preflight.py. No GPU inference was run or parity claimed.
Training PID41812 live at2225/6400, verifier PID46372 waiting; original gates
unchanged. Direct native learned video/app integration still pending.

## 25 October 2026 - prospective metrics validated and printed training prefix checked

79502live last1600/6400;64rounded25-step mean loss summaries match1600control.
Fullhistory check stillpending;wrong single-step audit oracle preserved/corrected.
13synthetic metric/qualification tests pass incl empty-support/per-scene safety.
No prospective evaluation/freeze;fullruntime not yetverified. See
production/AMBIENTCG_HOLDOUT_CANDIDATES_2026-10-25.md finalsection.
Verifier57516waiting;no concurrentGPUexport. Goal incomplete.


## 25 October 2026 - prospective evaluation refuses unqualified checkpoint

OriginalDEVnumeric/hash/requiredcheck guard implemented;7tests pass. Actual1600
preflight rejected before model/newdata evaluation. Evaluatorprepared with48case
completeness,one-checkpoint freeze,rawreference metrics/FLOAT masters;fullruntime
unrun. No freeze marker yet. See production/AMBIENTCG_HOLDOUT_CANDIDATES_2026-10-25.md
latestsection. Training79502last1325/6400,verifier57516waiting;goal incomplete.


## 25 October 2026 - prospective two-scene still protocol frozen and built

2screened ambientCG scenes x4views x3EV x8/10bit=48sources,24trueHDRreferences.
Pre-registered fixedrecipe;CPUbuild and independenthash/shape/dtype checks pass.
No model evaluation until originalDEVpass and checkpointfreeze. Not fullholdout
qualification;thirdscene excluded. See production/AMBIENTCG_HOLDOUT_CANDIDATES_2026-10-25.md
finalsection; corpus ambientcg_prospective_stills_20261025. Training79502last775/6400,
verifier57516waiting. Goal incomplete;no promotion.


## 25 October 2026 -2979 panorama comparisons completed and nearest-source sheets reviewed

6667terminal exit0:993x3scores,zeroerrors,3positivecontrols pass.3skyflags
retained;visualfields/clouds differ. All3nearest5sheets inspected. DaySky071A
vs hilly_terrain_01 terrain similarity unresolved;not cleared. Other2sources
can proceed protocol preparation within audit scope,not certifiedholdout.
See production/AMBIENTCG_HOLDOUT_CANDIDATES_2026-10-25.md latestsection.
Training79502 last525/6400,verifier57516waiting;goal incomplete.


## 25 October 2026 - partial panorama flags visually inspected without threshold changes

Training79502 last325/6400,verifier57516waiting. CPUaudit6667 last950/993,
PID42836 live. First2skyflags visually different terrain/clouds;originalflags
retained,plains_sunset still unreviewed. No certification. Finalreport pending;
after completion run tools/render_holdout_similarity_review.py and inspect3
sheets. See production/AMBIENTCG_HOLDOUT_CANDIDATES_2026-10-25.md.
No GPUexport concurrency;goal incomplete.


## 25 October 2026 - user-prioritized RUDRA training retry starts

User explicitly authorized stopping separate videoPID11592; identity verified,
only that process stopped. Existing79502 retry now trainingPID41812,launcher28076;
verifier57516 waits. Last25/6400. Pools/source snapshots/first1600schedule exact.
Full history check remains after completion;do not restart. CPUaudit6667 last
500/993 with skyflags needing review. No parallel GPUexport. See
production/PARTIAL_DURATION6400_RETRY_2026-10-25.md latestsection. Goal incomplete.


## 25 October 2026 - rotation/exposure-aware panorama overlap screen launched

3tests pass; audit6667 compares3new sources against993known panoramas, fixed
.98correlation flag, yaw/exposure controls and explicit errors. Last50/993;
no GPU/model use. Poll6667; inspect finalreport before eligibility. Still
quarantine,not independent-holdout proof. See production/AMBIENTCG_HOLDOUT_CANDIDATES_2026-10-25.md.
Training79502/verifier57516 still queued behind videoPID11592;goal incomplete.


## 25 October 2026 -79,083-file old-storage screen completes without candidate matches

Expanded sources+G:/rudra_v02 screen79,083media files,zero exact/path matches
or errors. Older-storage metadata scan complete/no references. Not a perceptual
near-duplicate proof;3candidates remain quarantine,not holdout-certified.
See production/AMBIENTCG_HOLDOUT_CANDIDATES_2026-10-25.md. Bothscreen jobs
terminal. Retry79502/57516 still queued behind live videoPID11592. No newmodel
results;goal incomplete.


## 25 October 2026 - provider colour declaration verified and float references prepared

Provider Blender files explicitly mark3EXRs LinearRec709; inspected autoexecoff.
OfficialOCIO conversion toACEScg unitexposure, no clamp,3FLOAT/AP1 masters with
exactreadback and roundtrip checks. Still quarantine,not model outputs/holdout.
Evidence G:/datasets/corpora/ambientcg_quarantine_acescg_20261025/report.json.
See production/AMBIENTCG_HOLDOUT_CANDIDATES_2026-10-25.md finalsection.
Retry79502/57516 remain queued;goal incomplete,no promotion.


## 25 October 2026 - alternate CC0 photographic HDR candidates screened and acquired

All10remaining PolyHaven candidates have prior source references. NewambientCG
3x2k candidates: selected families absent from historical text scan; commercial
CC0 receipt, ZIP sizes/CRC/SHA256 and finiteRGB checked. Not certified holdout.
EXRs lack chromaticities;confirm source basis before conversion;no model run.
See production/AMBIENTCG_HOLDOUT_CANDIDATES_2026-10-25.md. Retry79502/57516
still queued behind live videoPID11592. Goal incomplete;no promotion.


## 25 October 2026 - historical corpus references disqualify four holdout candidates

Reference scan85742 complete:all4already in run05/PeakNet corpora,3train/1val
in source manifest. Explicit holdout_eligibility=false saved; downloads preserved.
Name/content screen missed converted G:/rudra_v02/pairs_moves sources. Future
selection must screen historical scene IDs before download. No new holdout
coverage claimed. See production/HOLDOUT_CANDIDATES_2026-10-25.md final section.
Retry79502/57516 still queued behind videoPID11592;goal incomplete.


## 25 October 2026 - holdout candidates screened against local source inventory

28,969EXR/HDR paths screened,zero exact-byte duplicates or missing files.
Coordinate screen no known sourcewithin1km;265knownIDs lack coordinates,
road shares broader region. No independence claim;all4remain quarantine.
Historical reference scan85742 still live. Retry79502/57516 still queued behind
confirmed videoPID11592. See production/HOLDOUT_CANDIDATES_2026-10-25.md.
No GPU concurrency,threshold change or promotion;goal incomplete.


## 25 October 2026 - quarantined four new CC0 HDR source candidates

4x2k PolyHaven HDRIs,56.49MB, providerchecksums and finiteRGB verified; official
license/API receipts saved. Deterministic metadata-only selection; no model
evaluation. Remain quarantine pending ancestry/content/scene overlap audit.
Reference scan85742 LIVE; empty output so far is not no-match proof. See
production/HOLDOUT_CANDIDATES_2026-10-25.md. GPUretry79502/57516 still queued
behind videoPID11592;do not duplicate. Goal incomplete; no promotion.


## 25 October 2026 - HEVC Main10 procedural export transport passes

Two original10bit ramp frames preserve877 levels vslegacy256; sourceHEVC Main10
YUV codes exact. CPUguarded600 export FLOAT masters/readback and PQ16 video
decode exact;all10checks pass. Scope is procedural/lossless, not real footage
or model qualification. See production/ACES_HEVC10_TRANSPORT_2026-10-25.md.
Retry79502/57516 still live waiting on videoPID11592;do not duplicate or run
GPU tests. No quality thresholds changed;goal incomplete.


## 25 October 2026 - duration comparison verifier supports isolated retry

Summary helper now supports6400 retry paths and stronger checkpoint/identity/
TRAIN-audit checks. Completed1600 replay reproduces all saved metrics/failures
exactly in reports/duration_summary_verifier_20261025. No new quality claim.
Sessions79502/57516 confirmed live waiting; videoPID11592 still active.
See production/PARTIAL_DURATION6400_RETRY_2026-10-25.md for final summary command.
No duplicate launch; no concurrent native GPU test. Goal incomplete.


## 25 October 2026 - controlled6400 retry queued behind existing GPU job

Retry training session79502/launcherPID28076 waits for videoPID11592 and GPU
memory<6000MiB. Coordinator57516 waits for retry folder, then launcher lifetime.
Original failed6400 artifacts preserved. New root partial_pcgrad_duration6400_retry1_20261025
under G:/rudra_training. Identical trainer verified and registration saved.
Do not restart waiting jobs or run simultaneous native GPU checks. See
production/PARTIAL_DURATION6400_RETRY_2026-10-25.md for paths/verification.
Original31352/92141 terminal failed.1600 still latest completed;goal incomplete.


## 25 October 2026 - CUDA loader fix validated;6400 interrupted by GPU error

All4native targets build,394tests pass. CUDA export85883 explicitly stopped;
no GPU parity result. Training31352/PID35572 exit1 illegal CUDA memory access
after last375; coordinator92141 exit1 missing report. BOTH terminal, not live.
Fresh CUDA health passes; unrelated video generationPID11592 confirmed live,
left untouched. Defer retry until it ends; preserve failed6400 artifacts.
Next identical controlled6400 retry, with GPU workloads serialized. Exact cause
not established. See production/NATIVE_ACES_CUDA_LOADER_2026-10-25.md.
1600 latest completed; all3grading gates fail. Goal incomplete; no promotion.


## 25 October 2026 —1600 comparison complete;6400 controlled duration run launched

FixedTRAIN0EV1.705618->1.167375,photo .147135->.135141; aggregateEVs improve,
motion-3 worsens. All original DEVgrading failures retained;video/preservation
pass. Aggregate and hashes verified, priorjobs terminal. Bounded6400 unchanged
recipe launched31352/PID35572; verifier92141 waits. Pools/1600schedule prefix
exact; complete history check pending. No promotion. See
production/PARTIAL_DURATION1600_FINAL_2026-10-25.md.

## 25 October 2026 — provenance record clarified while fit audit runs

Read retained527row declared-data audit:399CC0 receipt matches,128procedural,
527content checks passing,no unresolved declared rows. Not a new source audit.
Rechecked root recipe hashes against current files:4/5 differ,reference matches.
Archive's earlier search records missing historical source; exact root replay
currently unavailable. Updated THIRD_PARTY_LICENSES.md to distinguish this
reproducibility gap from evidence of restricted initialization; no weight licence
or release authorization chosen. Coordinator17987 remains live in TRAIN fit.

## 25 October 2026 — duration1600 video and preservation pass

All8Hunyuan48 checks pass,.01378024EV flicker. Independent24frame preservation
passes all stages/frozen weights. Coordinator17987 now running fixed TRAIN fit.
Saved1600schedule covers962crops vs622first400; additional coverage quantified,
so improvement cannot be assigned solely to repeated exposures. Original3grade
failures persist. See production/PARTIAL_DURATION1600_INTERIM_2026-10-25.md.

## 25 October 2026 — duration1600 improves reconstruction but grading still fails

Paired120 and grading24 complete; grade input hashes correct. NewH .251418,
oldH .381858,aligned .008860 pass; legacy .778691 still worse thananalytic.
Grades18.763108/.073843/.004582 all fail original limits despite improving400.
Coordinator17987 live in video, then preservation/TRAINfit. Final comparison
helper prepared. See production/PARTIAL_DURATION1600_INTERIM_2026-10-25.md.

## 25 October 2026 — native float masters and PQ16 transport verified

Two720p nativeCPU/DirectML exports pass independent32bitFLOAT/AP1/mask checks;
CPU equalsPython, DMLwithinregistered tolerance. Derived FFV1 PQ16 decode exact.
1600training terminal, first400history exact; coordinator17987 evaluating. Grade
path bug corrected before queued evaluation, no threshold changes. Goal remains
incomplete. See production/NATIVE_ACES_MASTER_VERIFY_2026-10-25.md.

## 25 October 2026 — Windows native parity and394 tests pass

Root build cause was obsolete /permissive override, removed without dependency
edits. All4targets build on original compiler.394tests pass;2real720p frames
bit-exact C++/Python; research ONNX export passes CPU tolerances. Native jobs
terminal. No release/DirectML/native export qualification yet. Training97117
last1475/1600; coordinator17987 waiting. See production/NATIVE_ACES_WINDOWS_PARITY_2026-10-25.md.

## 25 October 2026 — native compiler/header failure isolated; compatible build launched

MSVC19.51 build failed in Torch headers. Installed19.44 passes minimal actual
header compile. New Ninja configuration succeeds; build39003 live. No Torch
patch/downgrade or training changes. Training97117 last725/1600; coordinator17987
waiting. Native runtime tests pending. See production/NATIVE_ACES_COMPILER_2026-10-25.md.

## 25 October 2026 — native Windows build configured with pinned OCIO

Machine vcpkg already has OCIO2.5.2; no download needed. CMake configuration
succeeds with LibTorch/OpenCV/ORT. Release build63214 live at parallel2; runtime
tests/parity pending. Training97117 last525/1600; coordinator17987 waiting.
Evidence: production/NATIVE_ACES_BUILD_2026-10-25.md. No qualification claim.

## 25 October 2026 — Windows native preflight blocker repaired during training

Corrected Torch root path and verified installed headers/library. Native full
suite failure now stops gate instead of allowing misleading ALL PASSED. Parser
validation passes; full native build remains unverified. Training97117 live at
325/1600, coordinator17987 waiting. No model promotion. See
production/NATIVE_ACES_PREFLIGHT_2026-10-25.md.

## 25 October 2026 — controlled1600 update run launched

Original deterministic uncappedPCGrad recipe, only duration changed. Pools and
first400schedule exact. Training session97117 live,last150/1600; coordinator17987
waits on training process, then requires400history identity before full quality,
video,preservation and fixedTRAINfit evaluation. Six focused tests pass.
No result/qualification yet. See production/PARTIAL_DURATION1600_2026-10-25.md.

## 25 October 2026 — difficult training crops fit with a reconstruction tradeoff

Eight finite-parameter checks agree in sign; max relative error .73615%.
Two100update fixed-batch diagnostic runs completed. Combined0EV25.666979->.875311,
but photo .499571->.602405 and-3EV576.61->620.93 regress. No promotion.
Frozen parameters exact;6tests pass. Initial graph-version failure repaired and
failed output retained. Next controlled deterministic1600 duration comparison,
requiring first400history identity then full gates, not a hard-crop release.
See production/PARTIAL_HARD_CROP_FIT_2026-10-25.md.

## 25 October 2026 — full TRAIN fit reveals concentrated residual error

622unique saved crops,1600frame and982607pixel exposures at four frozen states.
Uncapped raw0EV1.705618 versus analytic .080820; capped12.755371/direct12.996763.
Frame-weighted and both-domain errors also remain above analytic. Top10crops
account66.40% uncapped/72.05%direct error. Exact weights/masks, independent
analytic calibration and exposure counts verified; one aggregation test passes.
Audit terminal. Next hard-TRAIN-crop derivative and bounded fitting diagnostic,
not blind longer training or relaxed gates. See production/PARTIAL_TRAINING_FIT_2026-10-25.md.

## 25 October 2026 — direct projected updates pass+3EV,0EV remains unresolved

400deterministic updates with saved capped movement schedule; no Adam moments.
Requested lengths exact,actual99.262% afterrounding; pools/schedule/EMA exact.
Direct grades-3/+3 pass,0EV .103474 fails. Reconstruction/motion/detail/video/
preservation pass;15tests pass;alljobs terminal. Next fixed TRAIN fit comparison
over full saved supervision before choosing longer training. No promotion or
quality limit changes. See production/PARTIAL_PCGRAD_DIRECT_2026-10-25.md.

## 25 October 2026 — update cap improves -3/+3 grading but0EV regresses

Deterministic capped400 and saved deterministic control fully evaluated. Cap
passes reconstruction/motion/detail/video/preservation and-3EV;0EV .103647 and
+3EV .002594 fail. Actual400step bounds verified; pools/schedule/EMA exact.
13combined tests pass; alljobs terminal; no promotion/default/gate changes.
Next direct normalizedPCGrad with saved capped movement schedule to isolate
direction at matched movement budget. See production/PARTIAL_PCGRAD_CAP_2026-10-25.md.

## 25 October 2026 — applied-step probes validated through exact deterministic replay

Default-mode replay failed tensor/history identity; retained as separate diagnostic.
Opt-in deterministic2step and400step pairs match full model/history/pools/schedule.
Eight TRAIN probes: actualAdam raises total2/8,matched projected4/8; both1e-4
direction variants0/8. Seven tests pass; snapshots saved; alljobs terminal. Next
bounded actual-Adam L2cap comparison against saved deterministic control, both
requiring full quality evaluation. No promotion or threshold changes. See
production/PARTIAL_PCGRAD_STEP_REPLAY_2026-10-25.md.

## 25 October 2026 — PCGrad improves reconstruction, fails all grading gates

Opt-in algorithm1 projection implemented/tested. Exact pools/schedule/EMA versus
control;400updates complete. NewH .274009/oldH .380173 pass; -3/0/+3grading
18.927318/.078361/.010324 fail. Video/preservation pass;16tests pass; alljobs
terminal. Adam applied direction/length differs materially from locally checked
projected descent. Next diagnostic replay with measured post-step losses and
restored counterfactuals, final tensor identity required. No promotion or default
change. Evidence: production/PARTIAL_PCGRAD_2026-10-25.md.

## 25 October 2026 — local parameter-gradient predictions validated

At1e-4 symmetric L2 steps,all288 derivative signs match,median relative error
.000125,p95.022912. Same batches/baseline losses as prior audit; exact restoration.
Five tests pass; audit terminal. Local conflict confirmed, unseen-image benefit
unproven. Next bounded opt-in PCGrad400 comparison with original schedule/losses/
quality gates and actual Adam-step diagnostics. No production changes. See
production/PARTIAL_PARAMETER_PROBES_2026-10-25.md.

## 25 October 2026 — parameter-gradient conflict measured on TRAIN

32frozen evaluations: common-controlEMA direction adverse to photo10/32,
0EV1/32,temporal6/16; fixed0EV0/32. 0EVweighted norms usually much larger in
sample medians. Conflict is real locally but not sufficient explanation for
development grading. Three diagnostic tests pass; weights exact, audit terminal.
Next parameter-space symmetric finite-step validation before optimizer change.
See production/PARTIAL_OBJECTIVE_CONFLICT_2026-10-25.md.

## 25 October 2026 — group-preserving coverage trial fails qualification

400updates complete,1,009,513pixel exposures exactly match prepared schedule.
NewH .284811 passes but oldH .383616,0EV .094277,+3EV .002576 fail. Video and
preservation pass;14tests pass; alljobs terminal. Sampling alone has not solved
the tradeoff. Next TRAIN parameter-gradient conflict diagnostic, motivated by
PCGrad/CAGrad primary sources; no optimizer/default/gate change. See
production/PARTIAL_GROUP_COVERAGE_2026-10-25.md.

## 25 October 2026 — sampling shifts isolated; matched-group schedule ready

Source-only351TRAIN audit reproduces79still crops exactly. Eight excluded rows
have only111 reachable pixels; minimum64candidate rule explains all. Unconstrained
coverage shifts scene/exposure distribution. New schedule preserves every control
motion slot group and still batch while increasing motion union40.3% to412679.
Fourteen tests and actual400batch validation pass. No eligibility/loss/gate change.
Next controlled400 training/evaluation; no training job launched in this step.
See production/PARTIAL_SAMPLING_BALANCE_2026-10-25.md.

## 25 October 2026 — coverage schedule improves grading but misses two limits

400updates complete; motion union473275 versus294071 with identical pools and
still batches. 0EV .068653 improves21.4% but fails; +3EV .002155 now passes.
NewH .292812 misses5% threshold; oldH/motion/detail/video pass. Preservation exact,
13tests pass, alljobs terminal. No promotion or gate changes. Next TRAIN schedule
distribution and still-pool exclusion audit to guide controlled sampling work.
See production/PARTIAL_COVERAGE_SCHEDULE_2026-10-25.md.

## 25 October 2026 — scheduled partial-pixel coverage measured

All351 TRAIN rows hash-verified. Actual223frames,320158 unique partial pixels;
12.81% overall/27.88% reachable coverage. Motion schedule covers62.14% of its
pool support; still100%. 98 partial-containing frames never sampled,34 with
reachable support. Repeated exposures match independent calibration exactly.
Evidence supports a controlled coverage-oriented schedule experiment, not a
causal claim about grading. No parameter/gate changes; audit terminal.
See production/PARTIAL_TRAINING_COVERAGE_2026-10-25.md.

## 25 October 2026 — appearance constraint improves errors; coverage question remains

Fixed TRAIN8-scene crop sample has0/1 corrected pixels of1168 for EMA/fixed.
Old DEV partial region207494pixels has35.19%/35.42% correction. Raw models already
fail relative to analytic; correction improves average grading. Raw/final model
rankings differ. No causal conclusion from unequal TRAIN/DEV sampled regions.
Next audit scheduled TRAIN partial-pixel coverage before changing sampling/loss.
Four smoothing tests pass; both audits terminal; no model changes. Evidence:
production/PARTIAL_CONSTRAINT_GAP_2026-10-25.md.

## 25 October 2026 — fixed normalization trial rejected

400updates with byte-identical pools/schedule to EMA control.0EV .085134 and+3EV
.003077 improve slightly but still fail; oldH .387133 fails. NewH, aligned residual,
detail and video pass; preservation exact. All jobs terminal. No promotion/default
or threshold change. Next TRAIN raw/constrained pipeline mismatch audit before
more weighting changes. Evidence: production/PARTIAL_FIXED_SCALE_2026-10-25.md.

## 25 October 2026 — display gradients audited; fixed TRAIN scale prepared

Moderate directional probes agree in sign on96checks; tiny probes show noise.
+3EV EMA coefficient varies100x; no floor activation. Fixed denominators calibrated
on all1600scheduled TRAIN frames/982607pixels, no development inputs. Opt-in
fixed-scale training mode validates and snapshots values. Nine tests pass, audits
terminal, no weight change/promotion. Next matched fixed-scale temporal1 experiment
with original gates. Evidence: production/PARTIAL_DISPLAY_GRADIENT_2026-10-25.md.

## 25 October 2026 — partial temporal comparison completed; grading still fails

Matched400updates,538 motion pairs/79still crops,228 frames. Temporal1 passes newH
.285187, oldH .381566, aligned .009988, video .013195EV but fails0EV .087391 and
+3EV .003335. Temporal0 aligned slightly better; sampling change is a confound
against earlier independent-crop results. Exact frozen/output preservation;
nine tests pass, all jobs terminal, no promotion. Next TRAIN display-gradient
and relative-scaling audit before more fitting. Evidence:
production/PARTIAL_TEMPORAL_PAIR_2026-10-25.md.

## 25 October 2026 — matched phase training completed; neither qualifies

Two400-update partial runs with same832 centres/212frames and batch schedules.
Aligned geometry improves oldH to .382479(pass),0EV to .084570(still fail), and
motion residual to .010595(still fail). NewH .301169 fails. Both video48 pass;
full/outside-highlight preservation independently exact. Eleven tests pass.
All jobs terminal; no promotion/tolerance changes. Next aligned adjacent-pair
temporal training comparison with separate still replay and original gates.
Evidence: production/PARTIAL_PHASE_PAIR_2026-10-25.md.

## 25 October 2026 — partial crop phase mismatch measured and correction added

Eight TRAIN scenes: aligned max asinh error9.54e-7 versus shifted .182714.
Wider excluded boundary alone does not fix it. Explicit aligned-crops trainer
option added with geometry validation/snapshot; original control retained.
Seven tests pass; audit terminal, no weights changed or promoted. Next matched
legacy/aligned training with unchanged full quality gates and frozen-region check.
Evidence: production/PARTIAL_CROP_GEOMETRY_2026-10-25.md.

## 25 October 2026 — partial fine-tune preservation independently verified

All24 old DEV frames preserve fully clipped/non-highlight outputs exactly across
raw, hybrid64 and appearance-constrained stages. Frozen weight tensors exact;
partial tensors changed. No model qualification or tolerance adoption. Script
compiles; audit terminal. Trainer actually used229 eligible frames/916 crops from
351-row pool. Next measure arbitrary-phase/48px-border crop equivalence on TRAIN
before extending this recipe. Evidence: production/PARTIAL_PRESERVATION_2026-10-25.md.

## 24 October 2026 — trained context reaches remotely but adds no useful blind detail

Four TRAIN images,43 scheduled crops audited. Blind MSE parent7.25548/local7.30995/
coarse7.36265; both residuals worsen all four. Coarse responds to distant context
but remains spatially near-constant; detail error is essentially inherited.
Seven tests pass; corrected stable-variance rerun complete, no weights changed.
Next test a spatially varying coarse encoder/decoder bottleneck, not a global
mean alone, then matched bounded training with every original gate unchanged.
Evidence: production/ACES_LEARNED_CONTEXT_2026-10-24.md. Goal remains incomplete.

## 24 October 2026 — wider-context comparison complete; neither qualifies

Implemented frozen-parent zero-output local/coarse residuals. Matched600 updates,
3,251 parameters, initial weights and schedule verified; eight tests pass.
New highlight error .290632/.288728 passes, but full detail,0EV grading and
Hunyuan48 flicker .023176/.022415 fail. Coarse old highlight .384976 also fails.
All evaluations terminal; no default/promotion change. Goal remains incomplete.
Next audit actual learned context use and inherited/added error on certified-blind
TRAIN replay regions before further training. Evidence:
production/ACES_WIDE_CONTEXT_2026-10-24.md.

## 24 October 2026 — spatial context limit verified

Full branch conservative54px reach confirmed by independent structural tests and
trained perturbation probes. All351 TRAIN images audited: four replay images have
110,771 certified-blind white pixels with varying targets. Scheduled crops place
19.753% of replay full supervision there, across43/300 exposures. No motion pixels
meet the conservative criterion; this does not explain all failures or establish
irreducibility. Five tests pass; no parameter updates; all jobs terminal.
Next zero-initialized coarse full-image context residual, frozen existing branches,
crop-consistent conditioning, then bounded comparison and unchanged quality gates.
Evidence: production/ACES_RECEPTIVE_CONTEXT_2026-10-24.md.

## 24 October 2026 — full balanced-domain gradient audit complete

All100 TRAIN windows at initial/sequential/accumulated states:1800 sample-state
and300 window-state rows. Domain conflicts43/55/47; replay/motion norm medians
1.146/1.509/.740. Blanket replay dominance is not supported. Accumulated fits both
TRAIN domains better but both final models worsen6/8 new DEV scenes. Heavy clipping
improves much less in fitting. No parameters changed; all jobs terminal.
Eleven geometry/batch/temporal tests pass; first initial training loss reproduced.
Next audit usable spatial context/input ambiguity in fully clipped TRAIN regions,
without assuming missing context or selecting by DEV failures. No weight guess.
Evidence: production/ACES_BALANCED_DOMAIN_AUDIT_2026-10-24.md.

## 24 October 2026 — balanced accumulation comparison complete; both rejected

Same600-exposure schedules exact; all44 eligible scene/domain entries visited.
Sequential600 versus accumulated100 Adam updates, identical parent/seed/losses.
New H .337593 versus .370059; old H .350442 versus .324451; full detail .094291
versus .090731. Both fail new reconstruction/full detail/old detail/0EV grade.
Hunyuan48 .044291 versus .054989EV both fail .02; other preservation checks pass.
Parent and all120 outside-full outputs exact. Ten focused tests pass; counters
verify update budgets. All jobs terminal; no checkpoint/default promotion.
Next audit motion/replay gradient magnitude/direction and scene/exposure tradeoffs
on frozen schedule before targeted loss/representation changes; no blind extension.
Evidence: production/ACES_BALANCED_ACCUMULATION_2026-10-24.md.

## 24 October 2026 — all cyclic retention rotations complete; fixed-panel path rejected

Same18 TRAIN scenes rotated through update/anchor/monitor; all scenes monitored
once. Current-only accepts27/36, panel-aware7/36 (6,1,0 across folds). Monitor
photo mean regresses2.178% versus0.372%; panel-aware detail improves only0.121%
and temporal1.180%. Zero-update fold is not improvement. Regional regressions remain.
No main-trainer adoption, inference checkpoint or default change. All jobs terminal;
identity/coverage/initial-value/rollback/parent checks pass. Aggregator compiles.
Next bounded balanced multi-scene gradient-accumulation comparison on full eligible
TRAIN, explicit update/exposure budgets, research checkpoints and all original gates.
Evidence: production/ACES_ANCHOR_ROTATION_2026-10-24.md.

## 24 October 2026 — panel-aware projection moves, but monitor accuracy regresses

Three-arm matched12-attempt TRAIN pilot complete. Current-only12 accepted,
rejection-only0, panel-aware direction6. All protected aggregate anchor metrics
preserved, but monitor photo mean worsens1.43% and4/6 members regress (control2/6).
One anchor and one monitor near-region detail also regress. No adoption or model saved.
23/24-constraint NNLS projection feasibility/KKT verified; finite-step checks remain.
Eleven tests pass. Control decisions reproduce; largest metric drift3.82e-7.
Exact rollback/frozen-parent/original-branch checks pass; all jobs terminal.
Next cyclic role-rotation TRAIN check before broader retention-policy training.
Evidence: production/ACES_ANCHOR_DIRECTION12_2026-10-24.md.

## 24 October 2026 — strict anchor retention stalls; pilot complete

Eighteen disjoint TRAIN scenes,12 matched attempts per arm. Current-only accepts12;
strict six-member panel gate accepts0 and rejects all48 locally passing scale trials.
Exact rollback and optimizer counters verified; original branch restored. Current-only
means improve, but photo error worsens on1 anchor and2 monitor scenes; no aggregate
detail or temporal member regression over this short pilot. This does not reproduce
long-run detail degradation. Panel gate costs372 vs48 evaluation calls,88.73 vs12.01s.
Eight tests pass; all jobs terminal. No inference checkpoint/default change.
Next diagnose rejecting anchor objectives and test panel-aware update directions
under the same finite-step gate; do not adopt this stalled rejection-only policy.
Evidence: production/ACES_ANCHOR_RETENTION12_2026-10-24.md.

## 24 October 2026 — guarded600 failure diagnosis complete

Projected clipped-detail error worsens in all eight new scenes and all 48 medium/
heavy-clipping images. Fixed region audit (48 TRAIN, 96 coherent DEV) finds new
full-deep RGB-asinh gradient error +107.2% versus parent; selected TRAIN motion
regions also regress. Current-batch protection does not prove cross-scene retention.
Matched Hunyuan attribution reproduces failure at frame19->20; both mask and field
vary substantially, and colour constraints reduce but do not remove the error.
Four diagnostic tests pass; scene aggregates reproduce acceptance to1e-12.
All jobs terminal, no training/default change. Next: preregister a bounded TRAIN
anchor-panel retention comparison with unchanged DEV gates, not metric replacement.
Evidence: production/ACES_GUARDED_DIAGNOSIS_2026-10-24.md.

## 24 October 2026 — guarded600 complete; precision and transport verified

Both matched 600-attempt runs are terminal and rejected. Adam accepts 116 updates,
projected 381; projected improves new highlight error to .292532550 but misses the
fixed 5% target and full-detail gate. Old 0 EV grading and video stability also fail.
Research export CLI now preserves 16-bit decode precision; historical benchmark
API default remains 8-bit. Optional inverse-boundary repair passes 48-frame
appearance but flicker .021655090 EV still exceeds .02. Two four-frame EXR/16-bit
video exports independently decode bit-exact with correct metadata. 28 tests pass.
527 declared parent data rows audited (399 CC0, 128 procedural); historical source
reproducibility and distribution closure remain separate. No model promotion,
release, signing or native neural integration. All launched jobs terminal.
See production/ACES_GUARDED_MIXED600_2026-10-24.md,
production/ACES_INPUT_PRECISION_2026-10-24.md and
production/GOAL_REMAINING_2026-10-24.md for results and remaining requirements.

## 24 October2026 — protected updates retain advantage over five-step trajectories

Protected81 updates vsAdam45 across24 fixed TRAIN starts with persistent moments;
final trained-pair pass21/24 vs13/24, neighbouring pair18/24 vs10/24.
Median temporal improvement17.26% vs1.18%;13 protected trajectories stop early.
Neighbouring detail/temporal regressions remain.32 exact rejected-step rollbacks;
parent preserved and starting branch restored.5 tests pass; all jobs terminal.
No inference checkpoint or default change; no generalization/release claim.
Next bounded mixed-scene/replay comparison with equal attempt budgets and unchanged
quality gates. Evidence: production/ACES_PROTECTED_TRAJECTORY5_2026-10-24.md.

## 24 October2026 — protected update passes bounded matched TRAIN pilot

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

## 24 October2026 — gradient conflict and clipping-stratum regression localized

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

## 24 October2026 — active-union normalization improves flicker but fails quality

Matched600 updates complete. Hunyuan48 improves35.12% to0.043198491EV but fails0.02.
New H0.342798894 and fully-clipped detail0.088140910 worsen; old H0.326939908 improves.
Old overall/clipped detail and0EV grade fail; -3/+3EV pass. Aligned0.008791357 passes.
Parent and all120 outside-full outputs exact;96-frame shared/isolated audit passes.
302 finite gradient logs, temporal/(photo norm+detail norm) median0.02871/max0.94730;
this does not measure alignment. Next audit gradient conflict and scene regressions
before choosing another objective change.12 tests pass; all jobs terminal.
No promotion/default or actual export change. Goal incomplete.
Evidence: production/ACES_FULL_ACTIVE600_2026-10-24.md.

## 24 October2026 — temporal gradient dilution measured

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

## 24 October2026 — temporal support comparison rejected

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

## 24 October2026 — multi-scene full-context600 rejected after complete evaluation

-600 TRAIN updates complete; identical learned weights saved under shared/isolated
  routing. Parent/outside-full preserved exactly;96-frame independent routing audit.
- Isolated new H0.313047858 and clipped detail0.085965048 fail; old H improves to
  0.343326694. Shared routing damages partial reconstruction and performs worse.
- Hunyuan48 temporal0.061841547EV fails0.02; other7checks pass. Old clipped detail
  and0EV grade fail. No promotion/default or actual export change.
- Fixed48 TRAIN full fit improves9.8%/14.4% across motion/replay, demonstrating
  fitting gains without sufficient new-scene/video performance.
- Next measure full-support transitions versus learned-field variation before
  another registered change.16 tests pass; tools compile; all jobs terminal.
- Evidence: production/ACES_FULL_CONTEXT600_2026-10-24.md. Goal incomplete.

## 24 October2026 — isolated full branch passes bounded fitting pilot

- Added opt-in experimental v9 branch and strict factory support; existing context
  trainer rejects this architecture to prevent optimizing a frozen branch silently.
- Three matched600-update TRAIN arms complete. Full-only fit improves98.56%; the
  isolated/shared-full arms produce identical full-region results, but only isolation
  preserves partial/ordinary/shadow output exactly. Initial equivalence verified.
- Isolated boundary and high-region temporal errors improve versus initial; shared
  balanced still has better overall/partial/boundary fit. No generalization claim.
- Nine full/context/detail plus12 existing model/restorer/export tests pass; modules
  compile. Snapshots, predictions and reports saved; no production checkpoint/default.
- Next bounded multi-scene TRAIN comparison, then unchanged quality gates. All jobs
  terminal. Goal remains incomplete.
- Evidence: production/ACES_FULL_BRANCH_PILOT_2026-10-24.md.

## 24 October2026 — highlight strata and TRAIN gradient diagnosis complete

- Audited fixed48 TRAIN + complete96 coherent DEV with source-defined partial,
  full-boundary/near/deep regions. No optimization or default change.
- Sampler-v2 improves new deep clipping but worsens partial and old deep accuracy;
  new/old deep signed residuals have opposite signs. Colour constraint is not the
  main source of fully-clipped regression. TRAIN fitting also remains weak.
- Local gradient opposition in21/40 eligible TRAIN crops/frames supports testing
  separate full-region learning, not a claim that interference is the only cause.
- Next bounded TRAIN pilot: isolated full branch with exact initial/outside-support
  invariants and a shared-branch control, then unchanged quality gates. Not run yet.
- Two diagnostic tests pass; tools compile. All jobs terminal; goal incomplete.
- Evidence: production/ACES_HIGHLIGHT_STRATA_2026-10-24.md.

## 24 October 2026 — sampler-v2 controlled run complete; remaining goal mapped

- Same-parent/rate/seed6000 trial complete. Frozen weights exact; scene visits
  match pre-run v2 audit, zero empty replay updates. No checkpoint promotion.
- New72 H improves2.29% to0.308351923 but fails analytic; old24 H worsens4.59%
  against matched control, although it retains the older fixed-cap pass.
- Fully-clipped detail improves versus control but fails analytic on both sets.
  Old0EV grade fails; -3/+3EV pass. Aligned residual retains its fixed-cap pass.
- Hunyuan48 all eight checks pass, maximum0.014813336EV. No actual transport
  re-export for this rejected model. Fixed worst-control visual contact inspected.
- Saved complete seven-gate/app/export/provenance checklist; confirmed current
  research video reader reduces input to8-bit despite16-bit output capability.
- Primary-paper cross-check documented; no external data/code/weights imported.
- Next diagnose clipped interior/boundary and scene regressions before another
  registered change. All jobs terminal; broader goal incomplete.
- Evidence: production/ACES_SAMPLER_V2_2026-10-24.md and GOAL_REMAINING_2026-10-24.md.

## 24 October 2026 — rate comparison complete; empty replay updates repaired

- Matched6000 at1e-4 improves new DEV H2.21%, old coherent H5.29%, and aligned
  motion residual27.8% over1e-5. Hunyuan48 passes all checks at0.015537215EV.
- Reject adoption: new H remains worse than analytic; fully-clipped detail is
  worse on both new and old coherent DEV. Old0/+3EV grade gates also fail.
- Legacy sampler replay exactly matches recorded scene visits and finds61 empty
  replay updates. Opt-in v2 retains selected interior highlights and handles
  odd image dimensions. Its6000-decision audit finds zero empty updates and
  excludes one border-only row. This changes sampling; no v2 model trained yet.
- Crop-first preparation is bit-exact in tests; full training speedup unmeasured.
  15 focused tests, compilation and whitespace checks pass. No default promotion.
- Next: controlled v2 sampler training comparison, then existing DEV/detail/video
  gates. Full commercial/release objective remains incomplete.
- Evidence: production/ACES_FIT_SANITY_2026-10-24.md.

## 24 October 2026 — TRAIN sanity establishes fitting ability; matched rate trial

- Fixed TRAIN pair, identical-parent600-update arms: combined objective at1e-4
  cuts photo error88.7%, improves detail/motion and within-TRAIN phase probe.
- Matched1e-5 control fits more slowly; this identifies a useful optimization
  diagnostic, not new-scene validation. Frozen weights/outside support stay exact.
- Broad matched6000 trial running: same parent/seed/data/losses/sampling as prior
  balanced6000, only learning rate changes. No early stopping or default promotion.
- Six context/detail regression tests pass. Exact registrations, source snapshots,
  prediction arrays and inspected curves retained. Main goal incomplete.
- Evidence: production/ACES_FIT_SANITY_2026-10-24.md.

## 24 October 2026 — longer context training and de-banding evaluated; no promotion

- Completed balanced1800 plus further6000-step training. Latest old coherent H
  improves to0.36625540, but new H0.32270959, aligned residual and clipped detail fail.
- Hunyuan48 all appearance/motion checks pass (0.01903457EV); four-frame actual
  float32 EXR/16-bit PQ export independently verifies exact codes, tags and count.
- Fixed delivery verifier to use registered clip sets/counts, including small suites.
- Fixed48-image TRAIN audit shows new-domain fitting weakness, not only DEV failures.
- Original bounded curvature de-banding improves synthetic ramps but harms texture;
  guarded version preserves authored texture/motion yet fails small real grade/code
  nonregressions. Both rejected; no default/model/export integration.
- 38 combined regression tests pass. All launched jobs terminal; artifacts/snapshots
  and failures preserved. Goal incomplete; saved goal remains usageLimited.
- Next: TRAIN fitting/target ambiguity diagnosis before more same-recipe training.
  Evidence in production/ACES_CONTEXT_MOTION_2026-10-24.md and
  production/ACES_CURVATURE_DEQUANT_2026-10-24.md.

## 24 October 2026 — completed real exports; v8 context refinement

- Real constrained equal-head18-clip suite complete:17/18 quality passes; Hunyuan
  maximum0.03454340EV fails. All18 delivery verifications pass,70 exact EXR/PQ samples.
- Sampled81,799,936 pixels have zero >1-code appearance errors. Detail remains open.
- New CC0 planar-motion corpus288TRAIN/72DEV built with7/7 checks passing.
- First independent context600 model fails new/old coherent accuracy and detail;
  Hunyuan motion0.00996643EV passes but21 fully-clipped pixels fail appearance.
- Added explicit all-highlight colour constraint; new72 appearance passes.
- Balanced1800 training underway, replaying prior coherent TRAIN with stronger
  spatial/temporal objectives. No validation training or release promotion.
- 53 broad tests before constraint extension;12 targeted tests after it pass.

## 24 October 2026 — real motion exposes fully-clipped instability; context stage prepared

- Constrained equal head fixes sampled colour drift (selected105code error becomes0.54),
  but real Hunyuan transition fails0.02746EV. Isolated fully-clipped output is causal.
- Gradient-head loss, fully-clipped fallback and feather8 tested; each has quality
  failures. Feather8 passes Hunyuan48 motion but worsens fully-clipped detail6.28%.
- 49 regression tests pass; declared239-row data receipt/content audit passes.
- Frozen32TRAIN/8DEV additional CC0 scenes; known-planar-motion data generation
  running, registered600-step independent-context highlight stage prepared.
- Real18-clip export/delivery suite still finishing. No model/default promoted.
- Evidence: docs/production/ACES_REAL_EXPORT_DETAIL_FOLLOWUP_2026-10-24.md.

## 24 October 2026 — explicit highlight constraint improves coherent development

- Registered and evaluated equal-image head plus existing colour backtracking.
- Corrected coherent24 passes all checks: H MSE0.3693457 vs analytic0.4204337,
  all tested partial highlights within1code, ordinary/shadow support unchanged.
- Legacy24 still fails overall/highlight gates; no default or model promoted.
- 384 procedural moving-object frames pass complete identity/hash/appearance/
  luminance checks; worst transition0.00527555 EV, limit0.02.
- Eight colour stress contacts saved; blue contact visually inspected, centre
  maximum channel drift0.6221codes. This does not demonstrate restored detail.
- Tests pass; all jobs terminal. Next real generated colour/motion and grading/detail.
- Full evidence: docs/production/ACES_CONSTRAINED_HEAD_2026-10-24.md.

## 24 October 2026 — generated export verified; highlight colour remains a blocker

- Completed18 generated clips/826 frames: registered checks pass; all18 RGB16
  videos independently decode correctly and70 saved EXR samples match exactly.
- Additional colour audit reveals up to105-code changes in reconstructed highlights,
  outside the ordinary-only appearance check. No release promotion.
- Corrected a new-corpus storage-encoding error; earlier corpora/trials marked invalid.
  Rebuilt canonical storage with actual-loader read-back checks and a regression test.
- Corrected matched head trials: equal-image loss improves coherent highlight error
  15.71% versus v7 and9.02% versus analytic, but colour preservation and legacy natural
  gates fail. Both rejected; frozen ordinary/shadow behaviour verified unchanged.
- Kept export correctness/performance fixes; final combined regression run41/41 passes.
- All session jobs complete. Next is explicit highlight colour consistency and target
  alignment; full grading/detail/native/platform/release scope remains unfinished.

## 24 October 2026 — generated export review and colour fidelity diagnostic underway

- Hardened review completeness and evidence identity before evaluation; 18 named inputs.
- Four CogVideoX clips pass so far; first RGB16 delivery independently decoded exactly.
- Saturated blue stress test reveals ~75-code green drift hidden by ordinary-only R.
- Registered separate backtracking diagnostic; no change to v7 model or export default.

## 24 October 2026 — v1 analytic-shadow candidate passes registered acceptance

- Owner decision: no learned shadow radiance in v1. coverage4800 weights + analytic
  shadow policy pass all 12 registered checks; dark-motion negative luminance eliminated.
- Remaining before release: highlight detail, ramp ripples, native integration,
  platform/interop coverage, independent holdout, licensing.

## 24 October 2026 — seed study: edge supervision no effect; shadow strategy decision

- Five matched seeds: edge/control natural shadow ratio .9998, 3/5 wins. FAIL.
- 0 of 10 trained shadow branches pass all gates; seed noise (+-4%) exceeds most
  previously reported shadow differences. Proposed: analytic shadows for v1.

## 24 October 2026 — edge shadow trial narrowly fails; seed study registered

- Image-edge supervision gives the first shadow branch below coverage4800 on natural
  shadow MSE, but misses the registered 5% gain (4.81%), lifts 3 pixels and slightly
  raises dark-motion variation. FAIL, kept as evidence.
- Natural shadow evidence is 7 frames from 3 scenes; a registered 5-seed paired
  study will measure whether the gain is real before any further shadow change.

## 24 October 2026 — shadow context trial recorded; edge trial ready to launch

- Wide (radius40) shadow branch fails: worse original-dev shadow accuracy and
  .03054EV dark compressed motion. Local (radius8) is stable at .00800EV but still
  misses natural shadow accuracy versus coverage4800. No promotion.
- Image-edge supervision gap found (4/21 natural shadow images never supervised);
  fix tested. Registered paired trial and one-command PowerShell runner added.
- Codex usage limit reached; this session cannot run GPU jobs. Owner launches.

## 23 October 2026 — requantized consistency fails; shadow context limitation measured

- Controlled8-bit requantization of translated inputs worsens natural shadow
  error and dark motion (.025350EV worst compressed); strict trialFAIL. One
  tiny negativeY cancellation retained in results, no tolerance relaxation.
- Built separate original96TRAIN24dev known-linear shadow surfaces;4.9million
  TRAINshadow pixels, maxroundtrip.504120code, noexternalmedia. First partial
  build failed coverage and is marked unusable; correctedv2 passes two tests.
- Found2.05million allblack17x17 TRAINneighbourhoods with differing targets;
  currentradius8 shadow branch cannot distinguish them. Contact inspected.
  Next wider-context shadow learning with natural fidelity retained.
- Six consistency/shadow tests pass; all jobs complete, no promotion. Full HDR
  quality/platform/native/commercial-release scope remains unfinished.

## 23 October 2026 — shadow generalization audit and temporal regularization pair

- Fixed168 TRAIN crops show shadow branch improves training brightness/chroma;
  development regression motivates generalization work rather than blind scaling.
- Implemented/tested a CVPR2019-inspired transformed-error consistency loss.
  Matched600-step fine-tuning controls preserve frozen restoration/HDR weights.
- Both pass original natural gates but miss coverage4800 shadow accuracy.
  Treatment dark-motion error worsens versus matched control; compressed worst
  .021460EV exceeds.02. Full384 negativeY/lift0 each, but paired trial FAIL.
- Verified a quantization distribution mismatch on one TRAINcrop: bilinear
  transformation introduces fractional codes. This is a next hypothesis, not
  causal proof. Graded bands remain. Five focused tests pass; all jobs complete.
- Next quantization-aware TRAIN motion simulation with unchanged acceptance;
  broader HDR detail/platform/native/independent release scope remains open.

## 23 October 2026 — nonnegative shadow luminance experiment fails quality checks

- Implemented opt-in v5 square-root luminance/signed-chroma branch, frozen
  coverage4800 predictor/restorer. Nine tests pass; fresh1200 shadow-only run.
- Full384 synthetic motion frames now have zero negativeY, zero lift, R100%.
  However dark compressed worsttransitionmean excessEV.020933 exceeds.02;
  natural-shadow MSE regresses3.858% versus control. Registered trial FAIL.
- Hash-verified full24 predictions are bitexact outside shadow support;
  ordinary restoration/highlights preserved. Graded contact still shows bands.
  No promotion, no RGB clipping or relaxed thresholds. Exactblack square-field
  zero derivative remains a learnability concern. All jobs complete.
- Next improve shadow supervision/generalization/temporal behavior; full HDR
  detail, grading, platform/native and independent release scope remains open.

## 23 October 2026 — coverage176/fixed4800 passes restoration development checks

- Preserved natural48, earlier coloured procedural32 and balanced96 with
  explicit2/1/1 crop quotas. Fresh4800, frozen HDR predictor unchanged;
  all176 images visited, six focused tests pass.
- Natural codeMSE13.059% and legacy procedural88.640% better than analytic;
  both beat gradient1200. All nine ramp gradient errors56–81% below mixed1200.
  Registered comparison PASS; original five natural gates pass, zero lift.
- Visually inspected ramps still have steps/ripples; raw correction probe
  remains below.45code, so loosening.49 bound is not justified by saturation.
  Development pass is not G/H/S/T/platform/native/release completion.
- Reviewed primary dequantization research; no external data/code/weights added.
  Next separate shadow-luminance and HDR-detail work while preserving restoration.
- Temporal18: R100%,zero lift/newnegativeY/excludedcells. Worsttransitionmean
  excessEV .001846724/.000410060 increases versus v3 but remains below.02
  on this limited fixed-cell diagnostic; not releaseT. All jobs complete.

## 23 October 2026 — balanced training improves ramps but fails fidelity comparison

- Added tested explicit equal-domain crop sampling; fresh1200 on natural48 and
  balanced96, all144 visited and frozen HDR predictor unchanged.
- All nine ramp gradient errors improve67–82% over mixed1200. Inspected curves
  still have steps/ripples; no grading acceptance claim.
- Natural codeMSE .14393650 and original procedural .04064778 regress against
  gradient1200. Registered comparison FAIL; no model promotion. Natural original
  five gates still pass, zero shadow lift; supplemental balanced24 improves73.486%
  over analytic. Data/sampling intervention is confounded, not causal proof.
- Five focused tests pass, all jobs complete. Next preserve coloured procedural
  coverage while adding balanced cases; shadow/HDR detail and release scope open.

## 23 October 2026 — target gradients help; missing colour/intensity coverage repaired

- Opt-in gradient-supervised restorer improves natural codeMSE12.211% and
  procedural80.511% versus analytic. All ramp gradient errors improve, but
  bright-ramp gains17.8–22.7% miss the registered25% requirement. TrialFAIL;
  no threshold relaxation or promotion. Inspected plots still show steps.
- Found zero neutral/all-dark/all-bright pixels in the original procedural
  TRAIN supplement. Built separate balanced96TRAIN/24dev original examples;
  verified coverage and max target roundtrip.497786code. Not trained on yet.
- Natural TRAIN shadow targets contain no negative luminance (35607positive,
  51zero), clarifying the motion defect is not required by those targets.
- Four relevant tests pass, all jobs complete. Next controlled balanced
  restoration sampling and separately constrained shadow learning; full goal
  including HDR detail, video/platform/native and independent validation remains.

## 23 October 2026 — mixed restoration improves both domains; motion coverage grows

- Mixed300 restores natural fidelity but loses procedural gains; rejected.
  Registered fresh1200 with same settings except length improves ordinary
  codeMSE9.364% on natural24 and66.364% on procedural8. Combined restoration
  development checks pass; frozen HDR weights unchanged. No release promotion.
- Inspected ramp plots still show steps/ripples. Reduced plateau counts are
  not equivalent to banding-free grading; G remains incomplete.
- Built original moving-object/shadow/specular96frame scenes and H264 variants,
  with float32 reference frames; all576 unique frame/reference hashes checked.
  Baseline384frame audit preserves appearance and no-lift, but exposes negative
  shadow luminance absent from analytic/reference in checked frames. Recorded
  exclusions and defect; no signed-RGB clamp or temporal-release claim.
- Seven relevant tests pass; all jobs finished. Next address restoration step
  artifacts and shadow luminance while keeping natural fidelity; native,
  platform, real-motion and independent release validation still outstanding.

## 23 October 2026 — separate restoration branch learns but fails natural transfer

- Added opt-in v4 local-code restoration branch, keeping v2/v3 defaults and
  frozen HDR predictor. New trainer uses ordinary code loss without OCIO
  backward and checks frozen weights bitexact. Ancestry includes both datasets.
- Procedural8 error improves70.918%; ramp plateaus reduce substantially and
  gradient errors improve. Plots still show dark steps and bright ripples.
- Natural24 passes existing numerical checks, but ordinary code error worsens
  75.049% versus analytic. Candidate rejected for natural fidelity; no promotion
  or claim of restored HDR detail. Stored adverse evidence beside probePASS.
- Nine relevant tests pass, all jobs complete. Next train a mixed natural/
  procedural restoration branch and require fidelity in both domains.

## 23 October 2026 — restoration loss tested; original procedural coverage added

- Added opt-in direct ordinary-pixel display-code supervision, defaultOFF;
  four loss/sampler tests pass. Natural-scene300 trial reduces ordinary code
  MSE3.319%, but overall HDR improvement only1.66% fails5%gate and ramp banding
  is unchanged. Appearance/shadow safety pass; candidate not adopted.
- Built32 original procedural TRAIN and8 separate development examples, no
  external media. Hash/scene separation checked; target roundtrip max.492925code.
- Isolated procedural300 trial fails to learn meaningful restoration: codeMSE
  .0476% worse than analytic and unchanged ramp plateaus. A bounded65px box
  reference also worsens TRAIN error65.01%, so no smoothing shortcut adopted.
- Reports/configs saved, all jobs complete. Next test an isolated restoration
  branch with explicit local code differences; broader HDR/detail/video/native
  and independent release requirements remain intact.

## 23 October 2026 — original-HDR detail and smooth-ramp checks reveal quality gap

- Audited all24 saved development predictions against original HDR gradients
  on identical valid support. Model is slightly worse than analytic in highlight
  interiors and materially worse in the two eligible shadow cases. Experimental
  targets retain more reference structure. This is not solved by average MSE.
- Three known neutral ramps remain banded: at0EV plateau fractions equal the
  analytic baseline (76.648%,97.542%,94.302%). No grading-headroom acceptance.
- Candidate retains its numerical development pass but is not promoted. Saved
  detail_truth.json and grade_ramps_v2/report.json; two focused tests pass,
  all jobs complete. Next isolate direct restoration supervision before a
  separate reconstruction-detail objective trial. Full goal remains active.

## 23 October 2026 — moving-frame smoke and learned16-bit HDR output

- Lower_lr300 ran full-frame on18 existing CC0 camera-motion frames. Ordinary
  R100%, zero shadow lift, no negative luminance. Worst transition mean cell
  excess exposure.001559EV/.000003895EV across two clips; no cells excluded.
  Fixed-cell diagnostic only, not optical-flow aligned or release temporal proof.
- Added signed AP1 -> official ACES2 HDR1000 Rec2020 PQ -> FFV1 RGB16 bridge.
  Existing encoder now accepts explicitly declared pre-mastered PQ codes while
  keeping its previous default. No second shoulder or PQ round trip.
- Both9frame1280x720/24fps clips decoded exactly to expected16bit ACES codes;
  full-range/Rec2020/PQ/RGB tags verified. Artifacts under lower_lr300/video16.
  Display derivatives; float32EXR remains master. Native/player checks pending.
- 45 tests passed,2 existing delivery skips; all jobs complete. Longer motion,
  generated/compressed sources, grading-headroom and release evidence remain.

## 23 October 2026 — lower learning rate yields numerical development pass

- Fixed144 TRAIN crop audit found deterministic control300 worse than analytic
  and control200, motivating smaller updates rather than a larger model.
- Registered fresh300 Adam.0001 trial; sampler/RNG/contract match previous
  control, only learning rate and trainer argument code hash differ.
- CPU full24: overall target error7.8001% better than analytic, highlights and
  shadows improve, ordinary R passes, zero shadow pixels lift>1code. All five
  unchanged numerical development checks pass. No release promotion.
- Candidate G:/rudra_training/run05_aces_lower_lr300_20261023; acceptance,
  TRAIN curve and comparison_identity reports saved. Contact00/06/08/09 reviewed
  only; window reconstruction still limited. Headroom/temporal/platform and
  independent release evidence remain incomplete. Five focused tests pass.
- All training/evaluation jobs completed. Next validate detail/headroom and
  temporal behavior before broader training and complete native/video delivery.

## 23 October 2026 — deterministic comparison rejects both300-step candidates

- Completed registered weight0/1 comparison and CPU full24 evaluation;
  matched recipe/code/data/RNG mechanics pass. Both fail development gates.
- Overall target error: analytic .009646895, control .010594832, treatment
  .019774891. Treatment shadow error126.10% worse than matched control.
- Ordinary round trip passes both; shadow pixels lifted>1code:5 control,
  13359 treatment. No promotion, threshold relaxation or shadow-term adoption.
- Comparison: G:/rudra_training/run05_aces_deterministic_scaled300_20261023/comparison.json.
  Only contact00 of both visually reviewed. All associated jobs complete.
- Reconciled supplied Claude session with current evidence. Float32 transport
  exists, but useful grading headroom and full16-bit video integration remain
  unproven. Current development set is not an independent release holdout.

## 23 October 2026 — shadow-gradient imbalance measured; repeatability fixed

- TRAIN gradient probe: highlight median13.9157 versus shadow5.98e-6 under
  absolute objective; view-scaled shadow term produces median.31054.
- Matched300 weight0/1 trial mechanics pass; CPU full24 comparisonFAIL. Shadow
  error8.56%worse, overall/highlights regress, one shadow pixel lifts>1code.
  Term stays defaultOFF; no promotion or gate relaxation.
- Repeated control changes materially from previous run, exposing numerical
  instability in single-run comparisons. Added deterministic GPU settings;
  two independent60-step repeats match model/optimizer/RNG bit-for-bit.
- Deterministic mode now defaultsON. Five relevant tests pass; all jobs done.
  Next run reproducible comparisons before any further objective adoption.

## 23 October 2026 — varied spatial training and scene-disjoint development result

- Built48 CC0-metadata TRAIN images/21scenes and24 development images/8separate
  scenes with matched builder hashes. Exact-black reference edge case fixed;
  no artificial radiance floor. All ordinary target pixels within1code.
- Fresh v3 spatial300 completed:1196unique128crops, every training image visited,
  channels8,batch4,Adam.001. Registered recipe and input/code hashes saved.
- Full development24 target error improves23.73%, highlights improve, Rpasses
  and zero shadow pixels lift>1code. Shadow MSE regresses.0436%, so registered
  numerical gateFAIL; checkpoint remains experimental. No threshold relaxation.
- Partial visual review00/06/08 recorded; six relevant guard/sampler tests pass.
  Checkpoint and reports G:/rudra_training/run05_aces_spatial300_20261023.
  Next inspect TRAIN loss-scale balance; development is not independent holdout.

## 23 October 2026 — full-frame ACES audit and normalization correction

- GroupNorm four-crop smoke creates magenta highlight streaks on fullframes,
  worse than analytic10/12 despite ordinary R100%. Float32 export verified
  bitexact with independentOpenEXR; transport success does not mean quality.
- Added opt-in v3 per-pixel RMS normalization and locality regression test.
 17 focused tests pass. Same60step/four-crop recipe then full12 evaluation:
  mean target error70.40%lower than group,11/12images improve. Only6/12better
  than analytic, so no release claim. R100%,worst.108609code,0shadowlift>1code.
- Same trained stonehighlight inner64 contextRMSE .120579->.000132; reviewed
  contacts00/06/08 show large magenta streaks removed, residual edge errors.
- Checkpoint/reports in G:/rudra_training/run05_aces_pixel_rms_smoke_20261023.
  All jobs complete. Next broaden spatial sampling with v3; smoke not promoted.

## 23 October 2026 — shadow reference probes and ACEScg learning smoke

- Direct HDR shadow targets retain interior gradients almost exactly and obey
  lift bound, but boundary errors worsen in some cases. Smooth constrained
  log-exposure alternative improves two shadow detail cases, regresses another.
  Both full12 pilots preserve ordinary R100% with0shadow pixels>1code lift.
- Reviewed direct crops05/06/08 and smooth06/08 at0/+3EV; black seams remain.
 26 focused tests pass; targets remain experimental, not release qualified.
- Ran isolated four-crop ACEScg learnability probe, channels8/Adam.001/60steps.
  Loss.0298612->.00160594; ordinary37055pixelsR100%,worst.219385code.
  Completed20.81s on CUDA with CPU OCIO bridge. Not validation/generalization.
- Checkpoint/report: G:/rudra_training/run05_aces_learnability_smoke_20261023.
  No production model promotion; all jobs complete. Full-image learned audit next.

## 23 October 2026 — coherent shadow bound preserves detail, exposes edge trade-off

- Added optional per-component exposure bound with actual OCIO feasibility
  verification.24 focused tests pass, including unchanged relative detail and
  exact outside-mask identity. No default or legacy checkpoint changes.
- Full12 TRAIN pilot: ordinary R100%, zero shadow pixels >1code lift,
  max.4900001code, interior log-gradient scores unchanged from unbounded target.
- Some components darken nearly5stops. Grass boundary metric improves, stone
  cases6/8 at+3EV worsen. Reviewed contact, candidate stays experimental.
- Evidence in G:/datasets/corpora/run05_aces_region_shadow_pilot_20261023.
  No training launched; next address true-reference shadow detail and joins.

## 23 October 2026 — reference-detail audit and bounded-shadow trial

- Added provenance-checked interior log-luminance gradient diagnostic. Fully
  clipped stone targets retain97–100% reference gradient gain with boundary
  join; analytic inverse has zero. Three cases from same scene, not broad proof.
- Shadow target lift remains:6223/12478 pixels over1code. Tested opt-in viewed
  channel bound source+.49code in a separate12-image pilot; lift count becomes0,
  max.490038code, ordinary R100%, highlight metrics unchanged.
- Projection worsens shadow gradient errors; NOT adopted. Need coherent or
  joint gradient/brightness constraint instead of independently clipping pixels.
-23 focused tests pass. Shadow comparison contact inspected. All jobs complete,
  no training launched. Reports in both ACES boundary/bounded-shadow pilot folders.

## 23 October 2026 — boundary-aware AP1 target experiment completed

- Added opt-in original gradient-domain target join, with exact fixed ordinary
  pixels and signed AP1 support.10 focused tests pass; default remains scalar.
- Full12 TRAIN candidate preserves ordinary R100%, worst.493214code. Clipped
  evening-field sky seams visibly reduced; mean boundary discrepancy to analytic
  baseline31.613->1.776codes at0EV,47.970->4.607 at-3EV.
- Four fixed comparison sheets plus shadow regression contact inspected. AgX
  grass shadow boundary metric worsens; remaining banding/foreground artifacts
  also appear in analytic baseline. Reference detail retention is unproven.
- Candidate not adopted globally; no training. Evidence under
  G:/datasets/corpora/run05_aces_boundary_pilot_20261023/comparison.

## 23 October 2026 — signed AP1 model range fixed; target visual gate failed

- Experimental ACEScg v2 uses signed linear RGB residuals with shared scale.
  Old exponential range failed179 target pixels at4stops, still131 at12stops.
- Actual composition oracle on12 TRAIN targets: max viewed error .001808code
  across -3/0/+3EV, zero pixels over1code; not learned quality. Full12 random
  stress ordinary R100%, max.538467code, all outputs finite.18 tests pass.
- Inspected all3 pilot contact sheets. Clipped sky develops coloured seams;
  another sky retains broad banding. Current AP1 targets rejected for training,
  despite ordinary-pixel Rpass. Next correct boundary colour/gradient fitting.
- No new training or checkpoint migration. Review and evidence paths recorded
  in production/ACES_TARGET_PILOT_REVIEW_2026-10-23.md. Full goal remains open.

## 23 October 2026 — exact ACES real-image and float32 export proof

- Official CPU inverse/forward passes allpixels within1code on24 real SDRs;
  worst.689code. Float32 AP1 EXR reads bitexact; independentOpenEXR verification.
  Analytic inverse only: no learned detail, dequantization or physical RAW claim.
- Experimental exact-forward/finite-difference-backward bridge implemented;
  eleven view/gradient tests pass. Batch4x256 CPU-backed forward/backward about
  .84s total even forCUDAinput; not native/Script deployable or trainer-integrated.
- Gamut audit finds4.747% synthetic inverse colours have negative Rec2020
  values. Existing nonnegative clamp reducesR to95.468%; an explicit AP1 or
  signed-output contract is required before model migration. No fresh training.


## 23 October 2026 — signed recovery rejected; ACES and temporal feasibility findings

- Full24 recovery comparison: signed gradient0.7517% worse than matched
  control, clean and earlier-good-control guards fail. Four visual reviews
  inspected: both arms remove blue blotches; no signed-loss adoption.
- Short2x9frame camera smoke complete, same512/64 FP32 tiled inference.
  Global excess small; local camera/content changes confound fixed-cell values.
  No temporal release claim. v2 report fixes exposure-floor unit convention.
- Official ACES2 inverse LUT65/129/256 tested. Even256cube continuous samples
  only99.6017%within1code, below99.9%; exact inverse passes same randomsets.
  LUT axis/gradient regression passes; no approximation integrated.
- Next resolve exact ACES-view baseline/target/export compatibility; model
  banding, independent video/platform and native integration remain unfinished.


## 23 October 2026 — matched recovery mechanics verified; temporal smoke data ready

- Both recovery runs completed200; all matching checkpoint mechanics pass.
  CPU full24 control and signed evaluations running; no quality verdict.
- Recovery comparator now enforces earlier-good-control and zero-shadow-lift
  requirements; seven comparison/audit tests pass. Visual review remains required.
- Prepared two fixed-look/exposure CC0 camera-move9-frame snippets;18 saved
  frames hash/decode pass, first/last contact reviewed. Development smoke only;
  longer motion/platform testing and temporal inference remain incomplete.


## 23 October 2026 — recovery control finished; treatment and full24 audit started

- Control200 completed; signed .1 treatment launched34788. CPU control audit
  launched50712 (reused PID) under shadow_repair_comparison_20261023/control200.
- Added source-defined, target-referenced full-image shadow-colour diagnostic;
  two tests pass. Previous failed trial deep-shadow signedRMSE .614756 versus
  earlier control .006852; report counts and target gamut retained explicitly.
- Treatment, matched mechanics, full24 comparison and visual review pending.
  Acceptance still includes earlier good control and original release gates.


## 23 October 2026 — signed shadow objective and registered recovery experiment

- Confirmed spatial-only loss cannot identify constant wrong colour; added
  signed, target-anchored shadow radiance term, default OFF.15 focused tests
  pass, including exact trainer resume and sampler/RNG equivalence.
- TRAIN12 output/parameter probes finite. Original parent raw candidate/primary
  norm median.00938/max.06899; rejected parent median.01128/max11.159. Registered
  .1 weight. Full-frame context was not a broad worsening factor in TRAIN12.
- Recovery200 control launched37884 from rejected gradient400, spatial loss
  removed. Signed treatment waits for control. Both must also be compared with
  original nonspatial control; repairing a bad model alone is not adoption.
- No release claim; model/math dependencies now pinned for matched experiment.


## 23 October 2026 — reconstruction-gradient trial rejected after full evaluation

- Matched full24 result: gradient error23.1375% worse, clean group regresses;
  no adoption. R/relative H/S guards pass, original banding still fails.
- Three fixed visual comparisons inspected; treatment produces dark blue
  blotches in acoustical-shell image. Source-dark region23875pixels:11244
  gain>10codes in a channel, 50.785% have negative Rec709 linear channels.
  Control has neither issue. This is a shadow-colour failure despite small
  mean visible-lift statistic. Original zero-lift requirement is not met.
- Next investigate unclipped loss gradients and shadow chroma on TRAIN data,
  retaining all existing trial/release gates. No fresh training launched.
- Primary research and synthetic oracle show source-preservation budget limits
  achievable compression cleanup. One feasibility regression test passes;
  diagnostic only, no target or policy relaxation.


## 23 October 2026 — reconstruction treatment completed; complete graph export probe

- Both matched400 checkpoints finished; mechanics400.json passes identity,
  config, sampling/RNG and finite-weight checks. CUDA is not deterministic.
- Full24 CPU audits live: control worker29100 at20/24, treatment worker51456
  at6/24. Comparison and visual review pending; no quality/promotion claim.
- Added research-only complete-output TorchScript exporter, retaining picker
  baseline and bounded restoration. Reloaded graph exactly matches eager on
  five synthetic shapes and six hash-verified TRAIN crops across codecs.
- Six focused tests pass, including active restoration at unseen image size.
  Batch1/default controls only; native v1 and ACES integration remain pending.


## 23 October 2026 — treatment running; actual EXR transport and native export gap

- Reconstruction control completed400. Treatment .01 launched52116; full24
  control CPU audit launched45392. Final comparison remains pending.
- Current native v1 package lacks bounded restoration and picker-conditioned
  encoder parity. Exporter refuses unsupported models explicitly;3 tests pass.
- Exported an actual audited model prediction to float32 AP1 EXR with change
  mask. Decoded-file R passes; independent OpenEXR confirms channel types.
  Metadata explicitly retains current BT1886 research view, not ACES output.
- Native end-to-end model integration and final quality remain unfinished.

## 22 October 2026 — reconstruction-gradient comparison prepared and started

- Audited known-channel changes within partially clipped pixels; current
  pixel-level targets permit many. No retrospective target/policy change.
- Added opt-in reference spatial-gradient loss around reconstructed regions
  and two-pixel surroundings at -3/0/+3EV, without smoothing toward input.
- TRAIN12 gradient signal probe supports .01 trial weight;24 focused tests
  pass including exact CPU resume and unchanged input sampling.
- Matched400 control launched56092; treatment pending exit. Dequant term off.
  Registered full24 comparison and unchanged release requirements remain.

## 22 October 2026 — dequant branch and error attribution diagnosis

- TRAIN12 branch probes rule out saturation/vanishing gradients on sampled
  crops; auxiliary gradient aligns strongly with primary. Its spatial term is
  much weaker than its pixel term; clean crops acquire positive code bias.
- Full24 zeroEV diagnostic localizes almost all excess gradient error to
  reconstructed/boundary edges, not ordinary-region dequantization.
- Research-inspired bounded diffusion passes synthetic ramp test but fails
  real TRAIN pilot11/12 crops. Rejected for integration; no validation tuning.
- No additional training or promotion. Details/evidence in continuation audit.

## 22 October 2026 — auxiliary dequant trial fails; mask transport verified

- Full24 paired comparison completed: gradient error worsens0.03095%, clean
  group also regresses. Registered improvement/no-clean-regression gates fail.
  R/highlight/shadow guards pass; original banding still fails. No adoption.
- Four fixed review selections rendered and inspected; no obvious overview
  improvement. Diagnose branch signal/target attainability before more training.
- Python/native named EXR mask transport implemented;53 Python tests pass,
  2existing skips;14 native export tests pass. Native/Python byte parity and
  independent OpenEXR read pass; desktop Release RUDRA build passes.
- Model-to-mask and desktop request wiring unfinished; no release claim.

## 22 October 2026 — paired400 mechanics verified, image verdict pending

- Both runs completed400. Saved final states match code/data/parent, sampler,
  RNG and all config except treatment/output/ID; both weights finite.
- Parent full24 audit complete: reference gradients worse than analytic
  baseline, including uncompressed group. R passes; no broad quality pass.
- Control full24 GPU audit live session18247; auxiliary evaluation next.
- Official OCIO2.5.2 ACES2 inverse/forward tested on cube, boundary and grey
  samples: >=99.9% within1code for Rec1886 andsRGB; saturated exceptions exist.
  Separate feasibility evidence, no trained-model/native ACES integration.

## 22 October 2026 — control completed; auxiliary and full-image audit running

- Control400 completed and process exited; final checkpoint preserved.
- Auxiliary .003 run started sequentially (launcher34964). Config differences
  limited to loss weight/output/run ID. Keep training dependencies fixed.
- Full24 evaluation tool saves predictions and measures reference errors at
  seven exposures alongside contract metrics. Parent CPU audit live49988.
- Comparison checks input/reference coverage and missing measurements;
  aggregation/adoption criteria fixed before auxiliary results. Two new audit
  tests and two existing banding tests pass. No quality verdict yet.

## 22 October 2026 — corrected data and dequantization comparison

- Corrected624 corpus finished:1872 media verified, R624/624,600train/24val;
  original input hashes retained. No VAE rows or object-motion evidence.
- Added opt-in supervised code/gradient loss with matching-view safeguards;
  exact CPU resume and crop-RNG equivalence covered.23 focused tests pass,
  plus builder dependency-hash resume refusal test. Fixed zero sRGB gradient.
- Registered matched400-step control/auxiliary recipes, warm-starting strict
  policy diagnostic. Control launched (42500); auxiliary pending control exit.
  Full24 reference errors and existing quality guards determine trial result.
- Existing model still fails banding; no release or production-model claim.

## 22 October 2026 — versioned observed-clipping policy

- Original expansion2000 finished; final fullres still fails banding. No promotion.
- observed_clip_v1 preserves codes254 and below; integrated through builder,
  model config, both trainers and evaluator.42 focused tests pass.
- Corrected24 targets: same inputs,72 media verified,R24/24; bright >1code
  changes52,226->21. Sky target/model comparison visually confirms patch fix.
- Policy-only step1000 keeps exact weights; fullres Rallpass, highlights better
  than baseline on new6-pair mask, shadow0/3, banding FAIL. Not retrained.
- Full624 target rebuild running(session22197,worker11408),495rows observed;
  complete audits before a new training experiment. Details in continuation audit.

## 22 October 2026 — fullres1000 improvement, but target artifacts confirmed

- Fullres24: R passes; highlights1.920->.714stops, shadow visible lift0 on3
  measurable pairs, fidelity.03120->.02133. Banding still fails.
- Four visual reviews expose source-to-target sky clipping;24-target audit
  confirms45,066 newly clipped pixels in bright, initially unclipped regions.
  Existing R excludes these via the broad reconstruction mask. Fix policy
  before another experiment; retain existing metrics and failed data evidence.
- New reference banding diagnostics detect noise gaming;2 regression tests,
 7 combined dequant/metric tests pass. No production threshold inferred.
- Training live1820/2000; no promotion. See continuation audit.

## 22 October 2026 — explicit dequantization diagnostic

- Standalone code/reference-gradient loss prototype;5 tests pass, not enabled.
- Real12 TRAIN crops: step800 lowers mean code error but regresses both clean
  samples; corrected candidate gradients finite12/12. New loss needs scaling.
- CPU step1000 fullres24 evaluation running(session6187); GPU training live
  at1060/2000. No quality pass. See continuation audit for paths and limits.

## 22 October 2026 — desktop 16-bit video path completed and tested

- FFV1 RGB16/PQ MKV integrated into native movie encode/QC, CLI/queue, desktop
  movie/folder and headroom export. Float32 EXR labels corrected.
- Release builds pass;8 native encode/QC and5 distinct desktop tests pass.
  Actual16-bit codes preserved; queue/folder/headroom exports exercised.
- Headroom integration test fixture corrected from unsupported YUV420 to
  real YUV44410; existing HDR10 and new FFV1 both pass.
- Training live step680/2000, no quality/release pass. See continuation audit.

## 22 October 2026 — expanded training launched

- All624 saved targets pass R; full data/visual audits completed.
- Started run05_expansion_20261022,2000 scratch steps,2-hour cap, resume100,
  bounded/native/shadow objective. No quality pass; keep training code fixed.

## 22 October 2026 — full expansion audit and VAE preparation

-624 pairs complete;1872 media hashes/decode pass; all26 sheets reviewed.
  Full target contract still running; expansion2000 remains unlaunched.
- Official Wan Apache2.0 VAE pinned/downloaded/hashed; small GPU roundtrip passes.
  Builder weight identity/resume checks added;22 tests pass. No VAE data admitted.

## 22 October 2026 — shadow full-resolution result and partial QA

- Shadow loss lowers visible lift .631->.303; highlights1.263stops vs1.920
  baseline, R24/24. Still fails shadow/banding; no promotion.
- Expansion473/624 observed; sampled24 pairs passed hash/decode/visual review.
  Full audit pending;2000-step expansion recipe prepared but not launched.

## 22 October 2026 — CC0 expansion conversion

- Hash-verified200train+8val Poly Haven source scenes; conservative heldout-name
  family exclusion regression fixed/tested.624-pair conversion running in v2.
- Shadowloss200 completed with crop regression; fullres24 running. No promotion.
- Source selection and conversion are development work, not production clearance;
  visual quality, data overlap and temporal coverage remain unresolved.

## 22 October 2026 22:04 — target-relative shadow experiment

- Nativecrop fullres: R24/24 and highlight improvement, but shadow lift .631
  and fidelity regression. No promotion.
- Train dark-target audit supports stronger target-relative shadow loss;
  existing loss enabled in matched200-step run05_shadowloss_smoke_20261022.
  10 focused tests pass. Limited dark data and motion coverage remain unresolved.

## 22 October 2026 — native RGB16 delivery and crop experiment

- Native FFV1 RGB16 sequence encoding implemented and verified code-exact with
  correct PQ2020/full RGB tags.7 sequence tests pass; desktop integration pending.
- Native-crop200 completed; clean/hard PU -.890269/-.532466dB, no promotion.
  Full-resolution validation now running; see continuation audit for paths.

## 22 October 2026 21:58 — full-resolution bounded result and native crops

- Full-res24: R all pass; highlight error1.92->1.27stops and fidelity improved;
  shadow lift .284 and banding excess .287 fail. No promotion.
- Dataset resize broke source/target nonlinear relation; native crops restore
  clean target R from3/24 to24/24.19 tests pass. Legacy hard targets R0/24.
- Matched200-step run05_nativecrop_smoke_20261022 launched. Read continuation
  audit for exact evidence and research; main goal incomplete.

## 22 October 2026 — bounded smoke result

- Completed200 steps: clean PU +0.076043dB, hard -0.490607dB; log-L1
  0.00678516 vs initial0.00877800. Hard score fails; no promotion.
- Full-resolution24-frame evaluation started in run05_bounded_eval_20261022.
- Tiled preservation test passes; source collectors reject unknown commercial
  flags. Frozen active/previous corpora unchanged; provenance audit still needed.

## 22 October 2026 21:51 Cairo — bounded output diagnostic

- Corrected-data smoke completed200; clean/hard PU -7.62154/-0.34067dB.
- Experimental source-bounded correction head integrated;21 focused tests pass,
  including real model gradients and strict checkpoint reload.
- Matched200-step run05_bounded_smoke_20261022 started; worker27140 observed live.
  No production pass. Full details in production/RUN05_AUDIT_2026-10-22.md.

## 22 October 2026 evening — run05 smoke and target feasibility audit

- Full continuation evidence: docs/production/RUN05_AUDIT_2026-10-22.md.
- Run05 96-pair development corpus decoded/hashed; scratch200 completed but
  failed quality (clean/hard crop PU -4.714/-0.296dB). Full-resolution24-frame
  report confirms failure: raw R passes4/24, visible-shadow-lift mean0.519 on
  3 measurable pairs. Constraints restore R but retain shadow defects.
- Target conflict measured: pre-compression restoration targets cannot satisfy
  strict compressed-source matching. Explicit target modes introduced.
- Near-black storage floor independently found to break saved preservation
  targets; lower floor and file-roundtrip regression added, 19 tests pass.
  New real validation rebuild must be checked; no further training launched.
- Native float32 master export integrated and built; 12 tests pass. FFV1 RGB16
  transport tested byte-exact. Full model, temporal, commercial provenance and
  delivery validation remain incomplete; no release authorization.

## 22 October 2026 (later) — SI-HDR passes; luminance dark guard NOT confirmed

- SI-HDR v1.1 (181 photographic HDR captures, SIMULATED SDR, 15 location groups), frozen run02 best.pt
  without guard (the "unguarded" prediction of the guarded run; same Runner/protocol code):
  clip_95 full +0.75 [+0.50, +1.00], saturated +0.71 [+0.33, +1.09], unsaturated +0.62 [+0.29, +0.98];
  clip_97 full +1.12 [+0.66, +1.73], saturated +0.83 [+0.28, +1.37], unsaturated +1.09 [+0.54, +1.83] dB.
  A1, A2, A3 PASS at both clip levels; no A4 category flags. Weakest category: sun disk in frame
  (-0.00 / -0.21 dB saturated, 4 groups). A5 visual review of the 10 worst images still required.
  The standalone SI-HDR run (sihdr_run02_best_step16200.partial) never completed; not needed.
- Dark guard (validation-selected lo 0.2 / hi 1.0 nits, feather 8): validation clean +0.878 -> +0.922,
  hard +0.208 -> +0.211, dark slice 67 frames / 8 scenes clean +0.378 -> +0.488. Mataha re-score
  (DIAGNOSTIC ONLY): clean -0.345 -> +0.170, hard -0.107 -> +0.015 dB.
- Pre-registered confirmation: C1 PASS (A1/A3 hold); C2 PASS (night groups +0.72 / +0.64 dB, but
  unguarded was +0.93 / +1.39); C3 FAIL: guarded saturated-region gain is 0.27 / 0.49 dB below
  unguarded (limit 0.1); full frame -0.34 / -0.66 dB vs unguarded. C4 not scored: the guarded ToS run
  left an empty tos.partial. => Guard NOT confirmed. It must not ship as the default.
- Reading: the defect is specific to codec-crushed, noisy near-black (Mataha); clean photographic
  shadows (SI-HDR) are reconstructed well and a luminance-only guard throws that away. The
  production fix has to distinguish crushed/noisy blacks from real shadow detail -> training fix
  (configs/experiments/run03_dark_fix_DRAFT.json), with SI-HDR + ToS as the pre-registered checks.
  A guard re-designed after seeing SI-HDR would make SI-HDR non-independent for it.

## 22 October 2026 — frozen candidate FAILS the reserved test; dark-scene defect found

- Frozen candidate run02 best.pt (step 16,200, sha ad3d7910..., owner override recorded).
- Reserved test (Mataha, 260 frames, ALEXA 35 LogC4), equal-scene CI vs bt1886 baseline:
  clean -0.345 dB [-0.989, +0.181] (gate lower >= -0.1) FAIL; hard -0.107 [-0.238, +0.019] (gate > 0)
  FAIL. CVVDP (reported, not gated): clean 9.424 vs 9.542, hard 8.372 vs 8.510 JOD (model worse).
- Diagnosis: one dark interior shot dominates. A_0027C021 clean -6.3 dB, and again as z_mataha
  a_0027c021 -3.3 dB. Without those two the clean mean is +0.025 dB. Error is below 0.1 nits: the model
  lifts codec-noisy crushed blacks into saturated yellow/green blobs (up to +10 stops over baseline)
  with 512-px tile discontinuities. Visible defect; images in G: test_findings_run02_best_step16200/.
  Consistent with ToS 03_2a (dim interior) clean -1.38 dB. mataha/z_mataha share 6 shots (~20
  independent shots, not 26).
- Temporal (ToS v1, synthetic SDR): T1 overall flicker pass 6/6, T3 no popping 6/6, T4 reset 6/6;
  T2 moving-highlight excess FAIL 3/4 (04_5f hard, 06_1a clean+hard). Frozen verdict stands; diagnosis
  (relative-to-clipped-baseline definition; per-frame hard degradation) recorded separately in
  independent_protocol_20261021/tos_v1_findings_run02_best_step16200.json, not as a pass.
- Overlap screen SI-HDR vs 2,720 train targets: 0 flags at 0.97 (max 0.90). SI-HDR scoring running.
- Status: NOT production-ready. The reserved test kit is spent for this candidate and must not be
  used to select or tune a replacement. Next: fix the dark-scene defect, selected on validation plus
  a NEW dark-scene validation slice, then a fresh held-out test (owner decision).
- Step 1 prepared (no training): rudra/dark_guard.py (baseline-luminance-gated blend to the analytic
  baseline below lo..hi nits, optional feather) and training/dark_guard_eval.py, which re-scores the
  existing validation exports (no new inference), must reproduce test.json exactly, refuses non-validation
  runs, and applies a rule frozen in the file (val gate + dark slice >= -0.1 dB, >= 3 dark scenes).
  3 tests pass. Step 2 (training) drafted only: configs/experiments/run03_dark_fix_DRAFT.json.
- Owner: no further unused footage exists. Confirmation therefore pre-registered on the already-frozen
  independent sets (training/guarded_confirm.py, criteria C1-C4 fixed in code before any guarded score):
  SI-HDR v1.1 incl. night groups g09/g13/g15 and highlight non-regression, ToS 03_2a + T1/T3. Mataha
  re-score is diagnostic only (it revealed the defect). Limits: simulated/synthetic SDR, no real graded
  SDR, no unseen production; a production claim still needs new footage or customer acceptance tests.

## 21 October 2026 (later) — full-resolution shortlist results (owner-run, GPU)

- summary.json in production_eval_20261019/shortlist_run02_20261021: 5/5 evaluations complete.
  Gain vs bt1886 baseline, equal-scene 95% CI (251 frames / 25 scenes):
  run02 best.pt 16200 clean +0.878 [+0.371,+1.512], hard +0.208 [+0.087,+0.358];
  16000 clean +0.885, hard +0.159 [+0.031,+0.316]; 19000 clean +0.863, hard +0.143 [+0.043,+0.265].
  Old 7000 hard +0.076 [-0.034,+0.198] and old 11000 hard +0.062 [-0.030,+0.160] FAIL the hard gate.
  All three run02 candidates pass the validation PU gate and beat both references (19000 vs 7000 hard unresolved).
- Frozen rule proposes 16000 (0.05 dB tie band, clean tiebreak by 0.007 dB). Paired data: 16200 minus
  16000 hard +0.049 dB CI>0 (a_better), clean -0.007 unresolved. The tie band ignored a resolved paired
  difference; freezing 16200 needs a recorded owner override (freeze_candidate.py --owner-decision, added).
- Worst scenes common to all candidates: elmatareed a016c032 / a016c030 clean (-0.2 to -0.6 dB);
  inspect visually. Validation PU only; not temporal, CVVDP, independent or release evidence.
- Later 21 Oct: Overlap/SihdrEvaluate/TosBuild failed on compressed EXR (minimal reader). Added training/exr_any.py (OpenEXR, then OpenCV), partial-output dirs renamed only on success; SI-HDR protocol re-frozen as v1.1 (reader-only change, transforms/criteria identical, 0 scores under v1). Empty failed folders renamed _failed_* (kept). Freeze of run02 best.pt (16200) recorded with owner override; test kit prepared (260 frames / 26 scenes).

## 21 October 2026 — run02 completed; audit, frozen shortlist, evaluation and resume tooling

- Run02 verified from files: completed 20,000/20,000, 51,318 s, errors.log empty, 101 finite evals,
  20 numbered checkpoints (model+optimizer+scheduler) + best.pt step 16,200 (weights only). All 21
  sha256 recorded. Run folder untouched. PowerShell Status not run (agent VM).
- Audit (unchanged audit code, real torch, byte-identical copies): 12/21 pass. In-training PU is on
  256 px centre crops. Late-phase run02 clean +0.967 / hard +0.087 dB; old run late +0.831 / -0.002.
  Frozen shortlist before any full-res score: run02 best.pt(16200), 16000, 19000; refs old 7000/11000.
- Full-resolution comparison NOT run (no GPU; validation media on G:\datasets not granted). One-command
  batch: scripts/evaluate_production_shortlist.ps1 (training/production_shortlist_eval.py) with
  hash checks, evidence reuse, pairwise CIs, worst cases and a frozen selection rule.
- Frozen before scoring: SI-HDR protocol rudra-sihdr-v1 (G: independent_protocol_20261021) and ToS
  temporal recipe rudra-tos-temporal-v1 (thresholds T1-T4). New overlap screen, reserved-test kit
  (one candidate), candidate freeze/verify. Wrapper scripts/run_independent_eval.ps1.
- Resume-capable trainer (training/resumable_training.py, scripts/train_resumable_hdr.ps1): exact
  state incl. RNG/sampler/selection history/lineage, atomic + sidecar-verified saves with fallback,
  compatibility and provenance re-validation, stale-lock recovery, research-only warm start. CPU tests:
  bitwise parity with the scratch trainer and interrupted-vs-uninterrupted. Existing checkpoints are
  not exact-resumable (classified legacy / weights-only).
- Drafts: docs/production/ rights & lineage review, model card, I/O contract, operations/rollback/
  integration. Owner rights questions 1-4 remain. 169 passed / 1 skipped (CUDA-only) in cloud CPU.
- Status: implementation complete; validation pending; production not approved; release_authorized=false.

## 20 October 2026 — native1080 run lost; restart failed on CUDA

- Read-only check (agent VM, files only): `G:/rudra_training/fxtd_native1080_hdr_run01`
  was recreated at 21:23 UTC on 19 Oct. The earlier attempt's step_0001000-5000,
  best.pt and train.jsonl (last seen step 5,600 at 19:20 UTC) are no longer in
  that folder; the launcher never deletes, so it was removed outside it. Cause
  of the first stop is not recorded anywhere that remains.
- New attempt PID 14400 passed full preflight (6,462 files) then failed at the
  initial validation transfer: `torch.AcceleratorError: CUDA error: unknown
  error` (status failed, 21:31 UTC). No training steps, no checkpoints.
- Resume is not possible: no resume support, and no surviving checkpoints.
  Lost interim checkpoints had not passed the both-positive PU screen.
- Next: recover/check the GPU (reboot, nvidia-smi, nvlddmkm/TDR events), then
  Check and Start a fresh RunName (fxtd_native1080_hdr_run02). Keep the failed
  run01 folder for audit.
- Owner Check then failed: `Decoder dependency changed: pipeline/hdr_io.py`.
  Cause: candidate `commercial_review.json`/`ready.json` were rewritten outside
  the finalizer at 21:22-21:23 UTC 19 Oct pinning an hdr_io.py version
  (40ee964b) that exists nowhere; hdr_io.py was back at the reviewed/committed
  831e2e93 by 21:26. Agent restored only that pin to the originally reviewed
  value (evidence: preflight_result.json 06:25 UTC), kept as-found copies in
  the corpus `_receipt_incident_20261020/` with INCIDENT.md, and left ready.json
  for the official finalizer to reissue after full re-verification.
- nvlddmkm Event 153 (GPU reset) logged 18 Oct 17:54, 19 Oct 01:22, 20 Oct
  04:55 local: recurring driver resets; avoid other GPU apps during training.

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
