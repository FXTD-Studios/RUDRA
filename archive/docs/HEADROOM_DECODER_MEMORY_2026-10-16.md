# Headroom decoder memory: measured FFV1 profile

## Scope and decision

The previous continuous prototype removed repeated decoder starts but raised UHD
process-tree memory to about 1.2 GiB. This increment measures separate input,
filter, raw-output and output-queue settings before changing the float decoder.
Existing artist controls, colour transforms, delivery formats and app defaults
are unchanged. Headroom remains unavailable in the app; reader guards stay active.

Added diagnostic thread/queue overrides, separately sampled FFmpeg/ffprobe process
peaks, memory traces and a slow-consumer benchmark. These are benchmark-only
options, not new artist controls. Frozen probe binaries preserve pre-change
decoder behavior and the results can be reproduced from local generated media.

**34 diagnostic runs** (two repetitions per tested configuration across three
adaptive sweeps) retain every float frame hash, including comparison to the
previous frozen continuous prototype. The most useful measured change is reducing
FFV1 input workers from four to two. One input worker uses less memory but is
substantially slower; limiting filter threads did not show a useful additional
benefit. Raw output packing with one worker reduces memory further. A one-packet
output queue caps that queue without establishing a cap on all FFmpeg allocations.

The selected profile is **two FFV1 decoder threads, unchanged filter threading,
one raw-output thread and one queued output packet**. It is scoped to sources
whose codec is `ffv1`; other codecs retain their existing settings until measured.
The command is part of the existing content cache identity, so changed commands
automatically receive different cache keys. Provisional-frame and completion
receipt semantics are unchanged.

FFmpeg options apply to their corresponding input or output. The pinned binary's
rawvideo encoder reports frame-thread support; FFmpeg 7.1's encoder implementation
uses the CPU count for automatic threading and retains tasks per worker. This
supports investigating output threading but does not explain all observed memory
by itself. [FFmpeg option scoping](https://ffmpeg.org/ffmpeg.html#Description),
[FFmpeg 7.1 frame-thread encoder](https://ffmpeg.org/doxygen/7.1/frame__thread__encoder_8c_source.html).

The local 7.1.1 help text describes `thread_queue_size` only as a demuxer setting,
but the 7.1 source marks it for both input and output. Output placement is supported
and tested here. This queue setting is distinct from initial multi-stream muxing
queues. [FFmpeg 7.1 option definitions](https://ffmpeg.org/doxygen/7.1/ffmpeg__opt_8c_source.html),
[FFmpeg scheduler queues](https://ffmpeg.org/doxygen/7.1/ffmpeg__sched_8c_source.html).

These are process-level ablations, not a heap allocation profile proving the
ownership of each byte. No FFmpeg patch, binary replacement, new decoder library,
new training or dataset download is involved.

## Validation and performance

**328/328 native tests pass**, 45.745 seconds, with no failures/errors, disabled
or skipped tests. Added a command-scoping regression for FFV1 versus other codecs
and a real FFmpeg blocked-output lifecycle test. Both explicit cancellation and
destruction promptly release a decoder while a consumer retains its sole pixel
slot. Existing exact-range, source identity, malformed output, late failure,
receipt, policy compatibility and analytic/colour/delivery tests pass.

Evidence: `reports/headroom_memory_native_20261016.{json,log}`,
`headroom_memory_focused_20261016.{json,log}` and
`headroom_memory_build_20261016.log`. The pre-existing ORT API 22/17 startup
warning persists; no neural/backend acceptance is claimed. App/Python/GPU suites
were not rerun. App/packages were not rebuilt; the CLI test dependency was.

The **36 paired short-clip runs** are complete, with three repetitions of both
frozen binaries on the 24-frame HD/UHD fixtures. Stream/sequential float hashes
and unchanged legacy-control hashes all match. Median continuous-stream results:

| Measurement | HD before → after | UHD before → after |
| --- | ---: | ---: |
| Feed including hash and completion | 626.2 → 602.6 ms | 2393.1 → 2314.8 ms |
| Sampled process-tree working-set sum | 344.8 → 199.0 MiB | 1371.5 → 755.0 MiB |
| Reduction in sampled tree memory | 42.3% | 44.9% |

These feed timings exclude source preparation, grading, inference, display and
delivery encoding. Three repetitions do not establish a general speed guarantee.
Both variants use the same updated diagnostic monitor; no timed tests overlap
with builds, regression tests or fixture generation.

The **eight longer paired stream runs** also pass: two repetitions on each
240-frame HD/UHD synthetic clip, ten times the previous frame count (10.01 seconds
of source video). Generating both fixtures used 349.0 MiB; no source datasets were
accessed or changed. Median results:

| Measurement | HD before → after | UHD before → after |
| --- | ---: | ---: |
| Feed including hash and completion | 5.001 → 5.012 s | 23.036 → 20.637 s |
| Sampled process-tree working-set sum | 768.0 → 500.7 MiB | 2950.1 → 1899.7 MiB |
| Source preparation | 9.180 → 9.187 s | 34.738 → 35.356 s |

The longer samples retain a substantial memory reduction, but their absolute
memory use is higher than on the short clips. Traces show memory continuing to
rise while buffers fill; for example, candidate UHD middle-quarter medians of
1387–1409 MiB rise to 1696–1750 MiB in the following quarter. This is **not** proof
of a steady memory ceiling, absence of long-duration leaks or production playback
readiness. UHD baseline feed times vary from 20.97 to 25.11 seconds across only
two repeats; do not infer a general speed improvement from these medians.

The remaining **ten validation runs** also pass:

- Two 240-frame direct sequential references match both continuous variants.
- Four UHD slow-consumer runs (250 ms intentional delay while retaining each
  frame) preserve every frame and validate completion. Median sampled tree memory
  falls from **2409.3 to 1745.5 MiB**; feed times including intentional waits are
  8.604 to 8.518 seconds. These rates are not throughput measurements.
- Four batch-feeder controls preserve the existing fully verified publication
  semantics. Single paired runs: HD 3446.8→3243.0 ms and 405.1→354.2 MiB;
  UHD 22471.1→20954.7 ms and 753.6→553.8 MiB. One pair is a regression diagnostic,
  not sufficient evidence for a general speed claim.

In total, **88 benchmark runs passed: 34 diagnostic sweeps and 54 validation
runs**. Combined analysis verifies all float hashes across configurations,
short/long references and slow/batch consumers, plus unchanged legacy controls.
Short-fixture hashes also match the previous increment's frozen evidence. Every
continuous run uses one decoder, validates its receipt and stays within the
unchanged 256 MiB owned-pixel budget (observed one-frame peaks 24,883,200 bytes HD,
99,532,800 bytes UHD). New command settings produce new content cache keys;
keys stay stable across repetitions within each variant.

The longer test's sampled tree-memory reduction is **34.8% HD / 35.6% UHD**,
smaller than the short-test percentage. UHD remains only **11.63 fps** in the
long continuous diagnostic, before grading/display/encoding. The full pipeline
is not certified real time. The owner budget does not cover FFmpeg, source
probes, render buffers or caller copies. Working-set sums may double-count
shared pages and sampling can miss peaks; traces do not establish a hard cap.

## Evidence and reproduction

Artifacts under `reports/headroom_memory_20261016/`:

- `sweep`, `queue_sweep`, `refine`: adaptive diagnostic reports, commands, traces
  and per-process peak measurements. Single-input-worker slowdown and unhelpful
  filter-thread experiments are retained rather than silently omitted.
- `paired_short`, `paired_long`, `long_reference`, `slow_consumer`, `batch_control`:
  all final row reports, memory traces and summary files, with executable,
  FFmpeg and fixture SHA-256 plus per-frame XXH64 values.
- `comparison.json`, `summarize.py`: combined assertions and median/range
  calculations, including long-run memory windows and cross-stage parity.
- `baseline.exe`, `queue_probe.exe`, `candidate.exe`: frozen binaries. Baseline
  adds diagnostic options but retains the previous decoder settings; hash parity
  with the previous frozen stream is separately verified. `source_hashes.json`
  fingerprints the final implementation, tests and benchmark tools.
- `fixtures_240`: generated HD/UHD clips, tagged 10-bit YUV444 Rec.709 limited,
  24000/1001 fps with a five-second timestamp offset; 349.0 MiB combined.

Example reproduction with pinned FFmpeg and required native DLLs on PATH:

```powershell
.venv/Scripts/python.exe tools/benchmark_headroom_memory.py --binary reports/headroom_memory_20261016/candidate.exe --baseline reports/headroom_memory_20261016/baseline.exe --sources reports/headroom_memory_20261016/fixtures_240/hd_240frames_ffv1.mkv reports/headroom_memory_20261016/fixtures_240/uhd_240frames_ffv1.mkv --out reports/headroom_memory_long_recheck --modes stream --repeats 2
```

Use a new output directory. `--sweep --profiles decoder,filter,output,queue`
supports controlled diagnostics with the frozen `queue_probe.exe`;
`--modes stream --delay-ms 250` exercises a slow consumer. Default queue behavior
differs between baseline and candidate, so retain explicit profiles/frozen binaries
when reproducing ablations. The monitor caps each case at 180 seconds and only
terminates its own process tree on timeout. No case timed out in this run.

Build and whitespace checks pass. No app/package refresh, training, download,
dataset write, model promotion, commit or release was performed. No external
competitor, neural-quality or visual HDR acceptance result is claimed.

## Remaining work

Do not enable the app path based on these decoder benchmarks. Source preparation
still probes/hashes synchronously, taking about 35 seconds for the 10-second UHD
fixture. Next: shared protected-source ownership and
cancellable preparation, then an asynchronous adapter with memory accounting
across retained old/new generations. Session/undo, preview, scopes, export staging
and provenance must all honor the new policy and errors before UI enablement.
Decode completion alone still does not certify an encoded export.

Long production footage, other codecs, network/Unicode paths, platform portability,
commercial/model rights, calibrated HDR review, clean-machine acceptance and
release signing remain separate gates. The live app and release packages are not
updated by this work.
