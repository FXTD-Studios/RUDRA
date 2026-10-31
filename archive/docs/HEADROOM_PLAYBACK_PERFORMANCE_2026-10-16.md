# Headroom playback performance — 16 October 2026

The measured bottleneck is repeated decoder startup and repeated traversal of
earlier frames. This increment improves the isolated FFV1 feeder and reduces
decoder memory use. **The feeder is still too slow for real-time playback.**
Headroom remains unavailable in app/CLI selections; project compatibility guards
and the existing application pipelines remain intact.

## Implemented

- Added an explicit diagnostic executable, `rudra_headroom_frames_bench`, excluded
  from normal builds/installations, and `tools/benchmark_headroom_frames.py`.
  The runner generates tagged synthetic HD/UHD fixtures, preserves reports,
  alternates baseline/candidate order, measures preparation/feed/first-frame
  times, samples process-tree memory and checks every float frame's XXH64.
- Float decoding now uses four decoder threads. The initial 4/8/automatic thread
  sweep preserved all frame hashes; four threads reduced measured process memory
  substantially without a material HD timing penalty in that initial sweep.
  Legacy integer decoding and inference settings were not changed.
- The cache evicts enough eligible past frames for the next batch, while keeping
  retained consumer allocations inside its byte budget. Added actual decoder
  start counts and a peak pixel-allocation counter for diagnosis.
- `float_playback_decoder_command` uses input seeking only for probed FFV1 in
  Matroska `.mkv` files with an unambiguous validated constant frame clock.
  Selection boundaries lie halfway between adjacent frame timestamps; input
  seeking accounts for the container start, and absolute timestamp trimming
  selects the requested frames. The same float conversion and exact frame-count,
  finite-value, EOF, process-exit and cancellation checks remain in place.
- Other codecs/containers, missing timing metadata, negative starts and coarse
  clocks retain indexed float traversal. A new cache-identity component records
  the playback selection recipe and representative arguments, including container
  timing; the codec name and changed decoder arguments also change the identity.

The seek path follows FFmpeg's documented input-seek and timestamp-preservation
options. Acceptance comes from the pinned 7.1.1 executable and the tests below.
[FFmpeg option documentation](https://ffmpeg.org/ffmpeg.html#Main-options).

## Measured comparison

Three paired repetitions per mode/resolution: **36 runs**, two 24-frame synthetic
FFV1 clips at 24000/1001 fps with a five-second timestamp offset. Machine: Intel
i9-14900KF, 24 cores/32 logical processors, approximately 128 GiB RAM, Windows.
The source files total 37.2 MB. No source datasets, training, downloads or GPU work.

Times are medians for consuming all 24 frames, including the same consumer hash
work but excluding source preparation/probing/hashing. Memory is the median of
each run's sampled peak sum of process working sets, including preparation and
child processes. It is neither unique physical memory nor an exact OS peak.

| Feeder | Before | After | Measured speed ratio | Sampled tree peak before → after |
| --- | ---: | ---: | ---: | ---: |
| 1920x1080 | 6.781 s | 4.021 s | 1.69x | 544.8 → 417.1 MiB |
| 3840x2160 | 28.664 s | 24.454 s | 1.17x | 1348.4 → 762.9 MiB |

HD feed-time ranges were 6.389–7.193 s before and 3.555–4.022 s after; UHD ranges
were 27.774–31.780 s and 23.091–25.684 s. First verified frame medians improved
from 334 to 211 ms for HD and 1134 to 680 ms for UHD, after source preparation.

The unchanged legacy decoder is a timing control with different pixel semantics.
Direct sequential float decoding is the common-frame reference:

| Direct decode control | HD before → after | UHD before → after |
| --- | ---: | ---: |
| Sequential float | 0.632 → 0.582 s | 2.814 → 2.410 s |
| Legacy integer decode + float normalization | 0.610 → 0.622 s | 2.484 → 2.782 s |

The unchanged legacy control varies between runs, so these are local diagnostic
results with three repetitions, not general performance guarantees. Sequential
float timing ranges overlap. No competitor or overall HDR quality comparison is
claimed for this decode-only change. Existing analytic/color/delivery goldens pass.

Every float frame hash matched across the frozen baseline, candidate, feeder and
sequential paths. Legacy control hashes also matched before/after. The high-resolution
checks use XXH64; the small seek fixtures additionally compare every float sample.

The candidate's exact peak owned pixel allocations were 248,832,000 bytes for HD
and 199,065,600 bytes for UHD, inside the unchanged 256 MiB budget. The UHD budget
holds two full frames. Consumer retention causes single-frame prefetch batches:
the measured feeder still starts **23 decoders for 24 UHD frames**, versus 12 for
HD. Overall measured feeder rates are only **5.97 HD fps and 0.98 UHD fps**.
Even direct sequential UHD decode is only about 9.96 fps in this diagnostic.

Source preparation also remains expensive: candidate medians were 1.751 s for
HD and 4.757 s for UHD. Full timing probes and content hashes remain; no faster or
cancellable source-opening claim is made. Pixel-budget limits exclude FFmpeg,
metadata/probes, render buffers, caller copies and other source owners.

## Validation and reproducible evidence

- **317/317 native tests passed**, 66.788 s, no failures/errors/disabled/skipped
  tests. Report: `reports/headroom_perf_native_20261016.json` and `.log`.
- Added three native tests and extended identity coverage: batch reclamation,
  guarded seek selection, and exact sample parity over 18 combinations of 8/10-bit,
  25/24000:1001/30000:1001 fps and 0/5/123.456-second offsets. The 31-frame fixtures
  use 12-frame keyframe intervals and exercise seven ranges around boundaries.
  Additional coverage verifies video starting five seconds after audio.
- Existing cancellation, memory-pressure, source invalidation, malformed output,
  late decoder failures, project compatibility and HDR math tests still pass.
- An initial absolute-seek attempt failed on offset footage; failures were
  retained in `headroom_perf_focused_20261016`. Relative input seeking based on
  the actual container start fixed the issue without relaxing any output checks.
  An initial generated benchmark fixture lacked transfer metadata and was refused;
  corrected generation uses explicit frame colour tags. Initial evidence remains.
- Final builds: `headroom_perf_final_build_20261016.log` and
  `headroom_perf_verified_build_20261016.log`; no compiler warnings/errors.
  Whitespace checks pass, including new files.
- The test process still emits its pre-existing ONNX Runtime API 22/17 startup
  warning. These tests do not establish neural runtime readiness; resolve test
  DLL selection before model/backend acceptance. App/Python/GPU suites were not
  rerun, and the live application executable was not rebuilt.

Benchmark evidence under `reports/headroom_perf_20261016/`:

- `paired/summary.json`: all 36 measurements, source/FFmpeg/executable hashes and
  per-frame hashes; paired row reports and logs are retained alongside it.
- `comparison.json`: median/range calculations and verified before/after parity.
- `baseline.exe`, `candidate.exe`: frozen executables for this comparison.
- `hardware.json`, `source_hashes.json`: environment and relevant code fingerprints.
- `baseline_sweep/summary.json`: initial decoder thread sweep, separate from the
  paired comparison. `fixtures_tagged/` holds the valid generated media.

Reproduction (with the pinned FFmpeg and required DLLs on PATH):

```powershell
cmake --build build/native_phase3 --config Release --parallel 4 --target rudra_headroom_frames_bench
.venv/Scripts/python.exe tools/benchmark_headroom_frames.py --binary build/native_phase3/tests/Release/rudra_headroom_frames_bench.exe --compare-binary reports/headroom_perf_20261016/baseline.exe --fixtures reports/headroom_perf_20261016/fixtures_tagged --out reports/headroom_perf_recheck --repeats 3
```

Use a new output directory; the runner refuses an existing one. Measurements use
local synthetic short clips and do not certify long real-footage throughput,
Unicode/network paths, other codecs or portable platforms. Sampled working-set
sums can count shared pages multiple times and miss peaks between samples.

## Next action

Review a **continuous decoder design with a stronger reasoning model**, then
implement an isolated, bounded prototype. It must explicitly define when frames
may be previewed, when they are safe to reuse, and when an export is complete.
Do not quietly remove EOF/process-exit checks to publish unverified batches, or
simply enlarge the memory budget to hide the restart problem. Retain generation,
source-identity, late-failure and cancellation guarantees in tests. Benchmark the
prototype against these frozen executables and the direct sequential reference.

App/session/preview/scopes/export/provenance integration remains gated. Existing
read-lease portability, source-open cancellation, calibration/visual review,
commercial rights, signing and clean-machine gates remain unresolved. No training,
model promotion, installer refresh, commit or release was performed.
