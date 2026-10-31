# Asynchronous headroom playback — 17 October 2026

Status: implementation, native/concurrency validation and all 15 bounded
performance runs complete. Isolated native prototype; app headroom selection is
still disabled and default project-reader guards remain active.

## Implementation and contract

`HeadroomPlayback` schedules opening and seeking by generation. One decoding
worker performs source preparation, reads and completion. A separate cancellation
worker interrupts the old operation without executing probe/decoder waits on the
request thread. Only the latest pending request survives. New opens replace the
source selection; seeks reuse completed protected preparation. A seek during
preparation cancels/restarts that preparation with the newest range. Invalid
requests that can be checked immediately leave the active generation intact;
range checks needing a new source's metadata complete asynchronously.

Requests do no file I/O, decoding or joins. The caller can poll `video`, `take`
and `completion` with their default zero wait, or use a bounded wait off the UI
thread. `cancel` invalidates the generation immediately and schedules cleanup.
Destruction is blocking: it interrupts and joins both owned workers. External
callers must be joined before destroying the adapter. Worker failures are reported
as errors, not silent fallback to the legacy clamped path.

`HeadroomStream::close` explicitly interrupts/reaps before replacement, including
when a concurrent validator retains the old stream owner. This is necessary:
shared ownership alone would otherwise allow its process to outlive the worker's
last local reference. Decoders are closed before the next generation launches.
Preparation/decoder work and validation run outside the adapter state mutex;
results recheck the generation before publication or return.

Preview frames and completion receipts identify both adapter and generation.
Old or foreign tokens are rejected. Frames remain provisional until exact EOF,
successful child exit and current source identity have been verified. Late
failures clear queued output, revoke preview validation and issue no receipt.
Verified frame promotion is explicit. A decode receipt is still **not** an
encoder/delivery/file-publication receipt. The display consumer must validate
before presentation and serialize its final presentation decision with its own
seek handling; a validation call cannot prevent a later request invalidating it.

## Shared memory accounting

`HeadroomPixelBudget` reserves bytes **before** native frame allocation. An
immutable frame's shared storage holds the reservation, releasing it only after
its pixel storage is destroyed. Promoted frames retain the same reservation.
Frames borrowed from retired/destroyed decoder owners therefore still consume
the allowance. Busy leaves the next frame unread; releasing pixels permits it
to advance. Allocation/decoding failures release staging reservations.

The adapter uses one allowance across all of its generations (default 256 MiB),
plus a bounded ready queue (default two frames). It waits for memory before
starting a decoder, and never evicts a consumer's live pixels to make room. Queue
changes and cancellation wake the worker immediately; borrowed-pointer release
is checked at most every 5 ms while waiting for space.

This is **per adapter**, not a process-wide or GPU memory cap. FFmpeg buffers,
metadata and explicit caller copies are excluded. The separate verified batch
`HeadroomFrameSource` remains per-owner; do not mix an independent batch cache
into the same application's advertised shared cap without accounting for it.
The adapter supports paused one-frame ranges as well as continuous ranges.

## Verification

- Build passes without compiler warnings/errors; no failed compile this increment.
- Initial focused run: **42/42 tests pass**, 30.158 s. Two additional cases and
  final shutdown handling are included in the full native run below.
- **350/350 native tests pass**, 67.918 s, zero failures/errors/disabled/skipped.
  Fourteen new tests cover cross-owner pixel lifetime, retained/promoted handles,
  real sequence/seeks and +1 EV analytic-master parity, rapid seeks and old
  waiters, budget waits, both probe passes before/after EOF, source replacement
  and seek during preparation, late/malformed output, blocked read/EOF/exit,
  real FFmpeg backpressure/destruction, identity changes and foreign tokens.
  Existing colour, legacy, HDR delivery, cache and source-protection tests pass.
- **60/60 shuffled concurrency executions pass**: six race/lifetime cases,
  ten iterations, seeds 17017–17026. Not an exhaustive race proof or sanitizer run.
- The full suite's 31-request burst took **0.0141 ms** at the request API. This
  measures queue submission on a tiny fixture, not decode or application latency.
- Evidence: `reports/headroom_playback_native_20261017.{json,log}`,
  `headroom_playback_focused_20261017.{json,log}`,
  `headroom_playback_stress_20261017.log`, and
  `headroom_playback_20261017/stress_summary.json`.
- The pre-existing ORT API 22/17 warning persists. App/Python/GPU acceptance is
  not claimed; those tests were not rerun and app/packages were not rebuilt.

## Performance

**15/15 cases pass**, with 360 frame-hash comparisons, unchanged source/cache
identities and unchanged fixture hashes. Results also match the earlier frozen
FFV1 memory increment. No case timed out. Full evidence is in
`reports/headroom_playback_20261017/paired/summary.json`; reconciled checks and
medians are in `comparison.json`. The candidate executable and source fingerprints
are frozen alongside them.
The runner compares the previous frozen synchronous stream, current synchronous
stream (shared-budget storage added), and asynchronous adapter. It uses the
existing 24-frame HD/UHD FFV1 fixtures, two pairs in reversed order, plus one UHD
comparison with 100 ms consumer holds. Preparation is timed separately; async
feed begins with a prepared source so worker dispatch/queuing can be compared.
Every run checks all frame hashes, unchanged source identity, completion receipt,
one decoder start, expected frame count and the 256 MiB native pixel allowance.
Each case has a 180 s limit. No build/tests overlap timed cases.

| Measurement | Prior stream | Current stream | Background adapter |
|---|---:|---:|---:|
| HD 24-frame feed + hash | 521.43 ms | 527.20 ms | **518.66 ms** |
| UHD 24-frame feed + hash | 3003.26 ms | 3020.87 ms | **2281.34 ms** |
| HD first frame after preparation | 118.07 ms | 117.29 ms | 117.46 ms |
| UHD first frame after preparation | 421.56 ms | 414.14 ms | 419.52 ms |
| HD native pixel peak | 23.73 MiB | 23.73 MiB | 47.46 MiB |
| UHD native pixel peak | 94.92 MiB | 94.92 MiB | **189.84 MiB** |
| HD sampled process-tree peak | 164.77 MiB | 163.33 MiB | 166.41 MiB |
| UHD sampled process-tree peak | 1270.00 MiB | 1315.73 MiB | 936.62 MiB |
| UHD feed with 100 ms consumer holds | 6260.90 ms | 5866.51 ms | 3827.24 ms |
| UHD slow-consumer sampled tree peak | 1601.20 MiB | 1602.39 MiB | 1601.71 MiB |

Normal timings/tree peaks are medians of two runs per variant; pixel peaks are
maxima. Slow-consumer rows have one run per variant and deliberately include
consumer waits. Background request submission measured **0.0027–0.0104 ms**
after preparation. This is API dispatch, not GUI responsiveness or first-pixel
latency. Short UHD feed is 24.0% faster than the previous stream; HD is effectively
unchanged at this sample size. Current synchronous stream medians differ by
+1.1% HD / +0.6% UHD; these small differences are not a general regression claim.

The adapter overlaps decoding and consumption using a bounded queue; its native
pixel peak doubles here to two frames and stays below 256 MiB. Total process
memory is not bounded by that allowance; slow-consumer UHD still reaches about
1.56 GiB. UHD feeds **10.52 fps** on these samples, below real time before any
grade/render/display. Preparation still takes about 1.39 s HD / 4.94 s UHD for
these one-second clips and remains a separate bottleneck. Longer production
footage, multiple consumers and full application behaviour remain unvalidated.

Reproduce with pinned FFmpeg/ffprobe on PATH and a fresh output directory:

```powershell
.venv/Scripts/python.exe tools/benchmark_headroom_playback.py `
  --candidate reports/headroom_playback_20261017/candidate.exe `
  --baseline reports/headroom_preparation_20261016/candidate.exe `
  --sources reports/headroom_perf_20261016/fixtures_tagged/hd_24frames_ffv1.mkv `
            reports/headroom_perf_20261016/fixtures_tagged/uhd_24frames_ffv1.mkv `
  --out reports/headroom_playback_repeat --repeats 2
```

These are decode/hash tests, not application, render, display, export, neural
quality or external-competitor measurements. Process memory is sampled working-set
sums, not unique physical RAM or a hard bound. Small synthetic samples and local
load limit generalization. Initial full preparation and UHD decode speed still
require production-footage work; this increment does not change probing, colour
science, decoder settings, HDR delivery math or training.

## Next integration gate

Connect the isolated adapter to session/source-policy state and preview/paused
frame consumers, with undo/restore and stale-presentation checks. Keep app mode
disabled until scopes, controls, provenance and transactional export all support
the policy. Preserve the source-encoded float contract and existing analytic
colour/delivery path. Add application-level load/seek/close/undo/failure tests
before enabling any selection. Validate longer footage and multiple consumers;
the current short performance samples do not close those gates.

Existing commercial model/quality, dependency/codec rights, calibrated HDR review,
signing, clean-machine and macOS release gates remain. No training, model promotion,
downloads, source dataset edits, commit or release.
