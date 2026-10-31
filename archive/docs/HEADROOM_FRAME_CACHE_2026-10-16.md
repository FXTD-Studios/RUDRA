# Headroom frame cache and asynchronous feeder — 16 October 2026

Implemented and tested an isolated CPU source owner in
`native/video/include/rudra/video/headroom_frames.hpp` and
`native/video/src/headroom_frames.cpp`. The app still does not expose headroom
mode. This increment establishes correctness and bounded pixel storage; it is
not a real-time playback or production-release acceptance result.

## Completed

- `HeadroomFrameSource::open` acquires read leases, probes the source and computes
  the existing full source/decoder/policy identity once. It pins the resolved
  decoder executable instead of resolving PATH for each frame. No disk cache,
  legacy PNG cache, resizing, SDR normalization or model conversion is involved.
- On Windows, leases deny ordinary writes and replacement of the source and
  decoder while the owner is alive. An existing writer prevents opening. File
  identity, size and modification time are checked before decoding, publishing
  and returning cached data. Metadata changes invalidate the cache and preserve
  the original integrity error even when cancellation interrupts an active read.
  Content changes require closing/reopening; new content produces a new identity.
- `request({first,count})` and `take(index)` implement paused reads and asynchronous
  playback on the same exact frame-index path. New requests cancel old work;
  request generations prevent old batches from appearing after a seek. Busy
  timeouts, evicted frames, invalid ranges, source changes and decoder failures
  have distinct errors. There is no fallback to the integer decoder.
- The cache owns immutable float frames. Shape, byte count, contract and finite
  samples are verified. A batch is published only after the decoder verifies
  exact EOF and successful process exit. Truncated, extra, nonfinite and
  late-failing output never becomes a cache hit.
- Pixel storage has a configurable byte budget, default **256 MiB**, covering
  cached frames, decoding/staging reservations and frames still held by consumers
  after eviction. Full-resolution storage costs `width * height * 3 * 4` bytes.
  The default can hold two UHD frames (99,532,800 bytes each); a budget smaller
  than one frame is refused. These UHD figures are calculated, not RSS readings.
- Backpressure waits for consumers to release old frame handles; eviction cannot
  hide their memory use. Cancellation/destruction also works while waiting for
  space. Returned frames remain valid after the owner is destroyed.
- `FloatVideoDecoder::interrupt` now supports concurrent `finish` as well as
  `next`, including waiting for EOF or a child that closed stdout but has not
  exited. Process polling/reaping/killing are serialized inside this decoder.
  Callers must still serialize next/finish and join before close/destruction.
  The shared `Process` implementation and legacy decoder were not rewritten.

## Evidence

**314/314 native tests pass in 31.547 s**, including 12 new tests. No failures,
disabled tests, errors or skips. Evidence:

- `reports/headroom_frames_native_final_20261016.json` and `.log` — final source.
- `reports/headroom_frames_build_verified_20261016.log` — final successful build.
- Earlier focused run: 49/49 in 6.073 s,
  `reports/headroom_frames_focused_20261016.json` and `.log`. The final full run
  additionally covers active source invalidation during decoder cancellation.
- Intermediate complete run: 313/313 in 33.997 s,
  `reports/headroom_frames_native_20261016.json` and `.log`.

Real seven-frame FFV1 footage at 24000/1001 fps with a five-second timestamp offset
matches full sequential float decode exactly through playback, cached paused
reads and backward seeking. Applying the shared analytic headroom grade at +1 EV
also yields bit-identical linear Rec.2020 masters. Signed and above-white values
survive. Existing analytic/color/delivery goldens pass in the full suite.

The final test records **541.4313 ms** to open/probe/hash the tiny source plus the
actual FFmpeg executable and **276.6241 ms** for its seven-frame comparison loop.
The latter includes reference decoding and grading. These are single-run test
diagnostics on a 16x4 fixture, not production throughput, startup guarantees,
competitor comparisons or a claim of image-quality improvement.

Read-lease tests cover blocked content writes/replacement, an existing writer,
metadata-only invalidation of source and decoder, and changed identities after
reopen. Memory tests cover one/two-frame budgets, retained consumers, eviction,
seeks and destruction under pressure. Child-process tests cover cancellation
before/after pipe EOF and all malformed-output cases above.

Whitespace checks pass, including the new untracked files. The initial broad
rebuild emitted existing float-to-uint16 test warnings from `test_video_predict`
and `test_video_master`; final incremental builds have no warnings/errors.
No new app, Python, GPU or visual test run is claimed. Previous 60/60 app and
20/20 Python results remain prior evidence; those paths were not changed here.

## Limits and next acceptance criteria

1. **Measure and improve sustained decoding before app integration.** Each
   verified batch currently starts a decoder and traverses preceding frames via
   indexed trim. This can become quadratic work on a long movie. Batch size
   defaults to two, configurable from 1 to 32 within the byte budget. This
   correctness scaffold is not an efficient long-video feeder yet. Benchmark
   representative HD/UHD clips against direct sequential float decoding, then
   preserve the same frame/EOF/error guarantees in the optimized path.
2. Source opening uses full file hashes and the existing full-frame timing probe;
   perform it outside the UI thread. Opening/probing/hashing has no cancellation
   API yet. Only decoding and cache waits have bounded cancellation tests.
3. The byte budget covers pixel allocations owned by this source, including its
   borrowed frames. FFmpeg process memory, probe/metadata storage, caller-created
   copies, render buffers and other source owners are outside that budget.
   OS peak memory and sustained full-resolution throughput remain unmeasured.
4. Enforced read leases currently require Windows. Other platforms fail closed;
   portable source-lifetime protection needs implementation and testing. Current
   eligibility remains tagged progressive square-pixel 8/10-bit YUV444; no alpha,
   rotation or 4:2:0 support was added. Unicode/network-path acceptance is unrun.
5. Then connect session policy/undo, preview/display/scopes/probes and export jobs
   to this float path and the shared headroom renderer; write actual provenance.
   Keep the default project-reader compatibility guard until all consumers work.
   App paused/playback/export parity, encoded-output acceptance, GPU parity and
   calibrated HDR review remain required before UI enablement.

No source dataset edits, training, downloads, model promotion, live app rebuild,
installer refresh, commit or release. Test binaries and their CLI dependency were
rebuilt. Distribution packages remain stale. Commercial rights, signing,
clean-machine acceptance and visual review remain unresolved release gates.
