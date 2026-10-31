# Isolated float video decoder

16 October 2026. First implementation gate from the
[super-white review](SUPER_WHITES_REVIEW_2026-10-16.md).

## Implemented

- `native/media/include/rudra/media/video_decode_float.hpp` and its source add
  a separate FloatVideoDecoder. Existing VideoDecoder and its filters are
  unchanged; no application, CLI delivery or model caller selects the new API.
- ExtendedVideoFrame carries planar encoded RGB with signed/headroom values
  and the original input contract. It has no implicit SdrImage conversion.
  The contract describes original range, while output nominal black/white
  are normalized to 0/1; output is neither canonical sRGB nor scene-linear HDR.
- Public file opening requires tagged limited-range Rec.709-transfer, 8/10-bit
  YUV444, supported BT.709/BT.2020 matrix/primaries, progressive square pixels
  and no rotation or alpha. Unknown/contradictory tags, HDR, 420/422 and other
  depths are refused. Source eligibility is checked before launching FFmpeg.
- FFmpeg planar G/B/R bytes are read directly into final G/B/R positions in
  an RGB buffer, avoiding a second frame-sized buffer or reorder copy. No
  clamp, EOTF, gamut conversion, reconstruction or grading is applied.
- Dimensions/byte-count overflow, NaN/Inf, partial/missing/extra frames and
  process failures are checked. Terminal read failures cannot become success
  at finish. An early finish refuses without consuming the unread frames.
- Cancellation unblocks next() and returns Cancelled. API contract requires
  joining the reading/interrupting threads before finish/close/destruction;
  concurrent finish/close is not supported. Process/pipe infrastructure is reused.

## Verification

Release build passed; **274/274 native tests passed** in 39.194 seconds,
including the unchanged legacy tests and the 10 new tests. No app UI code
changed; the app suite and GPU benchmarks were not rerun for this isolated API.

10 focused tests pass with pinned FFmpeg 7.1.1. Tests cover plane order and
signed/headroom preservation, metadata, frame-count lifecycle, truncation at
all plane boundaries, all three nonfinite categories in each plane, invalid
sizes/overflow, startup/process failures (including a valid frame then failure),
empty output, blocked-read cancellation and prototype eligibility refusals.

The live test creates four tiny lossless FFV1 movies: 8/10-bit YUV444, each
with BT.709 and BT.2020 matrices. Neutral ramps and colored patches match an
independent double-precision matrix within 2e-6. All 19/79 tested above-white
levels remain distinct. The same movies through legacy VideoDecoder collapse
those levels to integer 65535. Negative values and >1 RGB color excursions
also survive. This is decoder fidelity evidence, not rendered HDR quality.

Initial focused run: 9/10 passed; the generated FFV1 fixture lacked transfer
and primaries tags, so the input gate correctly refused it. The fixture now
tags the raw input as well as the encoder output; validation was not weakened.
Initial report and build log are retained. A fixture narrowing warning was
also resolved with an explicit conversion; final build is clean of diagnostics.

Evidence:
- `reports/float_decoder_focused_pass_20261016.{json,log}`
- `reports/float_decoder_build_fix_20261016.log`
- Full native regression: `reports/float_decoder_native_20261016.{json,log}`

## Remaining / next step

Add and test a versioned, opt-in analytic BT.1886 baseline for extended input,
following the review's policy and explicit restrictions. Then share decoding
across paused preview, playback and both master/movie export, with separate
cache keys and saved project/provenance settings. This decoder is intentionally
not exposed in the UI yet: existing downstream clamps and clip fill would lose
or replace its recorded headroom.

Alpha, subsampled/compressed production inputs beyond these lossless fixtures,
other source curves, AI, calibration/reference, temporal and GPU parity, actual
throughput and calibrated visual review remain unvalidated for this new path.
No real-time performance claim. The new raw pipe uses twice the bytes of RGB48;
its final RGB buffer was already float in the existing architecture.

No app restart, package refresh, dataset writes, training, downloads, commit,
push or release occurred. Current release blockers remain unchanged.
