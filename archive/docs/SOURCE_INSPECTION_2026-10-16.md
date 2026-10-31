# Source inspection and next color-science review

First increment of the approved usability improvements, 16 October 2026.

## Implemented

The Source card shows cached movie stream headers: dimensions, pixel format,
primaries, transfer, matrix and range. Unknown tags remain unknown. It warns
about limited-range headroom loss and PQ/HLG input to the SDR workflow.
Warnings clear when the source changes. A stream-header probe is added to
movie opening; no extra full-frame scan is performed. This remains synchronous
like the existing open path; large/unresponsive media need future bounded,
asynchronous import handling.

Still sources currently receive profile guidance only. Per-frame ICC details,
input overrides and measured super-white detection are not implemented here.
Pixel decoding, source curves and delivery settings are unchanged.

## Stronger review before super-white implementation

Completed later on 16 October:
[measured super-white review](SUPER_WHITES_REVIEW_2026-10-16.md).
Use that report for the next implementation; the recommendations below were
the questions entering review.

Recommend Astra High/Extra High for the next review; routine UI/tests can use
Light. This is a task-risk recommendation, not a measured model benchmark.
Do not apply the earlier design note as a complete implementation recipe:

- Paused movie extraction in media/sequence.cpp uses FFmpeg PNG conversion;
  playback.cpp uses raw rgb48le; delivery uses decoder_filter. Audit all paths.
- Validate limited RGB versus limited YUV scaling with synthetic native-depth
  ramps before choosing an integer or float representation.
- Decide source-curve extensions, calibration/reference behavior and model
  inputs together. Do not pass new >1 values into unvalidated checkpoints.
- Define measured headroom versus true clipping from source codes, including
  per-channel cases. Do not simply change a threshold on expanded RGB.
- Require ordinary-range regression parity, distinct monotonic above-white
  output, CPU/GPU parity and temporal checks before enabling preservation.

The previous design's full-range conversion workaround is not a guarantee:
mapping headroom to an integer full-range ceiling can also discard detail.

## Remaining approved product work

Super-white preservation; actual still-profile inspection and safe input
overrides; additional still/sequence delivery options; short trial exports;
readable QC reports; robust export failure recovery. The current export sheet
already describes formats, encoding, destination and frames; improve it rather
than duplicate its guidance. Scene-aware reconstruction remains research work.

No training, downloads, source-dataset changes or release performed.

## Validation

Release build passed. App suite: 60/60 (59.805 s); native suite: 264/264
(35.656 s), with pinned FFmpeg 7.1.1. Reports:
`reports/source_inspection_app_20261016.{json,log}` and
`reports/source_inspection_native_20261016.{json,log}`. Whitespace check passed.
Updated RUDRA executable launched; manual visual layout review remains pending.
No throughput or image-quality gain is claimed for this metadata-only change.
