# Headroom grading and delivery boundary — 16 October 2026

The next integration prerequisite is implemented and tested: a shared CPU
render function applies the artist grade to the preserved float source, and
an explicit unit adapter passes its result to existing movie frame preparation.
**293/293 native tests pass. This is not yet an app-selectable mode.**

## Implemented

- `native/video/headroom_render`: `render_headroom_frame` returns a typed
  `HeadroomMaster`, linear Rec.2020 at **203 nits per unit**, carrying the policy
  and original source contract. Default-disabled opt-in remains mandatory.
- Reuses the existing master exposure, Rec.2020 luminance qualifiers and
  painted-mask sampling. Exposure precedes Region EV; masks gate the region
  adjustments, not global exposure. Neutral settings exactly retain the baseline.
- The model's 40,000-nit ceiling is not applied to this separate source-light
  path. There is no display shoulder in the linear master. Finite/range checks
  reject numerical overflow instead of silently substituting settings.
- `master_headroom_video_frame` explicitly rescales 203-nit units into the
  existing delivery code's 10,000-nit convention, once. This is a unit conversion,
  not SDR canonicalization, network inference, or the legacy composite.
  Existing peak/knee behavior, PQ/HLG/log encoding and statistics are reused.
  Unknown format identifiers fail rather than selecting a fallback.
- Existing production formulas, app settings and delivery callers are untouched.
  No new source cache, project field, CLI flag or UI option is exposed yet.

## Artist-control contract for this first mode

| Control | Behavior / restriction |
| --- | --- |
| Reconstruction | Analytic BT.1886, same source eligibility as the float decoder/baseline |
| Exposure | Supported, finite −8 to +8 EV |
| Region EV | Existing soft qualifiers; finite −8 to +8 EV, finite positive ordered thresholds |
| Painted masks | Supported with matching dimensions/planes and existing bands; loaded before rendering |
| SDR anchoring, chroma carry, grain settling | Must explicitly be off; their current implementations require canonical SDR |
| Recovery mode / residual strength / preservation | Must be neutral: all / 1 / on |
| Clip fill/peak overrides, calibration/reference | Still unsupported, rejected by baseline validation |
| Model checkpoint / tiled inference provenance | Rejected: this mode runs no model |
| Movie peak / knee | Existing delivery shoulder; does not mutate the linear master |
| Log delivery | Existing convention: no peak/knee shoulder |
| Temporal smoothing / static hold | Not part of this stateless API; integration must not claim they ran |

These are restrictions of the new opt-in path. No existing control was removed
from the current application. The render function refuses incompatible requests;
it does not switch artist settings off on the user's behalf.

## Measured evidence

- Release builds completed without compiler warnings/errors:
  `reports/headroom_render_build_20261016.log` and
  `reports/headroom_render_delivery_build_20261016.log`.
- Focused final run: **29/29**, **1.813 s**, zero skips/failures/errors:
  `reports/headroom_render_focused_20261016.{json,log}`.
- Complete native regression: **293/293**, **33.819 s**, zero
  disabled/skipped/failed tests: `reports/headroom_render_native_20261016.{json,log}`.
  Legacy color/master/delivery goldens and existing movie tests still pass.
- Ten new render tests cover units, exposure, soft qualifiers, colored-source
  gamut/qualifier order, mask application/resampling, validation, overflow,
  repeat/seek-order independence, delivery formats, and shoulder behavior.
- All **nine** existing movie profiles exercised at the frame-preparation level:
  five PQ/HLG profiles and four ProRes log variants. This is **not** nine newly
  encoded movie files or an end-to-end export/metadata acceptance result.
- Independent double-precision PQ and HLG expressions check emitted 16-bit
  frame codes; physical luminance checks detect a 203/10,000 unit mismatch.
- Four real lossless FFV1 fixtures (8/10-bit, BT.709/BT.2020) now pass through
  decoder → baseline → +1 EV grade → PQ movie spool and actual half-float EXR
  write/read, in both linear Rec.2020 and ACES AP0 containers. All **19/19**
  and **79/79** tested above-white steps remain distinct at these boundaries.
  The existing integer decoder control still collapses those source steps.
  Test EXRs are retained in unique temporary fixture folders.
- Neutral input code 1019 at 10 bits is approximately **249.743557 nits**
  before grading, **499.487115 nits** after +1 EV. Graded float agreement is
  within 0.0006 nits; half-EXR neutral agreement within 0.21 nits on these fixtures.
- No trained model/competitor comparison, visual improvement, real-time speed,
  or temporal-quality benefit is claimed. Seek-order equality establishes only
  that this renderer does not retain adaptive state.

## Integration still required

1. Persist a versioned opt-in policy without allowing older native/Python
   readers to silently reinterpret it. Keep the UI off until its callers agree.
2. Route paused frames and playback through the same float source contract;
   separate cache identity from legacy PNGs, including source identity, policy
   and decoding settings. Preserve cancellation and bounded memory.
3. Wire preview, scopes/probes and master/movie jobs to this shared render.
   GPU display must not run the old SDR composite on these pixels. Deliberately
   address every unsupported control and temporal setting in the UI/job contract.
4. Write policy, original input tags, actual grade and limitations into EXR/movie
   provenance. Existing sidecars' clipped/invented-pixel claims cannot be reused
   unchanged. Check half-float/log output limits before publication, including
   extreme artist grades; the tests here cover the stated fixtures only.
5. Measure preview/export agreement, actual encoded outputs, GPU parity,
   throughput/memory, temporal quality and calibrated-display appearance before
   product enablement. Current eligibility is still tagged 8/10-bit YUV444;
   common 4:2:0 sources need separate decoder validation.

No training, source-dataset writes, downloads, app restart, packaging, commit or
release occurred. The previous 60-app-test result was not rerun for this isolated
API addition. Existing legal, signing, clean-machine and HDR visual-review gates
remain open; old distribution packages remain stale.
