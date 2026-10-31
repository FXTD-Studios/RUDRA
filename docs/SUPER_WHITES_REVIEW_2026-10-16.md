# Super-white preservation review

16 October 2026. Review and bounded decoder experiment only; production pixel
processing is unchanged. This supersedes the implementation recommendations
in SUPER_WHITES_DESIGN.md, not the historical evidence of the current limit.

## Decision

Proceed with a **disabled-by-default floating-point decoder prototype**, then
a versioned analytic BT.1886 preservation path. Do not ship the earlier
limited-integer RGB proposal or globally remove clamps. Production enablement
requires the gates below. No new training is needed for this work.

## Measured evidence

Reproducer: `tools/superwhite_decode_review.py`. Final report:
`reports/superwhite_review_20261016_final/review.json` (commands, FFmpeg and
script SHA-256, exact samples and checks). Uses the pinned FFmpeg 7.1.1.

Four synthetic cases: 8/10-bit YUV444, each with BT.709 and BT.2020 nonconstant
luminance matrices. Neutral ramps and colored patches are compared with an
independent double-precision YCbCr-to-RGB matrix calculation. Complete ramps
also exercise reserved edge codes; that does not make those codes suitable
for broadcast delivery. Above-white counts below stop at 254 / 1019.

| Decoding path | Distinct R levels: 8-bit 236..254 | Distinct R levels: 10-bit 941..1019 |
| --- | ---: | ---: |
| Current export, full-range 16-bit RGB | 1 of 19 | 1 of 79 |
| Earlier proposal, limited-range 16-bit GBR | 1 of 19 | 1 of 79 |
| Candidate, full-range float GBR | 19 of 19 | 79 of 79 |
| Current playback conversion | 1 of 19 | 4 of 79 |
| Paused PNG conversion | 1 of 19 | 4 of 79 |

Both matrices give these counts. Float error versus the independent matrix is
at most **1.25e-7** over these cases; the preset gate is 2e-6. Code 1019 maps
to approximately **1.090183**, while current export maps it to 1.0. In-range
Y/Cb/Cr colored patches reach RGB **1.689558**: a universal 1.087 RGB ceiling
would also destroy recorded color excursions. This is signal preservation,
not proof of recovering the original scene's brightness or texture.

The limited-integer proposal fails concretely: this command outputs integer
RGB endpoints 0/65535 despite the requested limited output. Expanding them as
4096..60160 produces false black/white offsets and cannot restore lost levels.
Do not infer the actual RGB encoding from a requested range flag alone.

Float, requantized to uint16, matches current export at all tested nominal-range
neutral samples. This is narrow evidence, **not** bit parity for colored,
subsampled or compressed material. The current playback's nominal white is
approximately 0.996155 versus export's 1.0 on these fixtures. PNG/playback
also differ for the 8-bit fixture. Shared import behavior needs its own gate.

**20/20 final diagnostic checks pass.** These checks deliberately confirm
legacy/proposed-integer loss as well as float preservation. The first run
(`superwhite_review_20261016_a`) is retained: it incorrectly expected every
10-bit PNG headroom value to collapse to one. Measured 941..943 remain distinct;
the corrected check tests loss of levels, with the exact counts still reported.
Intermediate run `_b` is retained; `_final` matches the saved script hash.

No native app suite was rerun for this review-only script/doc change. Previous
60 app / 264 native results remain historical evidence, not new headroom tests.
No real-footage quality, GPU performance or temporal improvement is claimed.

## Architecture findings

| Stage | Current behavior / required change |
| --- | --- |
| Export decode | `media/src/video_probe.cpp:decoder_filter` plus `video_decode.cpp` produce packed integer RGB; add an explicit float path with correctly reordered GBR planes. Keep legacy path intact. |
| Paused preview / still master from movie | `media/src/sequence.cpp` writes a PNG; `main_window.cpp` reads it as a still. Cannot carry extended values. Use shared movie decoding, not a float value hidden in a PNG. |
| Playback | `app/playback.cpp` uses FFmpeg RGB48 conversion, not the export zscale chain. Route a new mode through the same input contract/decoder as export. |
| Input meaning | Preview takes decoded RGB directly; `video/predictor.cpp:prepare` canonicalizes transfer and converts primaries. New mode must specify these consistently, rather than copying either path blindly. |
| Canonicalization / baseline | `core/src/video_predict.cpp`, `baseline.cpp`, `source_curve.cpp` clamp at 1 and/or .995; the 256-knot corrections also clamp their coordinate. Carrying >1 alone has no effect. |
| Composite / GPU | `core/src/composite.cpp` and `render/shaders/composite.frag` clamp source channels. GPU input textures already support float; shader semantics and parity still need work. |
| Fill / invented map | Clip fill feathers from .97 to .995 and creates residual fields. It would treat preserved headroom as missing signal. It needs explicit protection or must be unavailable in the first prototype. |
| Grade / reference / view | Anchoring, calibration, reference curves and source comparison also contain bounded-code assumptions; inspect these before integration. Avoid a new path that only affects final export. |
| Python parity | `rudra/video.py`, `sdr2hdr.py`, `source_baseline.py`, `ui/sequence.py` have corresponding assumptions. Either implement matching opt-in semantics or explicitly reject the new policy there. |
| Cache / projects | Current movie cache token hashes only its resolved path. New mode needs a separate policy-version/input-contract/source-identity key; old cached PNGs must never satisfy float requests. Persist policy in projects and provenance. |

## Reconstruction policy

1. Preserve signed, encoded float RGB and source metadata at decode. Keep
   source primaries/transfer distinct from the artist's reconstruction curve.
   Do not introduce >1 tensors into the existing SdrImage/model contract by
   silently widening its meaning. Prefer an explicit extended-video frame
   type/conversion; keep the existing architecture and legacy types.
2. Initial eligibility: verified limited-range BT.709-transfer video, known
   supported matrix/primaries, analytic **BT.1886** selected. Start validation
   with the tested 8/10-bit YUV444 cases; add 420/422 and compressed formats
   before advertising ordinary movie support. Unsupported combinations get a
   clear refusal, not silent conversion or fallback.
3. First integrated prototype requires AI, automatic clip fill, calibration
   and reference fitting off. Do not silently change saved user settings.
   Retain grading/delivery controls only after verifying their input contract.
   BT.1886-shadows, ACES, Hable, AgX, camera-log and plain-clip extensions are
   separate follow-ups; their extrapolations are not interchangeable.
4. Candidate baseline: in the source RGB primaries, apply the zero-black
   display EOTF `L = 203 * max(E, 0)^2.4`, where E is normalized encoded RGB;
   extend the power function above 1. Convert resulting linear light to the
   working/output primaries. **203 nits and the above-white use are RUDRA policy**;
   they do not recover unknown camera exposure. Preserve signed source samples
   for diagnostics; negative rendered light is excluded by this initial EOTF,
   not because all footroom has been proved to be noise.
5. The existing native BT.1886 inverse clamps display light at .995, giving
   about **201.796 nits** at nominal white before other grade operations. Its
   implemented endpoint slope is zero. The earlier proposal to extrapolate
   that slope therefore preserves no headroom. A 203-nit power extension must
   be a versioned policy with a documented near-white difference; do not
   promise exact legacy equality and exact power-law white simultaneously.
   The candidate gives about **249.744 nits at neutral code 1019** before grade.
   These are equation evaluations, not measured rendered masters.
6. A high source sample is recorded signal, but may already contain sensor,
   grading or encoding clipping. Neither code 255 nor RGB >1 proves true scene
   saturation. Preserve measured excursions; do not automatically label them
   reconstructed/invented. Later fill protection needs per-channel provenance
   and a separate uncertainty policy, including clipping at nominal white.
7. Report source-luma above-white share separately from RGB channel excursions.
   Compute source statistics before resizing/gamut conversion, define alpha
   handling and denominator, and record unknown range/depth explicitly. A UI
   warning from tags remains a warning, never a measured detection result.

## Acceptance gates and next implementation

1. Add isolated float decoder API and tests first. Verify plane order, size,
   finite values, expected negative/above-white samples, EOF/cancel/error paths
   and alpha policy. No public default changes at this stage.
2. Add a versioned opt-in baseline with the restrictions above. Test exact
   black/white, continuity, monotonicity and source-primary hue behavior.
3. Integrate paused/playback/master/movie export together. Check matching
   full-resolution frames; preview downsampling differences must be explicit.
   Change cache namespace; test project reopen and provenance.
4. Disabled mode must retain existing pixel goldens and all app/native tests.
   Enabled mode must preserve tested headroom, with declared numerical budgets,
   distinct 10-bit ramp levels and no false invented overlay. Establish mixed
   ordinary/highlight neighborhood behavior before enabling automatic fill.
5. Add real encoded 420/422, range/tag errors, full-range controls, alpha,
   rotations, scene cuts, near-white noise and moving/specular highlights.
   Temporal holds must not flatten changing recorded highlights. Compare both
   analytic modes; do not substitute a decoder ramp for visual validation.
6. Run four-backend CPU/GPU numerical and latency gates, output QC, reference
   screenshots and calibrated HDR review before release enablement.

Cost estimate: raw float RGB at 4K is **94.9 MiB/frame**, versus 47.5 MiB for
RGB48; at 60 fps the raw float pipe alone carries **5.97 GB/s**. These are byte
counts, not measured speed. Existing SdrImage frame buffers are already float,
so this does not imply every cache doubles. Avoid an additional full-size copy
where ownership can be transferred; benchmark playback and export before
claiming real-time support. No dataset or model download is required.

## External references

- [FFmpeg zscale documentation](https://ffmpeg.org/ffmpeg-filters.html#zscale)
  describes input/output range and matrix options. Actual pinned-build behavior
  above comes from the local experiment, not an assumption about those flags.
- [ITU-R BT.1886](https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.1886-0-201103-I!!PDF-E.pdf),
  Annex 1, defines the reference EOTF and 10-bit normalization `(D-64)/876`.
  RUDRA's versioning, eligibility and 203-nit extension are engineering choices.

No production processing changes, source-dataset writes, training, app restart,
package build, commit or release occurred during this review.
