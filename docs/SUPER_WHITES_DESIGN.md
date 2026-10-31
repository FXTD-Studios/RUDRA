# Super-whites: design note (roadmap 1.2, audit P1-2)

**Superseded implementation proposal:** read
[the measured preservation review](SUPER_WHITES_REVIEW_2026-10-16.md).
The proposed limited-integer RGB decoding fails on pinned FFmpeg 7.1.1;
the native endpoint slope also flattens. Keep this note as historical context,
not an implementation prescription. Production preservation remains unimplemented.

16 October 2026. Design only. No code has changed, and 1.0 ships with the
limit described below.

## What happens today

Limited-range (TV) video stores white at code 235 and black at code 16
(8-bit; 940 and 64 at 10 bits). Many cameras and broadcast masters also write
real picture detail above white, at codes 236–254 ("super-whites"), and
below black, at codes 1–15 ("footroom").

RUDRA loses that detail before reconstruction:

1. `media/src/video_probe.cpp` `decoder_filter` decodes to full-range 16-bit
   (`zscale … rangein=limited:range=full`, `gbrp16le`). Code 235 becomes
   65535, and everything above it is clipped there. Footroom is clipped at 0.
2. `core/src/video_predict.cpp` `canonicalize_sdr` clamps to [0, 1] again
   (`predictor.cpp` calls it with `"full"`, because the decoder already
   converted).
3. So the clip detector sees code 255 where the source held measured detail,
   and the clip fill then invents highlights that the source actually
   recorded.

Stills are not affected: PNG, JPEG and TIFF are full range.

## Proposal

1. **Decode with headroom.** Use `zscale … rangein=limited:range=limited`
   (no range expansion) with a 16-bit output, then expand in RUDRA:
   `x = (code − 16·k) / (219·k)` in float. This keeps x > 1 for super-whites
   and x < 0 for footroom. The alternative is `gbrpf32le` with
   `range=full:npl=`; the integer path keeps the decoder bit-exact with
   today's for codes 16–235.
2. **Carry above-white through.**
   - `SdrImage` holds `[0, 1 + h]`, where `h` is the measured headroom: at
     most (254 − 235) / 219 ≈ 0.087.
   - `canonicalize_sdr` stops clamping above 1 for limited-range sources.
     Footroom below 0 is clamped, since black detail under 16 is noise far
     more often than signal; this needs confirming.
3. **Extend the analytic inverse above 1.0.** This is the real design
   decision.
   - **Option A, recommended:** continue each Source curve's inverse past
     display 1.0 along its own slope in log2 at code 235. This is
     monotonic, adds no new parameter, and is defined for every picker
     entry.
   - **Option B:** BT.1886 extended (code^2.4 past 1.0). Exact for BT.1886,
     but undefined for ACES, filmic and AgX.
4. **Mark it as measured, not clipped.** The clip mask (`max code ≥ 254`)
   counts only true 255 codes. Super-white pixels are excluded from the clip
   fill and the "invented" map, and the master sidecar reports
   `super_white_pct`.
5. **Gates.**
   - Full-range sources and limited-range sources with no code above 235 are
     bit-for-bit unchanged, against all existing goldens.
   - A synthetic 10-bit limited ramp 64→1019 gives distinct, monotonic
     master values above the 940 level.
   - The clip benches are no worse.
   - Flicker is measured on a real clip that has super-whites.

## Effort and risk

Large:
- the decoder, the canonicaliser, every Source curve inverse (native and
  Python twin), the clip detector, the composite shader (`composite.frag`
  reads the code in [0, 1]), the GPU parity cases, and the goldens;
- the Python Studio, if parity with it is to be kept.

Risk: a wrong inverse extension brightens highlights that the source
graded as white.

## Until then (1.0)

Review update, 16 October: the proposal above is not yet validated. Movie
preview and playback use additional decoding paths; see
[the source inspection review](SOURCE_INSPECTION_2026-10-16.md). The earlier
full-range workaround below is not a guarantee: range conversion can itself
clip headroom, and changing tags alone does not preserve measured detail.

Documented as a known limit in USER_GUIDE and BETA: limited-range video
detail above code 235 is clipped at decode, and the clip fill reconstructs
it. For footage where it matters, convert to full range with headroom
before RUDRA (for example in Resolve, with data levels set to full and the
super-whites kept).
