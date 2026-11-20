#pragma once
// The clip fill (research notes TASK_CLIPPED_HIGHLIGHTS, 12 Oct 2026): what the
// analytic reconstruction puts where the SDR clipped, instead of the flat
// ceiling inverse_aces_approx leaves there (every code at or above 0.995 is
// the same value, about 1276 nits at corpus EV 0).
//
// Three stages, all from the SDR alone, no model:
//
//   1. Partly clipped pixels (some channels at white, some not). The clipped
//      channel is estimated from the ones that survived and the colour ratios
//      of the unclipped pixels around it, carried in by a smooth fill. Never
//      below the ceiling it clipped at.
//   2. Fully clipped regions, large (sky, a blown window). Luminance is a
//      smooth fill in log2 Y from the region's own edge, and a gentle dome
//      that matches the falloff just outside it, capped at large_max_ev. Hue
//      comes from the edge and runs halfway to white at the deepest point.
//      No texture is made up.
//   3. Fully clipped regions, small (practical lamps, the sun, speculars). The
//      same fill with a peak-and-falloff profile: the peak from the region's
//      size and the slope of the glow around it, at most source_max_nits.
//   4. Near-clip detail (9 Nov 2026). An SDR's shoulder parks
//      a bright sky, a window, an open door at codes 230-254 with its structure
//      flattened to a few codes; a display renders it as a flat wall. The
//      structure that survives (the high-pass of the codes' log luminance,
//      Gaussian sigma detail_sigma_px) is amplified by detail_gain inside a
//      luma ramp (detail_from .. detail_to), the same on every channel (hue
//      held), below a noise floor (detail_floor_stops) left alone. 1.0 is off:
//      bit-identical. 1.5-2.0 is the range to judge.
//
// A ClipOverride sets one region's peak in nits (the artist's click). The
// result reaches the composite as Fields (clip_fill_fields), so the CPU
// composite, the GPU shader, the master and the movie export all take it
// without a line of their own: the fields are an input of all four.

#include <string>
#include <vector>

#include "rudra/core/composite.hpp"
#include "rudra/core/fields.hpp"
#include "rudra/core/image.hpp"

namespace rudra {

// The artist's peak for the clipped region at a point: x and y are the
// frame's normalised position (0..1, left/top), nits the region's peak
// (Rec.709 luma of the HDR, absolute). nits <= 0 is "no override".
struct ClipOverride {
    double x = 0.5, y = 0.5;
    double nits = 0.0;
};

struct ClipFillOptions {
    bool enabled = true;
    // A channel is clipped from soft_code (weight 0) to clip_code (weight 1);
    // the weight feathers every change in, so nothing steps at the edge.
    double soft_code = 0.97;
    double clip_code = 0.995;   // where inverse_aces_approx flattens: 254 and 255 are one value there
    // A fully clipped region covering at least this share of the frame is
    // large (stage 2); smaller ones are sources (stage 3).
    double large_fraction = 0.005;
    double large_max_ev = 1.5;          // a large region's dome, at most
    double source_min_ev = 0.0;         // a source's peak over its edge, at least
    double source_max_nits = 4000.0;    // and at most
    double max_partial_ev = 4.0;        // stage 1's estimate over the ceiling, at most
    // The dome's height is glow_scale x (glow slope x radius / 2): 1 extends
    // the glow's own slope to the centre. 0.5 was chosen on the clip benches'
    // first three scenes (bench/cp8_clip_ev0..2, mean error inside clipped
    // pixels over +0/+1/+2 EV) and holds on the other four; see
    // research notes TASK_CLIPPED_HIGHLIGHTS. Where the SDR barely clipped (+0 EV)
    // any dome overshoots: Strength, or a peak, is the artist's answer there.
    double glow_scale = 0.5;
    bool partial = true;                // stage 1 on
    bool slope_from_clean = false;      // the glow's slope from pixels no channel of which is near white
    std::vector<ClipOverride> overrides;
    // Stage 4: the near-clip detail gain, 1 = off (bit-identical), 1..3 sensible.
    double detail_gain = 1.0;
    double detail_from = 0.86;          // SDR Rec.709 luma (0..1) where the ramp starts
    double detail_to = 0.94;            // and where it reaches 1
    double detail_sigma_px = 6.0;       // the high-pass's Gaussian, in pixels of the frame
    double detail_floor_stops = 0.015;  // |high-pass| below this is noise (about one code at 250), left alone
    double detail_max_stops = 1.0;      // the gain's reach, at most, either way
    // 18 Nov 2026 (1.2.6, user report: a fire JPEG): a lossy 8-bit source.
    // JPEG's 8x8 blocks and 4:2:0 chroma put a few codes of noise on every
    // channel near white; stage 1 reads the surviving channels at full
    // per-pixel resolution, so that noise became a speckle of lifts up to
    // max_partial_ev (white blotches, blue and grey dots, a flat yellow-green
    // where R and G hit the cap together). smooth_px > 0 takes the stage-1
    // estimate and the final gain as a normalised Gaussian over the touched
    // pixels only (nothing spreads into clean pixels, nothing goes below the
    // ceiling). 0 is off: bit-identical. lossy_source_clip_options sets it.
    double smooth_px = 0.0;
    // Stage 1: where two channels clipped together (R and G at 255 in a
    // flame), the brighter one keeps the ratio the neighbours' chroma gives
    // over the other, up to max_partial_ev. Off: they stay at one ceiling, a
    // flat yellow beside the orange around it. Off by default (bit-identical);
    // lossy_source_clip_options turns it on.
    bool hue_order = false;
    // 1.2.7 (multi-role review of 1.2.6, 20 Nov 2026): stage 1 takes the clipped
    // channel's chroma from the clean pixels around it. Beside a dark silhouette
    // those can be a differently lit surface, and a few partly clipped pixels then
    // jump stops above the band they belong to (07V: 24 pixels under an engine
    // nacelle up to 21x white, MaxCLL 4,386 nits while the frame's 99.99th
    // percentile is 182). Each channel's lift is held to at most
    // partial_outlier_stops over the mean lift of the partly clipped pixels within
    // partial_outlier_sigma_px (a normalised Gaussian over those pixels only), so a
    // speck cannot outshine its own band; a lamp or tail light whose own pixels all
    // lift together keeps its lift. 0 is off: bit-identical to 1.2.6.
    double partial_outlier_stops = 1.0;
    double partial_outlier_sigma_px = 8.0;
    // 1.2.7: stage 1 carries the chroma of the clean pixels into the partly
    // clipped ones. A clean pixel more than chroma_from_stops below the clip
    // ceiling (in log2 luma) gives less of its chroma, none from chroma_to_stops:
    // a door or window seen from a dark, differently lit room borrowed the room's
    // colour (09V: the sky in the cargo door came out lavender, B/R 1.39 against
    // 1.09 in the source). A sky's own gradient, a flame's glow, sit within a stop
    // or two of the clip and keep theirs. to <= from is off: bit-identical to 1.2.6.
    double chroma_from_stops = 3.0;
    double chroma_to_stops = 6.0;
};

// The 1.2.6 clip fill exactly (no outlier hold, no near-white hue), for the
// parity tests and anyone who must reproduce a 1.2.6 master.
ClipFillOptions clip_fill_options_1_2_6(ClipFillOptions base = {});

// The options for a lossy 8-bit source (JPEG, WebP, an H.264/HEVC 8-bit
// movie): the gain smoothed over about a JPEG block, stage 1's lift held to
// 1.5 stops (a hue that runs to white with level, fire, a tungsten bulb, makes
// the surviving channels overstate it), and a wider feather so compression
// noise around code 247 does not switch pixels in and out. Everything else
// as `base` (the artist's peaks, detail gain, enabled).
ClipFillOptions lossy_source_clip_options(ClipFillOptions base);

// True for a file a lossy codec wrote (by extension): .jpg .jpeg .jpe .jfif
// .webp .avif .heic .heif, any case.
bool is_lossy_still_path(const std::string& path);

// True for a movie a lossy delivery codec wrote at 8 bits (ffprobe's codec_name and pix_fmt):
// H.264, HEVC, VP8/9, AV1, MPEG-2/4, MJPEG, WMV, Theora in an 8-bit pixel format. A 10-bit
// or wider one, and the mastering codecs (ProRes, DNxHD/HR, CineForm, FFV1, raw), are not:
// their noise near white is under a code. 1.2.6: the fire JPEG's fix, for footage.
bool is_lossy_video(const std::string& codec_name, const std::string& pix_fmt);

// Stage 4 alone: log2 gain, one plane, signed, zero outside the ramp; empty at
// detail_gain 1. Exposed for the tests and the acceptance harness.
PlanarBuffer near_clip_detail(const SdrImage& sdr, const ClipFillOptions& options = {});

// One fully clipped region, as the readout and the override see it.
struct ClipRegion {
    int pixels = 0;
    double fraction = 0.0;      // of the frame
    double radius_px = 0.0;     // deepest point's distance from the edge
    bool large = false;
    double edge_nits = 0.0;     // mean luma at the edge (before the dome)
    double peak_nits = 0.0;     // at the deepest point (after it)
    double glow_ev_per_px = 0.0;
    bool overridden = false;
    int x = 0, y = 0;           // the deepest point
};

struct ClipFill {
    // 3 x h x w: log2 of the filled value over the analytic baseline, per
    // channel. Zero wherever nothing clipped. Empty when nothing did at all.
    PlanarBuffer gain_log2;
    std::vector<ClipRegion> regions;
    double partial_pct = 0.0;   // pixels with some, not all, channels clipped
    double full_pct = 0.0;      // pixels with every channel clipped
    double detail_pct = 0.0;    // pixels stage 4 moved (clip_fill_fields fills it in)
};

// The fill in the analytic baseline's terms (baseline.hpp) at `corpus_ev`.
// `sdr` is code values; out-of-range codes are clamped as the composite does.
ClipFill reconstruct_clipped(const SdrImage& sdr, float corpus_ev, const ClipFillOptions& options = {});

// analytic_fields (fields.hpp) with the fill in them: a residual that takes
// the composite from its baseline to the filled value under the default
// controls (mode All, strength 1; Strength scales it, Off removes it), and a
// highlight mask of 1 wherever it changes anything, so Preserve keeps it.
// Disabled, or nothing clipped: exactly analytic_fields.
Fields clip_fill_fields(const SdrImage& sdr, const ModelConstants& model, const ClipFillOptions& options = {},
                        ClipFill* detail = nullptr);

// Smooth fill (push-pull, then relaxed): every plane's values where
// weight > 0 kept, the rest interpolated from them. At most 8 planes.
// Exposed for the tests.
void fill_missing(std::vector<std::vector<float>>& planes, const std::vector<float>& weight, int height, int width,
                  int relax_iterations = 2);

}  // namespace rudra
