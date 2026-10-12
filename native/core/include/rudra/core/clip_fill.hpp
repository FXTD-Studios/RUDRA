#pragma once
// The clip fill (claude/TASK_CLIPPED_HIGHLIGHTS.md, 12 Oct 2026): what the
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
//      comes from the edge. No texture is made up.
//   3. Fully clipped regions, small (practical lamps, the sun, speculars). The
//      same fill with a peak-and-falloff profile: the peak from the region's
//      size and the slope of the glow around it, at most source_max_nits.
//
// A ClipOverride sets one region's peak in nits (the artist's click). The
// result reaches the composite as Fields (clip_fill_fields), so the CPU
// composite, the GPU shader, the master and the movie export all take it
// without a line of their own: the fields are an input of all four.

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
    // claude/TASK_CLIPPED_HIGHLIGHTS.md. Where the SDR barely clipped (+0 EV)
    // any dome overshoots: Strength, or a peak, is the artist's answer there.
    double glow_scale = 0.5;
    bool partial = true;                // stage 1 on
    bool slope_from_clean = false;      // the glow's slope from pixels no channel of which is near white
    std::vector<ClipOverride> overrides;
};

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
