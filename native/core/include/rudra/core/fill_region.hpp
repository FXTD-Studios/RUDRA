#pragma once
// Where AI fill v3 acts on a frame (18 Nov 2026, 1.2.6 user feedback on 09V.mp4), the rule of
// rudra/genfill/masks.py in C++ so the app and the engine's "fill" view show what the fill does:
//
//   seen    max(R, G, B) code >= tau_high (250): the model is shown this area as context
//   filled  max code 255 (fully clipped), in 8-connected areas of at least min_share of the
//           frame (5e-5: about 100 px at 1080p). Only these pixels are changed (blend_core).
//   and, with shadows on (tau_low >= 0), the same for max code <= tau_low / code 0.
//
// A near-white sky of codes 250-254 is seen but not filled: the source still holds its picture
// there and v3 keeps it. Single clipped pixels scattered through it are left alone too (the fill
// landed in those specks only).
#include <cstdint>
#include <vector>

#include "rudra/core/image.hpp"

namespace rudra {

inline constexpr double kAiFillMinCoreShare = 5e-5;   // rudra/genfill/masks.py MIN_CORE_DEFAULT

enum class AiFillLabel : std::uint8_t { None = 0, SeenHigh = 1, FilledHigh = 2, SeenLow = 3, FilledLow = 4 };

struct AiFillRegion {
    int width = 0, height = 0;
    std::vector<std::uint8_t> label;   // AiFillLabel per pixel, image order
    double seen_pct = 0.0;             // share of the frame the model sees (filled included)
    double filled_pct = 0.0;           // share the fill changes
    double specks_pct = 0.0;           // fully clipped (or crushed) pixels left out as specks
    int areas = 0;                     // filled areas kept
    bool empty() const noexcept { return filled_pct <= 0.0; }
};

// The smallest area kept, in pixels (0 = keep all, as masks.py min_core_pixels).
int ai_fill_min_core_pixels(int width, int height, double share = kAiFillMinCoreShare);

// `top`: the SDR's largest code per pixel, 0..1 (code / 255), width x height.
// `blend_high`: the code from which pixels are filled (255 for v3; v4.3 region re-render fills its
// whole region, so tau_high, with min_share 0).
AiFillRegion ai_fill_region(const float* top, int width, int height, int tau_high = 250, int tau_low = -1,
                            double min_share = kAiFillMinCoreShare, int blend_high = 255);
// The same of an SDR image (max of R, G, B).
AiFillRegion ai_fill_region(const SdrImage& sdr, int tau_high = 250, int tau_low = -1, double min_share = kAiFillMinCoreShare,
                            int blend_high = 255);
// AI fill v4.3 region re-render's region: codes >= 240, every pixel of it redrawn (genfill_preview's request).
inline constexpr int kAiFillV4TauHigh = 240;

// The region as a picture for the viewer, linear light where 1.0 is SDR white: the SDR dimmed,
// the filled areas solid orange (shadows cyan), the seen-only areas tinted half way.
PlanarBuffer ai_fill_region_picture(const SdrImage& sdr, const AiFillRegion& region);

// The same tint over a display-referred picture (0..1, e.g. the engine's ACES view), in place.
void ai_fill_region_tint_display(PlanarBuffer& display, const AiFillRegion& region);

}  // namespace rudra
