#pragma once
// The viewer's range readout and SDR probe (docs/view.spec.md section 14,
// revision 6, 1 Nov 2026): where the source footage is clamped, how far the
// result goes above the SDR white, and what the result did inside the clamped
// area. Every number here is in absolute nits and ignores the viewer's
// exposure and gamma.
#include <optional>

#include "rudra/core/image.hpp"

namespace rudra {

struct RangeStats {
    std::size_t pixels = 0;
    // The source, from its codes (core/compare.hpp kSdrClipCode, kSdrCrushCode).
    double clamped_pct = 0.0;        // max(R, G, B) at code 254 or above (the Invented layer's "clipped")
    double clamped_all_pct = 0.0;    // all three channels at 254 or above
    double clamped_channel_pct[3] = {0.0, 0.0, 0.0};   // R, G, B each at 254 or above
    double crushed_pct = 0.0;        // max(R, G, B) at code 1 or below
    double sdr_range_stops = 0.0;    // log2(p99.5 / p0.5) of max(R, G, B) light, zeros left out; 0 when empty
    // The result (any composite or master in network units).
    double peak_nits = 0.0;          // max of max(R, G, B)
    double peak_stops_over_white = 0.0;   // log2(peak / 203); -inf-safe: 0 for a black frame
    double above_white_pct = 0.0;    // max(R, G, B) above 203 nits
    double above_1000_pct = 0.0;     // above 1 000 nits
    double hdr_range_stops = 0.0;    // as sdr_range_stops, on the result
    // Inside the clamped area (clamped_pct's pixels). Empty when nothing is clamped.
    std::size_t clamped_pixels = 0;
    double clamped_median_stops = 0.0;   // stops of max(R, G, B) over 203 nits, linear-method percentiles
    double clamped_p99_stops = 0.0;
    // Of the clamped pixels, the share the result left within 0.1 stop of the
    // plain inverse (the compare line's "changed" threshold, core/compare.hpp):
    // what was not reconstructed. Absent without a baseline.
    std::optional<double> clamped_unchanged_pct;
};

// `light`: sdr_light_rgba of the frame (core/view.hpp), the source as light
// and its clamp flags. `result` and `baseline` in network units, the frame's
// size; `baseline` may be null (the learned ACES route has none).
RangeStats range_stats(const std::vector<float>& light, const NetworkLinearImage& result,
                       const NetworkLinearImage* baseline);

// The source pixel under the pointer: its 8-bit codes and which are clamped.
struct SdrProbe {
    int x = 0, y = 0;
    int code[3] = {0, 0, 0};          // round(255 v)
    bool clamped[3] = {false, false, false};
    bool crushed = false;
    double nits[3] = {0.0, 0.0, 0.0};   // the light the Sdr source shows
};
// Round half up and clamped to the frame, as probe_pixel.
SdrProbe sdr_probe(const SdrImage& sdr, const std::vector<float>& light, double x, double y);

}  // namespace rudra
