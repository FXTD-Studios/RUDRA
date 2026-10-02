#pragma once
// What RUDRA changed against the analytic baseline, for the Compare readout.
//
// The network moves pixels mostly above diffuse white and in deep shadow, so at
// the default view peak (203 nits) the model and the baseline often clip to the
// same white and a wipe shows nothing. These numbers say how much of the frame
// changed, how much of that the current view peak can show, and a view peak
// that shows it. change_weight() is the per-pixel test the display pass tints
// with (core/view.cpp, render/shaders/display.frag), so the readout and the
// tint agree on what "changed" means.

#include <array>
#include <cstdint>

#include "rudra/core/image.hpp"

namespace rudra {

// A pixel changed when max(R, G, B) moved more than kChangeStops. The floor is
// added to both sides so that noise in near black is not counted as a change.
inline constexpr double kChangeStops = 0.1;
inline constexpr double kChangeFloorNits = 0.05;

// Log-spaced bins over the visibility and peak histograms.
inline constexpr int kCompareBins = 160;
inline constexpr double kCompareLoNits = 0.01, kCompareHiNits = 10000.0;

int compare_bin(double nits) noexcept;            // 0 .. kCompareBins - 1
double compare_bin_upper(int bin) noexcept;       // the bin's upper edge, nits

struct CompareStats {
    double changed_pct = 0.0;   // of all pixels
    double up_stops = 0.0;      // the largest brightening, >= 0
    double down_stops = 0.0;    // the largest darkening, <= 0
    // A view peak that shows the change: the 99.5th percentile of
    // max(model, baseline) over the changed pixels (a bin's upper edge), so one
    // specular does not push the view to 10 000 nits. 0 when nothing changed.
    double fit_nits = 0.0;
    // Percent of all pixels whose change first shows at a view peak in each bin.
    std::array<double, kCompareBins> visible_from{};

    // Percent of all pixels whose change shows when the picture clips at
    // `view_nits` (a bin's resolution, about 0.12 stop).
    double visible_pct(double view_nits) const noexcept;
};

// Over every pixel of the two composites (network units).
CompareStats compare_stats(const NetworkLinearImage& model, const NetworkLinearImage& baseline);

// The display pass's tint weight for one pixel, from max(R, G, B) of each side
// in nits: 0 below 0.05 stop, 1 from 0.15 stop, signed (+ brighter, - darker).
// fp32, evaluated as display.frag evaluates it.
float change_weight(float model_max_nits, float baseline_max_nits) noexcept;

// ---- the invented-pixel map (roadmap 3.5) ---------------------------------
//
// Where the output departs from what the SDR supports. A pixel that moved
// against the corrected baseline (|d| > kChangeStops) is
//   invented       where the SDR has no information there: clipped (its
//                  brightest channel at code 254 or above) or crushed (at 1 or
//                  below), so the value is the network's extrapolation;
//   reinterpreted  where the SDR has detail, so the network read the same
//                  code differently from the baseline's curve.
// Everything else follows the SDR. The viewer's Invented layer draws it
// (view.cpp, display.frag); the master's sidecar carries the shares.
inline constexpr float kSdrClipCode = 254.0f / 255.0f;
inline constexpr float kSdrCrushCode = 1.0f / 255.0f;

enum class Support : std::uint8_t { Follows = 0, Reinterpreted = 1, Invented = 2 };

// From max(R, G, B) of each side in nits and the SDR's max code (0..1).
Support support_of(float model_max_nits, float baseline_max_nits, float sdr_max_code) noexcept;
inline bool sdr_has_no_information(float sdr_max_code) noexcept {
    return sdr_max_code >= kSdrClipCode || sdr_max_code <= kSdrCrushCode;
}

struct SupportStats {
    double invented_pct = 0.0, reinterpreted_pct = 0.0;   // of all pixels
    double sdr_without_detail_pct = 0.0;                  // of all pixels: clipped or crushed in the SDR
};
SupportStats support_stats(const NetworkLinearImage& model, const NetworkLinearImage& baseline, const SdrImage& sdr);

}  // namespace rudra
