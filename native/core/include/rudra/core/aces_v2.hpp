#pragma once
// ACES v2 staged reconstruction (model contract v9 full_context +
// guarded_projected600 release recipe). Executable spec:
// tools/export_native_v2.py::staged_reference. Stages:
//   1. baseline = view.inverse(source)                     (OCIO, outside core)
//   2. G1(source, baseline) -> restored_source, highlight fields, full fields, weight
//   3. restored = view.inverse(restored_source)            (OCIO)
//   4. compose_v2                                          (this file)
//   5. smooth_clipped_hybrid                               (this file)
//   6. constrain_colour_v2 (needs view.forward)            (this file)
//   7. mask = weight on highlight support, 0 elsewhere
// Core stays dependency free: the view and the network are interfaces.
// All buffers are planar float32 (PlanarBuffer), 3 channels unless noted.

#include <cstddef>
#include <functional>
#include <string>

#include "rudra/core/image.hpp"
#include "rudra/platform/result.hpp"

namespace rudra::aces_v2 {

inline constexpr double kClipCode = 254.5 / 255.0;   // "fully clipped" channel threshold
inline constexpr double kLumaAp1[3] = {.2722287168, .6740817658, .0536895174};

// The declared ACES 2.0 SDR100 Rec.709 view (OCIO built-in
// cg-config-v4.0.0_aces-v2.0_ocio-v2.5). Pointwise; in place on packed RGB.
class AcesView {
public:
    virtual ~AcesView() = default;
    virtual void forward(float* rgb, std::size_t pixels) const = 0;   // ACEScg -> display codes
    virtual void inverse(float* rgb, std::size_t pixels) const = 0;   // display codes [0,1] -> ACEScg
    virtual std::string identity() const = 0;                        // "ocio 2.5.2 <config> <view>"
};

// Planar <-> packed helpers and whole-buffer view application.
std::vector<float> to_packed(const PlanarBuffer& rgb);
PlanarBuffer from_packed(const std::vector<float>& packed, int height, int width);
PlanarBuffer view_inverse(const AcesView& view, const PlanarBuffer& codes);
PlanarBuffer view_forward(const AcesView& view, const PlanarBuffer& acescg);

// Policy observed_clip_v1 (training/aces_model.py::reconstruction_weight). 1 channel.
PlanarBuffer reconstruction_weight(const PlanarBuffer& source);

struct G1Output {
    PlanarBuffer restored_source;   // 3ch display codes [0,1]
    PlanarBuffer highlight_fields;  // 3ch signed AP1 residual fields (partial highlights)
    PlanarBuffer full_fields;       // 3ch signed AP1 residual fields (fully clipped)
    PlanarBuffer weight;            // 1ch reconstruction weight
};
// The network stage. Implemented by a runtime (infer/), or a mock in tests.
using G1Fn = std::function<Result<G1Output>(const PlanarBuffer& source, const PlanarBuffer& baseline)>;

struct Composed {
    PlanarBuffer acescg;   // 3ch
    PlanarBuffer mask;     // 1ch: weight on highlight support (w>0 & max src>.5), else 0
};
// Stage 4. restored = view.inverse(g1.restored_source).
Composed compose_v2(const PlanarBuffer& source, const PlanarBuffer& baseline, const PlanarBuffer& restored,
                    const G1Output& g1);

struct SmoothingStats {
    std::size_t full_pixels = 0, interior_pixels = 0, components = 0;
};
// Stage 5, contract "hybrid64": sigma 64, ramp 1, start 0 (aces_clipped_smoothing.py,
// mode='hybrid'). Pixels outside the fully clipped support are returned bit-exact.
PlanarBuffer smooth_clipped_hybrid(const PlanarBuffer& prediction, const PlanarBuffer& source,
                                   double sigma = 64.0, SmoothingStats* stats = nullptr);

struct ConstraintStats {
    std::size_t support_pixels = 0, changed_pixels = 0, baseline_fallback_pixels = 0, remaining_display_errors = 0;
    std::size_t new_negative_luminance = 0;  // support pixels with Y<0 where the baseline has Y>=0
    // 3 Nov 2026: of the pixels still more than 1 code off, those where the plain ACES
    // inverse ITSELF is more than 1 code off (near-white tinted codes such as 253,255,255:
    // ACES 2.0 sends them to white, so no sane scene value displays them exactly). They
    // are the view's limit, not the reconstruction's, and are not in remaining_display_errors.
    std::size_t inverse_limit_pixels = 0;
};
// Stage 6 (aces_colour_backtrack.py::constrain_colour, tolerance 1 code): on highlight
// support, halve toward the baseline until the displayed codes are within 1 code of the
// source (8 halvings, then the baseline itself). Outside support bit-exact.
// forbid_new_negative_y (forbid_new_negative_y=True in Python): a support pixel also keeps
// halving while its AP1 luminance is negative and the baseline's is not.
PlanarBuffer constrain_colour_v2(const PlanarBuffer& source, const PlanarBuffer& prediction,
                                 const PlanarBuffer& baseline, const PlanarBuffer& weight, const AcesView& view,
                                 bool include_fully_clipped = true, ConstraintStats* stats = nullptr,
                                 bool forbid_new_negative_y = false);

// Stage 5b (1 Nov 2026, the user's tau controls): keep the
// learned result only where the source's brightest channel is at or above
// tau_high (highlights) or at or below tau_low (shadows), in 8-bit codes, with
// a 2-code ramp; everywhere between is the exact analytic inverse, which the
// ACES view returns to the source pixel for pixel. Off by default: then the
// frame is bit-exact with the model's own output (which also restores the mid
// band). tau_high 128..256 (256: no highlights), tau_low -1..64 (-1: no shadows).
struct UserBand {
    bool on = false;
    int tau_high = 254;
    int tau_low = 5;
    static constexpr int kRampCodes = 2;
    bool valid() const noexcept { return tau_high >= 128 && tau_high <= 256 && tau_low >= -1 && tau_low <= 64; }
};
// Clean shadows (4 Nov 2026, off by default): an edge-preserving spatial denoise of the
// master inside the shadow band, before the ceiling and the appearance constraint, so
// the SDR view still holds within 1 code. Weight 1 where the source's brightest channel
// is at or below max_code, 0 from max_code + 8. Each pixel there becomes a mean of its
// (2 radius + 1)^2 neighbours weighted by distance (sigma radius / 2) and by the
// difference in log2 luminance (sigma range_stops): flat noise averages out, edges stay.
// No invention: nothing is added that the neighbourhood does not hold.
struct CleanShadows {
    bool on = false;
    int max_code = 40;
    int radius = 3;
    float range_stops = 0.25f;
    bool valid() const noexcept { return max_code >= 1 && max_code <= 128 && radius >= 1 && radius <= 8 && range_stops > 0.0f; }
};
PlanarBuffer clean_shadows(const PlanarBuffer& acescg, const PlanarBuffer& source, const CleanShadows& c,
                           std::size_t* cleaned = nullptr);

struct BandStats {
    std::size_t learned_pixels = 0;   // weight 1: the learned result as it was
    std::size_t ramp_pixels = 0;      // 0 < weight < 1
    std::size_t analytic_pixels = 0;  // weight 0: the exact analytic inverse
};
// The AP1 luminance of the view's inverse of display white (1, 1, 1): RUDRA's highlight
// ceiling (128 in ACEScg for ACES 2.0 SDR 100 nits, about 26,000 nits at 203 per unit).
float inverse_white_luma(const AcesView& view);
// Scale every pixel brighter than `luma_max` (AP1 luminance) down to it, hue kept.
std::size_t cap_luminance(PlanarBuffer& acescg, float luma_max);

// The band's weight for one pixel's brightest channel (a code in 0..1).
float user_band_weight(float top, const UserBand& band) noexcept;
// prediction where the weight is 1, baseline where it is 0 (both bit-exact), a
// linear mix between. 1 channel weight returned in `weight_out` when given.
PlanarBuffer apply_user_band(const PlanarBuffer& prediction, const PlanarBuffer& baseline, const PlanarBuffer& source,
                             const UserBand& band, BandStats* stats = nullptr, PlanarBuffer* weight_out = nullptr);

struct FrameResult {
    PlanarBuffer acescg;   // the master: ACEScg scene-linear, signed, unclamped
    PlanarBuffer mask;     // rudra.reconstructed
    SmoothingStats smoothing;
    ConstraintStats constraint;
    BandStats band;        // all zero when the band is off
    // 3 Nov 2026: pixels brought down to the luminance of the inverse of display white
    // (inverse_white_luma): near-white tinted codes invert up to 7x brighter than white.
    std::size_t ceiling_pixels = 0;
    std::size_t clean_shadow_pixels = 0;   // pixels Clean shadows changed (0 when off)
    // Wall time of each stage, ms (3 Nov 2026): where a frame's time goes.
    struct StageMs {
        double inverse = 0, network = 0, compose = 0, smoothing = 0, band = 0, constraint = 0;
    } ms;
};
// Stages 1-7 for one frame of display codes in [0,1].
Result<FrameResult> run_frame(const PlanarBuffer& source, const AcesView& view, const G1Fn& g1,
                              bool smoothing = true, bool constrain = true, bool forbid_new_negative_y = false,
                              const UserBand& band = {}, const CleanShadows& clean = {});

}  // namespace rudra::aces_v2
