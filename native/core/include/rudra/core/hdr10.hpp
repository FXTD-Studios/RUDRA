#pragma once
// HDR10 and HLG mastering. Port of rudra/hdr10.py (float32, as the numpy is)
// and rudra/delivery/profiles.py (HLG in float64, then float32).

#include <optional>
#include <string_view>
#include <string>
#include <utility>
#include <vector>

#include "rudra/core/image.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

// SMPTE ST 2084, float32: nits -> [0,1] code, and back.
float pq_oetf(float nits) noexcept;
float pq_eotf(float code) noexcept;

// 12-bit PQ code of a luminance, as metadata._pq12 computes it.
int pq12(double nits) noexcept;

// What the delivery does with a pixel whose luminance fits the peak but one
// channel does not (a saturated colour: a clipped blue sky rebuilt to ~1200
// nits of blue at ~150 nits of luminance; validation finding F1, 5 Nov 2026).
//   Scale: the whole pixel scaled down until that channel fits. Hue and
//     saturation kept, luminance lost (0.16 stop at 1000 nits PQ, 0.70 HLG
//     on the F1 sky). The Python's (rudra/hdr10.py), kept for parity.
//   KeepLuminance: the pixel moved toward its own luminance (Rec.2020
//     weights) just far enough that the channel fits. Luminance and hue
//     direction kept, saturation given up, and only on pixels the display
//     cannot show at that peak anyway. Pixels already inside are untouched.
// A pixel whose luminance itself reaches the peak is scaled, as before.
enum class PeakFit { Scale, KeepLuminance };
const char* peak_fit_name(PeakFit fit) noexcept;   // "scale" / "luminance"
std::optional<PeakFit> peak_fit_from_name(std::string_view name) noexcept;

// One pixel, in place: every channel at most `top` (nits, or 1.0 for HLG
// scene light). `lum` is its Rec.2020 luminance. T is float or double.
template <typename T>
inline void fit_to_peak(T px[3], T lum, T top, PeakFit fit) noexcept {
    T cmax = px[0] > px[1] ? px[0] : px[1];
    cmax = cmax > px[2] ? cmax : px[2];
    if (!(cmax > top)) return;
    if (fit == PeakFit::KeepLuminance && lum >= T(0) && lum < top) {
        const T t = (top - lum) / (cmax - lum);   // cmax > top > lum: 0 <= t < 1
        for (int c = 0; c < 3; ++c) px[c] = lum + t * (px[c] - lum);
        return;
    }
    const T s = top / (cmax > T(1e-6) ? cmax : T(1e-6));
    for (int c = 0; c < 3; ++c) px[c] *= s;
}

// Hue-preserving shoulder in absolute nits from the network convention
// (1.0 = 10 000 nits); float32 throughout. 3 x h x w nits out.
Result<PlanarBuffer> master_to_peak(const PlanarBuffer& rgb_normalized, double peak_nits = 1000.0,
                                    std::optional<double> knee_nits = std::nullopt, PeakFit fit = PeakFit::Scale);

// (pq_rgb, mastered_nits)
Result<std::pair<PlanarBuffer, PlanarBuffer>> master_to_pq(const PlanarBuffer& rgb_normalized,
                                                           double peak_nits = 1000.0,
                                                           std::optional<double> knee_nits = std::nullopt);

// BT.2100 HLG, float64.
double hlg_oetf(double scene) noexcept;

// profiles.PROFILES: what ffmpeg is told for each delivery profile.
struct DeliveryProfile {
    std::string name, codec, encoder, transfer, pixel_format;
    int prores_profile = -1;   // -1: not ProRes
    std::string tag;
    std::string log;           // "acescct" / "logc4": scene-referred log, no peak or knee; empty: display-referred
};
// The Python's five, in its order (the parity tests hold this list to profiles.py).
const std::vector<DeliveryProfile>& delivery_profiles();
// Native only (9 Oct 2026): ProRes 422 HQ and 4444 in ACEScct (AP1) and ARRI
// LogC4 (AWG4). Tagged unknown primaries and transfer: there are no codes for them.
const std::vector<DeliveryProfile>& log_delivery_profiles();
// Either list, by name; nullptr when neither has it.
const DeliveryProfile* find_delivery_profile(std::string_view name);

// profiles.encode_master: (code values in [0,1], display light in nits), float32.
Result<std::pair<PlanarBuffer, PlanarBuffer>> encode_master(const PlanarBuffer& rgb_normalized,
                                                            const std::string& profile, double peak_nits = 1000.0,
                                                            std::optional<double> knee_nits = std::nullopt);

}  // namespace rudra
