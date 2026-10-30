#include "rudra/core/aces_v2_trims.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace rudra::aces_v2 {
namespace {
// ACEScg (AP1) luminance weights.
constexpr double kY[3] = {0.2722287168, 0.6740817658, 0.0536895174};
}  // namespace

PlanarBuffer apply_trims(const PlanarBuffer& master, const PlanarBuffer& baseline, const PlanarBuffer& mask,
                         const AcesTrims& t) {
    if (t.identity()) return master;
    PlanarBuffer out = master;
    const std::size_t n = master.plane_size();
    const double s = std::clamp(t.strength, 0.0, 1.5), sat = std::clamp(t.saturation, 0.0, 2.0);
    const double cap = t.peak_nits > 0 ? t.peak_nits / kAcesDiffuseNits : 0.0;   // in diffuse-white units
    const double knee = 0.75 * cap;
    const double gain = std::exp2(std::clamp(t.exposure_ev, -3.0, 3.0));
    const double con = std::clamp(t.contrast, 0.5, 1.5);
    for (std::size_t i = 0; i < n; ++i) {
        double rgb[3];
        for (int c = 0; c < 3; ++c) {
            const double b = baseline.plane(c)[i], m = master.plane(c)[i];
            rgb[c] = (b + s * (m - b)) * gain;
        }
        if (con != 1.0) {   // about 18 % grey, on luminance: the hue and the ratios stay
            const double y = kY[0] * rgb[0] + kY[1] * rgb[1] + kY[2] * rgb[2];
            if (y > 0) {
                const double k = 0.18 * std::pow(y / 0.18, con) / y;
                for (double& v : rgb) v *= k;
            }
        }
        const double w = std::clamp(double(mask.plane(0)[i]), 0.0, 1.0);
        if (w > 0) {
            double y = kY[0] * rgb[0] + kY[1] * rgb[1] + kY[2] * rgb[2];
            if (sat != 1.0) {
                double r2[3];
                for (int c = 0; c < 3; ++c) r2[c] = y + sat * (rgb[c] - y);
                for (int c = 0; c < 3; ++c) rgb[c] = rgb[c] + w * (r2[c] - rgb[c]);
                y = kY[0] * rgb[0] + kY[1] * rgb[1] + kY[2] * rgb[2];
            }
            if (cap > 0 && y > knee) {
                const double room = cap - knee;
                const double y2 = knee + room * (1.0 - std::exp(-(y - knee) / room));
                const double k = 1.0 + w * (y2 / y - 1.0);
                for (double& v : rgb) v *= k;
            }
        }
        for (int c = 0; c < 3; ++c) out.plane(c)[i] = static_cast<float>(rgb[c]);
    }
    return out;
}

std::string trims_text(const AcesTrims& t) {
    char b[192];
    if (t.peak_nits > 0)
        std::snprintf(b, sizeof b, "strength %.2f, saturation %.2f, peak %.0f nits", t.strength, t.saturation, t.peak_nits);
    else
        std::snprintf(b, sizeof b, "strength %.2f, saturation %.2f, no peak limit", t.strength, t.saturation);
    std::string out = b;
    if (t.exposure_ev != 0.0) {
        std::snprintf(b, sizeof b, ", exposure %+.2f EV", t.exposure_ev);
        out += b;
    }
    if (t.contrast != 1.0) {
        std::snprintf(b, sizeof b, ", contrast %.2f", t.contrast);
        out += b;
    }
    return out;
}

}  // namespace rudra::aces_v2
