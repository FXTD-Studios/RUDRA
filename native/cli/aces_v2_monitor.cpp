#include "aces_v2_monitor.hpp"

#include <algorithm>
#include <cmath>

#include "rudra/core/aces_v2_trims.hpp"
#include "rudra/core/gamut.hpp"
#include "rudra/core/hdr10.hpp"

namespace rudra {

AcesMonitor aces_monitor_from(const std::string& name, bool* ok) {
    if (ok) *ok = true;
    if (name.empty() || name == "scene") return AcesMonitor::Scene;
    if (name == "sdr100") return AcesMonitor::Sdr100;
    if (name == "hdr1000") return AcesMonitor::Hdr1000;
    if (ok) *ok = false;
    return AcesMonitor::Scene;
}

PlanarBuffer aces_monitor_planes(const PlanarBuffer& acescg, const aces_v2::AcesView& view, AcesMonitor m) {
    if (m == AcesMonitor::Scene) return acescg;
    const PlanarBuffer codes = aces_v2::view_forward(view, acescg);
    const bool sdr = m == AcesMonitor::Sdr100;
    // Display light in nits, in the display's primaries, then to AP1 / 203.
    const Mat3 to_ap1 = rgb_to_rgb_matrix(sdr ? Primaries::Rec709 : Primaries::Rec2020, Primaries::Ap1);
    PlanarBuffer out(3, codes.height(), codes.width());
    const std::size_t n = codes.plane_size();
    for (std::size_t i = 0; i < n; ++i) {
        double nits[3];
        for (int c = 0; c < 3; ++c) {
            const float v = std::clamp(codes.plane(c)[i], 0.0f, 1.0f);
            nits[c] = sdr ? 100.0 * std::pow(double(v), 2.4) : double(pq_eotf(v));   // BT.1886 (Lb = 0) or ST 2084
        }
        for (int c = 0; c < 3; ++c)
            out.plane(c)[i] = static_cast<float>(
                (to_ap1[std::size_t(c)][0] * nits[0] + to_ap1[std::size_t(c)][1] * nits[1] + to_ap1[std::size_t(c)][2] * nits[2]) /
                aces_v2::kAcesDiffuseNits);
    }
    return out;
}

PlanarBuffer aces_area_downscale(const PlanarBuffer& p, int max_side) {
    const int w = p.width(), h = p.height();
    if (max_side <= 0 || std::max(w, h) <= max_side) return p;
    const double s = double(max_side) / double(std::max(w, h));
    const int ow = std::max(1, int(std::lround(w * s))), oh = std::max(1, int(std::lround(h * s)));
    PlanarBuffer out(p.channels(), oh, ow);
    for (int y = 0; y < oh; ++y) {
        const int y0 = int(std::floor(double(y) * h / oh)), y1 = std::max(y0 + 1, int(std::floor(double(y + 1) * h / oh)));
        for (int x = 0; x < ow; ++x) {
            const int x0 = int(std::floor(double(x) * w / ow)), x1 = std::max(x0 + 1, int(std::floor(double(x + 1) * w / ow)));
            for (int c = 0; c < p.channels(); ++c) {
                double sum = 0;
                for (int yy = y0; yy < std::min(y1, h); ++yy)
                    for (int xx = x0; xx < std::min(x1, w); ++xx) sum += p.plane(c)[std::size_t(yy) * std::size_t(w) + std::size_t(xx)];
                out.plane(c)[std::size_t(y) * std::size_t(ow) + std::size_t(x)] =
                    static_cast<float>(sum / double((std::min(y1, h) - y0) * (std::min(x1, w) - x0)));
            }
        }
    }
    return out;
}

PlanarBuffer aces_code_plane(const PlanarBuffer& codes) {
    PlanarBuffer out(1, codes.height(), codes.width());
    for (std::size_t i = 0; i < codes.plane_size(); ++i)
        out.plane(0)[i] = std::max({codes.plane(0)[i], codes.plane(1)[i], codes.plane(2)[i]});
    return out;
}

}  // namespace rudra
