#include "rudra/core/range.hpp"

#include "rudra/core/compare.hpp"
#include "rudra/core/measure.hpp"
#include "rudra/core/view.hpp"

#include <algorithm>
#include <cmath>

namespace rudra {
namespace {
constexpr double kP = 10000.0;
constexpr double kWhite = 203.0;
constexpr double kChanged = 0.1;   // stops, core/compare.cpp's "changed"
constexpr double kFloor = 0.05;    // nits, the compare line's floor

// np.percentile's linear method by selection, not a sort: O(n) on a 4K frame.
// Equal to percentile_linear (core/measure.hpp); a test holds them together.
double pct_select(std::vector<double>& v, double q) {
    if (v.empty()) return 0.0;
    const double pos = q / 100.0 * double(v.size() - 1);
    const std::size_t lo = std::size_t(std::floor(pos));
    std::nth_element(v.begin(), v.begin() + std::ptrdiff_t(lo), v.end());
    const double a = v[lo];
    if (lo + 1 >= v.size()) return a;
    const double b = *std::min_element(v.begin() + std::ptrdiff_t(lo) + 1, v.end());
    return a + (b - a) * (pos - double(lo));
}

double range_of(std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double lo = pct_select(v, 0.5), hi = pct_select(v, 99.5);
    return lo > 0.0 && hi > 0.0 ? std::log2(hi / lo) : 0.0;
}
double pct(std::size_t a, std::size_t n) { return n ? 100.0 * double(a) / double(n) : 0.0; }
}  // namespace

RangeStats range_stats(const std::vector<float>& light, const NetworkLinearImage& result,
                       const NetworkLinearImage* baseline) {
    RangeStats s;
    const PlanarBuffer& r = result.buffer();
    const std::size_t n = r.plane_size();
    s.pixels = n;
    if (n == 0) return s;
    const bool have_light = light.size() == n * 4;
    const bool have_base = baseline && baseline->buffer().plane_size() == n;
    std::size_t clamped = 0, clamped_all = 0, ch[3] = {0, 0, 0}, crushed = 0, above_w = 0, above_k = 0, unchanged = 0;
    std::vector<double> sdr_v, hdr_v, in_clamp;
    sdr_v.reserve(n);
    hdr_v.reserve(n);
    double peak = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double m = std::max({double(r.plane(0)[i]), double(r.plane(1)[i]), double(r.plane(2)[i])}) * kP;
        peak = std::max(peak, m);
        if (m > kWhite) ++above_w;
        if (m > 1000.0) ++above_k;
        if (m > 0.0) hdr_v.push_back(m);
        if (!have_light) continue;
        const int flags = int(light[i * 4 + 3]);
        const double sm = std::max({double(light[i * 4]), double(light[i * 4 + 1]), double(light[i * 4 + 2])}) * kP;
        if (sm > 0.0) sdr_v.push_back(sm);
        const int clip = flags & (kFlagClipR | kFlagClipG | kFlagClipB);
        for (int k = 0; k < 3; ++k)
            if (clip & (1 << k)) ++ch[k];
        if (flags & kFlagCrush) ++crushed;
        if (clip == 0) continue;
        ++clamped;
        if (clip == (kFlagClipR | kFlagClipG | kFlagClipB)) ++clamped_all;
        in_clamp.push_back(m > 0.0 ? std::log2(m / kWhite) : -60.0);
        if (have_base) {
            const PlanarBuffer& b = baseline->buffer();
            const double bm = std::max({double(b.plane(0)[i]), double(b.plane(1)[i]), double(b.plane(2)[i])}) * kP;
            if (std::abs(std::log2((m + kFloor) / (bm + kFloor))) <= kChanged) ++unchanged;
        }
    }
    s.clamped_pct = pct(clamped, n);
    s.clamped_all_pct = pct(clamped_all, n);
    for (int k = 0; k < 3; ++k) s.clamped_channel_pct[k] = pct(ch[k], n);
    s.crushed_pct = pct(crushed, n);
    s.sdr_range_stops = range_of(sdr_v);
    s.peak_nits = peak;
    s.peak_stops_over_white = peak > 0.0 ? std::log2(peak / kWhite) : 0.0;
    s.above_white_pct = pct(above_w, n);
    s.above_1000_pct = pct(above_k, n);
    s.hdr_range_stops = range_of(hdr_v);
    s.clamped_pixels = clamped;
    if (!in_clamp.empty()) {
        s.clamped_median_stops = pct_select(in_clamp, 50.0);
        s.clamped_p99_stops = pct_select(in_clamp, 99.0);
        if (have_base) s.clamped_unchanged_pct = pct(unchanged, clamped);
    }
    return s;
}

SdrProbe sdr_probe(const SdrImage& sdr, const std::vector<float>& light, double x, double y) {
    SdrProbe p;
    const int w = sdr.width(), h = sdr.height();
    if (w <= 0 || h <= 0) return p;
    p.x = std::clamp(int(std::floor(x + 0.5)), 0, w - 1);
    p.y = std::clamp(int(std::floor(y + 0.5)), 0, h - 1);
    const std::size_t i = std::size_t(p.y) * std::size_t(w) + std::size_t(p.x);
    float mx = 0.0f;
    for (int k = 0; k < 3; ++k) {
        const float v = sdr.buffer().plane(k)[i];
        p.code[k] = int(std::lround(double(v) * 255.0));
        p.clamped[k] = v >= kSdrClipCode;
        mx = std::max(mx, v);
        if (light.size() == sdr.buffer().plane_size() * 4) p.nits[k] = double(light[i * 4 + std::size_t(k)]) * kP;
    }
    p.crushed = mx <= kSdrCrushCode;
    return p;
}

}  // namespace rudra
