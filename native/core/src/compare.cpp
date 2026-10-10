#include "rudra/core/compare.hpp"

#include <algorithm>
#include <cmath>

namespace rudra {
namespace {

constexpr double kPeak = 10000.0;   // network units to nits
const double kLogLo = std::log2(kCompareLoNits);
const double kLogSpan = std::log2(kCompareHiNits) - std::log2(kCompareLoNits);

}  // namespace

int compare_bin(double nits) noexcept {
    if (!(nits > kCompareLoNits)) return 0;
    const double t = (std::log2(nits) - kLogLo) / kLogSpan;
    return std::clamp(int(std::floor(t * kCompareBins)), 0, kCompareBins - 1);
}

double compare_bin_upper(int bin) noexcept {
    return std::exp2(kLogLo + kLogSpan * double(bin + 1) / kCompareBins);
}

double CompareStats::visible_pct(double view_nits) const noexcept {
    // Bins whose threshold is at or below the view peak; the view's own bin
    // counts, so the answer can be high by at most one bin.
    const int last = compare_bin(view_nits);
    double pct = 0.0;
    for (int i = 0; i <= last; ++i) pct += visible_from[std::size_t(i)];
    return std::min(pct, changed_pct);
}

CompareStats compare_stats(const NetworkLinearImage& model, const NetworkLinearImage& baseline) {
    CompareStats s;
    const PlanarBuffer& m = model.buffer();
    const PlanarBuffer& b = baseline.buffer();
    const std::size_t n = m.plane_size();
    if (n == 0 || b.plane_size() != n) return s;
    const double f = kChangeFloorNits;
    const double ratio = std::exp2(kChangeStops);
    std::array<std::size_t, kCompareBins> from{}, peaks{};
    std::size_t changed = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double mm = kPeak * std::max({double(m.plane(0)[i]), double(m.plane(1)[i]), double(m.plane(2)[i]), 0.0});
        const double bb = kPeak * std::max({double(b.plane(0)[i]), double(b.plane(1)[i]), double(b.plane(2)[i]), 0.0});
        if (!std::isfinite(mm) || !std::isfinite(bb)) continue;
        const double d = std::log2((mm + f) / (bb + f));
        if (std::abs(d) <= kChangeStops) continue;
        ++changed;
        s.up_stops = std::max(s.up_stops, d);
        s.down_stops = std::min(s.down_stops, d);
        // Clipped at a view peak V the two sides are min(., V): equal while V is
        // at or below the darker side, and the change shows once V clears it by
        // kChangeStops.
        const double lo = std::min(mm, bb), hi = std::max(mm, bb);
        ++from[std::size_t(compare_bin(ratio * (lo + f) - f))];
        ++peaks[std::size_t(compare_bin(hi))];
    }
    if (changed == 0) return s;
    const double total = double(n);
    s.changed_pct = 100.0 * double(changed) / total;
    for (int i = 0; i < kCompareBins; ++i) s.visible_from[std::size_t(i)] = 100.0 * double(from[std::size_t(i)]) / total;
    const double want = 0.995 * double(changed);
    double run = 0.0;
    for (int i = 0; i < kCompareBins; ++i) {
        run += double(peaks[std::size_t(i)]);
        if (run >= want) {
            s.fit_nits = compare_bin_upper(i);
            break;
        }
    }
    return s;
}

float change_weight(float model_max_nits, float baseline_max_nits) noexcept {
    const float f = float(kChangeFloorNits);
    const float d = std::log2((model_max_nits + f) / (baseline_max_nits + f));
    const float w = std::clamp((std::abs(d) - 0.05f) / 0.1f, 0.0f, 1.0f);
    return d < 0.0f ? -w : w;
}

Support support_of(float model_max_nits, float baseline_max_nits, float sdr_max_code) noexcept {
    const double f = kChangeFloorNits;
    const double d = std::log2((double(model_max_nits) + f) / (double(baseline_max_nits) + f));
    if (!(std::abs(d) > kChangeStops)) return Support::Follows;
    return sdr_has_no_information(sdr_max_code) ? Support::Invented : Support::Reinterpreted;
}

SupportStats support_stats(const NetworkLinearImage& model, const NetworkLinearImage& baseline, const SdrImage& sdr) {
    SupportStats s;
    const PlanarBuffer& m = model.buffer();
    const PlanarBuffer& b = baseline.buffer();
    const PlanarBuffer& c = sdr.buffer();
    const std::size_t n = m.plane_size();
    if (n == 0 || b.plane_size() != n || c.plane_size() != n) return s;
    std::size_t invented = 0, reinterpreted = 0, nothing = 0;
    auto max3 = [](const PlanarBuffer& p, std::size_t i) {
        return std::max(std::max(p.plane(0)[i], p.plane(1)[i]), p.plane(2)[i]);
    };
    for (std::size_t i = 0; i < n; ++i) {
        const float code = max3(c, i);
        nothing += sdr_has_no_information(code);
        const float mm = float(kPeak) * std::max(max3(m, i), 0.0f), bb = float(kPeak) * std::max(max3(b, i), 0.0f);
        if (!std::isfinite(mm) || !std::isfinite(bb)) continue;
        switch (support_of(mm, bb, code)) {
            case Support::Invented: ++invented; break;
            case Support::Reinterpreted: ++reinterpreted; break;
            case Support::Follows: break;
        }
    }
    const double total = double(n);
    s.invented_pct = 100.0 * double(invented) / total;
    s.reinterpreted_pct = 100.0 * double(reinterpreted) / total;
    s.sdr_without_detail_pct = 100.0 * double(nothing) / total;
    return s;
}

}  // namespace rudra
