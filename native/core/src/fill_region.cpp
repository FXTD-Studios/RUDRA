#include "rudra/core/fill_region.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace rudra {
namespace {

// 8-connected areas of `on`, smaller than min_pixels removed; the kept areas counted.
int drop_specks(std::vector<std::uint8_t>& on, int w, int h, int min_pixels) {
    const std::size_t n = std::size_t(w) * std::size_t(h);
    std::vector<std::int32_t> comp(n, -1);
    std::vector<std::int32_t> stack;
    std::vector<std::size_t> members;
    int kept = 0;
    for (std::size_t s = 0; s < n; ++s) {
        if (!on[s] || comp[s] >= 0) continue;
        members.clear();
        stack.assign(1, std::int32_t(s));
        comp[s] = 1;
        while (!stack.empty()) {
            const std::size_t p = std::size_t(stack.back());
            stack.pop_back();
            members.push_back(p);
            const int y = int(p / std::size_t(w)), x = int(p % std::size_t(w));
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int yy = y + dy, xx = x + dx;
                    if ((dy == 0 && dx == 0) || yy < 0 || xx < 0 || yy >= h || xx >= w) continue;
                    const std::size_t q = std::size_t(yy) * std::size_t(w) + std::size_t(xx);
                    if (on[q] && comp[q] < 0) {
                        comp[q] = 1;
                        stack.push_back(std::int32_t(q));
                    }
                }
        }
        if (int(members.size()) < min_pixels)
            for (std::size_t p : members) on[p] = 0;
        else
            ++kept;
    }
    return kept;
}

int code_of(float v) { return int(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); }

}  // namespace

int ai_fill_min_core_pixels(int width, int height, double share) {
    return share > 0.0 ? int(share * double(width) * double(height)) : 0;
}

AiFillRegion ai_fill_region(const float* top, int width, int height, int tau_high, int tau_low, double min_share, int blend_high) {
    AiFillRegion r;
    r.width = std::max(0, width);
    r.height = std::max(0, height);
    const std::size_t n = std::size_t(r.width) * std::size_t(r.height);
    r.label.assign(n, std::uint8_t(AiFillLabel::None));
    if (!top || n == 0) return r;
    const bool highs = tau_high <= 255, lows = tau_low >= 0;
    std::vector<std::uint8_t> hi_core(n, 0), lo_core(n, 0);
    std::size_t clipped = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const int c = code_of(top[i]);
        if (highs && c >= tau_high) {
            r.label[i] = std::uint8_t(AiFillLabel::SeenHigh);
            if (c >= blend_high) hi_core[i] = 1, ++clipped;
        } else if (lows && c <= tau_low) {
            r.label[i] = std::uint8_t(AiFillLabel::SeenLow);
            if (c <= 0) lo_core[i] = 1, ++clipped;
        }
    }
    const int min_px = ai_fill_min_core_pixels(r.width, r.height, min_share);
    if (min_px > 1) {
        r.areas = drop_specks(hi_core, r.width, r.height, min_px) + drop_specks(lo_core, r.width, r.height, min_px);
    } else {
        std::vector<std::uint8_t> a = hi_core, b = lo_core;
        r.areas = drop_specks(a, r.width, r.height, 1) + drop_specks(b, r.width, r.height, 1);
    }
    std::size_t seen = 0, filled = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (hi_core[i]) r.label[i] = std::uint8_t(AiFillLabel::FilledHigh);
        else if (lo_core[i]) r.label[i] = std::uint8_t(AiFillLabel::FilledLow);
        seen += r.label[i] != std::uint8_t(AiFillLabel::None);
        filled += hi_core[i] || lo_core[i];
    }
    r.seen_pct = 100.0 * double(seen) / double(n);
    r.filled_pct = 100.0 * double(filled) / double(n);
    r.specks_pct = 100.0 * double(clipped - filled) / double(n);
    return r;
}

AiFillRegion ai_fill_region(const SdrImage& sdr, int tau_high, int tau_low, double min_share, int blend_high) {
    const PlanarBuffer& b = sdr.buffer();
    std::vector<float> top(b.plane_size());
    for (std::size_t i = 0; i < top.size(); ++i) top[i] = std::max({b.plane(0)[i], b.plane(1)[i], b.plane(2)[i]});
    return ai_fill_region(top.data(), sdr.width(), sdr.height(), tau_high, tau_low, min_share, blend_high);
}

namespace {
constexpr float kOrange[3] = {1.0f, 0.55f, 0.0f}, kCyan[3] = {0.0f, 0.85f, 1.0f};   // display colours, as the engine's view
float to_linear(float v) { return std::pow(std::clamp(v, 0.0f, 1.0f), 2.4f); }
}  // namespace

void ai_fill_region_tint_display(PlanarBuffer& d, const AiFillRegion& r) {
    if (d.channels() < 3 || d.width() != r.width || d.height() != r.height) return;
    for (std::size_t i = 0; i < d.plane_size(); ++i) {
        const auto l = AiFillLabel(r.label[i]);
        const bool hi = l == AiFillLabel::SeenHigh || l == AiFillLabel::FilledHigh;
        const bool filled = l == AiFillLabel::FilledHigh || l == AiFillLabel::FilledLow;
        const float* col = hi ? kOrange : kCyan;
        for (int c = 0; c < 3; ++c) {
            float& v = d.plane(c)[i];
            const float dim = v * 0.5f;
            v = l == AiFillLabel::None ? dim : filled ? col[c] : 0.5f * dim + 0.5f * 0.6f * col[c];
        }
    }
}

PlanarBuffer ai_fill_region_picture(const SdrImage& sdr, const AiFillRegion& r) {
    PlanarBuffer d = sdr.buffer();
    ai_fill_region_tint_display(d, r);
    for (float& v : d.span()) v = to_linear(v);
    return d;
}

}  // namespace rudra
