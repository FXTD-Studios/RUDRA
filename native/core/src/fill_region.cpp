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
float smoothstep01(float x) noexcept {
    x = std::clamp(x, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}
}  // namespace

std::vector<float> ai_fill_v4_weight(const SdrImage& sdr, bool keep_detail) {
    const PlanarBuffer& b = sdr.buffer();
    const int w = sdr.width(), h = sdr.height();
    const std::size_t n = std::size_t(w) * std::size_t(h);
    std::vector<float> weight(n, 0.0f);
    if (n == 0 || b.channels() < 3) return weight;
    std::vector<std::uint8_t> region(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const int c0 = code_of(b.plane(0)[i]), c1 = code_of(b.plane(1)[i]), c2 = code_of(b.plane(2)[i]);
        const int hi = std::max({c0, c1, c2}), lo = std::min({c0, c1, c2});
        if (hi < kAiFillV4TauHigh) continue;
        region[i] = 1;
        // composite.neutral_weight (RERENDER_NEUTRAL = 0.78, 0.88)
        weight[i] = smoothstep01((float(lo) / float(std::max(hi, 1)) - 0.78f) / 0.10f);
    }
    if (!keep_detail) return weight;
    // composite.detail_keep_weight: the box's share in the region (an integral image)
    const int k = std::max(3, int(std::lround(25.0 * double(h) / 1080.0)) | 1), r = k / 2;
    std::vector<std::uint32_t> sum(std::size_t(w + 1) * std::size_t(h + 1), 0);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            sum[std::size_t(y + 1) * std::size_t(w + 1) + std::size_t(x + 1)] =
                region[std::size_t(y) * std::size_t(w) + std::size_t(x)] + sum[std::size_t(y) * std::size_t(w + 1) + std::size_t(x + 1)] +
                sum[std::size_t(y + 1) * std::size_t(w + 1) + std::size_t(x)] - sum[std::size_t(y) * std::size_t(w + 1) + std::size_t(x)];
    auto at = [&](int y, int x) { return sum[std::size_t(y) * std::size_t(w + 1) + std::size_t(x)]; };
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const std::size_t i = std::size_t(y) * std::size_t(w) + std::size_t(x);
            if (!region[i]) continue;
            // OpenCV's blur with BORDER_REPLICATE counts k x k always; clamping the box and dividing by k * k
            // matches it inside the frame and differs only within r pixels of its edge.
            const int y0 = std::max(0, y - r), y1 = std::min(h, y + r + 1), x0 = std::max(0, x - r), x1 = std::min(w, x + r + 1);
            const float cover = float(at(y1, x1) - at(y0, x1) - at(y1, x0) + at(y0, x0)) / float((y1 - y0) * (x1 - x0));
            weight[i] *= smoothstep01((cover - 0.45f) / 0.30f);
        }
    // areas under 0.05 % of the frame keep the master
    std::vector<std::uint8_t> big = region;
    drop_specks(big, w, h, std::max(1, int(5e-4 * double(w) * double(h))));
    for (std::size_t i = 0; i < n; ++i)
        if (!big[i]) weight[i] = 0.0f;
    return weight;
}

AiFillRegion ai_fill_region_v4(const SdrImage& sdr, bool keep_detail) {
    AiFillRegion r;
    r.width = sdr.width();
    r.height = sdr.height();
    const std::size_t n = std::size_t(r.width) * std::size_t(r.height);
    r.label.assign(n, std::uint8_t(AiFillLabel::None));
    if (n == 0) return r;
    const std::vector<float> wt = ai_fill_v4_weight(sdr, keep_detail);
    const PlanarBuffer& b = sdr.buffer();
    std::vector<std::uint8_t> filled(n, 0);
    std::size_t seen = 0, nfilled = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const int hi = code_of(std::max({b.plane(0)[i], b.plane(1)[i], b.plane(2)[i]}));
        if (hi < kAiFillV4TauHigh) continue;
        ++seen;
        if (wt[i] >= 0.5f) {
            r.label[i] = std::uint8_t(AiFillLabel::FilledHigh);
            filled[i] = 1;
            ++nfilled;
        } else {
            r.label[i] = std::uint8_t(AiFillLabel::SeenHigh);
        }
    }
    r.areas = drop_specks(filled, r.width, r.height, 1);
    r.seen_pct = 100.0 * double(seen) / double(n);
    r.filled_pct = 100.0 * double(nfilled) / double(n);
    r.specks_pct = 0.0;
    return r;
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
