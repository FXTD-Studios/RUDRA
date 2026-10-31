// ACES v2 staged reconstruction. See aces_v2.hpp. Arithmetic order and
// precision follow the Python reference so CPU results match it closely:
// float32 where the reference is torch/numpy float32, double where it is
// numpy float64 (the clipped smoothing).

#include "rudra/core/aces_v2.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace rudra::aces_v2 {
namespace {

// cv2.BORDER_REFLECT_101 index (gfedcb|abcdefgh|gfedcba).
inline int reflect101(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * n - 2 - i;
    }
    return i;
}

// cv2.getGaussianKernel(ksize, sigma, CV_64F) for ksize > 7.
std::vector<double> gaussian_kernel(int ksize, double sigma) {
    std::vector<double> k(static_cast<std::size_t>(ksize));
    const double scale2x = -0.5 / (sigma * sigma);
    double sum = 0.0;
    for (int i = 0; i < ksize; ++i) {
        const double x = i - (ksize - 1) * 0.5;
        k[i] = std::exp(scale2x * x * x);
        sum += k[i];
    }
    for (auto& v : k) v *= 1.0 / sum;
    return k;
}

// cv2.erode with a 3x3 cross, border constant 0, applied `iterations` times.
std::vector<std::uint8_t> erode_cross(const std::vector<std::uint8_t>& m, int h, int w, int iterations) {
    std::vector<std::uint8_t> cur = m, nxt(m.size());
    for (int it = 0; it < iterations; ++it) {
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const std::size_t i = static_cast<std::size_t>(y) * w + x;
                const bool v = cur[i] && y > 0 && cur[i - w] && y + 1 < h && cur[i + w] && x > 0 && cur[i - 1] &&
                               x + 1 < w && cur[i + 1];
                nxt[i] = v ? 1 : 0;
            }
        std::swap(cur, nxt);
    }
    return cur;
}

// 8-connected labelling; labels 1..n-1 in raster order of first pixel, 0 = background.
// (cv2.connectedComponents numbers components in raster order too; only the grouping
// matters for the per-component means.)
int label8(const std::vector<std::uint8_t>& m, int h, int w, std::vector<int>& labels) {
    labels.assign(m.size(), 0);
    int next = 1;
    std::vector<std::size_t> stack;
    for (std::size_t s = 0; s < m.size(); ++s) {
        if (!m[s] || labels[s]) continue;
        labels[s] = next;
        stack.push_back(s);
        while (!stack.empty()) {
            const std::size_t p = stack.back();
            stack.pop_back();
            const int py = static_cast<int>(p / w), px = static_cast<int>(p % w);
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int y = py + dy, x = px + dx;
                    if ((dy == 0 && dx == 0) || y < 0 || y >= h || x < 0 || x >= w) continue;
                    const std::size_t q = static_cast<std::size_t>(y) * w + x;
                    if (m[q] && !labels[q]) {
                        labels[q] = next;
                        stack.push_back(q);
                    }
                }
        }
        ++next;
    }
    return next;   // number of labels including background
}

inline float smoothstep_band(float top, double b0, double b1) {
    float t = (top - static_cast<float>(b0)) / static_cast<float>(b1 - b0);
    t = std::clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

}  // namespace

std::vector<float> to_packed(const PlanarBuffer& rgb) {
    const std::size_t n = rgb.plane_size();
    std::vector<float> out(n * 3);
    for (int c = 0; c < 3; ++c) {
        const float* p = rgb.plane(c);
        for (std::size_t i = 0; i < n; ++i) out[i * 3 + c] = p[i];
    }
    return out;
}

PlanarBuffer from_packed(const std::vector<float>& packed, int height, int width) {
    PlanarBuffer out(3, height, width);
    const std::size_t n = out.plane_size();
    for (int c = 0; c < 3; ++c) {
        float* p = out.plane(c);
        for (std::size_t i = 0; i < n; ++i) p[i] = packed[i * 3 + c];
    }
    return out;
}

PlanarBuffer view_inverse(const AcesView& view, const PlanarBuffer& codes) {
    auto packed = to_packed(codes);
    view.inverse(packed.data(), codes.plane_size());
    return from_packed(packed, codes.height(), codes.width());
}

PlanarBuffer view_forward(const AcesView& view, const PlanarBuffer& acescg) {
    auto packed = to_packed(acescg);
    view.forward(packed.data(), acescg.plane_size());
    return from_packed(packed, acescg.height(), acescg.width());
}

PlanarBuffer reconstruction_weight(const PlanarBuffer& source) {
    PlanarBuffer w(1, source.height(), source.width());
    const std::size_t n = source.plane_size();
    const float *r = source.plane(0), *g = source.plane(1), *b = source.plane(2);
    float* o = w.plane(0);
    for (std::size_t i = 0; i < n; ++i) {
        const float top = std::max(std::max(r[i], g[i]), b[i]);
        const float hi = smoothstep_band(top, 254.0 / 255.0, 1.0);
        const float lo = smoothstep_band(top, 1.0 / 255.0, 5.0 / 255.0);
        const float evidence = std::max(hi, 1.0f - lo);
        o[i] = std::clamp((evidence - 0.5f) * 2.0f, 0.0f, 1.0f);
    }
    return w;
}

Composed compose_v2(const PlanarBuffer& source, const PlanarBuffer& baseline, const PlanarBuffer& restored,
                    const G1Output& g1) {
    const int h = source.height(), wd = source.width();
    const std::size_t n = source.plane_size();
    Composed out{PlanarBuffer(3, h, wd), PlanarBuffer(1, h, wd)};
    const float* w = g1.weight.plane(0);
    const float clip = static_cast<float>(kClipCode);
    for (std::size_t i = 0; i < n; ++i) {
        const float s0 = source.plane(0)[i], s1 = source.plane(1)[i], s2 = source.plane(2)[i];
        const float top = std::max(std::max(s0, s1), s2), bottom = std::min(std::min(s0, s1), s2);
        const float wi = w[i];
        const bool high = wi > 0 && top > 0.5f;
        const bool full = wi > 0 && bottom >= clip;
        const bool shadow = wi > 0 && top <= 0.5f;
        const float scale = std::max(std::max(std::fabs(restored.plane(0)[i]), std::fabs(restored.plane(1)[i])),
                                     std::fabs(restored.plane(2)[i])) + 0.001f;
        const PlanarBuffer& fields = full ? g1.full_fields : g1.highlight_fields;
        for (int c = 0; c < 3; ++c) {
            const float rs = restored.plane(c)[i];
            float v = rs;
            if (shadow) {
                const float bl = baseline.plane(c)[i];
                v = wi >= 1.0f ? bl : wi * bl + (1.0f - wi) * rs;
            } else if (high) {
                v = rs + wi * (scale * fields.plane(c)[i]);
            }
            out.acescg.plane(c)[i] = v;
        }
        out.mask.plane(0)[i] = high ? wi : 0.0f;
    }
    return out;
}

PlanarBuffer smooth_clipped_hybrid(const PlanarBuffer& prediction, const PlanarBuffer& source, double sigma,
                                   SmoothingStats* stats) {
    const int h = source.height(), w = source.width();
    const std::size_t n = source.plane_size();
    const float clip = static_cast<float>(kClipCode);
    std::vector<std::uint8_t> full(n), ok(n);
    std::vector<double> ly(n, 0.0);
    std::size_t full_count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const float bottom = std::min(std::min(source.plane(0)[i], source.plane(1)[i]), source.plane(2)[i]);
        full[i] = bottom >= clip;
        full_count += full[i];
        const double y = static_cast<double>(prediction.plane(0)[i]) * kLumaAp1[0] +
                         static_cast<double>(prediction.plane(1)[i]) * kLumaAp1[1] +
                         static_cast<double>(prediction.plane(2)[i]) * kLumaAp1[2];
        ok[i] = full[i] && y > 1e-10;
        if (ok[i]) ly[i] = std::log2(std::max(y, 1e-10));
    }
    PlanarBuffer out = prediction;
    if (stats) *stats = SmoothingStats{full_count, 0, 0};
    if (std::none_of(ok.begin(), ok.end(), [](std::uint8_t v) { return v != 0; })) return out;

    // Support-normalised Gaussian of ly over ok pixels, evaluated only where needed (ok).
    const int ksize = static_cast<int>(4 * sigma) | 1, r = ksize / 2;
    const auto k = gaussian_kernel(ksize, sigma);
    int x0 = w, x1 = -1;
    std::vector<std::uint8_t> row_has(static_cast<std::size_t>(h), 0);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            if (ok[static_cast<std::size_t>(y) * w + x]) {
                row_has[y] = 1;
                x0 = std::min(x0, x);
                x1 = std::max(x1, x);
            }
    const int bw = x1 - x0 + 1;
    // Horizontal pass for every row (zero rows skipped), columns x0..x1.
    std::vector<double> hn(static_cast<std::size_t>(h) * bw, 0.0), hd(hn.size(), 0.0);
    for (int y = 0; y < h; ++y) {
        if (!row_has[y]) continue;
        const std::size_t base = static_cast<std::size_t>(y) * w;
        for (int x = x0; x <= x1; ++x) {
            double sn = 0.0, sd = 0.0;
            for (int j = 0; j < ksize; ++j) {
                const std::size_t q = base + reflect101(x + j - r, w);
                if (ok[q]) {
                    sn += k[j] * ly[q];
                    sd += k[j];
                }
            }
            hn[static_cast<std::size_t>(y) * bw + (x - x0)] = sn;
            hd[static_cast<std::size_t>(y) * bw + (x - x0)] = sd;
        }
    }
    std::vector<double> ls = ly;
    for (int y = 0; y < h; ++y)
        for (int x = x0; x <= x1; ++x) {
            const std::size_t i = static_cast<std::size_t>(y) * w + x;
            if (!ok[i]) continue;
            double sn = 0.0, sd = 0.0;
            for (int j = 0; j < ksize; ++j) {
                const int yy = reflect101(y + j - r, h);
                if (!row_has[yy]) continue;
                sn += k[j] * hn[static_cast<std::size_t>(yy) * bw + (x - x0)];
                sd += k[j] * hd[static_cast<std::size_t>(yy) * bw + (x - x0)];
            }
            ls[i] = sd > 1e-6 ? sn / std::max(sd, 1e-6) : ly[i];
        }

    // Flat log luminance per connected component of the eroded interior.
    auto interior = erode_cross(full, h, w, 2);
    for (std::size_t i = 0; i < n; ++i) interior[i] = interior[i] && ok[i];
    std::vector<int> labels;
    const int nl = label8(interior, h, w, labels);
    std::vector<double> sums(static_cast<std::size_t>(nl), 0.0);
    std::vector<std::size_t> counts(static_cast<std::size_t>(nl), 0);
    if (nl > 1) {
        for (std::size_t i = 0; i < n; ++i)
            if (interior[i]) {
                sums[labels[i]] += ls[i];
                ++counts[labels[i]];
            }
        for (std::size_t i = 0; i < n; ++i)
            if (interior[i]) ls[i] = sums[labels[i]] / static_cast<double>(std::max<std::size_t>(counts[labels[i]], 1));
    }
    // ramp 1, start 0: distance >= 1 for every support pixel, so the blend weight is ok.
    std::vector<double> o64(n * 3);
    for (std::size_t i = 0; i < n; ++i) {
        const double s = ok[i] ? std::exp2(ls[i] - ly[i]) : 1.0;
        for (int c = 0; c < 3; ++c) {
            const double p = prediction.plane(c)[i];
            o64[i * 3 + c] = full[i] ? p * s : p;
        }
    }
    if (nl > 1) {
        for (int c = 0; c < 3; ++c) {
            std::fill(sums.begin(), sums.end(), 0.0);
            for (std::size_t i = 0; i < n; ++i)
                if (interior[i]) sums[labels[i]] += o64[i * 3 + c];
            for (std::size_t i = 0; i < n; ++i)
                if (interior[i]) o64[i * 3 + c] = sums[labels[i]] / static_cast<double>(std::max<std::size_t>(counts[labels[i]], 1));
        }
    }
    for (std::size_t i = 0; i < n; ++i)
        if (full[i])
            for (int c = 0; c < 3; ++c) out.plane(c)[i] = static_cast<float>(o64[i * 3 + c]);
    if (stats) {
        stats->interior_pixels = static_cast<std::size_t>(std::count(interior.begin(), interior.end(), 1));
        stats->components = nl > 0 ? static_cast<std::size_t>(nl - 1) : 0;
    }
    return out;
}

PlanarBuffer constrain_colour_v2(const PlanarBuffer& source, const PlanarBuffer& prediction,
                                 const PlanarBuffer& baseline, const PlanarBuffer& weight, const AcesView& view,
                                 bool include_fully_clipped, ConstraintStats* stats, bool forbid_new_negative_y) {
    const std::size_t n = source.plane_size();
    // Same constants and float64 arithmetic as the Python guard (LUMA_AP1).
    auto luma = [](const PlanarBuffer& b, std::size_t i) {
        return static_cast<double>(b.plane(0)[i]) * 0.2722287168 + static_cast<double>(b.plane(1)[i]) * 0.6740817658 +
               static_cast<double>(b.plane(2)[i]) * 0.0536895174;
    };
    const float clip = static_cast<float>(kClipCode);
    PlanarBuffer result = prediction;
    std::vector<std::uint8_t> support(n), active(n);
    std::vector<float> alpha(n, 1.0f);
    for (std::size_t i = 0; i < n; ++i) {
        const float s0 = source.plane(0)[i], s1 = source.plane(1)[i], s2 = source.plane(2)[i];
        bool s = weight.plane(0)[i] > 0 && std::max(std::max(s0, s1), s2) > 0.5f;
        if (!include_fully_clipped) s = s && std::min(std::min(s0, s1), s2) < clip;
        support[i] = active[i] = s;
    }
    auto display_error = [&](const std::vector<std::size_t>& idx, std::vector<float>& err) {
        std::vector<float> packed(idx.size() * 3);
        for (std::size_t k = 0; k < idx.size(); ++k)
            for (int c = 0; c < 3; ++c) packed[k * 3 + c] = result.plane(c)[idx[k]];
        view.forward(packed.data(), idx.size());
        err.assign(idx.size(), 0.0f);
        for (std::size_t k = 0; k < idx.size(); ++k) {
            float e = 0.0f;
            for (int c = 0; c < 3; ++c) e = std::max(e, std::fabs(packed[k * 3 + c] - source.plane(c)[idx[k]]) * 255.0f);
            err[k] = e;
        }
    };
    std::vector<std::size_t> idx;
    std::vector<float> err;
    for (int step = 0; step < 9; ++step) {
        idx.clear();
        for (std::size_t i = 0; i < n; ++i)
            if (active[i]) idx.push_back(i);
        if (idx.empty()) break;
        display_error(idx, err);
        bool any = false;
        for (std::size_t k = 0; k < idx.size(); ++k) {
            bool ok = err[k] <= 1.0f;
            if (ok && forbid_new_negative_y) ok = luma(result, idx[k]) >= 0.0 || luma(baseline, idx[k]) < 0.0;
            if (ok) active[idx[k]] = 0;
            else any = true;
        }
        if (!any) break;
        for (std::size_t i = 0; i < n; ++i) {
            if (!active[i]) continue;
            alpha[i] = step < 8 ? alpha[i] * 0.5f : 0.0f;
            for (int c = 0; c < 3; ++c) {
                const float b = baseline.plane(c)[i];
                result.plane(c)[i] = b + alpha[i] * (prediction.plane(c)[i] - b);
            }
        }
    }
    if (stats) {
        ConstraintStats s;
        idx.clear();
        for (std::size_t i = 0; i < n; ++i)
            if (support[i]) {
                idx.push_back(i);
                ++s.support_pixels;
                s.changed_pixels += alpha[i] < 1.0f;
                s.baseline_fallback_pixels += alpha[i] == 0.0f;
                s.new_negative_luminance += luma(result, i) < 0.0 && luma(baseline, i) >= 0.0;
            }
        if (!idx.empty()) {
            display_error(idx, err);
            for (float e : err) s.remaining_display_errors += e > 1.0f;
        }
        *stats = s;
    }
    return result;
}

Result<FrameResult> run_frame(const PlanarBuffer& source, const AcesView& view, const G1Fn& g1, bool smoothing,
                              bool constrain, bool forbid_new_negative_y) {
    if (source.channels() != 3) return make_error(ErrorCode::InvalidArgument, "The frame must be RGB.");
    for (float v : source.span())
        if (!std::isfinite(v) || v < 0.0f || v > 1.0f)
            return make_error(ErrorCode::InvalidArgument, "Frame codes must be finite and within 0..1.",
                              "aces_v2::run_frame source outside [0,1]");
    const PlanarBuffer baseline = view_inverse(view, source);
    auto net = g1(source, baseline);
    if (!net) return net.error();
    const G1Output& g = net.value();
    const PlanarBuffer restored = view_inverse(view, g.restored_source);
    Composed composed = compose_v2(source, baseline, restored, g);
    FrameResult out;
    out.mask = std::move(composed.mask);
    out.acescg = smoothing ? smooth_clipped_hybrid(composed.acescg, source, 64.0, &out.smoothing)
                           : std::move(composed.acescg);
    if (constrain) out.acescg = constrain_colour_v2(source, out.acescg, baseline, g.weight, view, true, &out.constraint,
                                                    forbid_new_negative_y);
    for (float v : out.acescg.span())
        if (!std::isfinite(v)) return make_error(ErrorCode::BackendError, "The reconstruction produced a non-finite value.");
    return out;
}

}  // namespace rudra::aces_v2
