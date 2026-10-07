#include "rudra/core/view.hpp"

#include "rudra/core/compare.hpp"
#include "rudra/core/gamut.hpp"
#include "rudra/core/hdr10.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace rudra {
namespace {

constexpr float kPeak = 10000.0f;   // network units to nits

// Zone upper bounds (nits) and colours, docs/view.spec.md section 2.
constexpr std::array<float, 9> kZoneBelow{0.1f, 1.0f, 10.0f, 100.0f, 160.0f, 250.0f, 1000.0f, 4000.0f, 10000.0f};
constexpr std::array<std::array<float, 3>, 10> kZoneColour{{
    {0.169f, 0.122f, 0.239f}, {0.184f, 0.294f, 0.561f}, {0.184f, 0.561f, 0.722f},
    {0.200f, 0.627f, 0.416f}, {0.604f, 0.655f, 0.698f}, {0.914f, 0.929f, 0.945f},
    {0.910f, 0.765f, 0.290f}, {0.910f, 0.529f, 0.227f}, {0.816f, 0.263f, 0.184f},
    {0.780f, 0.290f, 0.780f},
}};

float lum2020(float r, float g, float b) noexcept {
    return r * kRec2020Luma[0] + g * kRec2020Luma[1] + b * kRec2020Luma[2];
}

float srgb_to_linear(float c) noexcept {
    c = std::clamp(c, 0.0f, 1.0f);
    return c > 0.04045f ? std::pow((c + 0.055f) / 1.055f, 2.4f) : c / 12.92f;
}

using Mat3f = std::array<float, 9>;
Mat3f to_float(const Mat3& m) {
    Mat3f f{};
    for (int i = 0; i < 9; ++i) f[std::size_t(i)] = float(m[std::size_t(i / 3)][std::size_t(i % 3)]);
    return f;
}

// Absolute nits in the target's primaries -> the value the swapchain takes.
float encode(const DisplayTarget& t, float nits) noexcept {
    return t.path == OutputPath::Hdr10 ? pq_oetf(nits) : nits / float(t.unit_nits);
}

}  // namespace

int false_colour_zone(float nits) noexcept {
    for (int i = 0; i < int(kZoneBelow.size()); ++i)
        if (nits < kZoneBelow[std::size_t(i)]) return i;
    return int(kZoneBelow.size());   // NaN lands here too, as in the shader
}

std::array<float, 3> false_colour(int zone) noexcept {
    return kZoneColour[std::size_t(std::clamp(zone, 0, int(kZoneColour.size()) - 1))];
}

float linear_to_srgb(float x) noexcept {
    x = std::clamp(x, 0.0f, 1.0f);
    return x > 0.0031308f ? 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f : 12.92f * x;
}

std::vector<float> sdr_max_codes(const SdrImage* sdr, int width, int height) {
    const std::size_t n = std::size_t(width) * std::size_t(height);
    std::vector<float> out(n, 0.5f);
    if (!sdr || sdr->width() != width || sdr->height() != height) return out;
    const PlanarBuffer& c = sdr->buffer();
    for (std::size_t i = 0; i < n; ++i) out[i] = std::max(std::max(c.plane(0)[i], c.plane(1)[i]), c.plane(2)[i]);
    return out;
}

namespace {
// core/master.cpp srgb_to_linear_d: the anchor's own decode, in double.
double srgb_to_linear_dbl(double c) noexcept {
    c = std::clamp(c, 0.0, 1.0);
    return c > 0.04045 ? std::pow((c + 0.055) / 1.055, 2.4) : c / 12.92;
}
}  // namespace

double anchor_hold(const NetworkLinearImage& model, const SdrImage& sdr, double knee, double softness) {
    // anchor_to_sdr's hold, on the composite in network units (x 10 000 = nits).
    const std::size_t n = model.buffer().plane_size();
    const PlanarBuffer& m = model.buffer();
    const PlanarBuffer& s = sdr.buffer();
    constexpr double eps = 1e-4, luma[3] = {0.2627, 0.6780, 0.0593};
    std::vector<double> want, band;
    want.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        double target = 0.0, actual = 0.0, mx = -1.0;
        for (int c = 0; c < 3; ++c) {
            const double v = s.plane(c)[i];
            target += srgb_to_linear_dbl(v) * luma[c];
            actual += double(m.plane(c)[i]) * 10000.0 * luma[c];
            mx = std::max(mx, v);
        }
        target *= 203.0;
        const double g = (target + eps) / (actual + eps);
        want.push_back(g);
        if (mx > knee - softness && mx < knee + softness && actual > 1e-9) band.push_back(g);
    }
    auto median = [](std::vector<double> v) {
        if (v.empty()) return 1.0;
        const std::size_t mid = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + std::ptrdiff_t(mid), v.end());
        double hi = v[mid];
        if (v.size() % 2 == 0) {
            const double lo = *std::max_element(v.begin(), v.begin() + std::ptrdiff_t(mid));
            return (lo + hi) * 0.5;
        }
        return hi;
    };
    const double hold = band.size() >= 64 ? median(std::move(band)) : median(std::move(want));
    return std::isfinite(hold) ? hold : 1.0;
}

float anchor_gain_f(float target_luma, float r, float g, float b, float max_code, float knee, float softness,
                    float hold) noexcept {
    const float eps = 1e-4f;
    const float target = target_luma * 203.0f;
    const float actual = (0.2627f * r + 0.6780f * g + 0.0593f * b) * kPeak;
    float gain = (target + eps) / (actual + eps);
    const float t = std::clamp((max_code - (knee - softness)) / (2.0f * softness), 0.0f, 1.0f);
    const float ramp = t * t * (3.0f - 2.0f * t);
    gain = gain * (1.0f - ramp) + hold * ramp;
    return std::isfinite(gain) ? gain : 1.0f;
}

PlanarBuffer render_view(const NetworkLinearImage& model, const NetworkLinearImage& baseline,
                         const ViewParams& p, const SdrImage* sdr) {
    assert(model.width() == baseline.width() && model.height() == baseline.height());
    const int w = model.width(), h = model.height();
    const PlanarBuffer& m = model.buffer();
    const PlanarBuffer& b = baseline.buffer();
    const bool wiping = p.wipe >= 0.0;
    const bool painting = p.paint_band >= 0 && p.masks && p.masks->has(p.paint_band);
    const PlanarBuffer& source = (!wiping && p.show == ViewSource::Baseline) ? b : m;
    // Uniforms reach the shader as fp32.
    const float scale = float(double(kPeak) / std::max(p.display_nits, 1e-3));
    const float wipe = wiping ? float(std::clamp(p.wipe, 0.0, 1.0)) : -1.0f;
    const float half_width = float(p.wipe_half_width);
    const float log_gain = std::log2(1.0f + float(std::max(p.diff_gain, 1.0)));

    const bool hdr = p.target.path != OutputPath::SdrPqSimulation;
    // HDR paths: the picture in absolute nits, clipped at the lower of the view
    // peak and the display's peak, never tone-mapped (ADR-005); overlays (false
    // colour, difference, the wipe handle) are graphics at the SDR white.
    const float ceiling = float(std::min(p.display_nits, p.target.peak_nits));
    const Mat3f picture_m = to_float(rgb_to_rgb_matrix(p.source, p.target.primaries));
    const Mat3f graphics_m = to_float(rgb_to_rgb_matrix(Primaries::Rec709, p.target.primaries));
    const float graphics_white = float(kDiffuseWhite.v);
    const bool anchoring = p.anchor && sdr != nullptr;
    const std::vector<float> codes =
        (p.mode == ViewMode::Invented || anchoring) ? sdr_max_codes(sdr, w, h) : std::vector<float>{};
    const float knee = float(p.anchor_knee), softness = float(p.anchor_softness), hold = float(p.anchor_hold);
    const PlanarBuffer* sdr_buf = anchoring ? &sdr->buffer() : nullptr;

    PlanarBuffer out(3, h, w);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float u = (float(x) + 0.5f) / float(w);
            const PlanarBuffer& pic = (wiping && u < wipe) ? b : source;
            float hr = pic.at(0, y, x), hg = pic.at(1, y, x), hb = pic.at(2, y, x);
            // The anchor (3.3) on the model's picture only: the baseline is the
            // inverse as it is, on its own or on the wipe's left.
            if (anchoring && &pic == &m) {
                const std::size_t i = std::size_t(y) * std::size_t(w) + std::size_t(x);
                const float target = 0.2627f * srgb_to_linear(sdr_buf->plane(0)[i]) +
                                     0.6780f * srgb_to_linear(sdr_buf->plane(1)[i]) +
                                     0.0593f * srgb_to_linear(sdr_buf->plane(2)[i]);
                const float gain = anchor_gain_f(target, hr, hg, hb, codes[i], knee, softness, hold);
                hr *= gain, hg *= gain, hb *= gain;
            }
            float c[3];
            switch (p.mode) {
                case ViewMode::FalseColour: {
                    const auto fc = false_colour(false_colour_zone(lum2020(hr, hg, hb) * kPeak));
                    c[0] = fc[0], c[1] = fc[1], c[2] = fc[2];
                    break;
                }
                case ViewMode::Difference: {
                    const float d = lum2020(std::abs(source.at(0, y, x) - b.at(0, y, x)),
                                            std::abs(source.at(1, y, x) - b.at(1, y, x)),
                                            std::abs(source.at(2, y, x) - b.at(2, y, x)));
                    const float v = std::clamp(std::log2(1.0f + d * kPeak) / log_gain, 0.0f, 1.0f);
                    c[0] = v * 0.95f, c[1] = v * 0.62f, c[2] = v * 0.28f;
                    break;
                }
                case ViewMode::Invented: {
                    const float g = kMapGrey * linear_to_srgb(lum2020(hr, hg, hb) * scale);
                    const float mn = std::max(std::max(m.at(0, y, x), m.at(1, y, x)), m.at(2, y, x)) * kPeak;
                    const float bn = std::max(std::max(b.at(0, y, x), b.at(1, y, x)), b.at(2, y, x)) * kPeak;
                    const float a = kMapMix * std::abs(change_weight(mn, bn));
                    const auto& col = sdr_has_no_information(codes[std::size_t(y) * std::size_t(w) + std::size_t(x)])
                                          ? kInventedColour : kReinterpretedColour;
                    for (int k = 0; k < 3; ++k) c[k] = g * (1.0f - a) + col[std::size_t(k)] * a;
                    break;
                }
                case ViewMode::Image:
                default:
                    if (hdr) {
                        c[0] = std::clamp(hr * kPeak, 0.0f, ceiling);
                        c[1] = std::clamp(hg * kPeak, 0.0f, ceiling);
                        c[2] = std::clamp(hb * kPeak, 0.0f, ceiling);
                    } else {
                        c[0] = linear_to_srgb(hr * scale);
                        c[1] = linear_to_srgb(hg * scale);
                        c[2] = linear_to_srgb(hb * scale);
                    }
                    break;
            }
            if (p.show_changes && p.mode == ViewMode::Image) {
                // Model against baseline at this pixel, whichever side is shown.
                const float mn = std::max(std::max(m.at(0, y, x), m.at(1, y, x)), m.at(2, y, x)) * kPeak;
                const float bn = std::max(std::max(b.at(0, y, x), b.at(1, y, x)), b.at(2, y, x)) * kPeak;
                const float cw = change_weight(mn, bn);
                if (cw != 0.0f) {
                    const auto& col = cw > 0.0f ? kChangeUp : kChangeDown;
                    const float a = kChangeTintMix * std::abs(cw);
                    for (int k = 0; k < 3; ++k) {
                        const float t = hdr ? graphics_white * srgb_to_linear(col[std::size_t(k)]) : col[std::size_t(k)];
                        c[k] = c[k] * (1.0f - a) + t * a;
                    }
                }
            }
            if (painting && p.mode == ViewMode::Image) {
                // The mask being painted (3.3), tinted over whichever side is shown.
                const float a = kMaskTintMix * p.masks->weight(p.paint_band, x, y, w, h);
                if (a > 0.0f)
                    for (int k = 0; k < 3; ++k) {
                        const float t = hdr ? graphics_white * srgb_to_linear(p.paint_tint[std::size_t(k)]) : p.paint_tint[std::size_t(k)];
                        c[k] = c[k] * (1.0f - a) + t * a;
                    }
            }
            const bool handle = wiping && std::abs(u - wipe) < half_width;
            if (!hdr) {
                if (handle)
                    for (float& v : c) v = 1.0f - v;
                for (int k = 0; k < 3; ++k) out.at(k, y, x) = c[k];
                continue;
            }
            // HDR: c is nits in the source primaries (image) or an SDR code (graphics).
            const bool picture = p.mode == ViewMode::Image;
            float n[3];
            for (int k = 0; k < 3; ++k) {
                if (picture) n[k] = handle ? ceiling - c[k] : c[k];
                else n[k] = graphics_white * srgb_to_linear(handle ? 1.0f - c[k] : c[k]);
            }
            const Mat3f& m3 = picture ? picture_m : graphics_m;
            for (int k = 0; k < 3; ++k) {
                const float v = m3[std::size_t(k) * 3] * n[0] + m3[std::size_t(k) * 3 + 1] * n[1] + m3[std::size_t(k) * 3 + 2] * n[2];
                out.at(k, y, x) = encode(p.target, v);
            }
        }
    }
    return out;
}

Rgb8Image render_view_rgb8(const NetworkLinearImage& model, const NetworkLinearImage& baseline,
                           const ViewParams& p, const SdrImage* sdr) {
    const PlanarBuffer f = render_view(model, baseline, p, sdr);
    Rgb8Image out{f.width(), f.height(), std::vector<std::uint8_t>(f.plane_size() * 3)};
    const std::size_t n = f.plane_size();
    for (std::size_t i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) {
            const float v = std::clamp(f.plane(k)[i], 0.0f, 1.0f);
            out.rgb[i * 3 + std::size_t(k)] = std::uint8_t(std::lround(v * 255.0f));
        }
    return out;
}

Reductions reduce_ladder(const PlanarBuffer& rgb) {
    int w = rgb.width(), h = rgb.height();
    std::vector<float> mx(std::size_t(w) * h), sm;
    for (std::size_t i = 0; i < mx.size(); ++i)
        mx[i] = std::max(std::max(rgb.plane(0)[i], rgb.plane(1)[i]), rgb.plane(2)[i]);
    sm = mx;
    while (w > 1 || h > 1) {
        const int dw = std::max(1, (w + 1) / 2), dh = std::max(1, (h + 1) / 2);
        std::vector<float> nm(std::size_t(dw) * dh), ns(std::size_t(dw) * dh);
        for (int y = 0; y < dh; ++y)
            for (int x = 0; x < dw; ++x) {
                float am = -1e30f, as = 0.0f;
                for (int j = 0; j < 2; ++j)
                    for (int i = 0; i < 2; ++i) {
                        const int px = 2 * x + i, py = 2 * y + j;
                        if (px >= w || py >= h) continue;
                        const std::size_t k = std::size_t(py) * w + px;
                        am = std::max(am, mx[k]);
                        as = as + sm[k];
                    }
                nm[std::size_t(y) * dw + x] = am;
                ns[std::size_t(y) * dw + x] = as;
            }
        mx.swap(nm);
        sm.swap(ns);
        w = dw;
        h = dh;
    }
    return {mx[0], sm[0]};
}

Probe probe_pixel(const NetworkLinearImage& model, const NetworkLinearImage& baseline, double x, double y) {
    const int w = model.width(), h = model.height();
    // Math.round: half up.
    const int px = std::clamp(int(std::floor(x + 0.5)), 0, w - 1);
    const int py = std::clamp(int(std::floor(y + 0.5)), 0, h - 1);
    auto one = [&](const NetworkLinearImage& img) {
        ProbeSample s;
        s.x = px;
        s.y = py;
        const float r = img.buffer().at(0, py, px), g = img.buffer().at(1, py, px), b = img.buffer().at(2, py, px);
        s.rgb_nits[0] = double(r) * 10000.0;
        s.rgb_nits[1] = double(g) * 10000.0;
        s.rgb_nits[2] = double(b) * 10000.0;
        s.nits = (double(r) * 0.2627 + double(g) * 0.6780 + double(b) * 0.0593) * 10000.0;
        return s;
    };
    return {one(model), one(baseline)};
}

}  // namespace rudra
