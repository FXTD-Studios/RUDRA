#include "rudra/core/video_predict.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

#include "rudra/core/baseline.hpp"
#include "rudra/core/gamut.hpp"
#include "rudra/core/view.hpp"

namespace rudra {

namespace {
constexpr double kLuma2020[3] = {0.2627, 0.6780, 0.0593};
constexpr double kLuma709[3] = {0.2126, 0.7152, 0.0722};
}  // namespace

Result<PlanarBuffer> canonicalize_sdr(const PlanarBuffer& rgb, std::string_view transfer, std::string_view range) {
    if (range != "full" && range != "limited")
        return make_error(ErrorCode::InvalidArgument, "value_range must be 'full' or 'limited'");
    if (transfer != "srgb" && transfer != "rec709" && transfer != "gamma22" && transfer != "gamma24")
        return make_error(ErrorCode::InvalidArgument, "transfer must be srgb, rec709, gamma22, or gamma24");
    const bool limited = range == "limited";
    const float lo = static_cast<float>(16.0 / 255.0), span = static_cast<float>(219.0 / 255.0);
    const float inv045 = static_cast<float>(1.0 / 0.45);
    PlanarBuffer out = rgb;
    for (float& v : out.span()) {
        float x = v;
        if (limited) x = (x - lo) / span;
        x = std::clamp(x, 0.0f, 1.0f);
        if (transfer == "srgb") {
            v = x;
            continue;
        }
        float linear;
        if (transfer == "rec709")
            linear = x < 0.081f ? x / 4.5f : std::pow((x + 0.099f) / 1.099f, inv045);
        else
            linear = std::pow(x, transfer == "gamma22" ? 2.2f : 2.4f);
        v = linear_to_srgb(linear);
    }
    return out;
}

PlanarBuffer srgb_code_to_rec2020(const PlanarBuffer& code, Primaries src) {
    const Mat3 md = rgb_to_rgb_matrix(src, Primaries::Rec2020);
    float m[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) m[i][j] = static_cast<float>(md[std::size_t(i)][std::size_t(j)]);
    PlanarBuffer out(3, code.height(), code.width());
    const std::size_t n = code.plane_size();
    const float* in[3] = {code.plane(0), code.plane(1), code.plane(2)};
    float* dst[3] = {out.plane(0), out.plane(1), out.plane(2)};
    for (std::size_t p = 0; p < n; ++p) {
        const float l[3] = {srgb_to_linear(in[0][p]), srgb_to_linear(in[1][p]), srgb_to_linear(in[2][p])};
        for (int i = 0; i < 3; ++i) dst[i][p] = linear_to_srgb(m[i][0] * l[0] + m[i][1] * l[1] + m[i][2] * l[2]);
    }
    return out;
}

PlanarBuffer area_resize(const PlanarBuffer& x, int height, int width) {
    // aten adaptive_avg_pool2d: bins [floor(o * in / out), ceil((o + 1) * in / out)),
    // the index arithmetic in float, a float sum, divided by the bin's height then width.
    const int ih = x.height(), iw = x.width();
    auto start = [](int o, int in, int out) {
        return static_cast<int>(std::floor(static_cast<float>(o * in) / static_cast<float>(out)));
    };
    auto end = [](int o, int in, int out) {
        return static_cast<int>(std::ceil(static_cast<float>((o + 1) * in) / static_cast<float>(out)));
    };
    PlanarBuffer out(x.channels(), height, width);
    for (int c = 0; c < x.channels(); ++c) {
        const float* src = x.plane(c);
        float* dst = out.plane(c);
        for (int oh = 0; oh < height; ++oh) {
            const int h0 = start(oh, ih, height), h1 = end(oh, ih, height);
            for (int ow = 0; ow < width; ++ow) {
                const int w0 = start(ow, iw, width), w1 = end(ow, iw, width);
                float sum = 0.0f;
                for (int y = h0; y < h1; ++y)
                    for (int xx = w0; xx < w1; ++xx) sum += src[static_cast<std::size_t>(y) * iw + xx];
                dst[static_cast<std::size_t>(oh) * width + ow] =
                    sum / static_cast<float>(h1 - h0) / static_cast<float>(w1 - w0);
            }
        }
    }
    return out;
}

std::vector<float> torch_linspace(float start, float end, int steps) {
    std::vector<float> v(static_cast<std::size_t>(std::max(steps, 0)));
    if (steps == 1) v[0] = start;
    if (steps <= 1) return v;
    // The step in float; each value as one fused multiply-add (the CPU kernel's
    // vector path), which the exact double product and sum reproduce.
    const float step = (end - start) / static_cast<float>(steps - 1);
    const int halfway = steps / 2;
    for (int i = 0; i < steps; ++i)
        v[std::size_t(i)] = i < halfway ? static_cast<float>(double(start) + double(step) * i)
                                        : static_cast<float>(double(end) - double(step) * (steps - i - 1));
    return v;
}

std::pair<double, bool> ShadowSmoother::update(const PlanarBuffer& thumbnail, double weight) {
    bool cut = previous_.empty();
    if (!cut) {
        double sum = 0.0;
        const std::span<const float> a = thumbnail.span(), b = std::as_const(previous_).span();
        for (std::size_t i = 0; i < a.size(); ++i) sum += std::fabs(static_cast<double>(a[i] - b[i]));
        cut = static_cast<double>(static_cast<float>(sum / static_cast<double>(a.size()))) >= cut_threshold_;
    }
    weight_ = (cut || !has_weight_) ? weight : retention_ * weight_ + (1.0 - retention_) * weight;
    has_weight_ = true;
    previous_ = thumbnail;
    if (cut) {
        held_.clear();
        pixel_luma_.clear();
        pixel_peak_.clear();
        pixel_gain_.clear();
    }
    return {weight_, cut};
}

std::vector<double> ShadowSmoother::hold(const std::string& key, const std::vector<double>& value) {
    auto it = held_.find(key);
    if (it == held_.end() || it->second.size() != value.size()) return held_[key] = value;
    std::vector<double>& h = it->second;
    for (std::size_t i = 0; i < h.size(); ++i) h[i] = retention_ * h[i] + (1.0 - retention_) * value[i];
    return h;
}

PlanarBuffer ShadowSmoother::hold_pixels(std::vector<double> luma, std::vector<double> peak,
                                         const std::vector<double>& linear, const PlanarBuffer& hdr) {
    if (!static_hold_ || retention_ == 0.0) return hdr;
    const int h = hdr.height(), w = hdr.width();
    const std::size_t n = hdr.plane_size();
    std::vector<double> gain(n);
    std::vector<char> negative(n);
    const float *h0 = hdr.plane(0), *h1 = hdr.plane(1), *h2 = hdr.plane(2);
    for (std::size_t i = 0; i < n; ++i) {
        const double yh = double(h0[i]) * kLuma2020[0] + double(h1[i]) * kLuma2020[1] + double(h2[i]) * kLuma2020[2];
        gain[i] = (yh + kStaticHdrFloor) / (linear[i] + kStaticSdrFloor);
        negative[i] = yh < 0.0;
    }
    if (pixel_luma_.size() != n) {   // a shot's first frame
        pixel_luma_ = std::move(luma);
        pixel_peak_ = std::move(peak);
        pixel_gain_ = std::move(gain);
        return hdr;
    }
    // |luma - previous|, its 3x3 mean with the edges repeated (np.pad
    // mode='edge'), summed in the Python's order.
    std::vector<double> diff(n);
    for (std::size_t i = 0; i < n; ++i) diff[i] = std::fabs(luma[i] - pixel_luma_[i]);
    PlanarBuffer out(3, h, w);
    float *o0 = out.plane(0), *o1 = out.plane(1), *o2 = out.plane(2);
    for (int y = 0; y < h; ++y) {
        const double* rows[3] = {diff.data() + std::size_t(std::max(y - 1, 0)) * w, diff.data() + std::size_t(y) * w,
                                 diff.data() + std::size_t(std::min(y + 1, h - 1)) * w};
        for (int x = 0; x < w; ++x) {
            const int xs[3] = {std::max(x - 1, 0), x, std::min(x + 1, w - 1)};
            double sum = 0.0;
            for (const double* row : rows)
                for (int xi : xs) sum = sum + row[xi];
            const double blurred = sum / 9.0;
            const double t = std::clamp((blurred - kStaticStill) / (kStaticMoved - kStaticStill), 0.0, 1.0);
            const std::size_t i = std::size_t(y) * w + x;
            const bool skip = std::max(peak[i], pixel_peak_[i]) >= kStaticClipped ||
                              std::min(peak[i], pixel_peak_[i]) <= kStaticCrushed || negative[i];
            const double hold = skip ? 0.0 : retention_ * (1.0 - t * t * (3.0 - 2.0 * t));
            const double held = hold * pixel_gain_[i] + (1.0 - hold) * gain[i];
            pixel_gain_[i] = held;
            const double f = held / gain[i];
            o0[i] = static_cast<float>(double(h0[i]) * f);
            o1[i] = static_cast<float>(double(h1[i]) * f);
            o2[i] = static_cast<float>(double(h2[i]) * f);
        }
    }
    pixel_luma_ = std::move(luma);
    pixel_peak_ = std::move(peak);
    return out;
}

double retention_for(std::string_view fps, double seconds) {
    if (!(seconds > 0.0)) return 0.0;
    const std::string f(fps);
    const auto slash = f.find('/');
    double rate = std::strtod(f.c_str(), nullptr);
    if (slash != std::string::npos) rate /= std::strtod(f.c_str() + slash + 1, nullptr);
    return std::exp(-1.0 / (rate * seconds));
}

double exposure_ev(std::span<const double> lum, int h, int w) {
    const std::vector<double> cells = exposure_cells(lum, h, w);
    double sum = 0.0;
    for (double v : cells) sum += v;
    return sum / double(cells.size());
}

std::vector<double> exposure_cells(std::span<const double> lum, int h, int w) {
    // rudra/video.py area_mean: an integral image (cumsum down, then across)
    // and adaptive average pooling's bins.
    const int H = kExposureThumbHeight, W = kExposureThumbWidth;
    std::vector<double> integral(std::size_t(h + 1) * (w + 1), 0.0);
    auto at = [&](int y, int x) -> double& { return integral[std::size_t(y) * (w + 1) + x]; };
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) at(y + 1, x + 1) = at(y, x + 1) + lum[std::size_t(y) * w + x];
    for (int y = 1; y <= h; ++y)
        for (int x = 1; x <= w; ++x) at(y, x) += at(y, x - 1);
    std::vector<double> cells;
    cells.reserve(std::size_t(H) * W);
    for (int i = 0; i < H; ++i) {
        const int y0 = (i * h) / H, y1 = ((i + 1) * h + H - 1) / H;
        for (int j = 0; j < W; ++j) {
            const int x0 = (j * w) / W, x1 = ((j + 1) * w + W - 1) / W;
            const double area = double(y1 - y0) * double(x1 - x0);
            const double m = (at(y1, x1) - at(y0, x1) - at(y1, x0) + at(y0, x0)) / area;
            cells.push_back(std::log2(std::max(m, 0.0) + 1.0 / 1024.0));
        }
    }
    return cells;
}

namespace {
double transfer_to_linear(double x, std::string_view transfer) {
    x = std::clamp(x, 0.0, 1.0);
    if (transfer == "srgb") return x <= 0.04045 ? x / 12.92 : std::pow((x + 0.055) / 1.055, 2.4);
    if (transfer == "rec709") return x < 0.081 ? x / 4.5 : std::pow((x + 0.099) / 1.099, 1.0 / 0.45);
    if (transfer == "gamma22") return std::pow(x, 2.2);
    return std::pow(x, 2.4);
}

// The transfer at every 16-bit code a decoded frame holds (k / 65535 in
// float32, as rgb_from_frame divides): the same values, a table look-up.
const std::vector<double>& code_table(std::string_view transfer) {
    static const auto make = [](std::string_view t) {
        std::vector<double> lut(65536);
        for (int k = 0; k < 65536; ++k) lut[std::size_t(k)] = transfer_to_linear(double(float(k) / 65535.0f), t);
        return lut;
    };
    static const std::vector<double> srgb = make("srgb"), rec709 = make("rec709"), g22 = make("gamma22"),
                                     g24 = make("gamma24");
    return transfer == "srgb" ? srgb : transfer == "rec709" ? rec709 : transfer == "gamma22" ? g22 : g24;
}
}  // namespace

std::vector<double> source_luminance(const PlanarBuffer& rgb, std::string_view transfer, std::string_view primaries) {
    const double* k = primaries == "rec2020" ? kLuma2020 : kLuma709;
    const std::vector<double>& lut = code_table(transfer);
    std::vector<double> out(rgb.plane_size(), 0.0);
    for (int c = 0; c < 3; ++c) {
        const float* p = rgb.plane(c);
        for (std::size_t i = 0; i < out.size(); ++i) {
            const float x = p[i];
            const long code = std::lround(static_cast<double>(x) * 65535.0);
            const double v = code >= 0 && code <= 65535 && float(code) / 65535.0f == x
                                 ? lut[std::size_t(code)]
                                 : transfer_to_linear(static_cast<double>(x), transfer);
            out[i] += v * k[c];
        }
    }
    return out;
}

std::pair<std::vector<double>, std::vector<double>> source_code_luma(const PlanarBuffer& rgb, std::string_view primaries) {
    const double* k = primaries == "rec2020" ? kLuma2020 : kLuma709;
    std::vector<double> luma(rgb.plane_size()), peak(rgb.plane_size());
    const float *r = rgb.plane(0), *g = rgb.plane(1), *b = rgb.plane(2);
    for (std::size_t i = 0; i < luma.size(); ++i) {
        luma[i] = double(r[i]) * k[0] + double(g[i]) * k[1] + double(b[i]) * k[2];
        peak[i] = std::max({double(r[i]), double(g[i]), double(b[i])});
    }
    return {std::move(luma), std::move(peak)};
}

std::vector<double> hdr_luminance(const PlanarBuffer& hdr) {
    std::vector<double> out(hdr.plane_size(), 0.0);
    for (int c = 0; c < 3; ++c) {
        const float* p = hdr.plane(c);
        for (std::size_t i = 0; i < out.size(); ++i) out[i] += static_cast<double>(p[i]) * kLuma2020[c];
    }
    for (double& v : out) v *= 100.0;
    return out;
}

std::vector<int> video_tile_starts(int length, int size, int overlap) {
    if (length <= size) return {0};
    std::vector<int> v;
    for (int s = 0; s <= length - size; s += size - overlap) v.push_back(s);
    if (v.back() != length - size) v.push_back(length - size);
    return v;
}

std::vector<float> video_tile_blend(int y, int x, int th, int tw, int h, int w, int overlap) {
    std::vector<float> wy(std::size_t(th), 1.0f), wx(std::size_t(tw), 1.0f);
    const int fy = std::min(overlap, th / 2), fx = std::min(overlap, tw / 2);
    if (y && fy) {
        const auto r = torch_linspace(0.001f, 1.0f, fy);
        std::copy(r.begin(), r.end(), wy.begin());
    }
    if (y + th < h && fy) {
        const auto r = torch_linspace(1.0f, 0.001f, fy);
        std::copy(r.begin(), r.end(), wy.end() - fy);
    }
    if (x && fx) {
        const auto r = torch_linspace(0.001f, 1.0f, fx);
        std::copy(r.begin(), r.end(), wx.begin());
    }
    if (x + tw < w && fx) {
        const auto r = torch_linspace(1.0f, 0.001f, fx);
        std::copy(r.begin(), r.end(), wx.end() - fx);
    }
    std::vector<float> blend(std::size_t(th) * std::size_t(tw));
    for (int r = 0; r < th; ++r)
        for (int c = 0; c < tw; ++c) blend[std::size_t(r) * std::size_t(tw) + std::size_t(c)] = wy[std::size_t(r)] * wx[std::size_t(c)];
    return blend;
}

}  // namespace rudra
