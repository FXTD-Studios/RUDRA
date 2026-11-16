#include "rudra/core/view_ocio.hpp"

#include "rudra/core/compare.hpp"
#include "rudra/core/gamut.hpp"
#include "rudra/core/view.hpp"

#include <algorithm>
#include <cmath>

namespace rudra {
namespace {

constexpr double kUnitsToWhite = 10000.0 / 203.0;   // viewer units (1.0 = 10 000 nits) to 1.0 = 203 nits

Mat3 inverse(const Mat3& m) {
    const double a = m[0][0], b = m[0][1], c = m[0][2];
    const double d = m[1][0], e = m[1][1], f = m[1][2];
    const double g = m[2][0], h = m[2][1], i = m[2][2];
    const double A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const double det = a * A + b * B + c * C;
    Mat3 r{};
    r[0] = {A / det, -(b * i - c * h) / det, (b * f - c * e) / det};
    r[1] = {B / det, (a * i - c * g) / det, -(a * f - c * d) / det};
    r[2] = {C / det, -(a * h - b * g) / det, (a * e - b * d) / det};
    return r;
}

std::array<float, 9> rows(const Mat3& m, double scale = 1.0) {
    std::array<float, 9> out{};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) out[std::size_t(r * 3 + c)] = float(m[std::size_t(r)][std::size_t(c)] * scale);
    return out;
}

// Row by row in the order the display pass sums them.
inline void mul(const std::array<float, 9>& m, const float h[3], float out[3]) noexcept {
    for (int k = 0; k < 3; ++k)
        out[k] = m[std::size_t(k) * 3] * h[0] + m[std::size_t(k) * 3 + 1] * h[1] + m[std::size_t(k) * 3 + 2] * h[2];
}

inline bool below_floor(const float v[3]) noexcept {
    const float mx = std::max(std::max(v[0], v[1]), v[2]);
    const float mn = std::min(std::min(v[0], v[1]), v[2]);
    return mn < -(kCheckFloor + kCheckRelative * std::max(mx, 0.0f));
}

}  // namespace

ViewMatrices view_matrices(Primaries source, WorkingSpace working, Primaries gamut, double nits_per_y) {
    ViewMatrices m;
    m.work = rows(rgb_to_rgb_matrix(source, working_primaries(working)), kUnitsToWhite);
    m.dec = rows(inverse(normalized_primary_matrix(Primaries::Rec709)), nits_per_y);
    m.chk = rows(rgb_to_rgb_matrix(source, gamut));
    m.ap0 = rows(rgb_to_rgb_matrix(source, Primaries::Ap0));
    return m;
}

bool out_of_gamut(const float h[3], const std::array<float, 9>& chk) noexcept {
    float g[3];
    mul(chk, h, g);
    return below_floor(g);
}

bool invalid_value(const float h[3], const std::array<float, 9>& work, const std::array<float, 9>& ap0) noexcept {
    if (!std::isfinite(h[0]) || !std::isfinite(h[1]) || !std::isfinite(h[2])) return true;
    float a[3], w[3];
    mul(ap0, h, a);
    if (below_floor(a)) return true;
    mul(work, h, w);
    return std::max(std::max(std::abs(w[0]), std::abs(w[1])), std::abs(w[2])) > kHalfMax;
}

PixelReading read_pixel(const float rgb[3], Primaries source, WorkingSpace working, Primaries gamut,
                        const OcioViewTransform* view) {
    PixelReading r;
    const ViewMatrices m = view_matrices(source, working, gamut, 203.0);
    const Mat3 wm = rgb_to_rgb_matrix(source, working_primaries(working));
    for (int k = 0; k < 3; ++k)
        r.working[std::size_t(k)] = (wm[std::size_t(k)][0] * rgb[0] + wm[std::size_t(k)][1] * rgb[1] +
                                     wm[std::size_t(k)][2] * rgb[2]) * kUnitsToWhite;
    r.out_of_gamut = std::isfinite(rgb[0]) && std::isfinite(rgb[1]) && std::isfinite(rgb[2]) && out_of_gamut(rgb, m.chk);
    if (!std::isfinite(rgb[0]) || !std::isfinite(rgb[1]) || !std::isfinite(rgb[2])) {
        r.invalid = true;
        r.why = (std::isnan(rgb[0]) || std::isnan(rgb[1]) || std::isnan(rgb[2])) ? "NaN" : "infinite";
    } else {
        float a[3], w[3];
        mul(m.ap0, rgb, a);
        mul(m.work, rgb, w);
        if (below_floor(a)) {
            r.invalid = true;
            r.why = "outside AP0";
        } else if (std::max(std::max(std::abs(w[0]), std::abs(w[1])), std::abs(w[2])) > kHalfMax) {
            r.invalid = true;
            r.why = "above half float";
        }
    }
    if (view && view->forward.cpu && view->working == working && !r.invalid) {
        float px[3];
        mul(m.work, rgb, px);
        view->forward.cpu(px, 1);
        for (int k = 0; k < 3; ++k) r.code[std::size_t(k)] = px[k];
        r.has_code = true;
    }
    return r;
}

std::vector<float> sdr_light_through(const SdrImage& sdr, const OcioInputTransform& t, Primaries source) {
    const PlanarBuffer& c = sdr.buffer();
    const std::size_t n = c.plane_size();
    std::vector<float> packed(n * 3);
    for (std::size_t i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) packed[i * 3 + std::size_t(k)] = c.plane(k)[i];
    if (t.cpu) t.cpu(packed.data(), n);
    const Mat3 m = rgb_to_rgb_matrix(working_primaries(t.working), source);
    const std::array<float, 9> back = rows(m, 1.0 / kUnitsToWhite);
    std::vector<float> out(n * 4);
    for (std::size_t i = 0; i < n; ++i) {
        float v[3] = {packed[i * 3], packed[i * 3 + 1], packed[i * 3 + 2]}, o[3];
        mul(back, v, o);
        int flags = 0;
        float mx = 0.0f;
        for (int k = 0; k < 3; ++k) {
            const float code = c.plane(k)[i];
            out[i * 4 + std::size_t(k)] = o[k];
            if (code >= kSdrClipCode) flags |= (1 << k);
            mx = std::max(mx, code);
        }
        if (mx <= kSdrCrushCode) flags |= kFlagCrush;
        out[i * 4 + 3] = float(flags);
    }
    return out;
}

std::string ocio_view_text(const OcioViewTransform& v) {
    return "OCIO · " + v.display + " · " + v.view + " · " + working_name(v.working);
}

}  // namespace rudra
