#include "rudra/core/log_encode.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "rudra/core/gamut.hpp"

namespace rudra {
namespace {

// ACEScct, S-2016-001 section 4.
constexpr double kCctXBrk = 0.0078125;
constexpr double kCctYBrk = 0.155251141552511;
constexpr double kCctA = 10.5402377416545;
constexpr double kCctB = 0.0729055341958355;

// ARRI LogC4, specification section 3.
const double kL4a = (std::pow(2.0, 18.0) - 16.0) / 117.45;
constexpr double kL4b = (1023.0 - 95.0) / 1023.0;
constexpr double kL4c = 95.0 / 1023.0;
const double kL4s = (7.0 * std::log(2.0) * std::pow(2.0, 7.0 - 14.0 * kL4c / kL4b)) / (kL4a * kL4b);
const double kL4t = (std::pow(2.0, 14.0 * (-kL4c / kL4b) + 6.0) - 64.0) / kL4a;

}  // namespace

double acescct_encode(double x) noexcept {
    return x <= kCctXBrk ? kCctA * x + kCctB : (std::log2(x) + 9.72) / 17.52;
}

double acescct_decode(double y) noexcept {
    return y <= kCctYBrk ? (y - kCctB) / kCctA : std::exp2(y * 17.52 - 9.72);
}

double logc4_encode(double e) noexcept {
    return e >= kL4t ? (std::log2(kL4a * e + 64.0) - 6.0) / 14.0 * kL4b + kL4c : (e - kL4t) / kL4s;
}

double logc4_decode(double p) noexcept {
    return p >= 0.0 ? (std::exp2(14.0 * (p - kL4c) / kL4b + 6.0) - 64.0) / kL4a : p * kL4s + kL4t;
}

const char* log_curve_name(LogCurve c) noexcept { return c == LogCurve::AcesCct ? "acescct" : "logc4"; }
const char* log_curve_label(LogCurve c) noexcept { return c == LogCurve::AcesCct ? "ACEScct" : "ARRI LogC4"; }
Primaries log_curve_primaries(LogCurve c) noexcept { return c == LogCurve::AcesCct ? Primaries::Ap1 : Primaries::Awg4; }
const char* log_gamut_label(LogCurve c) noexcept { return c == LogCurve::AcesCct ? "ACES AP1" : "ARRI Wide Gamut 4"; }

std::optional<LogCurve> log_curve_from_name(std::string_view n) noexcept {
    if (n == "acescct") return LogCurve::AcesCct;
    if (n == "logc4") return LogCurve::LogC4;
    return std::nullopt;
}

PlanarBuffer encode_log(const PlanarBuffer& in, LogCurve curve) {
    assert(in.channels() == 3);
    const Mat3 m = rgb_to_rgb_matrix(Primaries::Rec2020, log_curve_primaries(curve));
    const double to_scene = double(kPqPeak.v) / double(kDiffuseWhite.v);   // network units -> diffuse white = 1.0
    PlanarBuffer out(3, in.height(), in.width());
    const std::size_t n = in.plane_size();
    for (std::size_t i = 0; i < n; ++i) {
        const double v[3] = {in.plane(0)[i] * to_scene, in.plane(1)[i] * to_scene, in.plane(2)[i] * to_scene};
        for (int c = 0; c < 3; ++c) {
            const double lin = m[c][0] * v[0] + m[c][1] * v[1] + m[c][2] * v[2];
            const double code = curve == LogCurve::AcesCct ? acescct_encode(lin) : logc4_encode(lin);
            out.plane(c)[i] = static_cast<float>(std::clamp(code, 0.0, 1.0));
        }
    }
    return out;
}

}  // namespace rudra
