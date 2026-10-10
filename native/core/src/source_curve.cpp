#include "rudra/core/source_curve.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include "rudra/core/baseline.hpp"

namespace rudra {
namespace {

constexpr std::array<SourceCurve, 6> kAll = {SourceCurve::Unknown, SourceCurve::Aces,    SourceCurve::Hable,
                                             SourceCurve::Agx,     SourceCurve::CameraLog, SourceCurve::Clip};

// pipeline/sdr_render.py, float32 as numpy runs it; the constants are its.
float forward_aces(float x) noexcept {
    return std::clamp((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f), 0.0f, 1.0f);
}

float forward_hable(float x) noexcept {
    constexpr float a = 0.15f, b = 0.50f, c = 0.10f, d = 0.20f, e = 0.02f, f = 0.30f;
    const auto h = [](float v) { return (v * (a * v + c * b) + d * e) / (v * (a * v + b) + d * f) - e / f; };
    return std::clamp(h(x * 2.0f) / h(11.2f), 0.0f, 1.0f);
}

float forward_agx(float x) noexcept {
    constexpr float lo = -12.47f, hi = 4.03f, contrast = 1.0f;
    float t = (std::log2(std::max(x, std::exp2(lo))) - lo) / (hi - lo);
    t = std::clamp(t, 0.0f, 1.0f);
    const float s = 1.0f / (1.0f + std::exp(-(t - 0.6f) * 10.0f * contrast));
    const float s0 = 1.0f / (1.0f + std::exp(0.6f * 10.0f * contrast));
    const float s1 = 1.0f / (1.0f + std::exp(-0.4f * 10.0f * contrast));
    return std::pow(std::clamp((s - s0) / (s1 - s0), 0.0f, 1.0f), 2.2f);
}

float forward_camera_log(float x) noexcept {
    constexpr float contrast = 1.0f;
    float lg = std::log2(std::max(x, 1e-6f) / 0.18f) / 14.0f + 0.5f;
    lg = std::clamp(lg, 0.0f, 1.0f);
    const float k = 7.0f * contrast;
    const float s = 1.0f / (1.0f + std::exp(-(lg - 0.5f) * k));
    const float s0 = 1.0f / (1.0f + std::exp(0.5f * k));
    const float s1 = 1.0f / (1.0f + std::exp(-0.5f * k));
    return std::pow(std::clamp((s - s0) / (s1 - s0), 0.0f, 1.0f), 2.4f);
}

float forward_clip(float x) noexcept { return std::clamp(x, 0.0f, 1.0f); }

// A dense monotone table of (display, scene) over 22 stops of exposure, read
// by bisection and linear interpolation. Every curve above is non-decreasing.
struct InverseTable {
    static constexpr int kN = 8001;
    static constexpr double kLo = -16.0, kHi = 6.0;
    std::array<float, kN> x{}, y{};

    explicit InverseTable(SourceCurve c) {
        for (int i = 0; i < kN; ++i) {
            x[i] = static_cast<float>(std::exp2(kLo + (kHi - kLo) * i / (kN - 1)));
            y[i] = source_curve_forward(c, x[i]);
        }
        for (int i = 1; i < kN; ++i) y[i] = std::max(y[i], y[i - 1]);   // float noise never breaks the order
    }

    float at(float display) const noexcept {
        const float v = std::clamp(display, 0.0f, 0.995f);
        const auto it = std::lower_bound(y.begin(), y.end(), v);
        if (it == y.begin()) return 0.0f;
        const auto i = static_cast<int>(it - y.begin());
        if (it == y.end()) return x[kN - 1];
        const float y0 = y[i - 1], y1 = y[i];
        const float t = y1 > y0 ? (v - y0) / (y1 - y0) : 0.0f;
        return x[i - 1] + (x[i] - x[i - 1]) * t;
    }
};

const InverseTable& table(SourceCurve c) {
    static const InverseTable hable(SourceCurve::Hable), agx(SourceCurve::Agx), camera(SourceCurve::CameraLog),
        clip(SourceCurve::Clip);
    switch (c) {
        case SourceCurve::Hable: return hable;
        case SourceCurve::Agx: return agx;
        case SourceCurve::CameraLog: return camera;
        default: return clip;
    }
}

std::vector<float> build_params(SourceCurve c) {
    std::vector<float> p(1 + kSourceCurveKnots, 0.0f);
    for (int code = 0; code < kSourceCurveKnots; ++code) {
        const float display = srgb_to_linear(static_cast<float>(code) / 255.0f);
        const float want = source_curve_inverse(c, display);
        const float have = inverse_aces_approx(display);
        // Both inverses are 0 at code 0; the ratio there is a convention, and
        // the one that keeps the knot line flat into black is the code-1 value.
        p[1 + code] = (want > 0.0f && have > 0.0f) ? std::log2(want / have) : 0.0f;
    }
    p[1] = p[2];
    return p;
}

}  // namespace

std::string_view source_curve_id(SourceCurve c) noexcept {
    switch (c) {
        case SourceCurve::Aces: return "aces";
        case SourceCurve::Hable: return "hable";
        case SourceCurve::Agx: return "agx";
        case SourceCurve::CameraLog: return "camera_log";
        case SourceCurve::Clip: return "clip";
        default: return "unknown";
    }
}

std::optional<SourceCurve> parse_source_curve(std::string_view id) noexcept {
    for (SourceCurve c : kAll)
        if (source_curve_id(c) == id) return c;
    return std::nullopt;
}

std::span<const SourceCurve> all_source_curves() noexcept { return kAll; }

std::string_view source_curve_label(SourceCurve c) noexcept {
    switch (c) {
        case SourceCurve::Aces: return "ACES";
        case SourceCurve::Hable: return "Filmic";
        case SourceCurve::Agx: return "AgX";
        case SourceCurve::CameraLog: return "Camera log";
        case SourceCurve::Clip: return "Rec.709";
        default: return "Unknown";
    }
}

std::string_view source_curve_detail(SourceCurve c) noexcept {
    switch (c) {
        case SourceCurve::Aces: return "Narkowicz";
        case SourceCurve::Hable: return "Hable";
        case SourceCurve::Agx: return "sigmoid";
        case SourceCurve::CameraLog: return "log to 709";
        case SourceCurve::Clip: return "plain clip";
        default: return "ACES default";
    }
}

float source_curve_forward(SourceCurve c, float x) noexcept {
    switch (c) {
        case SourceCurve::Hable: return forward_hable(x);
        case SourceCurve::Agx: return forward_agx(x);
        case SourceCurve::CameraLog: return forward_camera_log(x);
        case SourceCurve::Clip: return forward_clip(x);
        default: return forward_aces(x);
    }
}

float source_curve_inverse(SourceCurve c, float display) noexcept {
    switch (c) {
        case SourceCurve::Unknown:
        case SourceCurve::Aces: return inverse_aces_approx(display);
        default: return table(c).at(display);
    }
}

std::span<const float> source_curve_params(SourceCurve c) {
    static const std::vector<float> none;
    static const std::vector<float> hable = build_params(SourceCurve::Hable), agx = build_params(SourceCurve::Agx),
                                    camera = build_params(SourceCurve::CameraLog), clip = build_params(SourceCurve::Clip);
    switch (c) {
        case SourceCurve::Hable: return hable;
        case SourceCurve::Agx: return agx;
        case SourceCurve::CameraLog: return camera;
        case SourceCurve::Clip: return clip;
        default: return none;
    }
}

std::vector<float> effective_curve_params(std::span<const float> model_params, SourceCurve source) {
    const std::span<const float> src = source_curve_params(source);
    if (src.empty()) return std::vector<float>(model_params.begin(), model_params.end());
    std::vector<float> out(src.begin(), src.end());
    if (model_params.size() < 3) return out;   // no CurveHead: the source alone
    // The model's knots, resampled at the source's codes, add in log2.
    out[0] += model_params[0];
    const int knots = kSourceCurveKnots;
    for (int i = 0; i < knots; ++i)
        out[1 + i] += curve_correction_log2(static_cast<float>(i) / (knots - 1), model_params) - model_params[0];
    return out;
}

}  // namespace rudra
