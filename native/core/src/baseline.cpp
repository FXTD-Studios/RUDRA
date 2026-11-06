#include "rudra/core/baseline.hpp"

#include <algorithm>
#include <cmath>

#include "parallel.hpp"

namespace rudra {
namespace {
// rudra/radiometry.py
constexpr float kA = 2.51f, kB = 0.03f, kC = 2.43f, kD = 0.59f, kE = 0.14f;
}  // namespace

float srgb_to_linear(float x) noexcept {
    x = std::clamp(x, 0.0f, 1.0f);
    return x <= 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f);
}

float inverse_aces_approx(float display_linear) noexcept {
    // Invert y = x(Ax+B) / (x(Cx+D)+E) for x; the larger root of the quadratic
    // (yC - A)x^2 + (yD - B)x + yE = 0. Clipped values sit just below one.
    const float y = std::clamp(display_linear, 0.0f, 0.995f);
    const float qa = y * kC - kA;
    const float qb = y * kD - kB;
    const float qc = y * kE;
    const float disc = std::max(qb * qb - 4.0f * qa * qc, 0.0f);
    const float denom = std::min(2.0f * qa, -1e-7f);
    const float s = std::sqrt(disc);
    const float root_a = (-qb - s) / denom;
    const float root_b = (-qb + s) / denom;
    return std::max(std::max(root_a, root_b), 0.0f);
}

float baseline_value(float sdr_code, float corpus_ev) noexcept {
    const float scale = static_cast<float>(std::exp2(-static_cast<double>(corpus_ev)) * (203.0 / 10000.0));
    return inverse_aces_approx(srgb_to_linear(sdr_code)) * scale;
}

namespace {
// baseline_value's scale, once per image instead of once per sample (the same
// float: the same expression).
float baseline_scale(float corpus_ev) noexcept {
    return static_cast<float>(std::exp2(-static_cast<double>(corpus_ev)) * (203.0 / 10000.0));
}
}  // namespace

// Every sample on its own (pow, sqrt, exp2), so the image splits over a few
// threads with the same result to the bit (4 Nov 2026: the viewer builds the
// baseline for every frame it shows, which took a 1600 px frame's whole time).
NetworkLinearImage analytic_baseline(const SdrImage& sdr, float corpus_ev) {
    PlanarBuffer out(3, sdr.height(), sdr.width());
    const auto in = sdr.buffer().span();
    auto o = out.span();
    const float scale = baseline_scale(corpus_ev);
    detail::parallel_chunks(in.size(), detail::worker_count(in.size()), [&](int, std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) o[i] = inverse_aces_approx(srgb_to_linear(in[i])) * scale;
    });
    return NetworkLinearImage(std::move(out));
}

float curve_correction_log2(float sdr_code, std::span<const float> params) noexcept {
    const int knots = static_cast<int>(params.size()) - 1;
    const float pos = std::clamp(sdr_code, 0.0f, 1.0f) * static_cast<float>(knots - 1);
    const int lo = std::min(static_cast<int>(std::floor(pos)), knots - 2);
    const float frac = pos - static_cast<float>(lo);
    const float v0 = params[1 + lo];
    const float v1 = params[2 + lo];
    return params[0] + v0 + (v1 - v0) * frac;
}

NetworkLinearImage corrected_baseline(const SdrImage& sdr, float corpus_ev, std::span<const float> params) {
    if (params.size() < 3) return analytic_baseline(sdr, corpus_ev);
    // One pass: the baseline, then its correction, sample by sample (as two passes did).
    PlanarBuffer out(3, sdr.height(), sdr.width());
    const auto in = sdr.buffer().span();
    auto o = out.span();
    const float scale = baseline_scale(corpus_ev);
    detail::parallel_chunks(in.size(), detail::worker_count(in.size()), [&](int, std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) {
            float v = inverse_aces_approx(srgb_to_linear(in[i])) * scale;
            v *= std::exp2(curve_correction_log2(in[i], params));
            o[i] = v;
        }
    });
    return NetworkLinearImage(std::move(out));
}

}  // namespace rudra
