#include "rudra/core/calibration.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>

#include "rudra/core/baseline.hpp"

namespace rudra {
namespace {

// Usable anchors, one per code, by code.
std::map<int, double> anchors_of(std::span<const CalibrationPoint> points) {
    std::map<int, double> out;
    for (const auto& p : points)
        if (p.code >= 0 && p.code < kSourceCurveKnots && p.nits > 0.0 && std::isfinite(p.nits)) out[p.code] = p.nits;
    return out;
}

}  // namespace

double source_code_nits(SourceCurve source, int code, float corpus_ev) noexcept {
    const float display = srgb_to_linear(float(code) / 255.0f);
    const double scene = source_curve_inverse(source, display);
    return scene * std::exp2(-double(corpus_ev)) * 203.0;
}

std::vector<float> calibration_params(std::span<const CalibrationPoint> points, SourceCurve source, float corpus_ev) {
    const auto anchors = anchors_of(points);
    if (anchors.empty()) return {};
    // log2(wanted / what the source inverse gives) at each anchor, then the
    // knot line through them.
    std::vector<std::pair<int, double>> pts;
    for (const auto& [code, nits] : anchors) {
        const double have = source_code_nits(source, code, corpus_ev);
        pts.emplace_back(code, have > 0.0 ? std::log2(nits / have) : 0.0);
    }
    std::vector<float> out(1 + kSourceCurveKnots, 0.0f);
    for (int code = 0; code < kSourceCurveKnots; ++code) {
        double d;
        if (code <= pts.front().first) d = pts.front().second;
        else if (code >= pts.back().first) d = pts.back().second;
        else {
            std::size_t i = 1;
            while (pts[i].first < code) ++i;
            const auto& [c0, d0] = pts[i - 1];
            const auto& [c1, d1] = pts[i];
            d = d0 + (d1 - d0) * double(code - c0) / double(c1 - c0);
        }
        out[1 + code] = float(d);
    }
    return out;
}

bool calibration_is_monotone(std::span<const CalibrationPoint> points, SourceCurve source, float corpus_ev) {
    const auto params = calibration_params(points, source, corpus_ev);
    if (params.empty()) return true;
    double prev = -1.0;
    for (int code = 1; code < kSourceCurveKnots; ++code) {   // code 0 is black under every curve
        const double n = source_code_nits(source, code, corpus_ev) * std::exp2(double(params[1 + code]));
        if (n < prev) return false;
        prev = n;
    }
    return true;
}

std::string calibration_summary(std::span<const CalibrationPoint> points, SourceCurve source, float corpus_ev) {
    const auto anchors = anchors_of(points);
    if (anchors.empty()) return "";
    if (!calibration_is_monotone(points, source, corpus_ev))
        return "These anchors do not describe a monotone curve. Not applied.";
    if (anchors.size() == 1) {
        const auto& [code, nits] = *anchors.begin();
        const double st = std::log2(nits / std::max(source_code_nits(source, code, corpus_ev), 1e-9));
        char b[96];
        std::snprintf(b, sizeof b, "1 anchor: exposure %+.2f stops to match it.", st);
        return b;
    }
    return std::to_string(anchors.size()) + " anchors: piecewise-linear in log2 between them, flat outside.";
}

}  // namespace rudra
