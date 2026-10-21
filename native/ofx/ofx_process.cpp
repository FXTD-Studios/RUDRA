#include "ofx_process.hpp"

#include <algorithm>
#include <cmath>

#include "rudra/core/gamut.hpp"
#include "rudra/core/hdr10.hpp"
#include "rudra/core/source_curve.hpp"
#include "rudra/deliver/master.hpp"

namespace rudra {

Result<void> render_sdr_to_hdr(const OfxGrade& g, const float* src, std::ptrdiff_t src_row, float* dst,
                               std::ptrdiff_t dst_row, int width, int height) {
    if (width <= 0 || height <= 0 || !src || !dst) return make_error(ErrorCode::InvalidArgument, "No picture to render");
    if (!parse_source_curve(g.source_curve)) return make_error(ErrorCode::InvalidArgument, "Unknown source curve", g.source_curve);
    if (g.output != "linear_rec2020" && g.output != "linear_acescg" && g.output != "pq_rec2020")
        return make_error(ErrorCode::InvalidArgument, "Unknown output", g.output);
    if (!(g.exposure_ev >= -3.0 && g.exposure_ev <= 3.0) || !(g.strength >= 0.0 && g.strength <= 1.0))
        return make_error(ErrorCode::InvalidArgument, "Exposure or strength out of range");
    const std::size_t n = std::size_t(width) * std::size_t(height);
    PlanarBuffer code(3, height, width);
    for (int y = 0; y < height; ++y) {
        const float* row = src + std::ptrdiff_t(y) * src_row;
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < 3; ++c) {
                const float v = row[std::ptrdiff_t(x) * 4 + c];
                code.plane(c)[std::size_t(y) * std::size_t(width) + std::size_t(x)] =
                    std::isfinite(v) ? std::clamp(v, 0.0f, 1.0f) : 0.0f;
            }
    }
    const SdrImage sdr(std::move(code));
    MasterRequest q;
    q.analytic = true;
    q.source_curve = g.source_curve;
    q.exposure_ev = g.exposure_ev;
    q.strength = g.strength;
    q.recovery_mode = g.recovery_mode;
    q.clip_fill = g.clip_fill;
    q.anchor = g.anchor;
    q.source_space = "rec709";
    q.container = "linear";
    auto m = grade_master(sdr, analytic_fields_for(sdr, q), FrameScalars{}, kAnalyticConstants, q);
    if (!m) return m.error();
    // Rec.709 nits to the output.
    PlanarBuffer nits(3, height, width);
    for (int c = 0; c < 3; ++c)
        for (std::size_t i = 0; i < n; ++i) nits.plane(c)[i] = float(m->nits.plane(c)[i]);
    const Primaries target = g.output == "linear_acescg" ? Primaries::Ap1 : Primaries::Rec2020;
    const PlanarBuffer out = convert_primaries(nits, Primaries::Rec709, target);
    const bool pq = g.output == "pq_rec2020";
    for (int y = 0; y < height; ++y) {
        float* row = dst + std::ptrdiff_t(y) * dst_row;
        const float* alpha = src + std::ptrdiff_t(y) * src_row;
        for (int x = 0; x < width; ++x) {
            const std::size_t i = std::size_t(y) * std::size_t(width) + std::size_t(x);
            for (int c = 0; c < 3; ++c) {
                const float v = std::max(out.plane(c)[i], 0.0f);
                row[std::ptrdiff_t(x) * 4 + c] = pq ? pq_oetf(v) : v / 203.0f;
            }
            row[std::ptrdiff_t(x) * 4 + 3] = alpha[std::ptrdiff_t(x) * 4 + 3];
        }
    }
    return {};
}

}  // namespace rudra

namespace rudra_ofx {

std::vector<std::string> source_ids() {
    std::vector<std::string> v;
    for (auto c : rudra::all_source_curves()) v.emplace_back(rudra::source_curve_id(c));
    return v;
}

std::vector<std::string> source_labels() {
    std::vector<std::string> v;
    for (auto c : rudra::all_source_curves()) v.emplace_back(rudra::source_curve_label(c));
    return v;
}

int default_source() {
    int i = 0;
    for (auto c : rudra::all_source_curves()) {
        if (c == rudra::SourceCurve::Bt1886) return i;
        ++i;
    }
    return 0;
}

std::string render(const std::string& source_curve, double exposure_ev, double strength, const std::string& recovery_mode,
                   bool clip_fill, bool anchor, const std::string& output, const float* src, std::ptrdiff_t src_row,
                   float* dst, std::ptrdiff_t dst_row, int width, int height) {
    rudra::OfxGrade g;
    g.source_curve = source_curve, g.exposure_ev = exposure_ev, g.strength = strength;
    g.recovery_mode = recovery_mode, g.clip_fill = clip_fill, g.anchor = anchor, g.output = output;
    auto r = rudra::render_sdr_to_hdr(g, src, src_row, dst, dst_row, width, height);
    return r ? std::string() : r.error().message;
}

}  // namespace rudra_ofx
