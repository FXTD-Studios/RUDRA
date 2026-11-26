#include "ofx_process.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <random>

#include "rudra/core/gamut.hpp"
#include "rudra/core/hdr10.hpp"
#include "rudra/core/source_curve.hpp"
#include "rudra/deliver/master.hpp"
#include "engine_client.hpp"
#include "fill_store.hpp"

namespace rudra {

namespace {

constexpr double kDiffuseWhiteNits = 203.0;

// The host's RGBA rows as the SDR's code planes, top row first: clamped to [0, 1], NaN as 0.
SdrImage codes_of(const float* src, std::ptrdiff_t src_row, int width, int height) {
    PlanarBuffer code(3, height, width);
    for (int y = 0; y < height; ++y) {
        const float* row = src + std::ptrdiff_t(y) * src_row;
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < 3; ++c) {
                const float v = row[std::ptrdiff_t(x) * 4 + c];
                code.plane(c)[std::size_t(y) * std::size_t(width) + std::size_t(x)] = std::isfinite(v) ? std::clamp(v, 0.0f, 1.0f) : 0.0f;
            }
    }
    return SdrImage(std::move(code));
}

// The master at 0 EV, in nits: Rec.709 primaries on the analytic route (double, as grade_master leaves
// them), ACEScg from the learned engine or a stored frame.
struct Master {
    Primaries primaries = Primaries::Rec709;
    std::vector<double> nits;   // 3 planes of width x height
};

Master master_from_ap1(const PlanarBuffer& ap1) {
    Master m{Primaries::Ap1, std::vector<double>(ap1.span().size())};
    const auto s = ap1.span();
    for (std::size_t i = 0; i < s.size(); ++i) m.nits[i] = std::max(double(s[i]), 0.0) * kDiffuseWhiteNits;
    return m;
}

// The master as ACEScg planes, 1.0 = diffuse white: what AI fill stores and draws on.
PlanarBuffer master_ap1(const Master& m, int width, int height) {
    PlanarBuffer p(3, height, width);
    auto o = p.span();
    for (std::size_t i = 0; i < o.size(); ++i) o[i] = float(m.nits[i] / kDiffuseWhiteNits);
    return m.primaries == Primaries::Ap1 ? p : convert_primaries(p, m.primaries, Primaries::Ap1);
}

Result<void> check_grade(const OfxGrade& g) {
    if (g.route != "analytic" && g.route != "learned") return make_error(ErrorCode::InvalidArgument, "Unknown route", g.route);
    if (!parse_source_curve(g.source_curve)) return make_error(ErrorCode::InvalidArgument, "Unknown source curve", g.source_curve);
    if (g.output != "linear_rec2020" && g.output != "linear_acescg" && g.output != "pq_rec2020")
        return make_error(ErrorCode::InvalidArgument, "Unknown output", g.output);
    if (!(g.exposure_ev >= -3.0 && g.exposure_ev <= 3.0) || !(g.strength >= 0.0 && g.strength <= 1.0))
        return make_error(ErrorCode::InvalidArgument, "Exposure or strength out of range");
    if (g.alpha != "source" && g.alpha != "limit" && g.alpha != "ai_fill") return make_error(ErrorCode::InvalidArgument, "Unknown alpha", g.alpha);
    if (const std::string why = reconstruction_region_error(g.region); g.region.enabled && !why.empty())
        return make_error(ErrorCode::InvalidArgument, "The luminance limit: " + why);
    return {};
}

std::string region_json(const OfxGrade& g) { return g.region.enabled ? reconstruction_region_json(g.region) : std::string(); }

Result<Master> make_master(const OfxGrade& g, const SdrImage& sdr) {
    if (g.route == "learned") {
        // The learned ACES 2.0 master (ACEScg, 1.0 = diffuse white), the luminance limit applied by the engine.
        auto planes = ofx_engine::learned_master(sdr, g.strength, region_json(g));
        if (!planes) return planes.error();
        return master_from_ap1(*planes);
    }
    MasterRequest q;
    q.analytic = true;
    q.source_curve = g.source_curve;
    q.exposure_ev = 0.0;   // applied after (write_output): a stored fill takes it the same way
    q.strength = g.strength;
    q.recovery_mode = g.recovery_mode;
    q.clip_fill = g.clip_fill;
    q.anchor = g.anchor;
    q.region = g.region;
    // The host hands over the timeline's own code values, as the app's movie frames:
    // BT.1886 reads them as video codes (code^2.4), as RUDRA 1.3.0 does (25 Nov 2026).
    q.own_codes = true;
    q.source_space = "rec709";
    q.container = "linear";
    auto m = grade_master(sdr, analytic_fields_for(sdr, q), FrameScalars{}, kAnalyticConstants, q);
    if (!m) return m.error();
    Master out{Primaries::Rec709, std::vector<double>(m->nits.span().begin(), m->nits.span().end())};
    return out;
}

// The luminance limit's weight per pixel (1 where the reconstruction acts in full, 0 where it does not):
// the app's region tint, max(highlight, shadow) at the SDR's Rec.709 luma.
PlanarBuffer limit_plane(const SdrImage& sdr, const ReconstructionRegion& r) {
    const int w = sdr.width(), h = sdr.height();
    PlanarBuffer a(1, h, w, 1.0f);
    if (!r.enabled) return a;
    const PlanarBuffer& s = sdr.buffer();
    for (std::size_t i = 0; i < a.plane_size(); ++i) {
        const float y = 0.2126f * s.plane(0)[i] + 0.7152f * s.plane(1)[i] + 0.0722f * s.plane(2)[i];
        float wh = 1.0f, ws = 1.0f;
        region_weights(y, r, wh, ws);
        a.plane(0)[i] = std::clamp(std::max(wh, ws), 0.0f, 1.0f);
    }
    return a;
}

// The master at the exposure to the host: PQ code, or linear with 1.0 = 203 nits, in the output's primaries.
// Alpha: the plane given, else the source's.
void write_output(const Master& m, double exposure_ev, const std::string& output, const float* src, std::ptrdiff_t src_row,
                  float* dst, std::ptrdiff_t dst_row, int width, int height, const PlanarBuffer* alpha) {
    const double gain = std::exp2(exposure_ev);
    const std::size_t n = std::size_t(width) * std::size_t(height);
    PlanarBuffer nits(3, height, width);
    for (int c = 0; c < 3; ++c)
        for (std::size_t i = 0; i < n; ++i) nits.plane(c)[i] = float(m.nits[std::size_t(c) * n + i] * gain);
    const Primaries target = output == "linear_acescg" ? Primaries::Ap1 : Primaries::Rec2020;
    const PlanarBuffer out = m.primaries == target ? std::move(nits) : convert_primaries(nits, m.primaries, target);
    const bool pq = output == "pq_rec2020";
    for (int y = 0; y < height; ++y) {
        float* row = dst + std::ptrdiff_t(y) * dst_row;
        const float* from = src + std::ptrdiff_t(y) * src_row;
        for (int x = 0; x < width; ++x) {
            const std::size_t i = std::size_t(y) * std::size_t(width) + std::size_t(x);
            for (int c = 0; c < 3; ++c) {
                const float v = std::max(out.plane(c)[i], 0.0f);
                row[std::ptrdiff_t(x) * 4 + c] = pq ? pq_oetf(v) : v / float(kDiffuseWhiteNits);
            }
            row[std::ptrdiff_t(x) * 4 + 3] = alpha ? alpha->plane(0)[i] : from[std::ptrdiff_t(x) * 4 + 3];
        }
    }
}

}  // namespace

ReconstructionRegion ofx_region(const rudra_ofx::Settings& s) {
    ReconstructionRegion r;
    if (s.limit == "off" || s.limit.empty()) return r;
    r.enabled = true;
    const float soft = float(s.limit_softness);
    if (s.limit == "band") {
        r.band = true;
        r.band_low = float(s.band_from), r.band_high = float(s.band_to), r.band_softness = soft;
        r.highlight_strength = r.shadow_strength = 1.0f;
        return r;
    }
    r.highlight_from = float(s.highlights_from), r.highlight_softness = soft;
    r.shadow_below = float(s.shadows_below), r.shadow_softness = soft;
    r.highlight_strength = (s.limit == "highlights" || s.limit == "both") ? 1.0f : 0.0f;
    r.shadow_strength = (s.limit == "shadows" || s.limit == "both") ? 1.0f : 0.0f;
    return r;
}

OfxGrade ofx_grade(const rudra_ofx::Settings& s) {
    OfxGrade g;
    g.route = s.route, g.source_curve = s.source_curve, g.exposure_ev = s.exposure_ev, g.strength = s.strength;
    g.recovery_mode = s.recovery_mode, g.clip_fill = s.clip_fill, g.anchor = s.anchor, g.output = s.output;
    g.region = ofx_region(s);
    g.alpha = s.alpha;
    return g;
}

Result<void> render_sdr_to_hdr(const OfxGrade& g, const float* src, std::ptrdiff_t src_row, float* dst,
                               std::ptrdiff_t dst_row, int width, int height) {
    if (width <= 0 || height <= 0 || !src || !dst) return make_error(ErrorCode::InvalidArgument, "No picture to render");
    if (auto ok = check_grade(g); !ok) return ok.error();
    const SdrImage sdr = codes_of(src, src_row, width, height);
    auto m = make_master(g, sdr);
    if (!m) return m.error();
    std::optional<PlanarBuffer> alpha;
    if (g.alpha == "limit") alpha = limit_plane(sdr, g.region);
    if (g.alpha == "ai_fill") alpha = PlanarBuffer(1, height, width, 0.0f);   // no fill here
    write_output(*m, g.exposure_ev, g.output, src, src_row, dst, dst_row, width, height, alpha ? &*alpha : nullptr);
    return {};
}

}  // namespace rudra

namespace rudra_ofx {

namespace {

// The fill's request: the instance, the clip, and everything that changes the master or the fill.
rudra::ofx_fill::FillRequest fill_request(const Settings& s, const rudra::OfxGrade& g) {
    rudra::ofx_fill::FillRequest r;
    r.shot_id = s.shot_id;
    r.first = s.first_frame, r.last = s.last_frame;
    r.condition = g.route == "learned" ? "aces" : "bt1886";   // the app's: the route's own inverse
    r.prompt = s.prompt;
    r.seed = s.seed;
    r.region_json = g.region.enabled ? rudra::reconstruction_region_json(g.region) : std::string();
    char b[256];
    std::snprintf(b, sizeof b, "%s|%s|%.6f|%s|%d|%d", g.route.c_str(), g.source_curve.c_str(), g.strength, g.recovery_mode.c_str(),
                  int(g.clip_fill), int(g.anchor));
    r.master_key = b;
    return r;
}

}  // namespace

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

Outcome render(const Settings& s, const float* src, std::ptrdiff_t src_row, float* dst, std::ptrdiff_t dst_row, int width, int height) {
    using namespace rudra;
    namespace fill = rudra::ofx_fill;
    Outcome out;
    if (width <= 0 || height <= 0 || !src || !dst) return {"No picture to render", {}};
    const OfxGrade g = ofx_grade(s);
    if (auto ok = check_grade(g); !ok) return {ok.error().message, {}};
    if (!s.ai_fill) {
        auto r = render_sdr_to_hdr(g, src, src_row, dst, dst_row, width, height);
        return {r ? std::string() : r.error().message, {}};
    }

    // AI fill (phase 3): the stored fill when it is done, else the master, stored while collecting.
    const SdrImage sdr = codes_of(src, src_row, width, height);
    const fill::FillRequest req = fill_request(s, g);
    const fill::Status st = fill::status(req);
    std::optional<PlanarBuffer> alpha;
    auto show = [&](const Master& m, const PlanarBuffer* generated) {
        if (g.alpha == "limit") alpha = limit_plane(sdr, g.region);
        if (g.alpha == "ai_fill") alpha = generated ? *generated : PlanarBuffer(1, height, width, 0.0f);
        write_output(m, g.exposure_ev, g.output, src, src_row, dst, dst_row, width, height, alpha ? &*alpha : nullptr);
    };
    if (st.state == fill::State::Done) {
        std::string why = "shown at full size only (this render is a proxy)";
        if (s.full_size) {
            if (auto f = fill::fill_frame(req, s.frame, sdr, &why)) {
                show(master_from_ap1(f->ap1), &f->generated);
                out.note = fill::status_text(st);
                return out;
            }
        }
        out.note = "AI fill: " + why + ". The master is shown.";
    } else if (st.state == fill::State::Running && s.full_size) {
        // The stored master while the fill draws: the learned engine stays off the GPU the fill is using.
        if (auto stored = fill::master_frame(req, s.frame, sdr)) {
            show(master_from_ap1(*stored), nullptr);
            out.note = fill::status_text(st);
            return out;
        }
    }
    auto m = make_master(g, sdr);
    if (!m) return {m.error().message, {}};
    if (st.state == fill::State::Collecting) {
        if (!s.full_size) {
            out.note = "AI fill: frames are collected from full-size renders only (turn the proxy off).";
        } else if (s.frame < req.first || s.frame > req.last) {
            // The clip's range is the host's (Settings::first_frame/last_frame): said, so a host whose frame
            // numbers do not match it shows why nothing is collected (26 Nov 2026).
            out.note = "AI fill: frame " + std::to_string(s.frame) + " is outside the clip (" + std::to_string(req.first) + " to " +
                       std::to_string(req.last) + "), so it is not stored.";
        } else if (auto last = fill::store_frame(req, s.frame, sdr, master_ap1(*m, width, height)); !last) {
            out.note = last.error().message;
        } else if (*last) {
            // Every frame is in: the fill starts. The learned engine is ended first, so the GPU is the fill's.
            auto cfg = ofx_engine::load_genfill_config();
            if (!cfg) {
                out.note = cfg.error().message;
            } else {
                ofx_engine::stop_engine();
                auto started = fill::start_job(req, *cfg);
                out.note = started ? "AI fill: every frame is in; the fill has started (about 20 minutes a 100 frames)."
                                   : started.error().message;
            }
        } else {
            out.note = fill::status_text(fill::status(req));
        }
    } else if (out.note.empty()) {
        out.note = fill::status_text(st);
        if (st.state == fill::State::None && fill::other_fill(req))
            out.note = "AI fill: out of date (a setting has changed since Generate). Generate again, or set it back.";
    }
    show(*m, nullptr);
    return out;
}

std::string render(const std::string& route, const std::string& source_curve, double exposure_ev, double strength,
                   const std::string& recovery_mode, bool clip_fill, bool anchor, const std::string& output, const float* src,
                   std::ptrdiff_t src_row, float* dst, std::ptrdiff_t dst_row, int width, int height) {
    Settings s;
    s.route = route, s.source_curve = source_curve, s.exposure_ev = exposure_ev, s.strength = strength;
    s.recovery_mode = recovery_mode, s.clip_fill = clip_fill, s.anchor = anchor, s.output = output;
    return render(s, src, src_row, dst, dst_row, width, height).error;
}

std::string ai_generate(const Settings& s) {
    const rudra::OfxGrade g = rudra::ofx_grade(s);
    if (auto ok = rudra::check_grade(g); !ok) return ok.error().message;
    if (auto cfg = rudra::ofx_engine::load_genfill_config(); !cfg) return cfg.error().message;
    auto r = rudra::ofx_fill::begin_collect(fill_request(s, g));
    return r ? std::string() : r.error().message;
}

std::string ai_cancel(const Settings& s) {
    auto r = rudra::ofx_fill::cancel(fill_request(s, rudra::ofx_grade(s)));
    return r ? std::string() : r.error().message;
}

std::string ai_clear(const Settings& s) {
    auto r = rudra::ofx_fill::clear(s.shot_id);
    return r ? std::string() : r.error().message;
}

std::string ai_status(const Settings& s) {
    namespace fill = rudra::ofx_fill;
    if (!s.ai_fill) return "AI fill is off.";
    if (s.shot_id.empty()) return fill::status_text({});
    const auto req = fill_request(s, rudra::ofx_grade(s));
    const fill::Status st = fill::status(req);
    if (st.state == fill::State::None && fill::other_fill(req))
        return "AI fill: out of date (a setting has changed since Generate). Generate again, or set it back.";
    return fill::status_text(st);
}

std::string new_shot_id() {
    std::random_device rd;
    const std::uint64_t t = std::uint64_t(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const std::uint64_t v = (std::uint64_t(rd()) << 32 ^ std::uint64_t(rd())) ^ (t * 0x9E3779B97F4A7C15ull);
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
    return b;
}

void stop_engine() { rudra::ofx_engine::stop_engine(); }

}  // namespace rudra_ofx
