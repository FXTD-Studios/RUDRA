#include "ofx_process.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <array>
#include <cctype>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <random>
#include <thread>

#include <nlohmann/json.hpp>

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

// The grade with no reconstruction anywhere (27 Nov 2026, the Mask input): the luminance limit on, both strengths
// 0, so both routes give their plain inverse, as they give it outside a limit.
OfxGrade base_grade(OfxGrade g) {
    g.region = ReconstructionRegion{};
    g.region.enabled = true;
    g.region.highlight_strength = g.region.shadow_strength = 0.0f;
    return g;
}

// x = base + mask * (x - base), in x's primaries.
void apply_mask(Master& x, Master base, const std::vector<float>& mask, int width, int height) {
    if (base.primaries != x.primaries) {
        PlanarBuffer b = master_ap1(base, width, height);
        if (x.primaries != Primaries::Ap1) b = convert_primaries(b, Primaries::Ap1, x.primaries);
        base.primaries = x.primaries;
        const auto v = b.span();
        for (std::size_t i = 0; i < v.size(); ++i) base.nits[i] = double(v[i]) * kDiffuseWhiteNits;
    }
    const std::size_t n = std::size_t(width) * std::size_t(height);
    for (int c = 0; c < 3; ++c)
        for (std::size_t i = 0; i < n; ++i) {
            const double m = std::clamp(double(mask[i]), 0.0, 1.0);
            const std::size_t k = std::size_t(c) * n + i;
            x.nits[k] = base.nits[k] + m * (x.nits[k] - base.nits[k]);
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
    r.steps = std::clamp(s.steps, 10, 50);
    // 1.3.2: 0 .. 2 in hundredths (a host's slider leaves float noise), and 1 without a prompt (nothing to steer).
    r.prompt_strength = s.prompt.empty() || !std::isfinite(s.prompt_strength)
                            ? 1.0
                            : std::round(std::clamp(s.prompt_strength, 0.0, 2.0) * 100.0) / 100.0;
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

namespace {

// AI fill's guide (1.3.2): until the fill is shown, a few words and a bar along the bottom of the picture say what is
// going on and what to do: the frames collected so far in green, the fill being drawn in blue, a problem in red. A user
// cannot be expected to know that the Status line changes only when a button is pressed, and a host may not show it at
// all. The words are drawn in capitals with the 5x7 glyphs below, in the output's own terms (1.0 is diffuse white in the
// linear outputs, about 0.58 in PQ). A ready fill is shown clean, and so is a frame the fill was not made for, so a
// render of a whole clip filled in part carries nothing.
enum class GuideKind { Info, Collecting, Drawing, Problem };

struct Guide {
    std::string text;   // empty: no guide
    double fraction = 0.0;
    GuideKind kind = GuideKind::Info;
};

using Rgb = std::array<float, 3>;

// Each row's five pixels, the leftmost the highest bit. A character missing here is drawn as a space.
const std::array<std::uint8_t, 7>* glyph(char c) {
    static const std::map<char, std::array<std::uint8_t, 7>> font = {
        {'A', {0x0e, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11}},
        {'B', {0x1e, 0x11, 0x11, 0x1e, 0x11, 0x11, 0x1e}},
        {'C', {0x0e, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0e}},
        {'D', {0x1e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1e}},
        {'E', {0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x1f}},
        {'F', {0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x10}},
        {'G', {0x0e, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0f}},
        {'H', {0x11, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11}},
        {'I', {0x0e, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0e}},
        {'J', {0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0c}},
        {'K', {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11}},
        {'L', {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1f}},
        {'M', {0x11, 0x1b, 0x15, 0x15, 0x11, 0x11, 0x11}},
        {'N', {0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11}},
        {'O', {0x0e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e}},
        {'P', {0x1e, 0x11, 0x11, 0x1e, 0x10, 0x10, 0x10}},
        {'Q', {0x0e, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0d}},
        {'R', {0x1e, 0x11, 0x11, 0x1e, 0x14, 0x12, 0x11}},
        {'S', {0x0f, 0x10, 0x10, 0x0e, 0x01, 0x01, 0x1e}},
        {'T', {0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}},
        {'U', {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e}},
        {'V', {0x11, 0x11, 0x11, 0x11, 0x11, 0x0a, 0x04}},
        {'W', {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0a}},
        {'X', {0x11, 0x11, 0x0a, 0x04, 0x0a, 0x11, 0x11}},
        {'Y', {0x11, 0x11, 0x0a, 0x04, 0x04, 0x04, 0x04}},
        {'Z', {0x1f, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1f}},
        {'0', {0x0e, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0e}},
        {'1', {0x04, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x0e}},
        {'2', {0x0e, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1f}},
        {'3', {0x1f, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0e}},
        {'4', {0x02, 0x06, 0x0a, 0x12, 0x1f, 0x02, 0x02}},
        {'5', {0x1f, 0x10, 0x1e, 0x01, 0x01, 0x11, 0x0e}},
        {'6', {0x06, 0x08, 0x10, 0x1e, 0x11, 0x11, 0x0e}},
        {'7', {0x1f, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}},
        {'8', {0x0e, 0x11, 0x11, 0x0e, 0x11, 0x11, 0x0e}},
        {'9', {0x0e, 0x11, 0x11, 0x0f, 0x01, 0x02, 0x0c}},
        {'-', {0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00}},
        {':', {0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00}},
        {'.', {0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x0c}},
        {',', {0x00, 0x00, 0x00, 0x00, 0x0c, 0x04, 0x08}},
        {';', {0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x08}},
        {'/', {0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x00}},
        {'\\', {0x00, 0x10, 0x08, 0x04, 0x02, 0x01, 0x00}},
        {'(', {0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02}},
        {')', {0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08}},
        {'[', {0x0e, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0e}},
        {']', {0x0e, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0e}},
        {'\'', {0x04, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00}},
        {'"', {0x0a, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00}},
        {'>', {0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08}},
        {'<', {0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02}},
        {'%', {0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03}},
        {'_', {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f}},
        {'?', {0x0e, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04}},
        {'!', {0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04}},
        {'+', {0x00, 0x04, 0x04, 0x1f, 0x04, 0x04, 0x00}},
        {'=', {0x00, 0x00, 0x1f, 0x00, 0x1f, 0x00, 0x00}},
        {'#', {0x0a, 0x0a, 0x1f, 0x0a, 0x1f, 0x0a, 0x0a}},
        {'&', {0x0c, 0x12, 0x14, 0x08, 0x15, 0x12, 0x0d}},
        {'*', {0x00, 0x04, 0x15, 0x0e, 0x15, 0x04, 0x00}},
    };
    const auto it = font.find(char(std::toupper(static_cast<unsigned char>(c))));
    return it == font.end() ? nullptr : &it->second;
}

void fill_rect(float* dst, std::ptrdiff_t dst_row, int width, int height, int x0, int y0, int w, int h, const Rgb& rgb) {
    for (int y = std::max(0, y0); y < std::min(height, y0 + h); ++y) {
        float* row = dst + std::ptrdiff_t(y) * dst_row;
        for (int x = std::max(0, x0); x < std::min(width, x0 + w); ++x) {
            float* p = row + std::ptrdiff_t(x) * 4;
            p[0] = rgb[0];
            p[1] = rgb[1];
            p[2] = rgb[2];
            p[3] = 1.0f;
        }
    }
}

// `text` in lines of at most `cols` characters, broken at spaces (a longer word is cut), at most `most` lines; the
// last one ends in ".." when there was more.
std::vector<std::string> wrap(const std::string& text, std::size_t cols, std::size_t most) {
    std::vector<std::string> lines;
    std::istringstream words(text);
    std::string line, w;
    bool more = false;
    while (words >> w) {
        if (w.size() > cols) w.resize(cols);
        if (line.empty()) {
            line = w;
        } else if (line.size() + 1 + w.size() <= cols) {
            line += " " + w;
        } else if (lines.size() + 1 >= most) {
            more = true;
            break;
        } else {
            lines.push_back(line);
            line = w;
        }
    }
    if (!line.empty()) lines.push_back(line);
    if (more && !lines.empty()) {
        std::string& last = lines.back();
        while (last.size() + 3 > cols) {   // whole words off its end, to make room for " .."
            const auto space = last.find_last_of(' ');
            if (space == std::string::npos || space == 0) {
                last.resize(cols > 3 ? cols - 3 : 0);
                break;
            }
            last.resize(space);
        }
        last += " ..";
    }
    return lines;
}

void draw_guide(float* dst, std::ptrdiff_t dst_row, int width, int height, const Guide& g, const std::string& output) {
    if (!dst || width <= 0 || height <= 0 || g.text.empty()) return;
    const float hi = output == "pq_rec2020" ? 0.58f : 1.0f, lo = 0.03f * hi;
    const Rgb dark{lo, lo, lo}, white{0.95f * hi, 0.95f * hi, 0.95f * hi};
    Rgb lit{0.5f * hi, 0.5f * hi, 0.5f * hi};                                 // grey
    if (g.kind == GuideKind::Collecting) lit = Rgb{0.15f * hi, hi, 0.25f * hi};   // green
    if (g.kind == GuideKind::Drawing) lit = Rgb{0.2f * hi, 0.45f * hi, hi};       // blue
    if (g.kind == GuideKind::Problem) lit = Rgb{hi, 0.25f * hi, 0.15f * hi};      // red
    // The bar, 9 rows at 1080p: the part done lit (all of it for a problem).
    const int rows = std::min(height, std::max(2, height / 120));
    const double part = g.kind == GuideKind::Problem ? 1.0 : std::clamp(g.fraction, 0.0, 1.0);
    fill_rect(dst, dst_row, width, height, 0, height - rows, width, rows, dark);
    fill_rect(dst, dst_row, width, height, 0, height - rows, int(std::lround(part * width)), rows, lit);
    // The words above it, white on dark, at most three lines: letters 20 x 28 pixels and 79 to a line at 1080p.
    const int s = std::max(1, height / 240), pad = 2 * s, line_h = 10 * s;
    const int cols = (width - 3 * s) / (6 * s);
    if (cols < 8) return;
    const std::vector<std::string> lines = wrap(g.text, std::size_t(cols), 3);
    std::size_t longest = 0;
    for (const auto& l : lines) longest = std::max(longest, l.size());
    const int box_w = int(longest) * 6 * s - s + 2 * pad, box_h = int(lines.size()) * line_h - 3 * s + 2 * pad;
    const int y_box = height - rows - box_h;
    if (lines.empty() || y_box < 0) return;
    fill_rect(dst, dst_row, width, height, 0, y_box, box_w, box_h, dark);
    for (std::size_t li = 0; li < lines.size(); ++li)
        for (std::size_t k = 0; k < lines[li].size(); ++k)
            if (const auto* gl = glyph(lines[li][k]))
                for (int r = 0; r < 7; ++r)
                    for (int c = 0; c < 5; ++c)
                        if (((*gl)[std::size_t(r)] >> (4 - c)) & 1)
                            fill_rect(dst, dst_row, width, height, pad + int(k) * 6 * s + c * s, y_box + pad + int(li) * line_h + r * s,
                                      s, s, white);
}

// The numbers in `text`, in order ("window 2 of 5, step 12 of 30": 2, 5, 12, 30).
std::vector<int> numbers_in(const std::string& text) {
    std::vector<int> out;
    for (std::size_t i = 0; i < text.size();) {
        if (!std::isdigit(static_cast<unsigned char>(text[i]))) {
            ++i;
            continue;
        }
        int v = 0;
        for (; i < text.size() && std::isdigit(static_cast<unsigned char>(text[i])); ++i) v = std::min(100000000, v * 10 + (text[i] - '0'));
        out.push_back(v);
    }
    return out;
}

// AI fill's clipped code (run.py --tau-high, ofx_job.recipe): a pixel at 240 of 255 or more in any channel.
constexpr int kClippedCode = 240;

// The brightest 8-bit code in the frame's SDR codes.
int brightest_code(const rudra::SdrImage& codes) {
    float top = 0.0f;
    for (const float v : codes.buffer().span())
        if (v > top) top = v;
    return int(std::lround(std::clamp(top, 0.0f, 1.0f) * 255.0f));
}

std::string keep_open(const std::string& host) { return " Keep " + host + " open; press Refresh status to update this."; }

// The words while the fill is drawn: how far it got, from the job's last word ("window 2 of 5, step 12 of 30").
Guide running_guide(const rudra::ofx_fill::Status& st, const std::string& host) {
    const std::string& m = st.message;
    if (m.find("reading") != std::string::npos)
        return {"RUDRA AI fill: reading the prompt (minutes the first time)." + keep_open(host), 0.0, GuideKind::Drawing};
    if (m.find("low-memory mode") != std::string::npos)   // ofx_job.py ran out of GPU memory and runs it again
        return {"RUDRA AI fill: not enough GPU memory, so it is drawn again in low-memory mode." + keep_open(host), 0.0,
                GuideKind::Drawing};
    if (m == "stopping") return {"RUDRA AI fill: stopping (Cancel was pressed).", 0.0, GuideKind::Drawing};
    if (const std::vector<int> v = numbers_in(m); m.rfind("window ", 0) == 0 && v.size() >= 2 && v[1] > 0) {
        const double within = v.size() >= 4 && v[3] > 0 ? std::clamp(double(v[2]) / v[3], 0.0, 1.0) : 0.0;
        const double part = (std::clamp(v[0], 1, v[1]) - 1 + within) / v[1];
        return {"RUDRA AI fill: drawing, " + std::to_string(int(part * 100.0)) + "% done." + keep_open(host), part, GuideKind::Drawing};
    }
    if (st.done > 0 && st.of > 0)
        return {"RUDRA AI fill: drawing, frame " + std::to_string(st.done) + " of " + std::to_string(st.of) + "." + keep_open(host),
                double(st.done) / st.of, GuideKind::Drawing};
    return {"RUDRA AI fill: starting (the model loads first)." + keep_open(host), 0.0, GuideKind::Drawing};
}

// The words for a fill that failed: the remedy for the usual causes, else the job's own word (its log's path left to
// the Status line).
Guide failed_guide(std::string m) {
    for (const char* tail : {" (log: ", " (its log: "})
        if (const auto at = m.find(tail); at != std::string::npos) m.erase(at);
    std::string low = m;
    for (char& c : low) c = char(std::tolower(static_cast<unsigned char>(c)));
    const auto has = [&](const char* w) { return low.find(w) != std::string::npos; };
    std::string text;
    if (has("out of memory") || has("outofmemory") || has("gpu memory"))
        text = "RUDRA AI fill failed: not enough GPU memory. Close other programs that use the GPU, then press Generate again.";
    else if (has("no space") || has("disk space") || has("disk is full"))
        text = "RUDRA AI fill failed: not enough disk space. Free some, then press Generate again.";
    else if (has("no word from"))
        text = "RUDRA AI fill stopped: " + m + ". Press Generate again.";
    else
        text = "RUDRA AI fill failed; press Generate to try again. Why: " + m;
    return {text, 1.0, GuideKind::Problem};
}

// The words while frames are collected: how many are in, and what to do.
Guide collecting_guide(const rudra::ofx_fill::Status& st, const std::string& what) {
    return {"RUDRA AI fill: " + std::to_string(st.done) + " of " + std::to_string(st.of) + " frames in. " + what,
            st.of > 0 ? double(st.done) / st.of : 0.0, GuideKind::Collecting};
}

// An AI fill message without its own "AI fill: " in front, for the guide's "RUDRA AI fill: ".
std::string bare(const std::string& m) {
    const std::string head = "AI fill: ";
    return m.rfind(head, 0) == 0 ? m.substr(head.size()) : m;
}

// Generate's last refusal for each shot (1.3.2): the guide says it, as a host may not show the Status line or a message.
std::mutex& refusal_mutex() {
    static std::mutex m;
    return m;
}

std::map<std::string, std::string>& refusals() {
    static auto* r = new std::map<std::string, std::string>();   // never destroyed: a render may outlive statics
    return *r;
}

void remember_refusal(const std::string& shot, const std::string& why) {
    std::lock_guard<std::mutex> lock(refusal_mutex());
    if (why.empty())
        refusals().erase(shot);
    else
        refusals()[shot] = why;
}

std::string refusal(const std::string& shot) {
    std::lock_guard<std::mutex> lock(refusal_mutex());
    const auto it = refusals().find(shot);
    return it == refusals().end() ? std::string() : it->second;
}

Outcome failure(std::string error) {
    Outcome o;
    o.error = std::move(error);
    return o;
}

}  // namespace

int flat_top_below_white(const float* rgba, std::ptrdiff_t row, int width, int height) {
    if (!rgba || width <= 0 || height <= 0) return 0;
    std::array<std::size_t, 256> hist{};
    std::size_t n = 0;
    for (int y = 0; y < height; y += 2) {
        const float* r = rgba + std::ptrdiff_t(y) * row;
        for (int x = 0; x < width; x += 2, ++n) {
            const float* p = r + std::ptrdiff_t(x) * 4;
            float m = std::max(p[0], std::max(p[1], p[2]));
            m = std::isfinite(m) ? std::clamp(m, 0.0f, 1.0f) : 0.0f;
            ++hist[std::size_t(std::lround(m * 255.0f))];
        }
    }
    int top = 255;
    while (top > 0 && hist[std::size_t(top)] == 0) --top;
    if (top >= kClippedCode || top < 16 || n == 0) return 0;
    std::size_t near = 0;
    for (int c = std::max(0, top - 2); c <= top; ++c) near += hist[std::size_t(c)];
    return double(near) >= 0.005 * double(n) ? top : 0;
}

namespace {

constexpr const char* kStyleNone = "OfxImageEffectColourManagementNone";
constexpr const char* kStyleBasic = "OfxImageEffectColourManagementBasic";

bool managed(const std::string& style) { return !style.empty() && style != kStyleNone; }

// The SDR Rec.709 display colourspaces RUDRA reads (the native config's names and labels, the generic SDR display).
bool sdr_display(const std::string& colourspace) {
    std::string c = colourspace;
    for (char& ch : c) ch = char(std::tolower(static_cast<unsigned char>(ch)));
    return c == "ofx_display_sdr" || c.find("rec1886_rec709") != std::string::npos || c.find("rec.1886 rec.709") != std::string::npos;
}

// What to do about a picture that is not the SDR clip, in the host's terms.
std::string host_advice(const std::string& host_label, bool is_managed) {
    if (host_label == "Resolve")
        return is_managed ? "Resolve did not hand RUDRA the Rec.709 codes it asks for: use a DaVinci YRGB project with the timeline "
                            "color space Rec.709 Gamma 2.4 (Project Settings > Color Management)."
                          : "In Resolve, turn on Enable OpenFX Color Management (Project Settings > Color Management), or use a "
                            "DaVinci YRGB project with the timeline color space Rec.709 Gamma 2.4.";
    if (host_label == "Nuke") return "In Nuke, read the clip with Raw Data on.";
    return "Give RUDRA the clip's own SDR Rec.709 codes.";
}

}  // namespace

std::string flat_top_advice(int code, const std::string& host_label, bool is_managed) {
    std::string s = "the picture's highlights stop flat at code " + std::to_string(code) +
                    " of 255, not 255: it is not the SDR clip RUDRA reads, so RUDRA finds nothing clipped to recover. ";
    if (code >= 233 && code <= 237)   // video white (235): a video-range clip read as full range
        return s + "It looks like a video-range clip read as full range" +
               (host_label == "Resolve" ? std::string(": set the clip's Data Levels to Video (Media Pool > Clip Attributes).")
                                        : std::string(": read it as video range."));
    return s + host_advice(host_label, is_managed);
}

std::vector<std::string> colour_source_preferences(const std::string& style) {
    if (!managed(style)) return {};
    if (style == kStyleBasic) return {"ofx_display_sdr"};
    return {"rec1886_rec709_display", "ofx_display_sdr"};   // Core, Full, OCIO (the studio config has the name as an alias)
}

std::string colour_output(const std::string& output, const std::string& style) {
    if (!managed(style)) return {};
    const bool pq = output == "pq_rec2020";
    if (style == kStyleBasic) return pq ? "ofx_display_hdr" : "ofx_scene_linear";
    if (pq) return "rec2100_pq_display";
    return output == "linear_acescg" ? "ACEScg" : "lin_rec2020";
}

std::string input_problem(const Settings& s) {
    const bool is_managed = managed(s.colour_style);
    if (is_managed && !s.input_colourspace.empty() && !sdr_display(s.input_colourspace))
        return "the host hands RUDRA the picture as " + s.input_colourspace +
               ", not the SDR Rec.709 codes it reads, so RUDRA finds nothing clipped to recover. " + host_advice(s.host_label, true);
    if (s.flat_top > 0) return flat_top_advice(s.flat_top, s.host_label, is_managed);
    return {};
}

FrameSpan resolve_frames(int playhead, int from, int to, double timeline_end, double clip_end) {
    FrameSpan r;
    r.first = from != 0 ? from : playhead;
    const auto near = [&r](double e) { return std::isfinite(e) && e >= r.first && e - r.first < kMaxFillFrames; };
    if (to != 0)
        r.last = to;
    else if (timeline_end > 0 && near(timeline_end))
        r.last = int(std::lround(timeline_end));
    else if (near(clip_end))
        r.last = int(std::lround(clip_end));
    else
        r.last = r.first + 99;
    r.last = std::max(r.last, r.first);
    return r;
}

Outcome render(const Settings& s, const float* src, std::ptrdiff_t src_row, float* dst, std::ptrdiff_t dst_row, int width, int height) {
    using namespace rudra;
    namespace fill = rudra::ofx_fill;
    Outcome out;
    if (width <= 0 || height <= 0 || !src || !dst) return failure("No picture to render");
    const OfxGrade g = ofx_grade(s);
    if (auto ok = check_grade(g); !ok) return failure(ok.error().message);
    const std::size_t n = std::size_t(width) * std::size_t(height);
    const bool masked = !s.mask.empty();
    if (masked && s.mask.size() != n) return failure("The mask is not the picture's size");
    if (!s.ai_fill && !masked) {
        auto r = render_sdr_to_hdr(g, src, src_row, dst, dst_row, width, height);
        return r ? Outcome{} : failure(r.error().message);
    }

    const SdrImage sdr = codes_of(src, src_row, width, height);
    // The Mask (27 Nov 2026): outside it the plain inverse, made once per frame when a mask is connected.
    std::optional<Master> base;
    std::string base_error;
    auto show = [&](Master m, const PlanarBuffer* generated) {
        if (masked) {
            if (!base) {
                auto b = make_master(base_grade(g), sdr);
                if (!b) {
                    base_error = b.error().message;
                    return;
                }
                base = std::move(*b);
            }
            apply_mask(m, *base, s.mask, width, height);
        }
        std::optional<PlanarBuffer> alpha;
        if (g.alpha == "limit") alpha = limit_plane(sdr, g.region);
        if (g.alpha == "ai_fill") alpha = generated ? *generated : PlanarBuffer(1, height, width, 0.0f);
        if (alpha && masked)
            for (std::size_t i = 0; i < n; ++i) alpha->plane(0)[i] *= std::clamp(s.mask[i], 0.0f, 1.0f);
        write_output(m, g.exposure_ev, g.output, src, src_row, dst, dst_row, width, height, alpha ? &*alpha : nullptr);
    };
    auto shown = [&](Outcome o) -> Outcome {
        if (!base_error.empty()) o.error = base_error;
        return o;
    };
    if (!s.ai_fill) {
        auto m = make_master(g, sdr);
        if (!m) return failure(m.error().message);
        show(std::move(*m), nullptr);
        return shown(out);
    }

    // AI fill (phase 3): the stored fill when it is done, else the master, stored while collecting. 1.3.2: and, until the
    // fill is shown, the guide in the picture (draw_guide), its words in Outcome::guide.
    const fill::FillRequest req = fill_request(s, g);
    const fill::Status st = fill::status(req);
    const std::string host = s.host_label.empty() ? std::string("the host") : s.host_label;
    const bool resolve = s.host_label == "Resolve";
    const bool inside = s.frame >= req.first && s.frame <= req.last;
    Guide guide;
    auto guided = [&](Outcome o) -> Outcome {
        if (!guide.text.empty()) {
            draw_guide(dst, dst_row, width, height, guide, g.output);
            o.guide = guide.text;
        }
        return shown(std::move(o));
    };
    if (st.state == fill::State::Done) {
        std::string why = "shown at full size only (this render is a proxy)";
        if (s.full_size) {
            if (auto f = fill::fill_frame(req, s.frame, sdr, &why)) {
                show(master_from_ap1(f->ap1), &f->generated);
                out.note = fill::status_text(st);
                if (st.generated == 0.0) {   // a fill that found nothing to draw says so; any other ready fill is clean
                    out.note = "AI fill: done, and it found nothing to draw in frames " + std::to_string(req.first) + " to " +
                               std::to_string(req.last) + " (nothing there clips at 240 and up).";
                    guide = {"RUDRA AI fill: done, and it found nothing to draw in frames " + std::to_string(req.first) + " to " +
                                 std::to_string(req.last) + ": nothing there clips (240 and up).",
                             1.0, GuideKind::Info};
                    return guided(out);
                }
                return shown(out);   // ready: the picture, clean
            }
        }
        out.note = "AI fill: " + why + ". The master is shown.";
        if (!s.full_size)   // Resolve draws a paused frame small first, then at full size
            guide = {resolve ? "RUDRA AI fill is ready; it shows on full-size frames (Resolve draws a paused frame small first)."
                             : "RUDRA AI fill is ready; it is shown at full size only (turn the proxy off).",
                     1.0, GuideKind::Info};
        else if (inside)
            guide = {"RUDRA AI fill: " + why + ".", 1.0, GuideKind::Problem};
        // else a frame the fill was not made for (From / To, or the clip's handles): the master, clean
    } else if (st.state == fill::State::Running && s.full_size) {
        // The stored master while the fill draws: the learned engine stays off the GPU the fill is using.
        if (auto stored = fill::master_frame(req, s.frame, sdr)) {
            show(master_from_ap1(*stored), nullptr);
            out.note = fill::status_text(st);
            guide = running_guide(st, host);
            return guided(out);
        }
    }
    auto m = make_master(g, sdr);
    if (!m) return failure(m.error().message);
    if (st.state == fill::State::Collecting) {
        if (!s.full_size) {
            out.note = "AI fill: frames are collected from full-size renders only (turn the proxy off).";
            guide = collecting_guide(st, resolve ? "This frame came at a reduced size, so it is not taken: play the clip through on the "
                                                   "Color page, with no timeline proxy."
                                                 : "Frames are taken from full-size renders only: turn the proxy and the viewer's "
                                                   "downscale off.");
        } else if (!inside) {
            // The clip's range is the host's (Settings::first_frame/last_frame): said, so a host whose frame
            // numbers do not match it shows why nothing is collected (26 Nov 2026).
            out.note = "AI fill: frame " + std::to_string(s.frame) + " is outside its frames (" + std::to_string(req.first) + " to " +
                       std::to_string(req.last) + "), so it is not stored.";
            // Resolve numbers a clip's frames by its source media, which its user does not see: words, not numbers.
            guide = collecting_guide(
                st, !resolve ? "Frame " + std::to_string(s.frame) + " is not one of them: play frames " + std::to_string(req.first) + " to " +
                                   std::to_string(req.last) + "."
                    : s.frame < req.first ? std::string("This frame comes before them: play on.")
                                          : std::to_string(st.of - st.done) + (st.of - st.done == 1 ? " was" : " were") +
                                                " missed: go back to where you pressed Generate and play through again on the Color page.");
        } else if (auto last = fill::store_frame(req, s.frame, sdr, master_ap1(*m, width, height)); !last) {
            out.note = last.error().message;
            guide = {"RUDRA AI fill: " + bare(last.error().message), st.of > 0 ? double(st.done) / st.of : 0.0, GuideKind::Problem};
        } else {
            const fill::Status now = fill::status(req);
            out.log = "AI fill: frame " + std::to_string(s.frame) + " stored, " + std::to_string(now.done) + " of " + std::to_string(now.of);
            if (*last) {
                // Every frame is in: the fill starts. The learned engine is ended first, so the GPU is the fill's.
                const auto start = [&]() -> Result<void> {
                    auto cfg = ofx_engine::load_genfill_config();
                    if (!cfg) return cfg.error();
                    ofx_engine::stop_engine();
                    return fill::start_job(req, *cfg);
                };
                const Result<void> started = start();
                out.note = started ? "AI fill: every frame is in; the fill has started (about 20 minutes a 100 frames at 30 steps)."
                                   : started.error().message;
                guide = started ? Guide{"RUDRA AI fill: all " + std::to_string(now.of) + " frames in; the fill is being drawn." + keep_open(host),
                                        0.0, GuideKind::Drawing}
                                : Guide{"RUDRA AI fill did not start: " + bare(started.error().message), 1.0, GuideKind::Problem};
            } else {
                out.note = fill::status_text(now);
                guide = collecting_guide(now, resolve ? "Play the clip through once on the Color page; the fill starts when all are in."
                                                      : "Play frames " + std::to_string(req.first) + " to " + std::to_string(req.last) +
                                                            " once; the fill starts when all are in.");
            }
        }
    } else if (out.note.empty()) {
        out.note = fill::status_text(st);
        if (st.state == fill::State::Running) {
            guide = running_guide(st, host);   // a proxy render, or a frame not stored, while it draws
        } else if (st.state == fill::State::Failed) {
            guide = failed_guide(st.message);
        } else if (st.state == fill::State::Cancelled) {
            guide = {"RUDRA AI fill cancelled. Press Generate to start again (the frames stored are kept).", 0.0, GuideKind::Info};
        } else if (st.state == fill::State::None) {
            if (const std::string why = refusal(s.shot_id); !why.empty()) {
                guide = {"RUDRA AI fill: Generate did not start. " + bare(why), 0.0, GuideKind::Problem};
            } else if (fill::other_fill(req)) {
                out.note = "AI fill: out of date (a setting has changed since Generate). Generate again, or set it back.";
                guide = {"RUDRA AI fill is out of date: a setting changed since Generate. Press Generate again, or set it back.", 0.0,
                         GuideKind::Info};
            } else if (const std::string problem = input_problem(s); !problem.empty()) {
                guide = {"RUDRA AI fill: " + problem, 0.0, GuideKind::Problem};
            } else if (const int top = brightest_code(sdr); top < kClippedCode) {
                // Nothing to fill here (27 Nov 2026: in Resolve every frame came with nothing clipped, and two fills drew
                // nothing without a word).
                guide = {"RUDRA AI fill: nothing in this frame is clipped (its brightest code is " + std::to_string(top) +
                             " of 255), and AI fill draws only where the picture clips (240 and up). If the source does clip, "
                             "RUDRA is not getting its Rec.709 codes: check the clip's data levels and the color management.",
                         0.0, GuideKind::Info};
            } else {
                guide = {resolve ? "RUDRA AI fill: put the playhead on the first frame to fill, then press Generate."
                                 : "RUDRA AI fill: press Generate, then play the frames once.",
                         0.0, GuideKind::Info};
            }
        }
    }
    show(std::move(*m), nullptr);
    return guided(out);
}

std::string render(const std::string& route, const std::string& source_curve, double exposure_ev, double strength,
                   const std::string& recovery_mode, bool clip_fill, bool anchor, const std::string& output, const float* src,
                   std::ptrdiff_t src_row, float* dst, std::ptrdiff_t dst_row, int width, int height) {
    Settings s;
    s.route = route, s.source_curve = source_curve, s.exposure_ev = exposure_ev, s.strength = strength;
    s.recovery_mode = recovery_mode, s.clip_fill = clip_fill, s.anchor = anchor, s.output = output;
    return render(s, src, src_row, dst, dst_row, width, height).error;
}

namespace {

std::string generate(const Settings& s) {
    // A To before the first frame made a fill of one frame without a word (Resolve, 27 Nov 2026: To 4, the playhead 26).
    if (s.to_frame != 0 && s.to_frame < s.first_frame)
        return "AI fill: To (" + std::to_string(s.to_frame) + ") is before the first frame to fill (" + std::to_string(s.first_frame) +
               "). Set To to 0 to fill to the clip's end, or to a frame after " + std::to_string(s.first_frame) + ".";
    if (const std::string problem = input_problem(s); !problem.empty()) return "AI fill: " + problem;
    const rudra::OfxGrade g = rudra::ofx_grade(s);
    if (auto ok = rudra::check_grade(g); !ok) return ok.error().message;
    auto cfg = rudra::ofx_engine::load_genfill_config();
    if (!cfg) return cfg.error().message;
    const auto req = fill_request(s, g);
    auto r = rudra::ofx_fill::begin_collect(req);
    if (!r) return r.error().message;
    // 1.3.2: a new prompt is read while the frames are collected, not after them (the fill waits for it if need be).
    (void)rudra::ofx_fill::start_prompt_embed(req, *cfg);
    return {};
}

}  // namespace

std::string ai_generate(const Settings& s) {
    const std::string why = generate(s);
    remember_refusal(s.shot_id, why);   // the guide says it until a Generate starts
    return why;
}

std::string ai_cancel(const Settings& s) {
    auto r = rudra::ofx_fill::cancel(fill_request(s, rudra::ofx_grade(s)));
    return r ? std::string() : r.error().message;
}

std::string ai_clear(const Settings& s) {
    remember_refusal(s.shot_id, {});
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
    std::string text = fill::status_text(st);
    if (st.state == fill::State::Collecting) {   // which frames are still to play (1.3.2)
        const std::vector<int> miss = fill::missing_frames(req);
        if (!miss.empty()) {
            std::size_t i = 1;
            while (i < miss.size() && miss[i] == miss[i - 1] + 1) ++i;
            const int a = req.first + miss.front(), b = req.first + miss[i - 1];
            text += a == b ? " Still to play: frame " + std::to_string(a) : " Still to play: frames " + std::to_string(a) + " to " + std::to_string(b);
            if (i < miss.size()) text += " and " + std::to_string(miss.size() - i) + " more";
            text += ".";
        }
    }
    if (st.state == fill::State::Collecting || st.state == fill::State::None) {
        // What Generate will cost (27 Nov 2026): 100 frames at 30 steps took about 20 minutes on the 1.3.1 test shot,
        // with no prompt; a prompt blended in (a strength other than 1) takes about a third longer (1.3.2).
        const int n = req.frames(),
                  minutes = std::max(1, int(std::lround(double(n) * req.steps * fill::prompt_cost(req) / 150.0)));
        text += " Frames " + std::to_string(req.first) + " to " + std::to_string(req.last) + " (" + std::to_string(n) +
                "): about " + std::to_string(minutes) + (minutes == 1 ? " minute" : " minutes") + " to draw at " + std::to_string(req.steps) +
                " steps.";
    }
    return text;
}

namespace {

// suggest.json (rudra/genfill/ofx_prompt.py) as the plug-in shows it: the first highlight suggestion for Prompt.
PromptSuggestion read_suggestion(const std::string& text) {
    PromptSuggestion out;
    const nlohmann::json j = nlohmann::json::parse(text, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        out.error = "its answer is not readable.";
        return out;
    }
    if (!j.value("ok", false)) {
        out.error = j.value("error", std::string("it failed."));
        return out;
    }
    out.note = j.value("describe_error", std::string());
    if (j.contains("suggestions") && j["suggestions"].is_array())
        for (const auto& s : j["suggestions"]) {
            if (!s.is_object() || s.value("region", std::string("highlight")) != "highlight") continue;
            if (out.text.empty()) {
                out.text = s.value("text", std::string());
                out.label = s.value("label", std::string());
                out.reason = s.value("reason", std::string());
            } else if (out.others.size() < 3) {
                out.others.push_back(s.value("label", std::string()));
            }
        }
    if (out.text.empty())
        out.error = "nothing is fully clipped in this frame, so AI fill has nothing to draw here: try a frame with clipped highlights.";
    else
        out.ready = true;
    return out;
}

}  // namespace

PromptSuggestion ai_suggest(const Settings& s, const unsigned char* rgb, int width, int height, double wait_seconds,
                            const std::function<bool(double)>& tick) {
    namespace fill = rudra::ofx_fill;
    PromptSuggestion out;
    if (!rgb || width <= 0 || height <= 0) {
        out.error = "no frame to read.";
        return out;
    }
    auto cfg = rudra::ofx_engine::load_genfill_config();
    if (!cfg) {
        out.error = cfg.error().message;
        return out;
    }
    rudra::PlanarBuffer code(3, height, width);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < 3; ++c)
                code.plane(c)[std::size_t(y) * std::size_t(width) + std::size_t(x)] =
                    float(rgb[(std::size_t(y) * std::size_t(width) + std::size_t(x)) * 3 + std::size_t(c)]) / 255.0f;
    // Florence-2 on the CPU while this shot's fill holds the GPU.
    const bool busy = !s.shot_id.empty() && fill::status(fill_request(s, rudra::ofx_grade(s))).state == fill::State::Running;
    if (auto started = fill::start_suggest(s.shot_id, rudra::SdrImage(std::move(code)), *cfg, busy); !started) {
        out.error = started.error().message;
        return out;
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        PromptSuggestion now = ai_suggestion(s);
        if (now.ready || !now.error.empty()) return now;
        const double gone = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (gone >= wait_seconds || (tick && !tick(std::min(1.0, gone / std::max(1e-3, wait_seconds))))) return now;
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
}

PromptSuggestion ai_suggestion(const Settings& s) {
    PromptSuggestion out;
    auto a = rudra::ofx_fill::suggest_answer(s.shot_id);
    if (!a) {
        out.error = a.error().message;
        return out;
    }
    if (!*a) return out;   // still being made
    return read_suggestion(**a);
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
