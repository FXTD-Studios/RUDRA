#include "rudra/deliver/master.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <cstdlib>
#include <random>
#include <regex>
#include <system_error>

#include <nlohmann/json.hpp>

#include "rudra/core/baseline.hpp"
#include "rudra/core/gamut.hpp"
#include "rudra/core/compare.hpp"
#include "rudra/core/master.hpp"
#include "rudra/core/measure.hpp"
#include "rudra/deliver/exr.hpp"
#include "rudra/deliver/sidecars.hpp"
#include "rudra/platform/hash.hpp"
#include "rudra/platform/pyjson.hpp"

#ifndef RUDRA_VERSION_LABEL
#define RUDRA_VERSION_LABEL "unknown"
#endif
#ifndef RUDRA_BUILD_COMMIT
#define RUDRA_BUILD_COMMIT "unknown"
#endif

namespace rudra {
namespace {

Result<Primaries> primaries_of(const std::string& s) {
    if (s == "rec709") return Primaries::Rec709;
    if (s == "rec2020") return Primaries::Rec2020;
    if (s == "p3d65") return Primaries::P3D65;
    if (s == "adobe_rgb") return Primaries::AdobeRgb;   // 16 Oct 2026: from a still's ICC profile
    return make_error(ErrorCode::InvalidArgument, "Unknown source space.", s);
}

Result<RecoveryMode> mode_of(const std::string& s) {
    if (s == "all") return RecoveryMode::All;
    if (s == "highlights") return RecoveryMode::Highlights;
    if (s == "shadows") return RecoveryMode::Shadows;
    if (s == "off") return RecoveryMode::Off;
    return make_error(ErrorCode::InvalidArgument, "recovery_mode must be all, highlights, shadows, or off", s);
}

// Python round(x, 1): the correctly rounded decimal, read back.
double py_round1(double x) {
    char b[64];
    std::snprintf(b, sizeof b, "%.1f", x);
    return std::strtod(b, nullptr);
}

// round(x, n) the same way.
double py_round(double x, int n) {
    char b[64];
    std::snprintf(b, sizeof b, "%.*f", n, x);
    return std::strtod(b, nullptr);
}

// How the analytic master filled where the SDR clipped, as the EXR and the
// sidecar record it.
std::string clip_fill_text(const MasterRequest& q) {
    if (!q.clip_fill) return "off (clipped values left at the inverse's ceiling)";
    std::string t = "on (core/clip_fill: partial channels, smooth fill, dome)";
    for (const auto& c : q.clip_peaks) {
        char b[96];
        std::snprintf(b, sizeof b, "; peak %.0f nits at (%.3f, %.3f)", c.nits, c.x, c.y);
        t += b;
    }
    return t;
}

pyjson::Value band_json(const MasterBand& b) {
    return pyjson::Dict{{"label", b.label}, {"low_nits", b.low_nits}, {"high_nits", b.high_nits}, {"ev", b.ev}};
}

}  // namespace

Result<MasterRequest> master_request_from_json(const std::string& text) {
    MasterRequest r;
    try {
        const auto j = nlohmann::json::parse(text);
        // A project is not a master-params file. Neither this reader nor its
        // legacy consumers may silently drop an unsupported input policy.
        if (j.contains("rudra_project"))
            return make_error(ErrorCode::Unsupported, "A RUDRA project cannot be read as legacy master parameters");
        if (j.contains("decode") && !j.at("decode").is_object())
            return make_error(ErrorCode::ParseError, "Invalid decode settings in master parameters");
        for (const auto* where : {&j, j.contains("decode") ? &j.at("decode") : &j}) {
            if (where->contains("headroom_policy") && where->at("headroom_policy") != "disabled")
                return make_error(ErrorCode::Unsupported, "This master-parameters path does not yet support headroom_policy");
        }
        r.checkpoint = j.value("checkpoint", std::string());
        r.analytic = j.value("analytic", false);
        r.preserve_outside = j.value("preserve_outside", true);
        r.recovery_mode = j.value("recovery_mode", std::string("all"));
        r.strength = j.value("strength", 1.0);
        r.region_softness_stops = j.value("region_softness_stops", 1.0);
        r.anchor = j.value("anchor", true);
        r.anchor_knee = j.value("anchor_knee", 0.9);
        r.carry_chroma = j.value("carry_chroma", true);
        r.chroma_knee = j.value("chroma_knee", 0.99);
        r.settle_grain = j.value("settle_grain", true);
        r.source_space = j.value("source_space", std::string("rec709"));
        r.container = j.value("container", std::string("aces"));
        r.exr_half = j.value("exr_half", true);
        r.source_curve = j.value("source_curve", std::string("unknown"));
        if (j.contains("calibration"))
            for (const auto& c : j.at("calibration"))
                r.calibration.push_back({c.at("code").get<int>(), c.at("nits").get<double>()});
        if (j.contains("reference")) {
            auto ref = reference_fit_from_json(j.at("reference").dump());
            if (!ref) return make_error(ErrorCode::ParseError, "The master parameters' reference is not a fit.", ref.error().message);
            r.reference = std::move(*ref);
        }
        if (j.contains("masks_file")) {
            r.masks_file = j.at("masks_file").get<std::string>();
            auto m = read_mask_set(r.masks_file);
            if (!m) return make_error(ErrorCode::ParseError, "The master parameters' masks could not be read.", m.error().message + ": " + m.error().detail);
            if (!m->empty()) r.masks = std::make_shared<const MaskSet>(std::move(*m));
        }
        if (j.contains("regions"))
            for (const auto& b : j.at("regions"))
                r.regions.push_back({b.value("label", std::string()), b.at("low_nits").get<double>(),
                                     b.at("high_nits").get<double>(), b.value("ev", 0.0)});
        r.clip_fill = j.value("clip_fill", true);
        r.exposure_ev = j.value("exposure_ev", 0.0);
        if (!std::isfinite(r.exposure_ev) || std::fabs(r.exposure_ev) > 8.0)
            return make_error(ErrorCode::InvalidArgument, "exposure_ev must be within -8 and +8 stops", std::to_string(r.exposure_ev));
        if (j.contains("clip_peaks"))
            for (const auto& c : j.at("clip_peaks"))
                r.clip_peaks.push_back({c.at("x").get<double>(), c.at("y").get<double>(), c.at("nits").get<double>()});
    } catch (const std::exception& e) {
        return make_error(ErrorCode::ParseError, "The master parameters are not valid JSON.", e.what());
    }
    return r;
}

std::string master_request_json(const MasterRequest& q) {
    nlohmann::json j{{"checkpoint", q.checkpoint},     {"preserve_outside", q.preserve_outside},
                     {"recovery_mode", q.recovery_mode}, {"strength", q.strength},
                     {"region_softness_stops", q.region_softness_stops}, {"anchor", q.anchor},
                     {"anchor_knee", q.anchor_knee},     {"carry_chroma", q.carry_chroma},
                     {"chroma_knee", q.chroma_knee},     {"settle_grain", q.settle_grain},
                     {"source_space", q.source_space},
                     {"container", q.container}};
    auto regions = nlohmann::json::array();
    for (const auto& b : q.regions)
        regions.push_back({{"label", b.label}, {"low_nits", b.low_nits}, {"high_nits", b.high_nits}, {"ev", b.ev}});
    j["regions"] = regions;
    if (q.source_curve != "unknown") j["source_curve"] = q.source_curve;
    if (!q.exr_half) j["exr_half"] = false;
    if (q.exposure_ev != 0.0) j["exposure_ev"] = q.exposure_ev;
    if (q.analytic) j["analytic"] = true;
    if (!q.calibration.empty()) {
        auto cal = nlohmann::json::array();
        for (const auto& c : q.calibration) cal.push_back({{"code", c.code}, {"nits", c.nits}});
        j["calibration"] = cal;
    }
    if (!q.reference.empty())
        j["reference"] = nlohmann::json::parse(reference_fit_json(q.reference));
    if (!q.masks_file.empty()) j["masks_file"] = q.masks_file;
    if (!q.clip_fill) j["clip_fill"] = false;
    if (!q.clip_peaks.empty()) {
        auto peaks = nlohmann::json::array();
        for (const auto& c : q.clip_peaks) peaks.push_back({{"x", c.x}, {"y", c.y}, {"nits", c.nits}});
        j["clip_peaks"] = peaks;
    }
    return j.dump();
}

ClipFillOptions clip_fill_options(const MasterRequest& q) {
    ClipFillOptions o;
    o.enabled = q.clip_fill;
    o.overrides = q.clip_peaks;
    return o;
}

Fields analytic_fields_for(const SdrImage& sdr, const MasterRequest& q, ClipFill* detail) {
    if (!q.clip_fill) return analytic_fields(sdr.height(), sdr.width());
    return clip_fill_fields(sdr, kAnalyticConstants, clip_fill_options(q), detail);
}

Result<GradedMaster> grade_master(const SdrImage& sdr, const Fields& fields, const FrameScalars& scalars,
                                  const ModelConstants& model, const MasterRequest& q, const HoldSettle& settle_hold) {
    auto mode = mode_of(q.recovery_mode);
    if (!mode) return mode.error();
    const auto source = parse_source_curve(q.source_curve);
    if (!source) return make_error(ErrorCode::InvalidArgument, "source_curve must be unknown, bt1886, bt1886_shadows, aces, hable, agx, camera_log, or clip", q.source_curve);
    CompositeParams cp;
    cp.mode = *mode;
    cp.strength = static_cast<float>(q.strength);
    cp.preserve_outside = q.preserve_outside;
    cp.source = *source;
    for (const auto& c : q.calibration)
        if (c.nits > 0.0 && c.code >= 0 && c.code <= 255) cp.calibration.push_back(c);
    if (!cp.calibration.empty() && !calibration_is_monotone(cp.calibration, cp.source, model.corpus_ev))
        return make_error(ErrorCode::InvalidArgument, "calibration anchors do not describe a monotone curve", q.source_curve);
    if (!q.reference.empty()) {
        if (int(q.reference.target_log2_nits.size()) != kSourceCurveKnots)
            return make_error(ErrorCode::InvalidArgument, "reference fit must carry 256 targets", q.reference.file);
        cp.reference = q.reference;
        cp.calibration.clear();   // a reference is an anchor at every code
    }
    if (q.masks && !q.masks->empty()) cp.masks = q.masks;   // painted masks (3.3), read by the chain
    GradedMaster g;
    g.network = composite(sdr, fields, scalars, model, cp);

    // The master chain, stage by stage (render_master_pixels without the
    // container step: write_aces_exr does that conversion itself).
    std::vector<RegionBand> bands;
    for (const auto& b : q.regions) bands.push_back({b.low_nits, b.high_nits, b.ev});
    if (bands.empty()) bands = default_region_bands();
    g.graded = any_graded(bands);
    g.nits = nits_from_network(g.network);
    // The grade goes LAST (core/master.cpp render_master_pixels, ui/server.py
    // since 9 Oct 2026): before the anchor, the anchor divided it back out.
    // The anchor follows the Source picker, its calibration and reference (13 Oct 2026).
    if (q.anchor) anchor_to_sdr(g.nits, sdr, q.anchor_knee, kAnchorSoftness, settle_hold, anchor_target_knots(cp));
    if (q.carry_chroma) carry_source_chroma(g.nits, sdr, q.chroma_knee);
    if (q.settle_grain) settle_highlight_grain(g.nits, sdr, q.anchor_knee);
    apply_region_ev(g.nits, bands, q.region_softness_stops, double(model.max_hdr) * 10000.0, cp.masks.get(), q.exposure_ev);
    g.params = std::move(cp);
    return g;
}

Result<MasterResult> write_master(const SdrImage& sdr, int source_bits, const Fields& fields,
                                  const FrameScalars& scalars, const ModelConstants& model, const MasterRequest& q,
                                  const std::filesystem::path& out) {
    auto src = primaries_of(q.source_space);
    if (!src) return src.error();
    if (q.container != "aces" && q.container != "linear")
        return make_error(ErrorCode::InvalidArgument, "container must be aces or linear", q.container);
    auto g = grade_master(sdr, fields, scalars, model, q);
    if (!g) return g.error();
    const CompositeParams& cp = g->params;
    const NetworkLinearImage& network = g->network;
    const NitsFrame& nits = g->nits;
    const bool graded = g->graded;
    const auto source = parse_source_curve(q.source_curve);
    // The painted masks (3.3): written beside the master.
    std::filesystem::path masks_path;
    if (cp.masks) {
        masks_path = out;
        masks_path.replace_extension(".masks.png");
        if (auto w = write_mask_set(masks_path, *cp.masks); !w) return w.error();
    }
    const PlanarBuffer linear = scene_linear(nits);

    const FrameStats stats = analyze_frame(nits, 0);
    const StaticMetadata md = maxcll_maxfall(std::span(&stats, 1));
    double peak = -INFINITY;
    for (double v : nits.span()) peak = std::max(peak, v);

    std::string region_text = "neutral";
    if (graded) {
        pyjson::List l;
        for (const auto& b : q.regions) l.push_back(band_json(b));
        region_text = pyjson::dumps(l);
    }
    ExrAttributes provenance{
        {"rudra:checkpoint", q.checkpoint},
        {"rudra:maxCLL", std::to_string(md.maxcll)},
        {"rudra:maxFALL", std::to_string(md.maxfall)},
        {"rudra:recoveryMode", q.recovery_mode},
        {"rudra:preserveOutside", q.preserve_outside ? "True" : "False"},
        {"rudra:regionEV", region_text},
        {"rudra:tiled", q.tiled ? "True" : "False"},
        {"rudra:anchored", q.anchor ? "True" : "False"},
        {"rudra:chromaCarried", q.carry_chroma ? "True" : "False"},
        {"rudra:grainSettled", q.settle_grain ? "True" : "False"},
        {"rudra:sourceSpace", q.source_space},
        {"rudra:appVersion", RUDRA_VERSION_LABEL},
        {"rudra:buildCommit", RUDRA_BUILD_COMMIT},
    };
    if (!q.checkpoint_sha256.empty()) provenance.emplace_back("rudra:checkpointSha256", q.checkpoint_sha256);
    std::string source_sha;
    if (!q.source_file.empty()) {
        auto h = sha256_file(q.source_file);
        if (!h) return h.error();
        source_sha = *h;
        provenance.emplace_back("rudra:sourceSha256", source_sha);
    }
    if (*source != SourceCurve::Unknown) provenance.emplace_back("rudra:sourceCurve", q.source_curve);
    if (q.exposure_ev != 0.0) {
        char b[32];
        std::snprintf(b, sizeof b, "%+.2f", q.exposure_ev);
        provenance.emplace_back("rudra:exposureEV", b);
    }
    if (q.analytic) provenance.emplace_back("rudra:reconstruction", "analytic (no model)");
    if (!q.exr_half) provenance.emplace_back("rudra:sampleType", "float32");
    if (q.analytic) provenance.emplace_back("rudra:clipFill", clip_fill_text(q));
    if (!cp.reference.empty()) provenance.emplace_back("rudra:reference", cp.reference.file);
    if (cp.masks) provenance.emplace_back("rudra:masks", masks_path.filename().string());
    const bool aces = q.container == "aces";
    // The linear container is scene-linear Rec.2020, converted from the
    // plate's primaries and stamped so (it carried the plate's primaries
    // untagged until 9 Oct 2026, as the Studio did).
    auto w = aces ? write_aces_exr(out, linear, *src, 1.0, provenance, q.exr_half)
                  : write_exr(out, convert_primaries(linear, *src, Primaries::Rec2020), q.exr_half, kRec2020Chromaticities,
                              provenance);
    if (!w) return w.error();

    pyjson::List regions_json;
    for (const auto& b : q.regions) regions_json.push_back(band_json(b));
    // Roadmap 3.5: where this reconstruction departs from what the SDR
    // supports, measured on the composite against its corrected baseline
    // (before the grade, which is the artist's, not the network's).
    const SupportStats support =
        support_stats(network, corrected_baseline(sdr, model.corpus_ev, baseline_curve_params(scalars, cp, model.corpus_ev)), sdr);
    auto pct3 = [](double v) {
        char b[64];
        std::snprintf(b, sizeof b, "%.3f", v);
        return std::strtod(b, nullptr);
    };
    pyjson::Dict sidecar_dict{
        {"maxcll_nits", md.maxcll},
        {"maxfall_nits", md.maxfall},
        {"peak_nits", py_round1(peak)},
        {"resolution", pyjson::List{sdr.width(), sdr.height()}},
        {"source_bits", source_bits},
        {"container", aces ? "ACES 2065-1 (AP0)" : "scene-linear Rec.2020"},
        {"transfer", "linear"},
        {"diffuse_white_nits", 203.0},
        {"checkpoint", q.checkpoint},
        {"recovery_mode", q.recovery_mode},
        {"residual_strength", q.strength},
        {"preserve_outside", q.preserve_outside},
        {"region_ev", graded ? pyjson::Value(regions_json) : pyjson::Value(nullptr)},
        {"region_softness_stops", graded ? pyjson::Value(q.region_softness_stops) : pyjson::Value(nullptr)},
        {"tiled", q.tiled},
        {"app_version", RUDRA_VERSION_LABEL},
        {"build_commit", RUDRA_BUILD_COMMIT},
        {"invented_pixels", pyjson::Dict{
             {"invented_pct", pct3(support.invented_pct)},
             {"reinterpreted_pct", pct3(support.reinterpreted_pct)},
             {"sdr_without_detail_pct", pct3(support.sdr_without_detail_pct)},
             {"threshold_stops", kChangeStops},
             {"basis", "composite vs corrected baseline, max(R,G,B); invented where the SDR's max code is >= 254 or <= 1"},
         }},
    };
    if (!q.checkpoint_sha256.empty()) sidecar_dict.emplace_back("checkpoint_sha256", q.checkpoint_sha256);
    if (!source_sha.empty()) sidecar_dict.emplace_back("source_sha256", source_sha);
    if (*source != SourceCurve::Unknown) sidecar_dict.emplace_back("source_curve", q.source_curve);
    if (q.exposure_ev != 0.0) sidecar_dict.emplace_back("exposure_ev", q.exposure_ev);
    if (q.analytic) sidecar_dict.emplace_back("reconstruction", "analytic (no model)");
    if (q.analytic) sidecar_dict.emplace_back("clip_fill", clip_fill_text(q));
    if (!cp.calibration.empty()) {
        pyjson::List cal;
        for (const auto& c : cp.calibration)
            cal.push_back(pyjson::Dict{{"code", std::int64_t(c.code)}, {"nits", c.nits}});
        sidecar_dict.emplace_back("calibration", cal);
    }
    if (!cp.reference.empty()) {
        const auto& rf = cp.reference;
        pyjson::List target;
        for (float v : rf.target_log2_nits) target.push_back(py_round(double(v), 4));
        sidecar_dict.emplace_back("reference", pyjson::Dict{
            {"file", rf.file},
            {"samples", std::int64_t(rf.samples)},
            {"codes_seen", std::int64_t(rf.codes_seen)},
            {"exposure_stops", py_round(reference_exposure(rf, cp.source, model.corpus_ev), 3)},
            {"residual_mean_stops", py_round(rf.residual_mean, 3)},
            {"residual_p95_stops", py_round(rf.residual_p95, 3)},
            {"target_log2_nits", target},
        });
    }
    if (cp.masks) {
        pyjson::List bands_json;
        pyjson::Dict coverage;
        for (int b : cp.masks->bands()) {
            bands_json.push_back(std::int64_t(b));
            coverage.emplace_back(std::to_string(b), py_round(cp.masks->coverage(b), 4));
        }
        sidecar_dict.emplace_back("masks", pyjson::Dict{
            {"file", masks_path.filename().string()},
            {"width", std::int64_t(cp.masks->width)},
            {"height", std::int64_t(cp.masks->height)},
            {"bands", bands_json},
            {"coverage", coverage},
        });
    }
    const pyjson::Value sidecar = std::move(sidecar_dict);
    std::filesystem::path side = out;
    side.replace_extension(".json");
    if (auto s = write_text_file(side, pyjson::dumps(sidecar, 2)); !s) return s.error();

    MasterResult r;
    r.maxcll = md.maxcll;
    r.maxfall = md.maxfall;
    r.peak_nits = py_round1(peak);
    r.width = sdr.width();
    r.height = sdr.height();
    r.source_bits = source_bits;
    r.exr = out;
    r.sidecar = side;
    return r;
}

namespace {

std::string strip(const std::string& s) {
    const auto a = s.find_first_not_of(" \t\r\n\f\v");
    if (a == std::string::npos) return {};
    const auto b = s.find_last_not_of(" \t\r\n\f\v");
    return s.substr(a, b - a + 1);
}

// Path.expanduser(): a leading ~ is the home folder.
std::filesystem::path expand_user(const std::string& s) {
    if (s.empty() || s[0] != '~' || (s.size() > 1 && s[1] != '/' && s[1] != '\\')) return std::filesystem::path(s);
    const char* home = std::getenv("HOME");
#ifdef _WIN32
    if (!home) home = std::getenv("USERPROFILE");
#endif
    return home ? std::filesystem::path(home) / std::filesystem::path(s.size() > 2 ? s.substr(2) : std::string())
                : std::filesystem::path(s);
}

std::string six(int v) {
    char b[16];
    std::snprintf(b, sizeof b, "%06d", v);
    return b;
}

}  // namespace

Result<std::vector<std::filesystem::path>> master_targets(const RenderPlan& plan) {
    namespace fs = std::filesystem;
    fs::path folder = expand_user(strip(plan.render_dir));
    if (strip(plan.render_dir).empty() || !folder.is_absolute())
        return make_error(ErrorCode::InvalidArgument, "Choose an absolute render folder on the Studio computer");
    std::error_code ec;
    const fs::path resolved = fs::weakly_canonical(folder, ec);   // Path.resolve(): links followed, . and .. gone
    if (!ec) folder = resolved;
    const std::string name = strip(plan.render_name);
    static const std::regex bad("[^A-Za-z0-9_.-]");
    if (name.empty() || name == "." || name == ".." || std::regex_search(name, bad))
        return make_error(ErrorCode::InvalidArgument,
                          "Render name must contain only letters, numbers, dots, underscores or hyphens");
    const long long count = plan.render_count, start = plan.frame_start;
    if (count < 1 || count > 100000 || start < 0 || start + count > 100000000)
        return make_error(ErrorCode::InvalidArgument, "Invalid frame range");
    if (!plan.sequence && count != 1) return make_error(ErrorCode::InvalidArgument, "Image render requires one frame");
    std::vector<fs::path> targets;
    for (long long i = 0; i < count; ++i)
        targets.push_back(folder / (plan.sequence ? name + "." + six(int(start + i)) + ".exr" : name + ".exr"));
    for (const auto& out : targets) {
        fs::path side = out;
        side.replace_extension(".json");
        if (fs::exists(out) || fs::exists(side))
            return make_error(ErrorCode::InvalidArgument, "Refusing to overwrite existing render: " + out.string());
    }
    return targets;
}

Result<MasterResult> publish_master(const SdrImage& sdr, int source_bits, const Fields& fields,
                                    const FrameScalars& scalars, const ModelConstants& model,
                                    const MasterRequest& request, const std::filesystem::path& out) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(out.parent_path(), ec);
    if (ec) return make_error(ErrorCode::IoError, "Cannot create the render folder.", ec.message());
    // Stage both files, then publish without replacing any existing file.
    std::random_device rd;
    fs::path staging;
    for (int attempt = 0; attempt < 16; ++attempt) {
        char tag[16];
        std::snprintf(tag, sizeof tag, "%08x", unsigned(rd()));
        staging = out.parent_path() / (std::string(".rudra-render-") + tag);
        if (fs::create_directory(staging, ec) && !ec) break;
        staging.clear();
    }
    if (staging.empty()) return make_error(ErrorCode::IoError, "Cannot stage the render.", out.parent_path().string());
    struct Cleanup {
        fs::path dir;
        ~Cleanup() {
            std::error_code e;
            fs::remove_all(dir, e);
        }
    } cleanup{staging};
    const fs::path staged = staging / out.filename();
    auto r = write_master(sdr, source_bits, fields, scalars, model, request, staged);
    if (!r) return r.error();
    fs::path staged_side = staged, side = out;
    staged_side.replace_extension(".json");
    side.replace_extension(".json");
    // A hard link fails if the name exists: nothing is ever replaced. Where
    // the file system has no links, a copy that refuses to overwrite.
    auto place = [](const fs::path& from, const fs::path& to) -> std::error_code {
        std::error_code e;
        fs::create_hard_link(from, to, e);
        if (e && !fs::exists(to)) {
            e.clear();
            fs::copy_file(from, to, fs::copy_options::none, e);
        }
        return e;
    };
    if (auto e = place(staged, out))
        return make_error(ErrorCode::IoError, "Refusing to overwrite existing render: " + out.string(), e.message());
    if (auto e = place(staged_side, side)) {
        fs::remove(out, ec);
        return make_error(ErrorCode::IoError, "Refusing to overwrite existing render: " + side.string(), e.message());
    }
    r->exr = out;
    r->sidecar = side;
    return r;
}

}  // namespace rudra
