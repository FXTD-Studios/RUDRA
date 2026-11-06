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
#include "rudra/core/view.hpp"
#include "rudra/platform/png8.hpp"
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

// The reconstruction region as JSON (spec revision 2), in field order.
nlohmann::json region_to_json(const ReconstructionRegion& r) {
    auto f = [](float v) { return py_round(double(v), 4); };
    return {{"band", r.band},
            {"highlight_from", f(r.highlight_from)}, {"highlight_softness", f(r.highlight_softness)},
            {"highlight_strength", f(r.highlight_strength)},
            {"shadow_below", f(r.shadow_below)}, {"shadow_softness", f(r.shadow_softness)},
            {"shadow_strength", f(r.shadow_strength)},
            {"band_low", f(r.band_low)}, {"band_high", f(r.band_high)}, {"band_softness", f(r.band_softness)}};
}

ReconstructionRegion region_from_json(const nlohmann::json& j) {
    ReconstructionRegion r;
    r.enabled = true;
    auto g = [&](const char* k, float d) { return static_cast<float>(j.value(k, double(d))); };
    r.band = j.value("band", false);
    r.highlight_from = g("highlight_from", r.highlight_from);
    r.highlight_softness = g("highlight_softness", r.highlight_softness);
    r.highlight_strength = g("highlight_strength", r.highlight_strength);
    r.shadow_below = g("shadow_below", r.shadow_below);
    r.shadow_softness = g("shadow_softness", r.shadow_softness);
    r.shadow_strength = g("shadow_strength", r.shadow_strength);
    r.band_low = g("band_low", r.band_low);
    r.band_high = g("band_high", r.band_high);
    r.band_softness = g("band_softness", r.band_softness);
    return r;
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
        if (j.contains("reconstruction_region") && !j.at("reconstruction_region").is_null()) {
            r.region = region_from_json(j.at("reconstruction_region"));
            if (const std::string why = reconstruction_region_error(r.region); !why.empty())
                return make_error(ErrorCode::InvalidArgument, why);
        }
        r.mattes = j.value("mattes", false);
        r.brackets = j.value("brackets", false);
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

std::string reconstruction_region_json(const ReconstructionRegion& r) { return region_to_json(r).dump(); }

Result<ReconstructionRegion> reconstruction_region_from_json(const std::string& json_text) {
    ReconstructionRegion r;
    try {
        const auto j = nlohmann::json::parse(json_text);
        if (!j.is_object()) return make_error(ErrorCode::ParseError, "The reconstruction region is not an object.");
        r = region_from_json(j);
    } catch (const std::exception& e) {
        return make_error(ErrorCode::ParseError, "The reconstruction region is not valid JSON.", e.what());
    }
    if (const std::string why = reconstruction_region_error(r); !why.empty())
        return make_error(ErrorCode::InvalidArgument, why);
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
    if (q.region.enabled) j["reconstruction_region"] = region_to_json(q.region);
    if (q.mattes) j["mattes"] = true;
    if (q.brackets) j["brackets"] = true;
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
    if (q.region.enabled)
        if (const std::string why = reconstruction_region_error(q.region); !why.empty())
            return make_error(ErrorCode::InvalidArgument, why);
    CompositeParams cp;
    cp.mode = *mode;
    cp.strength = static_cast<float>(q.strength);
    cp.preserve_outside = q.preserve_outside;
    cp.region = q.region;
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
    if (q.container != "aces" && q.container != "linear" && q.container != "acescg" && q.container != "rec709")
        return make_error(ErrorCode::InvalidArgument, "container must be aces, acescg, linear or rec709", q.container);
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
    if (q.region.enabled) provenance.emplace_back("rudra:reconstructionRegion", region_to_json(q.region).dump());
    if (!q.exr_half) provenance.emplace_back("rudra:sampleType", "float32");
    if (q.analytic) provenance.emplace_back("rudra:clipFill", clip_fill_text(q));
    if (!cp.reference.empty()) provenance.emplace_back("rudra:reference", cp.reference.file);
    if (cp.masks) provenance.emplace_back("rudra:masks", masks_path.filename().string());
    std::filesystem::path mattes_path, regions_path;
    std::array<double, kPixelClasses> class_share{};
    auto bracket_path = [&](double ev) {
        char b[32];
        std::snprintf(b, sizeof b, ".ev%+.0f.png", ev);
        std::filesystem::path p = out;
        return p.replace_extension(b);
    };
    if (q.brackets) {
        std::string names;
        for (double ev : kBracketStops) names += (names.empty() ? "" : ", ") + bracket_path(ev).filename().string();
        provenance.emplace_back("rudra:brackets", names);
    }
    if (q.mattes) {
        regions_path = out;
        regions_path.replace_extension(".regions.png");
        mattes_path = out;
        mattes_path.replace_extension(".mattes.exr");
        if (!q.mattes_dir.empty()) {   // a movie's frames: the mattes in their own folder
            std::error_code mec;
            std::filesystem::create_directories(q.mattes_dir, mec);
            if (mec) return make_error(ErrorCode::IoError, "Cannot create the mattes folder.", q.mattes_dir.string());
            mattes_path = q.mattes_dir / mattes_path.filename();
            regions_path = q.mattes_dir / regions_path.filename();
        }
        provenance.emplace_back("rudra:mattes", mattes_path.filename().string());
    }
    const bool aces = q.container == "aces";
    // RUDRA 1.1: ACEScg (AP1) and linear Rec.709 beside ACES 2065-1 and linear Rec.2020.
    const bool acescg = q.container == "acescg", rec709 = q.container == "rec709";
    const Primaries container_primaries = aces ? Primaries::Ap0 : acescg ? Primaries::Ap1 : rec709 ? Primaries::Rec709
                                                                                                  : Primaries::Rec2020;
    // The linear container is scene-linear Rec.2020, converted from the
    // plate's primaries and stamped so (it carried the plate's primaries
    // untagged until 9 Oct 2026, as the Studio did).
    const PlanarBuffer* alpha = q.alpha && q.alpha->height() == linear.height() && q.alpha->width() == linear.width()
                                    ? &*q.alpha : nullptr;
    if (q.alpha && !alpha) return make_error(ErrorCode::InvalidArgument, "The alpha is not the size of the picture.");
    if (alpha && !aces && !acescg) provenance.emplace_back("rudra:alpha", "the source's straight alpha as A; RGB premultiplied by it (OpenEXR associated alpha)");
    auto rec2020_rgba = [&] {   // the linear container with A: RGB premultiplied, as the ACES writers do
        PlanarBuffer rgb = convert_primaries(linear, *src, rec709 ? Primaries::Rec709 : Primaries::Rec2020);
        if (!alpha) return rgb;
        PlanarBuffer rgba(4, rgb.height(), rgb.width());
        for (std::size_t i = 0; i < rgb.plane_size(); ++i) {
            const float a = std::clamp(alpha->plane(0)[i], 0.0f, 1.0f);
            for (int c = 0; c < 3; ++c) rgba.plane(c)[i] = rgb.plane(c)[i] * a;
            rgba.plane(3)[i] = a;
        }
        return rgba;
    };
    auto w = aces     ? write_aces_exr(out, linear, *src, 1.0, provenance, q.exr_half, nullptr, {}, alpha)
             : acescg ? write_acescg_exr(out, linear, *src, 1.0, provenance, q.exr_half, nullptr, {}, alpha)
                      : write_exr(out, rec2020_rgba(), q.exr_half, rec709 ? kRec709Chromaticities : kRec2020Chromaticities,
                                  provenance);
    if (!w) return w.error();

    // RUDRA 1.1 priority 4: the brackets from the EXR as written, and the preview-to-export check.
    std::vector<std::filesystem::path> bracket_files;
    CodeDelta bracket_delta;
    if (q.brackets) {
        auto back = read_exr(out);
        if (!back) return back.error();
        const int bh = back->pixels.height(), bw = back->pixels.width();
        PlanarBuffer rgb(3, bh, bw);
        for (int c = 0; c < 3; ++c) std::copy_n(back->pixels.plane(c), rgb.plane_size(), rgb.plane(c));
        rgb = convert_primaries(rgb, container_primaries, *src);
        const float to_net = float(kDiffuseWhite.v) / 10000.0f;   // scene linear (203 nits = 1.0) -> network units
        PlanarBuffer exported(3, bh, bw), shown(3, bh, bw);
        for (std::size_t i = 0; i < rgb.plane_size(); ++i) {
            const float a = back->pixels.channels() == 4 ? back->pixels.plane(3)[i] : 1.0f;
            for (int c = 0; c < 3; ++c) {
                shown.plane(c)[i] = linear.plane(c)[i] * to_net;
                // premultiplied A: unpremultiply; where A is 0 there is no colour to compare
                exported.plane(c)[i] = a > 0.0f ? rgb.plane(c)[i] / a * to_net : shown.plane(c)[i];
            }
        }
        const NetworkLinearImage exp_img(std::move(exported)), shown_img(std::move(shown));
        for (double ev : kBracketStops) {
            const Rgb8Image from_file = exposure_bracket(exp_img, ev, *src);
            const CodeDelta d = code_delta(from_file, exposure_bracket(shown_img, ev, *src));
            bracket_delta.max_code_delta = std::max(bracket_delta.max_code_delta, d.max_code_delta);
            bracket_delta.pixels_differing = std::max(bracket_delta.pixels_differing, d.pixels_differing);
            const auto path = bracket_path(ev);
            if (auto wp = write_png8(path, Png8{from_file.width, from_file.height, 3, from_file.rgb}); !wp) return wp.error();
            bracket_files.push_back(path);
        }
    }

    pyjson::List regions_json;
    for (const auto& b : q.regions) regions_json.push_back(band_json(b));
    // Roadmap 3.5: where this reconstruction departs from what the SDR
    // supports, measured on the composite against its corrected baseline
    // (before the grade, which is the artist's, not the network's).
    const NetworkLinearImage baseline = corrected_baseline(sdr, model.corpus_ev, baseline_curve_params(scalars, cp, model.corpus_ev));
    const SupportStats support = support_stats(network, baseline, sdr);
    pyjson::List matte_names;
    if (q.mattes) {   // RUDRA 1.1 priority 3: the reconstruction's mattes, before the grade
        const CompositeParams plain = ungraded(cp);
        const bool graded_any = any_graded(cp.regions) || cp.exposure_ev != 0.0;
        const NetworkLinearImage recon = graded_any ? composite(sdr, fields, scalars, model, plain) : network;
        ReconstructionMattes m = reconstruction_mattes(sdr, fields, scalars, plain, recon, baseline);
        // RUDRA 1.1 priority 5: what each pixel is, how tightly the SDR pins it, and a picture of both.
        PixelClassMap classes = classify_pixels(sdr, recon, baseline);
        class_share = classes.share;
        {
            const Rgb8Image grey = exposure_bracket(recon, 0.0, *src);
            const int W = sdr.width(), H = sdr.height();
            Png8 pic{W, H, 3, std::vector<std::uint8_t>(std::size_t(W) * std::size_t(H) * 3)};
            auto edge = [&](int x, int y) {   // the region's boundary: weight 0 beside weight > 0
                if (!m.region) return false;
                const float* rw = m.region->plane(0);
                const bool in = rw[std::size_t(y) * std::size_t(W) + std::size_t(x)] > 0.0f;
                for (auto [dx, dy] : {std::pair{1, 0}, std::pair{-1, 0}, std::pair{0, 1}, std::pair{0, -1}}) {
                    const int u = x + dx, v = y + dy;
                    if (u < 0 || v < 0 || u >= W || v >= H) continue;
                    if ((rw[std::size_t(v) * std::size_t(W) + std::size_t(u)] > 0.0f) != in) return true;
                }
                return false;
            };
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    const std::size_t i = std::size_t(y) * std::size_t(W) + std::size_t(x);
                    const float l = (0.2126f * grey.rgb[i * 3] + 0.7152f * grey.rgb[i * 3 + 1] + 0.0722f * grey.rgb[i * 3 + 2]) / 255.0f;
                    const int k = classes.cls[i];
                    const float mix = k >= 2 ? kMapMix : 0.0f;
                    const bool on_edge = edge(x, y);
                    for (int c = 0; c < 3; ++c) {
                        float v = (1.0f - mix) * l * kMapGrey + mix * kPixelClassColour[std::size_t(k)][std::size_t(c)];
                        if (on_edge) v = 1.0f;
                        pic.data[i * 3 + std::size_t(c)] = std::uint8_t(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f));
                    }
                }
            if (auto wp = write_png8(regions_path, pic); !wp) return wp.error();
        }
        PlanarBuffer class_plane(1, sdr.height(), sdr.width());
        for (std::size_t i = 0; i < classes.cls.size(); ++i)
            class_plane.plane(0)[i] = float(classes.cls[i]) / float(kPixelClasses - 1);
        NamedPlanes planes;
        planes.emplace_back("rudra.class", std::move(class_plane));
        planes.emplace_back("rudra.confidence", std::move(classes.confidence));
        planes.emplace_back("rudra.highlight", std::move(m.highlight));
        planes.emplace_back("rudra.shadow", std::move(m.shadow));
        planes.emplace_back("rudra.changed", std::move(m.changed));
        if (m.region) planes.emplace_back("rudra.region", std::move(*m.region));
        if (cp.masks)
            for (int band : cp.masks->bands()) {
                PlanarBuffer p(1, sdr.height(), sdr.width());
                for (int y = 0; y < sdr.height(); ++y)
                    for (int x = 0; x < sdr.width(); ++x)
                        p.at(0, y, x) = cp.masks->weight(band, x, y, sdr.width(), sdr.height());
                planes.emplace_back("rudra.paint." + std::to_string(band), std::move(p));
            }
        for (const auto& pl : planes) matte_names.push_back(pl.first);
        ExrAttributes ma{{"rudra:mattesOf", out.filename().string()},
                         {"rudra:mattes", "highlight / shadow: the gate's priors as applied (mode, region, strength, "
                                          "recovery mask); changed: 1 where the reconstruction differs from the baseline; "
                                          "region: the luminance region; paint.N: painted mask N; class: 0 preserved, "
                                          "0.25 follows, 0.5 reconstructed, 0.75 estimated, 1 generated; confidence: "
                                          "how tightly the SDR pins the value (not a probability). Before the grade."},
                         {"rudra:appVersion", RUDRA_VERSION_LABEL}};
        if (auto wm = write_mattes_exr(mattes_path, planes, ma); !wm) return wm.error();
    }
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
        {"container", aces ? "ACES 2065-1 (AP0)" : acescg ? "ACEScg (AP1)" : rec709 ? "scene-linear Rec.709"
                                                                                 : "scene-linear Rec.2020"},
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
    if (q.brackets) {
        pyjson::List files, evs;
        for (const auto& f : bracket_files) files.push_back(f.filename().string());
        for (double ev : kBracketStops) evs.push_back(ev);
        sidecar_dict.emplace_back("brackets", pyjson::Dict{
            {"files", files}, {"ev", evs},
            {"view", "SDR, sRGB, Rec.709; diffuse white (203 nits) at code 255 at 0 EV; from the EXR as written"},
            {"preview_vs_export", pyjson::Dict{
                 {"max_code_delta", std::int64_t(bracket_delta.max_code_delta)},
                 {"pixels_differing_pct", py_round(100.0 * bracket_delta.pixels_differing, 3)},
                 {"basis", "the same brackets from the picture in memory (what the viewer shows) against the EXR read back; "
                           "a half master differs by its half-float rounding"}}}});
    }
    if (q.mattes) {
        sidecar_dict.emplace_back("mattes", pyjson::Dict{{"file", mattes_path.filename().string()},
                                                         {"channels", matte_names},
                                                         {"basis", "before the grade (no Region EV, no exposure)"}});
        sidecar_dict.emplace_back("pixel_classes", pyjson::Dict{
            {"picture", regions_path.filename().string()},
            {"preserved_pct", pct3(100.0 * class_share[0])},
            {"follows_pct", pct3(100.0 * class_share[1])},
            {"reconstructed_pct", pct3(100.0 * class_share[2])},
            {"estimated_pct", pct3(100.0 * class_share[3])},
            {"generated_pct", pct3(100.0 * class_share[4])},
            {"confidence", "a constraint score: 1 preserved / follows, 1/(1+|d|) reconstructed, "
                           "0.5/(1+|d|) estimated, 0 generated; d in stops. Not a probability"}});
    }
    if (q.region.enabled) {
        const auto rj = region_to_json(q.region);
        pyjson::Dict d;
        for (const auto& [k, v] : rj.items()) {
            if (v.is_boolean()) d.emplace_back(k, v.get<bool>());
            else d.emplace_back(k, v.get<double>());
        }
        d.emplace_back("outside", "the baseline, bit for bit (gate exactly 0)");
        sidecar_dict.emplace_back("reconstruction_region", pyjson::Value(d));
    }
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
    if (cp.masks) r.masks = masks_path;
    if (q.mattes) r.mattes = mattes_path;
    if (q.mattes) r.regions = regions_path;
    r.brackets = bracket_files;
    if (q.brackets) r.bracket_max_code_delta = bracket_delta.max_code_delta;
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
    // 5 Nov 2026: the files written beside the master go with it (the painted masks were
    // written into the staging folder and removed with it before this).
    std::vector<fs::path> placed;
    r->masks.clear();
    if (request.mattes_dir.empty()) {   // written in their own folder already when it is set
        r->mattes.clear();
        r->regions.clear();
    }
    r->brackets.clear();
    for (const char* ext : {".masks.png", ".mattes.exr", ".regions.png", ".ev-2.png", ".ev+0.png", ".ev+2.png"}) {
        fs::path from = staged, to = out;
        from.replace_extension(ext);
        to.replace_extension(ext);
        if (!fs::exists(from)) continue;
        if (auto e = place(from, to)) {
            for (const auto& p : placed) fs::remove(p, ec);
            fs::remove(side, ec);
            fs::remove(out, ec);
            return make_error(ErrorCode::IoError, "Refusing to overwrite existing render: " + to.string(), e.message());
        }
        placed.push_back(to);
        const std::string e(ext);
        if (e == ".masks.png") r->masks = to;
        else if (e == ".mattes.exr") r->mattes = to;
        else if (e == ".regions.png") r->regions = to;
        else r->brackets.push_back(to);
    }
    r->exr = out;
    r->sidecar = side;
    return r;
}

}  // namespace rudra
