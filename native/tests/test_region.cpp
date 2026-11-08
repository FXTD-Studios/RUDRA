// The reconstruction region (docs/composite.spec.md revision 2, RUDRA 1.1, 5 Nov 2026):
// core reconstruction_gate against its Python oracle (rudra/delivery/region.py, golden
// from tools/emit_region_golden.py), the composite's bit-exact preservation outside
// the region, and the region through masters, projects and the session.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include "rudra/core/baseline.hpp"
#include "rudra/core/composite.hpp"
#include "rudra/core/view.hpp"
#include "rudra/deliver/exr.hpp"
#include "rudra/deliver/master.hpp"
#include "rudra/deliver/sequence_encode.hpp"
#include "rudra/engine/session.hpp"
#include "rudra/platform/npy.hpp"

using namespace rudra;
namespace fs = std::filesystem;
using nlohmann::json;

namespace {

const fs::path kDir = fs::path(RUDRA_GOLDEN_DIR) / "composite";

json read_json(const fs::path& p) {
    std::ifstream in(p);
    return json::parse(in);
}

ReconstructionRegion region_of(const json& j) {
    ReconstructionRegion r;
    r.enabled = j.at("enabled").get<bool>();
    r.band = j.at("band").get<bool>();
    r.highlight_from = j.at("highlight_from").get<float>();
    r.highlight_softness = j.at("highlight_softness").get<float>();
    r.highlight_strength = j.at("highlight_strength").get<float>();
    r.shadow_below = j.at("shadow_below").get<float>();
    r.shadow_softness = j.at("shadow_softness").get<float>();
    r.shadow_strength = j.at("shadow_strength").get<float>();
    r.band_low = j.at("band_low").get<float>();
    r.band_high = j.at("band_high").get<float>();
    r.band_softness = j.at("band_softness").get<float>();
    return r;
}

RecoveryMode mode_of(const std::string& s) {
    if (s == "highlights") return RecoveryMode::Highlights;
    if (s == "shadows") return RecoveryMode::Shadows;
    if (s == "off") return RecoveryMode::Off;
    return RecoveryMode::All;
}

PlanarBuffer planar(const fs::path& file) {
    auto a = read_npy(file);
    EXPECT_TRUE(a) << file;
    if (!a) return PlanarBuffer(3, 1, 1);
    return PlanarBuffer(int(a->shape[0]), int(a->shape[1]), int(a->shape[2]), a->data);
}

struct Frame {
    SdrImage sdr;
    Fields fields;
    FrameScalars scalars;
    ModelConstants model;
};

// The "main" golden frame of tools/emit_composite_golden.py: a real model's fields.
Frame main_frame() {
    const json idx = read_json(kDir / "index.json");
    const json& f = idx.at("frames").at("main");
    auto file = [&](const char* k) { return kDir / f.at(k).at("file").get<std::string>(); };
    Frame fr{SdrImage(planar(file("sdr"))),
             Fields{planar(file("residual")), planar(file("highlight")), planar(file("shadow"))}, FrameScalars{},
             ModelConstants{idx.at("model").at("log_scale").get<float>(), idx.at("model").at("max_hdr").get<float>(),
                            idx.at("model").at("corpus_ev").get<float>()}};
    fr.scalars.shadow_weight = f.at("shadow_weight").get<float>();
    fr.scalars.curve_params = f.at("curve_params").get<std::vector<float>>();
    return fr;
}

float luma(const SdrImage& s, std::size_t i) {
    auto c = [&](int k) { return std::clamp(s.buffer().plane(k)[i], 0.0f, 1.0f); };
    return 0.2126f * c(0) + 0.7152f * c(1) + 0.0722f * c(2);
}

}  // namespace

TEST(ReconstructionRegion, GateMatchesThePythonOracle) {
    const json g = read_json(kDir / "region_gate.json");
    const auto y = g.at("y").get<std::vector<float>>();
    for (const auto& c : g.at("cases")) {
        CompositeParams p;
        p.region = region_of(c.at("region"));
        p.mode = mode_of(c.at("mode").get<std::string>());
        p.strength = c.at("strength").get<float>();
        const float sw = c.at("shadow_weight").get<float>();
        const auto want = c.at("gate").get<std::vector<float>>();
        ASSERT_EQ(want.size(), y.size());
        for (std::size_t i = 0; i < y.size(); ++i) {
            const float got = reconstruction_gate(y[i], sw, p);
            if (want[i] == 0.0f) ASSERT_EQ(got, 0.0f) << c.at("name") << " y " << y[i];   // the compact support is exact
            else ASSERT_NEAR(got, want[i], 2e-6f * std::max(1.0f, want[i])) << c.at("name") << " y " << y[i];
        }
    }
}

TEST(ReconstructionRegion, OffIsRevisionOneBitForBit) {
    const Frame fr = main_frame();
    CompositeParams rev1, off;
    off.region.highlight_from = 0.3f;   // ignored while off
    off.region.band = true;
    const auto a = composite(fr.sdr, fr.fields, fr.scalars, fr.model, rev1);
    const auto b = composite(fr.sdr, fr.fields, fr.scalars, fr.model, off);
    const auto sa = a.buffer().span(), sb = b.buffer().span();
    ASSERT_EQ(sa.size(), sb.size());
    for (std::size_t i = 0; i < sa.size(); ++i) ASSERT_EQ(sa[i], sb[i]) << i;
}

TEST(ReconstructionRegion, ARegionThatAdmitsEverythingIsRevisionOne) {
    const Frame fr = main_frame();
    CompositeParams rev1, all;
    all.region.enabled = true;
    all.region.band = true;
    all.region.band_low = 0.0f;
    all.region.band_high = 1.0f;
    all.region.band_softness = 0.0f;   // weight 1 at every luma, strengths 1
    const auto a = composite(fr.sdr, fr.fields, fr.scalars, fr.model, rev1);
    const auto b = composite(fr.sdr, fr.fields, fr.scalars, fr.model, all);
    const auto sa = a.buffer().span(), sb = b.buffer().span();
    for (std::size_t i = 0; i < sa.size(); ++i) ASSERT_EQ(sa[i], sb[i]) << i;
}

TEST(ReconstructionRegion, OutsideTheRegionIsTheBaselineBitForBit) {
    const Frame fr = main_frame();
    for (bool preserve : {true, false}) {
        CompositeParams p;
        p.preserve_outside = preserve;
        p.region.enabled = true;
        p.region.highlight_from = 0.9f;
        p.region.highlight_softness = 0.05f;
        p.region.shadow_below = 0.05f;
        p.region.shadow_softness = 0.02f;
        const auto out = composite(fr.sdr, fr.fields, fr.scalars, fr.model, p);
        const auto base = corrected_baseline(fr.sdr, fr.model.corpus_ev,
                                             baseline_curve_params(fr.scalars, p, fr.model.corpus_ev));
        const std::size_t n = std::size_t(fr.sdr.height()) * std::size_t(fr.sdr.width());
        std::size_t outside = 0, inside = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const float gate = reconstruction_gate(luma(fr.sdr, i), fr.scalars.shadow_weight, p);
            if (gate != 0.0f) {
                ++inside;
                continue;
            }
            ++outside;
            for (int c = 0; c < 3; ++c) ASSERT_EQ(out.buffer().plane(c)[i], base.buffer().plane(c)[i]) << i;
        }
        EXPECT_GT(outside, 0u);   // the frame has both, or the test proves nothing
        EXPECT_GT(inside, 0u);
    }
}

TEST(ReconstructionRegion, HighlightAndShadowStrengthsAreIndependent) {
    CompositeParams p;
    p.region.enabled = true;
    p.region.shadow_strength = 0.0f;
    EXPECT_EQ(reconstruction_gate(0.02f, 1.0f, p), 0.0f);    // shadows off
    EXPECT_GT(reconstruction_gate(0.98f, 1.0f, p), 0.9f);    // highlights untouched
    p.region.shadow_strength = 1.0f;
    p.region.highlight_strength = 0.0f;
    EXPECT_EQ(reconstruction_gate(0.98f, 1.0f, p), 0.0f);
    EXPECT_GT(reconstruction_gate(0.02f, 1.0f, p), 0.5f);
    EXPECT_EQ(reconstruction_gate(0.5f, 1.0f, p), 0.0f);     // mid-tones: outside both ramps
}

TEST(ReconstructionRegion, ChecksRefuseBadValues) {
    ReconstructionRegion r;
    EXPECT_TRUE(reconstruction_region_error(r).empty());
    r.band_low = 0.9f;
    r.band_high = 0.5f;
    EXPECT_FALSE(reconstruction_region_error(r).empty());
    r = {};
    r.highlight_softness = 0.8f;
    EXPECT_FALSE(reconstruction_region_error(r).empty());
    r = {};
    r.shadow_strength = std::nanf("");
    EXPECT_FALSE(reconstruction_region_error(r).empty());
}

TEST(ReconstructionRegion, MasterRequestCarriesItOnlyWhenOn) {
    MasterRequest q;
    EXPECT_EQ(master_request_json(q).find("reconstruction_region"), std::string::npos);
    q.region.enabled = true;
    q.region.band = true;
    q.region.band_low = 0.6f;
    q.region.band_high = 0.95f;
    q.region.highlight_strength = 1.5f;
    const auto back = master_request_from_json(master_request_json(q));
    ASSERT_TRUE(back) << back.error().message;
    EXPECT_TRUE(back->region.enabled);
    EXPECT_TRUE(back->region.band);
    EXPECT_FLOAT_EQ(back->region.band_low, 0.6f);
    EXPECT_FLOAT_EQ(back->region.band_high, 0.95f);
    EXPECT_FLOAT_EQ(back->region.highlight_strength, 1.5f);
    auto j = json::parse(master_request_json(q));
    j["reconstruction_region"]["band_low"] = 2.0;
    EXPECT_FALSE(master_request_from_json(j.dump()));   // refused by name, not clamped
}

TEST(ReconstructionRegion, SessionUndoesAndClamps) {
    Session s;
    EXPECT_FALSE(s.composite_params().region.enabled);
    s.set_region_enabled(true);
    EXPECT_TRUE(s.composite_params().region.enabled);
    ReconstructionRegion r = s.grade.region;
    r.highlight_from = 3.0f;
    r.shadow_strength = -1.0f;
    s.region_press();
    s.region_input(r);
    EXPECT_FLOAT_EQ(s.grade.region.highlight_from, 1.0f);
    EXPECT_FLOAT_EQ(s.grade.region.shadow_strength, 0.0f);
    EXPECT_TRUE(reconstruction_region_error(s.grade.region).empty());
    s.undo();
    EXPECT_FLOAT_EQ(s.grade.region.highlight_from, ReconstructionRegion{}.highlight_from);
    s.undo();
    EXPECT_FALSE(s.grade.region.enabled);
}

// ---- RUDRA 1.1 priority 3 (5 Nov 2026): the reconstruction's mattes ----

TEST(ReconstructionMattes, ZeroMattesMeanTheBaselineAndChangedIsExact) {
    const Frame fr = main_frame();
    CompositeParams p;
    p.region.enabled = true;
    p.region.highlight_from = 0.9f;
    p.region.highlight_softness = 0.05f;
    p.region.shadow_below = 0.05f;
    p.region.shadow_softness = 0.02f;
    const auto recon = composite(fr.sdr, fr.fields, fr.scalars, fr.model, p);
    const auto base = corrected_baseline(fr.sdr, fr.model.corpus_ev, baseline_curve_params(fr.scalars, p, fr.model.corpus_ev));
    const ReconstructionMattes m = reconstruction_mattes(fr.sdr, fr.fields, fr.scalars, p, recon, base);
    ASSERT_TRUE(m.region.has_value());
    const std::size_t n = std::size_t(fr.sdr.height()) * std::size_t(fr.sdr.width());
    std::size_t zero = 0, changed = 0;
    for (std::size_t i = 0; i < n; ++i) {
        bool differs = false;
        for (int c = 0; c < 3; ++c) differs = differs || recon.buffer().plane(c)[i] != base.buffer().plane(c)[i];
        ASSERT_EQ(m.changed.plane(0)[i], differs ? 1.0f : 0.0f) << i;          // exact by construction
        for (const PlanarBuffer* b : {&m.highlight, &m.shadow, &m.changed, &*m.region})
            ASSERT_TRUE(b->plane(0)[i] >= 0.0f && b->plane(0)[i] <= 1.0f) << i;
        if (m.highlight.plane(0)[i] == 0.0f && m.shadow.plane(0)[i] == 0.0f) {
            ++zero;
            ASSERT_FALSE(differs) << i;                                          // no matte, no change
        }
        changed += differs;
    }
    EXPECT_GT(zero, 0u);
    EXPECT_GT(changed, 0u);
}

TEST(ReconstructionMattes, TheModeDecidesWhichMatteIsLit) {
    const Frame fr = main_frame();
    CompositeParams p;
    p.mode = RecoveryMode::Highlights;
    const auto recon = composite(fr.sdr, fr.fields, fr.scalars, fr.model, p);
    const auto base = corrected_baseline(fr.sdr, fr.model.corpus_ev, baseline_curve_params(fr.scalars, p, fr.model.corpus_ev));
    const ReconstructionMattes m = reconstruction_mattes(fr.sdr, fr.fields, fr.scalars, p, recon, base);
    EXPECT_FALSE(m.region.has_value());
    for (float v : m.shadow.span()) ASSERT_EQ(v, 0.0f);
    EXPECT_TRUE(std::any_of(m.highlight.span().begin(), m.highlight.span().end(), [](float v) { return v > 0.0f; }));
}

TEST(ReconstructionMattes, TheMattesExrRoundTripsAndRefusesBadPlanes) {
    const fs::path dir = fs::temp_directory_path() / "rudra-mattes-exr";
    fs::remove_all(dir);
    PlanarBuffer a(1, 3, 4), b(1, 3, 4);
    for (std::size_t i = 0; i < a.plane_size(); ++i) {
        a.plane(0)[i] = float(i) / 11.0f;
        b.plane(0)[i] = 1.0f - float(i) / 11.0f;
    }
    NamedPlanes planes{{"rudra.shadow", a}, {"rudra.highlight", b}};
    ASSERT_TRUE(write_mattes_exr(dir / "m.mattes.exr", planes, {{"rudra:mattesOf", "m.exr"}}));
    auto img = read_exr(dir / "m.mattes.exr");
    ASSERT_TRUE(img) << img.error().message;
    ASSERT_EQ(img->other.size(), 2u);
    EXPECT_EQ(img->other[0].first, "rudra.highlight");                          // name order, as EXR asks
    for (const auto& [name, p] : img->other) {
        const PlanarBuffer& want = name == "rudra.shadow" ? a : b;
        for (std::size_t i = 0; i < want.plane_size(); ++i) ASSERT_EQ(p.plane(0)[i], want.plane(0)[i]) << name;   // float32: exact
    }
    PlanarBuffer bad(1, 3, 4, 1.5f);
    EXPECT_FALSE(write_mattes_exr(dir / "bad.exr", {{"rudra.x", bad}}));
    EXPECT_FALSE(write_mattes_exr(dir / "dup.exr", {{"rudra.x", a}, {"rudra.x", b}}));
    EXPECT_FALSE(write_mattes_exr(dir / "r.exr", {{"R", a}}));
    fs::remove_all(dir);
}

TEST(ReconstructionMattes, PublishedBesideTheMasterWithThePaintedMasks) {
    const Frame fr = main_frame();
    const fs::path dir = fs::temp_directory_path() / "rudra-mattes-publish";
    fs::remove_all(dir);
    MasterRequest q;
    q.analytic = true;
    q.mattes = true;
    q.region.enabled = true;
    auto masks = std::make_shared<MaskSet>();
    masks->width = fr.sdr.width();
    masks->height = fr.sdr.height();
    auto plane = empty_plane(masks->width, masks->height);
    (*plane)[0] = 255;
    masks->planes[1] = plane;
    q.masks = masks;
    q.regions = {{"highlights", 400.0, 2000.0, 0.5}, {"speculars", 2000.0, 8000.0, 0.0}};
    const Fields fields = analytic_fields_for(fr.sdr, q);
    const auto r = publish_master(fr.sdr, 8, fields, FrameScalars{}, kAnalyticConstants, q, dir / "SH010.exr");
    ASSERT_TRUE(r) << r.error().message << " " << r.error().detail;
    EXPECT_TRUE(fs::exists(dir / "SH010.mattes.exr"));
    EXPECT_TRUE(fs::exists(dir / "SH010.masks.png"));                           // was lost with the staging folder
    EXPECT_EQ(r->mattes, dir / "SH010.mattes.exr");
    auto img = read_exr(dir / "SH010.mattes.exr");
    ASSERT_TRUE(img);
    std::vector<std::string> names;
    for (const auto& [n, p] : img->other) names.push_back(n);
    EXPECT_EQ(names, (std::vector<std::string>{"rudra.changed", "rudra.class", "rudra.confidence", "rudra.highlight",
                                               "rudra.paint.1", "rudra.region", "rudra.shadow"}));
    EXPECT_TRUE(fs::exists(dir / "SH010.regions.png"));                         // priority 5: the class picture
    EXPECT_EQ(r->regions, dir / "SH010.regions.png");
    const json sj = read_json(dir / "SH010.json");   // closed before remove_all: Windows will not delete an open file
    EXPECT_EQ(sj.at("mattes").at("file"), "SH010.mattes.exr");
    // a second publish refuses, and leaves nothing half-written
    EXPECT_FALSE(publish_master(fr.sdr, 8, fields, FrameScalars{}, kAnalyticConstants, q, dir / "SH010.exr"));
    EXPECT_TRUE(fs::exists(dir / "SH010.mattes.exr"));
    fs::remove_all(dir);
}

// ---- RUDRA 1.1 priority 4 (5 Nov 2026): EV brackets, preview vs export, the display signal ----

TEST(Brackets, TwoStopsDownShowsTheHighlightsAndUpTheShadows) {
    // network units: 1.0 = 10 000 nits. 203 (SDR white), 812 (two stops over), 2 nits (deep shadow).
    PlanarBuffer p(3, 1, 3);
    const float nits[3] = {203.0f, 812.0f, 2.0f};
    for (int x = 0; x < 3; ++x)
        for (int c = 0; c < 3; ++c) p.at(c, 0, x) = nits[x] / 10000.0f;
    const NetworkLinearImage img(std::move(p));
    const Rgb8Image down = exposure_bracket(img, -2.0), zero = exposure_bracket(img, 0.0), up = exposure_bracket(img, 2.0);
    EXPECT_EQ(zero.rgb[0], 255);                        // the SDR white at 0 EV
    EXPECT_EQ(zero.rgb[3], 255);                        // 812 clips at 0 EV ...
    EXPECT_GE(down.rgb[3], 254);                        // ... and is white again two stops down
    EXPECT_LT(down.rgb[0], 160);                        // where 203 sits near mid grey
    EXPECT_GT(up.rgb[6], zero.rgb[6]);                  // +2 lifts the shadow
    EXPECT_EQ(code_delta(zero, zero).max_code_delta, 0);
    EXPECT_DOUBLE_EQ(code_delta(zero, down).pixels_differing, 2.0 / 3.0);   // 812 nits is white in both
}

TEST(Brackets, TheDisplaySignalSaysTransferPrimariesAndWhite) {
    const std::string pq = display_signal_text(DisplayTarget::hdr10(1000.0), 1000.0);
    EXPECT_NE(pq.find("PQ (ST 2084)"), std::string::npos);
    EXPECT_NE(pq.find("Rec.2020"), std::string::npos);
    EXPECT_NE(pq.find("reference white 203 nits"), std::string::npos);
    EXPECT_NE(pq.find("display peak 1,000 nits"), std::string::npos);
    EXPECT_NE(display_signal_text(DisplayTarget::scrgb(800.0), 800.0).find("scRGB linear (1.0 = 80 nits)"), std::string::npos);
    EXPECT_NE(display_signal_text(DisplayTarget::edr(1600.0), 1600.0).find("P3-D65"), std::string::npos);
    EXPECT_NE(display_signal_text(DisplayTarget::sdr(), 6496.0).find("6,496 nits shown as SDR white"), std::string::npos);
}

TEST(Brackets, PublishedWithTheMasterAndThePreviewMatchesTheExport) {
    const Frame fr = main_frame();
    for (bool half : {false, true}) {
        for (const char* container : {"aces", "linear"}) {
            const fs::path dir = fs::temp_directory_path() / (std::string("rudra-brackets-") + container + (half ? "-h" : "-f"));
            fs::remove_all(dir);
            MasterRequest q;
            q.analytic = true;
            q.brackets = true;
            q.exr_half = half;
            q.container = container;
            const Fields fields = analytic_fields_for(fr.sdr, q);
            const auto r = publish_master(fr.sdr, 8, fields, FrameScalars{}, kAnalyticConstants, q, dir / "SH010.exr");
            ASSERT_TRUE(r) << r.error().message << " " << r.error().detail;
            ASSERT_EQ(r->brackets.size(), 3u);
            for (const char* f : {"SH010.ev-2.png", "SH010.ev+0.png", "SH010.ev+2.png"}) EXPECT_TRUE(fs::exists(dir / f)) << f;
            const json sj = read_json(dir / "SH010.json");   // closed before remove_all: Windows will not delete an open file
            const int delta = sj.at("brackets").at("preview_vs_export").at("max_code_delta").get<int>();
            std::cout << container << (half ? " half" : " float") << ": max code delta " << delta << ", "
                      << sj.at("brackets").at("preview_vs_export").at("pixels_differing_pct") << " % differ\n";
            EXPECT_LE(delta, half ? 1 : 0) << container;   // the export is what was previewed (half: its rounding)
            fs::remove_all(dir);
        }
    }
}

// ---- RUDRA 1.1 priority 5 (5 Nov 2026): pixel classes and confidence ----

TEST(PixelClasses, EveryPixelHasOneClassAndTheyAgreeWithTheMattes) {
    const Frame fr = main_frame();
    CompositeParams p;
    p.region.enabled = true;
    p.region.highlight_from = 0.9f;
    p.region.highlight_softness = 0.05f;
    p.region.shadow_below = 0.05f;
    p.region.shadow_softness = 0.02f;
    const auto recon = composite(fr.sdr, fr.fields, fr.scalars, fr.model, p);
    const auto base = corrected_baseline(fr.sdr, fr.model.corpus_ev, baseline_curve_params(fr.scalars, p, fr.model.corpus_ev));
    const ReconstructionMattes m = reconstruction_mattes(fr.sdr, fr.fields, fr.scalars, p, recon, base);
    const PixelClassMap c = classify_pixels(fr.sdr, recon, base);
    double total = 0.0;
    for (double s : c.share) total += s;
    EXPECT_NEAR(total, 1.0, 1e-12);
    EXPECT_GT(c.share[0], 0.0);
    EXPECT_EQ(c.share[4], 0.0);                                         // no AI fill given
    for (std::size_t i = 0; i < c.cls.size(); ++i) {
        const float conf = c.confidence.plane(0)[i];
        ASSERT_TRUE(conf >= 0.0f && conf <= 1.0f);
        ASSERT_EQ(c.cls[i] == 0, m.changed.plane(0)[i] == 0.0f) << i;  // preserved == unchanged, exactly
        if (c.cls[i] <= 1) ASSERT_EQ(conf, 1.0f);
        if (c.cls[i] == 3) ASSERT_LE(conf, 0.5f);                       // the SDR only bounds an estimate
    }
}

TEST(PixelClasses, TheAiFillWinsAndHasNoConfidence) {
    const Frame fr = main_frame();
    CompositeParams p;
    const auto recon = composite(fr.sdr, fr.fields, fr.scalars, fr.model, p);
    const auto base = corrected_baseline(fr.sdr, fr.model.corpus_ev, baseline_curve_params(fr.scalars, p, fr.model.corpus_ev));
    PlanarBuffer gen(1, fr.sdr.height(), fr.sdr.width());
    gen.plane(0)[0] = 0.4f;
    const PixelClassMap c = classify_pixels(fr.sdr, recon, base, &gen);
    EXPECT_EQ(c.cls[0], std::uint8_t(PixelClass::Generated));
    EXPECT_EQ(c.confidence.plane(0)[0], 0.0f);
    EXPECT_DOUBLE_EQ(c.share[4], 1.0 / double(c.cls.size()));
}

// ---- RUDRA 1.1, the workflow gaps (5 Nov 2026) ----

TEST(Gaps, AFramesFolderNeverTakesTheMattesAsFrames) {
    const fs::path dir = fs::temp_directory_path() / "rudra-list-mattes";
    fs::remove_all(dir);
    fs::create_directories(dir);
    for (const char* n : {"m.000001.exr", "m.000001.mattes.exr", "m.000002.exr", "m.000002.MATTES.exr"})
        std::ofstream(dir / n) << "x";
    auto l = list_linear_frames(dir);
    ASSERT_TRUE(l);
    ASSERT_EQ(l->size(), 2u);
    EXPECT_EQ((*l)[0].filename(), "m.000001.exr");
    EXPECT_EQ((*l)[1].filename(), "m.000002.exr");
    fs::remove_all(dir);
}

TEST(Gaps, AMoviesMattesGoToTheirOwnFolder) {
    const Frame fr = main_frame();
    const fs::path dir = fs::temp_directory_path() / "rudra-movie-mattes";
    fs::remove_all(dir);
    MasterRequest q;
    q.analytic = true;
    q.mattes = true;
    q.container = "linear";
    q.mattes_dir = dir / "SH010_mattes";
    const auto r = publish_master(fr.sdr, 8, analytic_fields_for(fr.sdr, q), FrameScalars{}, kAnalyticConstants, q,
                                  dir / "SH010_frames" / "SH010.000001.exr");
    ASSERT_TRUE(r) << r.error().message;
    EXPECT_TRUE(fs::exists(dir / "SH010_mattes" / "SH010.000001.mattes.exr"));
    EXPECT_TRUE(fs::exists(dir / "SH010_mattes" / "SH010.000001.regions.png"));
    EXPECT_FALSE(fs::exists(dir / "SH010_frames" / "SH010.000001.mattes.exr"));
    EXPECT_EQ(r->mattes, dir / "SH010_mattes" / "SH010.000001.mattes.exr");
    auto l = list_linear_frames(dir / "SH010_frames");
    ASSERT_TRUE(l);
    EXPECT_EQ(l->size(), 1u);
    fs::remove_all(dir);
}

TEST(Gaps, ACEScgAndRec709ContainersRoundTripToThePreview) {
    const Frame fr = main_frame();
    for (const char* container : {"acescg", "rec709"}) {
        const fs::path dir = fs::temp_directory_path() / (std::string("rudra-container-") + container);
        fs::remove_all(dir);
        MasterRequest q;
        q.analytic = true;
        q.container = container;
        q.exr_half = false;
        q.brackets = true;
        const auto r = publish_master(fr.sdr, 8, analytic_fields_for(fr.sdr, q), FrameScalars{}, kAnalyticConstants, q,
                                      dir / "SH010.exr");
        ASSERT_TRUE(r) << container << ": " << r.error().message;
        const json sj = read_json(dir / "SH010.json");   // closed before remove_all: Windows will not delete an open file
        EXPECT_EQ(sj.at("brackets").at("preview_vs_export").at("max_code_delta").get<int>(), 0) << container;
        EXPECT_EQ(sj.at("container").get<std::string>(), std::string(container) == "acescg" ? "ACEScg (AP1)" : "scene-linear Rec.709");
        auto img = read_exr(dir / "SH010.exr");
        ASSERT_TRUE(img);
        const auto it = std::find_if(img->attributes.begin(), img->attributes.end(), [](const auto& a) { return a.first == "chromaticities"; });
        ASSERT_NE(it, img->attributes.end());
        float red_x = 0.0f;
        std::memcpy(&red_x, it->second.data(), 4);
        EXPECT_NEAR(red_x, std::string(container) == "acescg" ? 0.713f : 0.640f, 1e-6f) << container;
        fs::remove_all(dir);
    }
    MasterRequest bad;
    bad.container = "srgb";
    const Frame fr2 = main_frame();
    EXPECT_FALSE(write_master(fr2.sdr, 8, analytic_fields(fr2.sdr.height(), fr2.sdr.width()), FrameScalars{}, kAnalyticConstants, bad,
                              fs::temp_directory_path() / "rudra-bad-container.exr"));
}

TEST(Gaps, TheRegionOverlayTintsExactlyWhereTheRegionActs) {
    const Frame fr = main_frame();
    ReconstructionRegion rg;
    rg.enabled = true;
    rg.highlight_from = 0.9f;
    rg.highlight_softness = 0.05f;
    rg.shadow_strength = 0.0f;
    const int w = fr.sdr.width(), h = fr.sdr.height();
    const auto bytes = region_overlay_rgba8(&fr.sdr, rg, w, h);
    ASSERT_EQ(bytes.size(), std::size_t(w) * std::size_t(h) * 4);
    CompositeParams p;
    p.region = rg;
    for (std::size_t i = 0; i < std::size_t(w) * std::size_t(h); ++i) {
        const float gate = reconstruction_gate(luma(fr.sdr, i), fr.scalars.shadow_weight, p);
        if (gate == 0.0f) ASSERT_EQ(bytes[i * 4], 0) << i;   // no tint where the composite does nothing
    }
    const auto recon = composite(fr.sdr, fr.fields, fr.scalars, fr.model, p);
    ViewParams off = view_params(ViewMode::Image, 1000.0), on = off;
    on.region_overlay = true;
    on.overlay_region = rg;
    const PlanarBuffer a = render_view(recon, recon, off, &fr.sdr), b = render_view(recon, recon, on, &fr.sdr);
    std::size_t tinted = 0;
    for (std::size_t i = 0; i < std::size_t(w) * std::size_t(h); ++i) {
        const bool same = a.plane(0)[i] == b.plane(0)[i] && a.plane(1)[i] == b.plane(1)[i] && a.plane(2)[i] == b.plane(2)[i];
        ASSERT_EQ(same, bytes[i * 4] == 0) << i;
        tinted += !same;
    }
    EXPECT_GT(tinted, 0u);
    const RegionHistogram hist = region_histogram(fr.sdr, rg);
    double total = 0.0;
    for (double c : hist.counts) total += c;
    EXPECT_NEAR(total, 1.0, 1e-9);
    std::size_t reached = 0;   // 7 Nov 2026: what the tint shows (an 8-bit weight above 0)
    for (std::size_t i = 0; i < std::size_t(w) * std::size_t(h); ++i) reached += bytes[i * 4] > 0;
    EXPECT_NEAR(hist.reached, double(reached) / double(std::size_t(w) * std::size_t(h)), 1e-12);
    EXPECT_EQ(hist.weight.front(), 0.0f);
    EXPECT_EQ(hist.weight.back(), 1.0f);
}

// 7 Nov 2026 (release review R1): on the analytic route the tint is as strong as the composite's
// gate, not the region's own weight: a region from 0.5 tinted the mid-tones fully where the
// built-in priors leave the reconstruction under 5 %, and Highlights mode tinted the shadows.
TEST(Gaps, OnTheAnalyticRouteTheTintIsTheGate) {
    const Frame fr = main_frame();
    ReconstructionRegion rg;
    rg.enabled = true;
    rg.highlight_from = 0.5f;
    rg.highlight_softness = 0.1f;
    rg.shadow_below = 0.15f;
    const int w = fr.sdr.width(), h = fr.sdr.height();
    const std::size_t n = std::size_t(w) * std::size_t(h);
    for (RecoveryMode mode : {RecoveryMode::All, RecoveryMode::Highlights, RecoveryMode::Shadows}) {
        RegionOverlayGate g;
        g.analytic = true;
        g.mode = mode;
        g.strength = 1.5f;
        g.shadow_weight = fr.scalars.shadow_weight;
        const auto gate_bytes = region_overlay_rgba8(&fr.sdr, rg, w, h, g);
        const auto own_bytes = region_overlay_rgba8(&fr.sdr, rg, w, h);   // learned route, AI fill
        CompositeParams p;
        p.region = rg;
        p.mode = mode;
        p.strength = 1.5f;
        for (std::size_t i = 0; i < n; ++i) {
            const float y = luma(fr.sdr, i);
            const float gate = std::clamp(reconstruction_gate(y, fr.scalars.shadow_weight, p), 0.0f, 1.0f);
            ASSERT_EQ(gate_bytes[i * 4], std::uint8_t(std::lround(gate * 255.0f))) << i;
            if (own_bytes[i * 4] == 0) ASSERT_EQ(gate_bytes[i * 4], 0) << i;   // never outside the region
        }
        const RegionHistogram hg = region_histogram(fr.sdr, rg, 64, g), ho = region_histogram(fr.sdr, rg);
        EXPECT_LE(hg.reached, ho.reached + 1e-12);
        EXPECT_LE(hg.strong, hg.reached);
        const std::size_t mid = std::size_t(0.6 * 64);   // luma 0.6: inside the region, the prior near 0
        EXPECT_EQ(ho.weight[mid], 1.0f);
        EXPECT_LT(hg.weight[mid], 0.05f);
        if (mode == RecoveryMode::Highlights) EXPECT_EQ(hg.weight[1], 0.0f);   // the shadows: no tint
        if (mode == RecoveryMode::Shadows) EXPECT_EQ(hg.weight.back(), 0.0f);
    }
}

// 8 Nov 2026: the tint tells the shadow side from the highlight side (G of the overlay), and the
// histogram says which side each bin is on, on both routes.
TEST(Gaps, TheTintTellsShadowsFromHighlights) {
    const int w = 256, h = 4;
    SdrImage sdr(h, w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) sdr.buffer().at(c, y, x) = float(x) / 255.0f;   // a grey ramp, luma = x / 255
    ReconstructionRegion rg;
    rg.enabled = true;
    rg.highlight_from = 0.8f;
    rg.highlight_softness = 0.05f;
    rg.shadow_below = 0.2f;
    rg.shadow_softness = 0.05f;
    for (bool analytic : {false, true}) {
        RegionOverlayGate g;
        g.analytic = analytic;
        const auto bytes = region_overlay_rgba8(&sdr, rg, w, h, g);
        std::size_t shadows = 0, highlights = 0;
        for (int x = 0; x < w; ++x) {
            const std::size_t i = std::size_t(x) * 4;
            if (bytes[i] == 0) continue;
            if (x < 128) { EXPECT_EQ(bytes[i + 1], 255) << x; ++shadows; }
            else { EXPECT_EQ(bytes[i + 1], 0) << x; ++highlights; }
        }
        EXPECT_GT(shadows + highlights, 0u);
        if (!analytic) {
            EXPECT_GT(shadows, 0u);
            EXPECT_GT(highlights, 0u);
        }
        const RegionHistogram hist = region_histogram(sdr, rg, 64, g);
        ASSERT_EQ(hist.shadow.size(), 64u);
        EXPECT_EQ(hist.shadow.front(), 1);
        EXPECT_EQ(hist.shadow.back(), 0);
    }
    // In the viewer: a shadow pixel goes toward blue, a highlight one toward warm.
    ViewParams on;
    on.mode = ViewMode::Image;
    on.region_overlay = true;
    on.overlay_region = rg;
    EXPECT_GT(on.region_shadow_tint[2], on.region_shadow_tint[0]);
    EXPECT_GT(on.region_tint[0], on.region_tint[2]);
}
