// The reconstruction region (docs/composite.spec.md revision 2, RUDRA 1.1, 5 Nov 2026):
// core reconstruction_gate against its Python oracle (rudra/delivery/region.py, golden
// from tools/emit_region_golden.py), the composite's bit-exact preservation outside
// the region, and the region through masters, projects and the session.

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "rudra/core/baseline.hpp"
#include "rudra/core/composite.hpp"
#include "rudra/deliver/master.hpp"
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
