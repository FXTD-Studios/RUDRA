// The master exposure (13 Oct 2026): one control, in stops, that moves the
// whole picture. It is part of the grade -- after the anchor, before the
// Region EV bands, which read the exposed nits -- so the view (composite),
// the master (grade_master / apply_region_ev) and the anchor's target agree,
// and at 0 EV everything is bit for bit what it was.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>

#include "rudra/core/composite.hpp"
#include "rudra/core/fields.hpp"
#include "rudra/core/master.hpp"
#include "rudra/deliver/master.hpp"
#include "rudra/engine/project.hpp"
#include "rudra/engine/session.hpp"

using namespace rudra;

namespace {

SdrImage ramp(int h = 8, int w = 32) {
    PlanarBuffer p(3, h, w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) p.at(c, y, x) = 0.05f + 0.85f * float(x) / float(w - 1) * (c == 1 ? 1.0f : 0.9f);
    return SdrImage(std::move(p));
}

}  // namespace

TEST(Exposure, ZeroIsBitForBit) {
    const SdrImage sdr = ramp();
    const Fields f = analytic_fields(sdr.height(), sdr.width());
    CompositeParams p;
    p.source = SourceCurve::Bt1886;
    const auto a = composite(sdr, f, FrameScalars{}, kAnalyticConstants, p);
    p.exposure_ev = 0.0;
    std::vector<float> gain;
    const auto b = composite(sdr, f, FrameScalars{}, kAnalyticConstants, p, &gain);
    ASSERT_EQ(a.buffer().span().size(), b.buffer().span().size());
    for (std::size_t i = 0; i < a.buffer().span().size(); ++i) ASSERT_EQ(a.buffer().span()[i], b.buffer().span()[i]);
    for (float g : gain) ASSERT_EQ(g, 1.0f);
}

TEST(Exposure, StopsScaleThePictureAndTheAnchorsTarget) {
    const SdrImage sdr = ramp();
    const Fields f = analytic_fields(sdr.height(), sdr.width());
    CompositeParams p;
    p.source = SourceCurve::Bt1886;
    const auto base = composite(sdr, f, FrameScalars{}, kAnalyticConstants, p);
    for (double ev : {1.0, -1.5, 0.33}) {
        p.exposure_ev = ev;
        std::vector<float> gain;
        const auto out = composite(sdr, f, FrameScalars{}, kAnalyticConstants, p, &gain);
        const double k = std::exp2(ev);
        for (std::size_t i = 0; i < out.buffer().span().size(); ++i)
            EXPECT_NEAR(out.buffer().span()[i], base.buffer().span()[i] * k, 1e-6 + 2e-6 * base.buffer().span()[i] * k) << ev;
        // The view's anchor scales its target by this gain (ViewParams::grade_gain), so it keeps the exposure.
        for (float g : gain) EXPECT_NEAR(g, k, 1e-6 * k);
    }
}

TEST(Exposure, TheBandsReadTheExposedNits) {
    // A band from 100 to 400 nits: at +2 EV a 50-nit pixel becomes 200 and is in it.
    const SdrImage sdr = ramp();
    const Fields f = analytic_fields(sdr.height(), sdr.width());
    CompositeParams p;
    p.source = SourceCurve::Bt1886;
    p.regions = {{100.0, 400.0, 1.0}};
    p.region_softness_stops = 0.25;
    p.exposure_ev = 2.0;
    std::vector<float> gain;
    composite(sdr, f, FrameScalars{}, kAnalyticConstants, p, &gain);
    CompositeParams q = p;
    q.exposure_ev = 0.0;
    const auto plain = composite(sdr, f, FrameScalars{}, kAnalyticConstants, q);
    int moved = 0;
    for (std::size_t i = 0; i < gain.size(); ++i) {
        const double nits_at_0 = double(plain.buffer().plane(1)[i]) * 10000.0;
        if (nits_at_0 > 30.0 && nits_at_0 < 80.0) {   // 120 to 320 nits once exposed: inside the band
            EXPECT_GT(gain[i], 4.0f * 1.5f) << nits_at_0;
            ++moved;
        }
    }
    EXPECT_GT(moved, 0);
}

TEST(Exposure, TheMasterAppliesItAfterTheAnchor) {
    // apply_region_ev with the exposure: x 2^ev, clipped at the ceiling, no bands needed.
    NitsFrame n(1, 3);
    for (int c = 0; c < 3; ++c) {
        n.plane(c)[0] = 10.0;
        n.plane(c)[1] = 203.0;
        n.plane(c)[2] = 30000.0;
    }
    apply_region_ev(n, std::vector<RegionBand>{}, 1.0, 40000.0, nullptr, 1.0);
    EXPECT_DOUBLE_EQ(n.plane(0)[0], 20.0);
    EXPECT_DOUBLE_EQ(n.plane(1)[1], 406.0);
    EXPECT_DOUBLE_EQ(n.plane(2)[2], 40000.0);   // the ceiling
    NitsFrame same(1, 1);
    for (int c = 0; c < 3; ++c) same.plane(c)[0] = 7.0;
    apply_region_ev(same, std::vector<RegionBand>{}, 1.0, 40000.0, nullptr, 0.0);
    EXPECT_DOUBLE_EQ(same.plane(0)[0], 7.0);

    // grade_master: with the anchor on, the master's mid-tones still move by the exposure.
    const SdrImage sdr = ramp();
    const Fields f = analytic_fields(sdr.height(), sdr.width());
    MasterRequest q;
    q.analytic = true;
    q.clip_fill = false;
    q.source_curve = "bt1886";
    auto a = grade_master(sdr, f, FrameScalars{}, kAnalyticConstants, q);
    q.exposure_ev = 1.0;
    auto b = grade_master(sdr, f, FrameScalars{}, kAnalyticConstants, q);
    ASSERT_TRUE(a && b);
    const std::size_t mid = a->nits.plane_size() / 2;
    EXPECT_NEAR(b->nits.plane(1)[mid], 2.0 * a->nits.plane(1)[mid], 1e-9 * a->nits.plane(1)[mid] + 1e-9);
}

TEST(Exposure, TheRequestCarriesItOnlyWhenSet) {
    MasterRequest q;
    EXPECT_EQ(master_request_json(q).find("exposure_ev"), std::string::npos);
    q.exposure_ev = -0.5;
    const auto back = master_request_from_json(master_request_json(q));
    ASSERT_TRUE(back);
    EXPECT_DOUBLE_EQ(back->exposure_ev, -0.5);
    EXPECT_FALSE(master_request_from_json(R"({"exposure_ev": 12})"));
}

TEST(Exposure, TheSessionHoldsItWithTheGrade) {
    Session s;
    EXPECT_EQ(s.grade.exposure_ev, 0.0);
    const std::string before = s.params_json();
    EXPECT_EQ(before.find("exposure_ev"), std::string::npos);
    // A drag: one undo entry, many inputs.
    s.exposure_press();
    s.exposure_input(0.5);
    s.exposure_input(0.75);
    EXPECT_EQ(s.undo_depth(), 1u);
    EXPECT_DOUBLE_EQ(s.composite_params().exposure_ev, 0.75);
    EXPECT_NE(s.params_json().find("\"exposure_ev\":0.75"), std::string::npos);
    s.exposure_input(9.0);   // the slider's range
    EXPECT_DOUBLE_EQ(s.grade.exposure_ev, kExposureMax);
    s.undo();
    EXPECT_EQ(s.grade.exposure_ev, 0.0);
    EXPECT_EQ(s.params_json(), before);
    // The menu: thirds of a stop, and a reset.
    EXPECT_TRUE(Session::owns("exposure-up"));
    EXPECT_TRUE(s.run("exposure-up"));
    EXPECT_TRUE(s.run("exposure-up"));
    EXPECT_NEAR(s.grade.exposure_ev, 2.0 / 3.0, 1e-12);
    EXPECT_TRUE(s.run("exposure-down"));
    EXPECT_NEAR(s.grade.exposure_ev, 1.0 / 3.0, 1e-12);
    EXPECT_TRUE(s.run("exposure-reset"));
    EXPECT_EQ(s.grade.exposure_ev, 0.0);
    s.exposure_input(1.0);
    s.reset_recon();
    EXPECT_EQ(s.grade.exposure_ev, 0.0);
}

TEST(Exposure, ProjectsKeepIt) {
    Session s;
    s.exposure_press();
    s.exposure_input(-1.25);
    const Project p = project_from_session(s);
    const auto file = std::filesystem::temp_directory_path() / "rudra-exposure.rudra";
    auto back = parse_project(project_json(p, file), file);
    ASSERT_TRUE(back) << back.error().message;
    EXPECT_DOUBLE_EQ(back->grade.exposure_ev, -1.25);
    // A project from before 13 Oct 2026 has none: 0.
    std::string old = project_json(p, file);
    const auto at = old.find("\"exposure_ev\"");
    ASSERT_NE(at, std::string::npos);
    old.replace(at, std::string("\"exposure_ev\"").size(), "\"exposure_unused\"");
    auto legacy = parse_project(old, file);
    ASSERT_TRUE(legacy);
    EXPECT_EQ(legacy->grade.exposure_ev, 0.0);
}
