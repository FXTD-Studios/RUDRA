// The anchor follows the Source picker (13 Oct 2026). Below its knee the
// master's anchor puts each pixel on the light of its SDR code. That light was
// the sRGB reading at 203 nits whatever the picker said, so in every anchored
// master BT.1886 (the default), the calibration and the reference were undone
// below the knee. Now the target is the picker's curve with its calibration or
// reference; unknown and aces with neither keep the sRGB reading, bit for bit.

#include <gtest/gtest.h>

#include <cmath>

#include "rudra/core/baseline.hpp"
#include "rudra/core/composite.hpp"
#include "rudra/core/fields.hpp"
#include "rudra/core/master.hpp"
#include "rudra/core/view.hpp"
#include "rudra/deliver/master.hpp"

using namespace rudra;

namespace {

SdrImage grey_ramp(int w = 256) {
    PlanarBuffer p(3, 2, w);
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) p.at(c, y, x) = float(x) / float(w - 1);
    return SdrImage(std::move(p));
}

MasterRequest analytic(const std::string& curve) {
    MasterRequest q;
    q.analytic = true;
    q.clip_fill = false;
    q.settle_grain = false;
    q.source_curve = curve;
    return q;
}

}  // namespace

TEST(AnchorTarget, UnknownAndAcesKeepTheSrgbReading) {
    CompositeParams p;
    EXPECT_TRUE(anchor_target_knots(p).empty());
    p.source = SourceCurve::Aces;
    EXPECT_TRUE(anchor_target_knots(p).empty());
    EXPECT_EQ(anchor_target_log2(0.5f, {}), 0.0f);
    p.source = SourceCurve::Bt1886;
    EXPECT_EQ(anchor_target_knots(p).size(), std::size_t(kSourceCurveKnots));
    CompositeParams cal;
    cal.calibration = {{118, 60.0}};
    EXPECT_EQ(anchor_target_knots(cal).size(), std::size_t(kSourceCurveKnots));
}

TEST(AnchorTarget, Bt1886IsItsOwnDisplayReading) {
    CompositeParams p;
    p.source = SourceCurve::Bt1886;
    const auto knots = anchor_target_knots(p);
    for (int k : {16, 64, 118, 180, 235}) {
        const float code = float(k) / 255.0f;
        const double want = double(source_curve_inverse(SourceCurve::Bt1886, srgb_to_linear(code)));
        const double got = double(srgb_to_linear(code)) * std::exp2(double(anchor_target_log2(code, knots)));
        EXPECT_NEAR(got, want, 2e-4 * want + 1e-7) << k;
    }
}

TEST(AnchorTarget, TheAnchoredMasterKeepsTheCurveBelowTheKnee) {
    // Analytic, no fill: the composite is the picker's inverse, so the anchor
    // has nothing to move below the knee. Before, it pulled BT.1886 back to sRGB.
    const SdrImage sdr = grey_ramp();
    const Fields f = analytic_fields(sdr.height(), sdr.width());
    for (const char* curve : {"bt1886", "hable", "clip"}) {
        MasterRequest on = analytic(curve), off = analytic(curve);
        off.anchor = false;
        auto a = grade_master(sdr, f, FrameScalars{}, kAnalyticConstants, on);
        auto b = grade_master(sdr, f, FrameScalars{}, kAnalyticConstants, off);
        ASSERT_TRUE(a && b);
        for (int x = 8; x < 200; ++x) {   // codes 0.03 to 0.78, below the knee band
            const double u = b->nits.plane(1)[std::size_t(x)], v = a->nits.plane(1)[std::size_t(x)];
            EXPECT_NEAR(v, u, 2e-3 * u + 1e-6) << curve << " code " << x;
        }
    }
    // The ACES inverse is still anchored to the sRGB reading, as every master before.
    MasterRequest aces = analytic("unknown");
    auto a = grade_master(sdr, f, FrameScalars{}, kAnalyticConstants, aces);
    ASSERT_TRUE(a);
    const double srgb_118 = double(srgb_to_linear(118.0f / 255.0f)) * 203.0;
    EXPECT_NEAR(a->nits.plane(1)[118], srgb_118, 1e-3 * srgb_118);
}

TEST(AnchorTarget, TheCalibrationSurvivesTheAnchor) {
    const SdrImage sdr = grey_ramp();
    const Fields f = analytic_fields(sdr.height(), sdr.width());
    MasterRequest q = analytic("bt1886");
    q.calibration = {{118, 60.0}};
    auto a = grade_master(sdr, f, FrameScalars{}, kAnalyticConstants, q);
    ASSERT_TRUE(a);
    EXPECT_NEAR(a->nits.plane(1)[118], 60.0, 0.6);   // the artist's 60 nits, not the curve's or sRGB's
}

TEST(AnchorTarget, TheViewHoldAndGainFollowIt) {
    // The live view (anchor_hold, render_view) takes the same target: on an
    // analytic BT.1886 composite the anchor holds at 1 and moves nothing.
    const SdrImage sdr = grey_ramp();
    const Fields f = analytic_fields(sdr.height(), sdr.width());
    CompositeParams p;
    p.source = SourceCurve::Bt1886;
    const auto m = composite(sdr, f, FrameScalars{}, kAnalyticConstants, p);
    const auto knots = anchor_target_knots(p);
    EXPECT_NEAR(anchor_hold(m, sdr, 0.9, 0.04, nullptr, knots), 1.0, 2e-3);
    EXPECT_GT(std::abs(anchor_hold(m, sdr, 0.9, 0.04) - 1.0), 0.05);   // the old sRGB target would move it
    ViewParams v = view_params(ViewMode::Image, 1000.0);
    v.anchor = true;
    v.anchor_hold = anchor_hold(m, sdr, 0.9, 0.04, nullptr, knots);
    v.anchor_knots = knots;
    ViewParams plain = v;
    plain.anchor = false;
    const PlanarBuffer got = render_view(m, corrected_baseline(sdr, 0.0f, {}), v, &sdr);
    const PlanarBuffer want = render_view(m, corrected_baseline(sdr, 0.0f, {}), plain, &sdr);
    for (int x = 8; x < 200; ++x) EXPECT_NEAR(got.at(1, 0, x), want.at(1, 0, x), 2e-3 * want.at(1, 0, x) + 1e-6) << x;
}
