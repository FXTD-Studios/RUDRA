// Roadmap 3.1: the source curve. The forward curves are pipeline/sdr_render.py's
// (values pinned here were printed by that module), the inverses round-trip,
// and the per-code correction reproduces each inverse through the CurveHead
// path so the composite, the shader and the masters carry it unchanged.

#include <gtest/gtest.h>

#include <cmath>

#include "rudra/core/baseline.hpp"
#include "rudra/core/composite.hpp"
#include "rudra/core/source_curve.hpp"
#include "rudra/engine/session.hpp"

using namespace rudra;

namespace {

// pipeline/sdr_render.py, float32: curve_<name>(x) at x = 0.18, 1.0, 4.0.
struct Pin {
    SourceCurve curve;
    float x, y;
};
constexpr Pin kPins[] = {
    {SourceCurve::Aces, 0.18f, 0.26689893f},   {SourceCurve::Aces, 1.0f, 0.80379742f},  {SourceCurve::Aces, 4.0f, 0.97341704f},
    {SourceCurve::Hable, 0.18f, 0.12833844f},  {SourceCurve::Hable, 1.0f, 0.49291864f}, {SourceCurve::Hable, 4.0f, 0.91803002f},
    {SourceCurve::Agx, 0.18f, 0.24004444f},    {SourceCurve::Agx, 1.0f, 0.68272805f},   {SourceCurve::Agx, 4.0f, 0.91022730f},
    {SourceCurve::CameraLog, 0.18f, 0.18946457f}, {SourceCurve::CameraLog, 1.0f, 0.57168305f}, {SourceCurve::CameraLog, 4.0f, 0.83722943f},
    {SourceCurve::Clip, 0.18f, 0.18f},         {SourceCurve::Clip, 1.0f, 1.0f},         {SourceCurve::Clip, 4.0f, 1.0f},
};

}  // namespace

TEST(SourceCurve, IdsRoundTrip) {
    for (SourceCurve c : all_source_curves()) EXPECT_EQ(parse_source_curve(source_curve_id(c)), c);
    EXPECT_FALSE(parse_source_curve("narkowicz"));
    EXPECT_EQ(all_source_curves().size(), 6u);
}

TEST(SourceCurve, ForwardMatchesThePipeline) {
    for (const Pin& p : kPins) EXPECT_NEAR(source_curve_forward(p.curve, p.x), p.y, 2e-5f) << source_curve_id(p.curve) << " " << p.x;
}

TEST(SourceCurve, InversesRoundTrip) {
    for (SourceCurve c : all_source_curves()) {
        for (int code = 1; code < 255; ++code) {
            const float display = srgb_to_linear(float(code) / 255.0f);
            const float scene = source_curve_inverse(c, display);
            // Within one 8-bit code of the display value that made it.
            EXPECT_NEAR(source_curve_forward(c, scene), std::min(display, 0.995f), 1.0f / 255.0f) << source_curve_id(c) << " code " << code;
        }
        EXPECT_GT(source_curve_inverse(c, 0.995f), source_curve_inverse(c, 0.5f));
    }
    // Unknown and ACES are inverse_aces_approx itself.
    for (float y : {0.0f, 0.1f, 0.5f, 0.9f, 1.0f}) {
        EXPECT_EQ(source_curve_inverse(SourceCurve::Unknown, y), inverse_aces_approx(y));
        EXPECT_EQ(source_curve_inverse(SourceCurve::Aces, y), inverse_aces_approx(y));
    }
}

TEST(SourceCurve, CorrectionReproducesTheInverseAtEveryCode) {
    for (SourceCurve c : all_source_curves()) {
        const auto params = source_curve_params(c);
        if (c == SourceCurve::Unknown || c == SourceCurve::Aces) {
            EXPECT_TRUE(params.empty());
            continue;
        }
        ASSERT_EQ(params.size(), 1u + kSourceCurveKnots);
        EXPECT_EQ(params[0], 0.0f);
        for (int code = 2; code < 256; ++code) {
            const float sdr = float(code) / 255.0f;
            const float want = source_curve_inverse(c, srgb_to_linear(sdr));
            const float got = inverse_aces_approx(srgb_to_linear(sdr)) * std::exp2(curve_correction_log2(sdr, params));
            EXPECT_NEAR(got, want, std::max(want * 2e-4f, 1e-6f)) << source_curve_id(c) << " code " << code;
        }
    }
}

TEST(SourceCurve, EffectiveParamsMergeWithTheCurveHead) {
    const std::vector<float> none{0.0f};
    EXPECT_EQ(effective_curve_params(none, SourceCurve::Unknown), none);
    EXPECT_EQ(effective_curve_params(none, SourceCurve::Aces), none);
    const std::vector<float> head{0.25f, 0.0f, 0.5f, 1.0f};   // exposure + 3 knots, a ramp
    EXPECT_EQ(effective_curve_params(head, SourceCurve::Unknown), head);
    const auto merged = effective_curve_params(head, SourceCurve::Hable);
    const auto hable = source_curve_params(SourceCurve::Hable);
    ASSERT_EQ(merged.size(), hable.size());
    EXPECT_FLOAT_EQ(merged[0], 0.25f);
    // The head's ramp reads 0 -> 1 over the codes; it adds to the source's knots.
    for (int code = 0; code < 256; ++code) {
        const float sdr = float(code) / 255.0f;
        EXPECT_NEAR(merged[1 + code], hable[1 + code] + sdr, 1e-5f) << code;
    }
    // Both directions of the merge agree with corrected_baseline's reading of the sum.
    const float sdr = 0.6f;
    const float sum = curve_correction_log2(sdr, merged);
    EXPECT_NEAR(sum, curve_correction_log2(sdr, hable) + curve_correction_log2(sdr, head), 2e-3f);
}

TEST(SourceCurve, CompositeUnknownIsBitIdentical) {
    FrameScalars scalars;
    CompositeParams a, b;
    b.source = SourceCurve::Unknown;
    EXPECT_EQ(baseline_curve_params(scalars, a, -1.0f), scalars.curve_params);
    EXPECT_EQ(baseline_curve_params(scalars, b, -1.0f), scalars.curve_params);
    CompositeParams h;
    h.source = SourceCurve::Hable;
    EXPECT_EQ(baseline_curve_params(scalars, h, -1.0f).size(), 1u + kSourceCurveKnots);
}

TEST(SourceCurve, SessionOwnsTheChoice) {
    Session s;
    const std::string before = s.params_json();
    EXPECT_EQ(before.find("source_curve"), std::string::npos);
    EXPECT_EQ(s.composite_params().source, SourceCurve::Unknown);
    EXPECT_TRUE(Session::owns("source-hable"));
    EXPECT_FALSE(Session::owns("source-narkowicz"));
    EXPECT_TRUE(s.run("source-hable"));
    EXPECT_EQ(s.grade.source, "hable");
    EXPECT_EQ(s.composite_params().source, SourceCurve::Hable);
    EXPECT_NE(s.params_json().find("\"source_curve\":\"hable\""), std::string::npos);
    EXPECT_EQ(s.undo_depth(), 1u);
    s.set_source("hable");   // unchanged: no undo entry
    EXPECT_EQ(s.undo_depth(), 1u);
    s.set_source("not-a-curve");
    EXPECT_EQ(s.grade.source, "hable");
    s.undo();
    EXPECT_EQ(s.grade.source, "unknown");
    EXPECT_EQ(s.params_json(), before);
    s.redo();
    EXPECT_EQ(s.grade.source, "hable");
}

// ---- Roadmap 3.2: three-click calibration --------------------------------

TEST(Calibration, ExactAtTheAnchorsFlatOutsideLinearBetween) {
    const float ev = -1.0f;
    const std::vector<CalibrationPoint> pts = {{8, 0.3}, {118, 18.0}, {235, 400.0}};
    for (SourceCurve s : {SourceCurve::Unknown, SourceCurve::Hable}) {
        const auto p = calibration_params(pts, s, ev);
        ASSERT_EQ(p.size(), 1u + kSourceCurveKnots);
        EXPECT_EQ(p[0], 0.0f);
        for (const auto& a : pts)
            EXPECT_NEAR(source_code_nits(s, a.code, ev) * std::exp2(double(p[1 + a.code])), a.nits, a.nits * 1e-5)
                << source_curve_id(s) << " code " << a.code;
        // Flat outside the outer anchors, linear in log2 between.
        EXPECT_FLOAT_EQ(p[1 + 0], p[1 + 8]);
        EXPECT_FLOAT_EQ(p[1 + 255], p[1 + 235]);
        const float mid = (p[1 + 8] + p[1 + 118]) * 0.5f;
        EXPECT_NEAR(p[1 + 63], mid, 1e-5f);
        EXPECT_TRUE(calibration_is_monotone(pts, s, ev));
    }
}

TEST(Calibration, OneAnchorIsAnExposureShift) {
    const auto p = calibration_params(std::vector<CalibrationPoint>{{118, 36.0}}, SourceCurve::Aces, -1.0f);
    ASSERT_EQ(p.size(), 1u + kSourceCurveKnots);
    const float want = float(std::log2(36.0 / source_code_nits(SourceCurve::Aces, 118, -1.0f)));
    for (int code = 0; code < 256; ++code) EXPECT_NEAR(p[1 + code], want, 1e-6f) << code;
    EXPECT_NE(calibration_summary(std::vector<CalibrationPoint>{{118, 36.0}}, SourceCurve::Aces, -1.0f).find("1 anchor"),
              std::string::npos);
}

TEST(Calibration, UnusablePointsAreSkippedAndNonMonotoneRefused) {
    EXPECT_TRUE(calibration_params(std::vector<CalibrationPoint>{{118, 0.0}, {-1, 5.0}, {300, 5.0}}, SourceCurve::Aces, -1.0f).empty());
    EXPECT_EQ(calibration_summary(std::vector<CalibrationPoint>{{118, 0.0}}, SourceCurve::Aces, -1.0f), "");
    // A highlight darker than the grey: the curve would go down.
    const std::vector<CalibrationPoint> bad = {{118, 100.0}, {235, 10.0}};
    EXPECT_FALSE(calibration_is_monotone(bad, SourceCurve::Aces, -1.0f));
    EXPECT_EQ(calibration_summary(bad, SourceCurve::Aces, -1.0f).rfind("These anchors", 0), 0u);
    // The composite ignores a refused set and keeps the picker's curve.
    FrameScalars scalars;
    CompositeParams p;
    p.source = SourceCurve::Hable;
    p.calibration = bad;
    const auto hable = source_curve_params(SourceCurve::Hable);
    const auto got = baseline_curve_params(scalars, p, -1.0f);
    ASSERT_EQ(got.size(), hable.size());
    for (std::size_t i = 0; i < got.size(); ++i) EXPECT_EQ(got[i], hable[i]);
}

TEST(Calibration, SumsWithTheSourceAndTheHead) {
    const float ev = -1.0f;
    const std::vector<CalibrationPoint> pts = {{118, 60.0}};
    FrameScalars scalars;
    CompositeParams p;
    p.source = SourceCurve::Hable;
    p.calibration = pts;
    const auto got = baseline_curve_params(scalars, p, ev);
    const auto hable = source_curve_params(SourceCurve::Hable);
    const auto cal = calibration_params(pts, SourceCurve::Hable, ev);
    ASSERT_EQ(got.size(), hable.size());
    for (std::size_t i = 1; i < got.size(); ++i) EXPECT_NEAR(got[i], hable[i] + cal[i], 1e-6f) << i;
    // Through corrected_baseline the anchor lands on its nits.
    const float sdr = 118.0f / 255.0f;
    const double nits = inverse_aces_approx(srgb_to_linear(sdr)) * std::exp2(-double(ev)) * 203.0 *
                        std::exp2(double(curve_correction_log2(sdr, got)));
    EXPECT_NEAR(nits, 60.0, 60.0 * 1e-4);
    // No source, a CurveHead of 3 knots, one anchor: the head is resampled and the anchor still lands.
    scalars.curve_params = {0.1f, 0.0f, 0.2f, 0.4f};
    CompositeParams q;
    q.calibration = pts;
    const auto merged = baseline_curve_params(scalars, q, ev);
    ASSERT_EQ(merged.size(), 1u + kSourceCurveKnots);
    const double head_at = curve_correction_log2(sdr, scalars.curve_params);
    const double cal_at = calibration_params(pts, SourceCurve::Unknown, ev)[1 + 118];
    EXPECT_NEAR(curve_correction_log2(sdr, merged), head_at + cal_at, 2e-3);
}

TEST(Calibration, SessionOwnsTheAnchors) {
    Session s;
    const std::string before = s.params_json();
    s.set_calibration(1, 118, 18.0);
    EXPECT_EQ(s.undo_depth(), 1u);
    EXPECT_EQ(s.calibration_points().size(), 1u);
    EXPECT_NE(s.params_json().find("\"calibration\":[{\"code\":118,\"nits\":18}]"), std::string::npos);
    EXPECT_EQ(s.composite_params().calibration.size(), 1u);
    s.set_calibration(1, -1, 18.0);   // nothing changed: no undo entry
    EXPECT_EQ(s.undo_depth(), 1u);
    s.set_calibration(1, -1, 20.0);   // keeps the code
    EXPECT_EQ(s.calibration_points()[0].code, 118);
    EXPECT_EQ(s.calibration_points()[0].nits, 20.0);
    s.set_calibration(0, 8, 0.0);     // a code without nits: a slot, not a point
    EXPECT_EQ(s.calibration_points().size(), 1u);
    s.set_calibration(2, -1, 400.0);  // nits before any click: kept, but no point until a code lands
    EXPECT_EQ(s.calibration_points().size(), 1u);
    s.set_calibration(2, 235, 400.0);
    EXPECT_EQ(s.calibration_points().size(), 2u);
    s.clear_calibration();
    EXPECT_TRUE(s.calibration_points().empty());
    EXPECT_EQ(s.params_json(), before);
    s.undo();                         // back to the two points the clear took
    EXPECT_EQ(s.calibration_points().size(), 2u);
    s.clear_calibration();
    s.clear_calibration();            // already clear: no undo entry
    EXPECT_EQ(s.undo_depth(), 6u);
}
