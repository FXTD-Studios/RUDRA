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
    // Not the pipeline's: BT.1886 is the display decode run backwards, the
    // inverse BT.709 OETF of x^(1/2.4) (13 Oct 2026).
    {SourceCurve::Bt1886, 0.18f, 0.24952626f}, {SourceCurve::Bt1886, 1.0f, 1.0f},       {SourceCurve::Bt1886, 4.0f, 1.0f},
};

}  // namespace

TEST(SourceCurve, IdsRoundTrip) {
    for (SourceCurve c : all_source_curves()) EXPECT_EQ(parse_source_curve(source_curve_id(c)), c);
    EXPECT_FALSE(parse_source_curve("narkowicz"));
    EXPECT_EQ(all_source_curves().size(), 8u);
    EXPECT_EQ(source_curve_id(SourceCurve::Bt1886), "bt1886");
    EXPECT_EQ(source_curve_id(SourceCurve::Bt1886Shadows), "bt1886_shadows");
    // Appended last: every stored enum value keeps its meaning.
    EXPECT_EQ(int(SourceCurve::Bt1886), 6);
    EXPECT_EQ(int(SourceCurve::Bt1886Shadows), 7);
}

// 14 Oct 2026: BT.1886 shadows is BT.1886 from code 40 up, brighter below it,
// rising at every code, and the measured table at the codes it was fitted at.
TEST(SourceCurve, Bt1886ShadowsIsBt1886AboveTheKneeAndLiftedBelow) {
    const auto at = [](SourceCurve c, int code) { return source_curve_inverse(c, srgb_to_linear(float(code) / 255.0f)); };
    for (int code = 40; code < 256; ++code)
        EXPECT_EQ(at(SourceCurve::Bt1886Shadows, code), at(SourceCurve::Bt1886, code)) << code;
    for (int code = 1; code < 40; ++code) {
        EXPECT_GT(at(SourceCurve::Bt1886Shadows, code), at(SourceCurve::Bt1886, code)) << code;
        EXPECT_GT(at(SourceCurve::Bt1886Shadows, code + 1), at(SourceCurve::Bt1886Shadows, code)) << code;
    }
    EXPECT_EQ(at(SourceCurve::Bt1886Shadows, 0), 0.0f);
    // Pinned from tools/shadow_correction_eval.py (fit_v3.json): code 1, 20, 39 in log2 scene.
    EXPECT_NEAR(std::log2(at(SourceCurve::Bt1886Shadows, 1)), -14.685428f, 2e-3f);
    EXPECT_NEAR(std::log2(at(SourceCurve::Bt1886Shadows, 20)), -10.013963f, 2e-3f);
    EXPECT_NEAR(std::log2(at(SourceCurve::Bt1886Shadows, 39)), -8.253048f, 2e-3f);
    // About 0.2 nits at code 20 against BT.1886's 0.05.
    EXPECT_NEAR(at(SourceCurve::Bt1886Shadows, 20) * 203.0f, 0.1964f, 2e-3f);
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
    // A session starts on BT.1886 (13 Oct 2026), and says so in params().
    EXPECT_EQ(s.grade.source, "bt1886");
    EXPECT_NE(before.find("\"source_curve\":\"bt1886\""), std::string::npos);
    EXPECT_EQ(s.composite_params().source, SourceCurve::Bt1886);
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
    EXPECT_EQ(s.grade.source, "bt1886");
    EXPECT_EQ(s.params_json(), before);
    s.redo();
    EXPECT_EQ(s.grade.source, "hable");
    // "unknown" is still the ACES inverse, left out of params() as before 3.1.
    s.set_source("unknown");
    EXPECT_EQ(s.composite_params().source, SourceCurve::Unknown);
    EXPECT_EQ(s.params_json().find("source_curve"), std::string::npos);
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

// 13 Oct 2026: BT.1886 is the display's decode of a BT.709 video -- a code's
// scene light is code^2.4, 100 % at diffuse white -- of the light RUDRA reads
// from that code (the inverse BT.709 OETF, canonicalize_sdr "rec709"), and its
// knots carry it through the composite's per-code correction over the ACES inverse.
TEST(SourceCurve, Bt1886IsTheDisplayDecode) {
    const auto inverse_oetf = [](float v) { return v < 0.081f ? v / 4.5f : std::pow((v + 0.099f) / 1.099f, 1.0f / 0.45f); };
    for (int code = 1; code < 254; ++code) {
        const float c = float(code) / 255.0f;
        EXPECT_NEAR(source_curve_inverse(SourceCurve::Bt1886, inverse_oetf(c)), std::pow(c, 2.4f),
                    std::max(std::pow(c, 2.4f) * 2e-5f, 1e-7f)) << code;
    }
    const auto p = source_curve_params(SourceCurve::Bt1886);
    ASSERT_EQ(p.size(), std::size_t(1 + kSourceCurveKnots));
    const float d = srgb_to_linear(118.0f / 255.0f);   // the knots are at sRGB codes of that light
    EXPECT_NEAR(inverse_aces_approx(d) * std::exp2(p[1 + 118]), source_curve_inverse(SourceCurve::Bt1886, d), 2e-4f);
}

// 24 Nov 2026: a still, and a movie as the app reads it (each frame through ffmpeg to PNG), hold the source's own
// codes, not the video path's canonical ones. BT.1886 then reads each code as the video code it is: code^2.4. Before,
// the own code went through the sRGB decode and then the BT.709 OETF: 2.7 stops dark at code 26, 1.3 at 51, 0.36 at
// 128 (09V's frames: 1.95 stops at the median). The other curves read own codes as before.
TEST(SourceCurve, Bt1886ReadsOwnCodesAsTheVideoCodesTheyAre) {
    const auto own = source_curve_params(SourceCurve::Bt1886, true);
    ASSERT_EQ(own.size(), std::size_t(1 + kSourceCurveKnots));
    for (int code = 2; code < 255; ++code) {   // code 1 is code 2's ratio by convention; 255 meets the 0.995 clamp
        const double got = double(inverse_aces_approx(srgb_to_linear(float(code) / 255.0f))) * std::exp2(double(own[1 + std::size_t(code)]));
        EXPECT_NEAR(std::log2(got / std::pow(code / 255.0, 2.4)), 0.0, 0.01) << code;
    }
    const auto canonical = source_curve_params(SourceCurve::Bt1886, false);   // the video path: unchanged
    const double before = double(inverse_aces_approx(srgb_to_linear(26.0f / 255.0f))) * std::exp2(double(canonical[1 + 26]));
    EXPECT_LT(std::log2(before / std::pow(26.0 / 255.0, 2.4)), -2.5);   // what an own code got before
    // BT.1886 shadows: BT.1886 from where the video light's code reaches 40, its measured table below.
    const auto shadows = source_curve_params(SourceCurve::Bt1886Shadows, true);
    const double at51 = double(inverse_aces_approx(srgb_to_linear(51.0f / 255.0f))) * std::exp2(double(shadows[1 + 51]));
    EXPECT_NEAR(std::log2(at51 / std::pow(51.0 / 255.0, 2.4)), 0.0, 0.01);
    for (SourceCurve c : {SourceCurve::Unknown, SourceCurve::Aces, SourceCurve::Hable, SourceCurve::Agx, SourceCurve::CameraLog,
                          SourceCurve::Clip}) {
        const auto a = source_curve_params(c, true), b = source_curve_params(c, false);
        ASSERT_EQ(a.size(), b.size());
        for (std::size_t i = 0; i < a.size(); ++i) EXPECT_EQ(a[i], b[i]) << source_curve_id(c) << " " << i;
    }
    EXPECT_TRUE(reads_video_codes(SourceCurve::Bt1886));
    EXPECT_TRUE(reads_video_codes(SourceCurve::Bt1886Shadows));
    EXPECT_FALSE(reads_video_codes(SourceCurve::Clip));
    // Through the composite: the anchor's target and the session's params carry it.
    CompositeParams cp;
    cp.source = SourceCurve::Bt1886;
    cp.own_codes = true;
    const auto t = anchor_target_knots(cp);
    ASSERT_EQ(t.size(), std::size_t(kSourceCurveKnots));
    for (int code : {26, 51, 128})
        EXPECT_NEAR(t[std::size_t(code)], std::log2(std::pow(code / 255.0, 2.4) / double(srgb_to_linear(float(code) / 255.0f))), 0.01) << code;
    Session s;
    EXPECT_FALSE(s.composite_params().own_codes);
    s.set_own_codes(true);
    EXPECT_TRUE(s.own_codes());
    EXPECT_TRUE(s.composite_params().own_codes);
}
