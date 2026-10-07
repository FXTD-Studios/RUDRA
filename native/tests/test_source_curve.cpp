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
    EXPECT_EQ(baseline_curve_params(scalars, a), scalars.curve_params);
    EXPECT_EQ(baseline_curve_params(scalars, b), scalars.curve_params);
    CompositeParams h;
    h.source = SourceCurve::Hable;
    EXPECT_EQ(baseline_curve_params(scalars, h).size(), 1u + kSourceCurveKnots);
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
