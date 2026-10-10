// Roadmap 3.4: the reference match. A synthetic grade of a frame (the filmic
// curve +0.4 stops with a lifted shoulder, noise on top) comes back as the
// target at every code the frame covers; holes interpolate, the target is
// monotone, the residual is honest; the fit rides the calibration's knot
// path, replaces the anchors, survives its JSON, the session and a master
// request; and the loader reads what the masters write.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "rudra/core/baseline.hpp"
#include "rudra/core/calibration.hpp"
#include "rudra/core/composite.hpp"
#include "rudra/core/gamut.hpp"
#include "rudra/core/reference_fit.hpp"
#include "rudra/deliver/exr.hpp"
#include "rudra/deliver/master.hpp"
#include "rudra/engine/reference_image.hpp"
#include "rudra/engine/session.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {

constexpr float kEv = -1.0f;

double lift_at(int code) {
    const double t = std::clamp((code - 180.0) / 75.0, 0.0, 1.0);
    return 0.4 + 0.35 * t * t * (3.0 - 2.0 * t);
}

// A 96 x 96 frame covering codes 2..250 in every channel, offset per channel
// so the three planes land on different codes, and its "grade".
struct Synth {
    SdrImage sdr{96, 96};
    PlanarBuffer ref{3, 96, 96};
};
Synth synth(double noise_stops = 0.0) {
    Synth s;
    std::uint32_t rng = 7;
    auto next = [&] { rng = rng * 1664525u + 1013904223u; return double(rng >> 8) / double(1u << 24) - 0.5; };
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < 96; ++y)
            for (int x = 0; x < 96; ++x) {
                const int code = 2 + (x * 2 + y * 3 + c * 7) % 249;
                s.sdr.buffer().at(c, y, x) = float(code) / 255.0f;
                const double nits = source_code_nits(SourceCurve::Hable, code, kEv) * std::exp2(lift_at(code) + noise_stops * next());
                s.ref.at(c, y, x) = float(nits / 203.0);
            }
    return s;
}

}  // namespace

TEST(ReferenceFit, RecoversASyntheticGrade) {
    const Synth s = synth();
    auto fit = fit_reference(s.sdr, s.ref, "grade.exr");
    ASSERT_TRUE(fit) << fit.error().message;
    EXPECT_EQ(fit->file, "grade.exr");
    ASSERT_EQ(fit->target_log2_nits.size(), std::size_t(kSourceCurveKnots));
    EXPECT_GT(fit->codes_seen, 200);
    EXPECT_NEAR(reference_exposure(*fit, SourceCurve::Hable, kEv), 0.4, 0.03);
    for (int code : {40, 118, 200, 240}) {
        const double want = std::log2(source_code_nits(SourceCurve::Hable, code, kEv)) + lift_at(code);
        EXPECT_NEAR(fit->target_log2_nits[std::size_t(code)], want, 0.03) << code;
    }
    EXPECT_LT(fit->residual_mean, 0.02);
    EXPECT_LT(fit->residual_p95, 0.05);
    for (int code = 1; code < kSourceCurveKnots; ++code)
        EXPECT_GE(fit->target_log2_nits[std::size_t(code)], fit->target_log2_nits[std::size_t(code - 1)] - 1e-6f) << code;
    // Over the filmic curve the correction is the lift; over another curve it
    // is the same target, a different correction.
    const auto p = reference_params(*fit, SourceCurve::Hable, kEv);
    ASSERT_EQ(p.size(), 1u + kSourceCurveKnots);
    EXPECT_NEAR(p[1 + 118], lift_at(118), 0.03);
    const auto q = reference_params(*fit, SourceCurve::Aces, kEv);
    EXPECT_NEAR(q[1 + 118] - p[1 + 118],
                std::log2(source_code_nits(SourceCurve::Hable, 118, kEv) / source_code_nits(SourceCurve::Aces, 118, kEv)), 1e-4);
    EXPECT_NE(reference_summary(*fit, SourceCurve::Hable, kEv).find("codes"), std::string::npos);
}

TEST(ReferenceFit, NoiseLandsInTheResidualNotTheTarget) {
    const Synth s = synth(0.3);   // +-0.15 stops uniform
    auto fit = fit_reference(s.sdr, s.ref);
    ASSERT_TRUE(fit);
    EXPECT_NEAR(reference_exposure(*fit, SourceCurve::Hable, kEv), 0.4, 0.05);
    EXPECT_NEAR(fit->target_log2_nits[118], std::log2(source_code_nits(SourceCurve::Hable, 118, kEv)) + lift_at(118), 0.06);
    EXPECT_GT(fit->residual_mean, 0.04);
    EXPECT_LT(fit->residual_p95, 0.2);
}

TEST(ReferenceFit, RefusesTheWrongSizeAndAnEmptyReference) {
    const Synth s = synth();
    EXPECT_FALSE(fit_reference(s.sdr, PlanarBuffer(3, 48, 96)));
    EXPECT_FALSE(fit_reference(s.sdr, PlanarBuffer(4, 96, 96)));
    EXPECT_FALSE(fit_reference(s.sdr, PlanarBuffer(3, 96, 96, 0.0f)));   // nothing above zero
}

TEST(ReferenceFit, HolesInterpolateAndTheEndsAreFlat) {
    SdrImage sdr(8, 8);
    PlanarBuffer ref(3, 8, 8);
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < 64; ++i) {
            const int code = i < 32 ? 50 : 200;
            sdr.buffer().plane(c)[i] = float(code) / 255.0f;
            ref.plane(c)[i] = float(std::exp2(code == 50 ? 3.0 : 9.0) / 203.0);
        }
    auto fit = fit_reference(sdr, ref);
    ASSERT_TRUE(fit);
    EXPECT_EQ(fit->codes_seen, 2);
    EXPECT_NEAR(fit->target_log2_nits[10], 3.0, 1e-3);
    EXPECT_NEAR(fit->target_log2_nits[50], 3.0, 0.05);
    EXPECT_NEAR(fit->target_log2_nits[125], 6.0, 0.05);
    EXPECT_NEAR(fit->target_log2_nits[200], 9.0, 0.05);
    EXPECT_NEAR(fit->target_log2_nits[250], 9.0, 1e-3);
    EXPECT_EQ(fit->samples_per_code[125], 0);
}

TEST(ReferenceFit, ADescendingReferenceComesBackMonotone) {
    SdrImage sdr(16, 16);
    PlanarBuffer ref(3, 16, 16);
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < 256; ++i) {
            const int code = 20 + (i % 8) * 25;   // eight codes, 32 samples each
            sdr.buffer().plane(c)[i] = float(code) / 255.0f;
            ref.plane(c)[i] = float(std::exp2(10.0 - code / 50.0) / 203.0);   // brighter codes darker
        }
    auto fit = fit_reference(sdr, ref);
    ASSERT_TRUE(fit);
    for (int code = 1; code < kSourceCurveKnots; ++code)
        EXPECT_GE(fit->target_log2_nits[std::size_t(code)], fit->target_log2_nits[std::size_t(code - 1)] - 1e-6f);
    EXPECT_GT(fit->residual_p95, 0.5);   // and the residual says what the curve could not carry
    EXPECT_NE(reference_summary(*fit, SourceCurve::Hable, kEv).find("cannot carry"), std::string::npos);
}

TEST(ReferenceFit, SurvivesItsJson) {
    const Synth s = synth();
    auto fit = fit_reference(s.sdr, s.ref, "sh010.exr");
    ASSERT_TRUE(fit);
    auto back = reference_fit_from_json(reference_fit_json(*fit));
    ASSERT_TRUE(back) << back.error().message;
    EXPECT_EQ(*back, *fit);
    EXPECT_FALSE(reference_fit_from_json("{\"file\":\"x\"}"));
    EXPECT_FALSE(reference_fit_from_json("[]"));
}

TEST(ReferenceFit, ReplacesTheAnchorsOnTheBaselinePath) {
    const Synth s = synth();
    auto fit = fit_reference(s.sdr, s.ref);
    ASSERT_TRUE(fit);
    FrameScalars scalars;
    scalars.curve_params = {0.1f, 0.0f, 0.2f, 0.4f};
    CompositeParams p;
    p.source = SourceCurve::Hable;
    p.calibration = {{118, 1000.0}};   // would be a +5 stop shift
    p.reference = *fit;
    const auto merged = baseline_curve_params(scalars, p, kEv);
    ASSERT_EQ(merged.size(), 1u + kSourceCurveKnots);
    const float sdr = 118.0f / 255.0f;
    const double head_at = curve_correction_log2(sdr, scalars.curve_params);
    const double src_at = source_curve_params(SourceCurve::Hable)[1 + 118];
    const double ref_at = reference_params(*fit, SourceCurve::Hable, kEv)[1 + 118];
    EXPECT_NEAR(curve_correction_log2(sdr, merged), head_at + src_at + ref_at, 2e-3);
    EXPECT_NEAR(ref_at, lift_at(118), 0.03);
}

TEST(ReferenceFit, SessionOwnsTheReference) {
    const Synth s = synth();
    auto fit = fit_reference(s.sdr, s.ref, "grade.exr");
    ASSERT_TRUE(fit);
    Session sess;
    sess.set_calibration(1, 118, 18.0);
    const std::string anchored = sess.params_json();
    sess.set_reference(*fit);
    EXPECT_EQ(sess.undo_depth(), 2u);
    EXPECT_TRUE(sess.calibration_points().empty());                      // replaced
    EXPECT_NE(sess.params_json().find("\"reference\":{\"file\":\"grade.exr\""), std::string::npos);
    EXPECT_EQ(sess.params_json().find("\"calibration\""), std::string::npos);
    EXPECT_FALSE(sess.composite_params().reference.empty());
    sess.set_reference(*fit);                                             // unchanged: no undo entry
    EXPECT_EQ(sess.undo_depth(), 2u);
    sess.undo();
    EXPECT_EQ(sess.params_json(), anchored);                               // the anchors come back
    sess.redo();
    sess.clear_reference();
    EXPECT_TRUE(sess.grade.reference.empty());
    EXPECT_EQ(sess.params_json().find("\"reference\""), std::string::npos);
    sess.clear_reference();                                                // already clear
    EXPECT_EQ(sess.undo_depth(), 3u);
}

TEST(ReferenceFit, AMasterRequestCarriesIt) {
    const Synth s = synth();
    auto fit = fit_reference(s.sdr, s.ref, "grade.exr");
    ASSERT_TRUE(fit);
    MasterRequest q;
    q.source_curve = "hable";
    q.reference = *fit;
    auto r = master_request_from_json(master_request_json(q));
    ASSERT_TRUE(r) << r.error().message;
    EXPECT_EQ(r->reference, *fit);
    MasterRequest plain;
    EXPECT_EQ(master_request_json(plain).find("reference"), std::string::npos);
}

TEST(ReferenceFit, ReadsWhatTheMastersWrite) {
    const fs::path dir = fs::temp_directory_path() / "rudra_reference_fit_test";
    fs::create_directories(dir);
    PlanarBuffer rgb2020(3, 4, 6);
    for (int c = 0; c < 3; ++c)
        for (std::size_t i = 0; i < rgb2020.plane_size(); ++i) rgb2020.plane(c)[i] = 0.1f * float(c + 1) + 0.01f * float(i);
    // The linear container: Rec.2020 chromaticities, half floats.
    const fs::path linear = dir / "linear.exr";
    ASSERT_TRUE(write_exr(linear, rgb2020, false, kRec2020Chromaticities));
    auto got = read_reference(linear);
    ASSERT_TRUE(got) << got.error().message;
    const PlanarBuffer want = convert_primaries(rgb2020, Primaries::Rec2020, Primaries::Rec709);
    ASSERT_EQ(got->linear709.width(), 6);
    for (std::size_t i = 0; i < want.span().size(); ++i) EXPECT_NEAR(got->linear709.span()[i], want.span()[i], 1e-5f);
    EXPECT_NE(got->note.find("Rec.2020"), std::string::npos);
    // The ACES container: AP0, read back to the same Rec.709 values.
    const fs::path aces = dir / "aces.exr";
    ASSERT_TRUE(write_aces_exr(aces, rgb2020, Primaries::Rec2020));
    auto got2 = read_reference(aces);
    ASSERT_TRUE(got2) << got2.error().message;
    for (std::size_t i = 0; i < want.span().size(); ++i) EXPECT_NEAR(got2->linear709.span()[i], want.span()[i], 2e-3f);
    EXPECT_NE(got2->note.find("AP0"), std::string::npos);
    EXPECT_FALSE(read_reference(dir / "missing.exr"));
    fs::remove_all(dir);
}
