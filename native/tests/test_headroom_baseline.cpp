#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

#include "rudra/core/source_curve.hpp"
#include "rudra/video/headroom_baseline.hpp"

using namespace rudra;
namespace {
MasterRequest request(const std::string& primaries = "rec709") {
    MasterRequest q;
    q.analytic = true;
    q.source_curve = "bt1886";
    q.source_space = primaries;
    q.clip_fill = false;
    return q;
}
ExtendedVideoFrame frame(int width = 1, const std::string& primaries = "rec709") {
    return {PlanarBuffer(3, 1, width), {"rec709", primaries, primaries == "rec709" ? "bt709" : "bt2020nc", "limited"}};
}
constexpr auto kPolicy = HeadroomPolicy::Bt1886V1;
// Frozen Python golden in golden/composite/index.json, independent of the
// production matrix function. Columns give the transformed source primaries.
constexpr double k709to2020[3][3] = {
    {.627403895934699, .32928303837788375, .04331306568741722},
    {.06909728935823198, .9195403950754586, .011362315566309171},
    {.01639143887515023, .08801330787722578, .895595253247624}
};
}  // namespace

TEST(HeadroomBaseline, RequiresExplicitVersionedOptIn) {
    const auto source = frame();
    const auto q = request();
    EXPECT_EQ(headroom_policy_id(HeadroomPolicy::Disabled), "disabled");
    EXPECT_EQ(headroom_policy_id(kPolicy), "bt1886-headroom-v1");
    EXPECT_FALSE(reconstruct_headroom_baseline(source, q));
    const auto invalid = static_cast<HeadroomPolicy>(255);
    EXPECT_TRUE(headroom_policy_id(invalid).empty());
    EXPECT_FALSE(reconstruct_headroom_baseline(source, q, invalid));
    auto result = reconstruct_headroom_baseline(source, q, kPolicy);
    ASSERT_TRUE(result);
    EXPECT_EQ(result->policy, kPolicy);
    EXPECT_EQ(result->source_contract, source.source_contract);
    EXPECT_EQ(result->linear2020.encoding.primaries, Primaries::Rec2020);
    EXPECT_EQ(result->linear2020.encoding.transfer, Transfer::Linear);
    EXPECT_EQ(result->linear2020.encoding.nits_per_unit, 203.0f);
    static_assert(!std::is_convertible_v<HeadroomBaseline, NetworkLinearImage>);
}

TEST(HeadroomBaseline, MatchesAbsoluteNitsReferencesWithoutWhiteClipping) {
    // Decimal arithmetic at 60-digit precision; binary-exact encoded inputs.
    constexpr float code[] = {0, .25f, .5f, 1, 1.25f, 2};
    constexpr double nits[] = {0, 7.2870551895124408, 38.4613078752013513, 203,
                               346.8007828288835085, 1071.4404235475901386};
    for (const auto* primaries : {"rec709", "rec2020"}) {
        auto source = frame(6, primaries);
        for (int c = 0; c < 3; ++c) std::copy(std::begin(code), std::end(code), source.encoded_rgb.plane(c));
        auto result = reconstruct_headroom_baseline(source, request(primaries), kPolicy);
        ASSERT_TRUE(result);
        for (int c = 0; c < 3; ++c)
            for (int i = 0; i < 6; ++i)
                EXPECT_NEAR(double(result->linear2020.buffer().at(c, 0, i)) * 203, nits[i], 1e-4);
        EXPECT_EQ(result->linear2020.buffer().at(0, 0, 0), 0);
        EXPECT_EQ(result->linear2020.buffer().at(0, 0, 3), 1);
    }
}

TEST(HeadroomBaseline, RemainsMonotonicAcrossNominalWhiteAndAllHeadroomCodes) {
    for (int bits : {8, 10}) {
        const int k = 1 << (bits - 8), width = 1 << bits;
        auto source = frame(width);
        for (int c = 0; c < 3; ++c)
            for (int x = 0; x < width; ++x)
                source.encoded_rgb.at(c, 0, x) = float(x - 16 * k) / float(219 * k);
        auto result = reconstruct_headroom_baseline(source, request(), kPolicy);
        ASSERT_TRUE(result);
        const auto& out = result->linear2020.buffer();
        for (int c = 0; c < 3; ++c) {
            for (int x = 0; x <= 16 * k; ++x) EXPECT_EQ(out.at(c, 0, x), 0);
            for (int x = 16 * k + 1; x < 255 * k; ++x) EXPECT_GT(out.at(c, 0, x), out.at(c, 0, x - 1));
            EXPECT_EQ(out.at(c, 0, 235 * k), 1);
        }
    }
    auto near_white = frame(5, "rec2020");
    const float codes[] = {.999f, std::nextafter(1.0f, 0.0f), 1.0f, std::nextafter(1.0f, 2.0f), 1.001f};
    for (int c = 0; c < 3; ++c) std::copy(std::begin(codes), std::end(codes), near_white.encoded_rgb.plane(c));
    auto result = reconstruct_headroom_baseline(near_white, request("rec2020"), kPolicy);
    ASSERT_TRUE(result);
    const float* out = result->linear2020.buffer().plane(0);
    EXPECT_GT(out[2], out[1]);
    EXPECT_GT(out[3], out[2]);
    EXPECT_LT(out[3] - out[1], 1e-6f);
    EXPECT_NEAR((out[4] - out[0]) / (codes[4] - codes[0]), 2.4, .001);
}

TEST(HeadroomBaseline, AppliesPowerInSourcePrimariesBeforeLinearGamutConversion) {
    auto source = frame(4);
    for (int c = 0; c < 3; ++c) source.encoded_rgb.at(c, 0, c) = 1.25f;
    source.encoded_rgb.at(0, 0, 3) = 1.25f;
    source.encoded_rgb.at(1, 0, 3) = .5f;
    source.encoded_rgb.at(2, 0, 3) = .25f;
    auto result = reconstruct_headroom_baseline(source, request(), kPolicy);
    ASSERT_TRUE(result);
    constexpr double high = 346.8007828288835085 / 203;
    for (int c = 0; c < 3; ++c) {
        for (int primary = 0; primary < 3; ++primary)
            EXPECT_NEAR(result->linear2020.buffer().at(c, 0, primary), k709to2020[c][primary] * high, 2e-7);
        const double expected = k709to2020[c][0] * high + k709to2020[c][1] * (38.4613078752013513 / 203)
                              + k709to2020[c][2] * (7.2870551895124408 / 203);
        EXPECT_NEAR(result->linear2020.buffer().at(c, 0, 3), expected, 2e-7);
    }
    // Converting encoded RGB then applying the power is a different result.
    const double wrong_r = std::pow(k709to2020[0][0] * 1.25 + k709to2020[0][1] * .5 + k709to2020[0][2] * .25, 2.4);
    EXPECT_GT(std::abs(result->linear2020.buffer().at(0, 0, 3) - wrong_r), .01);
    source.source_contract = {"rec709", "rec2020", "bt2020nc", "limited"};
    auto identity = reconstruct_headroom_baseline(source, request("rec2020"), kPolicy);
    ASSERT_TRUE(identity);
    for (int primary = 0; primary < 3; ++primary)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(identity->linear2020.buffer().at(c, 0, primary), c == primary ? high : 0, 2e-7);
}

TEST(HeadroomBaseline, RetainsOriginalSignedSamplesAndDoesNotApplyGrade) {
    auto source = frame(1, "rec2020");
    source.encoded_rgb.at(0, 0, 0) = -.1f;
    source.encoded_rgb.at(1, 0, 0) = 1.25f;
    source.encoded_rgb.at(2, 0, 0) = 2.0f;
    const auto before = source.encoded_rgb.vector();
    auto q = request("rec2020");
    q.exposure_ev = 1.5;  // This API returns the baseline, never a graded master.
    auto result = reconstruct_headroom_baseline(source, q, kPolicy);
    ASSERT_TRUE(result);
    EXPECT_EQ(source.encoded_rgb.vector(), before);
    EXPECT_EQ(result->linear2020.buffer().at(0, 0, 0), 0);
    EXPECT_NEAR(result->linear2020.buffer().at(2, 0, 0) * 203.0, 1071.44042354759, 1e-4);
    EXPECT_EQ(q.exposure_ev, 1.5);
}

TEST(HeadroomBaseline, RefusesUnvalidatedReconstructionCombinations) {
    const auto source = frame();
    auto q = request(); q.analytic = false;
    EXPECT_FALSE(reconstruct_headroom_baseline(source, q, kPolicy));
    for (auto curve : all_source_curves()) {
        if (curve == SourceCurve::Bt1886) continue;
        q = request(); q.source_curve = std::string(source_curve_id(curve));
        EXPECT_FALSE(reconstruct_headroom_baseline(source, q, kPolicy));
    }
    q = request(); q.clip_fill = true;
    EXPECT_FALSE(reconstruct_headroom_baseline(source, q, kPolicy));
    EXPECT_TRUE(q.clip_fill);
    q = request(); q.clip_peaks.push_back({.5, .5, 1000});
    EXPECT_FALSE(reconstruct_headroom_baseline(source, q, kPolicy));
    q = request(); q.calibration.push_back({118, 18});
    EXPECT_FALSE(reconstruct_headroom_baseline(source, q, kPolicy));
    q = request(); q.reference.target_log2_nits = {0, 1};
    EXPECT_FALSE(reconstruct_headroom_baseline(source, q, kPolicy));
}

TEST(HeadroomBaseline, RefusesUnsupportedMetadataAndInvalidBuffers) {
    const auto q = request();
    for (const auto& contract : {
            InputContract{"srgb", "rec709", "bt709", "limited"},
            InputContract{"rec709", "rec709", "bt709", "full"},
            InputContract{"rec709", "p3d65", "bt709", "limited"},
            InputContract{"rec709", "rec709", "gbr", "limited"},
            InputContract{"rec709", "rec709", "bt709", ""}}) {
        auto source = frame(); source.source_contract = contract;
        EXPECT_FALSE(reconstruct_headroom_baseline(source, q, kPolicy));
    }
    auto mismatched = request("rec2020");
    EXPECT_EQ(reconstruct_headroom_baseline(frame(), mismatched, kPolicy).error().code, ErrorCode::ContractMismatch);
    auto source = frame(); source.encoded_rgb = PlanarBuffer();
    EXPECT_FALSE(reconstruct_headroom_baseline(source, q, kPolicy));
    source.encoded_rgb = PlanarBuffer(1, 1, 1);
    EXPECT_FALSE(reconstruct_headroom_baseline(source, q, kPolicy));
    source = frame(); source.encoded_rgb.vector().pop_back();
    EXPECT_FALSE(reconstruct_headroom_baseline(source, q, kPolicy));
}

TEST(HeadroomBaseline, RejectsNonfiniteAndOverflowWithoutModifyingInput) {
    for (float invalid : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::max()}) {
        for (int c = 0; c < 3; ++c) {
            auto source = frame(); source.encoded_rgb.at(c, 0, 0) = invalid;
            EXPECT_FALSE(reconstruct_headroom_baseline(source, request(), kPolicy));
            if (!std::isnan(invalid)) EXPECT_EQ(source.encoded_rgb.at(c, 0, 0), invalid);
            else EXPECT_TRUE(std::isnan(source.encoded_rgb.at(c, 0, 0)));
        }
    }
}

TEST(HeadroomBaseline, LegacyWhiteRemainsUnchangedAndPolicyDifferenceIsExplicit) {
    const double old_white = source_curve_inverse(SourceCurve::Bt1886, 1.0f) * 203.0;
    EXPECT_NEAR(old_white, 201.79570576109, 1e-4);
    auto source = frame();
    for (auto& code : source.encoded_rgb.span()) code = 1;
    auto result = reconstruct_headroom_baseline(source, request(), kPolicy);
    ASSERT_TRUE(result);
    EXPECT_EQ(result->linear2020.buffer().at(0, 0, 0) * 203.0, 203.0);
    EXPECT_GT(203.0 - old_white, 1.0);
    EXPECT_EQ(source_curve_inverse(SourceCurve::Bt1886, 1.0f) * 203.0, old_white);
}
