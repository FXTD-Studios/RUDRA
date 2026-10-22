#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

#include "rudra/core/hdr10.hpp"
#include "rudra/video/headroom_render.hpp"

using namespace rudra;
namespace {
constexpr auto kPolicy = HeadroomPolicy::Bt1886V1;
MasterRequest request() {
    MasterRequest q;
    q.analytic = true;
    q.source_curve = "bt1886";
    q.source_space = "rec2020";
    q.clip_fill = q.anchor = q.carry_chroma = q.settle_grain = false;
    return q;
}
ExtendedVideoFrame frame(int width = 1, float code = 1.25f) {
    return {PlanarBuffer(3, 1, width, code), {"rec709", "rec2020", "bt2020nc", "limited"}};
}
double nits(const HeadroomMaster& m, int x = 0, int c = 0) {
    return double(m.linear2020.buffer().at(c, 0, x)) * 203.0;
}
double pq_code(double nits_value) {
    // Independent double ST 2084 expression, not the production pq_oetf.
    const double p = std::pow(nits_value / 10000.0, 2610.0 / 16384.0);
    return std::pow((3424.0 / 4096.0 + 2413.0 / 128.0 * p) /
                    (1.0 + 2392.0 / 128.0 * p), 2523.0 / 32.0);
}
}  // namespace

TEST(HeadroomRender, NeutralIsTheBaselineAndKeepsItsContract) {
    auto source = frame(5);
    const float codes[] = {-.1f, 0, .5f, 1, 1.25f};
    for (int c = 0; c < 3; ++c)
        std::copy(std::begin(codes), std::end(codes), source.encoded_rgb.plane(c));
    const auto before = source.encoded_rgb.vector();
    const auto q = request();
    EXPECT_FALSE(render_headroom_frame(source, q));
    EXPECT_FALSE(render_headroom_frame(source, q, static_cast<HeadroomPolicy>(255)));
    auto b = reconstruct_headroom_baseline(source, q, kPolicy);
    auto m = render_headroom_frame(source, q, kPolicy);
    ASSERT_TRUE(b); ASSERT_TRUE(m);
    EXPECT_EQ(m->linear2020.buffer().vector(), b->linear2020.buffer().vector());
    EXPECT_EQ(m->source_contract, source.source_contract);
    EXPECT_EQ(m->policy, kPolicy);
    EXPECT_EQ(nits(*m, 3), 203);
    EXPECT_NEAR(nits(*m, 4), 346.8007828288835, 1e-4);
    EXPECT_EQ(source.encoded_rgb.vector(), before);
    static_assert(!std::is_convertible_v<HeadroomMaster, NetworkLinearImage>);
}

TEST(HeadroomRender, ExposureUses203NitUnitsAndIsNeverUndone) {
    auto source = frame();
    const double base = 346.8007828288835;
    for (double ev : {-8.0, -2.0, 0.0, 1.0, 8.0}) {
        auto q = request(); q.exposure_ev = ev;
        auto m = render_headroom_frame(source, q, kPolicy);
        ASSERT_TRUE(m) << m.error().message;
        EXPECT_NEAR(nits(*m), base * std::exp2(ev), base * std::exp2(ev) * 1e-7);
    }
    // No legacy 40000-nit model ceiling, including a near-zero grade.
    source = frame(1, 16);
    for (double ev : {0.0, 1e-6}) {
        auto q = request(); q.exposure_ev = ev;
        auto m = render_headroom_frame(source, q, kPolicy);
        ASSERT_TRUE(m);
        EXPECT_NEAR(nits(*m), 203.0 * std::pow(16.0, 2.4) * std::exp2(ev), .03);
        EXPECT_GT(nits(*m), 40000);
    }
}

TEST(HeadroomRender, RegionsReadExposedRec2020LightAndRespectSoftness) {
    auto source = frame(1, .5f);  // 38.4613 nits; exposure moves it into the band.
    auto q = request();
    q.exposure_ev = 1;
    q.regions = {{"mids", 60, 100, 1}};
    q.region_softness_stops = .1;
    auto m = render_headroom_frame(source, q, kPolicy);
    ASSERT_TRUE(m);
    EXPECT_NEAR(nits(*m), 38.46130787520135 * 4, 2e-5);
    // White at 203 nits is halfway up a two-stop ramp ending at 406 nits.
    source = frame(1, 1);
    q.exposure_ev = 0;
    q.regions = {{"soft", 406, 1000, 1}};
    q.region_softness_stops = 2;
    m = render_headroom_frame(source, q, kPolicy);
    ASSERT_TRUE(m);
    EXPECT_NEAR(nits(*m), 203 * std::sqrt(2.0), 5e-5);
    // Qualifiers see converted Rec.2020 RGB; Rec.709 red has ~43.17 nits
    // luminance, not 53.33 nits from treating the original RGB as Rec.2020.
    source = frame(1, 0);
    source.encoded_rgb.at(0, 0, 0) = 1;
    source.source_contract = {"rec709", "rec709", "bt709", "limited"};
    q = request(); q.source_space = "rec709";
    q.regions = {{"red", 40, 47, 1}}; q.region_softness_stops = .01;
    m = render_headroom_frame(source, q, kPolicy);
    ASSERT_TRUE(m);
    const double red2020[] = {.627403895934699, .06909728935823198, .01639143887515023};
    for (int c = 0; c < 3; ++c) EXPECT_NEAR(nits(*m, 0, c), 203 * red2020[c] * 2, 3e-5);
}

TEST(HeadroomRender, PaintedMasksGateRegionsButNotMasterExposure) {
    auto source = frame(2, 1);
    auto q = request();
    q.exposure_ev = 1;
    q.regions = {{"bright", 200, 1000, 1}};
    auto masks = std::make_shared<MaskSet>();
    masks->width = 2; masks->height = 1;
    masks->planes[0] = std::make_shared<MaskPlane>(MaskPlane{0, 255});
    q.masks = masks;
    auto m = render_headroom_frame(source, q, kPolicy);
    ASSERT_TRUE(m);
    EXPECT_EQ(nits(*m, 0), 406);
    EXPECT_EQ(nits(*m, 1), 812);
    EXPECT_EQ(*masks->planes[0], (MaskPlane{0, 255}));
    // A preview mask sampled at another frame size uses existing bilinear
    // sampling; these centres land halfway between adjacent mask pixels.
    masks->width = 4;
    masks->planes[0] = std::make_shared<MaskPlane>(MaskPlane{0, 128, 128, 254});
    m = render_headroom_frame(source, q, kPolicy);
    ASSERT_TRUE(m);
    EXPECT_NEAR(nits(*m, 0), 406 * std::exp2(double(64.0f / 255.0f)), 5e-5);
    EXPECT_NEAR(nits(*m, 1), 406 * std::exp2(double(191.0f / 255.0f)), 8e-5);
}

TEST(HeadroomRender, UnsupportedControlsAreRefusedInsteadOfIgnored) {
    const auto source = frame();
    auto refused = [&](MasterRequest q) {
        EXPECT_FALSE(render_headroom_frame(source, q, kPolicy));
        EXPECT_FALSE(render_headroom_preview(source, q, kPolicy));
    };
    auto q = request(); q.anchor = true; refused(q);
    q = request(); q.carry_chroma = true; refused(q);
    q = request(); q.settle_grain = true; refused(q);
    for (const char* mode : {"off", "highlights", "shadows", "invalid"}) {
        q = request(); q.recovery_mode = mode; refused(q);
    }
    q = request(); q.strength = .5; refused(q);
    q = request(); q.preserve_outside = false; refused(q);
    q = request(); q.analytic = false; refused(q);
    q = request(); q.clip_fill = true; refused(q);
    q = request(); q.calibration = {{128, 50}}; refused(q);
    q = request(); q.reference.target_log2_nits = {0}; refused(q);
    q = request(); q.checkpoint = "research-model"; refused(q);
    q = request(); q.checkpoint_sha256 = "hash"; refused(q);
    q = request(); q.tiled = true; refused(q);
    q = request(); q.container = "invalid"; refused(q);
}

TEST(HeadroomRender, InvalidGradesMasksAndNumericalOverflowFailClosed) {
    const auto source = frame();
    const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
    for (double v : {nan, inf, -inf, -8.01, 8.01}) {
        auto q = request(); q.exposure_ev = v;
        EXPECT_FALSE(render_headroom_frame(source, q, kPolicy));
        q = request(); q.regions = {{"invalid", 100, 500, v}};
        EXPECT_FALSE(render_headroom_frame(source, q, kPolicy));
    }
    for (double v : {nan, inf, -1.0}) {
        auto q = request(); q.region_softness_stops = v;
        EXPECT_FALSE(render_headroom_frame(source, q, kPolicy));
    }
    for (auto range : {std::array{0.0, 500.0}, std::array{-1.0, 500.0}, std::array{500.0, 100.0},
                       std::array{100.0, 100.0}, std::array{nan, 500.0}, std::array{100.0, inf}}) {
        auto q = request(); q.regions = {{"invalid", range[0], range[1], 0}};
        EXPECT_FALSE(render_headroom_frame(source, q, kPolicy));
    }
    auto q = request(); q.masks_file = "not-loaded.png";
    EXPECT_FALSE(render_headroom_frame(source, q, kPolicy));
    q = request(); q.regions = {{"one", 100, 500, 1}};
    auto mask = std::make_shared<MaskSet>();
    mask->planes[0] = std::make_shared<MaskPlane>(MaskPlane{255}); q.masks = mask;
    EXPECT_FALSE(render_headroom_frame(source, q, kPolicy)); // zero dimensions
    mask->width = 2; mask->height = 1;
    EXPECT_FALSE(render_headroom_frame(source, q, kPolicy)); // short plane
    mask->width = 1; mask->planes[3] = mask->planes[0];
    EXPECT_FALSE(render_headroom_frame(source, q, kPolicy)); // absent band
    q = request(); q.exposure_ev = 8;
    auto huge = frame(1, 2e15f);
    ASSERT_TRUE(reconstruct_headroom_baseline(huge, q, kPolicy));
    EXPECT_FALSE(render_headroom_frame(huge, q, kPolicy)); // grade overflows float
    huge.encoded_rgb.at(2, 0, 0) = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(render_headroom_frame(huge, request(), kPolicy));
}

TEST(HeadroomRender, StatelessFramesDoNotCarryGainsAcrossCutsOrSeekOrder) {
    const auto bright = frame(1, 1.25f), dark = frame(1, .1f);
    auto q = request(); q.exposure_ev = 1;
    q.regions = {{"highlights", 200, 1000, -.5}};
    auto first = render_headroom_frame(bright, q, kPolicy);
    ASSERT_TRUE(first);
    ASSERT_TRUE(render_headroom_frame(dark, q, kPolicy));
    auto again = render_headroom_frame(bright, q, kPolicy);
    ASSERT_TRUE(again);
    EXPECT_EQ(first->linear2020.buffer().vector(), again->linear2020.buffer().vector());
}

TEST(HeadroomRender, PreviewAndMasterMatchBulkDoubleReferenceBitForBit) {
    // Keep the pre-optimization NitsFrame path as an independent grade oracle.
    // Exercise coloured pixels, negatives, ordinary codes and super-whites,
    // both primaries, noninteger EV, overlapping bands and resampled masks.
    ExtendedVideoFrame source{PlanarBuffer(3, 7, 41), {"rec709", "rec709", "bt709", "limited"}};
    for (std::size_t i = 0; i < source.encoded_rgb.span().size(); ++i)
        source.encoded_rgb.span()[i] = float((i * 131 + 17) % 997) / 613.0f - .12f;
    const auto input = source.encoded_rgb.vector();
    auto masks = std::make_shared<MaskSet>();
    masks->width = 13; masks->height = 3;
    for (int band : {0, 2}) {
        auto plane = std::make_shared<MaskPlane>(39);
        for (std::size_t i = 0; i < plane->size(); ++i)
            (*plane)[i] = static_cast<std::uint8_t>((i * 73 + std::size_t(band) * 31) % 256);
        masks->planes[std::size_t(band)] = plane;
    }
    for (const char* primaries : {"rec709", "rec2020"}) {
        source.source_contract.primaries = primaries;
        source.source_contract.matrix = std::string(primaries) == "rec709" ? "bt709" : "bt2020nc";
        for (double exposure : {-8.0, -.3, 0.0, 1e-20, 1.0 / 3.0, 8.0}) {
            for (int mode = 0; mode < 4; ++mode) {
                SCOPED_TRACE(std::string(primaries) + "/" + std::to_string(exposure) + "/" + std::to_string(mode));
                auto q = request(); q.source_space = primaries; q.exposure_ev = exposure;
                if (mode == 1) q.regions = {{"neutral", 12, 2000, 0}};
                if (mode >= 2) {
                    q.regions = {{"low", .05, 12, -.7}, {"mid", 4, 100, .3}, {"high", 90, 8000, 1.1}};
                    q.region_softness_stops = mode == 2 ? 0 : 1.37;
                }
                if (mode == 3) q.masks = masks;
                auto base = reconstruct_headroom_baseline(source, q, kPolicy);
                ASSERT_TRUE(base);
                auto expected = base->linear2020;
                std::vector<RegionBand> bands;
                for (const auto& b : q.regions) bands.push_back({b.low_nits, b.high_nits, b.ev});
                if (q.exposure_ev != 0 || any_graded(bands)) {
                    NitsFrame nits(expected.height(), expected.width());
                    for (std::size_t i = 0; i < nits.span().size(); ++i)
                        nits.span()[i] = double(expected.buffer().span()[i]) * 203.0;
                    apply_region_ev(nits, bands, q.region_softness_stops, std::numeric_limits<double>::max(),
                                    q.masks.get(), q.exposure_ev);
                    for (std::size_t i = 0; i < nits.span().size(); ++i)
                        expected.buffer().span()[i] = static_cast<float>(nits.span()[i] / 203.0);
                }
                auto master = render_headroom_frame(source, q, kPolicy);
                auto preview = render_headroom_preview(source, q, kPolicy);
                ASSERT_TRUE(master); ASSERT_TRUE(preview);
                const auto want = expected.buffer().span();
                EXPECT_EQ(std::memcmp(want.data(), master->linear2020.buffer().span().data(), want.size_bytes()), 0);
                EXPECT_EQ(std::memcmp(want.data(), preview->master.linear2020.buffer().span().data(), want.size_bytes()), 0);
                const auto before = base->linear2020.buffer().span();
                EXPECT_EQ(std::memcmp(before.data(), preview->baseline.linear2020.buffer().span().data(), before.size_bytes()), 0);
                EXPECT_EQ(preview->master.source_contract, source.source_contract);
                EXPECT_EQ(preview->baseline.source_contract, source.source_contract);
                EXPECT_EQ(preview->master.policy, kPolicy); EXPECT_EQ(preview->baseline.policy, kPolicy);
                EXPECT_EQ(source.encoded_rgb.vector(), input);
            }
        }
    }
}

TEST(HeadroomRender, PreviewOutputsDoNotAliasOrModifySourceAndMasks) {
    auto source = frame(2);
    auto q = request(); q.exposure_ev = 1;
    q.regions = {{"high", 200, 1000, 1}};
    auto masks = std::make_shared<MaskSet>(); masks->width = 2; masks->height = 1;
    masks->planes[0] = std::make_shared<MaskPlane>(MaskPlane{0, 255}); q.masks = masks;
    const auto input = source.encoded_rgb.vector();
    auto preview = render_headroom_preview(source, q, kPolicy);
    ASSERT_TRUE(preview);
    const auto base = preview->baseline.linear2020.buffer().vector();
    EXPECT_NE(preview->master.linear2020.buffer().vector(), base);
    preview->master.linear2020.buffer().at(0, 0, 0) = 1234;
    EXPECT_EQ(preview->baseline.linear2020.buffer().vector(), base);
    preview->baseline.linear2020.buffer().at(0, 0, 1) = 999;
    EXPECT_EQ(source.encoded_rgb.vector(), input);
    EXPECT_EQ(*masks->planes[0], (MaskPlane{0, 255}));
}

TEST(HeadroomRender, PreviewRefusesInvalidPolicyContractMasksAndOverflow) {
    auto source = frame();
    auto q = request();
    EXPECT_FALSE(render_headroom_preview(source, q));
    EXPECT_FALSE(render_headroom_preview(source, q, static_cast<HeadroomPolicy>(255)));
    q.source_space = "rec709";
    EXPECT_FALSE(render_headroom_preview(source, q, kPolicy));
    q = request(); source.source_contract.range = "full";
    EXPECT_FALSE(render_headroom_preview(source, q, kPolicy));
    source = frame();
    auto masks = std::make_shared<MaskSet>(); masks->width = 2; masks->height = 1;
    masks->planes[0] = std::make_shared<MaskPlane>(MaskPlane{255}); q.masks = masks;
    EXPECT_FALSE(render_headroom_preview(source, q, kPolicy));
    q = request(); q.exposure_ev = 8;
    EXPECT_FALSE(render_headroom_preview(frame(1, 2e15f), q, kPolicy));
    q = request(); q.exposure_ev = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(render_headroom_preview(source, q, kPolicy));
    source.encoded_rgb.vector().pop_back();
    EXPECT_FALSE(render_headroom_preview(source, request(), kPolicy));
    source = frame(1, std::numeric_limits<float>::quiet_NaN());
    EXPECT_FALSE(render_headroom_preview(source, request(), kPolicy));
}

TEST(HeadroomRender, DeliveryBridgeUsesAbsoluteNitsForEveryExistingProfile) {
    auto m = render_headroom_frame(frame(), request(), kPolicy);
    ASSERT_TRUE(m);
    const auto before = m->linear2020.buffer().vector();
    const double expected = 346.8007828288835;
    for (const auto* profiles : {&delivery_profiles(), &log_delivery_profiles()}) {
        for (const auto& p : *profiles) {
            SCOPED_TRACE(p.name);
            auto sf = master_headroom_video_frame(*m, 7, p.name, 1000, std::nullopt, true);
            ASSERT_TRUE(sf) << sf.error().message;
            EXPECT_EQ(sf->channels, 3);
            EXPECT_EQ(sf->packed.size(), 3);
            ASSERT_TRUE(sf->stats);
            EXPECT_NEAR(sf->light.max_cll, expected, 2e-4);
            EXPECT_NEAR(sf->light.frame_average, expected, 2e-4);
            if (p.transfer == "smpte2084") {
                const int want = int(std::floor(pq_code(expected) * 65535.0 + .5));
                for (auto code : sf->packed) EXPECT_LE(std::abs(int(code) - want), 1);
            } else if (p.name == "hlg") {
                const double scene = std::pow(expected / 1000, 1.0 / 1.2);
                const double code = .17883277 * std::log(12 * scene - (1 - 4 * .17883277)) +
                                    (.5 - .17883277 * std::log(4 * .17883277));
                EXPECT_NEAR(sf->packed[0], std::floor(code * 65535 + .5), 1);
            }
        }
    }
    EXPECT_EQ(m->linear2020.buffer().vector(), before);
}

TEST(HeadroomRender, DeliveryShoulderIsAppliedOnceAndLeavesLinearMasterIntact) {
    auto m = render_headroom_frame(frame(), request(), kPolicy);
    ASSERT_TRUE(m);
    const double before = nits(*m);
    auto pq = master_headroom_video_frame(*m, 0, "hdr10", 250, 200);
    ASSERT_TRUE(pq);
    const double want = 200 + 50 * (1 - std::exp(-(before - 200) / 50));
    EXPECT_NEAR(pq->light.max_cll, want, 1e-4);
    EXPECT_EQ(nits(*m), before);
    for (const auto& p : log_delivery_profiles()) {
        auto log = master_headroom_video_frame(*m, 0, p.name, 250, 200);
        ASSERT_TRUE(log);
        EXPECT_NEAR(log->light.max_cll, before, 1e-4); // existing log: no shoulder
    }
    auto high = render_headroom_frame(frame(1, 16), request(), kPolicy);
    ASSERT_TRUE(high);
    pq = master_headroom_video_frame(*high, 0, "hdr10", 1000);
    ASSERT_TRUE(pq);
    EXPECT_LE(pq->light.max_cll, 1000);
    EXPECT_NEAR(pq->light.max_cll, 1000, .01);
}

TEST(HeadroomRender, DeliveryBridgeRejectsUnknownPolicyFormatAndBrokenFrames) {
    auto m = render_headroom_frame(frame(), request(), kPolicy);
    ASSERT_TRUE(m);
    EXPECT_FALSE(master_headroom_video_frame(*m, 0, "hdr-typo", 1000));
    EXPECT_FALSE(master_headroom_video_frame(*m, 0, "hdr10", 99));
    EXPECT_FALSE(master_headroom_video_frame(*m, 0, "hdr10", 1000, 1000));
    m->policy = HeadroomPolicy::Disabled;
    EXPECT_FALSE(master_headroom_video_frame(*m, 0, "hdr10", 1000));
    m->policy = kPolicy;
    for (float v : {-1.0f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::max()}) {
        m->linear2020.buffer().at(0, 0, 0) = v;
        EXPECT_FALSE(master_headroom_video_frame(*m, 0, "hdr10", 1000));
    }
    m->linear2020.buffer().vector().pop_back();
    EXPECT_FALSE(master_headroom_video_frame(*m, 0, "hdr10", 1000));
    m->linear2020 = Image<space::SceneLinear2020>();
    EXPECT_FALSE(master_headroom_video_frame(*m, 0, "hdr10", 1000));
}
