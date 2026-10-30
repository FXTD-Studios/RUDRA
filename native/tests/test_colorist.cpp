// The colorist's pieces of the learned ACES 2.0 route (28 Oct 2026): SMPTE
// timecode (core/timecode.hpp), the EXR timeCode / framesPerSecond / reelName
// attributes, and the trims on the master (core/aces_v2_trims.hpp).

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <random>

#include "rudra/core/aces_v2_trims.hpp"
#include "rudra/core/timecode.hpp"
#include "rudra/deliver/exr.hpp"

using namespace rudra;

TEST(Timecode, ParsesFormatsAndCountsNonDrop) {
    auto tc = parse_timecode("01:00:00:00");
    ASSERT_TRUE(tc);
    EXPECT_FALSE(tc->drop);
    EXPECT_EQ(format_timecode(*tc), "01:00:00:00");
    EXPECT_FALSE(parse_timecode("1:00:00"));
    EXPECT_FALSE(parse_timecode("01:61:00:00"));
    EXPECT_FALSE(parse_timecode("01:00:00:0x"));
    EXPECT_EQ(timecode_at("01:00:00:00", 24, 24.0), "01:00:01:00");
    EXPECT_EQ(timecode_at("01:00:00:00", 25 * 60, 25.0), "01:01:00:00");
    EXPECT_EQ(timecode_at("00:59:59:23", 1, 24000.0 / 1001.0), "01:00:00:00");   // 23.976 counts in 24s
    EXPECT_EQ(timecode_at("23:59:59:23", 1, 24.0), "00:00:00:00");                // a day wraps
    EXPECT_EQ(timecode_at("not a timecode", 3, 24.0), "");
}

TEST(Timecode, DropFrameSkipsTheRightLabels) {
    const double ntsc = 30000.0 / 1001.0;
    EXPECT_TRUE(drop_frame_rate(ntsc));
    EXPECT_FALSE(drop_frame_rate(25.0));
    EXPECT_EQ(timecode_at("00:00:59;29", 1, ntsc), "00:01:00;02");   // ;00 and ;01 do not exist
    EXPECT_EQ(timecode_at("00:09:59;29", 1, ntsc), "00:10:00;00");   // every tenth minute keeps them
    EXPECT_FALSE(timecode_to_frames(*parse_timecode("00:01:00;00"), 30));
    // An hour of 29.97 drop frame is 107892 frames.
    EXPECT_EQ(*timecode_to_frames(*parse_timecode("01:00:00;00"), 30), 107892);
    EXPECT_EQ(timecode_at("00:00:00;00", 107892, ntsc), "01:00:00;00");
    EXPECT_EQ(timecode_at("00:00:59;58", 2, 60000.0 / 1001.0), "00:01:00;04");   // 59.94 drops four
}

TEST(Timecode, ExrPackingIsBcd) {
    auto p = exr_timecode(*parse_timecode("01:02:03:04"));
    EXPECT_EQ(p[0], 0x01020304u);
    EXPECT_EQ(p[1], 0u);
    p = exr_timecode(*parse_timecode("10:59:58;29"));
    EXPECT_EQ(p[0], 0x10595800u | 0x29u | (1u << 6));
}

TEST(Timecode, TheExrHeaderCarriesTimecodeRateAndReel) {
    const auto d = std::filesystem::temp_directory_path() / ("rudra_tc_exr_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(d);
    PlanarBuffer px(3, 2, 2, 0.5f), mask(1, 2, 2, 0.0f);
    const auto typed = exr_timecode_attributes("01:00:00:12", 24000.0 / 1001.0, "A001C003");
    ASSERT_TRUE(write_acescg_exr(d / "f.exr", px, Primaries::Ap1, 1.0, {}, true, &mask, typed));
    auto r = read_exr(d / "f.exr");
    ASSERT_TRUE(r) << r.error().message;
    auto find = [&](const std::string& name) -> std::pair<std::string, std::vector<std::uint8_t>> {
        std::string type;
        for (const auto& [n, t] : r->attribute_types)
            if (n == name) type = t;
        for (const auto& [n, v] : r->attributes)
            if (n == name) return {type, v};
        return {};
    };
    const auto tc = find("timeCode");
    ASSERT_EQ(tc.first, "timecode");
    ASSERT_EQ(tc.second.size(), 8u);
    EXPECT_EQ(tc.second[0] | tc.second[1] << 8 | tc.second[2] << 16 | tc.second[3] << 24, 0x01000012);
    const auto fps = find("framesPerSecond");
    ASSERT_EQ(fps.first, "rational");
    EXPECT_EQ(fps.second[0] | fps.second[1] << 8 | fps.second[2] << 16, 24000);
    EXPECT_EQ(fps.second[4] | fps.second[5] << 8, 1001);
    EXPECT_EQ(find("reelName").first, "string");
    // Without them the header is what it always was.
    auto plain = exr_bytes(px, true, kAp1Chromaticities, {}, &mask);
    auto same = exr_bytes(px, true, kAp1Chromaticities, {}, &mask, {});
    ASSERT_TRUE(plain && same);
    EXPECT_EQ(*plain, *same);
}

namespace {
PlanarBuffer rgb(float r, float g, float b, int n = 4) {
    PlanarBuffer p(3, 1, n);
    for (int i = 0; i < n; ++i) {
        p.plane(0)[i] = r * float(i + 1);
        p.plane(1)[i] = g * float(i + 1);
        p.plane(2)[i] = b * float(i + 1);
    }
    return p;
}
double luma(const PlanarBuffer& p, int i) {
    return 0.2722287168 * p.plane(0)[i] + 0.6740817658 * p.plane(1)[i] + 0.0536895174 * p.plane(2)[i];
}
}  // namespace

TEST(AcesTrims, DefaultsChangeNothingAndStrengthBlendsToTheInverse) {
    const PlanarBuffer master = rgb(8, 6, 2), base = rgb(1, 0.9f, 0.8f);
    PlanarBuffer mask(1, 1, 4, 0.0f);
    mask.plane(0)[2] = mask.plane(0)[3] = 1.0f;
    aces_v2::AcesTrims t;
    EXPECT_TRUE(t.identity());
    const auto same = aces_v2::apply_trims(master, base, mask, t);
    for (std::size_t i = 0; i < master.span().size(); ++i) EXPECT_EQ(same.span()[i], master.span()[i]);
    t.strength = 0.0;
    const auto inv = aces_v2::apply_trims(master, base, mask, t);
    for (std::size_t i = 0; i < base.span().size(); ++i) EXPECT_FLOAT_EQ(inv.span()[i], base.span()[i]);
    t.strength = 0.5;
    const auto half = aces_v2::apply_trims(master, base, mask, t);
    EXPECT_FLOAT_EQ(half.plane(0)[0], 0.5f * (8 + 1));
}

TEST(AcesTrims, ExposureAndContrastTrimTheWholeMasterAndKeepTheHue) {
    // Review 28 Oct: a colorist's exposure and contrast on the learned master.
    const PlanarBuffer master = rgb(0.36f, 0.18f, 0.09f), base = master;
    PlanarBuffer mask(1, 1, 4, 0.0f);
    aces_v2::AcesTrims t;
    t.exposure_ev = 1.0;
    EXPECT_FALSE(t.identity());
    const auto up = aces_v2::apply_trims(master, base, mask, t);
    EXPECT_FLOAT_EQ(up.plane(0)[0], 0.72f);   // one stop, everywhere (the mask is empty)
    t = {};
    t.contrast = 1.5;
    const auto c = aces_v2::apply_trims(master, base, mask, t);
    const double y = luma(master, 0);
    EXPECT_NEAR(luma(c, 0), 0.18 * std::pow(y / 0.18, 1.5), 1e-6);
    EXPECT_NEAR(c.plane(0)[0] / c.plane(1)[0], 2.0, 1e-5);   // hue kept
    EXPECT_EQ(aces_v2::trims_text(t), "strength 1.00, saturation 1.00, no peak limit, contrast 1.50");
    t.exposure_ev = -0.5;
    EXPECT_EQ(aces_v2::trims_text(t), "strength 1.00, saturation 1.00, no peak limit, exposure -0.50 EV, contrast 1.50");
}

TEST(AcesTrims, SaturationAndPeakTouchOnlyTheRebuiltPixels) {
    const PlanarBuffer master = rgb(8, 6, 2), base = master;
    PlanarBuffer mask(1, 1, 4, 0.0f);
    mask.plane(0)[2] = mask.plane(0)[3] = 1.0f;
    aces_v2::AcesTrims t;
    t.saturation = 0.0;
    const auto grey = aces_v2::apply_trims(master, base, mask, t);
    EXPECT_FLOAT_EQ(grey.plane(0)[0], master.plane(0)[0]);   // outside the mask: untouched
    EXPECT_NEAR(grey.plane(0)[3], grey.plane(2)[3], 1e-4);   // inside: neutral, same luminance
    EXPECT_NEAR(luma(grey, 3), luma(master, 3), 1e-4);
    t = {};
    t.peak_nits = 1000.0;   // 4.93 x diffuse white
    const auto capped = aces_v2::apply_trims(master, base, mask, t);
    const double cap = 1000.0 / aces_v2::kAcesDiffuseNits;
    EXPECT_GT(luma(master, 3), cap);
    EXPECT_LE(luma(capped, 3), cap + 1e-5);   // at most the cap (float rounding at the top)
    EXPECT_GT(luma(capped, 3), 0.75 * cap);
    EXPECT_NEAR(capped.plane(0)[3] / capped.plane(1)[3], master.plane(0)[3] / master.plane(1)[3], 1e-5);   // hue kept
    EXPECT_FLOAT_EQ(capped.plane(0)[1], master.plane(0)[1]);   // outside the mask: untouched even above the cap
    EXPECT_EQ(aces_v2::trims_text(t), "strength 1.00, saturation 1.00, peak 1000 nits");
}
