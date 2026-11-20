// AI fill v3's region in C++ (core/fill_region.hpp), the rule of rudra/genfill/masks.py
// (tests/test_genfill_core.py test_blend_core_drops_specks_and_keeps_areas is its twin).
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "rudra/core/fill_region.hpp"

using namespace rudra;

namespace {
// 200 x 200 of near-white (252) with a speck every 7 rows and 5 columns, and one clipped area of
// 20 x 30 with a ring of near-white round it: the shot of the user's 09V.mp4 in small.
std::vector<float> door() {
    std::vector<float> top(200 * 200, 252.0f / 255.0f);
    for (int y = 0; y < 200; y += 7)
        for (int x = 0; x < 200; x += 5) top[std::size_t(y) * 200 + std::size_t(x)] = 1.0f;
    for (int y = 18; y < 42; ++y)
        for (int x = 98; x < 132; ++x) top[std::size_t(y) * 200 + std::size_t(x)] = 252.0f / 255.0f;
    for (int y = 20; y < 40; ++y)
        for (int x = 100; x < 130; ++x) top[std::size_t(y) * 200 + std::size_t(x)] = 1.0f;
    return top;
}
}  // namespace

TEST(AiFillRegion, SpecksAreSeenButNotFilled) {
    const auto top = door();
    const auto r = ai_fill_region(top.data(), 200, 200, 250, -1, 0.01 - 1e-9);   // 399 px kept at least
    EXPECT_EQ(r.areas, 1);
    EXPECT_NEAR(r.filled_pct, 100.0 * 600 / 40000, 1e-9);
    EXPECT_NEAR(r.seen_pct, 100.0, 1e-9);                       // all of it is 250 or more
    EXPECT_GT(r.specks_pct, 2.0);
    EXPECT_EQ(AiFillLabel(r.label[25 * 200 + 110]), AiFillLabel::FilledHigh);
    EXPECT_EQ(AiFillLabel(r.label[0]), AiFillLabel::SeenHigh);   // a speck: seen, left alone
    EXPECT_FALSE(r.empty());
}

TEST(AiFillRegion, OffKeepsEveryClippedPixelAndTheThresholdsHold) {
    const auto top = door();
    const auto all = ai_fill_region(top.data(), 200, 200, 250, -1, 0.0);
    EXPECT_NEAR(all.specks_pct, 0.0, 1e-12);
    EXPECT_GT(all.filled_pct, 100.0 * 600 / 40000);
    const auto none = ai_fill_region(top.data(), 200, 200, 256, -1);
    EXPECT_EQ(none.seen_pct, 0.0);
    EXPECT_TRUE(none.empty());
    EXPECT_EQ(ai_fill_min_core_pixels(1920, 1080), 103);         // as masks.py min_core_pixels
    EXPECT_EQ(ai_fill_min_core_pixels(4, 4), 0);
}

TEST(AiFillRegion, ANearWhiteDoorHasNothingToFill) {
    std::vector<float> top(320 * 180, 0.2f);
    for (int y = 20; y < 160; ++y)
        for (int x = 60; x < 160; ++x) top[std::size_t(y) * 320 + std::size_t(x)] = 253.0f / 255.0f;
    const auto r = ai_fill_region(top.data(), 320, 180);
    EXPECT_TRUE(r.empty());
    EXPECT_GT(r.seen_pct, 20.0);
}

TEST(AiFillRegion, ShadowsWhenOnAndThePictureTintsWhatItFills) {
    SdrImage sdr(10, 10, 0.5f);
    for (int y = 0; y < 5; ++y)
        for (int x = 0; x < 10; ++x)
            for (int c = 0; c < 3; ++c) sdr.buffer().at(c, y, x) = c == 0 ? 1.0f : 0.99f;   // max code 255
    for (int c = 0; c < 3; ++c) sdr.buffer().at(c, 9, 9) = 0.0f;
    const auto r = ai_fill_region(sdr, 250, 4, 0.0);
    EXPECT_EQ(AiFillLabel(r.label[0]), AiFillLabel::FilledHigh);
    EXPECT_EQ(AiFillLabel(r.label[99]), AiFillLabel::FilledLow);
    EXPECT_EQ(AiFillLabel(r.label[55]), AiFillLabel::None);
    EXPECT_EQ(ai_fill_region(sdr, 250, -1, 0.0).label[99], std::uint8_t(AiFillLabel::None));
    PlanarBuffer d = sdr.buffer();
    ai_fill_region_tint_display(d, r);
    EXPECT_FLOAT_EQ(d.at(0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(d.at(2, 0, 0), 0.0f);    // orange
    EXPECT_FLOAT_EQ(d.at(0, 9, 9), 0.0f);    // cyan
    EXPECT_FLOAT_EQ(d.at(0, 5, 5), 0.25f);   // the rest dimmed
    const PlanarBuffer lin = ai_fill_region_picture(sdr, r);
    EXPECT_FLOAT_EQ(lin.at(0, 0, 0), 1.0f);
    EXPECT_NEAR(lin.at(0, 5, 5), std::pow(0.25f, 2.4f), 1e-6);
}

// ---- 1.2.7 (multi-role review of 1.2.6): what v4.3 really redraws ----

namespace {
SdrImage frame_of(int w, int h, float r, float g, float b) {
    SdrImage s(h, w, 0.0f);   // (height, width)
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) s.buffer().at(0, y, x) = r, s.buffer().at(1, y, x) = g, s.buffer().at(2, y, x) = b;
    return s;
}
void paint(SdrImage& s, int x0, int y0, int x1, int y1, float r, float g, float b) {
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x) s.buffer().at(0, y, x) = r, s.buffer().at(1, y, x) = g, s.buffer().at(2, y, x) = b;
}
}  // namespace

TEST(AiFillRegionV4, ASaturatedSunsetIsSeenButNotRedrawn) {
    SdrImage s = frame_of(200, 100, 0.1f, 0.1f, 0.1f);
    paint(s, 0, 0, 200, 40, 1.0f, 200.0f / 255.0f, 110.0f / 255.0f);      // red at 255, green 200, blue 110
    paint(s, 60, 50, 140, 90, 252.0f / 255.0f, 1.0f, 1.0f);               // a near-white window
    const auto r = ai_fill_region_v4(s);
    EXPECT_NEAR(r.seen_pct, 100.0 * (200 * 40 + 80 * 40) / (200 * 100), 1e-9);
    EXPECT_NEAR(r.filled_pct, 100.0 * 80 * 40 / (200 * 100), 1e-9);       // the window only
    EXPECT_EQ(AiFillLabel(r.label[10]), AiFillLabel::SeenHigh);           // the sunset: tinted, the master
    EXPECT_EQ(AiFillLabel(r.label[std::size_t(70) * 200 + 100]), AiFillLabel::FilledHigh);
    // The codes-only region of 1.2.6 would have filled both.
    EXPECT_NEAR(ai_fill_region(s, kAiFillV4TauHigh, -1, 0.0, kAiFillV4TauHigh).filled_pct, r.seen_pct, 1e-9);
    // The guard's ramp (composite.neutral_weight): min/max 0.78 -> 0.88.
    const auto w = ai_fill_v4_weight(s);
    EXPECT_FLOAT_EQ(w[10], 0.0f);
    EXPECT_FLOAT_EQ(w[std::size_t(70) * 200 + 100], 1.0f);
    SdrImage mid = frame_of(4, 4, 1.0f, 0.83f, 0.83f);   // 212/255: half way up the ramp
    EXPECT_NEAR(ai_fill_v4_weight(mid)[0], 0.5f, 0.06f);
}

TEST(AiFillRegionV4, KeepDetailLeavesAGratingAndALampToTheMaster) {
    // 1080 lines so the 25-px box and the 0.05 % rule are as at full HD.
    SdrImage s = frame_of(1920, 1080, 0.05f, 0.05f, 0.05f);
    paint(s, 600, 100, 1300, 800, 1.0f, 1.0f, 1.0f);                       // the door
    for (int x = 600; x < 1300; x += 6) paint(s, x, 900, x + 2, 1000, 1.0f, 1.0f, 1.0f);   // a floor grating: thin bars
    paint(s, 100, 100, 120, 130, 1.0f, 1.0f, 1.0f);                        // a practical lamp, 600 px
    const auto plain = ai_fill_region_v4(s, false);
    const auto kept = ai_fill_region_v4(s, true);
    auto at = [&](const AiFillRegion& r, int x, int y) { return AiFillLabel(r.label[std::size_t(y) * 1920 + std::size_t(x)]); };
    EXPECT_EQ(at(plain, 600, 950), AiFillLabel::FilledHigh);
    EXPECT_EQ(at(plain, 110, 115), AiFillLabel::FilledHigh);
    EXPECT_EQ(at(kept, 600, 950), AiFillLabel::SeenHigh);    // the grating keeps its picture
    EXPECT_EQ(at(kept, 110, 115), AiFillLabel::SeenHigh);    // the lamp too
    EXPECT_EQ(at(kept, 950, 450), AiFillLabel::FilledHigh);  // the door is redrawn
    EXPECT_GT(kept.filled_pct, 0.97 * 100.0 * 700 * 700 / (1920.0 * 1080.0));
    EXPECT_LT(kept.filled_pct, plain.filled_pct);
    EXPECT_NEAR(kept.seen_pct, plain.seen_pct, 1e-9);
}
