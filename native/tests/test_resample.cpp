// RUDRA 1.3: Deliver's Output size (core/resample.hpp).
#include <rudra/core/block_grid.hpp>
#include <rudra/core/resample.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

using namespace rudra;

TEST(OutputSize, TheSizesKeepTheShapeAndAreEven) {
    EXPECT_EQ(output_size("source", 1920, 1080), (OutputSize{1920, 1080}));
    EXPECT_EQ(output_size("2x", 1920, 1080), (OutputSize{3840, 2160}));
    EXPECT_EQ(output_size("uhd", 1920, 1080), (OutputSize{3840, 2160}));
    EXPECT_EQ(output_size("hd", 3840, 2160), (OutputSize{1920, 1080}));
    EXPECT_EQ(output_size("uhd", 2048, 858), (OutputSize{3840, 1608}));   // scope 2.39
    EXPECT_EQ(output_size("hd", 4096, 2160), (OutputSize{1920, 1012}));
    EXPECT_EQ(output_size("2x", 1001, 501), (OutputSize{2002, 1002}));
    EXPECT_EQ(output_size("source", 1001, 501), (OutputSize{1001, 501}));   // the source as it is
    EXPECT_EQ(output_size("8k", 1920, 1080), (OutputSize{1920, 1080}));
    EXPECT_TRUE(valid_output_size("uhd"));
    EXPECT_FALSE(valid_output_size("4x"));
    EXPECT_NE(output_size_note("2x", 1920, 1080).find("3840 x 2160 from 1920 x 1080"), std::string::npos);
}

TEST(Resample, ALampGetsNoHaloAndNothingGoesBelowItsNeighbours) {
    // One 1,000-nit pixel and a 2-pixel lamp on a dim wall: Lanczos alone rings round both.
    PlanarBuffer x(3, 24, 24, 0.01f);
    for (int c = 0; c < 3; ++c) {
        x.at(c, 8, 8) = 1000.0f;
        x.at(c, 16, 15) = x.at(c, 16, 16) = 400.0f;
    }
    const PlanarBuffer y = resample_hdr(x, 48, 48);
    ASSERT_EQ(y.width(), 48);
    float lo = 1e9f, hi = -1e9f;
    for (float v : y.span()) lo = std::min(lo, v), hi = std::max(hi, v);
    EXPECT_GE(lo, 0.01f - 1e-6f);   // no dark ring, no negative
    EXPECT_LE(hi, 1000.0f);
    // Past the lamp's own neighbours the wall is untouched (Lanczos 3 reaches 3 source pixels).
    for (int yy = 0; yy < 48; ++yy)
        for (int xx = 0; xx < 48; ++xx) {
            const double sx = (xx + 0.5) / 2 - 0.5, sy = (yy + 0.5) / 2 - 0.5;
            const bool near_spot = std::abs(sx - 8) < 1.0 && std::abs(sy - 8) < 1.0;
            const bool near_lamp = std::abs(sy - 16) < 1.0 && sx > 14 - 1.0 && sx < 16 + 1.0;
            if (!near_spot && !near_lamp) ASSERT_NEAR(y.at(0, yy, xx), 0.01f, 1e-6f) << yy << "," << xx;
        }
}

TEST(Resample, AnEdgeStaysAnEdgeAndAFlatFieldStaysFlat) {
    PlanarBuffer x(1, 4, 16, 0.1f);
    for (int yy = 0; yy < 4; ++yy)
        for (int xx = 8; xx < 16; ++xx) x.at(0, yy, xx) = 50.0f;
    const PlanarBuffer y = resample_hdr(x, 8, 32);
    for (int xx = 1; xx < 32; ++xx) EXPECT_GE(y.at(0, 3, xx), y.at(0, 3, xx - 1));   // monotone: no overshoot
    EXPECT_FLOAT_EQ(y.at(0, 3, 0), 0.1f);
    EXPECT_FLOAT_EQ(y.at(0, 3, 31), 50.0f);
    const PlanarBuffer flat = resample_hdr(PlanarBuffer(3, 9, 7, 1.0f), 18, 14);
    for (float v : flat.span()) EXPECT_NEAR(v, 1.0f, 1e-6f);
}

TEST(Resample, ASmoothRampIsFollowedClosely) {
    PlanarBuffer x(1, 8, 64);
    for (int yy = 0; yy < 8; ++yy)
        for (int xx = 0; xx < 64; ++xx) x.at(0, yy, xx) = std::exp2(float(xx) / 8.0f);   // 8 stops
    const PlanarBuffer y = resample_hdr(x, 16, 128);
    for (int xx = 8; xx < 120; ++xx) {
        const double sx = (xx + 0.5) / 2 - 0.5;
        EXPECT_NEAR(y.at(0, 8, xx) / std::exp2(sx / 8.0), 1.0, 2e-3) << xx;
    }
}

TEST(Resample, DownscalingAveragesAndCannotRing) {
    PlanarBuffer x(1, 32, 32);
    for (int yy = 0; yy < 32; ++yy)
        for (int xx = 0; xx < 32; ++xx) x.at(0, yy, xx) = float((xx + yy) % 2);
    const PlanarBuffer y = resample_hdr(x, 16, 16);
    for (int yy = 1; yy < 15; ++yy)
        for (int xx = 1; xx < 15; ++xx) EXPECT_NEAR(y.at(0, yy, xx), 0.5f, 1e-5f);
    NitsFrame n(16, 16);
    for (int c = 0; c < 3; ++c) n.at(c, 5, 5) = 4000.0;
    const NitsFrame m = resample_hdr(n, 8, 8);
    double lo = 1e9, sum = 0;
    for (double v : m.span()) lo = std::min(lo, v), sum += v;
    EXPECT_GE(lo, 0.0);
    EXPECT_NEAR(sum / 3.0 / 64.0, 4000.0 / 256.0, 1e-6);   // the light is kept, spread over the smaller frame
}

// 1.3 (24 Nov 2026, audit): no side past 8192. 2x of a 6K or 8K source keeps the source's size
// (and Deliver greys 2x out for it); UHD and HD of the same source still resize.
TEST(OutputSize, NoSideIsWrittenPast8192) {
    EXPECT_TRUE(rudra::output_size_fits("2x", 3840, 2160));
    EXPECT_EQ(rudra::output_size("2x", 4096, 2160), (rudra::OutputSize{8192, 4320}));
    EXPECT_FALSE(rudra::output_size_fits("2x", 6144, 3456));
    EXPECT_EQ(rudra::output_size("2x", 6144, 3456), (rudra::OutputSize{6144, 3456}));
    EXPECT_FALSE(rudra::output_size_fits("2x", 7680, 4320));
    EXPECT_TRUE(rudra::output_size_fits("uhd", 7680, 4320));
    EXPECT_EQ(rudra::output_size("uhd", 7680, 4320), (rudra::OutputSize{3840, 2160}));
    EXPECT_TRUE(rudra::output_size_fits("source", 12288, 6480));   // the source is what it is
    EXPECT_EQ(rudra::output_size("source", 12288, 6480), (rudra::OutputSize{12288, 6480}));
    EXPECT_FALSE(rudra::output_size_fits("uhd", 1000, 4000));      // a tall frame: 3840 x 15360
}

// 1.3 (24 Nov 2026): Reconstruct > Clean compression measures the block grid and picks the cleanup strength.
TEST(BlockGrid, AGridOfBlocksMeasuresHighAndASmoothPictureOne) {
    const int w = 128, h = 96;
    std::vector<std::uint8_t> smooth(std::size_t(w) * h), blocks(std::size_t(w) * h);
    std::uint32_t seed = 7;
    auto rnd = [&seed] { seed = seed * 1664525u + 1013904223u; return int((seed >> 24) & 7); };
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const int ramp = 40 + (x + y) / 2;                            // a gradient with a little noise
            smooth[std::size_t(y) * w + x] = std::uint8_t(ramp + rnd());
            const int block = 40 + ((x / 8) * 13 + (y / 8) * 29) % 90;    // each 8x8 block a level of its own
            blocks[std::size_t(y) * w + x] = std::uint8_t(block + rnd());
        }
    const double s = block_grid(smooth.data(), w, h, w);
    const double b = block_grid(blocks.data(), w, h, w);
    EXPECT_NEAR(s, 1.0, 0.15) << s;
    EXPECT_GT(b, 2.0) << b;
    std::vector<std::uint8_t> flat(std::size_t(w) * h, 128);
    EXPECT_DOUBLE_EQ(block_grid(flat.data(), w, h, w), 1.0);
    EXPECT_DOUBLE_EQ(block_grid(nullptr, w, h, w), 1.0);
}

TEST(BlockGrid, TheCleanupIsTheFirstStrengthAtOrUnderTheTarget) {
    const int tried[] = {0, 2, 4, 6, 8};
    const double jpeg50[] = {1.66, 1.22, 1.06, 1.02, 1.01};      // the Meridian frames, 24 Nov 2026
    const double crf28[] = {1.08, 1.03, 1.02, 1.02, 1.01};
    const double stubborn[] = {1.40, 1.30, 1.25, 1.20, 1.15};
    EXPECT_EQ(choose_cleanup_qp(tried, jpeg50), 4);
    EXPECT_EQ(choose_cleanup_qp(tried, crf28), 0);               // nothing to clean
    EXPECT_EQ(choose_cleanup_qp(tried, stubborn), 8);            // the strongest tried
    EXPECT_EQ(choose_cleanup_qp(std::span<const int>(tried, 2), std::span<const double>(jpeg50, 3)), 0);
}
