// 1.3.1 (25 Nov 2026): the colour carry reads the SDR through the Source curve, as the anchor does.
// Under BT.1886 the anchor set each pixel's level from the curve but carry_source_chroma took its
// colour from the sRGB reading: analytic masters were 16 % less saturated than the SDR in the
// mid-tones and 38 % in the shadows (04.mp4, 24 Nov 2026). Now the carried channel ratios are the
// curve's own; with no knots (unknown, aces) the carry is the sRGB reading, bit for bit.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>

#include "rudra/core/baseline.hpp"
#include "rudra/core/composite.hpp"
#include "rudra/core/fields.hpp"
#include "rudra/core/master.hpp"
#include "rudra/core/source_curve.hpp"
#include "rudra/deliver/master.hpp"

using namespace rudra;

namespace {

// Coloured codes below the chroma knee: skin, foliage, sky, from shadows to bright.
SdrImage colour_patches(int h = 16, int w = 64, unsigned seed = 7) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.04f, 0.92f);
    PlanarBuffer p(3, h, w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) p.at(c, y, x) = u(rng);
    return SdrImage(std::move(p));
}

NitsFrame flat_nits(int h, int w, double v) {
    NitsFrame n(h, w);
    for (auto& x : n.span()) x = v;
    return n;
}

MasterRequest analytic(const std::string& curve, bool carry) {
    MasterRequest q;
    q.analytic = true;
    q.clip_fill = false;
    q.settle_grain = false;
    q.carry_chroma = carry;
    q.source_curve = curve;
    return q;
}

double saturation(double r, double g, double b) {
    const double mx = std::max({r, g, b}), mn = std::min({r, g, b});
    return mx > 0.0 ? (mx - mn) / mx : 0.0;
}

}  // namespace

TEST(ChromaSource, NoKnotsIsTheSrgbCarryBitForBit) {
    const SdrImage sdr = colour_patches();
    NitsFrame a = flat_nits(sdr.height(), sdr.width(), 50.0), b = a;
    carry_source_chroma(a, sdr);
    carry_source_chroma(b, sdr, kChromaKnee, kChromaSoftness, kChromaMaskSigma, {});
    EXPECT_TRUE(std::equal(a.span().begin(), a.span().end(), b.span().begin()));
}

TEST(ChromaSource, Bt1886KnotsCarryTheCurvesRatios) {
    const SdrImage sdr = colour_patches();
    CompositeParams p;
    p.source = SourceCurve::Bt1886;
    const auto knots = anchor_target_knots(p);
    NitsFrame n = flat_nits(sdr.height(), sdr.width(), 50.0);
    carry_source_chroma(n, sdr, kChromaKnee, kChromaSoftness, kChromaMaskSigma, knots);
    const PlanarBuffer& s = sdr.buffer();
    for (std::size_t i = 0; i < n.plane_size(); ++i) {
        double want[3], sum_w = 0.0, sum_g = 0.0;
        for (int c = 0; c < 3; ++c) {
            want[c] = double(source_curve_inverse(SourceCurve::Bt1886, srgb_to_linear(s.plane(c)[i])));
            sum_w += want[c], sum_g += n.plane(c)[i];
        }
        for (int c = 0; c < 3; ++c) EXPECT_NEAR(n.plane(c)[i] / sum_g, want[c] / sum_w, 1e-3) << i << " " << c;
    }
}

// The analytic master under BT.1886: before the carry its channels are already the curve's inverse,
// so the carry must leave their colour where it is (it took it to the sRGB reading before 1.3.1).
TEST(ChromaSource, TheAnalyticBt1886MasterKeepsTheSdrsSaturation) {
    const SdrImage sdr = colour_patches();
    const Fields f = analytic_fields(sdr.height(), sdr.width());
    auto on = grade_master(sdr, f, FrameScalars{}, kAnalyticConstants, analytic("bt1886", true));
    auto off = grade_master(sdr, f, FrameScalars{}, kAnalyticConstants, analytic("bt1886", false));
    ASSERT_TRUE(on && off);
    double worst = 0.0;
    for (std::size_t i = 0; i < on->nits.plane_size(); ++i) {
        const double a = saturation(on->nits.plane(0)[i], on->nits.plane(1)[i], on->nits.plane(2)[i]);
        const double b = saturation(off->nits.plane(0)[i], off->nits.plane(1)[i], off->nits.plane(2)[i]);
        worst = std::max(worst, std::abs(a - b));
    }
    EXPECT_LT(worst, 0.01);
}

// Unknown (the ACES inverse) has no knots: the carry is the sRGB reading, as in every master before.
TEST(ChromaSource, UnknownStillCarriesTheSrgbReading) {
    const SdrImage sdr = colour_patches();
    const Fields f = analytic_fields(sdr.height(), sdr.width());
    auto m = grade_master(sdr, f, FrameScalars{}, kAnalyticConstants, analytic("unknown", true));
    ASSERT_TRUE(m);
    const PlanarBuffer& s = sdr.buffer();
    for (std::size_t i = 0; i < m->nits.plane_size(); i += 37) {
        double sum_s = 0.0, sum_m = 0.0;
        for (int c = 0; c < 3; ++c) sum_s += srgb_to_linear(s.plane(c)[i]), sum_m += m->nits.plane(c)[i];
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(m->nits.plane(c)[i] / sum_m, double(srgb_to_linear(s.plane(c)[i])) / sum_s, 1e-3) << i;
    }
}
