// The OFX plug-in's picture (roadmap 5.1): the master chain's own frame, in
// the output the plug-in offers, whatever the host's row order.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "ofx_process.hpp"
#include "rudra/core/gamut.hpp"
#include "rudra/core/hdr10.hpp"
#include "rudra/deliver/master.hpp"

using namespace rudra;

namespace {

constexpr int kW = 32, kH = 18;

std::vector<float> rgba() {
    std::vector<float> p(std::size_t(kW) * kH * 4);
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) {
            float* q = &p[(std::size_t(y) * kW + std::size_t(x)) * 4];
            q[0] = float(x) / (kW - 1), q[1] = float(y) / (kH - 1), q[2] = x == y ? 1.0f : 0.4f, q[3] = 0.25f;
        }
    return p;
}

// The same frame through grade_master, in Rec.2020 nits.
PlanarBuffer reference(const OfxGrade& g) {
    const auto p = rgba();
    PlanarBuffer code(3, kH, kW);
    for (std::size_t i = 0; i < std::size_t(kW) * kH; ++i)
        for (int c = 0; c < 3; ++c) code.plane(c)[i] = p[i * 4 + std::size_t(c)];
    const SdrImage sdr(std::move(code));
    MasterRequest q;
    q.analytic = true, q.source_curve = g.source_curve, q.exposure_ev = g.exposure_ev, q.container = "linear";
    q.own_codes = true;   // the plug-in reads the host's codes as the app reads a movie's
    q.region = g.region;  // the luminance limit (26 Nov 2026): the app's region
    auto m = grade_master(sdr, analytic_fields_for(sdr, q), FrameScalars{}, kAnalyticConstants, q);
    EXPECT_TRUE(m);
    PlanarBuffer nits(3, kH, kW);
    for (int c = 0; c < 3; ++c)
        for (std::size_t i = 0; i < std::size_t(kW) * kH; ++i) nits.plane(c)[i] = float(m->nits.plane(c)[i]);
    return convert_primaries(nits, Primaries::Rec709, Primaries::Rec2020);
}

}  // namespace

TEST(OfxProcess, LinearRec2020IsTheMasterOver203) {
    const auto in = rgba();
    std::vector<float> out(in.size());
    OfxGrade g;
    g.exposure_ev = 0.5;
    ASSERT_TRUE(render_sdr_to_hdr(g, in.data(), kW * 4, out.data(), kW * 4, kW, kH));
    const auto ref = reference(g);
    for (std::size_t i = 0; i < std::size_t(kW) * kH; ++i) {
        for (int c = 0; c < 3; ++c) EXPECT_FLOAT_EQ(out[i * 4 + std::size_t(c)], std::max(ref.plane(c)[i], 0.0f) / 203.0f);
        EXPECT_EQ(out[i * 4 + 3], 0.25f);
    }
}

TEST(OfxProcess, PqIsTheSameLightEncoded) {
    const auto in = rgba();
    std::vector<float> lin(in.size()), pq(in.size());
    OfxGrade g;
    ASSERT_TRUE(render_sdr_to_hdr(g, in.data(), kW * 4, lin.data(), kW * 4, kW, kH));
    g.output = "pq_rec2020";
    ASSERT_TRUE(render_sdr_to_hdr(g, in.data(), kW * 4, pq.data(), kW * 4, kW, kH));
    for (std::size_t i = 0; i < lin.size(); ++i) {
        if (i % 4 == 3) continue;
        EXPECT_NEAR(pq_eotf(pq[i]), lin[i] * 203.0f, std::max(0.002f, lin[i] * 203.0f * 2e-4f));
    }
}

TEST(OfxProcess, RowsGoingUpGiveTheSamePicture) {
    // OFX rows run bottom up: a negative stride from the last row is the same frame.
    const auto in = rgba();
    std::vector<float> flipped(in.size()), a(in.size()), b(in.size());
    for (int y = 0; y < kH; ++y)
        std::copy(in.begin() + std::ptrdiff_t(y) * kW * 4, in.begin() + std::ptrdiff_t(y + 1) * kW * 4,
                  flipped.begin() + std::ptrdiff_t(kH - 1 - y) * kW * 4);
    OfxGrade g;
    ASSERT_TRUE(render_sdr_to_hdr(g, in.data(), kW * 4, a.data(), kW * 4, kW, kH));
    ASSERT_TRUE(render_sdr_to_hdr(g, flipped.data() + std::ptrdiff_t(kH - 1) * kW * 4, -kW * 4,
                                  b.data() + std::ptrdiff_t(kH - 1) * kW * 4, -kW * 4, kW, kH));
    for (int y = 0; y < kH; ++y)
        for (int i = 0; i < kW * 4; ++i)
            ASSERT_EQ(a[std::size_t(y) * kW * 4 + std::size_t(i)], b[std::size_t(kH - 1 - y) * kW * 4 + std::size_t(i)]);
}

TEST(OfxProcess, BadSettingsAreRefused) {
    const auto in = rgba();
    std::vector<float> out(in.size());
    OfxGrade g;
    g.source_curve = "kodak";
    EXPECT_FALSE(render_sdr_to_hdr(g, in.data(), kW * 4, out.data(), kW * 4, kW, kH));
    g = {};
    g.output = "log";
    EXPECT_FALSE(render_sdr_to_hdr(g, in.data(), kW * 4, out.data(), kW * 4, kW, kH));
    g = {};
    g.exposure_ev = 4.0;
    EXPECT_FALSE(render_sdr_to_hdr(g, in.data(), kW * 4, out.data(), kW * 4, kW, kH));
    EXPECT_FALSE(render_sdr_to_hdr({}, in.data(), kW * 4, out.data(), kW * 4, 0, kH));
}

TEST(OfxProcess, Bt1886ReadsTheHostsCodesAsVideo) {
    // 25 Nov 2026: a flat grey at codes 26, 51 and 128 comes out as code^2.4 (diffuse white = 1.0),
    // not 0.36-2.7 stops darker as before own codes.
    for (int code : {26, 51, 128}) {
        std::vector<float> in(std::size_t(kW) * kH * 4), out(in.size());
        for (std::size_t i = 0; i < std::size_t(kW) * kH; ++i)
            for (int c = 0; c < 4; ++c) in[i * 4 + std::size_t(c)] = c == 3 ? 1.0f : float(code) / 255.0f;
        OfxGrade g;
        g.source_curve = "bt1886";
        ASSERT_TRUE(render_sdr_to_hdr(g, in.data(), kW * 4, out.data(), kW * 4, kW, kH));
        const double want = std::pow(double(code) / 255.0, 2.4);
        const std::size_t mid = (std::size_t(kH / 2) * kW + std::size_t(kW / 2)) * 4;
        EXPECT_NEAR(std::log2(double(out[mid + 1]) / want), 0.0, 0.1) << "code " << code;
    }
}

// Phase 3 (26 Nov 2026): the luminance limit is the app's Reconstruct > Region, exactly.
TEST(OfxProcess, TheLuminanceLimitIsTheAppsRegion) {
    const auto in = rgba();
    std::vector<float> out(in.size());
    rudra_ofx::Settings s;
    s.limit = "highlights", s.highlights_from = 0.6, s.limit_softness = 0.05;
    OfxGrade g = ofx_grade(s);
    ASSERT_TRUE(g.region.enabled);
    EXPECT_EQ(g.region.highlight_strength, 1.0f);
    EXPECT_EQ(g.region.shadow_strength, 0.0f);
    ASSERT_TRUE(render_sdr_to_hdr(g, in.data(), kW * 4, out.data(), kW * 4, kW, kH));
    const auto ref = reference(g);
    for (std::size_t i = 0; i < std::size_t(kW) * kH; ++i)
        for (int c = 0; c < 3; ++c) EXPECT_FLOAT_EQ(out[i * 4 + std::size_t(c)], std::max(ref.plane(c)[i], 0.0f) / 203.0f);
    // A band and both edges map as the app's region does.
    s.limit = "band", s.band_from = 0.2, s.band_to = 0.7;
    g = ofx_grade(s);
    EXPECT_TRUE(g.region.band);
    EXPECT_FLOAT_EQ(g.region.band_low, 0.2f);
    EXPECT_FLOAT_EQ(g.region.band_high, 0.7f);
    s.limit = "both";
    g = ofx_grade(s);
    EXPECT_EQ(g.region.highlight_strength, 1.0f);
    EXPECT_EQ(g.region.shadow_strength, 1.0f);
    s.limit = "off";
    EXPECT_FALSE(ofx_grade(s).region.enabled);
}

TEST(OfxProcess, AlphaCanCarryTheLimit) {
    const auto in = rgba();
    std::vector<float> out(in.size());
    rudra_ofx::Settings s;
    s.limit = "shadows", s.shadows_below = 0.3, s.limit_softness = 0.1, s.alpha = "limit";
    ASSERT_EQ(rudra_ofx::render(s, in.data(), kW * 4, out.data(), kW * 4, kW, kH).error, "");
    const ReconstructionRegion r = ofx_region(s);
    for (std::size_t i = 0; i < std::size_t(kW) * kH; ++i) {
        const float y = 0.2126f * in[i * 4] + 0.7152f * in[i * 4 + 1] + 0.0722f * in[i * 4 + 2];
        float wh = 1.0f, ws = 1.0f;
        region_weights(y, r, wh, ws);
        EXPECT_FLOAT_EQ(out[i * 4 + 3], std::clamp(std::max(wh, ws), 0.0f, 1.0f)) << i;
    }
    s.alpha = "source", s.limit = "off";
    ASSERT_EQ(rudra_ofx::render(s, in.data(), kW * 4, out.data(), kW * 4, kW, kH).error, "");
    EXPECT_EQ(out[3], 0.25f);   // the source's alpha
}

TEST(OfxProcess, ABadLimitIsRefused) {
    const auto in = rgba();
    std::vector<float> out(in.size());
    rudra_ofx::Settings s;
    s.limit = "band", s.band_from = 0.8, s.band_to = 0.3;   // upside down
    EXPECT_NE(rudra_ofx::render(s, in.data(), kW * 4, out.data(), kW * 4, kW, kH).error, "");
    s.limit = "off", s.alpha = "everything";
    EXPECT_NE(rudra_ofx::render(s, in.data(), kW * 4, out.data(), kW * 4, kW, kH).error, "");
}

// 27 Nov 2026: the Mask input. 1 everywhere is the picture without a mask; 0 everywhere is the plain inverse (a
// limit with no reconstruction); half way between them where it is 0.5. The limit's alpha is the mask's too.
TEST(OfxProcess, TheMaskBlendsRudraIntoThePlainInverse) {
    const auto in = rgba();
    const std::size_t n = std::size_t(kW) * kH;
    std::vector<float> full(in.size()), none(in.size()), half(in.size()), plain(in.size()), out(in.size());
    rudra_ofx::Settings s;
    ASSERT_EQ(rudra_ofx::render(s, in.data(), kW * 4, full.data(), kW * 4, kW, kH).error, "");
    rudra_ofx::Settings p = s;   // the plain inverse: a limit with both strengths 0
    OfxGrade pg = ofx_grade(p);
    pg.region.enabled = true, pg.region.highlight_strength = pg.region.shadow_strength = 0.0f;
    ASSERT_TRUE(render_sdr_to_hdr(pg, in.data(), kW * 4, plain.data(), kW * 4, kW, kH));
    s.mask.assign(n, 1.0f);
    ASSERT_EQ(rudra_ofx::render(s, in.data(), kW * 4, out.data(), kW * 4, kW, kH).error, "");
    for (std::size_t i = 0; i < in.size(); ++i) EXPECT_NEAR(out[i], full[i], 1e-6f * std::max(1.0f, full[i])) << i;
    s.mask.assign(n, 0.0f);
    ASSERT_EQ(rudra_ofx::render(s, in.data(), kW * 4, none.data(), kW * 4, kW, kH).error, "");
    for (std::size_t i = 0; i < in.size(); ++i) EXPECT_NEAR(none[i], plain[i], 1e-6f * std::max(1.0f, plain[i])) << i;
    s.mask.assign(n, 0.5f);
    ASSERT_EQ(rudra_ofx::render(s, in.data(), kW * 4, half.data(), kW * 4, kW, kH).error, "");
    for (std::size_t i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c) {
            const std::size_t k = i * 4 + std::size_t(c);
            EXPECT_NEAR(half[k], 0.5f * (full[k] + plain[k]), 1e-5f * std::max(1.0f, full[k])) << k;
        }
    s.mask.assign(n, 0.5f), s.alpha = "limit";   // no limit: the alpha is the mask
    ASSERT_EQ(rudra_ofx::render(s, in.data(), kW * 4, out.data(), kW * 4, kW, kH).error, "");
    EXPECT_FLOAT_EQ(out[3], 0.5f);
    s.mask.assign(3, 1.0f);   // not the picture's size
    EXPECT_NE(rudra_ofx::render(s, in.data(), kW * 4, out.data(), kW * 4, kW, kH).error, "");
}

// 1.3.2: Resolve hands a plug-in its clip's frames by their source media numbers and does not say where a trimmed clip
// starts, so AI fill there takes From (else the playhead) to To (else the clip's end on Resolve's timeline, else the
// clip's frame range end, when either is near enough to be the clip's, else 100 frames).
TEST(OfxProcess, ResolvesAiFillFramesComeFromThePlayheadToTheClipsEnd) {
    using rudra_ofx::resolve_frames;
    auto f = resolve_frames(129, 0, 0, 225.0, 1.0e9);        // a trimmed clip: source frames 129..225
    EXPECT_EQ(f.first, 129);
    EXPECT_EQ(f.last, 225);
    f = resolve_frames(16, 0, 0, 0.0, 96.0);                  // no timeline suite: the clip's range end
    EXPECT_EQ(f.first, 16);
    EXPECT_EQ(f.last, 96);
    f = resolve_frames(16, 0, 0, 0.0, 1.0e9);                 // a placeholder range: 100 frames
    EXPECT_EQ(f.last, 115);
    f = resolve_frames(16, 0, 0, 16.0 + 2500.0, 96.0);        // a timeline end too far to be the clip's
    EXPECT_EQ(f.last, 96);
    f = resolve_frames(300, 0, 0, 225.0, 96.0);               // both behind the playhead: 100 frames from it
    EXPECT_EQ(f.first, 300);
    EXPECT_EQ(f.last, 399);
    f = resolve_frames(16, 1, 30, 225.0, 96.0);               // From and To as typed (Resolve's source frame numbers)
    EXPECT_EQ(f.first, 1);
    EXPECT_EQ(f.last, 30);
    f = resolve_frames(16, 0, 5, 225.0, 96.0);                // To before the first frame: one frame
    EXPECT_EQ(f.first, 16);
    EXPECT_EQ(f.last, 16);
}
