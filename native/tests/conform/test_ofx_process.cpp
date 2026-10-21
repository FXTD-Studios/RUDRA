// The OFX plug-in's picture (roadmap 5.1): the master chain's own frame, in
// the output the plug-in offers, whatever the host's row order.
#include <gtest/gtest.h>

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
