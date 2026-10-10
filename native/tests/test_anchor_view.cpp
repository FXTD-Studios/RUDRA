// Roadmap 3.3: the master's anchor stage, live on the viewer. The display
// pass's per-pixel gain (core/view.cpp anchor_gain_f with anchor_hold) must be
// the master chain's (core/master.cpp anchor_to_sdr) to fp32, so what the
// viewer shows with Anchor on is what the master writes.

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "rudra/core/baseline.hpp"
#include "rudra/core/master.hpp"
#include "rudra/core/view.hpp"

using namespace rudra;

namespace {

struct Pair {
    SdrImage sdr;
    NetworkLinearImage model;
};

// An SDR with a gradient, a clipped patch and noise, and a model that is the
// baseline pushed around: the anchor has real work to do on both sides of the knee.
Pair make_pair(int h, int w, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> jitter(-0.03f, 0.03f);
    PlanarBuffer s(3, h, w), m(3, h, w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const std::size_t i = std::size_t(y) * std::size_t(w) + std::size_t(x);
            const float base = float(x) / float(w - 1);
            const bool patch = x > w * 3 / 4 && y > h / 2;
            for (int c = 0; c < 3; ++c) {
                const float v = patch ? 1.0f : std::clamp(base + jitter(rng) + 0.02f * float(c), 0.0f, 1.0f);
                s.plane(c)[i] = v;
                // The baseline at -1 EV, then a stop up in the top rows and 0.7x in the bottom: a level the anchor undoes.
                const float lift = y < h / 2 ? 2.0f : 0.7f;
                m.plane(c)[i] = baseline_value(v, -1.0f) * lift * (patch ? 3.0f : 1.0f);
            }
        }
    return {SdrImage(std::move(s)), NetworkLinearImage(std::move(m))};
}

}  // namespace

TEST(AnchorView, TheLiveGainIsTheMastersToFp32) {
    const auto p = make_pair(48, 96, 7);
    for (double knee : {0.9, 0.8}) {
        const double hold = anchor_hold(p.model, p.sdr, knee);
        EXPECT_TRUE(std::isfinite(hold));
        NitsFrame nits = nits_from_network(p.model);
        anchor_to_sdr(nits, p.sdr, knee);
        const PlanarBuffer& s = p.sdr.buffer();
        const PlanarBuffer& m = p.model.buffer();
        double worst = 0.0;
        for (std::size_t i = 0; i < m.plane_size(); ++i) {
            float target = 0.0f, mx = 0.0f;
            const float luma[3] = {0.2627f, 0.6780f, 0.0593f};
            for (int c = 0; c < 3; ++c) {
                target += srgb_to_linear(s.plane(c)[i]) * luma[c];
                mx = std::max(mx, s.plane(c)[i]);
            }
            const float gain = anchor_gain_f(target, m.plane(0)[i], m.plane(1)[i], m.plane(2)[i], mx, float(knee), 0.04f, float(hold));
            for (int c = 0; c < 3; ++c) {
                const double want = nits.plane(c)[i];
                const double got = double(m.plane(c)[i] * gain) * 10000.0;
                worst = std::max(worst, std::abs(got - want) / std::max(want, 1e-3));
            }
        }
        EXPECT_LT(worst, 2e-4) << "knee " << knee;   // fp32 against the double chain
    }
}

TEST(AnchorView, UnclippedPixelsLandOnTheSdrLevel) {
    const auto p = make_pair(32, 64, 3);
    const double hold = anchor_hold(p.model, p.sdr, 0.9);
    const PlanarBuffer& s = p.sdr.buffer();
    const PlanarBuffer& m = p.model.buffer();
    for (std::size_t i = 0; i < m.plane_size(); ++i) {
        float target = 0.0f, mx = 0.0f;
        for (int c = 0; c < 3; ++c) {
            target += srgb_to_linear(s.plane(c)[i]) * (c == 0 ? 0.2627f : c == 1 ? 0.6780f : 0.0593f);
            mx = std::max(mx, s.plane(c)[i]);
        }
        if (mx >= 0.86f || target < 0.01f) continue;   // below the knee band, away from black
        const float gain = anchor_gain_f(target, m.plane(0)[i], m.plane(1)[i], m.plane(2)[i], mx, 0.9f, 0.04f, float(hold));
        const float luma = (0.2627f * m.plane(0)[i] + 0.6780f * m.plane(1)[i] + 0.0593f * m.plane(2)[i]) * gain * 10000.0f;
        EXPECT_NEAR(luma, target * 203.0f, target * 203.0f * 1e-3f) << i;
    }
}

TEST(AnchorView, OffAndBaselineAreUntouched) {
    const auto p = make_pair(16, 32, 1);
    const NetworkLinearImage b(p.sdr.buffer());   // any picture will do as the baseline here
    auto off = view_params(ViewMode::Image, 1000.0);
    auto on = off;
    on.anchor = true;
    on.anchor_hold = anchor_hold(p.model, p.sdr);
    const auto a = render_view(p.model, b, off, &p.sdr), c = render_view(p.model, b, on, &p.sdr);
    bool differs = false;
    for (std::size_t i = 0; i < a.span().size(); ++i) differs = differs || a.span()[i] != c.span()[i];
    EXPECT_TRUE(differs);
    // The baseline view is the inverse as it is, anchor or not.
    auto base_off = view_params(ViewMode::Image, 1000.0, ViewSource::Baseline), base_on = base_off;
    base_on.anchor = true;
    base_on.anchor_hold = on.anchor_hold;
    const auto d = render_view(p.model, b, base_off, &p.sdr), e = render_view(p.model, b, base_on, &p.sdr);
    for (std::size_t i = 0; i < d.span().size(); ++i) ASSERT_EQ(d.span()[i], e.span()[i]) << i;
    // Without the SDR there is nothing to anchor to: the picture is the composite's.
    const auto f = render_view(p.model, b, on, nullptr);
    for (std::size_t i = 0; i < a.span().size(); ++i) ASSERT_EQ(a.span()[i], f.span()[i]) << i;
}

// Since 9 Oct 2026 the master grades AFTER its anchor (Region EV used to be
// divided straight back out by it). The view does the same: the anchor's
// target carries the composite's grade gain, so below the knee a pushed band
// lands at the SDR's level times the push, and the hold is the ungraded one.
TEST(AnchorView, AGradeSurvivesTheAnchor) {
    const auto p = make_pair(32, 64, 11);
    const std::size_t n = p.model.buffer().plane_size();
    std::vector<float> gain(n);
    PlanarBuffer graded_buf = p.model.buffer();
    for (std::size_t i = 0; i < n; ++i) {
        gain[i] = (i % 3 == 0) ? 2.0f : (i % 3 == 1 ? 0.5f : 1.0f);   // +1 EV, -1 EV, none
        for (int c = 0; c < 3; ++c) graded_buf.plane(c)[i] *= gain[i];
    }
    const NetworkLinearImage graded(std::move(graded_buf));
    // The hold of the picture before its grade.
    EXPECT_NEAR(anchor_hold(graded, p.sdr, 0.9, 0.04, &gain), anchor_hold(p.model, p.sdr, 0.9), 1e-6);
    const PlanarBuffer& s = p.sdr.buffer();
    const PlanarBuffer& m = graded.buffer();
    int checked = 0;
    for (std::size_t i = 0; i < n; ++i) {
        float target = 0.0f, mx = 0.0f;
        for (int c = 0; c < 3; ++c) {
            target += srgb_to_linear(s.plane(c)[i]) * (c == 0 ? 0.2627f : c == 1 ? 0.6780f : 0.0593f);
            mx = std::max(mx, s.plane(c)[i]);
        }
        if (mx >= 0.86f || target < 0.01f) continue;
        const float g = anchor_gain_f(target * gain[i], m.plane(0)[i], m.plane(1)[i], m.plane(2)[i], mx, 0.9f, 0.04f, 1.0f);
        const float luma = (0.2627f * m.plane(0)[i] + 0.6780f * m.plane(1)[i] + 0.0593f * m.plane(2)[i]) * g * 10000.0f;
        EXPECT_NEAR(luma, target * 203.0f * gain[i], target * 203.0f * gain[i] * 1e-3f) << i;
        ++checked;
    }
    EXPECT_GT(checked, 100);
    // And render_view takes the gain: the pushed view differs from one without it.
    const NetworkLinearImage b(p.sdr.buffer());
    auto v = view_params(ViewMode::Image, 4000.0);
    v.anchor = true;
    v.anchor_hold = anchor_hold(graded, p.sdr, 0.9, 0.04, &gain);
    const auto without = render_view(graded, b, v, &p.sdr);
    v.grade_gain = &gain;
    const auto with = render_view(graded, b, v, &p.sdr);
    bool differs = false;
    for (std::size_t i = 0; i < with.span().size(); ++i) differs = differs || with.span()[i] != without.span()[i];
    EXPECT_TRUE(differs);
}
