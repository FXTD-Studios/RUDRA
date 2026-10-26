// ACES v2 staged stages (core/aces_v2.hpp) with a mock view: no OCIO, no
// LibTorch. Bit parity with the Python reference on real frames is measured by
// tools/aces_v2_parity (OCIO + LibTorch builds); these tests pin the contract.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include "rudra/core/aces_v2.hpp"

using namespace rudra;
using namespace rudra::aces_v2;

namespace {

// Invertible pointwise stand-in for the ACES view: display = x / (1 + x), x >= 0.
class MockView final : public AcesView {
public:
    void forward(float* rgb, std::size_t n) const override {
        for (std::size_t i = 0; i < n * 3; ++i) {
            const float x = std::max(rgb[i], 0.0f);
            rgb[i] = x / (1.0f + x);
        }
    }
    void inverse(float* rgb, std::size_t n) const override {
        for (std::size_t i = 0; i < n * 3; ++i) {
            const float y = std::min(rgb[i], 0.999f);
            rgb[i] = y / (1.0f - y);
        }
    }
    std::string identity() const override { return "mock"; }
};

PlanarBuffer fill3(int h, int w, float r, float g, float b) {
    PlanarBuffer p(3, h, w);
    for (std::size_t i = 0; i < p.plane_size(); ++i) {
        p.plane(0)[i] = r;
        p.plane(1)[i] = g;
        p.plane(2)[i] = b;
    }
    return p;
}

void set_px(PlanarBuffer& p, int y, int x, float r, float g, float b) {
    p.at(0, y, x) = r;
    p.at(1, y, x) = g;
    p.at(2, y, x) = b;
}

// A 40x48 frame: grey background, a fully clipped 16x20 block, a partially
// clipped strip and a crushed shadow strip.
PlanarBuffer test_frame() {
    PlanarBuffer s = fill3(40, 48, 0.4f, 0.35f, 0.3f);
    for (int y = 10; y < 26; ++y)
        for (int x = 12; x < 32; ++x) set_px(s, y, x, 1.0f, 1.0f, 1.0f);
    for (int x = 0; x < 48; ++x) set_px(s, 34, x, 1.0f, 0.8f, 0.6f);
    for (int x = 0; x < 48; ++x) set_px(s, 2, x, 0.0f, 0.0f, 2.0f / 255.0f);
    return s;
}

G1Output fake_g1(const PlanarBuffer& source, float hl, float full) {
    const int h = source.height(), w = source.width();
    G1Output g{source, fill3(h, w, hl, hl, hl), fill3(h, w, full, full, full), reconstruction_weight(source)};
    // textured full-branch fields, so smoothing has something to remove
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) g.full_fields.at(c, y, x) = full * (1.0f + 0.5f * std::sin(0.7f * static_cast<float>(x) + 1.3f * static_cast<float>(y) + static_cast<float>(c)));
    return g;
}

}  // namespace

TEST(AcesV2, ReconstructionWeightBands) {
    PlanarBuffer s = fill3(1, 4, 0.5f, 0.5f, 0.5f);
    set_px(s, 0, 1, 1.0f, 0.2f, 0.2f);   // clipped channel -> 1
    set_px(s, 0, 2, 0.0f, 0.0f, 0.0f);   // black -> 1
    set_px(s, 0, 3, 3.0f / 255, 0.0f, 0.0f);   // inside the low band: partial
    auto w = reconstruction_weight(s);
    EXPECT_EQ(w.at(0, 0, 0), 0.0f);
    EXPECT_EQ(w.at(0, 0, 1), 1.0f);
    EXPECT_EQ(w.at(0, 0, 2), 1.0f);
    EXPECT_GT(w.at(0, 0, 3), 0.0f);
    EXPECT_LT(w.at(0, 0, 3), 1.0f);
}

TEST(AcesV2, ComposeRoutesEachRegion) {
    MockView v;
    PlanarBuffer s = test_frame();
    PlanarBuffer base = view_inverse(v, s);
    G1Output g = fake_g1(s, 0.25f, 0.5f);
    PlanarBuffer restored = view_inverse(v, g.restored_source);
    Composed c = compose_v2(s, base, restored, g);
    // ordinary pixel: restored, mask 0
    EXPECT_EQ(c.acescg.at(1, 30, 5), restored.at(1, 30, 5));
    EXPECT_EQ(c.mask.at(0, 30, 5), 0.0f);
    // crushed shadow at w == 1: exactly the baseline, never a learned value, mask 0
    EXPECT_EQ(c.acescg.at(2, 2, 5), base.at(2, 2, 5));
    EXPECT_EQ(c.mask.at(0, 2, 5), 0.0f);
    // partial highlight: highlight fields, mask = weight
    const float scale = std::max({std::fabs(restored.at(0, 34, 5)), std::fabs(restored.at(1, 34, 5)), std::fabs(restored.at(2, 34, 5))}) + 0.001f;
    EXPECT_FLOAT_EQ(c.acescg.at(1, 34, 5), restored.at(1, 34, 5) + 1.0f * (scale * 0.25f));
    EXPECT_EQ(c.mask.at(0, 34, 5), 1.0f);
    // fully clipped: full fields
    EXPECT_GT(c.acescg.at(0, 15, 20), restored.at(0, 15, 20));
    EXPECT_NE(c.acescg.at(0, 15, 20), restored.at(0, 15, 20) + scale * 0.25f);
}

TEST(AcesV2, SmoothingFlattensInteriorAndKeepsOutsideExact) {
    MockView v;
    PlanarBuffer s = test_frame();
    G1Output g = fake_g1(s, 0.25f, 0.5f);
    Composed c = compose_v2(s, view_inverse(v, s), view_inverse(v, g.restored_source), g);
    SmoothingStats st;
    PlanarBuffer out = smooth_clipped_hybrid(c.acescg, s, 64.0, &st);
    EXPECT_EQ(st.full_pixels, 16u * 20u);
    EXPECT_EQ(st.components, 1u);
    EXPECT_EQ(st.interior_pixels, 12u * 16u);   // cross erosion x2 removes 2 px per side
    for (int y = 0; y < 40; ++y)
        for (int x = 0; x < 48; ++x) {
            const bool full = y >= 10 && y < 26 && x >= 12 && x < 32;
            for (int ch = 0; ch < 3; ++ch) {
                if (!full) {
                    EXPECT_EQ(out.at(ch, y, x), c.acescg.at(ch, y, x));
                }
                const bool interior = y >= 12 && y < 24 && x >= 14 && x < 30;
                if (interior) {
                    EXPECT_EQ(out.at(ch, y, x), out.at(ch, 15, 20));
                }
            }
        }
}

TEST(AcesV2, SmoothingIsNoOpWithoutClipping) {
    PlanarBuffer s = fill3(8, 8, 0.6f, 0.5f, 0.4f);
    PlanarBuffer p = fill3(8, 8, 1.0f, 2.0f, 3.0f);
    p.at(0, 3, 3) = 7.0f;
    PlanarBuffer out = smooth_clipped_hybrid(p, s);
    EXPECT_TRUE(std::equal(out.span().begin(), out.span().end(), p.span().begin()));
}

TEST(AcesV2, ConstraintBoundsDisplayErrorAndKeepsOutsideExact) {
    MockView v;
    PlanarBuffer s = test_frame();
    PlanarBuffer base = view_inverse(v, s);
    PlanarBuffer w = reconstruction_weight(s);
    PlanarBuffer pred = base;
    for (std::size_t i = 0; i < pred.plane_size(); ++i)
        for (int c = 0; c < 3; ++c) pred.plane(c)[i] = base.plane(c)[i] * 3.0f + 0.01f;   // far off the source
    ConstraintStats st;
    PlanarBuffer out = constrain_colour_v2(s, pred, base, w, v, true, &st);
    EXPECT_GT(st.support_pixels, 0u);
    EXPECT_EQ(st.remaining_display_errors, 0u);
    PlanarBuffer shown = view_forward(v, out);
    for (int y = 0; y < 40; ++y)
        for (int x = 0; x < 48; ++x) {
            const bool support = w.at(0, y, x) > 0 && std::max({s.at(0, y, x), s.at(1, y, x), s.at(2, y, x)}) > 0.5f;
            for (int c = 0; c < 3; ++c) {
                if (support) {
                    EXPECT_LE(std::fabs(shown.at(c, y, x) - s.at(c, y, x)) * 255.0f, 1.0f);
                } else {
                    EXPECT_EQ(out.at(c, y, x), pred.at(c, y, x));
                }
            }
        }
}

// Saturating stand-in: display = clamp(x, 0, 1), so a negative green on a clipped-blue
// pixel passes the 1-code display check (the case seen on the duration6400 run).
class SaturatingView final : public AcesView {
public:
    void forward(float* rgb, std::size_t n) const override {
        for (std::size_t i = 0; i < n * 3; ++i) rgb[i] = std::min(std::max(rgb[i], 0.0f), 1.0f);
    }
    void inverse(float* rgb, std::size_t n) const override { (void)rgb; (void)n; }
    std::string identity() const override { return "saturating"; }
};

TEST(AcesV2, LuminanceGuardRemovesNewNegativeY) {
    SaturatingView v;
    PlanarBuffer s = fill3(2, 2, 0.2f, 0.2f, 0.2f), base = s, pred = s, w(1, 2, 2);
    set_px(s, 0, 0, 0.0f, 0.0f, 1.0f);
    set_px(base, 0, 0, 0.0f, 0.01f, 5.0f);
    pred = base;
    set_px(pred, 0, 0, 0.0f, -5.0f, 30.0f);   // Y < 0, displays exactly
    for (std::size_t i = 0; i < w.plane_size(); ++i) w.plane(0)[i] = 0.0f;
    w.at(0, 0, 0) = 1.0f;
    auto y = [](const PlanarBuffer& b) {
        return 0.2722287168 * b.at(0, 0, 0) + 0.6740817658 * b.at(1, 0, 0) + 0.0536895174 * b.at(2, 0, 0);
    };
    ConstraintStats off, on;
    PlanarBuffer a = constrain_colour_v2(s, pred, base, w, v, true, &off, false);
    EXPECT_LT(y(a), 0.0);
    EXPECT_EQ(off.new_negative_luminance, 1u);
    PlanarBuffer b = constrain_colour_v2(s, pred, base, w, v, true, &on, true);
    EXPECT_GE(y(b), 0.0);
    EXPECT_EQ(on.new_negative_luminance, 0u);
    EXPECT_EQ(on.remaining_display_errors, 0u);
    EXPECT_EQ(on.changed_pixels, 1u);
    for (int c = 0; c < 3; ++c) EXPECT_EQ(b.at(c, 1, 1), pred.at(c, 1, 1));
}

TEST(AcesV2, RunFrameEndToEndAndRejectsBadInput) {
    MockView v;
    PlanarBuffer s = test_frame();
    G1Fn g1 = [](const PlanarBuffer& src, const PlanarBuffer&) -> Result<G1Output> { return fake_g1(src, 0.1f, 0.2f); };
    auto r = run_frame(s, v, g1);
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r->acescg.height(), 40);
    EXPECT_EQ(r->constraint.remaining_display_errors, 0u);
    EXPECT_EQ(r->smoothing.components, 1u);
    PlanarBuffer bad = s;
    bad.at(0, 0, 0) = 1.5f;
    EXPECT_FALSE(run_frame(bad, v, g1).ok());
    G1Fn failing = [](const PlanarBuffer&, const PlanarBuffer&) -> Result<G1Output> {
        return make_error(ErrorCode::BackendError, "no");
    };
    EXPECT_FALSE(run_frame(s, v, failing).ok());
}

// ---- engine job (decode and write injected) --------------------------------
#include <atomic>
#include <filesystem>
#include <map>
#include "rudra/engine/aces_v2_job.hpp"

TEST(AcesV2Job, WritesEveryFrameInOrderAndReports) {
    MockView v;
    const auto dir = std::filesystem::temp_directory_path() / "rudra_aces_v2_job_test";
    std::filesystem::remove_all(dir);
    std::map<std::string, std::size_t> written;
    AcesV2Request req{{"a.png", "b.png", "c.png"}, dir, true, "deadbeef"};
    G1Fn g1 = [](const PlanarBuffer& src, const PlanarBuffer&) -> Result<G1Output> { return fake_g1(src, 0.1f, 0.2f); };
    AcesV2Decode decode = [](const std::filesystem::path&) -> Result<AcesV2Decoded> { return AcesV2Decoded{test_frame(), 10}; };
    AcesV2Write write = [&](const std::filesystem::path& exr, const FrameResult& f,
                            const std::vector<std::pair<std::string, std::string>>& prov) -> Result<void> {
        written[exr.filename().string()] = f.mask.plane_size();
        EXPECT_EQ(prov[1].second, "deadbeef");
        return {};
    };
    AcesV2Job job(v, g1, req, decode, write);
    std::size_t seen = 0;
    auto out = job.run([&](const AcesV2FrameReport& r) { EXPECT_EQ(r.index, seen++); EXPECT_EQ(r.bits, 10); });
    EXPECT_FALSE(out.error.has_value());
    EXPECT_EQ(out.completed, 3u);
    EXPECT_TRUE(out.appearance_pass);
    EXPECT_EQ(written.size(), 3u);
    EXPECT_EQ(written.count("b.exr"), 1u);
    EXPECT_GT(out.frames[0].reconstructed_fraction, 0.0);
    // the same folder again is refused, nothing overwritten
    AcesV2Job again(v, g1, req, decode, write);
    EXPECT_TRUE(again.run().error.has_value());
    std::filesystem::remove_all(dir);
}

TEST(AcesV2Job, StopsOnDecodeFailureAndOnCancel) {
    MockView v;
    const auto dir = std::filesystem::temp_directory_path() / "rudra_aces_v2_job_test2";
    std::filesystem::remove_all(dir);
    G1Fn g1 = [](const PlanarBuffer& src, const PlanarBuffer&) -> Result<G1Output> { return fake_g1(src, 0.1f, 0.2f); };
    AcesV2Decode decode = [](const std::filesystem::path& p) -> Result<AcesV2Decoded> {
        if (p == "bad.png") return make_error(ErrorCode::IoError, "unreadable");
        return AcesV2Decoded{test_frame(), 8};
    };
    AcesV2Write write = [](const std::filesystem::path&, const FrameResult&,
                           const std::vector<std::pair<std::string, std::string>>&) -> Result<void> { return {}; };
    AcesV2Job job(v, g1, AcesV2Request{{"a.png", "bad.png", "c.png"}, dir, true, ""}, decode, write);
    auto out = job.run();
    EXPECT_EQ(out.completed, 1u);
    ASSERT_TRUE(out.error.has_value());
    std::filesystem::remove_all(dir);

    AcesV2Job job2(v, g1, AcesV2Request{{"a.png", "b.png", "c.png"}, dir, true, ""}, decode, write);
    auto out2 = job2.run([&](const AcesV2FrameReport&) { job2.cancel(); });
    EXPECT_TRUE(out2.cancelled);
    EXPECT_EQ(out2.completed, 1u);
    std::filesystem::remove_all(dir);

    std::atomic<bool> done{false};
    AcesV2Job job3(v, g1, AcesV2Request{{"a.png", "c.png"}, dir, true, ""}, decode, write);
    job3.start({}, [&](const AcesV2Outcome& o) { EXPECT_EQ(o.completed, 2u); done = true; });
    while (!done) std::this_thread::yield();
    std::filesystem::remove_all(dir);
}
