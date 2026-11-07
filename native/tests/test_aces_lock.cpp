// The lock (core/aces_lock.hpp, stage 6b) against its Python oracle
// (training/aces_partial_tighten.py) on golden/aces_lock/lock_parity_v1.bin, written by
// tools/make_lock_golden.py with the same toy view as below. Plus the contract: off is
// bit-exact, outside partial support is bit-exact, and the guarded result keeps every
// recorded channel within 1 code and every clipped channel clipped.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "rudra/core/aces_lock.hpp"
#include "rudra/core/aces_v2.hpp"

using namespace rudra;
using namespace rudra::aces_v2;

namespace {

// tests/test_aces_partial_tighten.py::MixingView: a channel-mixing matrix, then x / (1 + x).
class MixingView final : public AcesView {
public:
    static constexpr double M[3][3] = {{0.90, 0.08, 0.02}, {0.02, 0.96, 0.02}, {0.01, 0.04, 0.95}};
    void forward(float* rgb, std::size_t n) const override {
        for (std::size_t i = 0; i < n; ++i) {
            double x[3];
            for (int c = 0; c < 3; ++c) x[c] = std::max(static_cast<double>(rgb[i * 3 + c]), 0.0);
            for (int r = 0; r < 3; ++r) {
                const double y = x[0] * M[r][0] + x[1] * M[r][1] + x[2] * M[r][2];
                rgb[i * 3 + r] = static_cast<float>(y / (1.0 + y));
            }
        }
    }
    void inverse(float*, std::size_t) const override {}
    std::string identity() const override { return "mixing"; }
};

struct Golden {
    int h = 0, w = 0;
    PlanarBuffer source, constrained, weight;
    std::vector<double> dq;           // planar
    std::vector<std::uint8_t> used;   // planar
    struct Case {
        double strength = 0;
        bool dequant = false;
        int changed = 0, kept = 0, raised = 0, dequant_channels = 0;
        PlanarBuffer result;
    };
    std::vector<Case> cases;
};

template <class T>
T take(const char*& p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
}

PlanarBuffer take_hwc_f32(const char*& p, int h, int w, int ch) {
    PlanarBuffer b(ch, h, w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < ch; ++c) b.at(c, y, x) = take<float>(p);
    return b;
}

Golden load_golden() {
    const std::string path = std::string(RUDRA_GOLDEN_DIR) + "/aces_lock/lock_parity_v1.bin";
    std::ifstream f(path, std::ios::binary);
    EXPECT_TRUE(f.good()) << path;
    const std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    Golden g;
    if (bytes.size() < 16 || std::memcmp(bytes.data(), "RLK1", 4) != 0) {
        ADD_FAILURE() << "not a lock golden: " << path;
        return g;
    }
    const char* p = bytes.data() + 4;
    g.h = take<std::int32_t>(p);
    g.w = take<std::int32_t>(p);
    const int ncases = take<std::int32_t>(p);
    g.source = take_hwc_f32(p, g.h, g.w, 3);
    g.constrained = take_hwc_f32(p, g.h, g.w, 3);
    g.weight = take_hwc_f32(p, g.h, g.w, 1);
    const std::size_t n = std::size_t(g.h) * g.w;
    g.dq.resize(3 * n);
    g.used.resize(3 * n);
    for (std::size_t i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c) g.dq[c * n + i] = take<double>(p);
    for (std::size_t i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c) g.used[c * n + i] = take<std::uint8_t>(p);
    for (int k = 0; k < ncases; ++k) {
        Golden::Case cs;
        cs.strength = take<double>(p);
        cs.dequant = take<std::int32_t>(p) != 0;
        cs.changed = take<std::int32_t>(p);
        cs.kept = take<std::int32_t>(p);
        cs.raised = take<std::int32_t>(p);
        cs.dequant_channels = take<std::int32_t>(p);
        cs.result = take_hwc_f32(p, g.h, g.w, 3);
        g.cases.push_back(std::move(cs));
    }
    EXPECT_EQ(static_cast<std::size_t>(p - bytes.data()), bytes.size());
    return g;
}

std::vector<std::uint8_t> partial_support(const PlanarBuffer& s, const PlanarBuffer& w) {
    std::vector<std::uint8_t> m(s.plane_size());
    const float full = static_cast<float>(kClipCode);
    for (std::size_t i = 0; i < m.size(); ++i) {
        const float a = s.plane(0)[i], b = s.plane(1)[i], c = s.plane(2)[i];
        m[i] = w.plane(0)[i] > 0 && std::max({a, b, c}) > 0.5f && std::min({a, b, c}) < full;
    }
    return m;
}

}  // namespace

TEST(AcesLock, DequantizeMatchesThePythonOracle) {
    const Golden g = load_golden();
    ASSERT_FALSE(g.cases.empty());
    const auto sup = partial_support(g.source, g.weight);
    const Dequantized dq = dequantize_recorded(g.source, &sup, 200, 1.5);
    ASSERT_EQ(dq.values.size(), g.dq.size());
    double worst = 0;
    std::size_t used_diff = 0;
    for (std::size_t i = 0; i < g.dq.size(); ++i) {
        worst = std::max(worst, std::fabs(dq.values[i] - g.dq[i]) * 255.0);
        used_diff += dq.used[i] != g.used[i];
    }
    EXPECT_EQ(used_diff, 0u);
    EXPECT_LT(worst, 1e-9) << "codes";   // the same float64 operations in the same order
}

TEST(AcesLock, LockMatchesThePythonOracleForEveryRegisteredSetting) {
    const Golden g = load_golden();
    ASSERT_EQ(g.cases.size(), 3u);
    MixingView view;
    for (const auto& cs : g.cases) {
        LockSettings st = LockSettings::registered();
        st.strength = cs.strength;
        st.dequant = cs.dequant;
        LockStats stats;
        const PlanarBuffer out = lock_observed(g.source, g.constrained, g.weight, view, st, &stats);
        SCOPED_TRACE("strength " + std::to_string(cs.strength) + (cs.dequant ? " dequant" : " source"));
        EXPECT_EQ(int(stats.changed_pixels), cs.changed);
        EXPECT_EQ(int(stats.kept_pixels), cs.kept);
        EXPECT_EQ(int(stats.strength_raised_pixels), cs.raised);
        EXPECT_EQ(int(stats.dequant_channels), cs.dequant_channels);
        double worst = 0;
        std::size_t exact = 0;
        for (std::size_t i = 0; i < out.span().size(); ++i) {
            const double a = out.span()[i], b = cs.result.span()[i];
            worst = std::max(worst, std::fabs(a - b) / std::max(std::fabs(b), 1e-6));
            exact += a == b;
        }
        EXPECT_LT(worst, 1e-5) << "relative";
        std::printf("[lock parity] strength %.2f %s: %zu / %zu values bit-exact, worst relative %.3g\n", cs.strength,
                    cs.dequant ? "dequant" : "source", exact, out.span().size(), worst);
    }
}

TEST(AcesLock, OffIsBitExactAndOutsideSupportIsUntouched) {
    const Golden g = load_golden();
    MixingView view;
    // run_frame's default leaves the lock off; here the function itself on a frame with no weight.
    PlanarBuffer zero(1, g.h, g.w, 0.0f);
    LockStats s;
    const PlanarBuffer none = lock_observed(g.source, g.constrained, zero, view, LockSettings::registered(), &s);
    EXPECT_EQ(s.support_pixels, 0u);
    EXPECT_TRUE(std::equal(none.span().begin(), none.span().end(), g.constrained.span().begin()));
    const auto sup = partial_support(g.source, g.weight);
    const PlanarBuffer out = lock_observed(g.source, g.constrained, g.weight, view, LockSettings::registered());
    for (std::size_t i = 0; i < sup.size(); ++i) {
        if (sup[i]) continue;
        for (int c = 0; c < 3; ++c) ASSERT_EQ(out.plane(c)[i], g.constrained.plane(c)[i]) << i;
    }
}

TEST(AcesLock, GuardedResultStaysInsideTheAppearanceBounds) {
    const Golden g = load_golden();
    MixingView view;
    const PlanarBuffer out = lock_observed(g.source, g.constrained, g.weight, view, LockSettings::registered());
    const PlanarBuffer shown = view_forward(view, out);
    const auto sup = partial_support(g.source, g.weight);
    std::size_t changed = 0;
    for (std::size_t i = 0; i < sup.size(); ++i) {
        bool moved = false;
        for (int c = 0; c < 3; ++c) moved = moved || out.plane(c)[i] != g.constrained.plane(c)[i];
        if (!sup[i] || !moved) continue;
        ++changed;
        for (int c = 0; c < 3; ++c) {
            const double s = g.source.plane(c)[i], d = shown.plane(c)[i];
            if (s < kClipCode) EXPECT_LE(std::fabs(d - s) * 255.0, 1.0 + 1e-6) << i;
            else EXPECT_GE(d, kClipCode) << i;
        }
    }
    EXPECT_GT(changed, 1000u);
}

TEST(AcesLock, RunFrameRefusesABadStrength) {
    MixingView view;
    PlanarBuffer src(3, 4, 4, 0.5f);
    G1Fn g1 = [](const PlanarBuffer& s, const PlanarBuffer& b) -> Result<G1Output> {
        return G1Output{s, PlanarBuffer(3, s.height(), s.width()), PlanarBuffer(3, s.height(), s.width()),
                        PlanarBuffer(1, b.height(), b.width())};
    };
    LockSettings bad = LockSettings::registered();
    bad.strength = 0.0;
    EXPECT_FALSE(run_frame(src, view, g1, true, true, false, {}, {}, bad));
}

// Not a gate, and off in CI: the cost on a full HD frame that is ALL smooth partial
// highlights (the worst case; real frames have a few percent). Run it with
//   rudra_tests --gtest_filter=AcesLock.* --gtest_also_run_disabled_tests
TEST(AcesLock, DISABLED_TimingOnAFullHdFrameIsReported) {
    const int h = 1080, w = 1920;
    MixingView view;
    PlanarBuffer src(3, h, w), cur(3, h, w), weight(1, h, w, 1.0f);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            double sc[3] = {40.0 + 900.0 * x / (w - 1), 1.4 + 1.6 * y / (h - 1), 0.8 + 0.5 * x / (w - 1)};
            float px[3] = {float(sc[0]), float(sc[1]), float(sc[2])};
            view.forward(px, 1);
            for (int c = 0; c < 3; ++c) {
                src.at(c, y, x) = float(std::nearbyint(px[c] * 255.0) / 255.0);
                cur.at(c, y, x) = float(sc[c] * (1.0 + 0.01 * ((x * 7 + y * 13 + c) % 5 - 2)));
            }
        }
    LockStats s;
    const auto t = std::chrono::steady_clock::now();
    (void)lock_observed(src, cur, weight, view, LockSettings::registered(), &s);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    std::printf("[lock timing] 1920x1080, %zu support pixels: dequant %.0f ms, solve %.0f ms, total %.0f ms\n",
                s.support_pixels, s.ms_dequant, s.ms_solve, ms);
    EXPECT_GT(s.changed_pixels, 0u);
}
