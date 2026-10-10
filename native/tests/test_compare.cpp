// The compare line and the Changes tint: what the model changed against the
// baseline, how much of it a view peak shows, and the peak that shows it.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "rudra/core/compare.hpp"
#include "rudra/core/readouts.hpp"
#include "rudra/core/view.hpp"

using namespace rudra;

namespace {

constexpr float kNits = 1.0f / 10000.0f;   // nits to network units
constexpr double kInf = std::numeric_limits<double>::infinity();

// A 10 x 10 pair, grey at `base` nits; `n` pixels of the model set to `model`.
struct Pair {
    NetworkLinearImage model, baseline;
};
Pair pair(float base, int n, float model) {
    Pair p{NetworkLinearImage(10, 10, base * kNits), NetworkLinearImage(10, 10, base * kNits)};
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c) p.model.buffer().at(c, i / 10, i % 10) = model * kNits;
    return p;
}

}  // namespace

TEST(Compare, NothingChanged) {
    const Pair p = pair(100.0f, 0, 0.0f);
    const CompareStats s = compare_stats(p.model, p.baseline);
    EXPECT_EQ(s.changed_pct, 0.0);
    EXPECT_EQ(s.fit_nits, 0.0);
    const CompareText t = compare_text(s, 203.0, kInf);
    EXPECT_FALSE(t.warn);
    EXPECT_FALSE(t.fit_shown);
    EXPECT_NE(t.line.find("matches the baseline"), std::string::npos);
}

TEST(Compare, NearBlackNoiseIsNotAChange) {
    // 0.001 against 0.004 nits is two stops, and invisible: under the floor.
    const Pair p = pair(0.001f, 50, 0.004f);
    EXPECT_EQ(compare_stats(p.model, p.baseline).changed_pct, 0.0);
}

TEST(Compare, HighlightsAboveTheViewPeakAreCountedAndHidden) {
    // The case beta 2 shipped with: the baseline at 1 000 nits, the model at
    // 4 000 on 5 % of the frame. At a 203-nit view both clip to white.
    const Pair p = pair(1000.0f, 5, 4000.0f);
    const CompareStats s = compare_stats(p.model, p.baseline);
    EXPECT_DOUBLE_EQ(s.changed_pct, 5.0);
    EXPECT_NEAR(s.up_stops, 2.0, 1e-4);
    EXPECT_EQ(s.down_stops, 0.0);
    EXPECT_EQ(s.visible_pct(203.0), 0.0);
    EXPECT_EQ(s.visible_pct(900.0), 0.0);
    EXPECT_DOUBLE_EQ(s.visible_pct(1200.0), 5.0);   // clears the darker side by 0.1 stop
    EXPECT_GE(s.fit_nits, 4000.0);
    EXPECT_LT(s.fit_nits, 4000.0 * 1.1);

    const CompareText t = compare_text(s, 203.0, kInf);
    EXPECT_TRUE(t.warn);
    EXPECT_EQ(t.line, "RUDRA changed 5.0% of this frame, up to +2.0 stops; at 203 nits almost none of it shows");
    ASSERT_TRUE(t.fit_shown);
    EXPECT_DOUBLE_EQ(t.fit_peak_ev, 4.5);   // 203 * 2^4.5 = 4 593, the first half-EV step over 4 000
    EXPECT_EQ(t.fit_label, "Show at 4,593 nits");

    // At the fitted peak all of it shows and the button goes away.
    const CompareText at = compare_text(s, 203.0 * std::exp2(4.5), kInf);
    EXPECT_FALSE(at.warn);
    EXPECT_FALSE(at.fit_shown);
    EXPECT_EQ(at.line, "RUDRA changed 5.0% of this frame, up to +2.0 stops; all of it shows at 4,593 nits");
}

TEST(Compare, AnHdrDisplayCapsTheFit) {
    const Pair p = pair(1000.0f, 5, 4000.0f);
    const CompareStats s = compare_stats(p.model, p.baseline);
    // An 800-nit panel: the fit stops at the first step over the panel.
    const CompareText t = compare_text(s, 203.0, 800.0);
    ASSERT_TRUE(t.fit_shown);
    EXPECT_DOUBLE_EQ(t.fit_peak_ev, 2.0);   // 812 nits
    // Already at the panel's peak: no button, and the line says why.
    const CompareText at = compare_text(s, 812.0, 800.0);
    EXPECT_TRUE(at.warn);
    EXPECT_FALSE(at.fit_shown);
    EXPECT_NE(at.line.find("above this display's peak"), std::string::npos) << at.line;
}

TEST(Compare, DarkeningAndBothDirections) {
    Pair p = pair(10.0f, 10, 2.0f);   // 10 % darker by about 2.3 stops
    for (int c = 0; c < 3; ++c) p.model.buffer().at(c, 9, 9) = 40.0f * kNits;
    const CompareStats s = compare_stats(p.model, p.baseline);
    EXPECT_DOUBLE_EQ(s.changed_pct, 11.0);
    EXPECT_LT(s.down_stops, -2.0);
    EXPECT_GT(s.up_stops, 1.9);
    const CompareText t = compare_text(s, 203.0, kInf);
    EXPECT_FALSE(t.warn);   // all of it below the view peak
    EXPECT_NE(t.line.find("+2.0 and −2.3 stops"), std::string::npos) << t.line;
}

TEST(Compare, ChangeWeightIsSignedAndRamps) {
    EXPECT_EQ(change_weight(100.0f, 100.0f), 0.0f);
    EXPECT_EQ(change_weight(100.0f * std::exp2(0.04f), 100.0f), 0.0f);
    EXPECT_NEAR(change_weight(100.0f * std::exp2(0.10f), 100.0f), 0.5f, 1e-3);
    EXPECT_EQ(change_weight(400.0f, 100.0f), 1.0f);
    EXPECT_EQ(change_weight(25.0f, 100.0f), -1.0f);
}

TEST(Compare, TheChangesTintMarksOnlyChangedPixels) {
    const Pair p = pair(1000.0f, 5, 4000.0f);
    ViewParams v = view_params(ViewMode::Image, 203.0);
    const PlanarBuffer plain = render_view(p.model, p.baseline, v);
    v.show_changes = true;
    const PlanarBuffer tinted = render_view(p.model, p.baseline, v);
    // Changed (row 0, x < 5): white mixed toward amber; unchanged: as before.
    for (int k = 0; k < 3; ++k) {
        EXPECT_NEAR(tinted.at(k, 0, 0), 1.0f * (1.0f - kChangeTintMix) + kChangeUp[std::size_t(k)] * kChangeTintMix, 1e-6);
        EXPECT_EQ(tinted.at(k, 0, 7), plain.at(k, 0, 7));
        EXPECT_EQ(tinted.at(k, 5, 5), plain.at(k, 5, 5));
    }
    // The baseline shown, or a wipe: the tint still marks where the model differs.
    v.show = ViewSource::Baseline;
    EXPECT_EQ(render_view(p.model, p.baseline, v).at(2, 0, 0), tinted.at(2, 0, 0));
    // Other layers are never tinted.
    v.mode = ViewMode::Difference;
    const PlanarBuffer diff_t = render_view(p.model, p.baseline, v);
    v.show_changes = false;
    const PlanarBuffer diff = render_view(p.model, p.baseline, v);
    EXPECT_EQ(diff_t.at(0, 0, 0), diff.at(0, 0, 0));
}

TEST(Compare, TheTintOnHdrPathsIsGraphicsAtTheSdrWhite) {
    const Pair p = pair(1000.0f, 5, 4000.0f);
    ViewParams v = view_params(ViewMode::Image, 1000.0);
    v.target = DisplayTarget::scrgb(1000.0);
    v.source = Primaries::Rec709;
    v.show_changes = true;
    const PlanarBuffer out = render_view(p.model, p.baseline, v);
    // scRGB: 1.0 is 80 nits. The pixel is 1 000 nits mixed toward 203 * amber.
    const float a = kChangeTintMix;
    const float amber_r = 203.0f * std::pow((0.95f + 0.055f) / 1.055f, 2.4f);
    EXPECT_NEAR(out.at(0, 0, 0) * 80.0f, 1000.0f * (1.0f - a) + amber_r * a, 0.05f);
}

// ---- the invented-pixel map (roadmap 3.5) ----------------------------------

namespace {

// The pair above, with an SDR whose first `clipped` pixels are at code 1.0.
SdrImage sdr_with(int clipped, float code = 0.5f) {
    SdrImage s(10, 10, code);
    for (int i = 0; i < clipped; ++i)
        for (int c = 0; c < 3; ++c) s.buffer().at(c, i / 10, i % 10) = 1.0f;
    return s;
}

}  // namespace

TEST(Invented, ClassifiesByWhatTheSdrHas) {
    EXPECT_EQ(support_of(100.0f, 100.0f, 1.0f), Support::Follows);
    EXPECT_EQ(support_of(400.0f, 100.0f, 1.0f), Support::Invented);         // clipped
    EXPECT_EQ(support_of(0.5f, 2.0f, 0.0f), Support::Invented);             // crushed
    EXPECT_EQ(support_of(400.0f, 100.0f, 0.6f), Support::Reinterpreted);    // the SDR has detail
    EXPECT_EQ(support_of(400.0f, 100.0f, 253.0f / 255.0f), Support::Reinterpreted);
    EXPECT_EQ(support_of(400.0f, 100.0f, 254.0f / 255.0f), Support::Invented);
}

TEST(Invented, SharesOfTheFrame) {
    // 5 changed pixels: 3 where the SDR clipped, 2 where it has detail; 1 more
    // clipped pixel the model left alone.
    const Pair p = pair(1000.0f, 5, 4000.0f);
    const SupportStats s = support_stats(p.model, p.baseline, sdr_with(3));
    EXPECT_DOUBLE_EQ(s.invented_pct, 3.0);
    EXPECT_DOUBLE_EQ(s.reinterpreted_pct, 2.0);
    EXPECT_DOUBLE_EQ(s.sdr_without_detail_pct, 3.0);
    const std::string t = support_text(s);
    EXPECT_NE(t.find("Invented (magenta) 3.0%"), std::string::npos) << t;
    EXPECT_NE(t.find("Reinterpreted (cyan) 2.0%"), std::string::npos) << t;
}

TEST(Invented, TheLayerColoursByClass) {
    const Pair p = pair(1000.0f, 5, 4000.0f);
    const SdrImage sdr = sdr_with(3);
    const ViewParams v = view_params(ViewMode::Invented, 203.0);
    const PlanarBuffer out = render_view(p.model, p.baseline, v, &sdr);
    const float g = kMapGrey * linear_to_srgb(4000.0f / 203.0f);   // clips to 1: grey 0.6
    const float a = kMapMix;
    for (int k = 0; k < 3; ++k) {
        EXPECT_NEAR(out.at(k, 0, 0), g * (1.0f - a) + kInventedColour[std::size_t(k)] * a, 1e-6) << k;       // clipped SDR
        EXPECT_NEAR(out.at(k, 0, 4), g * (1.0f - a) + kReinterpretedColour[std::size_t(k)] * a, 1e-6) << k;  // SDR detail
        EXPECT_NEAR(out.at(k, 5, 5), kMapGrey * linear_to_srgb(1.0f), 1e-6) << k;                            // unchanged
    }
    // Without the SDR every change reads as reinterpreted.
    const PlanarBuffer blind = render_view(p.model, p.baseline, v);
    EXPECT_EQ(blind.at(0, 0, 0), out.at(0, 0, 4));
    EXPECT_EQ(sdr_max_codes(nullptr, 2, 2), std::vector<float>(4, 0.5f));
}
