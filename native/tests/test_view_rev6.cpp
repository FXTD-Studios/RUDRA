// docs/view.spec.md revision 6 (1 Nov 2026): the viewer exposure and gamma,
// the Sdr source and the A | B wipe, the clamp and screen-clip zebras, the
// Range layer, and the range readout (core/range.hpp). The browser goldens
// (test_viewer.cpp) hold revision 5, which the defaults must still produce.
#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "rudra/core/compare.hpp"
#include "rudra/core/baseline.hpp"
#include "rudra/core/composite.hpp"
#include "rudra/core/measure.hpp"
#include "rudra/core/range.hpp"
#include "rudra/core/view.hpp"

using namespace rudra;

namespace {

constexpr float kP = 10000.0f;

// A 16 x 8 frame: SDR codes, a model and a baseline in network units.
struct Frame {
    SdrImage sdr{8, 16};
    NetworkLinearImage model{8, 16}, base{8, 16};
    void sdr_px(int x, int y, int r, int g, int b) {
        sdr.buffer().at(0, y, x) = float(r) / 255.0f;
        sdr.buffer().at(1, y, x) = float(g) / 255.0f;
        sdr.buffer().at(2, y, x) = float(b) / 255.0f;
    }
    void nits(NetworkLinearImage& im, int x, int y, float n) {
        for (int k = 0; k < 3; ++k) im.buffer().at(k, y, x) = n / kP;
    }
};

Frame random_frame(unsigned seed) {
    Frame f;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> code(0.0f, 1.0f), lin(0.0f, 0.2f);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 16; ++x)
            for (int k = 0; k < 3; ++k) {
                f.sdr.buffer().at(k, y, x) = std::round(code(rng) * 255.0f) / 255.0f;
                f.model.buffer().at(k, y, x) = lin(rng);
                f.base.buffer().at(k, y, x) = lin(rng) * 0.5f;
            }
    f.sdr_px(0, 0, 255, 255, 255);   // clamped, all channels
    f.sdr_px(1, 0, 255, 120, 90);    // clamped, red only
    f.sdr_px(2, 0, 0, 1, 0);         // crushed
    return f;
}

bool same(const PlanarBuffer& a, const PlanarBuffer& b) {
    if (a.plane_size() != b.plane_size()) return false;
    for (int k = 0; k < 3; ++k)
        for (std::size_t i = 0; i < a.plane_size(); ++i)
            if (a.plane(k)[i] != b.plane(k)[i]) return false;
    return true;
}

}  // namespace

TEST(ViewRev6, DefaultsAreRevisionFive) {
    // Every revision-6 field at its default is the revision-5 pass, bit for bit,
    // with or without the SDR given (the browser goldens hold the rest).
    Frame f = random_frame(1);
    for (ViewMode mode : {ViewMode::Image, ViewMode::FalseColour, ViewMode::Difference, ViewMode::Invented})
        for (double wipe : {-1.0, 0.4}) {
            ViewParams p = view_params(mode, 400.0, ViewSource::Model, wipe);
            p.show_changes = mode == ViewMode::Image;
            const auto with = render_view(f.model, f.base, p, &f.sdr);
            ViewParams q = p;
            q.view_ev = 0.0, q.view_gamma = 1.0, q.wipe_left = ViewSource::Baseline, q.wipe_right = ViewSource::Model;
            EXPECT_TRUE(same(with, render_view(f.model, f.base, q, &f.sdr)));
        }
}

TEST(ViewRev6, ExposureIsAGainBeforeThePeak) {
    Frame f = random_frame(2);
    for (double ev : {-3.0, -1.0, 2.0}) {
        ViewParams a = view_params(ViewMode::Image, 203.0);
        a.view_ev = ev;
        const ViewParams b = view_params(ViewMode::Image, 203.0 / std::exp2(ev));
        const auto va = render_view(f.model, f.base, a), vb = render_view(f.model, f.base, b);
        for (int k = 0; k < 3; ++k)
            for (std::size_t i = 0; i < va.plane_size(); ++i) EXPECT_NEAR(va.plane(k)[i], vb.plane(k)[i], 2e-6f);
    }
}

TEST(ViewRev6, GammaPivotsOnTheViewPeak) {
    Frame f;
    f.nits(f.model, 3, 3, 50.75f);   // a quarter of the 203-nit peak
    ViewParams p = view_params(ViewMode::Image, 203.0);
    p.view_gamma = 2.0;
    const auto v = render_view(f.model, f.base, p);
    EXPECT_NEAR(v.at(0, 3, 3), linear_to_srgb(0.5f), 1e-6f);
    f.nits(f.model, 4, 3, 203.0f);   // the peak itself does not move
    EXPECT_NEAR(render_view(f.model, f.base, p).at(0, 3, 4), 1.0f, 1e-6f);
}

TEST(ViewRev6, SdrSourceIsTheClampedPlateAsLight) {
    Frame f;
    f.sdr_px(5, 2, 255, 255, 255);
    f.nits(f.model, 5, 2, 1600.0f);   // what the reconstruction made of it
    ViewParams p = view_params(ViewMode::Image, 203.0, ViewSource::Sdr);
    p.view_ev = -2.0;
    const auto v = render_view(f.model, f.base, p, &f.sdr);
    EXPECT_NEAR(v.at(0, 2, 5), linear_to_srgb(0.25f), 1e-6f);   // flat white, stopped down: grey
    p.show = ViewSource::Model;
    EXPECT_FLOAT_EQ(render_view(f.model, f.base, p, &f.sdr).at(0, 2, 5), 1.0f);   // 1600 / 4 > 203: still bright
    // Without the SDR, Sdr falls back to the baseline.
    p.show = ViewSource::Sdr;
    const ViewParams base = view_params(ViewMode::Image, 203.0, ViewSource::Baseline);
    ViewParams bq = base;
    bq.view_ev = -2.0;
    EXPECT_TRUE(same(render_view(f.model, f.base, p, nullptr), render_view(f.model, f.base, bq, nullptr)));
}

TEST(ViewRev6, WipeTakesAnyTwoSources) {
    Frame f;
    for (int x = 0; x < 16; ++x) {
        f.sdr_px(x, 1, 255, 255, 255);
        f.nits(f.model, x, 1, 800.0f);
    }
    ViewParams p = view_params(ViewMode::Image, 1000.0, ViewSource::Model, 0.5, 0.0);
    p.wipe_left = ViewSource::Sdr;
    p.wipe_right = ViewSource::Model;
    const auto v = render_view(f.model, f.base, p, &f.sdr);
    EXPECT_NEAR(v.at(1, 1, 2), linear_to_srgb(0.203f), 1e-6f);    // SDR white at a 1000-nit peak
    EXPECT_NEAR(v.at(1, 1, 13), linear_to_srgb(0.8f), 1e-6f);      // the model's 800 nits
}

TEST(ViewRev6, ClampZebraMarksTheSourceAtAnyExposure) {
    Frame f;
    for (int x = 0; x < 16; ++x)
        for (int y = 0; y < 8; ++y) f.sdr_px(x, y, 128, 128, 128);
    for (int y = 0; y < 8; ++y) {
        f.sdr_px(0, y, 255, 255, 255);
        f.sdr_px(1, y, 255, 100, 100);
        f.sdr_px(2, y, 0, 0, 0);
    }
    for (double ev : {0.0, -4.0}) {
        ViewParams on = view_params(ViewMode::Image, 203.0);
        on.view_ev = ev;
        on.clamp_zebra = true;
        ViewParams off = on;
        off.clamp_zebra = false;
        const auto a = render_view(f.model, f.base, on, &f.sdr), b = render_view(f.model, f.base, off, &f.sdr);
        for (int y = 0; y < 8; ++y)
            for (int x = 0; x < 16; ++x) {
                const int d = (x + y) & 7;
                const bool striped = (x == 0 && d < 3) || (x == 1 && d < 1) || (x == 2 && d < 3);
                const bool differs = a.at(0, y, x) != b.at(0, y, x) || a.at(2, y, x) != b.at(2, y, x);
                EXPECT_EQ(differs, striped) << "x " << x << " y " << y << " ev " << ev;
            }
        EXPECT_NEAR(a.at(0, 0, 0), kZebraMix * kClampAllColour[0], 1e-6f);   // black picture under it
        EXPECT_NEAR(a.at(2, 0, 2), kZebraMix * kCrushColour[2], 1e-6f);
    }
}

TEST(ViewRev6, ScreenClipZebraFollowsTheExposure) {
    Frame f;
    for (int x = 0; x < 16; ++x)
        for (int y = 0; y < 8; ++y) f.nits(f.model, x, y, 1000.0f);
    ViewParams p = view_params(ViewMode::Image, 203.0);
    p.clip_zebra = true;
    ViewParams off = p;
    off.clip_zebra = false;
    EXPECT_FALSE(same(render_view(f.model, f.base, p), render_view(f.model, f.base, off)));
    p.view_ev = off.view_ev = -3.0;   // 125 nits on a 203 peak: nothing clips
    EXPECT_TRUE(same(render_view(f.model, f.base, p), render_view(f.model, f.base, off)));
}

TEST(ViewRev6, RangeLayerColoursStopsOverTheSdrWhite) {
    Frame f;
    const float nits[] = {0.5f, 50.0f, 203.0f, 300.0f, 600.0f, 1200.0f, 2000.0f, 4000.0f};
    for (int x = 0; x < 8; ++x) f.nits(f.model, x, 4, nits[x]);
    ViewParams p = view_params(ViewMode::Range, 203.0);
    const auto v = render_view(f.model, f.base, p, &f.sdr);
    for (int x = 0; x < 8; ++x) {
        const auto want = range_colour(std::log2(nits[x] / 203.0f));
        for (int k = 0; k < 3; ++k) EXPECT_FLOAT_EQ(v.at(k, 4, x), want[std::size_t(k)]) << x;
    }
    EXPECT_FLOAT_EQ(range_colour(-7.0f)[0], 0.08f);
    EXPECT_FLOAT_EQ(range_colour(std::log2(0.0f))[0], 0.08f);   // black
    EXPECT_FLOAT_EQ(range_colour(0.5f)[2], 0.85f);
    EXPECT_FLOAT_EQ(range_colour(9.0f)[0], 0.90f);
    // The source's clamp is hatched; the view exposure does not reach the layer.
    f.sdr_px(0, 0, 255, 255, 255);
    p.view_ev = -3.0;
    p.view_gamma = 2.2;
    const auto h = render_view(f.model, f.base, p, &f.sdr);
    EXPECT_GT(h.at(0, 0, 0), 0.5f);   // (0 + 0) & 7 == 0: hatched
    EXPECT_FLOAT_EQ(h.at(0, 4, 3), v.at(0, 4, 3));
}

TEST(ViewRev6, OverlaysIgnoreTheViewerControls) {
    Frame f = random_frame(3);
    for (ViewMode mode : {ViewMode::FalseColour, ViewMode::Difference, ViewMode::Invented, ViewMode::Range}) {
        ViewParams a = view_params(mode, 300.0);
        ViewParams b = a;
        b.view_ev = -3.0;
        b.view_gamma = 2.0;
        EXPECT_TRUE(same(render_view(f.model, f.base, a, &f.sdr), render_view(f.model, f.base, b, &f.sdr)))
            << int(mode);
    }
}

TEST(ViewRev6, HdrPathsTakeTheGainBeforeTheirClip) {
    Frame f;
    f.nits(f.model, 0, 0, 800.0f);
    ViewParams p = view_params(ViewMode::Image, 1000.0);
    p.target = DisplayTarget::edr(1000.0, Primaries::Rec709);
    p.view_ev = -1.0;
    EXPECT_NEAR(render_view(f.model, f.base, p).at(0, 0, 0), 400.0f / 203.0f, 1e-4f);
    p.view_ev = 1.0;
    EXPECT_NEAR(render_view(f.model, f.base, p).at(0, 0, 0), 1000.0f / 203.0f, 1e-4f);   // 1600 clipped at the peak
}

TEST(RangeStats, CountsTheClampAndTheHeadroomExactly) {
    Frame f;
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 16; ++x) {
            f.sdr_px(x, y, 128, 128, 128);
            f.nits(f.model, x, y, 50.0f);
            f.nits(f.base, x, y, 50.0f);
        }
    // 8 all-channel clamped pixels: 6 reconstructed to 812 nits (+2 st), 2 left as the inverse.
    for (int x = 0; x < 8; ++x) {
        f.sdr_px(x, 0, 255, 255, 255);
        f.nits(f.base, x, 0, 203.0f);
        f.nits(f.model, x, 0, x < 6 ? 812.0f : 200.0f);   // 200: within 0.1 stop of the inverse
    }
    f.sdr_px(8, 0, 254, 10, 10);   // red clamped at code 254
    f.nits(f.model, 8, 0, 2030.0f);
    f.nits(f.base, 8, 0, 203.0f);
    f.sdr_px(9, 0, 1, 1, 0);       // crushed
    const auto light = sdr_light_rgba(&f.sdr);
    const RangeStats s = range_stats(light, f.model, &f.base);
    EXPECT_EQ(s.pixels, 128u);
    EXPECT_EQ(s.clamped_pixels, 9u);
    EXPECT_DOUBLE_EQ(s.clamped_pct, 100.0 * 9 / 128);
    EXPECT_DOUBLE_EQ(s.clamped_all_pct, 100.0 * 8 / 128);
    EXPECT_DOUBLE_EQ(s.clamped_channel_pct[0], 100.0 * 9 / 128);
    EXPECT_DOUBLE_EQ(s.clamped_channel_pct[1], 100.0 * 8 / 128);
    EXPECT_DOUBLE_EQ(s.crushed_pct, 100.0 / 128);
    EXPECT_NEAR(s.peak_nits, 2030.0, 0.01);
    EXPECT_NEAR(s.peak_stops_over_white, std::log2(2030.0 / 203.0), 1e-4);
    EXPECT_DOUBLE_EQ(s.above_white_pct, 100.0 * 7 / 128);
    EXPECT_DOUBLE_EQ(s.above_1000_pct, 100.0 / 128);
    EXPECT_NEAR(s.clamped_median_stops, 2.0, 1e-4);
    ASSERT_TRUE(s.clamped_unchanged_pct.has_value());
    EXPECT_DOUBLE_EQ(*s.clamped_unchanged_pct, 100.0 * 2 / 9);
    EXPECT_GT(s.hdr_range_stops, 0.0);
    EXPECT_GT(s.sdr_range_stops, 0.0);
    // The learned route has no baseline: no "unchanged" share.
    EXPECT_FALSE(range_stats(light, f.model, nullptr).clamped_unchanged_pct.has_value());
}

TEST(RangeStats, ClampedShareIsTheCountOfCodesAt254) {
    // The acceptance check: equal to a direct count of max(R, G, B) >= 254.
    Frame f = random_frame(4);
    std::size_t count = 0;
    for (std::size_t i = 0; i < f.sdr.buffer().plane_size(); ++i) {
        int mx = 0;
        for (int k = 0; k < 3; ++k) mx = std::max(mx, int(std::lround(f.sdr.buffer().plane(k)[i] * 255.0f)));
        if (mx >= 254) ++count;
    }
    const RangeStats s = range_stats(sdr_light_rgba(&f.sdr), f.model, &f.base);
    EXPECT_EQ(s.clamped_pixels, count);
}

TEST(RangeStats, UnclippedFrameHasNoClampedArea) {
    Frame f;
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 16; ++x) f.sdr_px(x, y, 200, 180, 120);
    const RangeStats s = range_stats(sdr_light_rgba(&f.sdr), f.model, &f.base);
    EXPECT_EQ(s.clamped_pixels, 0u);
    EXPECT_DOUBLE_EQ(s.clamped_pct, 0.0);
    EXPECT_FALSE(s.clamped_unchanged_pct.has_value());
}

TEST(RangeStats, SdrProbeReadsCodesAndFlags) {
    Frame f;
    f.sdr_px(3, 2, 255, 254, 17);
    const auto light = sdr_light_rgba(&f.sdr);
    const SdrProbe p = sdr_probe(f.sdr, light, 3.4, 1.6);
    EXPECT_EQ(p.x, 3);
    EXPECT_EQ(p.y, 2);
    EXPECT_EQ(p.code[0], 255);
    EXPECT_EQ(p.code[1], 254);
    EXPECT_EQ(p.code[2], 17);
    EXPECT_TRUE(p.clamped[0] && p.clamped[1]);
    EXPECT_FALSE(p.clamped[2]);
    EXPECT_FALSE(p.crushed);
    EXPECT_NEAR(p.nits[0], 203.0, 1e-3);
}

TEST(RangeStats, SelectionPercentilesEqualTheSortedOnes) {
    // range.cpp selects instead of sorting; the numbers must be numpy's.
    Frame f = random_frame(5);
    for (int x = 0; x < 6; ++x) {
        f.sdr_px(x, 3, 255, 255, 255);
        f.nits(f.model, x, 3, 100.0f * float(x + 1) * float(x + 1));
    }
    const RangeStats s = range_stats(sdr_light_rgba(&f.sdr), f.model, &f.base);
    std::vector<double> in;
    const auto light = sdr_light_rgba(&f.sdr);
    for (std::size_t i = 0; i < f.model.buffer().plane_size(); ++i)
        if (int(light[i * 4 + 3]) & 7) {
            const double m = std::max({double(f.model.buffer().plane(0)[i]), double(f.model.buffer().plane(1)[i]),
                                       double(f.model.buffer().plane(2)[i])}) * 10000.0;
            in.push_back(m > 0.0 ? std::log2(m / 203.0) : -60.0);
        }
    std::vector<double> a = in, b = in;
    EXPECT_NEAR(s.clamped_median_stops, percentile_linear(a, 50.0), 1e-12);
    EXPECT_NEAR(s.clamped_p99_stops, percentile_linear(b, 99.0), 1e-12);
}

// ---- the session's viewer controls (engine/session.hpp) ----
#include "rudra/engine/session.hpp"

TEST(ViewerSession, ExposureIsViewStateNotTheGrade) {
    Session s;
    std::uint32_t seen = 0;
    s.on_change([&](std::uint32_t w) { seen |= w; });
    const std::string params = s.params_json();
    const std::size_t undo = s.undo_depth();
    s.set_view_ev(-3.0);
    s.set_view_gamma(2.2);
    s.toggle_clamp_zebra();
    s.toggle_clip_zebra();
    s.set_wipe_sources("sdr", "model");
    EXPECT_EQ(s.params_json(), params);    // never in a render's or a master's settings
    EXPECT_EQ(s.undo_depth(), undo);       // never undone
    EXPECT_EQ(seen & Session::Grade, 0u);  // never a grade change
    EXPECT_NE(seen & Session::View, 0u);
    EXPECT_DOUBLE_EQ(s.grade.exposure_ev, 0.0);   // the master exposure is another control
}

TEST(ViewerSession, FStopToggleAndLimits) {
    Session s;
    s.set_view_ev(-12.0);
    EXPECT_DOUBLE_EQ(s.view_ev, -10.0);
    s.set_view_ev(-2.5);
    s.toggle_view_ev();
    EXPECT_DOUBLE_EQ(s.view_ev, 0.0);
    s.toggle_view_ev();
    EXPECT_DOUBLE_EQ(s.view_ev, -2.5);
    s.nudge_view_ev(0.5);
    EXPECT_DOUBLE_EQ(s.view_ev, -2.0);
    s.set_view_gamma(9.0);
    EXPECT_DOUBLE_EQ(s.view_gamma, 5.0);
    s.reset_viewer();
    EXPECT_TRUE(s.viewer_neutral());
    s.toggle_view_ev();   // back to the last value that was not 0
    EXPECT_DOUBLE_EQ(s.view_ev, -2.0);
}

TEST(ViewerSession, WipeSourcesAndFlip) {
    Session s;
    s.set_wipe_sources("sdr", "model");
    ASSERT_TRUE(s.wipe.has_value());
    EXPECT_EQ(s.wipe_left, "sdr");
    s.flip_ab();
    EXPECT_EQ(s.wipe_left, "model");
    EXPECT_EQ(s.wipe_right, "sdr");
    s.set_wipe_sources("nonsense", "model");   // refused, nothing changes
    EXPECT_EQ(s.wipe_left, "model");
    s.toggle_wipe();                            // off: flip swaps the shown source
    s.set_show("model");
    s.flip_ab();
    EXPECT_EQ(s.show, "model" == s.wipe_right ? s.wipe_left : s.wipe_right);
}

TEST(ViewRev6, SdrSourceReadsTheSourceCurve) {
    // With the Source picker's knots the Sdr source is that curve's light (here
    // BT.1886, gamma 2.4 at diffuse white), not the sRGB reading the app left
    // the viewer with before 1 Nov 2026.
    CompositeParams bt;
    bt.source = SourceCurve::Bt1886;
    const std::vector<float> knots = anchor_target_knots(bt);
    ASSERT_FALSE(knots.empty());
    SdrImage sdr(1, 3);
    const float codes[3] = {0.25f, 0.5f, 0.9f};
    for (int x = 0; x < 3; ++x)
        for (int k = 0; k < 3; ++k) sdr.buffer().at(k, 0, x) = codes[x];
    const auto plain = sdr_light_rgba(&sdr);
    const auto read = sdr_light_rgba(&sdr, knots);
    // The masters' own BT.1886 reading of the same codes: the composite's baseline
    // at 0 EV (what anchor_target_knots samples), so the viewer and the master agree.
    CompositeParams curve_only;
    curve_only.source = SourceCurve::Bt1886;
    const NetworkLinearImage base =
        corrected_baseline(SdrImage(sdr.buffer()), 0.0f, baseline_curve_params(FrameScalars{}, curve_only, 0.0f));
    for (int x = 0; x < 3; ++x) {
        const double want = base.buffer().at(1, 0, x);
        EXPECT_NEAR(read[std::size_t(x) * 4 + 1], want, want * 0.01) << codes[x];   // knot interpolation only
        EXPECT_NE(read[std::size_t(x) * 4], plain[std::size_t(x) * 4]);
    }
    EXPECT_NEAR(read[2 * 4 + 3], plain[2 * 4 + 3], 0.0f);   // the flags do not depend on the reading
}
