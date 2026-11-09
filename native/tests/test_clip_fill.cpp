// The clip fill (core/clip_fill.hpp): what the analytic reconstruction puts
// where the SDR clipped. Synthetic scenes rendered through the corpus's own
// forward curve (exposure, ACES Narkowicz, sRGB), so the truth is known.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include "rudra/core/baseline.hpp"
#include "rudra/core/clip_fill.hpp"
#include "rudra/core/composite.hpp"

using namespace rudra;

namespace {

constexpr float kEv = kAnalyticConstants.corpus_ev;

float scale() { return float(std::exp2(-double(kEv)) * (203.0 / 10000.0)); }

// The forward render baseline_value inverts: network linear -> SDR code.
float forward_code(float network) {
    const float x = std::max(network / scale(), 0.0f);
    const float a = (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);
    const float y = std::clamp(a, 0.0f, 1.0f);
    const float c = y <= 0.0031308f ? y * 12.92f : 1.055f * std::pow(y, 1.0f / 2.4f) - 0.055f;
    return std::round(std::clamp(c, 0.0f, 1.0f) * 255.0f) / 255.0f;   // 8-bit, as delivered
}

SdrImage render(const PlanarBuffer& truth) {
    PlanarBuffer s(3, truth.height(), truth.width());
    for (std::size_t i = 0; i < truth.span().size(); ++i) s.span()[i] = forward_code(truth.span()[i]);
    return SdrImage(std::move(s));
}

float nits(float network) { return network * 10000.0f; }
float luma(const PlanarBuffer& b, std::size_t i) {
    return 0.2126f * b.plane(0)[i] + 0.7152f * b.plane(1)[i] + 0.0722f * b.plane(2)[i];
}

// Applies a fill to the analytic baseline.
PlanarBuffer filled(const SdrImage& sdr, const ClipFill& f) {
    PlanarBuffer out = analytic_baseline(sdr, kEv).buffer();
    if (f.gain_log2.empty()) return out;
    for (std::size_t i = 0; i < out.span().size(); ++i) out.span()[i] *= std::exp2(f.gain_log2.span()[i]);
    return out;
}

// A sky: a vertical gradient, blue over the top where only blue clips, and a
// sun disc in it that clips every channel.
PlanarBuffer sky(int h, int w, bool sun) {
    PlanarBuffer t(3, h, w);
    const float white = 203.0f / 10000.0f;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float k = 1.0f - float(y) / float(h - 1);          // 1 at the top
            const float lum = white * (1.0f + 3.0f * k);              // 203 .. 812 nits
            t.at(0, y, x) = lum * 0.55f;
            t.at(1, y, x) = lum * 0.85f;
            t.at(2, y, x) = lum * 2.4f;
            if (sun) {
                const float d = std::hypot(float(x) - float(w) * 0.5f, float(y) - float(h) * 0.3f);
                const float glow = 60.0f * white * std::exp(-d * d / 60.0f);   // a peak of ~12 000 nits
                for (int c = 0; c < 3; ++c) t.at(c, y, x) += glow;
            }
        }
    return t;
}

}  // namespace

TEST(ClipFill, NothingClippedIsTheAnalyticReconstruction) {
    PlanarBuffer s(3, 16, 16);
    for (std::size_t i = 0; i < s.span().size(); ++i) s.span()[i] = 0.9f * float(i % 97) / 96.0f;
    const SdrImage sdr(std::move(s));
    ClipFill detail;
    const Fields f = clip_fill_fields(sdr, kAnalyticConstants, {}, &detail);
    EXPECT_TRUE(detail.gain_log2.empty());
    for (float v : f.residual.span()) ASSERT_EQ(v, 0.0f);
    for (float v : f.highlight.span()) ASSERT_EQ(v, 0.0f);
}

TEST(ClipFill, DisabledIsTheAnalyticReconstruction) {
    const SdrImage sdr = render(sky(48, 64, true));
    ClipFillOptions off;
    off.enabled = false;
    const Fields f = clip_fill_fields(sdr, kAnalyticConstants, off);
    for (float v : f.residual.span()) ASSERT_EQ(v, 0.0f);
}

TEST(ClipFill, FillMissingIsExactOnAPlane) {
    const int h = 40, w = 50;
    std::vector<std::vector<float>> p(1, std::vector<float>(std::size_t(h * w)));
    std::vector<float> wt(p[0].size(), 1.0f);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const std::size_t i = std::size_t(y * w + x);
            p[0][i] = 0.3f * float(x) - 0.2f * float(y) + 1.0f;
            if (std::hypot(x - 25.0, y - 20.0) < 12.0) wt[i] = 0.0f, p[0][i] = 99.0f;
        }
    fill_missing(p, wt, h, w, 400);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            ASSERT_NEAR(p[0][std::size_t(y * w + x)], 0.3f * float(x) - 0.2f * float(y) + 1.0f, 0.05f) << x << "," << y;
}

// Stage 1: blue clipped, red and green kept. The fill puts blue back near the
// truth (the sky's ratio, carried from below), never under the ceiling.
TEST(ClipFill, APartlyClippedSkyGetsItsBlueBack) {
    const int h = 64, w = 48;
    const PlanarBuffer truth = sky(h, w, false);
    const SdrImage sdr = render(truth);
    const ClipFill f = reconstruct_clipped(sdr, kEv);
    ASSERT_GT(f.partial_pct, 10.0);
    EXPECT_EQ(f.full_pct, 0.0);
    const PlanarBuffer base = analytic_baseline(sdr, kEv).buffer();
    const PlanarBuffer out = filled(sdr, f);
    double err_flat = 0.0, err_fill = 0.0;
    int count = 0;
    for (std::size_t i = 0; i < base.plane_size(); ++i) {
        if (sdr.buffer().plane(2)[i] < 0.999f || sdr.buffer().plane(1)[i] > 0.99f) continue;   // blue alone clipped
        ASSERT_GE(out.plane(2)[i], base.plane(2)[i]);
        err_flat += std::abs(std::log2(base.plane(2)[i] / truth.plane(2)[i]));
        err_fill += std::abs(std::log2(out.plane(2)[i] / truth.plane(2)[i]));
        ++count;
    }
    ASSERT_GT(count, 100);
    err_flat /= count;
    err_fill /= count;
    std::cout << "[ partial ] mean |EV error| in blue: flat " << err_flat << ", filled " << err_fill << "\n";
    EXPECT_LT(err_fill, err_flat * 0.35);
    EXPECT_LT(err_fill, 0.3);
}

// Stage 3: the sun clips every channel. The disc gets a peak over its edge,
// at least source_min_ev, at most source_max_nits, and the falloff is smooth.
TEST(ClipFill, ASunGetsAPeakAndAFalloff) {
    const int h = 96, w = 128;
    const SdrImage sdr = render(sky(h, w, true));
    ClipFillOptions o;
    o.large_fraction = 0.1;   // a 128 x 96 frame: the disc is a few percent of it
    o.source_min_ev = 1.0;    // the floor the test holds the peak to
    const ClipFill f = reconstruct_clipped(sdr, kEv, o);
    ASSERT_GT(f.full_pct, 0.0);
    const auto sun = std::max_element(f.regions.begin(), f.regions.end(),
                                      [](const ClipRegion& a, const ClipRegion& b) { return a.pixels < b.pixels; });
    ASSERT_NE(sun, f.regions.end());
    EXPECT_FALSE(sun->large);
    EXPECT_GE(sun->peak_nits, 1276.0 * std::exp2(o.source_min_ev) * 0.95);   // over the ceiling
    EXPECT_LE(sun->peak_nits, o.source_max_nits * 1.05);
    EXPECT_GT(sun->glow_ev_per_px, 0.0);
    std::cout << "[ sun ] " << sun->pixels << " px, radius " << sun->radius_px << ", edge " << sun->edge_nits
              << " nits, peak " << sun->peak_nits << " nits, glow " << sun->glow_ev_per_px << " EV/px\n";
    // The peak sits at the disc's centre and falls off toward the edge.
    const PlanarBuffer out = filled(sdr, f);
    const std::size_t centre = std::size_t(sun->y) * std::size_t(w) + std::size_t(sun->x);
    EXPECT_NEAR(sun->x, w / 2, 2);
    EXPECT_NEAR(sun->y, int(h * 0.3f), 2);
    float prev = nits(luma(out, centre));
    for (int dx = 1; dx < 6; ++dx) {
        const float v = nits(luma(out, centre + std::size_t(dx)));
        EXPECT_LE(v, prev * 1.001f) << dx;
        prev = v;
    }
}

// 31 Oct 2026: a blown sun counted as a large region (a wide, low-contrast
// shot) carried the blue sky's hue all the way in and came out cyan. Its core
// now runs toward white like a source's: less colourful than its edge.
TEST(ClipFill, ALargeRegionsCoreRunsTowardWhite) {
    const int h = 96, w = 128;
    const SdrImage sdr = render(sky(h, w, true));
    ClipFillOptions o;
    o.large_fraction = 0.001;   // the sun is a large region here
    const ClipFill f = reconstruct_clipped(sdr, kEv, o);
    const auto sun = std::max_element(f.regions.begin(), f.regions.end(),
                                      [](const ClipRegion& a, const ClipRegion& b) { return a.pixels < b.pixels; });
    ASSERT_NE(sun, f.regions.end());
    ASSERT_TRUE(sun->large);
    const PlanarBuffer out = filled(sdr, f);
    auto spread = [&](int x, int y) {   // log2 of the brightest channel over the dimmest
        const std::size_t i = std::size_t(y) * std::size_t(w) + std::size_t(x);
        const float r = out.plane(0)[i], g = out.plane(1)[i], b = out.plane(2)[i];
        return std::log2(std::max({r, g, b}) / std::min({r, g, b}));
    };
    int edge = sun->x;   // the last fully clipped pixel to the right of the deepest point
    while (edge + 1 < w && sdr.buffer().plane(0)[std::size_t(sun->y) * std::size_t(w) + std::size_t(edge + 1)] >= 0.995f)
        ++edge;
    ASSERT_GT(edge, sun->x);
    EXPECT_LT(spread(sun->x, sun->y), 0.75f * spread(edge, sun->y));
}

// Stage 2: a blown band across the top of the frame is a large region: its
// dome is gentle and capped at large_max_ev.
TEST(ClipFill, ALargeBlownSkyIsCappedAndSmooth) {
    const int h = 80, w = 80;
    PlanarBuffer truth(3, h, w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) {
                const float k = 1.0f - float(y) / float(h - 1);   // a sky brightening to the top
                truth.at(c, y, x) = (1.0f + 14.0f * k * k) * (c == 2 ? 1.1f : 1.0f) * 203.0f / 10000.0f;
            }
    const SdrImage sdr = render(truth);
    ClipFillOptions o;
    const ClipFill f = reconstruct_clipped(sdr, kEv, o);
    ASSERT_EQ(f.regions.size(), 1u);
    const ClipRegion& r = f.regions[0];
    EXPECT_TRUE(r.large);
    EXPECT_LE(std::log2(r.peak_nits / r.edge_nits), o.large_max_ev + 1e-6);
    // Never below what clipped, everywhere.
    const PlanarBuffer base = analytic_baseline(sdr, kEv).buffer();
    const PlanarBuffer out = filled(sdr, f);
    for (std::size_t i = 0; i < out.span().size(); ++i) ASSERT_GE(out.span()[i], base.span()[i]);
    // Smooth: no step between neighbours more than a tenth of a stop beyond
    // the one the baseline itself takes there (8-bit codes near white are
    // far apart after the inverse; that step is the SDR's, not the fill's).
    for (int y = 1; y < h; ++y)
        for (int x = 1; x < w; ++x) {
            const std::size_t i = std::size_t(y * w + x);
            for (const std::size_t j : {i - std::size_t(w), i - 1}) {
                const float step = std::abs(std::log2(luma(out, i) / luma(out, j)));
                const float own = std::abs(std::log2(luma(base, i) / luma(base, j)));
                ASSERT_LT(step, own + 0.1f) << x << "," << y;
            }
        }
}

// Step 4: a click on a region sets its peak in nits.
TEST(ClipFill, AnOverrideSetsTheRegionsPeak) {
    const int h = 96, w = 128;
    const SdrImage sdr = render(sky(h, w, true));
    ClipFillOptions o;
    o.overrides.push_back({0.5, 0.3, 2500.0});
    const ClipFill f = reconstruct_clipped(sdr, kEv, o);
    const auto sun = std::find_if(f.regions.begin(), f.regions.end(), [](const ClipRegion& r) { return r.overridden; });
    ASSERT_NE(sun, f.regions.end());
    EXPECT_NEAR(sun->peak_nits, 2500.0, 2500.0 * 0.08);
    // A click on nothing clipped changes nothing.
    ClipFillOptions miss;
    miss.overrides.push_back({0.02, 0.98, 9000.0});
    const ClipFill g = reconstruct_clipped(sdr, kEv, miss);
    for (const auto& r : g.regions) EXPECT_FALSE(r.overridden);
}

// The fields carry the fill through composite() exactly (to fp32) under the
// defaults; Off is the baseline, and Strength scales it in log.
TEST(ClipFill, TheCompositeAppliesTheFill) {
    const int h = 96, w = 128;
    const SdrImage sdr = render(sky(h, w, true));
    ClipFill detail;
    const Fields fields = clip_fill_fields(sdr, kAnalyticConstants, {}, &detail);
    const PlanarBuffer want = filled(sdr, detail);
    const PlanarBuffer base = analytic_baseline(sdr, kEv).buffer();
    CompositeParams p;
    const PlanarBuffer got = composite(sdr, fields, FrameScalars{}, kAnalyticConstants, p).buffer();
    for (std::size_t i = 0; i < got.span().size(); ++i)
        ASSERT_NEAR(got.span()[i], want.span()[i], 1e-4f * std::max(1.0f, want.span()[i])) << i;
    p.mode = RecoveryMode::Off;
    const PlanarBuffer off = composite(sdr, fields, FrameScalars{}, kAnalyticConstants, p).buffer();
    for (std::size_t i = 0; i < off.span().size(); ++i) ASSERT_NEAR(off.span()[i], base.span()[i], 1e-6f * std::max(1.0f, base.span()[i])) << i;
    p.mode = RecoveryMode::All;
    p.strength = 0.5f;
    const PlanarBuffer half = composite(sdr, fields, FrameScalars{}, kAnalyticConstants, p).buffer();
    for (std::size_t i = 0; i < half.span().size(); ++i) {
        ASSERT_GE(half.span()[i], base.span()[i] * 0.9999f);
        ASSERT_LE(half.span()[i], want.span()[i] * 1.0001f);
    }
}

// ---- stage 4 (9 Nov 2026): the near-clip detail -----------------------------

namespace {
// A bright sky with a faint structure parked just under white: a 203..450-nit
// gradient (codes about 225..248 through the corpus curve) with a 10 % sinusoidal
// ripple, which the shoulder flattens to two or three codes.
PlanarBuffer shoulder_sky(int h, int w) {
    PlanarBuffer t(3, h, w);
    const float white = 203.0f / 10000.0f;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float k = 1.0f - float(y) / float(h - 1);
            const float ripple = 1.0f + 0.10f * std::sin(float(x) * 0.5f);
            const float lum = white * (1.0f + 1.2f * k) * ripple;
            t.at(0, y, x) = lum * 0.9f;
            t.at(1, y, x) = lum;
            t.at(2, y, x) = lum * 1.1f;
        }
    return t;
}
}  // namespace

TEST(ClipFillDetail, OffIsBitIdenticalAndOnAmplifiesTheStructureHueHeld) {
    const int h = 64, w = 160;
    const SdrImage sdr = render(shoulder_sky(h, w));
    ClipFillOptions off;   // detail_gain 1
    EXPECT_TRUE(near_clip_detail(sdr, off).empty());
    ClipFill d_off, d_on;
    const Fields f_off = clip_fill_fields(sdr, kAnalyticConstants, off, &d_off);
    ClipFillOptions on;
    on.detail_gain = 2.0;
    const Fields f_on = clip_fill_fields(sdr, kAnalyticConstants, on, &d_on);
    EXPECT_EQ(d_off.detail_pct, 0.0);
    EXPECT_GT(d_on.detail_pct, 5.0);   // the bright band moved
    const PlanarBuffer g = near_clip_detail(sdr, on);
    ASSERT_EQ(g.channels(), 1);
    // Signed, within the cap, and only where the codes are bright.
    float lo = 0.0f, hi = 0.0f;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float v = g.at(0, y, x);
            lo = std::min(lo, v);
            hi = std::max(hi, v);
            const float yl = luma(sdr.buffer(), std::size_t(y * w + x));
            if (yl < float(on.detail_from)) ASSERT_EQ(v, 0.0f) << x << "," << y;
            ASSERT_LE(std::abs(v), float(on.detail_max_stops));
        }
    EXPECT_LT(lo, -0.01f);
    EXPECT_GT(hi, 0.01f);
    // Through the composite: every pixel moves by the stage's gain exactly (to fp32), the
    // channel ratios stay the baseline's (hue held), and the ripple's contrast grows.
    CompositeParams p;
    const PlanarBuffer a = composite(sdr, f_off, FrameScalars{}, kAnalyticConstants, p).buffer();
    const PlanarBuffer b = composite(sdr, f_on, FrameScalars{}, kAnalyticConstants, p).buffer();
    for (std::size_t i = 0; i < a.plane_size(); ++i) {
        const float ya = luma(a, i), yb = luma(b, i);
        ASSERT_GT(ya, 0.0f);
        ASSERT_NEAR(std::log2(yb / ya), g.span()[i], 2e-3f) << i;
        for (int c = 0; c < 3; ++c)
            ASSERT_NEAR(b.plane(c)[i] / yb, a.plane(c)[i] / ya, 2e-3f) << i << " channel " << c;
    }
    // The gain follows the ripple (brighter where the scene was), not the gradient or noise.
    double sxy = 0.0, sxx = 0.0, syy = 0.0;
    for (int y = 0; y < h; ++y)
        for (int x = 20; x < w - 20; ++x) {
            const float v = g.at(0, y, x);
            if (v == 0.0f) continue;
            const double r = std::sin(double(x) * 0.5);
            sxy += v * r;
            sxx += double(v) * v;
            syy += r * r;
        }
    ASSERT_GT(sxx, 0.0);
    EXPECT_GT(sxy / std::sqrt(sxx * syy), 0.7);
}

TEST(ClipFillDetail, TheNoiseFloorLeavesOneCodeOfDitherAlone) {
    // A flat bright field with one code of dither: nothing to amplify.
    const int h = 32, w = 64;
    PlanarBuffer s(3, h, w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) s.at(c, y, x) = ((x + y) & 1 ? 249.0f : 250.0f) / 255.0f;
    const SdrImage sdr(std::move(s));
    ClipFillOptions on;
    on.detail_gain = 2.0;
    const PlanarBuffer g = near_clip_detail(sdr, on);
    ASSERT_FALSE(g.empty());
    for (float v : g.span()) ASSERT_EQ(v, 0.0f);
}

// ---- the settings (12 Oct 2026): request JSON, session, project ------------

#include "rudra/deliver/master.hpp"
#include "rudra/engine/project.hpp"
#include "rudra/engine/session.hpp"

TEST(ClipFillSettings, TheRequestJsonCarriesThemOnlyWhenNotTheDefault) {
    MasterRequest q;
    EXPECT_EQ(master_request_json(q).find("clip_"), std::string::npos);   // masters before 12 Oct read the same
    q.clip_fill = false;
    q.clip_peaks.push_back({0.25, 0.75, 2500.0});
    auto back = master_request_from_json(master_request_json(q));
    ASSERT_TRUE(back);
    EXPECT_FALSE(back->clip_fill);
    ASSERT_EQ(back->clip_peaks.size(), 1u);
    EXPECT_EQ(back->clip_peaks[0].x, 0.25);
    EXPECT_EQ(back->clip_peaks[0].y, 0.75);
    EXPECT_EQ(back->clip_peaks[0].nits, 2500.0);
    EXPECT_TRUE(master_request_from_json("{}")->clip_fill);
    // 9 Nov 2026: the near-clip detail gain, in the JSON only when not 1.
    MasterRequest d;
    EXPECT_EQ(master_request_json(d).find("highlight_detail"), std::string::npos);
    d.highlight_detail = 1.8;
    auto dback = master_request_from_json(master_request_json(d));
    ASSERT_TRUE(dback);
    EXPECT_EQ(dback->highlight_detail, 1.8);
    EXPECT_EQ(clip_fill_options(*dback).detail_gain, 1.8);
    EXPECT_EQ(master_request_from_json("{}")->highlight_detail, 1.0);
    EXPECT_FALSE(master_request_from_json(R"({"highlight_detail": 9})"));
}

TEST(ClipFillSettings, TheDetailGainIsUndoneWithTheGradeAndKeptByAProject) {
    Session s;
    EXPECT_EQ(s.grade.highlight_detail, 1.0);
    EXPECT_EQ(s.params_json().find("highlight_detail"), std::string::npos);
    s.highlight_detail_press();
    s.highlight_detail_input(1.75);
    EXPECT_EQ(s.grade.highlight_detail, 1.75);
    EXPECT_NE(s.params_json().find("\"highlight_detail\":1.75"), std::string::npos) << s.params_json();
    s.highlight_detail_input(7.0);   // clamped to the range
    EXPECT_EQ(s.grade.highlight_detail, kHighlightDetailMax);
    s.undo();
    EXPECT_EQ(s.grade.highlight_detail, 1.0);
    Project p;
    p.grade.highlight_detail = 2.0;
    const auto file = std::filesystem::temp_directory_path() / "detail.rudra";
    auto back = parse_project(project_json(p, file), file);
    ASSERT_TRUE(back) << back.error().message;
    EXPECT_EQ(back->grade.highlight_detail, 2.0);
    auto old = parse_project(R"({"rudra_project": 1, "grade": {"mode": "all"}})", file);
    ASSERT_TRUE(old);
    EXPECT_EQ(old->grade.highlight_detail, 1.0);
}

TEST(ClipFillSettings, AnalyticFieldsForFollowsTheRequest) {
    const SdrImage sdr = render(sky(48, 64, true));
    MasterRequest q;
    const Fields on = analytic_fields_for(sdr, q);
    q.clip_fill = false;
    const Fields off = analytic_fields_for(sdr, q);
    float most = 0.0f;
    for (float v : on.highlight.span()) most = std::max(most, v);
    EXPECT_EQ(most, 1.0f);
    for (float v : off.residual.span()) ASSERT_EQ(v, 0.0f);
}

TEST(ClipFillSettings, ThePeaksAreUndoneWithTheGrade) {
    Session s;
    EXPECT_TRUE(s.grade.clip_fill);
    s.set_clip_peak(0.5, 0.5, 3000.0);
    s.set_clip_peak(0.505, 0.5, 1500.0);   // within 2 %: replaces it
    ASSERT_EQ(s.grade.clip_peaks.size(), 1u);
    EXPECT_EQ(s.grade.clip_peaks[0].nits, 1500.0);
    s.set_clip_peak(0.2, 0.2, 800.0);
    EXPECT_EQ(s.grade.clip_peaks.size(), 2u);
    s.set_clip_peak(0.2, 0.2, 0.0);        // 0 removes the one there
    EXPECT_EQ(s.grade.clip_peaks.size(), 1u);
    s.toggle_clip_fill();
    EXPECT_FALSE(s.grade.clip_fill);
    s.undo();
    EXPECT_TRUE(s.grade.clip_fill);
    s.clear_clip_peaks();
    EXPECT_TRUE(s.grade.clip_peaks.empty());
    s.undo();
    EXPECT_EQ(s.grade.clip_peaks.size(), 1u);
}

TEST(ClipFillSettings, AProjectKeepsThem) {
    Project p;
    p.grade.clip_fill = false;
    p.grade.clip_peaks.push_back({0.1, 0.9, 4000.0});
    const auto file = std::filesystem::temp_directory_path() / "clip.rudra";
    auto back = parse_project(project_json(p, file), file);
    ASSERT_TRUE(back) << back.error().message;
    EXPECT_FALSE(back->grade.clip_fill);
    ASSERT_EQ(back->grade.clip_peaks.size(), 1u);
    EXPECT_EQ(back->grade.clip_peaks[0].nits, 4000.0);
    // A project from before 12 Oct 2026: the fill on, no peaks.
    auto old = parse_project(R"({"rudra_project": 1, "grade": {"mode": "all"}})", file);
    ASSERT_TRUE(old);
    EXPECT_TRUE(old->grade.clip_fill);
    EXPECT_TRUE(old->grade.clip_peaks.empty());
}
