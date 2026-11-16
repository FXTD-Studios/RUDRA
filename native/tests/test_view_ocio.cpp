// docs/view.spec.md revision 7 (17 Nov 2026): the colour-managed viewer. The CPU display
// pass (render_view, what display.frag is held to) with an OCIO view, the working spaces,
// the original through an input transform and the range checks, on stand-in stages; and,
// with OpenColorIO in the build, the built-in ACES 2.0 config's own views.
#include <gtest/gtest.h>

#include <cmath>
#include <set>

#include "rudra/core/baseline.hpp"
#include "rudra/core/compare.hpp"
#include "rudra/core/gamut.hpp"
#include "rudra/core/hdr10.hpp"
#include "rudra/core/view.hpp"
#include "rudra/core/view_ocio.hpp"
#ifdef RUDRA_HAVE_OCIO
#include "rudra/infer/ocio_viewer.hpp"
#endif

using namespace rudra;

namespace {

constexpr float kP = 10000.0f;

// A stand-in view: code = w / (1 + w) per channel of the working-space value; its decode
// gives the codes back as XYZ of Rec.709 light, so the HDR path shows code x nits per Y.
std::shared_ptr<OcioViewTransform> fake_view(bool decodes = true, double nits_per_y = 100.0) {
    auto v = std::make_shared<OcioViewTransform>();
    v->id = "fake";
    v->display = "Fake display";
    v->view = "Fake view";
    v->forward.cpu = [](float* rgb, std::size_t n) {
        for (std::size_t i = 0; i < n * 3; ++i) rgb[i] = rgb[i] / (1.0f + std::abs(rgb[i]));
    };
    v->decodes = decodes;
    const Mat3 m = normalized_primary_matrix(Primaries::Rec709);
    v->decode.cpu = [m](float* rgb, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            const double r = rgb[i * 3], g = rgb[i * 3 + 1], b = rgb[i * 3 + 2];
            for (int k = 0; k < 3; ++k) rgb[i * 3 + std::size_t(k)] = float(m[std::size_t(k)][0] * r + m[std::size_t(k)][1] * g + m[std::size_t(k)][2] * b);
        }
    };
    v->nits_per_y = nits_per_y;
    v->hdr = nits_per_y == 100.0;
    return v;
}

NetworkLinearImage flat(int w, int h, float r, float g, float b) {
    NetworkLinearImage im(h, w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            im.buffer().at(0, y, x) = r;
            im.buffer().at(1, y, x) = g;
            im.buffer().at(2, y, x) = b;
        }
    return im;
}

// What the stand-in view makes of a Rec.709 pixel at view exposure `ev`.
std::array<float, 3> expected_code(const float h[3], double ev = 0.0, WorkingSpace w = WorkingSpace::AcesCg) {
    const ViewMatrices m = view_matrices(Primaries::Rec709, w, Primaries::Ap1, 100.0);
    const float g = float(std::exp2(ev));
    const float hv[3] = {h[0] * g, h[1] * g, h[2] * g};
    std::array<float, 3> out{};
    for (int k = 0; k < 3; ++k) {
        const float v = m.work[std::size_t(k) * 3] * hv[0] + m.work[std::size_t(k) * 3 + 1] * hv[1] + m.work[std::size_t(k) * 3 + 2] * hv[2];
        out[std::size_t(k)] = std::clamp(v / (1.0f + std::abs(v)), 0.0f, 1.0f);
    }
    return out;
}

}  // namespace

TEST(ViewOcio, TheViewRunsOnTheWorkingSpaceAtTheViewExposure) {
    const float h[3] = {0.0203f, 0.01f, 0.004f};   // 203 nits red-ish, Rec.709
    const NetworkLinearImage im = flat(4, 2, h[0], h[1], h[2]);
    ViewParams p = view_params(ViewMode::Image, 203.0);
    p.ocio = fake_view();
    for (double ev : {0.0, 1.5, -2.0}) {
        p.view_ev = ev;
        const PlanarBuffer out = render_view(im, im, p);
        const auto want = expected_code(h, ev);
        for (int k = 0; k < 3; ++k) EXPECT_NEAR(out.at(k, 1, 2), want[std::size_t(k)], 1e-6) << "ev " << ev << " channel " << k;
    }
}

TEST(ViewOcio, TheViewGammaIsOnTheCodesAsInNuke) {
    const float h[3] = {0.004f, 0.004f, 0.004f};
    const NetworkLinearImage im = flat(2, 2, h[0], h[1], h[2]);
    ViewParams p = view_params(ViewMode::Image, 203.0);
    p.ocio = fake_view();
    p.view_gamma = 2.0;
    const PlanarBuffer out = render_view(im, im, p);
    const auto code = expected_code(h);
    EXPECT_NEAR(out.at(0, 0, 0), std::sqrt(code[0]), 1e-5);
}

TEST(ViewOcio, AnHdrSwapchainShowsTheViewsLight) {
    const float h[3] = {0.0203f, 0.0203f, 0.0203f};
    const NetworkLinearImage im = flat(2, 2, h[0], h[1], h[2]);
    ViewParams p = view_params(ViewMode::Image, 203.0);
    p.target = DisplayTarget::scrgb(1000.0);
    p.ocio = fake_view(true, 100.0);
    const PlanarBuffer out = render_view(im, im, p);
    const auto code = expected_code(h);
    // White stays white through XYZ and back: code x 100 nits, written at 80 nits per 1.0.
    for (int k = 0; k < 3; ++k) EXPECT_NEAR(out.at(k, 0, 0), code[std::size_t(k)] * 100.0f / 80.0f, 2e-4) << k;
    // An SDR display's view puts its peak at the SDR white.
    p.ocio = fake_view(true, 203.0);
    const PlanarBuffer sdr = render_view(im, im, p);
    EXPECT_NEAR(sdr.at(1, 0, 0), code[1] * 203.0f / 80.0f, 4e-4);
    // A display that cannot be decoded: its codes as an SDR graphic at the SDR white.
    p.ocio = fake_view(false);
    const PlanarBuffer gfx = render_view(im, im, p);
    EXPECT_NEAR(gfx.at(1, 0, 0), 203.0f * srgb_to_linear(code[1]) / 80.0f, 4e-4);
}

TEST(ViewOcio, ClippedByTheViewIsReadOnItsCodes) {
    // 2 000 000 nits: the stand-in's code is within half a 10-bit code of 1.
    const NetworkLinearImage im = flat(8, 8, 200.0f, 200.0f, 200.0f);
    ViewParams p = view_params(ViewMode::Image, 203.0);
    p.ocio = fake_view();
    p.clip_zebra = true;
    const PlanarBuffer out = render_view(im, im, p);
    // The clip stripes are on the (x - y) diagonal, two pixels of every eight.
    const float stripe = out.at(2, 0, 0), plain = out.at(2, 0, 4);
    EXPECT_NEAR(stripe, plain * (1.0f - kZebraMix) + kScreenClipColour[2] * kZebraMix, 1e-5);
    // A pixel well inside the view is never striped.
    const NetworkLinearImage dim = flat(8, 8, 0.002f, 0.002f, 0.002f);
    const PlanarBuffer o2 = render_view(dim, dim, p);
    EXPECT_FLOAT_EQ(o2.at(2, 0, 0), o2.at(2, 0, 4));
}

TEST(ViewOcio, TheChecksMarkOutOfGamutAndInvalidPixels) {
    NetworkLinearImage im = flat(8, 8, 0.01f, 0.01f, 0.01f);
    im.buffer().at(0, 0, 0) = std::nanf("");           // invalid: NaN
    // A green of Rec.2020 (in Rec.709: a channel below zero): outside Rec.709, inside AP1.
    im.buffer().at(0, 1, 1) = -0.0027f;
    im.buffer().at(1, 1, 1) = 0.0111f;
    im.buffer().at(2, 1, 1) = 0.0012f;
    for (int k = 0; k < 3; ++k) im.buffer().at(k, 2, 2) = 1500.0f;   // 15 000 000 nits: above half float in ACEScg
    im.buffer().at(1, 3, 3) = -0.2f;                    // outside AP0: no colour at all
    ViewParams p = view_params(ViewMode::Image, 203.0);
    p.invalid_zebra = true;
    p.gamut_zebra = true;
    p.gamut = Primaries::Rec709;
    const PlanarBuffer out = render_view(im, im, p);
    for (int k = 0; k < 3; ++k) {
        EXPECT_FLOAT_EQ(out.at(k, 0, 0), kInvalidColour[std::size_t(k)]) << "NaN";
        EXPECT_FLOAT_EQ(out.at(k, 2, 2), kInvalidColour[std::size_t(k)]) << "above half float";
        EXPECT_FLOAT_EQ(out.at(k, 3, 3), kInvalidColour[std::size_t(k)]) << "outside AP0";
    }
    // Out of gamut on the bands (rows 0-2 of every 8), mixed over the picture.
    EXPECT_GT(std::abs(out.at(1, 1, 1) - out.at(1, 1, 0)), 0.05f);
    EXPECT_FLOAT_EQ(out.at(1, 4, 5), out.at(1, 4, 6));   // inside, no band
    // AP1 holds that Rec.709 negative: no band.
    p.gamut = Primaries::Ap1;
    const PlanarBuffer ap1 = render_view(im, im, p);
    const PlanarBuffer plain = render_view(im, im, view_params(ViewMode::Image, 203.0));
    for (int k = 0; k < 3; ++k) EXPECT_FLOAT_EQ(ap1.at(k, 1, 1), plain.at(k, 1, 1)) << k;
}

TEST(ViewOcio, TheProbeReadsTheWorkingSpace) {
    const float white[3] = {0.0203f, 0.0203f, 0.0203f};   // diffuse white, 203 nits
    for (WorkingSpace w : {WorkingSpace::AcesCg, WorkingSpace::Aces2065}) {
        const PixelReading r = read_pixel(white, Primaries::Rec709, w, Primaries::Ap1);
        for (int k = 0; k < 3; ++k) EXPECT_NEAR(r.working[std::size_t(k)], 1.0, 1e-4) << working_name(w);
        EXPECT_FALSE(r.invalid);
        EXPECT_FALSE(r.out_of_gamut);
        EXPECT_FALSE(r.has_code);
    }
    const float red[3] = {0.0203f, 0.0f, 0.0f};
    const PixelReading cg = read_pixel(red, Primaries::Rec709, WorkingSpace::AcesCg, Primaries::Ap1);
    const PixelReading ap0 = read_pixel(red, Primaries::Rec709, WorkingSpace::Aces2065, Primaries::Ap1);
    const Mat3 m = rgb_to_rgb_matrix(Primaries::Rec709, Primaries::Ap1);
    EXPECT_NEAR(cg.working[0], m[0][0], 1e-6);
    EXPECT_GT(std::abs(cg.working[1] - ap0.working[1]), 1e-3);   // the spaces differ off the grey axis
    auto view = fake_view();
    const PixelReading coded = read_pixel(white, Primaries::Rec709, WorkingSpace::AcesCg, Primaries::Ap1, view.get());
    ASSERT_TRUE(coded.has_code);
    EXPECT_NEAR(coded.code[0], 0.5f, 1e-4);   // 1 / (1 + 1)
    const float nan3[3] = {std::nanf(""), 0.0f, 0.0f};
    EXPECT_EQ(read_pixel(nan3, Primaries::Rec709, WorkingSpace::AcesCg, Primaries::Ap1).why, "NaN");
}

TEST(ViewOcio, TheOriginalComesInThroughItsInputTransform) {
    SdrImage sdr(1, 3);
    const float codes[3][3] = {{1.0f, 1.0f, 1.0f}, {0.5f, 0.25f, 0.0f}, {0.0f, 0.0f, 0.002f}};
    for (int x = 0; x < 3; ++x)
        for (int k = 0; k < 3; ++k) sdr.buffer().at(k, 0, x) = codes[x][k];
    OcioInputTransform t;
    t.id = "codes as ACEScg";
    t.cpu = [](float*, std::size_t) {};   // the codes are the working values
    const std::vector<float> l = sdr_light_through(sdr, t, Primaries::Rec709);
    ASSERT_EQ(l.size(), 12u);
    const Mat3 m = rgb_to_rgb_matrix(Primaries::Ap1, Primaries::Rec709);
    for (int x = 0; x < 3; ++x)
        for (int k = 0; k < 3; ++k) {
            const double want = (m[std::size_t(k)][0] * codes[x][0] + m[std::size_t(k)][1] * codes[x][1] +
                                 m[std::size_t(k)][2] * codes[x][2]) * 203.0 / 10000.0;
            EXPECT_NEAR(l[std::size_t(x) * 4 + std::size_t(k)], want, 1e-6);
        }
    // The clamp flags are the plate's, whatever the way in.
    EXPECT_EQ(int(l[3]), kFlagClipR | kFlagClipG | kFlagClipB);
    EXPECT_EQ(int(l[11]), kFlagCrush);
    // The display pass reads it wherever the SDR source is shown.
    ViewParams p = view_params(ViewMode::Image, 203.0, ViewSource::Sdr);
    p.original = std::make_shared<OcioInputTransform>(t);
    EXPECT_EQ(view_sdr_light(&sdr, p), l);
    p.original.reset();
    EXPECT_EQ(view_sdr_light(&sdr, p), sdr_light_rgba(&sdr));
}

#ifdef RUDRA_HAVE_OCIO
namespace {
const ocio_viewer::Config& builtin() {
    static auto c = ocio_viewer::Config::open({});
    EXPECT_TRUE(c.ok()) << (c.ok() ? "" : c.error().detail);
    return *c.value();
}
}  // namespace

TEST(OcioViewer, TheBuiltInConfigHasTheViewsAskedFor) {
    const auto& c = builtin();
    std::set<std::string> have;
    for (const auto& p : c.presets()) have.insert(p.display + "|" + p.view);
    for (const char* want : {"sRGB - Display|ACES 2.0 - SDR 100 nits (Rec.709)",
                             "Rec.1886 Rec.709 - Display|ACES 2.0 - SDR 100 nits (Rec.709)",
                             "Rec.2100-PQ - Display|ACES 2.0 - HDR 1000 nits (P3 D65)",
                             "P3-D65 - Display|ACES 2.0 - SDR 100 nits (P3 D65)", "sRGB - Display|Raw"})
        EXPECT_TRUE(have.count(want)) << want;
    // Every display and view of the config can be shown, and decoded for an HDR screen,
    // with its samplers after the display pass's own (binding 5) and none shared.
    for (const std::string& d : c.displays())
        for (const std::string& v : c.views(d)) {
            auto t = c.view(d, v, WorkingSpace::AcesCg);
            ASSERT_TRUE(t.ok()) << d << " / " << v;
            EXPECT_TRUE(t.value()->decodes) << d << " / " << v;
            std::set<int> bindings;
            for (const auto* st : {&t.value()->forward, &t.value()->decode})
                for (const OcioTexture& tx : st->textures) {
                    EXPECT_GE(tx.binding, int(ocio_viewer::kFirstOcioBinding));
                    EXPECT_TRUE(bindings.insert(tx.binding).second) << d << " / " << v << " " << tx.sampler;
                    EXPECT_EQ(tx.values.size(), std::size_t(tx.width) * tx.height * tx.depth * tx.channels);
                }
            EXPECT_NE(t.value()->forward.glsl.find("vec4 ocioView("), std::string::npos);
        }
}

TEST(OcioViewer, AcesCgMidGreyThroughTheSrgbView) {
    auto v = builtin().view("sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)", WorkingSpace::AcesCg);
    ASSERT_TRUE(v.ok());
    float g[3] = {0.18f, 0.18f, 0.18f};
    v.value()->forward.cpu(g, 1);
    EXPECT_NEAR(g[0], 0.34919f, 2e-5);   // PyOpenColorIO 2.5.2, the same config and view
    EXPECT_FALSE(v.value()->hdr);
    EXPECT_DOUBLE_EQ(v.value()->nits_per_y, 203.0);
}

TEST(OcioViewer, AnHdr10ScreenShowsThePqViewAsItsOwnCodes) {
    auto v = builtin().view("Rec.2100-PQ - Display", "ACES 2.0 - HDR 1000 nits (Rec.2020)", WorkingSpace::AcesCg);
    ASSERT_TRUE(v.ok());
    EXPECT_TRUE(v.value()->hdr);
    const int w = 64;
    NetworkLinearImage im(1, w);
    for (int x = 0; x < w; ++x) {
        const float l = std::exp2(-10.0f + 16.0f * float(x) / float(w - 1)) * 203.0f / kP;
        im.buffer().at(0, 0, x) = l;
        im.buffer().at(1, 0, x) = l * 0.6f;
        im.buffer().at(2, 0, x) = l * 0.3f;
    }
    ViewParams p = view_params(ViewMode::Image, 203.0);
    p.ocio = v.value();
    p.target = DisplayTarget::hdr10(10000.0);
    const PlanarBuffer out = render_view(im, im, p);
    const ViewMatrices m = view_matrices(Primaries::Rec709, WorkingSpace::AcesCg, Primaries::Ap1, 100.0);
    for (int x = 0; x < w; ++x) {
        float px[3], h[3] = {im.buffer().at(0, 0, x), im.buffer().at(1, 0, x), im.buffer().at(2, 0, x)};
        for (int k = 0; k < 3; ++k)
            px[k] = m.work[std::size_t(k) * 3] * h[0] + m.work[std::size_t(k) * 3 + 1] * h[1] + m.work[std::size_t(k) * 3 + 2] * h[2];
        v.value()->forward.cpu(px, 1);
        for (int k = 0; k < 3; ++k) EXPECT_NEAR(out.at(k, 0, x), std::clamp(px[k], 0.0f, 1.0f), 0.5f / 1023.0f) << x << " " << k;
    }
}

TEST(OcioViewer, TheOriginalLooksAsGradedThroughTheInverseOfItsView) {
    const auto& c = builtin();
    auto in = c.input("inverse:sRGB - Display\tACES 2.0 - SDR 100 nits (Rec.709)", WorkingSpace::AcesCg);
    auto v = c.view("sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)", WorkingSpace::AcesCg);
    ASSERT_TRUE(in.ok() && v.ok());
    SdrImage sdr(1, 16);
    for (int x = 0; x < 16; ++x) {
        sdr.buffer().at(0, 0, x) = float(x) / 15.0f;
        sdr.buffer().at(1, 0, x) = float(x) / 30.0f;
        sdr.buffer().at(2, 0, x) = 0.2f;
    }
    ViewParams p = view_params(ViewMode::Image, 203.0, ViewSource::Sdr);
    p.ocio = v.value();
    p.original = in.value();
    const NetworkLinearImage none(1, 16);
    const PlanarBuffer out = render_view(none, none, p, &sdr);
    for (int x = 0; x < 16; ++x)
        for (int k = 0; k < 3; ++k) EXPECT_NEAR(out.at(k, 0, x), sdr.buffer().at(k, 0, x), 0.5f / 255.0f) << x << " " << k;
}

TEST(OcioViewer, BothWorkingSpacesShowTheSamePicture) {
    const auto& c = builtin();
    auto cg = c.view("Rec.1886 Rec.709 - Display", "ACES 2.0 - SDR 100 nits (Rec.709)", WorkingSpace::AcesCg);
    auto ap0 = c.view("Rec.1886 Rec.709 - Display", "ACES 2.0 - SDR 100 nits (Rec.709)", WorkingSpace::Aces2065);
    ASSERT_TRUE(cg.ok() && ap0.ok());
    const NetworkLinearImage im = flat(4, 1, 0.03f, 0.012f, 0.004f);
    ViewParams p = view_params(ViewMode::Image, 203.0);
    p.ocio = cg.value();
    const PlanarBuffer a = render_view(im, im, p);
    p.ocio = ap0.value();
    p.working = WorkingSpace::Aces2065;
    const PlanarBuffer b = render_view(im, im, p);
    for (int k = 0; k < 3; ++k) EXPECT_NEAR(a.at(k, 0, 0), b.at(k, 0, 0), 2e-5);
}
#endif
