// Roadmap 3.3: painted masks. The brush paints what the page's demo paints,
// a set samples at any frame size, the gain takes the weights, the composite
// and the master chain gate their bands by them, the PNG beside the master
// round-trips through the codec with no library, the session undoes a stroke
// at a time, and a master request carries the file.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>

#include "rudra/core/composite.hpp"
#include "rudra/core/masks.hpp"
#include "rudra/core/master.hpp"
#include "rudra/core/view.hpp"
#include "rudra/deliver/master.hpp"
#include "rudra/engine/session.hpp"
#include "rudra/platform/png8.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {

std::shared_ptr<MaskSet> painted(int w, int h, int band, double x0, double y0, double x1, double y1, Brush brush = {}) {
    auto set = std::make_shared<MaskSet>();
    set->width = w;
    set->height = h;
    auto plane = empty_plane(w, h);
    paint_stroke(*plane, w, h, x0, y0, x1, y1, brush);
    set->planes[std::size_t(band)] = std::move(plane);
    return set;
}

}  // namespace

TEST(Masks, TheBrushIsSoftRoundAndFlows) {
    MaskPlane m(64 * 64, 0);
    Brush b;
    b.size_px = 20;
    b.softness = 0.5;
    b.flow = 1.0;
    paint_stroke(m, 64, 64, 32, 32, 32, 32, b);   // one stamp
    EXPECT_EQ(m[32 * 64 + 32], 255);                // full at the centre
    EXPECT_GT(m[32 * 64 + 38], 0);                  // inside the ramp
    EXPECT_LT(m[32 * 64 + 38], 255);
    EXPECT_EQ(m[32 * 64 + 43], 0);                  // past the radius
    EXPECT_EQ(m[10 * 64 + 10], 0);
    MaskPlane soft(64 * 64, 0);
    b.flow = 0.3;
    paint_stroke(soft, 64, 64, 32, 32, 32, 32, b);
    EXPECT_NEAR(soft[32 * 64 + 32], 77, 1);          // flow: 30 % of the way up
    paint_stroke(soft, 64, 64, 32, 32, 32, 32, b);
    EXPECT_NEAR(soft[32 * 64 + 32], 130, 2);         // and 30 % of what is left
    b.erase = true;
    b.flow = 1.0;
    paint_stroke(soft, 64, 64, 32, 32, 32, 32, b);
    EXPECT_EQ(soft[32 * 64 + 32], 0);                // erased
    // A drag covers the segment, not just its ends.
    MaskPlane line(64 * 64, 0);
    b.erase = false;
    b.softness = 0.0;
    paint_stroke(line, 64, 64, 5, 32, 59, 32, b);
    for (int x = 5; x <= 59; ++x) EXPECT_EQ(line[32 * 64 + x], 255) << x;
    invert_plane(line);
    EXPECT_EQ(line[32 * 64 + 32], 0);
    EXPECT_EQ(line[2 * 64 + 2], 255);
}

TEST(Masks, ASetSamplesAtAnyFrameSize) {
    auto set = painted(40, 20, 1, 0, 10, 39, 10, Brush{6.0, 0.0, 1.0, false});
    EXPECT_TRUE(set->has(1));
    EXPECT_FALSE(set->has(0));
    EXPECT_EQ(set->bands(), std::vector<int>{1});
    EXPECT_GT(set->coverage(1), 0.2);
    EXPECT_LT(set->coverage(1), 0.5);
    EXPECT_EQ(set->coverage(0), 0.0);
    EXPECT_FLOAT_EQ(set->weight(0, 5, 5, 40, 20), 1.0f);     // none: 1 everywhere
    EXPECT_FLOAT_EQ(set->weight(1, 20, 10, 40, 20), 1.0f);   // on the stroke
    EXPECT_FLOAT_EQ(set->weight(1, 20, 1, 40, 20), 0.0f);    // off it
    // At twice the size the stroke is twice as wide, bilinear in between.
    EXPECT_FLOAT_EQ(set->weight(1, 40, 20, 80, 40), 1.0f);
    EXPECT_FLOAT_EQ(set->weight(1, 40, 2, 80, 40), 0.0f);
    float ws[kMaxMaskBands];
    set->weights(40, 20, 80, 40, ws);
    EXPECT_FLOAT_EQ(ws[0], 1.0f);
    EXPECT_FLOAT_EQ(ws[1], 1.0f);
    const auto rgba = set->rgba8();
    ASSERT_EQ(rgba.size(), std::size_t(40 * 20 * 4));
    EXPECT_EQ(rgba[(10 * 40 + 20) * 4 + 0], 255);   // band 0 absent: 255
    EXPECT_EQ(rgba[(10 * 40 + 20) * 4 + 1], 255);   // band 1 on the stroke
    EXPECT_EQ(rgba[(1 * 40 + 20) * 4 + 1], 0);
}

TEST(Masks, TheGainTakesTheWeights) {
    const std::vector<RegionBand> bands = {{400.0, 2000.0, 1.0}, {2000.0, 8000.0, 0.0}, {0.05, 12.0, -1.0}};
    const double bright[3] = {1000.0, 1000.0, 1000.0};
    EXPECT_NEAR(region_ev_gain(bright, bands, 1.0), 2.0, 1e-9);
    const float none[kMaxMaskBands] = {1.0f, 1.0f, 1.0f, 1.0f};
    EXPECT_NEAR(region_ev_gain(bright, bands, 1.0, none), 2.0, 1e-9);
    const float half[kMaxMaskBands] = {0.5f, 1.0f, 1.0f, 1.0f};
    EXPECT_NEAR(region_ev_gain(bright, bands, 1.0, half), std::exp2(0.5), 1e-6);
    const float off[kMaxMaskBands] = {0.0f, 1.0f, 1.0f, 1.0f};
    EXPECT_NEAR(region_ev_gain(bright, bands, 1.0, off), 1.0, 1e-9);
    const double dark[3] = {1.0, 1.0, 1.0};
    const float shadow_off[kMaxMaskBands] = {1.0f, 1.0f, 0.0f, 1.0f};
    EXPECT_NEAR(region_ev_gain(dark, bands, 1.0), 0.5, 1e-9);
    EXPECT_NEAR(region_ev_gain(dark, bands, 1.0, shadow_off), 1.0, 1e-9);
}

TEST(Masks, TheCompositeAndTheMasterChainGateByThem) {
    const int w = 32, h = 16;
    SdrImage sdr(h, w, 0.9f);   // bright everywhere
    Fields fields;
    fields.residual = PlanarBuffer(3, h, w, 0.0f);
    fields.shadow = PlanarBuffer(1, h, w, 0.0f);
    fields.highlight = PlanarBuffer(1, h, w, 0.0f);
    FrameScalars sc;
    const ModelConstants model;
    CompositeParams p;
    p.regions = {{100.0, 100000.0, 1.0}};   // +1 stop on everything bright
    const auto plain = composite(sdr, fields, sc, model, CompositeParams{});
    const auto graded = composite(sdr, fields, sc, model, p);
    p.masks = painted(w, h, 0, 0, 8, 10, 8, Brush{4.0, 0.0, 1.0, false});   // the left end only
    const auto masked = composite(sdr, fields, sc, model, p);
    EXPECT_NEAR(masked.buffer().at(0, 8, 4), graded.buffer().at(0, 8, 4), 1e-6);   // inside: graded
    EXPECT_NEAR(masked.buffer().at(0, 8, 28), plain.buffer().at(0, 8, 28), 1e-6);  // outside: untouched
    EXPECT_NEAR(graded.buffer().at(0, 8, 28), plain.buffer().at(0, 8, 28) * 2.0f, 1e-5);
    // The master chain at twice the size, the mask the preview's.
    NetworkLinearImage big(h * 2, w * 2, 0.05f);
    NitsFrame nits = nits_from_network(big);
    apply_region_ev(nits, p.regions, 1.0, 40000.0, p.masks.get());
    EXPECT_NEAR(nits.plane(0)[16 * 64 + 8], 1000.0, 1e-3);    // inside, +1 stop
    EXPECT_NEAR(nits.plane(0)[16 * 64 + 56], 500.0, 1e-3);    // outside
    MasterParams mp;
    mp.regions = p.regions;
    mp.masks = p.masks;
    mp.anchor = mp.carry_chroma = mp.settle_grain = false;
    mp.container = MasterContainer::SceneLinear;
    SdrImage big_sdr(h * 2, w * 2, 0.9f);
    const MasterPixels out = render_master_pixels(big, big_sdr, model, mp);
    EXPECT_NEAR(out.pixels.at(0, 16, 8) / out.pixels.at(0, 16, 56), 2.0, 1e-5);
}

TEST(Masks, TheTintFollowsTheMaskOnTheView) {
    const int w = 16, h = 8;
    NetworkLinearImage m(h, w, 0.01f), b(h, w, 0.01f);
    ViewParams v = view_params(ViewMode::Image, 203.0);
    const PlanarBuffer plain = render_view(m, b, v);
    v.paint_band = 0;
    v.masks = painted(w, h, 0, 0, 4, 6, 4, Brush{3.0, 0.0, 1.0, false});
    const PlanarBuffer tinted = render_view(m, b, v);
    EXPECT_NE(tinted.at(0, 4, 2), plain.at(0, 4, 2));   // under the stroke
    EXPECT_EQ(tinted.at(0, 4, 14), plain.at(0, 4, 14));  // away from it
    EXPECT_NEAR(tinted.at(0, 4, 2), plain.at(0, 4, 2) * (1.0f - kMaskTintMix) + v.paint_tint[0] * kMaskTintMix, 1e-6);
    v.paint_band = 2;   // a band with no plane: no tint
    EXPECT_EQ(render_view(m, b, v).at(0, 4, 2), plain.at(0, 4, 2));
}

TEST(Masks, ThePngCodecRoundTripsAndReadsFilteredFiles) {
    const fs::path dir = fs::temp_directory_path() / "rudra_masks_test";
    fs::create_directories(dir);
    auto set = painted(50, 30, 2, 5, 15, 45, 15, Brush{8.0, 0.7, 0.9, false});
    set->planes[0] = empty_plane(50, 30);
    (*std::const_pointer_cast<MaskPlane>(set->planes[0]))[7 * 50 + 7] = 128;
    const fs::path png = dir / "m.masks.png";
    ASSERT_TRUE(write_mask_set(png, *set));
    auto back = read_mask_set(png);
    ASSERT_TRUE(back) << back.error().message;
    EXPECT_EQ(*back, *set);
    EXPECT_FALSE(back->has(1));   // never painted: read back as none
    // The codec itself: every channel count, and a file with the other filters.
    for (int ch : {1, 2, 3, 4}) {
        Png8 img;
        img.width = 7;
        img.height = 5;
        img.channels = ch;
        for (int i = 0; i < 35 * ch; ++i) img.data.push_back(std::uint8_t(i * 37));
        auto dec = decode_png8(png8_bytes(img));
        ASSERT_TRUE(dec) << ch;
        EXPECT_EQ(dec->channels, ch);
        EXPECT_EQ(dec->data, img.data) << ch;
    }
    // A 2x4 RGBA PNG whose rows use Sub, Up, Average and Paeth, deflated by
    // zlib at level 9 (pixel (x, y, k) = (53 y + 17 x + 7 k) mod 256).
    static const std::uint8_t filtered[] = {
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x02,
        0x00, 0x00, 0x00, 0x04, 0x08, 0x06, 0x00, 0x00, 0x00, 0xA4, 0xEF, 0xEE, 0x39, 0x00, 0x00, 0x00, 0x1F, 0x49, 0x44, 0x41,
        0x54, 0x78, 0xDA, 0x63, 0x64, 0x60, 0xE7, 0x13, 0x15, 0x04, 0x02, 0x26, 0x53, 0x28, 0x60, 0x0E, 0x08, 0x0E, 0x8F, 0x52,
        0x06, 0x02, 0x16, 0x10, 0x0F, 0x24, 0x05, 0x00, 0x54, 0x29, 0x05, 0x19, 0x90, 0xB7, 0xCE, 0x39, 0x00, 0x00, 0x00, 0x00,
        0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82};
    auto f = decode_png8(filtered);
    ASSERT_TRUE(f) << f.error().detail;
    EXPECT_EQ(f->width, 2);
    EXPECT_EQ(f->height, 4);
    EXPECT_EQ(f->channels, 4);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 2; ++x)
            for (int k = 0; k < 4; ++k)
                EXPECT_EQ(f->data[std::size_t((y * 2 + x) * 4 + k)], (53 * y + 17 * x + 7 * k) % 256) << y << x << k;
    EXPECT_FALSE(decode_png8(std::vector<std::uint8_t>{1, 2, 3}));
    fs::remove_all(dir);
}

TEST(Masks, TheSessionUndoesAStrokeAtATime) {
    Session s;
    EXPECT_EQ(s.painted_band(), -1);
    s.stroke(64, 64, 10, 10, 20, 10, false);   // nothing armed: nothing painted
    EXPECT_FALSE(s.grade.masks);
    s.arm_mask(0);
    EXPECT_EQ(s.painted_band(), 0);
    Brush b;
    b.size_px = 10;
    b.softness = 0.0;
    b.flow = 1.0;
    s.set_brush(b);
    const std::string before = s.params_json();
    s.stroke_begin();
    s.stroke(64, 64, 10, 10, 20, 10, false);
    s.stroke(64, 64, 20, 10, 30, 10, false);
    s.stroke_end();
    ASSERT_TRUE(s.grade.masks && s.grade.masks->has(0));
    EXPECT_EQ(s.undo_depth(), 1u);                      // one stroke, one step
    EXPECT_NE(s.params_json().find("\"masks\":[0]"), std::string::npos);
    EXPECT_EQ(s.composite_params().masks, s.grade.masks);
    const double cov1 = s.grade.masks->coverage(0);
    s.stroke_begin();
    s.stroke(64, 64, 40, 40, 50, 40, false);
    s.stroke_end();
    EXPECT_EQ(s.undo_depth(), 2u);
    EXPECT_GT(s.grade.masks->coverage(0), cov1);
    s.undo();
    EXPECT_NEAR(s.grade.masks->coverage(0), cov1, 1e-12);
    s.invert_mask();
    EXPECT_NEAR(s.grade.masks->coverage(0), 1.0 - cov1, 1e-3);
    s.undo();
    s.arm_mask(1);
    s.stroke(64, 64, 5, 50, 5, 50, false);   // begins its own step
    s.stroke_end();
    EXPECT_EQ(s.grade.masks->bands(), (std::vector<int>{0, 1}));
    EXPECT_NE(s.params_json().find("\"masks\":[0,1]"), std::string::npos);
    s.clear_mask(1);
    EXPECT_EQ(s.grade.masks->bands(), std::vector<int>{0});
    s.clear_masks();
    EXPECT_FALSE(s.grade.masks);
    EXPECT_EQ(s.params_json(), before);
    s.undo();
    EXPECT_TRUE(s.grade.masks && s.grade.masks->has(0));
    // A frame of another size starts a fresh set.
    s.arm_mask(0);
    s.stroke(32, 32, 5, 5, 5, 5, false);
    EXPECT_EQ(s.grade.masks->width, 32);
    s.arm_mask(-1);
    EXPECT_EQ(s.painted_band(), -1);
}

TEST(Masks, AMasterRequestCarriesTheFile) {
    const fs::path dir = fs::temp_directory_path() / "rudra_masks_request";
    fs::create_directories(dir);
    auto set = painted(20, 10, 0, 2, 5, 18, 5, Brush{4.0, 0.0, 1.0, false});
    const fs::path png = dir / "shot.masks.png";
    ASSERT_TRUE(write_mask_set(png, *set));
    MasterRequest q;
    q.masks = set;
    q.masks_file = png.string();
    auto r = master_request_from_json(master_request_json(q));
    ASSERT_TRUE(r) << r.error().message;
    ASSERT_TRUE(r->masks);
    EXPECT_EQ(*r->masks, *set);
    EXPECT_EQ(r->masks_file, png.string());
    MasterRequest plain;
    EXPECT_EQ(master_request_json(plain).find("masks_file"), std::string::npos);
    EXPECT_FALSE(master_request_from_json("{\"masks_file\":\"" + (dir / "missing.png").generic_string() + "\"}"));
    fs::remove_all(dir);
}
