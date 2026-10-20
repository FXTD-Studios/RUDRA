// The still decoder against rudra/decode.py (goldens from tools/emit_decode_golden.py).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

#include <nlohmann/json.hpp>

#include "rudra/media/still.hpp"
#include "rudra/platform/npy.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {

const fs::path kDir = fs::path(RUDRA_GOLDEN_DIR) / "decode";

nlohmann::json index_json() {
    std::ifstream in(kDir / "index.json");
    return nlohmann::json::parse(in);
}

}  // namespace

TEST(Decode, EveryFixtureMatchesDecodeSdr) {
    const auto idx = index_json();
    ASSERT_GE(idx.at("cases").size(), 15u);
    for (const auto& c : idx.at("cases")) {
        const std::string file = c.at("file");
        SCOPED_TRACE(file);
        auto r = decode_sdr_file(kDir / file);
        if (c.contains("refused")) {
            ASSERT_FALSE(r);
            EXPECT_EQ(r.error().code, ErrorCode::Unsupported);
            EXPECT_EQ(r.error().message, c.at("refused").get<std::string>());
            continue;
        }
        ASSERT_TRUE(r) << r.error().message;
        EXPECT_EQ(r->bits, c.at("bits").get<int>());
        EXPECT_EQ(r->distinct_codes, c.at("distinct_codes").get<int>());
        EXPECT_EQ(r->source, c.at("source").get<std::string>());
        auto want = read_npy(kDir / c.at("expected").get<std::string>());
        ASSERT_TRUE(want);
        const auto got = r->rgb.buffer().span();
        ASSERT_EQ(got.size(), want->data.size());
        // Lossless formats decode to identical integers, so identical floats.
        // JPEG is a decoder's arithmetic: libjpeg-turbo's ISLOW IDCT and fancy
        // upsampling are specified to the bit, so it is held exact too.
        std::size_t diff = 0;
        for (std::size_t i = 0; i < got.size(); ++i) diff += got[i] != want->data[i];
        EXPECT_EQ(diff, 0u) << diff << " of " << got.size() << " values differ";
    }
}

TEST(Decode, RejectsGarbageAndEmpty) {
    EXPECT_FALSE(decode_sdr(std::span<const std::uint8_t>{}));
    const std::uint8_t junk[] = {1, 2, 3, 4, 5, 6, 7, 8};
    auto r = decode_sdr(junk);
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().code, ErrorCode::Unsupported);
    EXPECT_FALSE(decode_sdr_file(kDir / "does_not_exist.png"));
}

#ifdef RUDRA_HAVE_STILL_DECODE
// The preview size (ui/server.py _fit, tests/golden/fit): the same floats.
TEST(Decode, FitIsTheStudiosPreviewDownscale) {
    std::ifstream in(std::string(RUDRA_GOLDEN_DIR) + "/fit/index.json");
    const auto idx = nlohmann::json::parse(in);
    ASSERT_GE(idx["cases"].size(), 5u);
    const std::string dir = std::string(RUDRA_GOLDEN_DIR);
    for (const auto& c : idx["cases"]) {
        auto a = read_npy(dir + "/decode/" + c["input"].get<std::string>());
        auto e = read_npy(dir + "/fit/" + c["expected"].get<std::string>());
        ASSERT_TRUE(a && e);
        const int h = int(a->shape[1]), w = int(a->shape[2]);
        PlanarBuffer b(3, h, w);
        std::copy(a->data.begin(), a->data.end(), b.span().begin());
        const SdrImage out = fit_max_side(SdrImage(std::move(b)), c["max_side"].get<int>());
        ASSERT_EQ(out.buffer().height(), int(e->shape[1])) << c["expected"];
        ASSERT_EQ(out.buffer().width(), int(e->shape[2])) << c["expected"];
        double worst = 0;
        const auto got = out.buffer().span();
        for (std::size_t i = 0; i < got.size(); ++i) worst = std::max(worst, double(std::abs(got[i] - e->data[i])));
        EXPECT_LE(worst, 1e-6) << c["expected"];   // OpenCV's float INTER_AREA, both sides
    }
}
#endif

// Roadmap 1.1 (16 Oct 2026): decode_sdr reads the file's own colour tag and
// applies its EXIF orientation. A fixture PNG with an eXIf and an iCCP chunk
// spliced in after IHDR (CRCs computed, as libpng checks them).
namespace {

std::uint32_t crc32(const std::uint8_t* p, std::size_t n, std::uint32_t c = 0xffffffffu) {
    for (std::size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1u)));
    }
    return c;
}

std::vector<std::uint8_t> splice(std::vector<std::uint8_t> png, const char* type, const std::vector<std::uint8_t>& data) {
    std::vector<std::uint8_t> c;
    auto be = [&](std::uint32_t v) { for (int s = 24; s >= 0; s -= 8) c.push_back(std::uint8_t(v >> s)); };
    be(std::uint32_t(data.size()));
    c.insert(c.end(), type, type + 4);
    c.insert(c.end(), data.begin(), data.end());
    be(~crc32(c.data() + 4, c.size() - 4));
    const std::size_t after_ihdr = 8 + 12 + 13;
    png.insert(png.begin() + std::ptrdiff_t(after_ihdr), c.begin(), c.end());
    return png;
}

std::vector<std::uint8_t> read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

}  // namespace

TEST(Decode, ExifOrientationIsApplied) {
    const auto png = read_file(fs::path(RUDRA_GOLDEN_DIR) / "master" / "still_png8.png");
    auto plain = decode_sdr(png);
    ASSERT_TRUE(plain);
    // eXIf: big-endian TIFF, IFD0 with orientation 6 (rotate 90 clockwise to show).
    const std::vector<std::uint8_t> tiff = {'M', 'M', 0, 42, 0, 0, 0, 8, 0, 1, 0x01, 0x12, 0, 3, 0, 0, 0, 1, 0, 6, 0, 0, 0, 0, 0, 0};
    auto turned = decode_sdr(splice(png, "eXIf", tiff));
    ASSERT_TRUE(turned);
    EXPECT_EQ(turned->colour.orientation, 6);
    const auto& a = plain->rgb.buffer();
    const auto& b = turned->rgb.buffer();
    ASSERT_EQ(b.width(), a.height());
    ASSERT_EQ(b.height(), a.width());
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < b.height(); ++y)
            for (int x = 0; x < b.width(); ++x) ASSERT_EQ(b.at(c, y, x), a.at(c, a.height() - 1 - x, y));
    EXPECT_EQ(plain->colour.orientation, 1);
    EXPECT_EQ(plain->colour.primaries, Primaries::Rec709);
}

TEST(Decode, IccProfileSetsThePrimaries) {
    const auto png = read_file(fs::path(RUDRA_GOLDEN_DIR) / "master" / "still_png8.png");
    // A Display P3 profile: header, rXYZ / gXYZ / bXYZ (D50), wrapped in a stored-block zlib stream.
    std::vector<std::uint8_t> icc(128, 0);
    std::memcpy(icc.data() + 16, "RGB ", 4);
    std::memcpy(icc.data() + 36, "acsp", 4);
    std::memcpy(icc.data() + 12, "mntr", 4);   // a display profile, XYZ PCS, D50: as libpng expects
    std::memcpy(icc.data() + 20, "XYZ ", 4);
    const std::uint8_t d50[12] = {0, 0, 0xf6, 0xd6, 0, 1, 0, 0, 0, 0, 0xd3, 0x2d};
    std::memcpy(icc.data() + 68, d50, 12);
    auto be = [](std::vector<std::uint8_t>& v, std::uint32_t x) { for (int s = 24; s >= 0; s -= 8) v.push_back(std::uint8_t(x >> s)); };
    be(icc, 3);
    const double p3[9] = {0.5151, 0.2412, -0.0011, 0.2920, 0.6922, 0.0419, 0.1571, 0.0666, 0.7841};
    const char* sig[3] = {"rXYZ", "gXYZ", "bXYZ"};
    for (int i = 0; i < 3; ++i) {
        icc.insert(icc.end(), sig[i], sig[i] + 4);
        be(icc, std::uint32_t(128 + 4 + 36 + 20 * i));
        be(icc, 20);
    }
    for (int i = 0; i < 3; ++i) {
        icc.insert(icc.end(), {'X', 'Y', 'Z', ' ', 0, 0, 0, 0});
        for (int k = 0; k < 3; ++k) be(icc, std::uint32_t(std::int32_t(std::lround(p3[3 * i + k] * 65536.0))));
    }
    for (int k = 0; k < 4; ++k) icc[std::size_t(k)] = std::uint8_t(icc.size() >> (24 - 8 * k));   // profile size
    std::vector<std::uint8_t> z = {0x78, 0x01, 1, std::uint8_t(icc.size()), std::uint8_t(icc.size() >> 8),
                                   std::uint8_t(~icc.size()), std::uint8_t(~icc.size() >> 8)};
    z.insert(z.end(), icc.begin(), icc.end());
    std::uint32_t a = 1, s = 0;
    for (auto c : icc) { a = (a + c) % 65521; s = (s + a) % 65521; }
    be(z, (s << 16) | a);
    std::vector<std::uint8_t> iccp = {'P', '3', 0, 0};
    iccp.insert(iccp.end(), z.begin(), z.end());
    auto tagged = decode_sdr(splice(png, "iCCP", iccp));
    ASSERT_TRUE(tagged) << tagged.error().message;
    EXPECT_EQ(tagged->colour.primaries, Primaries::P3D65);
    EXPECT_STREQ(primaries_id(tagged->colour.primaries), "p3d65");
    // The pixels are the same codes; only the master's conversion changes.
    const auto plain = decode_sdr(png);
    ASSERT_TRUE(plain);
    EXPECT_TRUE(std::equal(plain->rgb.buffer().span().begin(), plain->rgb.buffer().span().end(),
                           tagged->rgb.buffer().span().begin()));
}
