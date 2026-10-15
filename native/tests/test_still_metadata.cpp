// Roadmap 1.1 (16 Oct 2026): a still's own colour tag and EXIF orientation,
// read from its bytes (media/still_metadata.hpp). Synthetic files, built here:
// ICC profiles with each primaries set's D50 colorants, zlib streams from
// Python's zlib (fixed and dynamic Huffman), PNG / JPEG / TIFF / WebP wrappers.

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "rudra/media/still_metadata.hpp"

using namespace rudra;

namespace {

using B = std::vector<std::uint8_t>;

void put32(B& b, std::uint32_t v) {
    for (int s = 24; s >= 0; s -= 8) b.push_back(std::uint8_t(v >> s));
}
void put32le(B& b, std::uint32_t v) {
    for (int s = 0; s < 32; s += 8) b.push_back(std::uint8_t(v >> s));
}
void put16(B& b, unsigned v) {
    b.push_back(std::uint8_t(v >> 8));
    b.push_back(std::uint8_t(v));
}
void put16le(B& b, unsigned v) {
    b.push_back(std::uint8_t(v));
    b.push_back(std::uint8_t(v >> 8));
}
void put_s(B& b, const char* s, std::size_t n) { b.insert(b.end(), s, s + n); }

B hex(const char* h) {
    B b;
    for (std::size_t i = 0; h[i] && h[i + 1]; i += 2) b.push_back(std::uint8_t(std::stoi(std::string(h + i, 2), nullptr, 16)));
    return b;
}

std::uint32_t adler32(const B& d) {
    std::uint32_t a = 1, s = 0;
    for (auto c : d) {
        a = (a + c) % 65521;
        s = (s + a) % 65521;
    }
    return (s << 16) | a;
}

// A minimal RGB ICC profile: header, rXYZ / gXYZ / bXYZ, a v2 description.
B icc(const double rgb[9], const std::string& desc, const char* space = "RGB ") {
    B tags;
    struct Tag { const char* sig; B data; };
    std::vector<Tag> t;
    for (int i = 0; i < 3; ++i) {
        B x;
        put_s(x, "XYZ \0\0\0\0", 8);
        for (int k = 0; k < 3; ++k) put32(x, std::uint32_t(std::int32_t(std::lround(rgb[3 * i + k] * 65536.0))));
        t.push_back({i == 0 ? "rXYZ" : i == 1 ? "gXYZ" : "bXYZ", x});
    }
    B d;
    put_s(d, "desc\0\0\0\0", 8);
    put32(d, std::uint32_t(desc.size() + 1));
    put_s(d, desc.c_str(), desc.size() + 1);
    t.push_back({"desc", d});
    B out(128, 0);
    std::memcpy(out.data() + 16, space, 4);
    std::memcpy(out.data() + 36, "acsp", 4);
    put32(out, std::uint32_t(t.size()));
    std::uint32_t off = 128 + 4 + 12 * std::uint32_t(t.size());
    for (const auto& e : t) {
        put_s(out, e.sig, 4);
        put32(out, off);
        put32(out, std::uint32_t(e.data.size()));
        off += std::uint32_t(e.data.size());
    }
    for (const auto& e : t) out.insert(out.end(), e.data.begin(), e.data.end());
    const std::uint32_t size = std::uint32_t(out.size());
    for (int s = 0; s < 4; ++s) out[std::size_t(s)] = std::uint8_t(size >> (24 - 8 * s));
    return out;
}

constexpr double kSrgb[9] = {0.4360, 0.2225, 0.0139, 0.3851, 0.7169, 0.0971, 0.1430, 0.0606, 0.7139};
constexpr double kP3[9] = {0.5151, 0.2412, -0.0011, 0.2920, 0.6922, 0.0419, 0.1571, 0.0666, 0.7841};
constexpr double k2020[9] = {0.6735, 0.2790, -0.0019, 0.1657, 0.6753, 0.0300, 0.1250, 0.0456, 0.7969};
constexpr double kAdobe[9] = {0.6097, 0.3111, 0.0195, 0.2053, 0.6257, 0.0609, 0.1492, 0.0632, 0.7446};
constexpr double kOdd[9] = {0.5000, 0.3000, 0.0000, 0.3000, 0.6000, 0.1000, 0.1600, 0.1000, 0.7200};

// zlib with stored (uncompressed) deflate blocks: enough to wrap an ICC here.
B zlib_stored(const B& d) {
    B z = {0x78, 0x01};
    std::size_t at = 0;
    do {
        const std::size_t n = std::min<std::size_t>(65535, d.size() - at);
        z.push_back(at + n == d.size() ? 1 : 0);
        put16le(z, unsigned(n));
        put16le(z, unsigned(~n & 0xffff));
        z.insert(z.end(), d.begin() + std::ptrdiff_t(at), d.begin() + std::ptrdiff_t(at + n));
        at += n;
    } while (at < d.size());
    put32(z, adler32(d));
    return z;
}

void chunk(B& png, const char* type, const B& data) {
    put32(png, std::uint32_t(data.size()));
    put_s(png, type, 4);
    png.insert(png.end(), data.begin(), data.end());
    put32(png, 0);   // CRC: not checked by the reader
}

B png_with(const std::vector<std::pair<const char*, B>>& chunks) {
    B png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    chunk(png, "IHDR", B(13, 0));
    for (const auto& [t, d] : chunks) chunk(png, t, d);
    chunk(png, "IDAT", {});
    chunk(png, "IEND", {});
    return png;
}

B tiff_orientation(unsigned o, bool little) {
    B t;
    if (little) {
        put_s(t, "II", 2);
        put16le(t, 42);
        put32le(t, 8);
        put16le(t, 1);
        put16le(t, 0x0112);
        put16le(t, 3);
        put32le(t, 1);
        put16le(t, o);
        put16le(t, 0);
        put32le(t, 0);
    } else {
        put_s(t, "MM", 2);
        put16(t, 42);
        put32(t, 8);
        put16(t, 1);
        put16(t, 0x0112);
        put16(t, 3);
        put32(t, 1);
        put16(t, o);
        put16(t, 0);
        put32(t, 0);
    }
    return t;
}

}  // namespace

TEST(StillMetadata, InflatesPythonsZlib) {
    // zlib.compress(b"hello hello hello RUDRA", 1): fixed Huffman.
    const B fixed = hex("7801cb48cdc9c957c8402283425d821c01667b081b");
    const B a = zlib_inflate(fixed);
    EXPECT_EQ(std::string(a.begin(), a.end()), "hello hello hello RUDRA");
    // 600 random letters at level 9: dynamic Huffman; checked by the stream's own Adler-32.
    const B dyn = hex(
        "78da2d51810a052108fb157f4d68602005bafeffcde31dd459eadc96e33ea0e00b41606f10b5b1c0f716d6ba7306b7b6c532e8ead1a16821"
        "d956dbfcace324cca89a68b7c30462c31cad1d56cbdd3c7b1d404d8251f15b2ab4aba2e348342800e3bd0d9c5bc42eda76bbe2852fd9a2352"
        "4cb73392cf209ea98d5d5877e43961ae11a2f7ef505cc0b514e176c9fee413337d14080219c12149f46eba04495853fa691576b040d6bd2d2"
        "8fb9cf90a6b353a49fab1157721afb09ec9e2bc1c318eb3c64b60456cdd087306c29a05069cb29a7c5e5d88a1da21b2135b262da3b6d04584"
        "4f5441c87114764e557460a814749fd8325d91b57469adeea3352f36cb8ca63c8d2b9da717d530a94f03a9a753d922179c087afdaad2740c5e"
        "aaf5b67ed4f2b3960eadbdfeb1aff5d8f8ac544cdd32041518c1f7fb9e2ad");
    const B b = zlib_inflate(dyn);
    ASSERT_EQ(b.size(), 600u);
    const std::uint32_t want = (std::uint32_t(dyn[dyn.size() - 4]) << 24) | (std::uint32_t(dyn[dyn.size() - 3]) << 16) |
                               (std::uint32_t(dyn[dyn.size() - 2]) << 8) | dyn[dyn.size() - 1];
    EXPECT_EQ(adler32(b), want);
    // Stored blocks, and garbage refused.
    const B text(70000, 'x');
    EXPECT_EQ(zlib_inflate(zlib_stored(text)), text);
    EXPECT_TRUE(zlib_inflate(hex("78da0102030405060708")).empty());
    EXPECT_TRUE(zlib_inflate({}).empty());
    // A size limit holds.
    EXPECT_TRUE(zlib_inflate(zlib_stored(text), 1000).empty());
}

TEST(StillMetadata, IccPrimariesAreRecognisedByTheirColorants) {
    std::string d;
    EXPECT_EQ(icc_primaries(icc(kSrgb, "sRGB IEC61966-2.1"), &d), Primaries::Rec709);
    EXPECT_EQ(d, "sRGB IEC61966-2.1");
    EXPECT_EQ(icc_primaries(icc(kP3, "Display P3")), Primaries::P3D65);
    EXPECT_EQ(icc_primaries(icc(k2020, "ITU-R BT.2020")), Primaries::Rec2020);
    EXPECT_EQ(icc_primaries(icc(kAdobe, "Adobe RGB (1998)")), Primaries::AdobeRgb);
    EXPECT_FALSE(icc_primaries(icc(kOdd, "Camera RGB")));
    EXPECT_FALSE(icc_primaries(icc(kSrgb, "grey", "GRAY")));
    EXPECT_FALSE(icc_primaries(B(40, 0)));
}

TEST(StillMetadata, PngIccCicpSrgbChrmAndExif) {
    // iCCP: name, NUL, method 0, zlib(profile).
    auto iccp = [](const B& profile) {
        B d;
        put_s(d, "P3\0\0", 4);
        const B z = zlib_stored(profile);
        d.insert(d.end(), z.begin(), z.end());
        return d;
    };
    StillColour c = read_still_colour(png_with({{"iCCP", iccp(icc(kP3, "Display P3"))}}));
    EXPECT_TRUE(c.tagged);
    EXPECT_TRUE(c.recognised);
    EXPECT_EQ(c.primaries, Primaries::P3D65);
    EXPECT_EQ(c.profile, "Display P3");
    EXPECT_NE(c.note().find("Display P3"), std::string::npos);

    c = read_still_colour(png_with({{"iCCP", iccp(icc(kOdd, "Camera RGB"))}}));
    EXPECT_TRUE(c.tagged);
    EXPECT_FALSE(c.recognised);
    EXPECT_EQ(c.primaries, Primaries::Rec709);
    EXPECT_NE(c.note().find("not one RUDRA knows"), std::string::npos);

    c = read_still_colour(png_with({{"cICP", B{9, 13, 0, 1}}}));
    EXPECT_EQ(c.primaries, Primaries::Rec2020);
    c = read_still_colour(png_with({{"cICP", B{12, 13, 0, 1}}}));
    EXPECT_EQ(c.primaries, Primaries::P3D65);
    c = read_still_colour(png_with({{"sRGB", B{0}}}));
    EXPECT_TRUE(c.tagged);
    EXPECT_EQ(c.primaries, Primaries::Rec709);

    B chrm;   // white, red, green, blue x / y * 100000: Adobe RGB
    for (unsigned v : {31270u, 32900u, 64000u, 33000u, 21000u, 71000u, 15000u, 6000u}) put32(chrm, v);
    c = read_still_colour(png_with({{"cHRM", chrm}}));
    EXPECT_EQ(c.primaries, Primaries::AdobeRgb);

    c = read_still_colour(png_with({{"eXIf", tiff_orientation(6, false)}}));
    EXPECT_EQ(c.orientation, 6);
    EXPECT_FALSE(c.tagged);
    EXPECT_EQ(c.primaries, Primaries::Rec709);
    // Untagged: Rec.709, as every master before.
    c = read_still_colour(png_with({}));
    EXPECT_FALSE(c.tagged);
    EXPECT_EQ(c.primaries, Primaries::Rec709);
    EXPECT_EQ(c.orientation, 1);
}

TEST(StillMetadata, JpegIccInPartsAndExif) {
    const B profile = icc(kP3, "Display P3");
    B j = {0xFF, 0xD8};
    // APP1 Exif, orientation 8, little-endian TIFF.
    B exif;
    put_s(exif, "Exif\0\0", 6);
    const B t = tiff_orientation(8, true);
    exif.insert(exif.end(), t.begin(), t.end());
    j.push_back(0xFF);
    j.push_back(0xE1);
    put16(j, unsigned(exif.size() + 2));
    j.insert(j.end(), exif.begin(), exif.end());
    // APP2 ICC_PROFILE in two parts, written out of order.
    const std::size_t half = profile.size() / 2;
    for (int part : {2, 1}) {
        B seg;
        put_s(seg, "ICC_PROFILE\0", 12);
        seg.push_back(std::uint8_t(part));
        seg.push_back(2);
        if (part == 1) seg.insert(seg.end(), profile.begin(), profile.begin() + std::ptrdiff_t(half));
        else seg.insert(seg.end(), profile.begin() + std::ptrdiff_t(half), profile.end());
        j.push_back(0xFF);
        j.push_back(0xE2);
        put16(j, unsigned(seg.size() + 2));
        j.insert(j.end(), seg.begin(), seg.end());
    }
    j.push_back(0xFF);
    j.push_back(0xDA);   // start of scan
    const StillColour c = read_still_colour(j);
    EXPECT_EQ(c.orientation, 8);
    EXPECT_EQ(c.primaries, Primaries::P3D65);
    EXPECT_TRUE(c.recognised);
}

TEST(StillMetadata, WebpChunks) {
    const B profile = icc(kAdobe, "Adobe RGB (1998)");
    B w;
    put_s(w, "RIFF", 4);
    put32le(w, 0);
    put_s(w, "WEBP", 4);
    put_s(w, "ICCP", 4);
    put32le(w, std::uint32_t(profile.size()));
    w.insert(w.end(), profile.begin(), profile.end());
    if (profile.size() & 1) w.push_back(0);
    const StillColour c = read_still_colour(w);
    EXPECT_EQ(c.primaries, Primaries::AdobeRgb);
}

TEST(StillMetadata, OrientationsMapEveryPixel) {
    // 3 x 2, one channel, values = stored index: 0 1 2 / 3 4 5.
    const std::vector<float> p = {0, 1, 2, 3, 4, 5};
    const auto o6 = apply_orientation(p, 1, 3, 2, 6);   // rotate 90 clockwise to show
    EXPECT_EQ(o6.width, 2);
    EXPECT_EQ(o6.height, 3);
    EXPECT_EQ(o6.data, (std::vector<float>{3, 0, 4, 1, 5, 2}));
    const auto o8 = apply_orientation(p, 1, 3, 2, 8);   // rotate 90 counter-clockwise
    EXPECT_EQ(o8.data, (std::vector<float>{2, 5, 1, 4, 0, 3}));
    EXPECT_EQ(apply_orientation(p, 1, 3, 2, 3).data, (std::vector<float>{5, 4, 3, 2, 1, 0}));
    EXPECT_EQ(apply_orientation(p, 1, 3, 2, 2).data, (std::vector<float>{2, 1, 0, 5, 4, 3}));
    EXPECT_EQ(apply_orientation(p, 1, 3, 2, 4).data, (std::vector<float>{3, 4, 5, 0, 1, 2}));
    EXPECT_EQ(apply_orientation(p, 1, 3, 2, 5).data, (std::vector<float>{0, 3, 1, 4, 2, 5}));
    EXPECT_EQ(apply_orientation(p, 1, 3, 2, 7).data, (std::vector<float>{5, 2, 4, 1, 3, 0}));
    EXPECT_EQ(apply_orientation(p, 1, 3, 2, 1).data, p);
}
