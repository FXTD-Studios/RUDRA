#include "rudra/media/still_metadata.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace rudra {
namespace {

using Bytes = std::span<const std::uint8_t>;

std::uint32_t be32(const std::uint8_t* p) noexcept {
    return (std::uint32_t(p[0]) << 24) | (std::uint32_t(p[1]) << 16) | (std::uint32_t(p[2]) << 8) | p[3];
}
std::uint16_t be16(const std::uint8_t* p) noexcept { return std::uint16_t((p[0] << 8) | p[1]); }
std::uint32_t le32(const std::uint8_t* p) noexcept {
    return (std::uint32_t(p[3]) << 24) | (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[1]) << 8) | p[0];
}

// ---- inflate (RFC 1951), after Mark Adler's puff: small, bounded, no tables kept

struct Inflater {
    Bytes in;
    std::size_t pos = 0;
    std::uint32_t bitbuf = 0;
    int bitcnt = 0;
    std::vector<std::uint8_t> out;
    std::size_t limit;
    bool bad = false;

    int bits(int need) {
        std::uint32_t val = bitbuf;
        while (bitcnt < need) {
            if (pos >= in.size()) { bad = true; return 0; }
            val |= std::uint32_t(in[pos++]) << bitcnt;
            bitcnt += 8;
        }
        bitbuf = val >> need;
        bitcnt -= need;
        return int(val & ((1u << need) - 1));
    }

    struct Huffman {
        std::array<short, 16> count{};
        std::array<short, 288> symbol{};
    };

    int decode(const Huffman& h) {
        int code = 0, first = 0, index = 0;
        for (int len = 1; len < 16; ++len) {
            code |= bits(1);
            if (bad) return -1;
            const int count = h.count[std::size_t(len)];
            if (code - count < first) return h.symbol[std::size_t(index + (code - first))];
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        bad = true;
        return -1;
    }

    static bool construct(Huffman& h, const short* length, int n) {
        h.count.fill(0);
        for (int s = 0; s < n; ++s) h.count[std::size_t(length[s])]++;
        if (h.count[0] == n) return true;   // no codes: complete, but decoding will fail
        int left = 1;
        for (int len = 1; len < 16; ++len) {
            left <<= 1;
            left -= h.count[std::size_t(len)];
            if (left < 0) return false;   // over-subscribed
        }
        std::array<short, 16> offs{};
        for (int len = 1; len < 15; ++len) offs[std::size_t(len + 1)] = short(offs[std::size_t(len)] + h.count[std::size_t(len)]);
        for (int s = 0; s < n; ++s)
            if (length[s] != 0) h.symbol[std::size_t(offs[std::size_t(length[s])]++)] = short(s);
        return true;
    }

    bool put(std::uint8_t b) {
        if (out.size() >= limit) { bad = true; return false; }
        out.push_back(b);
        return true;
    }

    bool codes(const Huffman& lencode, const Huffman& distcode) {
        static constexpr short lbase[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
                                            35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
        static constexpr short lext[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
        static constexpr short dbase[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129,
                                            193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
        static constexpr short dext[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
        for (;;) {
            int symbol = decode(lencode);
            if (symbol < 0 || bad) return false;
            if (symbol < 256) {
                if (!put(std::uint8_t(symbol))) return false;
            } else if (symbol == 256) {
                return true;
            } else {
                symbol -= 257;
                if (symbol >= 29) return false;
                const int len = lbase[symbol] + bits(lext[symbol]);
                const int ds = decode(distcode);
                if (ds < 0 || ds >= 30 || bad) return false;
                const std::size_t dist = std::size_t(dbase[ds] + bits(dext[ds]));
                if (bad || dist > out.size()) return false;
                for (int i = 0; i < len; ++i)
                    if (!put(out[out.size() - dist])) return false;
            }
        }
    }

    bool stored() {
        bitbuf = 0;
        bitcnt = 0;
        if (pos + 4 > in.size()) return false;
        const unsigned len = unsigned(in[pos]) | (unsigned(in[pos + 1]) << 8);
        const unsigned nlen = unsigned(in[pos + 2]) | (unsigned(in[pos + 3]) << 8);
        pos += 4;
        if (len != (~nlen & 0xffffu) || pos + len > in.size()) return false;
        for (unsigned i = 0; i < len; ++i)
            if (!put(in[pos + i])) return false;
        pos += len;
        return true;
    }

    bool fixed() {
        static Huffman lencode, distcode;
        static bool built = false;
        if (!built) {
            short lengths[288];
            int s = 0;
            for (; s < 144; ++s) lengths[s] = 8;
            for (; s < 256; ++s) lengths[s] = 9;
            for (; s < 280; ++s) lengths[s] = 7;
            for (; s < 288; ++s) lengths[s] = 8;
            construct(lencode, lengths, 288);
            for (s = 0; s < 30; ++s) lengths[s] = 5;
            construct(distcode, lengths, 30);
            built = true;
        }
        return codes(lencode, distcode);
    }

    bool dynamic() {
        static constexpr short order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
        const int nlen = bits(5) + 257, ndist = bits(5) + 1, ncode = bits(4) + 4;
        if (bad || nlen > 286 || ndist > 30) return false;
        short lengths[320] = {};
        for (int i = 0; i < ncode; ++i) lengths[order[i]] = short(bits(3));
        Huffman lencode, distcode;
        if (bad || !construct(lencode, lengths, 19)) return false;
        int index = 0;
        while (index < nlen + ndist) {
            int symbol = decode(lencode);
            if (symbol < 0 || bad) return false;
            if (symbol < 16) {
                lengths[index++] = short(symbol);
            } else {
                int len = 0, rep;
                if (symbol == 16) {
                    if (index == 0) return false;
                    len = lengths[index - 1];
                    rep = 3 + bits(2);
                } else if (symbol == 17) {
                    rep = 3 + bits(3);
                } else {
                    rep = 11 + bits(7);
                }
                if (bad || index + rep > nlen + ndist) return false;
                while (rep--) lengths[index++] = short(len);
            }
        }
        if (lengths[256] == 0) return false;
        if (!construct(lencode, lengths, nlen) || !construct(distcode, lengths + nlen, ndist)) return false;
        return codes(lencode, distcode);
    }

    bool run() {
        int last;
        do {
            last = bits(1);
            const int type = bits(2);
            if (bad) return false;
            const bool ok = type == 0 ? stored() : type == 1 ? fixed() : type == 2 ? dynamic() : false;
            if (!ok || bad) return false;
        } while (!last);
        return true;
    }
};

// ---- ICC

struct Colorants {
    std::array<double, 9> rgb;   // rXYZ, gXYZ, bXYZ
};

// D50-adapted (Bradford) colorants, as ICC v2/v4 profiles store them.
constexpr std::pair<Primaries, Colorants> kKnown[] = {
    {Primaries::Rec709, {{0.4360, 0.2225, 0.0139, 0.3851, 0.7169, 0.0971, 0.1430, 0.0606, 0.7139}}},
    {Primaries::P3D65, {{0.5151, 0.2412, -0.0011, 0.2920, 0.6922, 0.0419, 0.1571, 0.0666, 0.7841}}},
    {Primaries::Rec2020, {{0.6735, 0.2790, -0.0019, 0.1657, 0.6753, 0.0300, 0.1250, 0.0456, 0.7969}}},
    {Primaries::AdobeRgb, {{0.6097, 0.3111, 0.0195, 0.2053, 0.6257, 0.0609, 0.1492, 0.0632, 0.7446}}},
};

std::optional<std::array<double, 3>> icc_xyz(Bytes icc, std::uint32_t off, std::uint32_t size) {
    if (size < 20 || std::size_t(off) + size > icc.size() || std::memcmp(icc.data() + off, "XYZ ", 4) != 0) return std::nullopt;
    std::array<double, 3> v{};
    for (int i = 0; i < 3; ++i) v[std::size_t(i)] = double(std::int32_t(be32(icc.data() + off + 8 + 4 * i))) / 65536.0;
    return v;
}

std::string icc_text(Bytes icc, std::uint32_t off, std::uint32_t size) {
    if (std::size_t(off) + size > icc.size() || size < 12) return {};
    const std::uint8_t* p = icc.data() + off;
    if (std::memcmp(p, "desc", 4) == 0) {   // v2 textDescriptionType: ASCII count, then the text
        const std::uint32_t n = be32(p + 8);
        if (12 + std::size_t(n) > size) return {};
        std::string s(reinterpret_cast<const char*>(p + 12), n);
        while (!s.empty() && s.back() == '\0') s.pop_back();
        return s;
    }
    if (std::memcmp(p, "mluc", 4) == 0 && size >= 28) {   // v4: the first record, UTF-16BE, kept to ASCII
        const std::uint32_t len = be32(p + 20), at = be32(p + 24);
        if (std::size_t(at) + len > size) return {};
        std::string s;
        for (std::uint32_t i = 0; i + 1 < len; i += 2) {
            const std::uint16_t u = be16(p + at + i);
            s.push_back(u < 0x80 ? char(u) : '?');
        }
        return s;
    }
    return {};
}

int tiff_orientation(Bytes t) {
    if (t.size() < 8) return 1;
    const bool le = t[0] == 'I' && t[1] == 'I';
    if (!le && !(t[0] == 'M' && t[1] == 'M')) return 1;
    auto u16 = [&](std::size_t o) -> unsigned { return le ? unsigned(t[o] | (t[o + 1] << 8)) : unsigned(be16(t.data() + o)); };
    auto u32 = [&](std::size_t o) -> std::uint32_t { return le ? le32(t.data() + o) : be32(t.data() + o); };
    const std::uint32_t ifd = u32(4);
    if (std::size_t(ifd) + 2 > t.size()) return 1;
    const unsigned n = u16(ifd);
    for (unsigned i = 0; i < n; ++i) {
        const std::size_t e = std::size_t(ifd) + 2 + 12 * i;
        if (e + 12 > t.size()) break;
        if (u16(e) == 0x0112) {
            const unsigned v = u16(e + 8);
            return v >= 1 && v <= 8 ? int(v) : 1;
        }
    }
    return 1;
}

// TIFF IFD0 tag 34675 (InterColorProfile): its bytes.
Bytes tiff_icc(Bytes t) {
    if (t.size() < 8) return {};
    const bool le = t[0] == 'I' && t[1] == 'I';
    auto u16 = [&](std::size_t o) -> unsigned { return le ? unsigned(t[o] | (t[o + 1] << 8)) : unsigned(be16(t.data() + o)); };
    auto u32 = [&](std::size_t o) -> std::uint32_t { return le ? le32(t.data() + o) : be32(t.data() + o); };
    const std::uint32_t ifd = u32(4);
    if (std::size_t(ifd) + 2 > t.size()) return {};
    const unsigned n = u16(ifd);
    for (unsigned i = 0; i < n; ++i) {
        const std::size_t e = std::size_t(ifd) + 2 + 12 * i;
        if (e + 12 > t.size()) break;
        if (u16(e) == 34675) {
            const std::uint32_t count = u32(e + 4), off = u32(e + 8);
            if (std::size_t(off) + count <= t.size()) return t.subspan(off, count);
        }
    }
    return {};
}

std::optional<Primaries> classify_xy(const std::array<double, 8>& c) {   // rx ry gx gy bx by wx wy
    struct K { Primaries p; double v[8]; };
    static constexpr K known[] = {
        {Primaries::Rec709, {0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290}},
        {Primaries::P3D65, {0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.3127, 0.3290}},
        {Primaries::Rec2020, {0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290}},
        {Primaries::AdobeRgb, {0.64, 0.33, 0.21, 0.71, 0.15, 0.06, 0.3127, 0.3290}},
    };
    for (const auto& k : known) {
        bool ok = true;
        for (int i = 0; i < 8 && ok; ++i) ok = std::abs(c[std::size_t(i)] - k.v[i]) <= 0.005;
        if (ok) return k.p;
    }
    return std::nullopt;
}

void from_icc(StillColour& out, Bytes icc) {
    out.tagged = true;
    out.tag = "icc";
    std::string desc;
    const auto p = icc_primaries(icc, &desc);
    out.profile = desc;
    if (p) {
        out.primaries = *p;
        out.recognised = true;
    } else {
        out.primaries = Primaries::Rec709;
        out.recognised = false;
    }
}

void read_png(Bytes b, StillColour& out) {
    std::size_t pos = 8;
    bool have_icc = false, have_cicp = false;
    while (pos + 12 <= b.size()) {
        const std::uint32_t len = be32(b.data() + pos);
        const std::uint8_t* type = b.data() + pos + 4;
        if (pos + 12 + std::size_t(len) > b.size()) break;
        const Bytes data = b.subspan(pos + 8, len);
        if (std::memcmp(type, "cICP", 4) == 0 && len >= 4) {
            have_cicp = true;
            out.tagged = true;
            out.tag = "cicp";
            const int cp = data[0];
            out.recognised = cp == 1 || cp == 9 || cp == 12;
            out.primaries = cp == 9 ? Primaries::Rec2020 : cp == 12 ? Primaries::P3D65 : Primaries::Rec709;
            out.profile = "cICP colour primaries " + std::to_string(cp);
        } else if (std::memcmp(type, "iCCP", 4) == 0 && !have_cicp) {
            const auto nul = std::find(data.begin(), data.end(), std::uint8_t(0));
            if (nul != data.end() && std::size_t(nul - data.begin()) + 2 <= data.size()) {
                const std::size_t at = std::size_t(nul - data.begin()) + 2;   // name, NUL, method byte
                const auto icc = zlib_inflate(data.subspan(at));
                if (!icc.empty()) {
                    from_icc(out, icc);
                    have_icc = true;
                } else {
                    out.tagged = true;
                    out.tag = "icc";
                    out.recognised = false;
                    out.profile = "unreadable iCCP";
                }
            }
        } else if (std::memcmp(type, "sRGB", 4) == 0 && !have_icc && !have_cicp) {
            out.tagged = true;
            out.tag = "srgb";
            out.primaries = Primaries::Rec709;
            out.recognised = true;
            out.profile = "sRGB";
        } else if (std::memcmp(type, "cHRM", 4) == 0 && len >= 32 && !have_icc && !have_cicp && out.tag != "srgb") {
            std::array<double, 8> c{};
            const int order[8] = {2, 3, 4, 5, 6, 7, 0, 1};   // file: white, red, green, blue
            for (int i = 0; i < 8; ++i) c[std::size_t(i)] = double(be32(data.data() + 4 * order[i])) / 100000.0;
            out.tagged = true;
            out.tag = "chrm";
            const auto p = classify_xy(c);
            out.recognised = bool(p);
            out.primaries = p.value_or(Primaries::Rec709);
            out.profile = "cHRM chromaticities";
        } else if (std::memcmp(type, "eXIf", 4) == 0) {
            out.orientation = tiff_orientation(data);
        } else if (std::memcmp(type, "IDAT", 4) == 0 || std::memcmp(type, "IEND", 4) == 0) {
            break;   // colour chunks come before the image data
        }
        pos += 12 + std::size_t(len);
    }
}

void read_jpeg(Bytes b, StillColour& out) {
    std::size_t pos = 2;
    std::vector<std::pair<int, std::vector<std::uint8_t>>> icc_parts;
    while (pos + 4 <= b.size() && b[pos] == 0xFF) {
        const std::uint8_t marker = b[pos + 1];
        if (marker == 0xD8 || (marker >= 0xD0 && marker <= 0xD7) || marker == 0x01) { pos += 2; continue; }
        if (marker == 0xDA || marker == 0xD9) break;   // start of scan: no more metadata
        const std::size_t len = be16(b.data() + pos + 2);
        if (len < 2 || pos + 2 + len > b.size()) break;
        const Bytes seg = b.subspan(pos + 4, len - 2);
        if (marker == 0xE1 && seg.size() > 6 && std::memcmp(seg.data(), "Exif\0\0", 6) == 0) {
            out.orientation = tiff_orientation(seg.subspan(6));
        } else if (marker == 0xE2 && seg.size() > 14 && std::memcmp(seg.data(), "ICC_PROFILE\0", 12) == 0) {
            icc_parts.push_back({seg[12], std::vector<std::uint8_t>(seg.begin() + 14, seg.end())});
        }
        pos += 2 + len;
    }
    if (!icc_parts.empty()) {
        std::sort(icc_parts.begin(), icc_parts.end(), [](const auto& a, const auto& c) { return a.first < c.first; });
        std::vector<std::uint8_t> icc;
        for (const auto& [n, part] : icc_parts) icc.insert(icc.end(), part.begin(), part.end());
        from_icc(out, icc);
    }
}

void read_webp(Bytes b, StillColour& out) {
    std::size_t pos = 12;
    while (pos + 8 <= b.size()) {
        const std::uint32_t len = le32(b.data() + pos + 4);
        if (pos + 8 + std::size_t(len) > b.size()) break;
        const Bytes data = b.subspan(pos + 8, len);
        if (std::memcmp(b.data() + pos, "ICCP", 4) == 0) from_icc(out, data);
        if (std::memcmp(b.data() + pos, "EXIF", 4) == 0) {
            const Bytes t = data.size() > 6 && std::memcmp(data.data(), "Exif\0\0", 6) == 0 ? data.subspan(6) : data;
            out.orientation = tiff_orientation(t);
        }
        pos += 8 + std::size_t(len) + (len & 1u);
    }
}

}  // namespace

const char* primaries_id(Primaries p) noexcept {
    switch (p) {
        case Primaries::Rec709: return "rec709";
        case Primaries::Rec2020: return "rec2020";
        case Primaries::P3D65: return "p3d65";
        case Primaries::AdobeRgb: return "adobe_rgb";
        case Primaries::Ap0: return "ap0";
        case Primaries::Ap1: return "ap1";
        case Primaries::Awg4: return "awg4";
    }
    return "rec709";
}

std::string StillColour::note() const {
    static const char* names[] = {"Rec.709 / sRGB", "Rec.2020", "Display P3", "ACES AP0", "ACES AP1", "ARRI Wide Gamut 4",
                                  "Adobe RGB (1998)"};
    const char* name = names[std::size_t(primaries)];
    std::string s;
    if (!tagged) s = "no colour profile: read as Rec.709 / sRGB";
    else if (!recognised)
        s = "colour profile '" + (profile.empty() ? std::string("unnamed") : profile) +
            "' is not one RUDRA knows: read as Rec.709 / sRGB, check the colours";
    else s = std::string(name) + " (" + (profile.empty() ? tag : profile) + "): mastered from " + name;
    if (orientation != 1) s += "; EXIF orientation " + std::to_string(orientation) + " applied";
    return s;
}

std::vector<std::uint8_t> zlib_inflate(Bytes data, std::size_t limit) {
    if (data.size() < 6) return {};
    const unsigned cmf = data[0], flg = data[1];
    if ((cmf & 0x0f) != 8 || ((cmf << 8) | flg) % 31 != 0 || (flg & 0x20)) return {};   // deflate, checksum, no dictionary
    Inflater inf{data.subspan(2), 0, 0, 0, {}, limit};
    if (!inf.run()) return {};
    return std::move(inf.out);
}

std::optional<Primaries> icc_primaries(Bytes icc, std::string* description) {
    if (icc.size() < 132 || std::memcmp(icc.data() + 36, "acsp", 4) != 0) return std::nullopt;
    const std::uint32_t n = be32(icc.data() + 128);
    std::optional<std::array<double, 3>> r, g, b;
    for (std::uint32_t i = 0; i < n && 132 + 12 * std::size_t(i + 1) <= icc.size(); ++i) {
        const std::uint8_t* e = icc.data() + 132 + 12 * i;
        const std::uint32_t off = be32(e + 4), size = be32(e + 8);
        if (std::memcmp(e, "rXYZ", 4) == 0) r = icc_xyz(icc, off, size);
        else if (std::memcmp(e, "gXYZ", 4) == 0) g = icc_xyz(icc, off, size);
        else if (std::memcmp(e, "bXYZ", 4) == 0) b = icc_xyz(icc, off, size);
        else if (std::memcmp(e, "desc", 4) == 0 && description) *description = icc_text(icc, off, size);
    }
    if (std::memcmp(icc.data() + 16, "RGB ", 4) != 0 || !r || !g || !b) return std::nullopt;
    const std::array<double, 9> got = {(*r)[0], (*r)[1], (*r)[2], (*g)[0], (*g)[1], (*g)[2], (*b)[0], (*b)[1], (*b)[2]};
    for (const auto& [p, k] : kKnown) {
        bool ok = true;
        for (std::size_t i = 0; i < 9 && ok; ++i) ok = std::abs(got[i] - k.rgb[i]) <= 0.01;
        if (ok) return p;
    }
    return std::nullopt;
}

StillColour read_still_colour(Bytes b) {
    StillColour out;
    if (b.size() >= 8 && std::memcmp(b.data(), "\x89PNG\r\n\x1a\n", 8) == 0) read_png(b, out);
    else if (b.size() >= 4 && b[0] == 0xFF && b[1] == 0xD8) read_jpeg(b, out);
    else if (b.size() >= 8 && ((b[0] == 'I' && b[1] == 'I' && b[2] == 42) || (b[0] == 'M' && b[1] == 'M' && b[3] == 42))) {
        out.orientation = tiff_orientation(b);
        if (const Bytes icc = tiff_icc(b); !icc.empty()) from_icc(out, icc);
    } else if (b.size() >= 12 && std::memcmp(b.data(), "RIFF", 4) == 0 && std::memcmp(b.data() + 8, "WEBP", 4) == 0) {
        read_webp(b, out);
    }
    return out;
}

Oriented apply_orientation(std::span<const float> planes, int channels, int w, int h, int o) {
    Oriented r;
    const bool swap = o >= 5 && o <= 8;
    r.width = swap ? h : w;
    r.height = swap ? w : h;
    r.data.resize(planes.size());
    for (int c = 0; c < channels; ++c)
        for (int y = 0; y < r.height; ++y)
            for (int x = 0; x < r.width; ++x) {
                int sx = x, sy = y;   // the stored pixel shown at (x, y)
                switch (o) {
                    case 2: sx = w - 1 - x; break;
                    case 3: sx = w - 1 - x; sy = h - 1 - y; break;
                    case 4: sy = h - 1 - y; break;
                    case 5: sx = y; sy = x; break;
                    case 6: sx = y; sy = h - 1 - x; break;
                    case 7: sx = w - 1 - y; sy = h - 1 - x; break;
                    case 8: sx = w - 1 - y; sy = x; break;
                    default: break;
                }
                r.data[(std::size_t(c) * r.height + y) * r.width + x] = planes[(std::size_t(c) * h + sy) * w + sx];
            }
    return r;
}

}  // namespace rudra
