#include "rudra/platform/png8.hpp"

#include <array>
#include <cstring>
#include <fstream>

namespace rudra {
namespace {

std::uint32_t crc_table_entry(std::uint32_t n) {
    for (int k = 0; k < 8; ++k) n = (n & 1) ? 0xEDB88320u ^ (n >> 1) : n >> 1;
    return n;
}

std::uint32_t crc32(const std::uint8_t* p, std::size_t n, std::uint32_t crc = 0xFFFFFFFFu) {
    static const auto table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i) t[i] = crc_table_entry(i);
        return t;
    }();
    for (std::size_t i = 0; i < n; ++i) crc = table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return crc;
}

std::uint32_t adler32(const std::uint8_t* p, std::size_t n) {
    std::uint32_t a = 1, b = 0;
    for (std::size_t i = 0; i < n; ++i) {
        a = (a + p[i]) % 65521u;
        b = (b + a) % 65521u;
    }
    return (b << 16) | a;
}

void put_u32(std::vector<std::uint8_t>& o, std::uint32_t v) {
    o.push_back(std::uint8_t(v >> 24));
    o.push_back(std::uint8_t(v >> 16));
    o.push_back(std::uint8_t(v >> 8));
    o.push_back(std::uint8_t(v));
}

void chunk(std::vector<std::uint8_t>& o, const char* type, const std::vector<std::uint8_t>& body) {
    put_u32(o, std::uint32_t(body.size()));
    std::vector<std::uint8_t> tb(type, type + 4);
    tb.insert(tb.end(), body.begin(), body.end());
    o.insert(o.end(), tb.begin(), tb.end());
    put_u32(o, crc32(tb.data(), tb.size()) ^ 0xFFFFFFFFu);
}

// ---- inflate -------------------------------------------------------------
struct BitReader {
    std::span<const std::uint8_t> in;
    std::size_t pos = 0;
    std::uint32_t bitbuf = 0;
    int bitcnt = 0;
    bool bad = false;
    std::uint32_t bits(int n) {
        while (bitcnt < n) {
            if (pos >= in.size()) { bad = true; return 0; }
            bitbuf |= std::uint32_t(in[pos++]) << bitcnt;
            bitcnt += 8;
        }
        const std::uint32_t v = bitbuf & ((1u << n) - 1u);
        bitbuf >>= n;
        bitcnt -= n;
        return v;
    }
    void align() { bitbuf = 0; bitcnt = 0; }
};

struct Huffman {
    std::array<std::uint16_t, 16> count{};
    std::vector<std::uint16_t> symbol;
    bool build(const std::uint8_t* lengths, int n) {
        count.fill(0);
        for (int i = 0; i < n; ++i) ++count[lengths[i]];
        count[0] = 0;
        std::array<std::uint16_t, 16> offs{};
        for (int i = 1; i < 16; ++i) offs[i] = std::uint16_t(offs[i - 1] + count[i - 1]);
        symbol.assign(std::size_t(n), 0);
        for (int i = 0; i < n; ++i)
            if (lengths[i]) symbol[offs[lengths[i]]++] = std::uint16_t(i);
        return true;
    }
    int decode(BitReader& br) const {
        int code = 0, first = 0, index = 0;
        for (int len = 1; len < 16; ++len) {
            code |= int(br.bits(1));
            const int c = count[len];
            if (code - c < first) return symbol[std::size_t(index + (code - first))];
            index += c;
            first += c;
            first <<= 1;
            code <<= 1;
            if (br.bad) return -1;
        }
        return -1;
    }
};

bool inflate(std::span<const std::uint8_t> zlib, std::vector<std::uint8_t>& out) {
    if (zlib.size() < 6) return false;
    BitReader br{zlib.subspan(2)};   // past the zlib header
    static const std::uint16_t lbase[] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
    static const std::uint16_t lext[] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
    static const std::uint16_t dbase[] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
    static const std::uint16_t dext[] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
    bool last = false;
    while (!last) {
        last = br.bits(1) != 0;
        const std::uint32_t type = br.bits(2);
        if (br.bad) return false;
        if (type == 0) {
            br.align();
            if (br.pos + 4 > br.in.size()) return false;
            const std::uint32_t len = br.in[br.pos] | (std::uint32_t(br.in[br.pos + 1]) << 8);
            const std::uint32_t nlen = br.in[br.pos + 2] | (std::uint32_t(br.in[br.pos + 3]) << 8);
            br.pos += 4;
            if ((len ^ 0xFFFFu) != nlen || br.pos + len > br.in.size()) return false;
            out.insert(out.end(), br.in.begin() + std::ptrdiff_t(br.pos), br.in.begin() + std::ptrdiff_t(br.pos + len));
            br.pos += len;
            continue;
        }
        Huffman lit, dist;
        if (type == 1) {
            std::uint8_t l[288], d[30];
            int i = 0;
            for (; i < 144; ++i) l[i] = 8;
            for (; i < 256; ++i) l[i] = 9;
            for (; i < 280; ++i) l[i] = 7;
            for (; i < 288; ++i) l[i] = 8;
            for (i = 0; i < 30; ++i) d[i] = 5;
            lit.build(l, 288);
            dist.build(d, 30);
        } else if (type == 2) {
            const int nlen = int(br.bits(5)) + 257, ndist = int(br.bits(5)) + 1, ncode = int(br.bits(4)) + 4;
            static const int order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
            std::uint8_t lengths[320] = {};
            for (int i = 0; i < ncode; ++i) lengths[order[i]] = std::uint8_t(br.bits(3));
            Huffman lencode;
            lencode.build(lengths, 19);
            int idx = 0;
            std::memset(lengths, 0, sizeof lengths);
            while (idx < nlen + ndist) {
                int sym = lencode.decode(br);
                if (sym < 0) return false;
                if (sym < 16) lengths[idx++] = std::uint8_t(sym);
                else {
                    int len = 0, rep;
                    if (sym == 16) {
                        if (idx == 0) return false;
                        len = lengths[idx - 1];
                        rep = 3 + int(br.bits(2));
                    } else if (sym == 17) rep = 3 + int(br.bits(3));
                    else rep = 11 + int(br.bits(7));
                    if (idx + rep > nlen + ndist) return false;
                    while (rep--) lengths[idx++] = std::uint8_t(len);
                }
            }
            lit.build(lengths, nlen);
            dist.build(lengths + nlen, ndist);
        } else return false;
        for (;;) {
            int sym = lit.decode(br);
            if (sym < 0 || br.bad) return false;
            if (sym < 256) out.push_back(std::uint8_t(sym));
            else if (sym == 256) break;
            else {
                sym -= 257;
                if (sym >= 29) return false;
                const int len = lbase[sym] + int(br.bits(lext[sym]));
                const int ds = dist.decode(br);
                if (ds < 0 || ds >= 30) return false;
                const std::size_t d = dbase[ds] + br.bits(dext[ds]);
                if (d > out.size()) return false;
                const std::size_t start = out.size() - d;
                for (int i = 0; i < len; ++i) out.push_back(out[start + std::size_t(i)]);
            }
        }
    }
    return true;
}

int paeth(int a, int b, int c) {
    const int p = a + b - c, pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
    return (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
}

}  // namespace

std::vector<std::uint8_t> png8_bytes(const Png8& img) {
    const std::size_t row = std::size_t(img.width) * std::size_t(img.channels);
    std::vector<std::uint8_t> raw;
    raw.reserve((row + 1) * std::size_t(img.height));
    for (int y = 0; y < img.height; ++y) {
        raw.push_back(0);   // filter none
        raw.insert(raw.end(), img.data.begin() + std::ptrdiff_t(row * std::size_t(y)),
                   img.data.begin() + std::ptrdiff_t(row * std::size_t(y + 1)));
    }
    std::vector<std::uint8_t> z{0x78, 0x01};
    std::size_t pos = 0;
    do {
        const std::size_t len = std::min<std::size_t>(65535, raw.size() - pos);
        const bool last = pos + len == raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back(std::uint8_t(len));
        z.push_back(std::uint8_t(len >> 8));
        z.push_back(std::uint8_t(~len));
        z.push_back(std::uint8_t((~len) >> 8));
        z.insert(z.end(), raw.begin() + std::ptrdiff_t(pos), raw.begin() + std::ptrdiff_t(pos + len));
        pos += len;
    } while (pos < raw.size());
    put_u32(z, adler32(raw.data(), raw.size()));

    std::vector<std::uint8_t> o{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<std::uint8_t> ihdr;
    put_u32(ihdr, std::uint32_t(img.width));
    put_u32(ihdr, std::uint32_t(img.height));
    static const std::uint8_t colour[5] = {0, 0, 4, 2, 6};
    ihdr.insert(ihdr.end(), {8, colour[img.channels], 0, 0, 0});
    chunk(o, "IHDR", ihdr);
    chunk(o, "IDAT", z);
    chunk(o, "IEND", {});
    return o;
}

Result<void> write_png8(const std::filesystem::path& path, const Png8& img) {
    if (img.width <= 0 || img.height <= 0 || img.channels < 1 || img.channels > 4 ||
        img.data.size() != std::size_t(img.width) * std::size_t(img.height) * std::size_t(img.channels))
        return make_error(ErrorCode::InvalidArgument, "PNG image is not 8-bit with 1 to 4 channels at its size", path.string());
    const auto bytes = png8_bytes(img);
    std::ofstream out(path, std::ios::binary);
    if (!out) return make_error(ErrorCode::IoError, "The PNG could not be written.", path.string());
    out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    if (!out) return make_error(ErrorCode::IoError, "The PNG could not be written.", path.string());
    return {};
}

Result<Png8> decode_png8(std::span<const std::uint8_t> b) {
    auto bad = [&](const char* why) { return make_error(ErrorCode::ParseError, "Not a PNG this reader takes.", why); };
    static const std::uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (b.size() < 8 || std::memcmp(b.data(), sig, 8) != 0) return bad("signature");
    std::size_t pos = 8;
    Png8 img;
    std::vector<std::uint8_t> idat;
    int bit_depth = 0, colour = -1, interlace = 0;
    auto u32 = [&](std::size_t at) {
        return (std::uint32_t(b[at]) << 24) | (std::uint32_t(b[at + 1]) << 16) | (std::uint32_t(b[at + 2]) << 8) | b[at + 3];
    };
    while (pos + 12 <= b.size()) {
        const std::uint32_t len = u32(pos);
        const char* type = reinterpret_cast<const char*>(b.data() + pos + 4);
        if (pos + 12 + len > b.size()) return bad("truncated chunk");
        const std::uint8_t* body = b.data() + pos + 8;
        if (std::memcmp(type, "IHDR", 4) == 0) {
            if (len != 13) return bad("IHDR");
            img.width = int(u32(pos + 8));
            img.height = int(u32(pos + 12));
            bit_depth = body[8];
            colour = body[9];
            interlace = body[12];
        } else if (std::memcmp(type, "IDAT", 4) == 0) {
            idat.insert(idat.end(), body, body + len);
        } else if (std::memcmp(type, "IEND", 4) == 0) break;
        pos += 12 + len;
    }
    if (bit_depth != 8 || interlace != 0) return bad("8-bit non-interlaced only");
    switch (colour) {
        case 0: img.channels = 1; break;
        case 4: img.channels = 2; break;
        case 2: img.channels = 3; break;
        case 6: img.channels = 4; break;
        default: return bad("palette");
    }
    if (img.width <= 0 || img.height <= 0) return bad("size");
    std::vector<std::uint8_t> raw;
    if (!inflate(idat, raw)) return bad("deflate stream");
    const std::size_t row = std::size_t(img.width) * std::size_t(img.channels);
    if (raw.size() != (row + 1) * std::size_t(img.height)) return bad("data size");
    img.data.assign(row * std::size_t(img.height), 0);
    const int bpp = img.channels;
    for (int y = 0; y < img.height; ++y) {
        const std::uint8_t filter = raw[(row + 1) * std::size_t(y)];
        const std::uint8_t* in = raw.data() + (row + 1) * std::size_t(y) + 1;
        std::uint8_t* out = img.data.data() + row * std::size_t(y);
        const std::uint8_t* up = y ? out - row : nullptr;
        for (std::size_t i = 0; i < row; ++i) {
            const int a = i >= std::size_t(bpp) ? out[i - std::size_t(bpp)] : 0;
            const int bb = up ? up[i] : 0;
            const int c = (up && i >= std::size_t(bpp)) ? up[i - std::size_t(bpp)] : 0;
            int v = in[i];
            switch (filter) {
                case 0: break;
                case 1: v += a; break;
                case 2: v += bb; break;
                case 3: v += (a + bb) / 2; break;
                case 4: v += paeth(a, bb, c); break;
                default: return bad("filter");
            }
            out[i] = std::uint8_t(v);
        }
    }
    return img;
}

Result<Png8> read_png8(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return make_error(ErrorCode::NotFound, "The PNG could not be opened.", path.string());
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto r = decode_png8(bytes);
    if (!r) return make_error(r.error().code, r.error().message, path.string() + ": " + r.error().detail);
    return r;
}

}  // namespace rudra
