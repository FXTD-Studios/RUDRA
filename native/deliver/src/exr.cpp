#include "rudra/deliver/exr.hpp"

#include "rudra/core/timecode.hpp"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>

#include "exr_openexr.hpp"
#include "rudra/core/gamut.hpp"
#include "rudra/platform/io_error.hpp"
#include "rudra/platform/pyjson.hpp"

#if RUDRA_HAVE_ZLIB
#include <zlib.h>
#endif

namespace rudra {

const Chromaticities kAp0Chromaticities{0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077, 0.32168, 0.33767};
const Chromaticities kAp1Chromaticities{0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767};
const Chromaticities kRec2020Chromaticities{0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290};
const Chromaticities kRec709Chromaticities{0.640, 0.330, 0.300, 0.600, 0.150, 0.060, 0.3127, 0.3290};

namespace {

constexpr std::int32_t kMagic = 20000630;

void put_i32(std::vector<std::uint8_t>& o, std::int32_t v) {
    const auto u = static_cast<std::uint32_t>(v);
    for (int k = 0; k < 4; ++k) o.push_back(std::uint8_t(u >> (8 * k)));
}
void put_u64(std::vector<std::uint8_t>& o, std::uint64_t v) {
    for (int k = 0; k < 8; ++k) o.push_back(std::uint8_t(v >> (8 * k)));
}
void put_f32(std::vector<std::uint8_t>& o, float f) { put_i32(o, std::bit_cast<std::int32_t>(f)); }
void put_str(std::vector<std::uint8_t>& o, const std::string& s) {
    o.insert(o.end(), s.begin(), s.end());
    o.push_back(0);
}
void attr(std::vector<std::uint8_t>& o, const std::string& name, const std::string& type,
          const std::vector<std::uint8_t>& payload) {
    put_str(o, name);
    put_str(o, type);
    put_i32(o, static_cast<std::int32_t>(payload.size()));
    o.insert(o.end(), payload.begin(), payload.end());
}

}  // namespace

std::uint16_t float_to_half(float f) noexcept {
    const std::uint32_t x = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    const std::uint32_t abs = x & 0x7fffffffu;
    if (abs >= 0x7f800000u) {   // inf or nan
        return std::uint16_t(sign | (abs > 0x7f800000u ? 0x7e00u : 0x7c00u));
    }
    if (abs >= 0x477ff000u) return std::uint16_t(sign | 0x7c00u);   // rounds to inf
    if (abs < 0x38800000u) {   // half subnormal or zero
        if (abs < 0x33000000u) return std::uint16_t(sign);   // below half the smallest subnormal
        const int e = int(abs >> 23);
        const std::uint32_t mant = (abs & 0x7fffffu) | 0x800000u;
        const int shift = 126 - e;   // the value in units of 2^-24 is mant >> shift
        const std::uint32_t q = mant >> shift;
        const std::uint32_t rem = mant & ((1u << (shift)) - 1u);
        const std::uint32_t half = 1u << (shift - 1);
        std::uint32_t r = q;
        if (rem > half || (rem == half && (q & 1u))) ++r;
        return std::uint16_t(sign | r);
    }
    // Normal: rebias exponent, round the 13 dropped mantissa bits to nearest even.
    std::uint32_t h = ((abs - 0x38000000u) >> 13);
    const std::uint32_t rem = abs & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return std::uint16_t(sign | h);
}

bool exr_zip_available() noexcept { return RUDRA_HAVE_ZLIB != 0; }

const std::vector<ExrCompression>& exr_compressions() {
    static const std::vector<ExrCompression> all = {ExrCompression::Zip,   ExrCompression::Zips, ExrCompression::Piz,
                                                    ExrCompression::Rle,   ExrCompression::None, ExrCompression::Dwaa,
                                                    ExrCompression::Dwab,  ExrCompression::B44,  ExrCompression::B44a,
                                                    ExrCompression::Pxr24};
    return all;
}

bool exr_compression_available(ExrCompression c) noexcept {
    switch (c) {
        case ExrCompression::None: return true;
        case ExrCompression::Zip:
        case ExrCompression::Zips: return exr_zip_available();
        default: return exr_detail::openexr_available();
    }
}

bool exr_compression_lossy(ExrCompression c) noexcept {
    return c == ExrCompression::Pxr24 || c == ExrCompression::B44 || c == ExrCompression::B44a || c == ExrCompression::Dwaa ||
           c == ExrCompression::Dwab;
}

const char* exr_compression_id(ExrCompression c) noexcept {
    switch (c) {
        case ExrCompression::None: return "none";
        case ExrCompression::Zip: return "zip";
        case ExrCompression::Zips: return "zips";
        case ExrCompression::Rle: return "rle";
        case ExrCompression::Piz: return "piz";
        case ExrCompression::Pxr24: return "pxr24";
        case ExrCompression::B44: return "b44";
        case ExrCompression::B44a: return "b44a";
        case ExrCompression::Dwaa: return "dwaa";
        case ExrCompression::Dwab: return "dwab";
    }
    return "none";
}

const char* exr_compression_label(ExrCompression c) noexcept {
    switch (c) {
        case ExrCompression::None: return "uncompressed";
        case ExrCompression::Zip: return "ZIP";
        case ExrCompression::Zips: return "ZIPS";
        case ExrCompression::Rle: return "RLE";
        case ExrCompression::Piz: return "PIZ";
        case ExrCompression::Pxr24: return "PXR24";
        case ExrCompression::B44: return "B44";
        case ExrCompression::B44a: return "B44A";
        case ExrCompression::Dwaa: return "DWAA";
        case ExrCompression::Dwab: return "DWAB";
    }
    return "uncompressed";
}

std::optional<ExrCompression> parse_exr_compression(const std::string& id) {
    std::string l = id;
    for (char& ch : l) ch = char(std::tolower(static_cast<unsigned char>(ch)));
    for (ExrCompression c : exr_compressions())
        if (l == exr_compression_id(c)) return c;
    return std::nullopt;
}

ExrCompression exr_compression_from(const std::string& id, bool zip, int zip_lines) noexcept {
    if (!id.empty())
        if (const auto c = parse_exr_compression(id)) return *c;
    return !zip ? ExrCompression::None : zip_lines == 1 ? ExrCompression::Zips : ExrCompression::Zip;
}

namespace {
// The bytes of a scanline EXR whose channel i is `names[i]`, read from `planes[i]`
// (w x h floats each), written in name order as the format asks; uncompressed, or ZIP.
std::vector<std::uint8_t> planes_bytes(const std::vector<std::string>& names, const std::vector<const float*>& planes,
                                       int w, int h, bool half, const std::optional<Chromaticities>& chroma,
                                       const ExrAttributes& attributes, const ExrTypedAttributes& typed,
                                       ExrCompression compression = ExrCompression::None);

constexpr int kZipLines = 16;   // OpenEXR's ZIP_COMPRESSION: 16 scanlines a block

// What RUDRA's own writer does: None, ZIP and ZIPS (a build without zlib writes ZIP and ZIPS uncompressed, as before).
bool own_writer(ExrCompression c) { return c == ExrCompression::None || c == ExrCompression::Zip || c == ExrCompression::Zips; }

// The value as it goes into the file: NaN 0, +inf the largest float, -inf 0, half clipped to its range.
float clean_value(float v, bool half) {
    if (std::isnan(v)) v = 0.0f;
    else if (v == std::numeric_limits<float>::infinity()) v = 3.4e38f;
    else if (v == -std::numeric_limits<float>::infinity()) v = 0.0f;
    if (half) v = std::clamp(v, -65504.0f, 65504.0f);
    return v;
}

// The file's bytes: RUDRA's own writer for None / ZIP / ZIPS, the OpenEXR library for every other compression (1.3.2),
// with the values cleaned the same way.
Result<std::vector<std::uint8_t>> encode_planes(const std::vector<std::string>& names, const std::vector<const float*>& planes,
                                                int w, int h, bool half, const std::optional<Chromaticities>& chroma,
                                                const ExrAttributes& attributes, const ExrTypedAttributes& typed,
                                                ExrCompression compression) {
    if (own_writer(compression)) return planes_bytes(names, planes, w, h, half, chroma, attributes, typed, compression);
    if (!exr_compression_available(compression))
        return make_error(ErrorCode::Unsupported, std::string("This build of RUDRA writes no ") + exr_compression_label(compression) +
                                                      " EXR (it has no OpenEXR library): choose ZIP, ZIPS or none.");
    const std::size_t n = std::size_t(w) * std::size_t(h);
    std::vector<std::vector<float>> cleaned(planes.size());
    std::vector<const float*> ptrs;
    for (std::size_t i = 0; i < planes.size(); ++i) {
        cleaned[i].resize(n);
        for (std::size_t k = 0; k < n; ++k) cleaned[i][k] = clean_value(planes[i][k], half);
        ptrs.push_back(cleaned[i].data());
    }
    return exr_detail::openexr_encode(names, ptrs, w, h, half, chroma, attributes, typed, compression);
}

// OpenEXR's ZIP (ImfZip.cpp): the bytes split into even and odd halves, then each a delta of the
// one before (+128), then zlib. A block that does not shrink is stored as it is (the reader knows
// it by its size). Empty when this build has no zlib.
std::vector<std::uint8_t> zip_block(const std::vector<std::uint8_t>& raw) {
#if RUDRA_HAVE_ZLIB
    const std::size_t n = raw.size();
    std::vector<std::uint8_t> t(n);
    std::size_t a = 0, b = (n + 1) / 2;
    for (std::size_t i = 0; i < n; ++i) t[(i & 1u) ? b++ : a++] = raw[i];
    for (std::size_t i = n; i-- > 1;) t[i] = std::uint8_t(int(t[i]) - int(t[i - 1]) + 128 + 256);
    uLongf size = compressBound(uLong(n));
    std::vector<std::uint8_t> z(size);
    if (compress2(z.data(), &size, t.data(), uLong(n), 6) != Z_OK || size >= n) return raw;
    z.resize(size);
    return z;
#else
    (void)raw;
    return {};
#endif
}

// The inverse: `expect` bytes back, or nullopt (bad data, or no zlib in this build).
std::optional<std::vector<std::uint8_t>> unzip_block(const std::uint8_t* data, std::size_t size, std::size_t expect) {
    if (size == expect) return std::vector<std::uint8_t>(data, data + size);   // stored as it is
#if RUDRA_HAVE_ZLIB
    std::vector<std::uint8_t> t(expect);
    uLongf got = uLongf(expect);
    if (expect == 0 || uncompress(t.data(), &got, data, uLong(size)) != Z_OK || got != expect) return std::nullopt;
    for (std::size_t i = 1; i < expect; ++i) t[i] = std::uint8_t(int(t[i - 1]) + int(t[i]) - 128);
    std::vector<std::uint8_t> out(expect);
    std::size_t a = 0, b = (expect + 1) / 2;
    for (std::size_t i = 0; i < expect; ++i) out[i] = t[(i & 1u) ? b++ : a++];
    return out;
#else
    (void)data;
    return std::nullopt;
#endif
}
}  // namespace

Result<std::vector<std::uint8_t>> exr_bytes(const PlanarBuffer& rgb, bool half,
                                            const std::optional<Chromaticities>& chroma, const ExrAttributes& attributes,
                                            const PlanarBuffer* reconstruction_mask, const ExrTypedAttributes& typed,
                                            ExrCompression compression) {
    const int n_ch = rgb.channels();
    if (n_ch != 3 && n_ch != 4) return make_error(ErrorCode::InvalidArgument, "expected a 3 or 4 channel image");
    const int w = rgb.width(), h = rgb.height();
    if (reconstruction_mask && (reconstruction_mask->channels() != 1 ||
        reconstruction_mask->width() != w || reconstruction_mask->height() != h ||
        !std::all_of(reconstruction_mask->span().begin(), reconstruction_mask->span().end(),
            [](float v) { return std::isfinite(v) && v >= 0 && v <= 1; })))
        return make_error(ErrorCode::InvalidArgument, "reconstruction mask must be finite HxW values in [0, 1]");
    std::vector<std::string> names = {"R", "G", "B"};
    if (n_ch == 4) names.push_back("A");
    if (reconstruction_mask) names.push_back("rudra.reconstructed");
    std::vector<const float*> planes;
    for (int c = 0; c < n_ch; ++c) planes.push_back(rgb.plane(c));
    if (reconstruction_mask) planes.push_back(reconstruction_mask->plane(0));
    return encode_planes(names, planes, w, h, half, chroma, attributes, typed, compression);
}

Result<void> write_mattes_exr(const std::filesystem::path& path, const NamedPlanes& planes,
                              const ExrAttributes& attributes, ExrCompression compression) {
    if (planes.empty()) return make_error(ErrorCode::InvalidArgument, "a mattes EXR needs at least one matte");
    const int w = planes.front().second.width(), h = planes.front().second.height();
    std::vector<std::string> names;
    std::vector<const float*> data;
    for (const auto& [name, p] : planes) {
        if (name.empty() || name == "R" || name == "G" || name == "B" || name == "A" ||
            std::find(names.begin(), names.end(), name) != names.end())
            return make_error(ErrorCode::InvalidArgument, "matte names must be unique and not R, G, B or A", name);
        if (p.channels() != 1 || p.width() != w || p.height() != h ||
            !std::all_of(p.span().begin(), p.span().end(), [](float v) { return std::isfinite(v) && v >= 0 && v <= 1; }))
            return make_error(ErrorCode::InvalidArgument, "a matte must be finite HxW values in [0, 1], all one size", name);
        names.push_back(name);
        data.push_back(p.plane(0));
    }
    const auto encoded = encode_planes(names, data, w, h, false, std::nullopt, attributes, {}, compression);
    if (!encoded) return encoded.error();
    const std::vector<std::uint8_t>& bytes = *encoded;
    std::error_code ec;
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream out(path, std::ios::binary);
    if (!out) return make_error(ErrorCode::IoError, "Cannot write the mattes EXR.", path.string());
    out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    out.close();
    if (!out) return make_error(ErrorCode::IoError, "Cannot write the mattes EXR.", path.string());
    return {};
}

namespace {
std::vector<std::uint8_t> planes_bytes(const std::vector<std::string>& names, const std::vector<const float*>& planes,
                                       int w, int h, bool half, const std::optional<Chromaticities>& chroma,
                                       const ExrAttributes& attributes, const ExrTypedAttributes& typed,
                                       ExrCompression compression) {
    const bool zip = (compression == ExrCompression::Zip || compression == ExrCompression::Zips) && exr_zip_available();
    const int zip_lines = compression == ExrCompression::Zips ? 1 : kZipLines;   // ZIPS: one scanline a block
    std::vector<int> order(names.size());
    for (std::size_t i = 0; i < names.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return names[a] < names[b]; });

    std::vector<std::uint8_t> head;
    {
        std::vector<std::uint8_t> ch;
        for (int i : order) {
            put_str(ch, names[i].c_str());
            put_i32(ch, half ? 1 : 2);
            ch.insert(ch.end(), {0, 0, 0, 0});
            put_i32(ch, 1);
            put_i32(ch, 1);
        }
        ch.push_back(0);
        attr(head, "channels", "chlist", ch);
    }
    attr(head, "compression", "compression", {std::uint8_t(!zip ? 0 : zip_lines == 1 ? 2 : 3)});
    std::vector<std::uint8_t> box;
    for (int v : {0, 0, w - 1, h - 1}) put_i32(box, v);
    attr(head, "dataWindow", "box2i", box);
    attr(head, "displayWindow", "box2i", box);
    attr(head, "lineOrder", "lineOrder", {0});
    std::vector<std::uint8_t> f1, v2;
    put_f32(f1, 1.0f);
    put_f32(v2, 0.0f);
    put_f32(v2, 0.0f);
    attr(head, "pixelAspectRatio", "float", f1);
    attr(head, "screenWindowCenter", "v2f", v2);
    attr(head, "screenWindowWidth", "float", f1);
    if (chroma) {
        std::vector<std::uint8_t> c;
        for (double v : *chroma) put_f32(c, static_cast<float>(v));
        attr(head, "chromaticities", "chromaticities", c);
    }
    for (const auto& [k, v] : attributes) attr(head, k, "string", std::vector<std::uint8_t>(v.begin(), v.end()));
    for (const auto& t : typed) attr(head, t.name, t.type, t.payload);
    head.push_back(0);

    const int bpp = half ? 2 : 4;
    const std::int64_t row_size = std::int64_t(w) * bpp * static_cast<std::int64_t>(names.size());
    const std::int64_t chunk = 8 + row_size;
    const std::int64_t data_start = 8 + std::int64_t(head.size()) + 8 * std::int64_t(h);

    std::vector<std::uint8_t> out;
    if (!zip) {
        out.reserve(std::size_t(data_start + chunk * h));
        put_i32(out, kMagic);
        put_i32(out, 2);
        out.insert(out.end(), head.begin(), head.end());
        for (int y = 0; y < h; ++y) put_u64(out, std::uint64_t(data_start + std::int64_t(y) * chunk));
    }

    auto clean = [half](float v) { return clean_value(v, half); };
    if (zip) {   // blocks of 16 scanlines (ZIPS: 1), each zipped; the offset table one entry a block
        const int blocks = (h + zip_lines - 1) / zip_lines;
        std::vector<std::vector<std::uint8_t>> zipped(static_cast<std::size_t>(blocks));
        std::vector<std::uint8_t> raw;
        for (int bi = 0; bi < blocks; ++bi) {
            raw.clear();
            for (int y = bi * zip_lines; y < std::min(h, (bi + 1) * zip_lines); ++y)
                for (int c : order) {
                    const float* row = planes[std::size_t(c)] + std::size_t(y) * std::size_t(w);
                    for (int x = 0; x < w; ++x) {
                        const float v = clean(row[x]);
                        if (half) {
                            const std::uint16_t hv = float_to_half(v);
                            raw.push_back(std::uint8_t(hv));
                            raw.push_back(std::uint8_t(hv >> 8));
                        } else {
                            put_f32(raw, v);
                        }
                    }
                }
            zipped[std::size_t(bi)] = zip_block(raw);
        }
        put_i32(out, kMagic);
        put_i32(out, 2);
        out.insert(out.end(), head.begin(), head.end());
        std::uint64_t at = 8 + head.size() + 8 * std::uint64_t(blocks);
        for (const auto& d : zipped) {
            put_u64(out, at);
            at += 8 + d.size();
        }
        for (int bi = 0; bi < blocks; ++bi) {
            put_i32(out, bi * zip_lines);
            put_i32(out, static_cast<std::int32_t>(zipped[std::size_t(bi)].size()));
            out.insert(out.end(), zipped[std::size_t(bi)].begin(), zipped[std::size_t(bi)].end());
        }
        return out;
    }
    for (int y = 0; y < h; ++y) {
        put_i32(out, y);
        put_i32(out, static_cast<std::int32_t>(row_size));
        for (int c : order) {
            const float* row = planes[std::size_t(c)] + std::size_t(y) * std::size_t(w);
            for (int x = 0; x < w; ++x) {
                const float v = clean(row[x]);
                if (half) {
                    const std::uint16_t hv = float_to_half(v);
                    out.push_back(std::uint8_t(hv));
                    out.push_back(std::uint8_t(hv >> 8));
                } else {
                    put_f32(out, v);
                }
            }
        }
    }
    return out;
}
}  // namespace

Result<void> write_exr(const std::filesystem::path& path, const PlanarBuffer& rgb, bool half,
                       const std::optional<Chromaticities>& chroma, const ExrAttributes& attributes,
                       const PlanarBuffer* reconstruction_mask, const ExrTypedAttributes& typed,
                       ExrCompression compression) {
    auto bytes = exr_bytes(rgb, half, chroma, attributes, reconstruction_mask, typed, compression);
    if (!bytes) return bytes.error();
    std::error_code ec;
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), ec);
    // errno is cleared first so a reason left over from earlier is never
    // reported as this one's, and the file is closed here, not by the
    // destructor: the last buffered bytes reach the disk on close, and a full
    // disk there used to pass as a written master.
    errno = 0;
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        const int err = errno;
        return make_error(ErrorCode::IoError,
                          "The EXR file could not be created (" + io_failure_reason(path, err, bytes->size()) + ").",
                          path.string());
    }
    f.write(reinterpret_cast<const char*>(bytes->data()), static_cast<std::streamsize>(bytes->size()));
    f.close();
    if (!f) {
        const int err = errno;
        std::string reason = io_failure_reason(path, err, bytes->size());
        std::filesystem::remove(path, ec);   // never leave a truncated master behind
        return make_error(ErrorCode::IoError, "The EXR file could not be written (" + reason + ").", path.string());
    }
    return {};
}

const char* primaries_name(Primaries p) noexcept {
    switch (p) {
        case Primaries::Rec709: return "rec709";
        case Primaries::Rec2020: return "rec2020";
        case Primaries::P3D65: return "p3d65";
        case Primaries::Ap0: return "ap0";
        case Primaries::Ap1: return "ap1";
        case Primaries::Awg4: return "awg4";
        case Primaries::AdobeRgb: return "adobe_rgb";
    }
    return "?";
}

namespace {

Result<void> write_aces_like(const std::filesystem::path& path, const PlanarBuffer& rgb, Primaries source,
                             Primaries dst, double exposure, const ExrAttributes& provenance, bool ap0, bool half,
                             const PlanarBuffer* reconstruction_mask, const ExrTypedAttributes& typed,
                             const PlanarBuffer* alpha, ExrCompression compression = ExrCompression::None) {
    if (alpha && (alpha->channels() != 1 || alpha->height() != rgb.height() || alpha->width() != rgb.width()))
        return make_error(ErrorCode::InvalidArgument, "The alpha must be one plane the size of the picture.");
    PlanarBuffer scaled = rgb;
    const float k = static_cast<float>(exposure);
    for (float& v : scaled.span()) v = v * k;
    PlanarBuffer out = convert_primaries(scaled, source, dst);
    if (alpha) {   // RGB premultiplied by A, then A: OpenEXR's associated alpha
        PlanarBuffer rgba(4, rgb.height(), rgb.width());
        for (std::size_t i = 0; i < out.plane_size(); ++i) {
            const float a = std::clamp(alpha->plane(0)[i], 0.0f, 1.0f);
            for (int c = 0; c < 3; ++c) rgba.plane(c)[i] = out.plane(c)[i] * a;
            rgba.plane(3)[i] = a;
        }
        out = std::move(rgba);
    }
    ExrAttributes attrs;
    ExrTypedAttributes all_typed = typed;
    // SMPTE ST 2065-4 says acesImageContainerFlag is an int (= 1), only on a file
    // that is the container: AP0, RGB(A), half float, uncompressed scanlines (28 Oct 2026 review).
    // 1.2.6 (readiness report, checklist item 3): a float32 AP0 file (the analytic route's grading
    // master) carried the flag too, but the container is half float only: it is labelled instead.
    // 1.3.2 (28 Nov 2026): the AP0 file takes Deliver's EXR compression too; compressed, it is not the container
    // either (ST 2065-4 is uncompressed), and is labelled with its compression instead.
    const bool compressed = compression != ExrCompression::None && exr_compression_available(compression);
    if (ap0 && !reconstruction_mask && half && !compressed) all_typed.push_back({"acesImageContainerFlag", "int", {1, 0, 0, 0}});
    else
        attrs.emplace_back("rudra:container", !ap0                 ? "ACEScg (AP1)"
                                              : reconstruction_mask ? "AP0 linear with reconstruction mask"
                                              : compressed ? std::string("AP0 linear, ") + exr_compression_label(compression) +
                                                                 " compressed: not the SMPTE ST 2065-4 container (uncompressed)"
                                                           : "AP0 linear, float32: not the SMPTE ST 2065-4 container (half float)");
    attrs.emplace_back("rudra:sourceSpace", primaries_name(source));
    attrs.emplace_back("rudra:exposureConvention", "0.18 = 18% grey (scene), diffuse white = 1.0");
    attrs.emplace_back("rudra:exposureScale", pyjson::repr(exposure));
    if (alpha) attrs.emplace_back("rudra:alpha", "the source's straight alpha as A; RGB premultiplied by it (OpenEXR associated alpha)");
    if (!provenance.empty()) {
        pyjson::Dict d;
        for (const auto& [kk, v] : provenance) d.emplace_back(kk, v);
        attrs.emplace_back("rudra:provenance", pyjson::dumps(d, -1, true));
    }
    return write_exr(path, out, half, ap0 ? kAp0Chromaticities : kAp1Chromaticities, attrs, reconstruction_mask, all_typed,
                     compression);
}

}  // namespace

Result<void> write_aces_exr(const std::filesystem::path& path, const PlanarBuffer& rgb, Primaries source,
                            double exposure, const ExrAttributes& provenance, bool half, const PlanarBuffer* reconstruction_mask,
                            const ExrTypedAttributes& typed, const PlanarBuffer* alpha, ExrCompression compression) {
    return write_aces_like(path, rgb, source, Primaries::Ap0, exposure, provenance, true, half, reconstruction_mask, typed, alpha,
                           compression);
}

Result<void> write_acescg_exr(const std::filesystem::path& path, const PlanarBuffer& rgb, Primaries source,
                              double exposure, const ExrAttributes& provenance, bool half, const PlanarBuffer* reconstruction_mask,
                              const ExrTypedAttributes& typed, const PlanarBuffer* alpha, ExrCompression compression) {
    return write_aces_like(path, rgb, source, Primaries::Ap1, exposure, provenance, false, half, reconstruction_mask, typed, alpha,
                           compression);
}

ExrTypedAttributes exr_timecode_attributes(const std::string& timecode, double fps, const std::string& reel) {
    ExrTypedAttributes out;
    auto u32 = [](std::vector<std::uint8_t>& b, std::uint32_t v) {
        for (int i = 0; i < 4; ++i) b.push_back(std::uint8_t(v >> (8 * i)));
    };
    if (fps > 0) {
        // framesPerSecond as the exact rational for the NTSC rates (24000/1001 ...).
        std::int32_t num = std::int32_t(std::lround(fps * 1000.0)), den = 1000;
        if (std::abs(fps - std::round(fps)) < 1e-6) num = std::int32_t(std::lround(fps)), den = 1;
        else if (std::abs(fps * 1.001 - std::round(fps * 1.001)) < 1e-3)
            num = std::int32_t(std::lround(fps * 1.001)) * 1000, den = 1001;
        std::vector<std::uint8_t> r;
        u32(r, std::uint32_t(num));
        u32(r, std::uint32_t(den));
        out.push_back({"framesPerSecond", "rational", std::move(r)});
    }
    if (auto tc = parse_timecode(timecode)) {
        const auto packed = exr_timecode(*tc);
        std::vector<std::uint8_t> b;
        u32(b, packed[0]);
        u32(b, packed[1]);
        out.push_back({"timeCode", "timecode", std::move(b)});
    }
    if (!reel.empty()) out.push_back({"reelName", "string", std::vector<std::uint8_t>(reel.begin(), reel.end())});
    return out;
}

namespace {
float half_to_float(std::uint16_t h) noexcept {
    const std::uint32_t sign = std::uint32_t(h & 0x8000u) << 16;
    std::uint32_t exp = (h >> 10) & 0x1fu, mant = h & 0x3ffu;
    std::uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            int e = -1;
            do { ++e; mant <<= 1; } while (!(mant & 0x400u));
            bits = sign | std::uint32_t(127 - 15 - e) << 23 | (mant & 0x3ffu) << 13;
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | mant << 13;
    } else {
        bits = sign | (exp + 112) << 23 | mant << 13;
    }
    return std::bit_cast<float>(bits);
}
}  // namespace

Result<ExrImage> read_exr(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return make_error(ErrorCode::NotFound, "The EXR file could not be opened.", path.string());
    const std::vector<std::uint8_t> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto bad = [&](const char* why) { return make_error(ErrorCode::ParseError, "The EXR file is not readable.", path.string() + ": " + why); };
    // The file is not trusted (security review, 11 Nov 2026): a reference image or a master can come
    // from anywhere. Every read is checked against the file's size first, a scanline must name a row
    // inside the data window, and nothing is allocated until the file is seen to hold the pixels its
    // header promises. Before this a crafted scanline row wrote outside the pixel buffer.
    const std::size_t n = raw.size();
    auto has = [&](std::size_t at, std::size_t len) { return at <= n && len <= n - at; };
    std::size_t pos = 0;
    auto i32 = [&](std::size_t at) {   // the caller has checked has(at, 4)
        std::int32_t v;
        std::memcpy(&v, raw.data() + at, 4);
        return v;
    };
    if (n < 8 || i32(0) != kMagic) return bad("not an EXR file");
    if (i32(4) & 0x1a00) return bad("tiled, deep or multi-part");
    pos = 8;
    auto cstr = [&](std::size_t& at, std::string& s) {   // false: no terminator before the end of the file
        const std::size_t start = at;
        while (at < n && raw[at] != 0) ++at;
        if (at >= n) return false;
        s.assign(raw.begin() + std::ptrdiff_t(start), raw.begin() + std::ptrdiff_t(at));
        ++at;
        return true;
    };
    ExrImage img;
    std::vector<std::string> channels;
    int x0 = 0, y0 = 0, x1 = -1, y1 = -1, compression = 0;
    for (;;) {
        if (pos >= n) return bad("truncated header");
        if (raw[pos] == 0) break;
        std::string name, type;
        if (!cstr(pos, name) || !cstr(pos, type) || !has(pos, 4)) return bad("truncated header");
        const std::int32_t declared = i32(pos);
        pos += 4;
        if (declared < 0 || !has(pos, static_cast<std::size_t>(declared))) return bad("truncated header");
        const auto size = static_cast<std::size_t>(declared);
        std::vector<std::uint8_t> payload(raw.begin() + std::ptrdiff_t(pos), raw.begin() + std::ptrdiff_t(pos + size));
        if (type == "chlist") {
            const std::size_t end = pos + size;
            std::size_t c = pos;
            for (;;) {
                if (c >= end) return bad("bad channel list");
                if (raw[c] == 0) break;
                std::string channel;
                if (!cstr(c, channel) || c > end || end - c < 16) return bad("bad channel list");
                const std::int32_t pixel_type = i32(c);
                if (pixel_type < 0 || pixel_type > 2) return bad("bad channel list");
                channels.push_back(std::move(channel));
                img.pixel_types.push_back(pixel_type);
                c += 16;
            }
        } else if (type == "compression") {
            if (size < 1) return bad("bad compression attribute");
            compression = payload[0];
        } else if (name == "dataWindow") {
            if (size < 16) return bad("bad data window");
            x0 = i32(pos); y0 = i32(pos + 4); x1 = i32(pos + 8); y1 = i32(pos + 12);
        }
        img.attribute_types.emplace_back(name, type);
        img.attributes.emplace_back(name, std::move(payload));
        pos += size;
    }
    ++pos;
    // 1.2.6 (item 8): ZIPS (2, one scanline a block) and ZIP (3, sixteen) besides uncompressed; 1.3.2: RLE (1), PIZ (4),
    // PXR24 (5), B44 (6), B44A (7), DWAA (8) and DWAB (9) through the OpenEXR library, when the build has it.
    const bool by_openexr = compression == 1 || (compression >= 4 && compression <= 9);
    if (compression != 0 && compression != 2 && compression != 3 && !by_openexr) return bad("compressed with a method RUDRA does not read");
    if (by_openexr && !exr_detail::openexr_available())
        return bad("compressed with a method this build of RUDRA does not read (it has no OpenEXR library)");
    if ((compression == 2 || compression == 3) && !exr_zip_available()) return bad("ZIP-compressed, and this build has no zlib");
    // In 64 bits: x1 - x0 + 1 overflows an int for a hostile window.
    const std::int64_t w64 = std::int64_t(x1) - std::int64_t(x0) + 1, h64 = std::int64_t(y1) - std::int64_t(y0) + 1;
    if (w64 <= 0 || h64 <= 0 || channels.empty()) return bad("empty data window");
    constexpr std::int64_t kMaxSide = 1 << 20;   // far past any frame; keeps the products below in range
    if (w64 > kMaxSide || h64 > kMaxSide) return bad("data window too large");
    const int w = static_cast<int>(w64), h = static_cast<int>(h64);
    // The file must hold the offset table and h scanlines of every channel before anything is allocated.
    std::size_t line_bytes = 8;   // each scanline: its row, its size, then the channels
    const std::vector<std::uint8_t>* px = &raw;   // the scanlines as uncompressed ones
    std::vector<std::uint8_t> flat;
    if (compression == 0) {
        for (int t : img.pixel_types) {
            line_bytes += std::size_t(w) * (t == 1 ? 2 : 4);
            if (line_bytes > n) return bad("truncated data");
        }
        if (!has(pos, 8 * std::size_t(h))) return bad("truncated data");
        pos += 8 * std::size_t(h);
        if ((n - pos) / line_bytes < std::size_t(h)) return bad("truncated data");
    } else if (by_openexr) {
        // The same budget as ZIP's before anything is decoded (a hostile header can claim any size).
        std::uint64_t pixel_bytes = 0;
        for (int t : img.pixel_types) pixel_bytes += std::uint64_t(w) * (t == 1 ? 2u : 4u);
        constexpr std::uint64_t kMaxDecoded = 2ull << 30;
        if (std::uint64_t(h) * (pixel_bytes + 8) > kMaxDecoded) return bad("more pixel data than RUDRA reads from one file");
        auto decoded = exr_detail::openexr_decode_flat(raw, channels, img.pixel_types, x0, y0, w, h);
        if (!decoded) return make_error(ErrorCode::ParseError, "The EXR file is not readable.", path.string() + ": " + decoded.error().detail);
        flat = std::move(*decoded);
        px = &flat;
        pos = 0;
    } else {
        // Each block unzipped into flat as the uncompressed layout (row, size, channels), so the
        // reading below is the same. Nothing past what zlib can expand the file to is allocated.
        std::uint64_t pixel_bytes = 0;
        for (int t : img.pixel_types) pixel_bytes += std::uint64_t(w) * (t == 1 ? 2u : 4u);
        line_bytes += std::size_t(pixel_bytes);
        if (std::uint64_t(h) * pixel_bytes > 1100ull * n + (1ull << 20)) return bad("data larger than the file can hold");
        // A fixed budget too (code review, 19 Nov 2026): the file's size alone can be padded by an
        // unrelated attribute. 2 GiB of pixels is an 8K frame with dozens of float channels.
        constexpr std::uint64_t kMaxDecoded = 2ull << 30;
        if (std::uint64_t(h) * (pixel_bytes + 8) > kMaxDecoded) return bad("more pixel data than RUDRA reads from one file");
        const int lines = compression == 3 ? 16 : 1;
        const std::int64_t blocks = (h64 + lines - 1) / lines;
        if (!has(pos, 8 * std::size_t(blocks))) return bad("truncated data");
        pos += 8 * std::size_t(blocks);
        flat.reserve(std::size_t(h) * line_bytes);
        std::vector<bool> seen(std::size_t(blocks), false);
        for (std::int64_t b = 0; b < blocks; ++b) {
            if (!has(pos, 8)) return bad("truncated data");
            const std::int64_t first = std::int64_t(i32(pos)) - std::int64_t(y0);
            const std::int32_t size = i32(pos + 4);
            pos += 8;
            if (first < 0 || first >= h64 || first % lines != 0) return bad("block outside the data window");
            if (seen[std::size_t(first / lines)]) return bad("a block twice");
            seen[std::size_t(first / lines)] = true;
            if (size < 0 || !has(pos, std::size_t(size))) return bad("truncated data");
            // Each block on its own: zlib expands at most about 1032 times, so a block claiming more than
            // that is refused before its buffer is allocated (a stored block is its own size).
            if (const std::uint64_t want = std::uint64_t(std::min<std::int64_t>(lines, h64 - first)) * pixel_bytes;
                want != std::uint64_t(size) && want > 1100ull * std::uint64_t(size) + 1024)
                return bad("a ZIP block claims more than it can hold");
            const int count = int(std::min<std::int64_t>(lines, h64 - first));
            auto block = unzip_block(raw.data() + pos, std::size_t(size), std::size_t(count) * std::size_t(pixel_bytes));
            if (!block) return bad("a ZIP block does not decompress");
            pos += std::size_t(size);
            for (int l = 0; l < count; ++l) {
                put_i32(flat, std::int32_t(first + l + y0));
                put_i32(flat, std::int32_t(pixel_bytes));
                const auto* s = block->data() + std::size_t(l) * std::size_t(pixel_bytes);
                flat.insert(flat.end(), s, s + pixel_bytes);
            }
        }
        px = &flat;
        pos = 0;
    }
    const std::size_t pn = px->size();
    auto phas = [&](std::size_t at, std::size_t len) { return at <= pn && len <= pn - at; };
    auto pi32 = [&](std::size_t at) {
        std::int32_t v;
        std::memcpy(&v, px->data() + at, 4);
        return v;
    };
    static const char* order[] = {"R", "G", "B", "A"};
    const int nout = int(std::count(channels.begin(), channels.end(), std::string("A"))) ? 4 : 3;
    img.pixels = PlanarBuffer(nout, h, w);
    if (std::find(channels.begin(), channels.end(), "rudra.reconstructed") != channels.end())
        img.reconstruction_mask.emplace(1, h, w);
    const bool all_half = std::all_of(img.pixel_types.begin(), img.pixel_types.end(), [](int t) { return t == 1; });
    if (all_half) img.half_bits.assign(std::size_t(nout) * std::size_t(w) * std::size_t(h), 0);
    for (int r = 0; r < h; ++r) {
        if (!phas(pos, 8)) return bad("truncated data");
        const std::int64_t row = std::int64_t(pi32(pos)) - std::int64_t(y0);
        if (row < 0 || row >= h64) return bad("scanline outside the data window");
        const int y = static_cast<int>(row);
        pos += 8;
        for (std::size_t c = 0; c < channels.size(); ++c) {
            int plane = -1;
            for (int k = 0; k < nout; ++k)
                if (channels[c] == order[k]) plane = k;
            const bool half = img.pixel_types[c] == 1;
            const std::size_t bytes = std::size_t(w) * (half ? 2 : 4);
            if (!phas(pos, bytes)) return bad("truncated data");
            const bool is_mask = channels[c] == "rudra.reconstructed";
            PlanarBuffer* other = nullptr;
            if (plane < 0 && !is_mask) {   // any other channel, kept by name (5 Nov 2026)
                auto it = std::find_if(img.other.begin(), img.other.end(), [&](const auto& p) { return p.first == channels[c]; });
                if (it == img.other.end()) {
                    img.other.emplace_back(channels[c], PlanarBuffer(1, h, w));
                    it = img.other.end() - 1;
                }
                other = &it->second;
            }
            {
                for (int x = 0; x < w; ++x) {
                    float v;
                    if (half) {
                        std::uint16_t hv;
                        std::memcpy(&hv, px->data() + pos + std::size_t(x) * 2, 2);
                        v = half_to_float(hv);
                        if (all_half && plane >= 0) img.half_bits[(std::size_t(plane) * std::size_t(h) + std::size_t(y)) * std::size_t(w) + std::size_t(x)] = hv;
                    } else {
                        std::memcpy(&v, px->data() + pos + std::size_t(x) * 4, 4);
                    }
                    if (is_mask) img.reconstruction_mask->at(0, y, x) = v;
                    else if (other) other->at(0, y, x) = v;
                    else img.pixels.at(plane, y, x) = v;
                }
            }
            pos += bytes;
        }
    }
    return img;
}

std::string ocio_config_text() {
    auto matrix_yaml = [](const Mat3& m) {
        std::string s = std::string(8, ' ') + "- !<MatrixTransform> {matrix: [";
        const double m4[16] = {m[0][0], m[0][1], m[0][2], 0, m[1][0], m[1][1], m[1][2], 0,
                               m[2][0], m[2][1], m[2][2], 0, 0, 0, 0, 1};
        for (int i = 0; i < 16; ++i) {
            char b[64];
            std::snprintf(b, sizeof b, "%.10f", m4[i]);
            s += b;
            if (i < 15) s += ", ";
        }
        return s + "]}";
    };
    struct Space { const char* name; Primaries src; const char* label; };
    const Space spaces[] = {{"RUDRA Scene Linear (Rec.2020)", Primaries::Rec2020, "Rec.2020"},
                            {"Linear Rec.709", Primaries::Rec709, "Linear Rec.709"},
                            {"ACEScg", Primaries::Ap1, "ACEScg"}};
    std::string blocks;
    for (const auto& sp : spaces) {
        blocks += "  - !<ColorSpace>\n    name: " + std::string(sp.name) +
                  "\n    family: Scene\n    encoding: scene-linear\n    bitdepth: 32f\n    description: |\n      Linear " +
                  sp.label +
                  " to ACES2065-1 via Bradford CAT.\n      Generated by rudra.delivery.aces; matrices derived from "
                  "primaries, not copied.\n    to_scene_reference: !<GroupTransform>\n      children:\n" +
                  matrix_yaml(rgb_to_rgb_matrix(sp.src, Primaries::Ap0)) + "\n";
    }
    return std::string(
           "ocio_profile_version: 2\n"
           "\n"
           "description: RUDRA delivery working spaces (pair with the official ACES OCIO config for display transforms)\n"
           "name: rudra-delivery\n"
           "\n"
           "roles:\n"
           "  scene_linear: RUDRA Scene Linear (Rec.2020)\n"
           "  aces_interchange: ACES2065-1\n"
           "  default: RUDRA Scene Linear (Rec.2020)\n"
           "  data: Raw\n"
           "\n"
           "file_rules:\n"
           "  - !<Rule> {name: Default, colorspace: default}\n"
           "\n"
           "colorspaces:\n"
           "  - !<ColorSpace>\n"
           "    name: ACES2065-1\n"
           "    family: Scene\n"
           "    encoding: scene-linear\n"
           "    bitdepth: 32f\n"
           "    description: SMPTE ST 2065-1 AP0 primaries, ACES white. Reference space.\n"
           "\n"
           "  - !<ColorSpace>\n"
           "    name: Raw\n"
           "    family: Utility\n"
           "    encoding: data\n"
           "    isdata: true\n"
           "\n") +
           blocks;
}

}  // namespace rudra
