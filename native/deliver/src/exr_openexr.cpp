#include "exr_openexr.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>

#if RUDRA_HAVE_OPENEXR
// OpenEXR's and Imath's headers are not held to RUDRA's warnings (cmake/RudraWarnings.cmake): a copy fetched and built
// with RUDRA (cmake/RudraDeps.cmake) is not a system include, so its conversions and shadowing would be reported here.
#if defined(_MSC_VER)
#pragma warning(push, 0)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <ImfChannelList.h>
#include <ImfChromaticitiesAttribute.h>
#include <ImfCompression.h>
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfIO.h>
#include <ImfInputFile.h>
#include <ImfIntAttribute.h>
#include <ImfOpaqueAttribute.h>
#include <ImfOutputFile.h>
#include <ImfRationalAttribute.h>
#include <ImfStandardAttributes.h>
#include <ImfStringAttribute.h>
#include <ImfTimeCodeAttribute.h>
#include <Iex.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif

namespace rudra::exr_detail {

#if RUDRA_HAVE_OPENEXR
namespace {

// The file being written, in memory: OpenEXR seeks back to fill in the offset table.
class MemoryOut : public Imf::OStream {
public:
    MemoryOut() : Imf::OStream("<memory>") {}
    void write(const char c[], int n) override {
        if (n <= 0) return;
        const std::uint64_t end = pos_ + std::uint64_t(n);
        if (end > bytes.size()) bytes.resize(std::size_t(end));
        std::memcpy(bytes.data() + pos_, c, std::size_t(n));
        pos_ = end;
    }
    std::uint64_t tellp() override { return pos_; }
    void seekp(std::uint64_t pos) override { pos_ = pos; }
    std::vector<std::uint8_t> bytes;

private:
    std::uint64_t pos_ = 0;
};

// A file already read into memory. Its size is given, so OpenEXR runs its consistency checks: the file is not
// trusted (read_exr's security review, 11 Nov 2026).
class MemoryIn : public Imf::IStream {
public:
    explicit MemoryIn(const std::vector<std::uint8_t>& b) : Imf::IStream("<memory>"), b_(b) {}
    bool read(char c[], int n) override {
        if (n < 0 || pos_ > b_.size() || std::size_t(n) > b_.size() - pos_) throw Iex::InputExc("Unexpected end of the EXR file.");
        std::memcpy(c, b_.data() + pos_, std::size_t(n));
        pos_ += std::size_t(n);
        return pos_ < b_.size();
    }
    std::uint64_t tellg() override { return pos_; }
    void seekg(std::uint64_t pos) override { pos_ = std::size_t(std::min<std::uint64_t>(pos, b_.size())); }
    std::int64_t size() override { return std::int64_t(b_.size()); }
    bool isStatelessRead() const override { return true; }
    std::int64_t read(void* buf, std::uint64_t sz, std::uint64_t offset) override {
        if (offset >= b_.size()) return 0;
        const std::uint64_t n = std::min<std::uint64_t>(sz, b_.size() - offset);
        std::memcpy(buf, b_.data() + offset, std::size_t(n));
        return std::int64_t(n);
    }

private:
    const std::vector<std::uint8_t>& b_;
    std::size_t pos_ = 0;
};

Imf::Compression imf_compression(ExrCompression c) {
    switch (c) {
        case ExrCompression::None: return Imf::NO_COMPRESSION;
        case ExrCompression::Rle: return Imf::RLE_COMPRESSION;
        case ExrCompression::Zips: return Imf::ZIPS_COMPRESSION;
        case ExrCompression::Zip: return Imf::ZIP_COMPRESSION;
        case ExrCompression::Piz: return Imf::PIZ_COMPRESSION;
        case ExrCompression::Pxr24: return Imf::PXR24_COMPRESSION;
        case ExrCompression::B44: return Imf::B44_COMPRESSION;
        case ExrCompression::B44a: return Imf::B44A_COMPRESSION;
        case ExrCompression::Dwaa: return Imf::DWAA_COMPRESSION;
        case ExrCompression::Dwab: return Imf::DWAB_COMPRESSION;
    }
    return Imf::NO_COMPRESSION;
}

std::uint32_t u32_at(const std::vector<std::uint8_t>& b, std::size_t at) {
    return std::uint32_t(b[at]) | std::uint32_t(b[at + 1]) << 8 | std::uint32_t(b[at + 2]) << 16 | std::uint32_t(b[at + 3]) << 24;
}

// A typed attribute (exr_timecode_attributes, acesImageContainerFlag ...) as OpenEXR's own type, so it is written
// exactly as RUDRA's writer writes it; any other type goes in as its raw bytes.
void insert_typed(Imf::Header& header, const ExrTyped& t) {
    const auto& p = t.payload;
    if (t.type == "int" && p.size() == 4) {
        header.insert(t.name, Imf::IntAttribute(int(std::int32_t(u32_at(p, 0)))));
    } else if (t.type == "rational" && p.size() == 8) {
        header.insert(t.name, Imf::RationalAttribute(Imf::Rational(int(std::int32_t(u32_at(p, 0))), u32_at(p, 4))));
    } else if (t.type == "timecode" && p.size() == 8) {
        header.insert(t.name, Imf::TimeCodeAttribute(Imf::TimeCode(u32_at(p, 0), u32_at(p, 4))));
    } else if (t.type == "string") {
        header.insert(t.name, Imf::StringAttribute(std::string(p.begin(), p.end())));
    } else {
        header.insert(t.name, Imf::OpaqueAttribute(t.type.c_str(), long(p.size()), p.data()));
    }
}

}  // namespace

bool openexr_available() noexcept { return true; }

Result<std::vector<std::uint8_t>> openexr_encode(const std::vector<std::string>& names, const std::vector<const float*>& planes,
                                                 int w, int h, bool half, const std::optional<Chromaticities>& chroma,
                                                 const ExrAttributes& attributes, const ExrTypedAttributes& typed,
                                                 ExrCompression compression) {
    if (names.size() != planes.size() || names.empty() || w <= 0 || h <= 0)
        return make_error(ErrorCode::InvalidArgument, "No picture to write as an EXR.");
    try {
        Imf::Header header(w, h);
        header.compression() = imf_compression(compression);
        const std::size_t n = std::size_t(w) * std::size_t(h);
        // The values as they go into the file: half through RUDRA's own rounding (float_to_half, as the uncompressed and
        // ZIP files have them), so a lossless compression holds the very same halves.
        std::vector<std::vector<std::uint16_t>> halves;
        std::vector<std::vector<float>> floats;
        Imf::FrameBuffer fb;
        for (std::size_t i = 0; i < names.size(); ++i) {
            header.channels().insert(names[i], Imf::Channel(half ? Imf::HALF : Imf::FLOAT));
            if (half) {
                halves.emplace_back(n);
                std::uint16_t* d = halves.back().data();
                for (std::size_t k = 0; k < n; ++k) d[k] = float_to_half(planes[i][k]);
                fb.insert(names[i], Imf::Slice(Imf::HALF, reinterpret_cast<char*>(d), sizeof(std::uint16_t),
                                               sizeof(std::uint16_t) * std::size_t(w)));
            } else {
                floats.emplace_back(planes[i], planes[i] + n);
                fb.insert(names[i], Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(floats.back().data()), sizeof(float),
                                               sizeof(float) * std::size_t(w)));
            }
        }
        if (chroma) {
            const auto& c = *chroma;
            Imf::addChromaticities(header, Imf::Chromaticities(Imath::V2f(float(c[0]), float(c[1])), Imath::V2f(float(c[2]), float(c[3])),
                                                               Imath::V2f(float(c[4]), float(c[5])), Imath::V2f(float(c[6]), float(c[7]))));
        }
        for (const auto& [k, v] : attributes) header.insert(k, Imf::StringAttribute(v));
        for (const auto& t : typed) insert_typed(header, t);
        MemoryOut out;
        {
            Imf::OutputFile file(out, header);
            file.setFrameBuffer(fb);
            file.writePixels(h);
        }   // the offset table is written when the file closes
        return std::move(out.bytes);
    } catch (const std::exception& e) {
        return make_error(ErrorCode::IoError, std::string("OpenEXR could not write the ") + exr_compression_label(compression) + " EXR.",
                          e.what());
    }
}

Result<std::vector<std::uint8_t>> openexr_decode_flat(const std::vector<std::uint8_t>& file, const std::vector<std::string>& channels,
                                                      const std::vector<int>& pixel_types, int x0, int y0, int w, int h) {
    if (channels.size() != pixel_types.size() || channels.empty() || w <= 0 || h <= 0)
        return make_error(ErrorCode::ParseError, "The EXR file is not readable.", "empty data window");
    try {
        MemoryIn in(file);
        Imf::InputFile f(in);
        const Imath::Box2i dw = f.header().dataWindow();
        if (dw.min.x != x0 || dw.min.y != y0 || dw.max.x - dw.min.x + 1 != w || dw.max.y - dw.min.y + 1 != h)
            return make_error(ErrorCode::ParseError, "The EXR file is not readable.", "its data window is not the one its header says");
        // Each channel into its own buffer in its own type (UINT 4, HALF 2, FLOAT 4 bytes a value).
        const std::size_t n = std::size_t(w) * std::size_t(h);
        std::vector<std::vector<char>> bufs(channels.size());
        Imf::FrameBuffer fb;
        std::size_t pixel_bytes = 0;
        for (std::size_t c = 0; c < channels.size(); ++c) {
            const int t = pixel_types[c];
            const std::size_t size = t == 1 ? 2 : 4;
            pixel_bytes += size * std::size_t(w);
            bufs[c].assign(n * size, 0);
            const Imf::PixelType pt = t == 0 ? Imf::UINT : t == 1 ? Imf::HALF : Imf::FLOAT;
            // OpenEXR addresses a slice by absolute pixel coordinates: the base sits (x0, y0) before the buffer.
            char* base = bufs[c].data() - (std::ptrdiff_t(x0) + std::ptrdiff_t(y0) * std::ptrdiff_t(w)) * std::ptrdiff_t(size);
            fb.insert(channels[c], Imf::Slice(pt, base, size, size * std::size_t(w)));
        }
        f.setFrameBuffer(fb);
        f.readPixels(dw.min.y, dw.max.y);
        std::vector<std::uint8_t> flat;
        flat.reserve(std::size_t(h) * (8 + pixel_bytes));
        auto put_i32 = [&flat](std::int32_t v) {
            const auto u = std::uint32_t(v);
            for (int k = 0; k < 4; ++k) flat.push_back(std::uint8_t(u >> (8 * k)));
        };
        for (int y = 0; y < h; ++y) {
            put_i32(y0 + y);
            put_i32(std::int32_t(pixel_bytes));
            for (std::size_t c = 0; c < channels.size(); ++c) {
                const std::size_t size = pixel_types[c] == 1 ? 2 : 4;
                const char* row = bufs[c].data() + std::size_t(y) * std::size_t(w) * size;
                flat.insert(flat.end(), row, row + std::size_t(w) * size);
            }
        }
        return flat;
    } catch (const std::exception& e) {
        return make_error(ErrorCode::ParseError, "The EXR file is not readable.", e.what());
    }
}

#else   // no OpenEXR library in this build: None, ZIP and ZIPS only (exr.cpp)

bool openexr_available() noexcept { return false; }

Result<std::vector<std::uint8_t>> openexr_encode(const std::vector<std::string>&, const std::vector<const float*>&, int, int, bool,
                                                 const std::optional<Chromaticities>&, const ExrAttributes&, const ExrTypedAttributes&,
                                                 ExrCompression compression) {
    return make_error(ErrorCode::Unsupported,
                      std::string("This build of RUDRA writes no ") + exr_compression_label(compression) + " EXR (it has no OpenEXR library).");
}

Result<std::vector<std::uint8_t>> openexr_decode_flat(const std::vector<std::uint8_t>&, const std::vector<std::string>&,
                                                      const std::vector<int>&, int, int, int, int) {
    return make_error(ErrorCode::Unsupported, "The EXR file is not readable.",
                      "compressed with a method this build of RUDRA does not read (it has no OpenEXR library)");
}

#endif

}  // namespace rudra::exr_detail
