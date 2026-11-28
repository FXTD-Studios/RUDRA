#include "rudra/media/still.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <cstdio>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#if RUDRA_MEDIA_HAVE_OPENEXR
#include <ImathBox.h>
#include <ImfChannelList.h>
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfInputFile.h>
#endif

namespace rudra {
namespace {

template <class T>
std::size_t distinct_green(const cv::Mat& m) {
    std::vector<T> g;
    g.reserve(m.total());
    const int ch = m.channels();
    const int gi = ch >= 3 ? 1 : 0;   // BGR: green is channel 1 either way
    for (int y = 0; y < m.rows; ++y) {
        const T* row = m.ptr<T>(y);
        for (int x = 0; x < m.cols; ++x) g.push_back(row[x * ch + gi]);
    }
    std::sort(g.begin(), g.end());
    return static_cast<std::size_t>(std::unique(g.begin(), g.end()) - g.begin());
}

// Every stored value on the 8-bit lattice of a 16-bit container, either
// scaled (n * 257) or shifted (n * 256): an 8-bit frame padded up.
bool padded_8bit(const cv::Mat& m) {
    const int ch = std::min(m.channels(), 3);
    for (const unsigned step : {65535u / 255u, 65536u / 256u}) {
        bool all = true;
        for (int y = 0; y < m.rows && all; ++y) {
            const std::uint16_t* row = m.ptr<std::uint16_t>(y);
            for (int x = 0; x < m.cols && all; ++x)
                for (int c = 0; c < ch; ++c)
                    if (row[x * m.channels() + c] % step) { all = false; break; }
        }
        if (all) return true;
    }
    return false;
}

}  // namespace

// OpenCV reads EXR only when asked to (OPENCV_IO_ENABLE_OPENEXR, read once, at its first EXR).
static void enable_openexr() {
    static const bool once = [] {
#ifdef _WIN32
        _putenv_s("OPENCV_IO_ENABLE_OPENEXR", "1");
#else
        setenv("OPENCV_IO_ENABLE_OPENEXR", "1", 0);
#endif
        return true;
    }();
    (void)once;
}

Result<DecodedStill> decode_sdr(std::span<const std::uint8_t> bytes) {
    if (bytes.empty()) return make_error(ErrorCode::ParseError, "The image file is empty.");
    // 1.3.2: an EXR frame (by its magic number) holds linear light. It is read as Rec.709 SDR, clipped at 1.0 and
    // encoded with BT.1886 (code = L^(1/2.4)): the display codes RUDRA reads. Values above 1.0 are what the SDR clipped.
    const bool exr = bytes.size() >= 4 && bytes[0] == 0x76 && bytes[1] == 0x2f && bytes[2] == 0x31 && bytes[3] == 0x01;
    if (exr) enable_openexr();
    const cv::Mat buf(1, static_cast<int>(bytes.size()), CV_8UC1, const_cast<std::uint8_t*>(bytes.data()));
    cv::Mat m;
    try {
        m = cv::imdecode(buf, cv::IMREAD_UNCHANGED);
    } catch (const cv::Exception& e) {
        return make_error(ErrorCode::ParseError, "The image could not be decoded.", e.what());
    }
    if (m.empty())
        return make_error(ErrorCode::Unsupported, exr ? "This EXR could not be read: this build's OpenCV reads no EXR."
                                                      : "This image format is not readable by this build.");
    const int depth = m.depth(), ch = m.channels();
    if (depth != CV_8U && depth != CV_16U && depth != CV_32F)
        return make_error(ErrorCode::Unsupported, "Only 8-bit, 16-bit and float images are read.");
    if (ch != 1 && ch != 3 && ch != 4)
        return make_error(ErrorCode::Unsupported, "Only grey, RGB and RGBA images are read.");

    const int h = m.rows, w = m.cols;
    DecodedStill out;
    out.source = exr ? "opencv (EXR: linear Rec.709, as BT.1886 codes)" : "opencv";
    PlanarBuffer rgb(3, h, w);
    // BGR(A) -> RGB planes; grey replicated.
    auto channel_of = [&](int c) { return ch == 1 ? 0 : 2 - c; };

    if (depth == CV_32F) {
        float top = -std::numeric_limits<float>::infinity();
        for (int y = 0; y < h; ++y) {
            const float* row = m.ptr<float>(y);
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < std::min(ch, 3); ++c)
                    if (!std::isnan(row[x * ch + c])) top = std::max(top, row[x * ch + c]);
        }
        if (!exr && top > 1.0f + 1e-4f) {
            char msg[256];
            std::snprintf(msg, sizeof msg,
                          "float input peaks at %.3f: this is scene-linear HDR, not an SDR frame. RUDRA "
                          "reconstructs display-encoded SDR; an HDR file does not need reconstructing.",
                          double(top));
            return make_error(ErrorCode::Unsupported, msg);
        }
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < h; ++y) {
                const float* row = m.ptr<float>(y);
                for (int x = 0; x < w; ++x) {
                    const float v = row[x * ch + channel_of(c)];
                    rgb.at(c, y, x) = !exr ? std::clamp(v, 0.0f, 1.0f)
                                           : std::isfinite(v) ? std::pow(std::clamp(v, 0.0f, 1.0f), 1.0f / 2.4f) : 0.0f;
                }
            }
        out.bits = 32;
        out.distinct_codes = static_cast<int>(distinct_green<float>(m));
        if (ch == 4) {
            PlanarBuffer a(1, h, w);
            for (int y = 0; y < h; ++y) {
                const float* row = m.ptr<float>(y);
                for (int x = 0; x < w; ++x) a.at(0, y, x) = std::clamp(row[x * ch + 3], 0.0f, 1.0f);
            }
            out.alpha = std::move(a);
        }
    } else if (depth == CV_16U) {
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < h; ++y) {
                const std::uint16_t* row = m.ptr<std::uint16_t>(y);
                for (int x = 0; x < w; ++x) rgb.at(c, y, x) = float(row[x * ch + channel_of(c)]) / 65535.0f;
            }
        out.bits = padded_8bit(m) ? 8 : 16;
        out.distinct_codes = static_cast<int>(distinct_green<std::uint16_t>(m));
        if (ch == 4) {
            PlanarBuffer a(1, h, w);
            for (int y = 0; y < h; ++y) {
                const std::uint16_t* row = m.ptr<std::uint16_t>(y);
                for (int x = 0; x < w; ++x) a.at(0, y, x) = float(row[x * ch + 3]) / 65535.0f;
            }
            out.alpha = std::move(a);
        }
    } else {
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < h; ++y) {
                const std::uint8_t* row = m.ptr<std::uint8_t>(y);
                for (int x = 0; x < w; ++x) rgb.at(c, y, x) = float(row[x * ch + channel_of(c)]) / 255.0f;
            }
        out.bits = 8;
        if (ch == 4) {
            PlanarBuffer a(1, h, w);
            for (int y = 0; y < h; ++y) {
                const std::uint8_t* row = m.ptr<std::uint8_t>(y);
                for (int x = 0; x < w; ++x) a.at(0, y, x) = float(row[x * ch + 3]) / 255.0f;
            }
            out.alpha = std::move(a);
        }
        out.distinct_codes = static_cast<int>(distinct_green<std::uint8_t>(m));
    }
    out.colour = read_still_colour(bytes);
    if (out.colour.orientation != 1) {
        const Oriented o = apply_orientation(rgb.span(), 3, w, h, out.colour.orientation);
        PlanarBuffer turned(3, o.height, o.width);
        std::copy(o.data.begin(), o.data.end(), turned.span().begin());
        rgb = std::move(turned);
    }
    out.rgb = SdrImage(std::move(rgb));
    return out;
}

#if RUDRA_MEDIA_HAVE_OPENEXR
// 1.3.2 (28 Nov 2026, PR #105's Windows CI): an EXR is read with the OpenEXR library RUDRA builds with, not through
// OpenCV. OpenCV's EXR decoder is off unless OPENCV_IO_ENABLE_OPENEXR reaches it, and whether the variable set here
// reaches it depends on how that OpenCV was built: the Windows runner's refused it ("The image could not be decoded").
// Every compression; half and float both read as float. The same picture as the OpenCV path: linear Rec.709 SDR,
// clipped at 1.0, as BT.1886 codes; a grey (Y) file is replicated.
static Result<DecodedStill> decode_exr_openexr(const std::filesystem::path& path) {
    try {
        const std::u8string u8 = path.u8string();   // OpenEXR takes UTF-8 names, also on Windows
        Imf::InputFile file(std::string(u8.begin(), u8.end()).c_str());
        const Imath::Box2i dw = file.header().dataWindow();
        const long long w64 = static_cast<long long>(dw.max.x) - dw.min.x + 1, h64 = static_cast<long long>(dw.max.y) - dw.min.y + 1;
        if (w64 <= 0 || h64 <= 0 || w64 > 65536 || h64 > 65536)
            return make_error(ErrorCode::Unsupported, "This EXR's picture is empty or larger than 65536 pixels a side.");
        const int w = static_cast<int>(w64), h = static_cast<int>(h64);
        const Imf::ChannelList& channels = file.header().channels();
        const bool rgb = channels.findChannel("R") && channels.findChannel("G") && channels.findChannel("B");
        const bool grey = !rgb && channels.findChannel("Y");
        if (!rgb && !grey) return make_error(ErrorCode::Unsupported, "Only RGB, RGBA and grey (Y) EXRs are read.");
        const bool has_alpha = channels.findChannel("A") != nullptr;
        const std::size_t n = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
        std::vector<float> r(n), g(rgb ? n : 0), b(rgb ? n : 0), a(has_alpha ? n : 0);
        Imf::FrameBuffer fb;
        // OpenEXR addresses a slice by the data window's own coordinates: the base is moved back by its origin.
        const std::ptrdiff_t origin = static_cast<std::ptrdiff_t>(dw.min.x) + static_cast<std::ptrdiff_t>(dw.min.y) * w;
        auto slice = [&](const char* name, std::vector<float>& v, double fill) {
            char* base = reinterpret_cast<char*>(v.data()) - origin * static_cast<std::ptrdiff_t>(sizeof(float));
            fb.insert(name, Imf::Slice(Imf::FLOAT, base, sizeof(float), sizeof(float) * static_cast<std::size_t>(w), 1, 1, fill));
        };
        if (rgb) {
            slice("R", r, 0.0), slice("G", g, 0.0), slice("B", b, 0.0);
        } else {
            slice("Y", r, 0.0);
        }
        if (has_alpha) slice("A", a, 1.0);
        file.setFrameBuffer(fb);
        file.readPixels(dw.min.y, dw.max.y);

        auto code = [](float v) { return std::isfinite(v) ? std::pow(std::clamp(v, 0.0f, 1.0f), 1.0f / 2.4f) : 0.0f; };
        PlanarBuffer rgb_codes(3, h, w);
        const std::vector<float>* src[3] = {&r, rgb ? &g : &r, rgb ? &b : &r};
        for (int c = 0; c < 3; ++c) {
            float* dst = rgb_codes.plane(c);
            const std::vector<float>& s = *src[c];
            for (std::size_t i = 0; i < n; ++i) dst[i] = code(s[i]);
        }
        DecodedStill out;
        out.source = "openexr (EXR: linear Rec.709, as BT.1886 codes)";
        out.bits = 32;
        std::vector<float> green(src[1]->begin(), src[1]->end());
        std::sort(green.begin(), green.end());
        out.distinct_codes = static_cast<int>(std::unique(green.begin(), green.end()) - green.begin());
        if (has_alpha) {
            PlanarBuffer alpha(1, h, w);
            float* dst = alpha.plane(0);
            for (std::size_t i = 0; i < n; ++i) dst[i] = std::isfinite(a[i]) ? std::clamp(a[i], 0.0f, 1.0f) : 0.0f;
            out.alpha = std::move(alpha);
        }
        out.rgb = SdrImage(std::move(rgb_codes));
        return out;
    } catch (const std::exception& e) {
        return make_error(ErrorCode::ParseError, "This EXR could not be read.", e.what());
    }
}
#endif

Result<DecodedStill> decode_sdr_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return make_error(ErrorCode::NotFound, "The image file could not be opened.", path.string());
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
#if RUDRA_MEDIA_HAVE_OPENEXR
    if (bytes.size() >= 4 && bytes[0] == 0x76 && bytes[1] == 0x2f && bytes[2] == 0x31 && bytes[3] == 0x01) {   // EXR
        auto e = decode_exr_openexr(path);
        if (!e) return make_error(e.error().code, e.error().message, path.string() + (e.error().detail.empty() ? "" : ": " + e.error().detail));
        return e;
    }
#endif
    auto r = decode_sdr(bytes);
    if (!r) return make_error(r.error().code, r.error().message, path.string() + (r.error().detail.empty() ? "" : ": " + r.error().detail));
    return r;
}

}  // namespace rudra

namespace rudra {

SdrImage fit_max_side(const SdrImage& rgb, int max_side) {
    const auto& b = rgb.buffer();
    const int h = b.height(), w = b.width();
    if (max_side <= 0 || std::max(h, w) <= max_side) return rgb;
    // int(width * ratio), as the Python truncates.
    const double ratio = double(max_side) / double(std::max(h, w));
    const int nw = std::max(1, int(double(w) * ratio)), nh = std::max(1, int(double(h) * ratio));
    cv::Mat src(h, w, CV_32FC3);
    for (int y = 0; y < h; ++y) {
        float* row = src.ptr<float>(y);
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) row[x * 3 + c] = b.at(c, y, x);
    }
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(nw, nh), 0, 0, cv::INTER_AREA);
    PlanarBuffer out(3, nh, nw);
    for (int y = 0; y < nh; ++y) {
        const float* row = dst.ptr<float>(y);
        for (int x = 0; x < nw; ++x)
            for (int c = 0; c < 3; ++c) out.at(c, y, x) = row[x * 3 + c];
    }
    return SdrImage(std::move(out));
}

}  // namespace rudra
