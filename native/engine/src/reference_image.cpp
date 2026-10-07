#include "rudra/engine/reference_image.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <tuple>

#include "rudra/core/color.hpp"
#include "rudra/core/gamut.hpp"
#include "rudra/core/hdr10.hpp"
#include "rudra/deliver/exr.hpp"
#include "rudra/media/still.hpp"

namespace rudra {
namespace {

std::string lower_ext(const std::filesystem::path& p) {
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return e;
}

// The primaries an EXR's chromaticities name, by the red primary's x.
std::pair<Primaries, const char*> primaries_of_chromaticities(const std::vector<std::uint8_t>& payload) {
    if (payload.size() < 32) return {Primaries::Rec2020, "Rec.2020 assumed"};
    float rx;
    std::memcpy(&rx, payload.data(), 4);
    if (std::fabs(rx - 0.7347f) < 0.003f) return {Primaries::Ap0, "AP0 chromaticities"};
    if (std::fabs(rx - 0.713f) < 0.003f) return {Primaries::Ap1, "AP1 chromaticities"};
    if (std::fabs(rx - 0.708f) < 0.003f) return {Primaries::Rec2020, "Rec.2020 chromaticities"};
    if (std::fabs(rx - 0.680f) < 0.003f) return {Primaries::P3D65, "P3 chromaticities"};
    if (std::fabs(rx - 0.640f) < 0.003f) return {Primaries::Rec709, "Rec.709 chromaticities"};
    return {Primaries::Rec2020, "Rec.2020 assumed"};
}

}  // namespace

Result<ReferenceImage> read_reference(const std::filesystem::path& path) {
    const std::string ext = lower_ext(path);
    ReferenceImage out;
    if (ext == ".exr") {
        auto img = read_exr(path);
        if (!img) return img.error();
        Primaries prim = Primaries::Rec2020;
        const char* how = "no chromaticities, Rec.2020 assumed";
        for (const auto& [name, payload] : img->attributes)
            if (name == "chromaticities") std::tie(prim, how) = primaries_of_chromaticities(payload);
        PlanarBuffer rgb = img->pixels;
        if (rgb.channels() == 4) {
            PlanarBuffer three(3, rgb.height(), rgb.width());
            for (int c = 0; c < 3; ++c) std::copy(rgb.plane(c), rgb.plane(c) + rgb.plane_size(), three.plane(c));
            rgb = std::move(three);
        }
        if (rgb.channels() != 3) return make_error(ErrorCode::InvalidArgument, "reference EXR must carry R, G and B", path.string());
        out.linear709 = prim == Primaries::Rec709 ? rgb : convert_primaries(rgb, prim, Primaries::Rec709);
        out.note = std::string("EXR, ") + how;
        return out;
    }
    auto still = decode_sdr_file(path);
    if (!still) return still.error();
    if (still->bits < 10)
        return make_error(ErrorCode::InvalidArgument, "reference must be a 16-bit PQ frame or an EXR", path.string());
    const SdrImage& codes = still->rgb;
    PlanarBuffer nits2020(3, codes.height(), codes.width());
    const float k = 1.0f / kDiffuseWhite.v;
    for (int c = 0; c < 3; ++c) {
        const float* s = codes.buffer().plane(c);
        float* d = nits2020.plane(c);
        for (std::size_t i = 0; i < nits2020.plane_size(); ++i) d[i] = pq_eotf(s[i]) * k;
    }
    out.linear709 = convert_primaries(nits2020, Primaries::Rec2020, Primaries::Rec709);
    out.note = (ext == ".tif" || ext == ".tiff") ? "TIFF 16-bit PQ, Rec.2020" : "PNG 16-bit PQ, Rec.2020";
    return out;
}

}  // namespace rudra
