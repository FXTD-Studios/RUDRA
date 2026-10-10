#pragma once
// The reference frame for the reference match (roadmap 3.4): the graded HDR
// of the frame on screen, read into what core/reference_fit.hpp takes,
// scene-linear Rec.709 with 1.0 = 203 nits.
//
// EXR (uncompressed scanline, the files RUDRA masters write and the Python
// reads): scene-linear as stored, diffuse white at 1.0; the primaries from
// the header's chromaticities (AP0 for the ACES container, Rec.2020 for the
// linear one, Rec.709 or P3 when a file says so), Rec.2020 when there are
// none. PNG and TIFF at 16 bits: PQ code values (ST 2084) in Rec.2020,
// through the EOTF to nits.

#include <filesystem>
#include <string>

#include "rudra/core/image.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

struct ReferenceImage {
    PlanarBuffer linear709;   // 3 planes, 1.0 = 203 nits
    std::string note;         // what was read and how: "EXR, AP0 chromaticities", "PNG 16-bit PQ"
};

Result<ReferenceImage> read_reference(const std::filesystem::path& path);

}  // namespace rudra
