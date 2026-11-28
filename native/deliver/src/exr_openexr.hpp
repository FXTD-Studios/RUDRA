#pragma once
// The EXR compressions RUDRA's own writer does not do (1.3.2, 28 Nov 2026): RLE, PIZ, PXR24, B44, B44A, DWAA and
// DWAB, written and read by the OpenEXR library (exr.cpp keeps None, ZIP and ZIPS byte for byte as before). Only
// built with RUDRA_HAVE_OPENEXR; without it exr_compression_available() says no and these are never called.
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "rudra/deliver/exr.hpp"

namespace rudra::exr_detail {

// Whether this build has the OpenEXR library.
bool openexr_available() noexcept;

// A scanline EXR of channels `names` (any order: OpenEXR writes them sorted), `planes[i]` w x h floats each, as
// half or float32, values already cleaned (exr.cpp); the header with `chroma`, the string and the typed attributes.
Result<std::vector<std::uint8_t>> openexr_encode(const std::vector<std::string>& names, const std::vector<const float*>& planes,
                                                 int w, int h, bool half, const std::optional<Chromaticities>& chroma,
                                                 const ExrAttributes& attributes, const ExrTypedAttributes& typed,
                                                 ExrCompression compression);

// The pixels of the scanline EXR in `file` as uncompressed scanlines are laid out (each row: its y, its size in bytes,
// then each channel's w values in its own pixel type, in file order), for read_exr's own reading. `channels` and
// `pixel_types` in file order, the data window from (x0, y0), w x h, all as read_exr parsed the header.
Result<std::vector<std::uint8_t>> openexr_decode_flat(const std::vector<std::uint8_t>& file, const std::vector<std::string>& channels,
                                                      const std::vector<int>& pixel_types, int x0, int y0, int w, int h);

}  // namespace rudra::exr_detail
