#pragma once
// EXR, ACES and OCIO writers. Port of rudra/delivery/exr.py and aces.py,
// byte for byte: uncompressed scanline OpenEXR 2.0 written directly (no
// OpenEXR library, like the Python), so the file the native app writes is
// the file the Python writes, header included.

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "rudra/core/color.hpp"
#include "rudra/core/image.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

using ExrAttributes = std::vector<std::pair<std::string, std::string>>;   // string attributes, in order
using Chromaticities = std::array<double, 8>;   // rx ry gx gy bx by wx wy
// Attributes of other types (28 Oct 2026): name, OpenEXR type ("timecode",
// "rational", "string" ...), payload bytes. Written after the string ones.
struct ExrTyped {
    std::string name, type;
    std::vector<std::uint8_t> payload;
};
using ExrTypedAttributes = std::vector<ExrTyped>;
// timeCode (SMPTE, timecode), framesPerSecond (rational) and reelName (string)
// for a frame; an invalid timecode leaves timeCode out, an empty reel reelName.
ExrTypedAttributes exr_timecode_attributes(const std::string& timecode, double fps, const std::string& reel);

extern const Chromaticities kAp0Chromaticities;
extern const Chromaticities kAp1Chromaticities;
extern const Chromaticities kRec2020Chromaticities;
extern const Chromaticities kRec709Chromaticities;   // RUDRA 1.1: the linear Rec.709 container

// 1.2.6 (readiness report, checklist item 8): lossless ZIP (OpenEXR's zlib in blocks of 16
// scanlines, compression 3), for the ACEScg and linear masters: what Nuke, Resolve and every
// OpenEXR reader take, a third to a half of the size. A build without zlib writes None. (Until 1.3.2 the ACES 2065-1
// file was always the uncompressed SMPTE ST 2065-4 container; see below.)
// 1.3 (24 Nov 2026): ZIPS as well (OpenEXR's ZIPS_COMPRESSION, compression 2: the same zlib, one
// scanline a block: a little larger than ZIP, quicker to read a few lines of); Deliver's EXR compression.
enum class ExrCompression { None, Zip, Zips, Rle, Piz, Pxr24, B44, B44a, Dwaa, Dwab };
// Whether this build can write and read ZIP (zlib found at configure time).
bool exr_zip_available() noexcept;

// 1.3.2 (28 Nov 2026): every OpenEXR compression in Deliver, on the analytic and the learned route. None, ZIP and
// ZIPS are written by RUDRA's own writer above (byte for byte as before); RLE, PIZ, PXR24, B44, B44A, DWAA and DWAB
// by the OpenEXR library, when the build has it (RUDRA_HAVE_OPENEXR), which also reads them back. Lossy: PXR24
// (float32 kept to 24 bits; half is lossless), B44 / B44A (half only; float32 channels are stored as they are) and
// DWAA / DWAB (RGB lossy; other channels, the masks among them, lossless). The ACES 2065-1 file takes any of them
// too: compressed, it is no longer the SMPTE ST 2065-4 container and says so (no acesImageContainerFlag).
// Every compression, in the order Deliver lists them.
const std::vector<ExrCompression>& exr_compressions();
// Whether this build writes and reads `c`.
bool exr_compression_available(ExrCompression c) noexcept;
// Whether `c` changes values (PXR24, B44, B44A, DWAA, DWAB).
bool exr_compression_lossy(ExrCompression c) noexcept;
// Its id in requests and settings: "none", "zip", "zips", "rle", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab".
const char* exr_compression_id(ExrCompression c) noexcept;
// Its name as OpenEXR and the hosts write it ("ZIP", "DWAA" ...).
const char* exr_compression_label(ExrCompression c) noexcept;
// An id back (case ignored); nullopt for an unknown one.
std::optional<ExrCompression> parse_exr_compression(const std::string& id);
// A request's compression: `id` when it names one, else the 1.2.6 / 1.3 keys (exr_zip, exr_zip_lines).
ExrCompression exr_compression_from(const std::string& id, bool zip, int zip_lines) noexcept;

// float32 -> IEEE half, round to nearest even (numpy's astype(np.float16)).
std::uint16_t float_to_half(float f) noexcept;

// write_exr as bytes: 3 or 4 planes, NaN -> 0, +inf -> 3.4e38, -inf -> 0,
// half clipped to +-65504. Optional mask is a matching one-plane finite [0,1]
// image stored as rudra.reconstructed, separate from RGB and alpha.
Result<std::vector<std::uint8_t>> exr_bytes(const PlanarBuffer& rgb, bool half = true,
                                            const std::optional<Chromaticities>& chromaticities = std::nullopt,
                                            const ExrAttributes& attributes = {},
                                            const PlanarBuffer* reconstruction_mask = nullptr,
                                            const ExrTypedAttributes& typed = {},
                                            ExrCompression compression = ExrCompression::None);
Result<void> write_exr(const std::filesystem::path& path, const PlanarBuffer& rgb, bool half = true,
                       const std::optional<Chromaticities>& chromaticities = std::nullopt,
                       const ExrAttributes& attributes = {},
                       const PlanarBuffer* reconstruction_mask = nullptr, const ExrTypedAttributes& typed = {},
                       ExrCompression compression = ExrCompression::None);

// write_aces_exr / write_acescg_exr: scene linear in `source`, diffuse white
// 1.0; `provenance` goes in as one sorted-keys JSON string attribute.
// `alpha` (1 Nov 2026): a matching one-plane straight alpha in [0, 1] (the
// source's), written as channel A with RGB premultiplied by it: OpenEXR's
// associated alpha, what Nuke and Resolve expect from an EXR. The header says so
// (rudra:alpha). The reconstruction mask is not premultiplied.
Result<void> write_aces_exr(const std::filesystem::path& path, const PlanarBuffer& rgb_linear,
                            Primaries source = Primaries::Rec2020, double exposure_scale = 1.0,
                            const ExrAttributes& provenance = {}, bool half = true,
                            const PlanarBuffer* reconstruction_mask = nullptr, const ExrTypedAttributes& typed = {},
                            const PlanarBuffer* alpha = nullptr, ExrCompression compression = ExrCompression::None);
Result<void> write_acescg_exr(const std::filesystem::path& path, const PlanarBuffer& rgb_linear,
                              Primaries source = Primaries::Rec2020, double exposure_scale = 1.0,
                              const ExrAttributes& provenance = {}, bool half = true,
                              const PlanarBuffer* reconstruction_mask = nullptr, const ExrTypedAttributes& typed = {},
                              const PlanarBuffer* alpha = nullptr, ExrCompression compression = ExrCompression::None);

// A mattes EXR (RUDRA 1.1, 5 Nov 2026): one-plane float32 channels by name (e.g.
// "rudra.highlight"), each finite in [0, 1] and of one size, uncompressed scanlines,
// no RGB. What Nuke / Resolve read as named channels beside the master.
using NamedPlanes = std::vector<std::pair<std::string, PlanarBuffer>>;
// 1.3.2: `compression` as the master's (Deliver's EXR compression).
Result<void> write_mattes_exr(const std::filesystem::path& path, const NamedPlanes& planes,
                              const ExrAttributes& attributes = {}, ExrCompression compression = ExrCompression::None);

// read_exr for scanline files, uncompressed or ZIP / ZIPS (1.2.6), (exr.py's reader), and (1.3.2) every other
// OpenEXR compression through the OpenEXR library when the build has it: the planes in
// R, G, B(, A) order, and the header's attributes in file order as raw bytes.
struct ExrImage {
    PlanarBuffer pixels;
    std::optional<PlanarBuffer> reconstruction_mask;
    NamedPlanes other;   // every other channel, one plane each, in file order (5 Nov 2026)
    std::vector<std::pair<std::string, std::string>> attribute_types;   // name -> type
    std::vector<std::pair<std::string, std::vector<std::uint8_t>>> attributes;   // name -> payload
    std::vector<int> pixel_types;   // per channel, file order: 1 half, 2 float
    std::vector<std::uint16_t> half_bits;   // raw halves, R, G, B(, A) planes, when every channel is half
};
Result<ExrImage> read_exr(const std::filesystem::path& path);

// generate_ocio_config's text.
std::string ocio_config_text();

// The name Python uses for a primaries set ("rec709", "rec2020", ...).
const char* primaries_name(Primaries p) noexcept;

}  // namespace rudra
