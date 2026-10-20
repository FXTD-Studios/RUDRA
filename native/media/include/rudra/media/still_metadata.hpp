#pragma once
// What a still says about its own colour and orientation (roadmap 1.1, 16 Oct
// 2026). OpenCV's IMREAD_UNCHANGED keeps every code but reads no metadata, so
// a Display P3 phone photo was mastered as Rec.709 and a portrait one came out
// sideways. This reads, from the file's bytes and without a colour library:
//   - the embedded ICC profile (PNG iCCP, JPEG APP2, TIFF tag 34675, WebP ICCP):
//     its rXYZ/gXYZ/bXYZ against the D50-adapted primaries of sRGB/Rec.709,
//     Display P3, Rec.2020 and Adobe RGB (1998); and its description;
//   - PNG cICP (primaries code points 1, 9, 12), sRGB and cHRM chunks;
//   - the EXIF orientation (JPEG APP1, PNG eXIf, TIFF IFD0, WebP EXIF).
// An untagged file is Rec.709, as before. A profile that matches none of the
// four is reported (`recognised` false) and read as Rec.709: never a silent
// conversion, never a silent guess.

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "rudra/core/color.hpp"

namespace rudra {

struct StillColour {
    Primaries primaries = Primaries::Rec709;   // what the master converts from
    bool tagged = false;                       // the file said something about its colour
    bool recognised = true;                    // ... and it was one of the four
    std::string tag;                           // "icc", "cicp", "srgb", "chrm" or "" (untagged)
    std::string profile;                       // the ICC description, when there is one
    int orientation = 1;                       // EXIF 1..8, 1 = as stored
    // One line for the log and the Source panel: what was found and what is done.
    std::string note() const;
};

StillColour read_still_colour(std::span<const std::uint8_t> bytes);

// Primaries from an ICC profile's colorant tags (D50, s15Fixed16), within 0.01
// per component; nullopt for none of the four, a non-RGB or a malformed profile.
std::optional<Primaries> icc_primaries(std::span<const std::uint8_t> icc, std::string* description = nullptr);

// zlib (RFC 1950/1951) inflate, for PNG iCCP. Empty on malformed input.
std::vector<std::uint8_t> zlib_inflate(std::span<const std::uint8_t> data, std::size_t limit = 64u << 20);

// EXIF orientation applied to planes stored row-major (c, y, x): the result's
// width and height swap for 5..8.
struct Oriented {
    std::vector<float> data;
    int width = 0, height = 0;
};
Oriented apply_orientation(std::span<const float> planes, int channels, int width, int height, int orientation);

const char* primaries_id(Primaries p) noexcept;   // master source_space ids: rec709, p3d65, rec2020, adobe_rgb ...

}  // namespace rudra
