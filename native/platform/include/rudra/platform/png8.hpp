#pragma once
// A small PNG codec for 8-bit images (the painted masks, roadmap 3.3), with
// no library: the writer emits zlib stored blocks (no compression, a mask at
// the preview's size is a few megabytes), the reader inflates fixed and
// dynamic Huffman blocks and undoes the five scanline filters. 8-bit grey,
// grey+alpha, RGB and RGBA, non-interlaced; anything else is refused.

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "rudra/platform/result.hpp"

namespace rudra {

struct Png8 {
    int width = 0, height = 0, channels = 0;   // channels 1, 2, 3 or 4
    std::vector<std::uint8_t> data;            // height x width x channels, top-down
};

std::vector<std::uint8_t> png8_bytes(const Png8& image);
Result<void> write_png8(const std::filesystem::path& path, const Png8& image);
Result<Png8> decode_png8(std::span<const std::uint8_t> bytes);
Result<Png8> read_png8(const std::filesystem::path& path);

}  // namespace rudra
