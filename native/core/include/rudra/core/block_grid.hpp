#pragma once
// RUDRA 1.3 (24 Nov 2026): how strongly a picture shows a compression block grid, and the cleanup strength
// that takes it away without smoothing more than that (Reconstruct > Clean compression).
#include <cstddef>
#include <cstdint>
#include <span>

namespace rudra {

// Luma, 8-bit, row-major (`stride` bytes a row): the mean step across the 8-pixel block edges, horizontal and
// vertical, over the mean step everywhere else. A clean frame measures 1.00; on Netflix Meridian frames
// (24 Nov 2026) JPEG quality 50 measured 1.66, quality 75 1.38, x264 CRF 35 1.22 and CRF 28 1.10. A picture
// with no steps at all (flat) measures 1.0.
double block_grid(const std::uint8_t* luma, int width, int height, std::ptrdiff_t stride) noexcept;

// The cleanup strength: the first of `tried` (ffmpeg spp's qp, ascending; 0 = no cleanup) whose cleaned frames
// measured at or under `target`, `measured[i]` being block_grid of the frames cleaned at tried[i]; the last
// tried when none did. On the Meridian frames the first qp at or under 1.10 gained 0.03-0.43 dB PSNR on average
// at every level tried and lost at most 0.08 dB on any one frame; a target of 1.05 smoothed too much (up to
// -2.6 dB). Empty or mismatched: 0.
int choose_cleanup_qp(std::span<const int> tried, std::span<const double> measured, double target = 1.10) noexcept;

}  // namespace rudra
