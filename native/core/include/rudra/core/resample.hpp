#pragma once
// RUDRA 1.3 (23 Nov 2026): Deliver's Output size. A clean resize of a scene-linear HDR
// frame, never an AI upscale: no detail is invented.
//
// Upscaling is Lanczos 3, done in linear light, with each pass held between the two
// source samples nearest the new one (an anti-ringing clamp, as mpv's antiring 1.0).
// Lanczos alone rings by up to about a tenth of an edge's step, which next to a
// 1,000-nit lamp is a visible dark halo, or a negative; the clamp keeps every new
// sample inside its neighbours, so a highlight gains no halo and nothing goes below
// its darkest neighbour. Downscaling is a tent of the step's width: weights that are
// never negative, so it cannot ring at all.

#include <rudra/core/image.hpp>
#include <rudra/core/master.hpp>

#include <string>

namespace rudra {

struct OutputSize {
    int width = 0, height = 0;
    bool operator==(const OutputSize&) const = default;
};

// "source", "2x", "uhd" (3840 wide) or "hd" (1920 wide).
bool valid_output_size(const std::string& choice);

// 1.3 (24 Nov 2026, audit): the largest side a resize writes. 2x of a 6K or 8K source would be
// 12K-16K (about 1.6 GB a float frame, and past HEVC's 8192 x 4320): such a choice keeps the
// source's size instead, and Deliver greys it out (output_size_fits).
constexpr int kMaxOutputSide = 8192;

// The size a width x height frame is written at. "source" (and anything unknown) keeps
// it as it is; the others keep the picture's shape and round both sides to even, which
// ProRes 4:2:2 and HEVC 4:2:0 need. A size with a side over kMaxOutputSide is not written:
// the source's size is returned.
OutputSize output_size(const std::string& choice, int width, int height);
// Whether the choice gives that size for this source ("source" always does).
bool output_size_fits(const std::string& choice, int width, int height);

// "3840 x 2160 from 1920 x 1080 (2x, Lanczos 3 with anti-ringing)": the EXR attribute.
std::string output_size_note(const std::string& choice, int src_width, int src_height);

PlanarBuffer resample_hdr(const PlanarBuffer& x, int height, int width);
NitsFrame resample_hdr(const NitsFrame& x, int height, int width);

}  // namespace rudra
