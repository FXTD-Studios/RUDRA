#pragma once
// Inputs of the ACES v2 master (27 Oct 2026): stills and videos. The ACES path
// takes display code values as they are (the SDR came through the ACES 2.0
// view), so a video is decoded to full-range 16-bit RGB with its YCbCr matrix
// (swscale, accurate rounding, full chroma: the Python export's decode)
// and range, and its transfer and primaries tags are not used. An untagged
// YCbCr video is read as BT.709, limited range, as tools/rudra_aces_export.py
// reads it. Every frame of a video is mastered, named <stem>_<NNNNNN>.exr.

#include <filesystem>
#include <memory>
#include <string>

#include "rudra/media/video_decode.hpp"
#include "rudra/media/video_probe.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

bool aces_is_still(const std::filesystem::path& p);   // .png .tif .tiff .jpg .jpeg .bmp .webp
bool aces_is_video(const std::filesystem::path& p);   // .mp4 .mov .mkv .avi .m4v .mxf .webm .mpg .mpeg .ts .mts .m2ts .wmv

// The decode contract for the ACES path from ffprobe's view of the stream.
VideoArgs aces_video_args(const VideoStreamInfo& stream);
// Probes the file and returns what the decoder needs (frame count in clock.frames).
Result<VideoSource> aces_open_video(const std::filesystem::path& file);
// Effective bits of the stored samples (8, 10, 12 or 16), from the pixel format.
int aces_video_bits(const std::string& pix_fmt);
// "<stem>_000012.exr"
std::string aces_video_exr_name(const std::filesystem::path& video, int index);

// Every frame in order (full-range 16-bit RGB).
Result<std::unique_ptr<VideoDecoder>> aces_open_video_decoder(const std::filesystem::path& file, const VideoSource& video,
                                                              const std::filesystem::path& log);
// One frame by index (0-based), for previews; decodes from the start up to it.
Result<RawFrame16> aces_read_video_frame(const std::filesystem::path& file, const VideoSource& video, int index,
                                         const std::filesystem::path& log);

}  // namespace rudra
