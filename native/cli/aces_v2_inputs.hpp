#pragma once
// The frames an ACES v2 master reads (27 Oct 2026): stills, a folder of
// stills, or videos, in order. Shared by `aces-v2` and `aces-v2-serve`.

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "rudra/core/image.hpp"
#include "rudra/media/aces_frames.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

struct AcesFrameIn {
    PlanarBuffer codes;        // display code values in [0, 1]
    int bits = 8;
    std::string source;        // "a.png", or "clip.mp4 frame 12"
    std::string source_path;   // the file it came from
    std::string exr;           // output file name
    // A movie's frame (28 Oct 2026): its timecode (the source's start timecode
    // plus the frame number; "" without one), reel name and rate.
    int frame = -1;
    std::string timecode, reel;
    double fps = 0;
};

// A movie's start timecode and reel name from its ffprobe streams (a tmcd
// track's or the video stream's tags "timecode" and "reel_name"); "" when absent.
struct AcesSourceTimecode {
    std::string start, reel;
};
AcesSourceTimecode aces_source_timecode(const std::string& streams_json);
double aces_video_fps(const VideoSource& vs);

// A folder: its stills, sorted. A still or a video: itself. Anything else: none.
std::vector<std::filesystem::path> aces_expand_input(const std::filesystem::path& input);

class AcesInputs {
public:
    // Probes every video, names every output, and refuses two frames mapping to
    // one EXR name (case-insensitive) before anything is mastered.
    static Result<std::unique_ptr<AcesInputs>> open(const std::vector<std::filesystem::path>& sources,
                                                    const std::filesystem::path& decode_log);
    ~AcesInputs();
    std::size_t total() const noexcept { return total_; }
    // The first video's frame rate, else 24 (stills).
    double fps() const;
    // Only frames first .. last (inclusive) of a single video (a shot's in and out).
    Result<void> set_range(int first, int last);
    // The next frame; call total() times.
    Result<AcesFrameIn> next();

private:
    AcesInputs() = default;
    struct Item {
        std::filesystem::path path;
        bool video = false;
        VideoSource source;
    };
    std::vector<Item> items_;
    std::filesystem::path log_;
    std::size_t total_ = 0, item_ = 0;
    int frame_ = 0;
    int first_ = 0, count_ = -1;   // a range of the one video (-1: every frame)
    std::unique_ptr<VideoDecoder> decoder_;
};

// One frame of a still (frame ignored) or a video, for the preview.
Result<AcesFrameIn> aces_read_one(const std::filesystem::path& source, int frame, const std::filesystem::path& decode_log);

}  // namespace rudra
