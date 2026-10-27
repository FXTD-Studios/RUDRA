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
};

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
    std::unique_ptr<VideoDecoder> decoder_;
};

// One frame of a still (frame ignored) or a video, for the preview.
Result<AcesFrameIn> aces_read_one(const std::filesystem::path& source, int frame, const std::filesystem::path& decode_log);

}  // namespace rudra
