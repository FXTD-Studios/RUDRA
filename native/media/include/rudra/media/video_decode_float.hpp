#pragma once
// Experimental, isolated headroom decoder. No app/default pipeline uses it.
// See docs/SUPER_WHITES_REVIEW_2026-10-16.md before integrating this type.

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "rudra/core/image.hpp"
#include "rudra/media/video_probe.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {
class Process;

// Encoded RGB in the source primaries/transfer, nominal black 0 and white 1.
// Negative and >1 samples survive. This is NOT SdrImage or scene-linear HDR.
// source_contract describes the original file (including its original range),
// not the normalized values in encoded_rgb. There is no implicit conversion.
struct ExtendedVideoFrame {
    PlanarBuffer encoded_rgb;
    InputContract source_contract;
};

// Zero-based decoded-frame indices, not a rounded timestamp seek. A paused
// frame uses count=1; playback/export can use the same path for a range.
struct FloatFrameRange {
    int first = 0, count = 0;
};

// Validates the prototype's narrow eligibility before constructing arguments:
// known limited-range Rec.709 transfer, 8/10-bit YUV444, no alpha/rotation.
Result<std::vector<std::string>> float_decoder_command(const std::string& ffmpeg,
    const std::filesystem::path& source, const VideoSource& video,
    std::optional<FloatFrameRange> range = std::nullopt);

// Source clock MUST have been validated by open_video_source. For FFV1 in
// Matroska, seek/trim between adjacent validated timestamps (not at a rounded
// frame timestamp). Other codecs, containers or ambiguous clocks keep indexed
// traversal. Count/EOF/process validation is unchanged. Pixel conversion is shared.
inline constexpr std::string_view kFloatPlaybackSelection = "ffv1-mkv-cfr-midpoint-v1";
Result<std::vector<std::string>> float_playback_decoder_command(const std::string& ffmpeg,
    const std::filesystem::path& source, const VideoSource& video, FloatFrameRange range);

class FloatVideoDecoder {
public:
    ~FloatVideoDecoder();
    FloatVideoDecoder(const FloatVideoDecoder&) = delete;
    FloatVideoDecoder& operator=(const FloatVideoDecoder&) = delete;

    static Result<std::unique_ptr<FloatVideoDecoder>> open(const std::filesystem::path& source,
        const VideoSource& video, const std::filesystem::path& log,
        std::optional<FloatFrameRange> range = std::nullopt);
    // Low-level pipe seam for tests: producer must emit planar gbrpf32le,
    // exactly frames frames. Contract is retained, not applied or validated.
    static Result<std::unique_ptr<FloatVideoDecoder>> open_command(const std::vector<std::string>& argv,
        int width, int height, int frames, const InputContract& contract, const std::filesystem::path& log);

    int frames() const noexcept { return frames_; }
    int next_index() const noexcept { return index_; } // relative to the requested range
    Result<ExtendedVideoFrame> next();
    // Refuses unread, partial, extra, nonfinite or failed streams. next()'s
    // terminal errors remain failures here; an early finish can be retried.
    Result<void> finish();
    void close();
    // May run concurrently with next() OR finish(). The reader must serialize
    // next/finish; join both reader and interrupter before close/destruction.
    // The process stays allocated until close, including after finish.
    void interrupt();

private:
    FloatVideoDecoder() = default;
    Error remember(Error error);
    void stop_process();
    std::unique_ptr<Process> process_;
    std::mutex process_mutex_; // serializes kill/running/wait (including POSIX reaping)
    std::filesystem::path log_;
    InputContract contract_;
    std::optional<Error> failure_;
    std::atomic<bool> interrupted_{false};
    bool finished_ = false;
    int width_ = 0, height_ = 0, frames_ = 0, index_ = 0;
};
}  // namespace rudra
