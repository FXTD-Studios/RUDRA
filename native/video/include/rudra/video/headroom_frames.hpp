#pragma once

#include <chrono>
#include <memory>

#include "rudra/video/headroom_source.hpp"

namespace rudra {
struct HeadroomFrameLimits {
    // Pixel allocations owned by this source, including decoder staging and
    // frames still held by consumers after eviction. Excludes FFmpeg/OS memory,
    // metadata, and any copies a caller explicitly creates. Full resolution.
    std::size_t pixel_bytes = 256u * 1024u * 1024u;
    int batch_frames = 2;
};

struct HeadroomFrameStats {
    std::size_t frame_bytes = 0, capacity_frames = 0, live_pixel_bytes = 0;
    std::size_t cached_frames = 0;
    std::uint64_t decoded_frames = 0, cache_hits = 0, decoder_starts = 0;
    std::size_t peak_pixel_bytes = 0;
    bool complete = false;
};

// Isolated CPU source owner + bounded memory cache + asynchronous feeder.
// Windows currently supplies enforced read leases; other platforms fail closed.
// Open probes and hashes under leases and may be expensive; run off the UI thread.
// Share preparation across owners without repeating probe/hash. Budgets remain
// per owner. Observed identity changes require fresh preparation, not request().
// Changes of source/policy/decoder require a new owner. No persistent cache or PNG.
class HeadroomFrameSource {
public:
    using Frame = std::shared_ptr<const ExtendedVideoFrame>;
    static Result<std::unique_ptr<HeadroomFrameSource>> open(
        const std::filesystem::path& source, HeadroomPolicy policy,
        const std::filesystem::path& decoder, HeadroomFrameLimits limits = {},
        const std::filesystem::path& log = {}, HeadroomPrepareOptions options = {});
    static Result<std::unique_ptr<HeadroomFrameSource>> open(PreparedHeadroomSource::Handle prepared,
        HeadroomFrameLimits limits = {}, const std::filesystem::path& log = {});
    ~HeadroomFrameSource();
    HeadroomFrameSource(const HeadroomFrameSource&) = delete;
    HeadroomFrameSource& operator=(const HeadroomFrameSource&) = delete;

    const VideoSource& video() const;
    const std::string& cache_id() const;
    // A new request cancels the previous range, retaining only verified cache
    // entries. A paused frame is {index,1}; playback uses {first,count}.
    Result<void> request(FloatFrameRange range);
    // Monotonic within one request; use request() for a backward seek. A cached
    // earlier frame can still be read. Busy = timeout, NotFound = evicted frame;
    // cancellation and decoder failures have distinct errors. No silent fallback.
    // Release old Frame handles to let a memory-limited feeder advance.
    Result<Frame> take(int index, std::chrono::milliseconds wait = std::chrono::milliseconds(4000));
    void cancel();
    HeadroomFrameStats stats() const;

private:
    HeadroomFrameSource();
    struct Impl;
    std::unique_ptr<Impl> d_;
};
}  // namespace rudra
