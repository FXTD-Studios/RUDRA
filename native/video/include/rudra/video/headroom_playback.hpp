#pragma once

#include "rudra/video/headroom_stream.hpp"

namespace rudra {
namespace headroom_detail { struct PlaybackIdentity; }

class HeadroomPlaybackFrame {
public:
    const ExtendedVideoFrame& pixels() const { return preview_.pixels(); }
    int index() const { return preview_.index(); }
    std::uint64_t generation() const { return generation_; }
private:
    friend class HeadroomPlayback;
    HeadroomPlaybackFrame(HeadroomPreviewFrame preview,
        std::shared_ptr<const headroom_detail::PlaybackIdentity> identity, std::uint64_t generation)
        : preview_(std::move(preview)), identity_(std::move(identity)), generation_(generation) {}
    HeadroomPreviewFrame preview_;
    std::shared_ptr<const headroom_detail::PlaybackIdentity> identity_;
    std::uint64_t generation_;
};

class HeadroomPlaybackReceipt {
private:
    friend class HeadroomPlayback;
    HeadroomPlaybackReceipt(HeadroomDecodeReceipt receipt,
        std::shared_ptr<const headroom_detail::PlaybackIdentity> identity, std::uint64_t generation)
        : receipt_(std::move(receipt)), identity_(std::move(identity)), generation_(generation) {}
    HeadroomDecodeReceipt receipt_;
    std::shared_ptr<const headroom_detail::PlaybackIdentity> identity_;
    std::uint64_t generation_;
};

struct HeadroomPlaybackLimits {
    std::size_t pixel_bytes = 256u * 1024u * 1024u;
    std::size_t queued_frames = 2;
};
struct HeadroomPlaybackStats {
    std::uint64_t generation = 0, decoded_frames = 0, decoder_starts = 0;
    std::size_t queued_frames = 0;
    HeadroomPixelStats pixels;
    bool preparing = false, complete = false;
};

// Isolated asynchronous adapter; not an app/session/export integration.
// One decoding worker reaps the previous decoder before starting a new one.
// A cancellation worker interrupts probe/read/finish off the request thread.
// Requests only replace the pending generation: no file I/O, decoding or joins.
// Shared pixel accounting includes old borrowed/promoted frames after seeks.
// It does not bound FFmpeg or explicit caller copies. Releasing old frames is
// required for progress when they fill the budget; Busy does not skip a frame.
//
// take() returns PROVISIONAL pixels. Validate again before presentation, and
// serialize the final presentation decision with the caller's seek handling.
// A successful check is not a promise that a later request cannot invalidate it.
// Receipts prove this generation's decode, never export/file publication.
// All calls except destruction are thread-safe. Join external callers before
// destruction, which cancels and joins the two owned workers.
class HeadroomPlayback {
public:
    using Generation = std::uint64_t;
    using Frame = HeadroomPlaybackFrame;
    using Receipt = HeadroomPlaybackReceipt;
    static Result<std::unique_ptr<HeadroomPlayback>> create(HeadroomPlaybackLimits limits = {},
                                                           const std::filesystem::path& log = {});
    ~HeadroomPlayback();
    HeadroomPlayback(const HeadroomPlayback&) = delete;
    HeadroomPlayback& operator=(const HeadroomPlayback&) = delete;
    Result<Generation> open(const std::filesystem::path& source, HeadroomPolicy policy,
        const std::filesystem::path& decoder, FloatFrameRange range,
        const std::filesystem::path& probe_executable = {});
    Result<Generation> open(PreparedHeadroomSource::Handle prepared, FloatFrameRange range);
    // While preparing, replaces the requested range and cancels/restarts the
    // unfinished preparation. Once ready, shares that preparation across seeks.
    Result<Generation> seek(FloatFrameRange range);
    Result<VideoSource> video(Generation generation, std::chrono::milliseconds wait = {});
    Result<Frame> take(Generation generation, std::chrono::milliseconds wait = {});
    Result<Receipt> completion(Generation generation, std::chrono::milliseconds wait = {});
    Result<void> validate_preview(const Frame& frame);
    Result<void> validate_completion(const Receipt& receipt);
    Result<HeadroomFrameSource::Frame> verified_frame(const Frame& frame, const Receipt& receipt);
    void cancel(); // invalidates immediately; worker cleanup is asynchronous
    HeadroomPlaybackStats stats() const;
private:
    HeadroomPlayback();
    struct Impl;
    std::unique_ptr<Impl> d_;
};
} // namespace rudra
