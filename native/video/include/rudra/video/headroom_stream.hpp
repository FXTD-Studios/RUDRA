#pragma once

#include <memory>
#include "rudra/video/headroom_budget.hpp"
#include "rudra/video/headroom_frames.hpp"

namespace rudra {
namespace headroom_detail { struct StreamIdentity; }

// Immutable pixels for provisional preview or temporary export staging only.
// No implicit conversion to a verified cache Frame. Retained handles count
// against the stream's pixel budget. Pixels may outlive the owner, but their
// validity cannot: validate with that owner before use, including presentation.
class HeadroomPreviewFrame {
public:
    const ExtendedVideoFrame& pixels() const { return *frame_; }
    int index() const { return index_; }
private:
    friend class HeadroomStream;
    HeadroomPreviewFrame(std::shared_ptr<const ExtendedVideoFrame> frame,
                         std::shared_ptr<const headroom_detail::StreamIdentity> identity, int index)
        : frame_(std::move(frame)), identity_(std::move(identity)), index_(index) {}
    std::shared_ptr<const ExtendedVideoFrame> frame_;
    std::shared_ptr<const headroom_detail::StreamIdentity> identity_;
    int index_;
};

// Proof of decode completion for ONE owner/range; never an encoder, delivery,
// or file-publication receipt. Must be validated by the originating live owner.
class HeadroomDecodeReceipt {
private:
    friend class HeadroomStream;
    explicit HeadroomDecodeReceipt(std::shared_ptr<const headroom_detail::StreamIdentity> identity)
        : identity_(std::move(identity)) {}
    std::shared_ptr<const headroom_detail::StreamIdentity> identity_;
};

// Isolated synchronous pull prototype. Run reads/finish on a background thread.
// One immutable range per owner: seek/retry needs a NEW owner, so old frames or
// receipts cannot be accepted as the new generation. Existing batch feeder and
// production readers are unchanged. No persistent cache or export publication.
// cancel() can interrupt next()/finish(); all other decoding calls serialize.
// Join callers before destruction. Reuse a prepared source across new owners;
// A shared budget optionally caps retained pixels across overlapping owners.
class HeadroomStream {
public:
    static Result<std::unique_ptr<HeadroomStream>> open(
        const std::filesystem::path& source, HeadroomPolicy policy,
        const std::filesystem::path& decoder, std::size_t pixel_bytes = 256u * 1024u * 1024u,
        const std::filesystem::path& log = {}, HeadroomPrepareOptions options = {});
    static Result<std::unique_ptr<HeadroomStream>> open(PreparedHeadroomSource::Handle prepared,
        std::size_t pixel_bytes = 256u * 1024u * 1024u, const std::filesystem::path& log = {},
        std::shared_ptr<HeadroomPixelBudget> shared_budget = {});
    ~HeadroomStream();
    HeadroomStream(const HeadroomStream&) = delete;
    HeadroomStream& operator=(const HeadroomStream&) = delete;
    const VideoSource& video() const;
    const std::string& cache_id() const;
    Result<void> start(FloatFrameRange range);
    // Busy if retained handles fill the budget; release a handle and retry.
    // No frame is skipped or decoder restarted under memory pressure.
    Result<HeadroomPreviewFrame> next();
    Result<void> validate_preview(const HeadroomPreviewFrame& frame);
    // All expected frames must have been returned. Exact EOF, successful child
    // exit, current identity and non-cancellation are required. Idempotent.
    Result<HeadroomDecodeReceipt> finish();
    Result<void> validate_completion(const HeadroomDecodeReceipt& receipt);
    // Only retained pixels can be promoted after finish. Check the receipt
    // again on later reuse; naked shared_ptrs cannot be physically revoked.
    Result<HeadroomFrameSource::Frame> verified_frame(
        const HeadroomPreviewFrame& frame, const HeadroomDecodeReceipt& receipt);
    void cancel(); // terminal; invalidates previews and completion, even after finish
    // Blocking interruption + reaping. Use off the UI thread before launching
    // a replacement decoder, even if another validator retains this owner.
    void close();
    HeadroomFrameStats stats() const;
private:
    HeadroomStream();
    struct Impl;
    std::unique_ptr<Impl> d_;
};
}  // namespace rudra
