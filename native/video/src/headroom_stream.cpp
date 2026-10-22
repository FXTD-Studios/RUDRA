#include "rudra/video/headroom_stream.hpp"

#include <algorithm>
#include <limits>
#include <mutex>

namespace rudra {
namespace fs = std::filesystem;
namespace headroom_detail { struct StreamIdentity {}; }
namespace {
std::string utf8(const fs::path& path) { const auto s = path.u8string(); return {s.begin(), s.end()}; }
Error busy() { return make_error(ErrorCode::Busy, "Headroom stream operation is already running"); }
Error invalid_token() { return make_error(ErrorCode::InvalidArgument, "Frame or receipt belongs to another headroom stream"); }
}

struct HeadroomStream::Impl {
    PreparedHeadroomSource::Handle prepared;
    std::shared_ptr<HeadroomPixelBudget> budget;
    const VideoSource& video() const { return prepared->video(); }
    fs::path log;
    std::size_t bytes = 0, capacity = 0, reserved = 0, peak = 0;
    int delivered = 0;
    std::uint64_t starts = 0;
    bool complete = false;
    std::optional<FloatFrameRange> range;
    std::optional<Error> failure;
    const std::shared_ptr<const headroom_detail::StreamIdentity> identity =
        std::make_shared<const headroom_detail::StreamIdentity>();
    std::vector<std::weak_ptr<const ExtendedVideoFrame>> allocations;
    std::unique_ptr<FloatVideoDecoder> decoder;
    mutable std::mutex mutex;
    std::mutex operation;

    std::size_t live() const {
        return reserved + std::size_t(std::count_if(allocations.begin(), allocations.end(),
            [](const auto& p) { return !p.expired(); }));
    }
    Error fail(Error error) {
        if (!failure) failure = std::move(error); // preserve integrity/first failure
        complete = false;
        if (decoder) decoder->interrupt();
        return *failure;
    }
    Result<void> verify() {
        if (failure) return *failure;
        auto valid = prepared->verify(); if (!valid) return fail(valid.error());
        return {};
    }
};

HeadroomStream::HeadroomStream() : d_(std::make_unique<Impl>()) {}
HeadroomStream::~HeadroomStream() { cancel(); }

Result<std::unique_ptr<HeadroomStream>> HeadroomStream::open(const fs::path& source,
    HeadroomPolicy policy, const fs::path& decoder, std::size_t pixel_bytes, const fs::path& log,
    HeadroomPrepareOptions options) {
    if (policy != HeadroomPolicy::Bt1886V1)
        return make_error(ErrorCode::Unsupported, "Headroom stream requires an explicit supported policy");
    if (!pixel_bytes) return make_error(ErrorCode::InvalidArgument, "Invalid headroom stream budget");
    auto prepared = PreparedHeadroomSource::prepare(source, policy, decoder, options);
    if (!prepared) return prepared.error();
    return open(std::move(*prepared), pixel_bytes, log);
}

Result<std::unique_ptr<HeadroomStream>> HeadroomStream::open(PreparedHeadroomSource::Handle prepared,
    std::size_t pixel_bytes, const fs::path& log, std::shared_ptr<HeadroomPixelBudget> shared_budget) {
    if (!prepared || !pixel_bytes) return make_error(ErrorCode::InvalidArgument, "Invalid headroom source or budget");
    try {
        std::unique_ptr<HeadroomStream> out(new HeadroomStream());
        auto& d = *out->d_;
        d.prepared = std::move(prepared);
        auto valid = d.verify(); if (!valid) return valid.error();
        const auto w = std::size_t(d.video().stream.width), h = std::size_t(d.video().stream.height);
        if (w > std::numeric_limits<std::size_t>::max() / h / (3 * sizeof(float)))
            return make_error(ErrorCode::InvalidArgument, "Headroom frame size overflows");
        d.bytes = w * h * 3 * sizeof(float); d.capacity = pixel_bytes / d.bytes;
        if (!d.capacity) return make_error(ErrorCode::InvalidArgument, "Headroom budget cannot hold one full frame");
        d.budget = shared_budget ? std::move(shared_budget) : std::make_shared<HeadroomPixelBudget>(pixel_bytes);
        if (d.budget->stats().limit < d.bytes)
            return make_error(ErrorCode::InvalidArgument, "Shared headroom budget cannot hold one full frame");
        d.log = log;
        return out;
    } catch (const std::exception& e) {
        return make_error(ErrorCode::IoError, "Cannot open headroom stream", e.what());
    }
}

const VideoSource& HeadroomStream::video() const { return d_->video(); }
const std::string& HeadroomStream::cache_id() const { return d_->prepared->cache_id(); }

Result<void> HeadroomStream::start(FloatFrameRange range) {
    std::unique_lock operation(d_->operation, std::try_to_lock);
    if (!operation.owns_lock()) return busy();
    std::lock_guard lock(d_->mutex);
    auto valid = d_->verify(); if (!valid) return valid.error();
    if (d_->range) return make_error(ErrorCode::InvalidArgument, "Headroom stream already started; open a new owner to seek");
    if (range.first < 0 || range.first >= d_->video().clock.frames || range.count <= 0 ||
        range.count > d_->video().clock.frames - range.first)
        return make_error(ErrorCode::InvalidArgument, "Headroom stream range is outside the movie");
    auto command = float_playback_decoder_command(utf8(d_->prepared->decoder_path()), d_->prepared->source_path(), d_->video(), range);
    if (!command) return command.error();
    auto decoder = FloatVideoDecoder::open_command(*command, d_->video().stream.width, d_->video().stream.height,
                                                 range.count, d_->video().contract, d_->log);
    if (!decoder) return d_->fail(decoder.error());
    d_->range = range; d_->decoder = std::move(*decoder); ++d_->starts;
    return {};
}

Result<HeadroomPreviewFrame> HeadroomStream::next() {
    std::unique_lock operation(d_->operation, std::try_to_lock);
    if (!operation.owns_lock()) return busy();
    std::unique_lock lock(d_->mutex);
    auto valid = d_->verify(); if (!valid) return valid.error();
    if (!d_->range || d_->delivered == d_->range->count)
        return make_error(ErrorCode::InvalidArgument, "Start a range or finish after reading its expected frames");
    std::erase_if(d_->allocations, [](const auto& p) { return p.expired(); });
    if (d_->live() >= d_->capacity)
        return make_error(ErrorCode::Busy, "Release a retained headroom frame before decoding another");
    try {
        auto reservation = d_->budget->reserve(d_->bytes);
        if (!reservation) return reservation.error();
        d_->reserved = 1; d_->peak = std::max(d_->peak, d_->live() * d_->bytes);
        lock.unlock();
        auto frame = d_->decoder->next(); // finite and exact frame bytes checked by decoder
        lock.lock();
        d_->reserved = 0;
        valid = d_->verify(); if (!valid) return valid.error();
        if (!frame) return d_->fail(frame.error());
        const auto& b = frame->encoded_rgb;
        if (b.channels() != 3 || b.width() != d_->video().stream.width || b.height() != d_->video().stream.height ||
            b.span().size_bytes() != d_->bytes || frame->source_contract != d_->video().contract)
            return d_->fail(make_error(ErrorCode::IntegrityError, "Invalid headroom stream payload"));
        struct Storage {
            // Pixels must be destroyed before their allowance is released.
            std::shared_ptr<HeadroomPixelBudget::Reservation> reservation;
            ExtendedVideoFrame frame;
        };
        auto storage = std::make_shared<Storage>(Storage{std::move(*reservation), std::move(*frame)});
        std::shared_ptr<const ExtendedVideoFrame> held(storage, &storage->frame);
        d_->allocations.push_back(held);
        return HeadroomPreviewFrame(std::move(held), d_->identity, d_->range->first + d_->delivered++);
    } catch (const std::exception& e) {
        if (!lock.owns_lock()) lock.lock();
        d_->reserved = 0;
        return d_->fail(make_error(ErrorCode::IoError, "Headroom stream read failed", e.what()));
    }
}

Result<void> HeadroomStream::validate_preview(const HeadroomPreviewFrame& frame) {
    std::lock_guard lock(d_->mutex);
    if (frame.identity_ != d_->identity || !frame.frame_) return invalid_token();
    return d_->verify();
}

Result<HeadroomDecodeReceipt> HeadroomStream::finish() {
    std::unique_lock operation(d_->operation, std::try_to_lock);
    if (!operation.owns_lock()) return busy();
    std::unique_lock lock(d_->mutex);
    auto valid = d_->verify(); if (!valid) return valid.error();
    if (!d_->range || d_->delivered != d_->range->count)
        return make_error(ErrorCode::InvalidArgument, "Read all expected headroom frames before completion");
    if (d_->complete) return HeadroomDecodeReceipt(d_->identity);
    try {
        lock.unlock();
        auto done = d_->decoder->finish();
        lock.lock();
        valid = d_->verify(); if (!valid) return valid.error();
        if (!done) return d_->fail(done.error());
        d_->complete = true;
        return HeadroomDecodeReceipt(d_->identity);
    } catch (const std::exception& e) {
        if (!lock.owns_lock()) lock.lock();
        return d_->fail(make_error(ErrorCode::IoError, "Headroom stream completion failed", e.what()));
    }
}

Result<void> HeadroomStream::validate_completion(const HeadroomDecodeReceipt& receipt) {
    std::lock_guard lock(d_->mutex);
    if (receipt.identity_ != d_->identity) return invalid_token();
    auto valid = d_->verify(); if (!valid) return valid.error();
    if (!d_->complete) return make_error(ErrorCode::Busy, "Headroom decode has not completed");
    return {};
}

Result<HeadroomFrameSource::Frame> HeadroomStream::verified_frame(
    const HeadroomPreviewFrame& frame, const HeadroomDecodeReceipt& receipt) {
    std::lock_guard lock(d_->mutex);
    if (frame.identity_ != d_->identity || !frame.frame_ || receipt.identity_ != d_->identity) return invalid_token();
    auto valid = d_->verify(); if (!valid) return valid.error();
    if (!d_->complete) return make_error(ErrorCode::Busy, "Cannot reuse provisional headroom pixels");
    return frame.frame_;
}

void HeadroomStream::cancel() {
    std::lock_guard lock(d_->mutex);
    d_->fail(make_error(ErrorCode::Cancelled, "Headroom stream was cancelled"));
}
void HeadroomStream::close() {
    cancel();
    std::lock_guard operation(d_->operation);
    std::lock_guard lock(d_->mutex);
    if (d_->decoder) d_->decoder->close();
}
HeadroomFrameStats HeadroomStream::stats() const {
    std::lock_guard lock(d_->mutex);
    return {d_->bytes, d_->capacity, d_->live() * d_->bytes, 0,
            std::uint64_t(d_->delivered), 0, d_->starts, d_->peak, d_->complete};
}
}  // namespace rudra
