#include "rudra/video/headroom_frames.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <map>
#include <mutex>
#include <thread>


namespace rudra {
namespace fs = std::filesystem;
namespace {
Error cancelled() { return make_error(ErrorCode::Cancelled, "Headroom frame request was cancelled"); }


std::string utf8(const fs::path& path) {
    const auto s = path.u8string(); return {s.begin(), s.end()};
}
}  // namespace

struct HeadroomFrameSource::Impl {
    PreparedHeadroomSource::Handle prepared;
    const VideoSource& video() const { return prepared->video(); }
    fs::path log;
    HeadroomFrameLimits limits;
    std::size_t frame_bytes = 0, capacity = 0, reserved = 0;
    std::uint64_t generation = 0, decoded = 0, hits = 0, decoder_starts = 0;
    std::size_t peak_pixels = 0;
    std::optional<FloatFrameRange> range;
    int floor = 0;
    bool pending = false, stopping = false, complete = false, worker_failed = false;
    std::optional<Error> failure;
    std::map<int, Frame> cache;
    std::vector<std::weak_ptr<const ExtendedVideoFrame>> allocations;
    std::unique_ptr<FloatVideoDecoder> active;
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::thread worker;

    Result<void> verify() const {
        return prepared->verify();
    }
    std::size_t live() const {
        return reserved + std::size_t(std::count_if(allocations.begin(), allocations.end(),
            [](const auto& p) { return !p.expired(); }));
    }
    void fail(Error error) {
        failure = std::move(error); cache.clear(); complete = false; changed.notify_all();
    }
    void run() {
        try { run_checked(); }
        catch (const std::exception& e) {
            std::lock_guard lock(mutex);
            active.reset(); reserved = 0; worker_failed = true;
            fail(make_error(ErrorCode::IoError, "Headroom feeder failed", e.what()));
        }
    }
    void run_checked() {
        std::unique_lock lock(mutex);
        while (!stopping) {
            changed.wait(lock, [&] { return stopping || pending; });
            if (stopping) break;
            pending = false;
            const auto epoch = generation;
            const int end = range->first + range->count;
            int next = range->first;
            while (!stopping && epoch == generation && !failure) {
                next = std::max(next, floor);
                if (next >= end) { complete = true; changed.notify_all(); break; }
                if (cache.contains(next)) { ++next; continue; }
                std::erase_if(allocations, [](const auto& p) { return p.expired(); });
                // Keep unread ahead frames. Old/past frames are evictable; live
                // consumer handles still consume their allocation slots.
                const auto desired = std::min({capacity, std::size_t(limits.batch_frames), std::size_t(end - next)});
                for (auto it = cache.begin(); capacity - live() < desired && it != cache.end();) {
                    if (it->first < floor || it->first >= end) it = cache.erase(it);
                    else ++it;
                }
                const auto room = capacity - live();
                if (!room) {
                    // Consumer shared_ptr release has no callback into this
                    // owner. A short bounded wait also responds to cancel/seek.
                    changed.wait_for(lock, std::chrono::milliseconds(10));
                    continue;
                }
                int count = int(std::min({room, std::size_t(limits.batch_frames), std::size_t(end - next)}));
                for (int i = 1; i < count; ++i)
                    if (cache.contains(next + i)) { count = i; break; }
                auto valid = verify();
                if (!valid) { fail(valid.error()); break; }
                auto command = float_playback_decoder_command(utf8(prepared->decoder_path()), prepared->source_path(), video(), FloatFrameRange{next, count});
                if (!command) { fail(command.error()); break; }
                auto decoder = FloatVideoDecoder::open_command(*command, video().stream.width, video().stream.height,
                                                               count, video().contract, log);
                if (!decoder) { fail(decoder.error()); break; }
                active = std::move(*decoder);
                ++decoder_starts;
                reserved = std::size_t(count);
                peak_pixels = std::max(peak_pixels, live() * frame_bytes);
                auto* reading = active.get();
                lock.unlock();
                std::vector<Frame> batch;
                std::optional<Error> error;
                for (int i = 0; i < count; ++i) {
                    auto frame = reading->next();
                    if (!frame) { error = frame.error(); break; }
                    const auto& b = frame->encoded_rgb;
                    if (b.channels() != 3 || b.width() != video().stream.width || b.height() != video().stream.height ||
                        b.span().size_bytes() != frame_bytes || frame->source_contract != video().contract ||
                        !std::all_of(b.span().begin(), b.span().end(), [](float v) { return std::isfinite(v); })) {
                        error = make_error(ErrorCode::IntegrityError, "Invalid headroom cache payload"); break;
                    }
                    batch.push_back(std::make_shared<const ExtendedVideoFrame>(std::move(*frame)));
                }
                if (!error) { auto done = reading->finish(); if (!done) error = done.error(); }
                lock.lock();
                active.reset(); // no interrupter can enter while mutex is held
                if (stopping || epoch != generation) { batch.clear(); reserved = 0; break; }
                if (failure) { batch.clear(); reserved = 0; break; }
                if (!error) { auto checked = verify(); if (!checked) error = checked.error(); }
                if (error) {
                    batch.clear(); reserved = 0;
                    if (!failure) fail(*error); // preserve an earlier source-integrity failure
                    break;
                }
                // Publish only after exact EOF and successful process exit for
                // the entire batch. No truncated/late-failing batch is cached.
                for (int i = 0; i < count; ++i) {
                    if (next + i >= floor) {
                        allocations.push_back(batch[std::size_t(i)]);
                        cache.emplace(next + i, std::move(batch[std::size_t(i)]));
                    }
                }
                batch.clear(); reserved = 0;
                decoded += std::uint64_t(count);
                next += count;
                changed.notify_all();
            }
        }
    }
};

HeadroomFrameSource::HeadroomFrameSource() : d_(std::make_unique<Impl>()) {}
HeadroomFrameSource::~HeadroomFrameSource() {
    {
        std::lock_guard lock(d_->mutex);
        d_->stopping = true;
        if (d_->active) d_->active->interrupt();
        d_->changed.notify_all();
    }
    if (d_->worker.joinable()) d_->worker.join();
}

Result<std::unique_ptr<HeadroomFrameSource>> HeadroomFrameSource::open(
    const fs::path& source, HeadroomPolicy policy, const fs::path& decoder,
    HeadroomFrameLimits limits, const fs::path& log, HeadroomPrepareOptions options) {
    if (policy != HeadroomPolicy::Bt1886V1)
        return make_error(ErrorCode::Unsupported, "Headroom frame source requires an explicit supported policy");
    if (!limits.pixel_bytes || limits.batch_frames < 1 || limits.batch_frames > 32)
        return make_error(ErrorCode::InvalidArgument, "Invalid headroom cache limits");
    auto prepared = PreparedHeadroomSource::prepare(source, policy, decoder, options);
    if (!prepared) return prepared.error();
    return open(std::move(*prepared), limits, log);
}

Result<std::unique_ptr<HeadroomFrameSource>> HeadroomFrameSource::open(PreparedHeadroomSource::Handle prepared,
    HeadroomFrameLimits limits, const fs::path& log) {
    if (!prepared || !limits.pixel_bytes || limits.batch_frames < 1 || limits.batch_frames > 32)
        return make_error(ErrorCode::InvalidArgument, "Invalid headroom source or cache limits");
    try {
        std::unique_ptr<HeadroomFrameSource> out(new HeadroomFrameSource());
        auto& d = *out->d_;
        d.prepared = std::move(prepared);
        auto valid = d.verify(); if (!valid) return valid.error();
        if (d.video().stream.width <= 0 || d.video().stream.height <= 0)
            return make_error(ErrorCode::InvalidArgument, "Invalid headroom frame dimensions");
        const auto w = std::size_t(d.video().stream.width), h = std::size_t(d.video().stream.height);
        if (w > std::numeric_limits<std::size_t>::max() / h / (3 * sizeof(float)))
            return make_error(ErrorCode::InvalidArgument, "Headroom frame size overflows");
        d.frame_bytes = w * h * 3 * sizeof(float);
        d.capacity = limits.pixel_bytes / d.frame_bytes;
        if (!d.capacity) return make_error(ErrorCode::InvalidArgument, "Headroom cache budget cannot hold one full frame");
        d.limits = limits; d.log = log;
        d.worker = std::thread([p = out.get()] { p->d_->run(); });
        return out;
    } catch (const std::exception& e) {
        return make_error(ErrorCode::IoError, "Cannot open headroom frame source", e.what());
    }
}

const VideoSource& HeadroomFrameSource::video() const { return d_->video(); }
const std::string& HeadroomFrameSource::cache_id() const { return d_->prepared->cache_id(); }

Result<void> HeadroomFrameSource::request(FloatFrameRange range) {
    std::lock_guard lock(d_->mutex);
    if (d_->worker_failed) return *d_->failure;
    if (range.first < 0 || range.first >= d_->video().clock.frames || range.count <= 0 ||
        range.count > d_->video().clock.frames - range.first)
        return make_error(ErrorCode::InvalidArgument, "Headroom request is outside the movie");
    auto checked = d_->verify();
    if (!checked) { d_->fail(checked.error()); if (d_->active) d_->active->interrupt(); return checked.error(); }
    if (d_->active) d_->active->interrupt();
    ++d_->generation; d_->range = range; d_->floor = range.first;
    d_->failure.reset(); d_->pending = true; d_->complete = false;
    d_->changed.notify_all();
    return {};
}

Result<HeadroomFrameSource::Frame> HeadroomFrameSource::take(int index, std::chrono::milliseconds wait) {
    std::unique_lock lock(d_->mutex);
    if (wait.count() < 0) return make_error(ErrorCode::InvalidArgument, "Invalid headroom frame wait");
    if (d_->failure) return *d_->failure;
    if (!d_->range || index < d_->range->first || index >= d_->range->first + d_->range->count)
        return make_error(ErrorCode::InvalidArgument, "Frame is outside the active headroom request");
    auto checked = d_->verify();
    if (!checked) { d_->fail(checked.error()); if (d_->active) d_->active->interrupt(); return checked.error(); }
    if (index < d_->floor && !d_->cache.contains(index))
        return make_error(ErrorCode::NotFound, "Headroom frame was evicted; request it again");
    d_->floor = std::max(d_->floor, index);
    const auto epoch = d_->generation;
    d_->changed.notify_all();
    d_->changed.wait_for(lock, wait, [&] {
        return d_->failure || epoch != d_->generation || d_->cache.contains(index) || d_->complete;
    });
    if (epoch != d_->generation) return cancelled();
    if (d_->failure) return *d_->failure;
    checked = d_->verify();
    if (!checked) { d_->fail(checked.error()); if (d_->active) d_->active->interrupt(); return checked.error(); }
    if (auto it = d_->cache.find(index); it != d_->cache.end()) { ++d_->hits; return it->second; }
    if (d_->complete) return make_error(ErrorCode::NotFound, "Headroom frame is no longer cached; request it again");
    return make_error(ErrorCode::Busy, "Headroom frame is not ready; release old frames or wait");
}

void HeadroomFrameSource::cancel() {
    std::lock_guard lock(d_->mutex);
    ++d_->generation; d_->pending = false;
    d_->fail(cancelled());
    if (d_->active) d_->active->interrupt();
}

HeadroomFrameStats HeadroomFrameSource::stats() const {
    std::lock_guard lock(d_->mutex);
    return {d_->frame_bytes, d_->capacity, d_->live() * d_->frame_bytes,
            d_->cache.size(), d_->decoded, d_->hits, d_->decoder_starts, d_->peak_pixels, d_->complete};
}
}  // namespace rudra
