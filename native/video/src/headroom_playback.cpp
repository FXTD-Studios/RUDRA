#include "rudra/video/headroom_playback.hpp"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace rudra {
namespace fs = std::filesystem;
namespace headroom_detail { struct PlaybackIdentity {}; }
namespace {
Error cancelled() { return make_error(ErrorCode::Cancelled, "Headroom playback generation was cancelled"); }
Error foreign() { return make_error(ErrorCode::InvalidArgument, "Frame or receipt belongs to another playback owner"); }
Error busy() { return make_error(ErrorCode::Busy, "Headroom playback is not ready; release old frames or wait"); }
bool valid_range(FloatFrameRange r, const PreparedHeadroomSource::Handle& source) {
    return r.first >= 0 && r.count > 0 && (!source ||
        (r.first < source->video().clock.frames && r.count <= source->video().clock.frames - r.first));
}
}

struct HeadroomPlayback::Impl {
    struct Selection {
        PreparedHeadroomSource::Handle prepared;
        fs::path source, decoder, probe;
        HeadroomPolicy policy = HeadroomPolicy::Bt1886V1;
    };
    HeadroomPlaybackLimits limits;
    fs::path log;
    std::shared_ptr<HeadroomPixelBudget> budget;
    const std::shared_ptr<const headroom_detail::PlaybackIdentity> identity =
        std::make_shared<const headroom_detail::PlaybackIdentity>();
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::thread worker, interrupter;
    bool closing = false, pending = false, preparing = false, complete = false, worker_failed = false;
    Generation generation = 0, active_generation = 0, interrupted_generation = 0;
    std::uint64_t decoded = 0, starts = 0; // lifetime totals, including superseded reads
    std::optional<Selection> selected;
    FloatFrameRange range;
    std::stop_source stop;
    std::shared_ptr<HeadroomStream> active;
    std::deque<Frame> queue;
    std::optional<HeadroomDecodeReceipt> receipt;
    std::optional<Error> failure;

    // All *_locked helpers require mutex. No decoder or OS calls under mutex.
    Result<void> status_locked(Generation epoch) const {
        if (closing || epoch != generation) return cancelled();
        if (failure) return *failure;
        if (!selected) return make_error(ErrorCode::InvalidArgument, "Open a headroom source first");
        return {};
    }
    void fail_locked(Error error) {
        if (!failure) failure = std::move(error);
        queue.clear(); receipt.reset(); preparing = complete = false;
        changed.notify_all();
    }
    Result<Generation> request_locked(Selection selection, FloatFrameRange requested) {
        if (closing || worker_failed) return failure.value_or(cancelled());
        if (!valid_range(requested, selection.prepared))
            return make_error(ErrorCode::InvalidArgument, "Invalid headroom playback range");
        selected = std::move(selection); range = requested; ++generation;
        pending = true; preparing = !selected->prepared; complete = false;
        failure.reset(); queue.clear(); receipt.reset(); changed.notify_all();
        return generation;
    }
    bool current(Generation epoch) const {
        std::lock_guard lock(mutex);
        return !closing && epoch == generation && !failure;
    }
    void fail(Generation epoch, Error error) {
        std::lock_guard lock(mutex);
        if (!closing && epoch == generation) fail_locked(std::move(error));
    }
    void interrupt_loop() {
        std::unique_lock lock(mutex);
        for (;;) {
            changed.wait(lock, [&] { return closing || (active_generation != 0 &&
                active_generation != generation && interrupted_generation != active_generation); });
            const bool end = closing;
            auto cancelling = stop;
            auto owner = active;
            interrupted_generation = active_generation;
            lock.unlock();
            cancelling.request_stop();
            if (owner) owner->cancel();
            owner.reset(); // destruction/reaping must also stay off the caller
            if (end) return;
            lock.lock();
        }
    }
    bool wait_for_room(Generation epoch, std::size_t frame_bytes) {
        std::unique_lock lock(mutex);
        while (!closing && epoch == generation && !failure) {
            const auto pixels = budget->stats();
            if (queue.size() < limits.queued_frames && frame_bytes <= pixels.limit - pixels.live) return true;
            // Borrowed frame release has no adapter callback. Poll only under
            // backpressure; seek/cancel/destruction wake this immediately.
            changed.wait_for(lock, std::chrono::milliseconds(5));
        }
        return false;
    }
    void decode(Generation epoch, Selection selection, FloatFrameRange requested, std::stop_token token) {
        if (!selection.prepared) {
            auto prepared = PreparedHeadroomSource::prepare(selection.source, selection.policy,
                selection.decoder, {token, selection.probe});
            if (!prepared) { fail(epoch, prepared.error()); return; }
            selection.prepared = std::move(*prepared);
        }
        if (!current(epoch)) return;
        if (!valid_range(requested, selection.prepared)) {
            fail(epoch, make_error(ErrorCode::InvalidArgument, "Headroom playback range is outside the movie")); return;
        }
        auto opened = HeadroomStream::open(selection.prepared, limits.pixel_bytes, log, budget);
        if (!opened) { fail(epoch, opened.error()); return; }
        auto owner = std::shared_ptr<HeadroomStream>(std::move(*opened));
        {
            std::lock_guard lock(mutex);
            if (closing || epoch != generation) return;
            selected->prepared = selection.prepared;
            active = owner; preparing = false; changed.notify_all();
        }
        const auto bytes = owner->stats().frame_bytes;
        if (!wait_for_room(epoch, bytes)) return;
        auto started = owner->start(requested);
        if (!started) { fail(epoch, started.error()); return; }
        { std::lock_guard lock(mutex); ++starts; }
        for (int i = 0; i < requested.count; ++i) {
            if (!wait_for_room(epoch, bytes)) return;
            auto frame = owner->next();
            if (!frame) { fail(epoch, frame.error()); return; }
            auto valid = owner->validate_preview(*frame);
            if (!valid) { fail(epoch, valid.error()); return; }
            std::lock_guard lock(mutex);
            ++decoded;
            if (closing || epoch != generation || failure) return;
            queue.push_back(Frame(std::move(*frame), identity, epoch)); changed.notify_all();
        }
        auto done = owner->finish(); // includes exact EOF and exit; cancellable
        if (!done) { fail(epoch, done.error()); return; }
        auto valid = owner->validate_completion(*done);
        if (!valid) { fail(epoch, valid.error()); return; }
        std::lock_guard lock(mutex);
        if (closing || epoch != generation || failure) return;
        receipt = std::move(*done); complete = true; changed.notify_all();
    }
    void work_loop() {
        for (;;) {
            std::unique_lock lock(mutex);
            changed.wait(lock, [&] { return closing || pending ||
                (active && active_generation != generation); });
            if (closing) {
                auto retired = std::move(active);
                lock.unlock();
                if (retired) retired->close();
                return;
            }
            if (!pending) {
                auto retired = std::move(active);
                lock.unlock();
                retired->close();
                continue;
            }
            const auto epoch = generation;
            const auto selection = *selected;
            const auto requested = range;
            auto retired = std::move(active);
            stop = std::stop_source{};
            const auto token = stop.get_token();
            active_generation = epoch; pending = false;
            lock.unlock();
            if (retired) retired->close();
            retired.reset(); // join/reap old decoder before any new launch
            try { decode(epoch, selection, requested, token); }
            catch (const std::exception& e) {
                std::shared_ptr<HeadroomStream> owner;
                { std::lock_guard guard(mutex); owner = active; }
                if (owner) owner->cancel();
                fail(epoch, make_error(ErrorCode::IoError, "Headroom playback failed", e.what()));
            }
            std::shared_ptr<HeadroomStream> failed_owner;
            { std::lock_guard guard(mutex);
              if (epoch == generation && failure) failed_owner = active; }
            if (failed_owner) failed_owner->close();
        }
    }
    void run() {
        try { work_loop(); }
        catch (const std::exception& e) {
            std::lock_guard lock(mutex);
            worker_failed = true;
            fail_locked(make_error(ErrorCode::IoError, "Headroom playback worker failed", e.what()));
        }
    }
    // Validate outside adapter mutex; generation recheck rejects concurrent seek.
    Result<void> checked(Generation epoch, Result<void> valid) {
        std::lock_guard lock(mutex);
        auto status = status_locked(epoch); if (!status) return status.error();
        if (!valid) fail_locked(valid.error());
        return valid;
    }
};

HeadroomPlayback::HeadroomPlayback() : d_(std::make_unique<Impl>()) {}
Result<std::unique_ptr<HeadroomPlayback>> HeadroomPlayback::create(HeadroomPlaybackLimits limits, const fs::path& log) {
    if (!limits.pixel_bytes || !limits.queued_frames || limits.queued_frames > 32)
        return make_error(ErrorCode::InvalidArgument, "Invalid headroom playback limits");
    try {
        std::unique_ptr<HeadroomPlayback> out(new HeadroomPlayback());
        out->d_->limits = limits; out->d_->log = log;
        out->d_->budget = std::make_shared<HeadroomPixelBudget>(limits.pixel_bytes);
        out->d_->interrupter = std::thread([p = out.get()] { p->d_->interrupt_loop(); });
        out->d_->worker = std::thread([p = out.get()] { p->d_->run(); });
        return out;
    } catch (const std::exception& e) { return make_error(ErrorCode::IoError, "Cannot create headroom playback", e.what()); }
}
HeadroomPlayback::~HeadroomPlayback() {
    {
        std::lock_guard lock(d_->mutex);
        d_->closing = true; ++d_->generation; d_->queue.clear(); d_->changed.notify_all();
    }
    if (d_->interrupter.joinable()) d_->interrupter.join();
    if (d_->worker.joinable()) d_->worker.join();
}
Result<HeadroomPlayback::Generation> HeadroomPlayback::open(const fs::path& source, HeadroomPolicy policy,
    const fs::path& decoder, FloatFrameRange range, const fs::path& probe_executable) {
    if (policy != HeadroomPolicy::Bt1886V1)
        return make_error(ErrorCode::Unsupported, "Playback requires an explicit supported headroom policy");
    if (source.empty() || decoder.empty()) return make_error(ErrorCode::InvalidArgument, "Missing headroom source or decoder");
    std::lock_guard lock(d_->mutex);
    return d_->request_locked({{}, source, decoder, probe_executable, policy}, range);
}
Result<HeadroomPlayback::Generation> HeadroomPlayback::open(PreparedHeadroomSource::Handle prepared, FloatFrameRange range) {
    if (!prepared) return make_error(ErrorCode::InvalidArgument, "Missing prepared headroom source");
    std::lock_guard lock(d_->mutex);
    return d_->request_locked({std::move(prepared), {}, {}, {}, HeadroomPolicy::Bt1886V1}, range);
}
Result<HeadroomPlayback::Generation> HeadroomPlayback::seek(FloatFrameRange range) {
    std::lock_guard lock(d_->mutex);
    if (!d_->selected) return make_error(ErrorCode::InvalidArgument, "Open a headroom source before seeking");
    return d_->request_locked(*d_->selected, range);
}

Result<VideoSource> HeadroomPlayback::video(Generation epoch, std::chrono::milliseconds wait) {
    if (wait.count() < 0) return make_error(ErrorCode::InvalidArgument, "Invalid playback wait");
    std::unique_lock lock(d_->mutex);
    d_->changed.wait_for(lock, wait, [&] { return !d_->status_locked(epoch) ||
        (!d_->preparing && d_->selected->prepared); });
    auto valid = d_->status_locked(epoch); if (!valid) return valid.error();
    if (d_->preparing || !d_->selected->prepared) return busy();
    return d_->selected->prepared->video();
}
Result<HeadroomPlayback::Frame> HeadroomPlayback::take(Generation epoch, std::chrono::milliseconds wait) {
    if (wait.count() < 0) return make_error(ErrorCode::InvalidArgument, "Invalid playback wait");
    std::unique_lock lock(d_->mutex);
    d_->changed.wait_for(lock, wait, [&] { return !d_->status_locked(epoch) || !d_->queue.empty() || d_->complete; });
    auto valid = d_->status_locked(epoch); if (!valid) return valid.error();
    if (d_->queue.empty()) return d_->complete ?
        make_error(ErrorCode::NotFound, "All frames in the playback range were taken") : busy();
    auto frame = std::move(d_->queue.front()); d_->queue.pop_front(); d_->changed.notify_all();
    auto owner = d_->active;
    lock.unlock();
    valid = d_->checked(epoch, owner->validate_preview(frame.preview_));
    if (!valid) return valid.error();
    return frame;
}
Result<HeadroomPlayback::Receipt> HeadroomPlayback::completion(Generation epoch, std::chrono::milliseconds wait) {
    if (wait.count() < 0) return make_error(ErrorCode::InvalidArgument, "Invalid playback wait");
    std::unique_lock lock(d_->mutex);
    d_->changed.wait_for(lock, wait, [&] { return !d_->status_locked(epoch) || d_->receipt.has_value(); });
    auto valid = d_->status_locked(epoch); if (!valid) return valid.error();
    if (!d_->receipt) return busy();
    auto receipt = *d_->receipt;
    auto owner = d_->active;
    lock.unlock();
    valid = d_->checked(epoch, owner->validate_completion(receipt));
    if (!valid) return valid.error();
    return Receipt(std::move(receipt), d_->identity, epoch);
}
Result<void> HeadroomPlayback::validate_preview(const Frame& frame) {
    std::shared_ptr<HeadroomStream> owner;
    {
        std::lock_guard lock(d_->mutex);
        if (frame.identity_ != d_->identity) return foreign();
        auto valid = d_->status_locked(frame.generation_); if (!valid) return valid.error();
        owner = d_->active;
    }
    return d_->checked(frame.generation_, owner->validate_preview(frame.preview_));
}
Result<void> HeadroomPlayback::validate_completion(const Receipt& receipt) {
    std::shared_ptr<HeadroomStream> owner;
    {
        std::lock_guard lock(d_->mutex);
        if (receipt.identity_ != d_->identity) return foreign();
        auto valid = d_->status_locked(receipt.generation_); if (!valid) return valid.error();
        owner = d_->active;
    }
    return d_->checked(receipt.generation_, owner->validate_completion(receipt.receipt_));
}
Result<HeadroomFrameSource::Frame> HeadroomPlayback::verified_frame(const Frame& frame, const Receipt& receipt) {
    std::shared_ptr<HeadroomStream> owner;
    {
        std::lock_guard lock(d_->mutex);
        if (frame.identity_ != d_->identity || receipt.identity_ != d_->identity) return foreign();
        auto valid = d_->status_locked(frame.generation_); if (!valid) return valid.error();
        valid = d_->status_locked(receipt.generation_); if (!valid) return valid.error();
        owner = d_->active;
    }
    auto pixels = owner->verified_frame(frame.preview_, receipt.receipt_);
    auto valid = d_->checked(frame.generation_, pixels ? Result<void>{} : Result<void>(pixels.error()));
    if (!valid) return valid.error();
    return pixels;
}
void HeadroomPlayback::cancel() {
    std::lock_guard lock(d_->mutex);
    ++d_->generation; d_->pending = false; d_->failure.reset(); d_->fail_locked(cancelled());
}
HeadroomPlaybackStats HeadroomPlayback::stats() const {
    std::lock_guard lock(d_->mutex);
    return {d_->generation, d_->decoded, d_->starts, d_->queue.size(), d_->budget->stats(), d_->preparing, d_->complete};
}
} // namespace rudra
