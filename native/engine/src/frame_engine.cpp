#include "rudra/engine/frame_engine.hpp"

#include <algorithm>

namespace rudra {
namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

FrameEngine::FrameEngine(FrameLoader loader, FrameInfer infer, EngineOptions options)
    : loader_(std::move(loader)), infer_(std::move(infer)), opt_(options),
      worker_([this] { run(); }) {}

FrameEngine::~FrameEngine() {
    {
        std::lock_guard lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    worker_.join();
}

void FrameEngine::set_sequence(int count) {
    std::lock_guard lk(mu_);
    generation_.bump();
    stats_.cancelled += queue_.size();
    queue_.clear();
    cache_.clear();
    lru_.clear();
    count_ = std::max(0, count);
    wanted_ = -1;
}

int FrameEngine::count() const {
    std::lock_guard lk(mu_);
    return count_;
}

void FrameEngine::on_ready(std::function<void(const ReadyFrame&)> cb) {
    std::lock_guard lk(mu_);
    ready_cb_ = std::move(cb);
}

bool FrameEngine::is_cached(int index) const {
    std::lock_guard lk(mu_);
    return cache_.count(index) != 0;
}

EngineStats FrameEngine::stats() const {
    std::lock_guard lk(mu_);
    EngineStats s = stats_;
    s.cached = int(cache_.size());
    return s;
}

std::shared_ptr<ReadyFrame> FrameEngine::lookup(int index) {
    auto it = cache_.find(index);
    if (it == cache_.end()) return nullptr;
    lru_.erase(it->second.second);
    lru_.push_front(index);
    it->second.second = lru_.begin();
    return it->second.first;
}

void FrameEngine::set_playing(bool on) {
    std::lock_guard lk(mu_);
    playing_ = on;
}

bool FrameEngine::playing() const {
    std::lock_guard lk(mu_);
    return playing_;
}

bool FrameEngine::ahead_of_playhead(int index) const {
    if (!playing_ || wanted_ < 0 || count_ <= 0) return false;
    const int d = ((index - wanted_) % count_ + count_) % count_;
    return d >= 1 && d <= std::max(opt_.play_ahead, opt_.read_ahead);
}

bool FrameEngine::wanted(const Job& job) const {
    return generation_.is_current(job.generation) || ahead_of_playhead(job.index);
}

void FrameEngine::insert(std::shared_ptr<ReadyFrame> f) {
    const int index = f->index;
    if (auto it = cache_.find(index); it != cache_.end()) {
        // A full-size frame replaces its proxy; nothing replaces a full one.
        const bool old_proxy = it->second.first->fields && it->second.first->fields->proxy;
        const bool new_full = f->fields && !f->fields->proxy;
        if (old_proxy && new_full) it->second.first = std::move(f);
        return;
    }
    lru_.push_front(index);
    cache_.emplace(index, std::make_pair(std::move(f), lru_.begin()));
    // Evict the least recent, never the frame on screen.
    while (int(cache_.size()) > std::max(1, opt_.cache_frames)) {
        auto victim = std::prev(lru_.end());
        if (*victim == wanted_ && lru_.size() > 1) victim = std::prev(victim);
        cache_.erase(*victim);
        lru_.erase(victim);
    }
}

Generation::value_type FrameEngine::show(int index) {
    std::lock_guard dl(deliver_mu_);   // a delivery under way finishes before the request moves on
    std::shared_ptr<ReadyFrame> hit;
    Generation::value_type g;
    std::function<void(const ReadyFrame&)> cb;
    {
        std::lock_guard lk(mu_);
        if (count_ <= 0) return generation_.current();
        index = std::clamp(index, 0, count_ - 1);
        g = generation_.bump();
        wanted_ = index;
        if (playing_) {
            // Playback: what is queued ahead of the new playhead stays queued
            // (it is the next ticks' frames); the rest is cancelled.
            std::deque<Job> keep;
            for (Job j : queue_) {
                if (ahead_of_playhead(j.index) && !j.refine) {
                    j.generation = g;
                    j.priority = Priority::ReadAhead;
                    keep.push_back(j);
                } else {
                    ++stats_.cancelled;
                }
            }
            queue_ = std::move(keep);
        } else {
            // Cancellation: nothing queued for an older generation will run.
            stats_.cancelled += queue_.size();
            queue_.clear();
        }
        asked_at_ = std::chrono::steady_clock::now();
        ++stats_.shown;
        hit = lookup(index);
        if (hit) {
            ++stats_.cache_hits;
            stats_.last_request_to_ready_ms = 0.0;
            cb = ready_cb_;
            // Stopped on a playback proxy: shown now, and inferred again at full size.
            if (!playing_ && hit->fields && hit->fields->proxy)
                queue_.push_front({index, Priority::Interactive, g, true});
        } else {
            queue_.push_back({index, Priority::Interactive, g});
        }
        // Read ahead, as the Studio's prefetch: the frames after this one,
        // wrapping; further while playing. Not twice.
        const int ahead = playing_ ? std::max(opt_.play_ahead, opt_.read_ahead) : opt_.read_ahead;
        for (int k = 1; k <= ahead && k < count_; ++k) {
            const int i = (index + k) % count_;
            if (cache_.count(i)) continue;
            if (std::any_of(queue_.begin(), queue_.end(), [&](const Job& j) { return j.index == i; })) continue;
            queue_.push_back({i, Priority::ReadAhead, g});
        }
    }
    cv_.notify_all();
    if (hit && cb) {
        ReadyFrame f = *hit;
        f.generation = g;
        f.from_cache = true;
        cb(f);
    }
    return g;
}

void FrameEngine::deliver(const ReadyFrame& f) {
    std::lock_guard dl(deliver_mu_);   // checked and handed over with no show() between them
    std::function<void(const ReadyFrame&)> cb;
    {
        std::lock_guard lk(mu_);
        if (!generation_.is_current(f.generation) || f.index != wanted_) {
            ++stats_.stale_dropped;
            return;
        }
        stats_.last_request_to_ready_ms = ms_since(asked_at_);
        cb = ready_cb_;
    }
    if (cb) cb(f);
}

void FrameEngine::run() {
    for (;;) {
        Job job;
        {
            std::unique_lock lk(mu_);
            cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            // Interactive first; otherwise in the order read-ahead was queued.
            auto it = std::find_if(queue_.begin(), queue_.end(), [](const Job& j) { return j.priority == Priority::Interactive; });
            if (it == queue_.end()) it = queue_.begin();
            job = *it;
            queue_.erase(it);
            if (!wanted(job)) {
                ++stats_.cancelled;
                continue;
            }
            // A playback proxy the stopped playhead asks for is inferred again
            // at full size (also when it landed after the request was queued).
            auto cached = lookup(job.index);
            const bool refine = cached && cached->fields && cached->fields->proxy && !playing_ &&
                                (job.refine || job.priority == Priority::Interactive);
            if (cached && !refine) {
                if (job.priority == Priority::Interactive) {
                    ReadyFrame f = *cached;
                    f.generation = job.generation;
                    f.from_cache = true;
                    lk.unlock();
                    deliver(f);
                }
                continue;
            }
        }

        auto f = std::make_shared<ReadyFrame>();
        f->index = job.index;
        f->generation = job.generation;
        const auto t0 = std::chrono::steady_clock::now();
        auto sdr = loader_(job.index);
        f->decode_ms = ms_since(t0);
        if (!sdr) {
            f->error = sdr.error();
        } else {
            // A seek while decoding makes a read-ahead frame not worth inferring now.
            bool still = true;
            {
                std::lock_guard lk(mu_);
                still = job.priority == Priority::Interactive || wanted(job);
                if (!still) ++stats_.cancelled;
            }
            if (!still) continue;
            const auto t1 = std::chrono::steady_clock::now();
            auto fr = infer_(*sdr);
            f->infer_ms = ms_since(t1);
            if (!fr) f->error = fr.error();
            else {
                f->sdr = std::make_shared<const SdrImage>(std::move(*sdr));
                f->fields = std::make_shared<const FrameResult>(std::move(*fr));
            }
        }
        bool asked = job.priority == Priority::Interactive;
        {
            std::lock_guard lk(mu_);
            if (f->error) ++stats_.errors;
            else {
                ++stats_.inferred;
                insert(f);
            }
            asked = asked || f->index == wanted_;
        }
        // The frame asked for is delivered even when it failed, so the UI can say why.
        if (asked) deliver(*f);
    }
}

}  // namespace rudra
