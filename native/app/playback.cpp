#include "playback.hpp"

#include <chrono>
#include <cstdio>

#include "rudra/media/video_decode.hpp"
#include "rudra/media/video_probe.hpp"
#include "rudra/platform/process.hpp"
#ifdef RUDRA_HAVE_STILL_DECODE
#include "rudra/media/still.hpp"
#endif

namespace rudra::app {

struct MovieFeeder::Impl {
    std::unique_ptr<VideoDecoder> decoder;
};

Result<std::unique_ptr<MovieFeeder>> MovieFeeder::start(const std::filesystem::path& movie, double fps, int start,
                                                        int count, int max_side, int ahead) {
    if (start < 0 || start >= count) return make_error(ErrorCode::InvalidArgument, "playback starts outside the movie");
    auto ffmpeg = find_executable("ffmpeg");
    if (!ffmpeg) return make_error(ErrorCode::NotFound, "ffmpeg is not on PATH");
    auto video = open_video_source(movie, VideoArgs{});
    if (!video) return video.error();
    const int w = video->stream.width, h = video->stream.height;
    char ss[64];
    std::snprintf(ss, sizeof ss, "%.6f", start / std::max(fps, 1e-6));
    // As the per-frame path decodes (media/sequence.cpp): the first video
    // stream, ffmpeg's own conversion to RGB, accurate seek; raw 16-bit RGB.
    const std::vector<std::string> argv = {ffmpeg->string(), "-v", "error", "-nostdin", "-accurate_seek", "-ss", ss,
                                           "-i", movie.string(), "-map", "0:v:0", "-frames:v",
                                           std::to_string(count - start), "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"};
    const auto log = std::filesystem::temp_directory_path() / "rudra-playback-decode.log";
    auto d = VideoDecoder::open_command(argv, w, h, 3, count - start, log);
    if (!d) return d.error();
    std::unique_ptr<MovieFeeder> f(new MovieFeeder());
    f->impl_ = std::make_unique<Impl>();
    f->impl_->decoder = std::move(*d);
    f->start_ = start, f->count_ = count, f->max_side_ = max_side, f->ahead_ = std::max(1, ahead);
    f->next_ = start;
    f->worker_ = std::thread([p = f.get()] { p->run(); });
    return f;
}

void MovieFeeder::stop() {
    {
        std::lock_guard lk(mu_);
        if (stop_) return;
        stop_ = true;
    }
    cv_.notify_all();
    if (impl_ && impl_->decoder) impl_->decoder->interrupt();   // ends a read in flight
}

MovieFeeder::~MovieFeeder() {
    stop();
    if (worker_.joinable()) worker_.join();
}

void MovieFeeder::run() {
    for (;;) {
        {
            std::unique_lock lk(mu_);
            cv_.wait(lk, [&] { return stop_ || int(ready_.size()) < ahead_; });
            if (stop_ || next_ >= count_) return;
        }
        auto raw = impl_->decoder->next();
        std::lock_guard lk(mu_);
        if (stop_) return;
        if (!raw) {
            failed_ = true;
            cv_.notify_all();
            return;
        }
        if (next_ < floor_) {   // skipped past while playing: decoded, not kept
            ++next_;
            continue;
        }
        SdrImage img = rgb_from_frame(*raw);
#ifdef RUDRA_HAVE_STILL_DECODE
        if (max_side_ > 0) img = fit_max_side(img, max_side_);   // the preview's size, as the paused path
#endif
        ready_.emplace(next_++, std::move(img));
        cv_.notify_all();
    }
}

std::optional<SdrImage> MovieFeeder::take(int index, int wait_ms) {
    std::unique_lock lk(mu_);
    // Behind what this decoder still holds or will make: not from here.
    if (!covers(index) || (index < next_ && !ready_.count(index))) return std::nullopt;
    // Everything before it is no longer wanted: the room goes to what is ahead
    // (a skip to keep time, or a scrub forward while playing).
    if (index > floor_) floor_ = index;
    ready_.erase(ready_.begin(), ready_.lower_bound(index));
    cv_.notify_all();
    const bool got = cv_.wait_for(lk, std::chrono::milliseconds(wait_ms), [&] {
        return stop_ || failed_ || ready_.count(index) || next_ > index;
    });
    auto it = ready_.find(index);
    if (!got || it == ready_.end()) return std::nullopt;
    SdrImage out = std::move(it->second);
    // What is behind the playhead is not wanted again: room for what is ahead.
    ready_.erase(ready_.begin(), std::next(it));
    cv_.notify_all();
    return out;
}

}  // namespace rudra::app
