#pragma once
// Playback of a movie (10 Oct 2026). Opening a movie frame by frame seeks
// with ffmpeg and round-trips a PNG for every frame (media/sequence.cpp), far
// too slow to play. While a movie plays, one ffmpeg decodes it in order from
// the playhead on its own thread, a few frames ahead, at the preview's size;
// the frame engine's loader takes them from here, and falls back to the
// per-frame path for anything this decoder is not producing.

#include <condition_variable>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#include "rudra/core/image.hpp"
#include "rudra/platform/result.hpp"

namespace rudra::app {

class MovieFeeder {
public:
    ~MovieFeeder();
    MovieFeeder(const MovieFeeder&) = delete;
    MovieFeeder& operator=(const MovieFeeder&) = delete;

    // Decodes `movie` from frame `start` (of `count`, at `fps`) in order, at
    // most `ahead` frames waiting, each fitted to `max_side` (0: full size).
    static Result<std::unique_ptr<MovieFeeder>> start(const std::filesystem::path& movie, double fps, int start,
                                                      int count, int max_side, int ahead = 8);

    // Frame `index` if this decoder makes it: waits for it (up to `wait_ms`)
    // when it is still to come; nullopt for a frame behind its start, after a
    // failure, or on timeout. A frame taken is gone from the buffer.
    std::optional<SdrImage> take(int index, int wait_ms = 4000);
    int first() const noexcept { return start_; }
    // Ends the decoder and releases anyone waiting in take() (who get nullopt).
    void stop();
    bool covers(int index) const noexcept { return index >= start_ && index < count_; }

private:
    MovieFeeder() = default;
    void run();
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int start_ = 0, count_ = 0, max_side_ = 0, ahead_ = 8;
    std::mutex mu_;
    std::condition_variable cv_;
    std::map<int, SdrImage> ready_;
    int next_ = 0;            // the next frame the decoder will produce
    int floor_ = 0;           // frames below it are not wanted any more (a skip ahead): decoded and dropped
    bool stop_ = false, failed_ = false;
    std::thread worker_;
};

}  // namespace rudra::app
