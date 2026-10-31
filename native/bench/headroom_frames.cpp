// Explicit diagnostic target; not installed or included in the test suite.
#include <algorithm>
#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <sstream>
#include <thread>

#include <nlohmann/json.hpp>
#include "rudra/media/video_decode.hpp"
#include "rudra/platform/hash.hpp"
#include "rudra/video/headroom_frames.hpp"
#include "rudra/video/headroom_stream.hpp"
#include "rudra/video/headroom_playback.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

using namespace rudra;
using Clock = std::chrono::steady_clock;
namespace {
double ms(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
template<class T> T checked(Result<T> r) {
    if (!r) throw std::runtime_error(r.error().message + ": " + r.error().detail);
    return std::move(r).value();
}
void checked(Result<void> r) { if (!r) throw std::runtime_error(r.error().message + ": " + r.error().detail); }

// Three independent seek owners, with identical decoding and budgets. Only
// preparation reuse differs. Compare hashes with a full sequential reference.
nlohmann::json seek_benchmark(const std::filesystem::path& file, bool reuse) {
    nlohmann::json report{{"schema", "headroom-preparation-benchmark-v1"}, {"source", file.string()},
                          {"reuse", reuse}, {"ranges", nlohmann::json::array()}};
    const auto total = Clock::now();
    auto ffmpeg = checked(require_executable("ffmpeg"));
    PreparedHeadroomSource::Handle shared;
    int frames = 0;
    for (int seek = 0; seek < 3; ++seek) {
        const auto opening = Clock::now();
        if (!shared || !reuse) shared = checked(PreparedHeadroomSource::prepare(file, HeadroomPolicy::Bt1886V1, ffmpeg));
        const auto prepared_ms = ms(opening);
        const auto binding = Clock::now();
        auto owner = checked(HeadroomStream::open(shared));
        const auto bind_ms = ms(binding);
        frames = owner->video().clock.frames;
        const int first = seek == 0 ? 0 : (seek == 1 ? frames - 1 : frames / 2);
        const auto reading = Clock::now();
        checked(owner->start({first, 1}));
        auto frame = checked(owner->next());
        checked(owner->validate_preview(frame));
        const auto first_ms = ms(reading);
        const auto hash = xxh64(std::as_bytes(frame.pixels().encoded_rgb.span()));
        auto receipt = checked(owner->finish()); checked(owner->validate_completion(receipt));
        checked(owner->verified_frame(frame, receipt));
        const auto stats = owner->stats();
        report["ranges"].push_back({{"first", first}, {"prepare_ms", prepared_ms}, {"bind_ms", bind_ms},
            {"first_frame_ms", first_ms}, {"decode_finish_ms", ms(reading)}, {"xxh64", hash},
            {"cache_id", owner->cache_id()}, {"decoder_starts", stats.decoder_starts},
            {"peak_pixel_bytes", stats.peak_pixel_bytes}, {"complete", stats.complete}});
    }
    report["total_ms"] = ms(total); report["ok"] = true;
    report["frames"] = frames; report["width"] = shared->video().stream.width;
    report["height"] = shared->video().stream.height;
    return report;
}
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::cerr << "headroom-frames-bench movie feeder|stream|sequential|legacy|reuse|reopen|cancel output.json [decoder_threads|decoder,filter,output[,queue]|delay=ms]\n";
        return 2;
    }
    nlohmann::json report{{"schema", "headroom-frame-benchmark-v1"}, {"mode", argv[2]}, {"source", argv[1]}};
    try {
        const std::filesystem::path file(argv[1]);
        const std::string mode(argv[2]);
        if (mode == "cancel") {
            if (argc != 4) throw std::runtime_error("Cancellation benchmark takes no additional options");
            auto ffmpeg = checked(require_executable("ffmpeg"));
            rudra::stop_source stop;
            auto work = std::async(std::launch::async, [&] {
                return PreparedHeadroomSource::prepare(file, HeadroomPolicy::Bt1886V1, ffmpeg, {stop.get_token()});
            });
            if (work.wait_for(std::chrono::milliseconds(100)) != std::future_status::timeout)
                throw std::runtime_error("Fixture completed preparation before the cancellation measurement");
            const auto cancelling = Clock::now(); stop.request_stop();
            auto result = work.get();
            report["cancel_ms"] = ms(cancelling); report["cancel_after_ms"] = 100;
            if (result || result.error().code != ErrorCode::Cancelled)
                throw std::runtime_error("Cancelled preparation unexpectedly published or returned another error");
            report["ok"] = true;
        } else if (mode == "reuse" || mode == "reopen") {
            if (argc != 4) throw std::runtime_error("Seek benchmark takes no additional options");
            report = seek_benchmark(file, mode == "reuse"); report["mode"] = mode;
        } else {
        if (mode != "feeder" && mode != "stream" && mode != "async" && mode != "sequential" && mode != "legacy") throw std::runtime_error("Unknown mode");
        int delay_ms = 0;
        if (argc == 5 && (mode == "stream" || mode == "async") && std::string(argv[4]).starts_with("delay=")) {
            const auto value = std::string(argv[4]).substr(6); std::size_t used = 0;
            delay_ms = std::stoi(value, &used);
            if (used != value.size() || delay_ms < 0 || delay_ms > 250)
                throw std::runtime_error("Stream consumer delay must be 0..250 ms");
        } else if (argc == 5 && mode != "sequential")
            throw std::runtime_error("Use sequential thread options or stream delay=milliseconds");
        report["consumer_delay_ms"] = delay_ms;
        const auto prepare = Clock::now();
        auto ffmpeg = checked(require_executable("ffmpeg"));
        std::unique_ptr<HeadroomFrameSource> feeder;
        std::unique_ptr<HeadroomStream> stream;
        std::unique_ptr<HeadroomPlayback> playback;
        PreparedHeadroomSource::Handle prepared;
        HeadroomPlayback::Generation generation = 0;
        VideoSource video;
        if (mode == "feeder") {
            feeder = checked(HeadroomFrameSource::open(file, HeadroomPolicy::Bt1886V1, ffmpeg));
            video = feeder->video(); report["cache_id"] = feeder->cache_id();
        } else if (mode == "stream") {
            stream = checked(HeadroomStream::open(file, HeadroomPolicy::Bt1886V1, ffmpeg));
            video = stream->video(); report["cache_id"] = stream->cache_id();
        } else if (mode == "async") {
            prepared = checked(PreparedHeadroomSource::prepare(file, HeadroomPolicy::Bt1886V1, ffmpeg));
            video = prepared->video(); report["cache_id"] = prepared->cache_id();
            playback = checked(HeadroomPlayback::create());
        } else video = checked(open_video_source(file, VideoArgs{}));
        report["prepare_ms"] = ms(prepare);
        report["width"] = video.stream.width; report["height"] = video.stream.height;
        report["frames"] = video.clock.frames; report["fps"] = video.clock.fps;
        const auto start = Clock::now();
        std::unique_ptr<FloatVideoDecoder> sequential;
        std::unique_ptr<VideoDecoder> legacy;
        if (feeder) checked(feeder->request({0, video.clock.frames}));
        else if (stream) checked(stream->start({0, video.clock.frames}));
        else if (playback) {
            const auto dispatch = Clock::now();
            generation = checked(playback->open(prepared, {0, video.clock.frames}));
            report["submit_ms"] = ms(dispatch);
        }
        else if (mode == "sequential") {
            auto command = checked(float_decoder_command(ffmpeg.string(), file, video));
            if (argc == 5) {
                std::vector<int> counts;
                std::istringstream options(argv[4]); std::string value;
                while (std::getline(options, value, ',')) {
                    std::size_t used = 0; const int count = std::stoi(value, &used);
                    if (used != value.size() || count < 0 || count > 32)
                        throw std::runtime_error("Invalid thread count");
                    counts.push_back(count);
                }
                if ((counts.size() != 1 && counts.size() != 3 && counts.size() != 4) || counts.front() == 0 ||
                    (counts.size() == 4 && counts[3] == 0))
                    throw std::runtime_error("Use decoder threads or decoder,filter,output[,queue] (0 = auto for filter/output)");
                auto at = std::find(command.begin(), command.end(), "-i");
                command.insert(at, {"-threads", std::to_string(counts[0])});
                report["decoder_threads"] = counts[0];
                if (counts.size() >= 3) {
                    command.insert(command.end() - 1, {"-filter_threads", std::to_string(counts[1]),
                        "-threads:v", std::to_string(counts[2])});
                    report["filter_threads"] = counts[1]; report["output_threads"] = counts[2];
                }
                if (counts.size() == 4) {
                    command.insert(command.end() - 1, {"-thread_queue_size", std::to_string(counts[3])});
                    report["output_queue_packets"] = counts[3];
                }
            }
            report["decode_argv"] = command;
            sequential = checked(FloatVideoDecoder::open_command(command, video.stream.width, video.stream.height,
                video.clock.frames, video.contract, {}));
        } else legacy = checked(VideoDecoder::open(file, video, {}));
        double hash_ms = 0;
        std::size_t observed_pixels = 0;
        auto hashes = nlohmann::json::array();
        for (int i = 0; i < video.clock.frames; ++i) {
            HeadroomFrameSource::Frame held;
            std::optional<HeadroomPreviewFrame> provisional;
            std::optional<HeadroomPlaybackFrame> background;
            ExtendedVideoFrame frame;
            SdrImage sdr;
            std::span<const float> pixels;
            if (feeder) {
                held = checked(feeder->take(i, std::chrono::seconds(45)));
                pixels = held->encoded_rgb.span();
                observed_pixels = std::max(observed_pixels, feeder->stats().live_pixel_bytes);
            } else if (stream) {
                provisional.emplace(checked(stream->next()));
                checked(stream->validate_preview(*provisional));
                pixels = provisional->pixels().encoded_rgb.span();
                observed_pixels = std::max(observed_pixels, stream->stats().live_pixel_bytes);
            } else if (playback) {
                background.emplace(checked(playback->take(generation, std::chrono::seconds(45))));
                checked(playback->validate_preview(*background));
                if (background->index() != i) throw std::runtime_error("Asynchronous frame index mismatch");
                pixels = background->pixels().encoded_rgb.span();
                observed_pixels = std::max(observed_pixels, playback->stats().pixels.live);
            } else if (sequential) { frame = checked(sequential->next()); pixels = frame.encoded_rgb.span(); }
            else { sdr = rgb_from_frame(checked(legacy->next())); pixels = sdr.buffer().span(); }
            if (!i) report["first_frame_ms"] = ms(start);
            const auto hashing = Clock::now();
            hashes.push_back(xxh64(std::as_bytes(pixels)));
            hash_ms += ms(hashing);
            if (delay_ms) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
        if (sequential) checked(sequential->finish());
        if (legacy) checked(legacy->finish());
        if (stream) {
            auto receipt = checked(stream->finish());
            checked(stream->validate_completion(receipt));
            report["decode_receipt_validated"] = true;
        }
        if (playback) {
            auto receipt = checked(playback->completion(generation, std::chrono::seconds(45)));
            checked(playback->validate_completion(receipt));
            const auto stats = playback->stats();
            report["decode_receipt_validated"] = true;
            report["frame_bytes"] = std::size_t(video.stream.width) * std::size_t(video.stream.height) * 3 * sizeof(float);
            report["observed_pixel_bytes"] = observed_pixels;
            report["decoded_frames"] = stats.decoded_frames;
            report["decoder_starts"] = stats.decoder_starts;
            report["peak_pixel_bytes"] = stats.pixels.peak;
            report["shared_pixel_limit"] = stats.pixels.limit;
        }
        if (feeder || stream) {
            const auto deadline = Clock::now() + std::chrono::seconds(3);
            while (feeder && !feeder->stats().complete && Clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            const auto stats = feeder ? feeder->stats() : stream->stats();
            if (!stats.complete) throw std::runtime_error("Feeder did not complete");
            report["frame_bytes"] = stats.frame_bytes;
            report["capacity_frames"] = stats.capacity_frames;
            report["observed_pixel_bytes"] = observed_pixels;
            report["decoded_frames"] = stats.decoded_frames;
            report["decoder_starts"] = stats.decoder_starts;
            report["peak_pixel_bytes"] = stats.peak_pixel_bytes;
        }
        report["feed_ms"] = ms(start);
        report["consumer_hash_ms"] = hash_ms;
        report["feed_fps_including_hash"] = 1000 * video.clock.frames / report["feed_ms"].get<double>();
        report["frame_xxh64"] = hashes;
        report["ok"] = true;
        }
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS_EX memory{}; memory.cb = sizeof memory;
        if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory), sizeof memory)) {
            report["parent_peak_working_set_bytes"] = memory.PeakWorkingSetSize;
            report["parent_peak_commit_bytes"] = memory.PeakPagefileUsage;
        }
#endif
    } catch (const std::exception& e) { report["ok"] = false; report["error"] = e.what(); }
    std::ofstream out(argv[3]);
    if (!out) { std::cerr << "Cannot write benchmark report\n"; return 2; }
    out << report.dump(2) << '\n';
    std::cout << report.value("mode", "") << ": " << (report["ok"].get<bool>() ? "pass" : report.value("error", "failed")) << '\n';
    return report["ok"].get<bool>() ? 0 : 1;
}
