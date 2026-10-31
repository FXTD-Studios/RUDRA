#include <gtest/gtest.h>

#include <chrono>
#include <climits>
#include <filesystem>
#include <fstream>
#include <future>
#include <thread>
#include <type_traits>
#include <nlohmann/json.hpp>
#include "rudra/platform/hash.hpp"

#include "rudra/video/headroom_frames.hpp"
#include "rudra/video/headroom_render.hpp"
#include "rudra/video/headroom_stream.hpp"
#include "rudra/video/headroom_playback.hpp"
#include "rudra/engine/headroom_presentation.hpp"

using namespace rudra;
using namespace std::chrono_literals;
namespace {
namespace fs = std::filesystem;
constexpr std::size_t kFrameBytes = 16 * 4 * 3 * sizeof(float);
constexpr auto kPolicy = HeadroomPolicy::Bt1886V1;

class HeadroomFrames : public ::testing::Test {
protected:
    fs::path dir, movie, ffmpeg;
    void SetUp() override {
#ifndef _WIN32
        GTEST_SKIP() << "Enforced source leases currently require Windows";
#endif
        auto exe = find_executable("ffmpeg");
        if (!exe || !find_executable("ffprobe")) GTEST_SKIP() << "FFmpeg and ffprobe required";
        ffmpeg = *exe;
        dir = fs::temp_directory_path() / ("rudra-headroom-frames-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(dir);
        movie = dir / "source.mkv";
        std::vector<std::uint16_t> codes(7 * 16 * 4 * 3, 512);
        for (int f = 0; f < 7; ++f)
            for (int y = 0; y < 4; ++y)
                for (int x = 0; x < 16; ++x)
                    codes[std::size_t(f * 16 * 4 * 3 + y * 16 + x)] = std::uint16_t(930 + x + f * 10);
        const auto raw = dir / "source.yuv";
        { std::ofstream out(raw, std::ios::binary);
          out.write(reinterpret_cast<const char*>(codes.data()), std::streamsize(codes.size() * 2)); }
        auto encoded = run_process({ffmpeg.string(), "-v", "error", "-nostdin", "-f", "rawvideo",
            "-pixel_format", "yuv444p10le", "-video_size", "16x4", "-framerate", "24000/1001",
            "-color_range", "tv", "-color_trc", "bt709", "-color_primaries", "bt709", "-colorspace", "bt709",
            "-i", raw.string(), "-frames:v", "7", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv444p10le",
            "-color_range", "tv", "-color_trc", "bt709", "-color_primaries", "bt709", "-colorspace", "bt709",
            "-output_ts_offset", "5", movie.string()});
        ASSERT_TRUE(encoded); ASSERT_EQ(encoded->exit_code, 0) << encoded->err;
    }
    Result<std::unique_ptr<HeadroomFrameSource>> open(std::size_t slots = 2, bool real = false,
                                                    const fs::path& path = {}) {
        return HeadroomFrameSource::open(path.empty() ? movie : path, kPolicy,
            real ? ffmpeg : fs::path(RUDRA_HEADROOM_PRODUCER_PATH), {slots * kFrameBytes, 2}, dir / "decode.log");
    }
    bool complete(HeadroomFrameSource& source) {
        const auto until = std::chrono::steady_clock::now() + 3s;
        while (!source.stats().complete && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(2ms);
        return source.stats().complete;
    }
    Result<std::unique_ptr<HeadroomStream>> stream(std::size_t slots = 2, bool real = false,
                                                 const fs::path& path = {}) {
        return HeadroomStream::open(path.empty() ? movie : path, kPolicy,
            real ? ffmpeg : fs::path(RUDRA_HEADROOM_PRODUCER_PATH), slots * kFrameBytes, dir / "stream.log");
    }
    fs::path probe_fixture(const std::string& name) {
        const auto path = dir / (name + ".mkv"); fs::copy_file(movie, path);
        auto probe = probe_video(movie);
        if (!probe) throw std::runtime_error(probe.error().message);
        std::ofstream(path.string() + ".streams.json") << probe->streams_json;
        std::ofstream(path.string() + ".frames.json") << probe->frames_json;
        return path;
    }
    std::string probe_calls(const fs::path& file) {
        std::ifstream in(file.string() + ".probe-calls");
        return {(std::istreambuf_iterator<char>(in)), {}};
    }
    Result<std::unique_ptr<HeadroomPlayback>> playback(std::size_t slots = 2, std::size_t queued = 1) {
        return HeadroomPlayback::create({slots * kFrameBytes, queued}, dir / "playback.log");
    }
};
}

TEST_F(HeadroomFrames, SessionPresentationKeepsHeadroomAndRejectsOldGradeSeekAndOwner) {
    Session session(Session::SourceSupport::HeadroomV1);
    ASSERT_TRUE(session.set_headroom_policy(kPolicy));
    session.grade.clip_fill = session.anchor = session.carry_chroma = false;
    auto bridge = HeadroomPresentation::create({4 * kFrameBytes, 1}); ASSERT_TRUE(bridge);
    auto other = HeadroomPresentation::create({4 * kFrameBytes, 1}); ASSERT_TRUE(other);
    ASSERT_TRUE((*bridge)->set_session(session, false, false, false));
    ASSERT_TRUE((*other)->set_session(session, false, false, false));
    ASSERT_TRUE((*bridge)->open(movie, ffmpeg, {0, 1}));
    auto original = (*bridge)->render_next(5s); ASSERT_TRUE(original) << original.error().message;
    EXPECT_GT(original->master().linear2020.buffer().at(0, 0, 15), 1);
    EXPECT_TRUE(std::ranges::equal(original->master().linear2020.buffer().span(),
                                    original->baseline().linear2020.buffer().span()));
    int shown = -1;
    auto display = [&](const auto& f) { shown = f.index(); };
    ASSERT_TRUE((*bridge)->present(*original, display)); EXPECT_EQ(shown, 0);
    EXPECT_FALSE((*other)->present(*original, display));
    session.nudge_exposure(1);
    ASSERT_TRUE((*bridge)->set_session(session, false, false, false));
    EXPECT_FALSE((*bridge)->present(*original, display));
    ASSERT_TRUE((*bridge)->seek({0, 1}));
    auto graded = (*bridge)->render_next(5s); ASSERT_TRUE(graded) << graded.error().message;
    for (std::size_t i = 0; i < graded->master().linear2020.buffer().span().size(); ++i)
        EXPECT_FLOAT_EQ(graded->master().linear2020.buffer().span()[i],
                        2 * original->master().linear2020.buffer().span()[i]);
    session.undo();
    ASSERT_TRUE((*bridge)->set_session(session, false, false, false));
    EXPECT_FALSE((*bridge)->present(*graded, display));
    ASSERT_TRUE((*bridge)->seek({3, 1}));
    auto paused = (*bridge)->render_next(5s); ASSERT_TRUE(paused) << paused.error().message;
    ASSERT_TRUE((*bridge)->present(*paused, display)); EXPECT_EQ(shown, 3);
    (*bridge)->cancel();
    EXPECT_FALSE((*bridge)->present(*paused, display)); EXPECT_EQ(shown, 3);
    EXPECT_LE((*bridge)->stats().pixels.live, 4 * kFrameBytes);
}

TEST_F(HeadroomFrames, PresentationRejectsOutOfOrderFramesAndInvalidControls) {
    Session session(Session::SourceSupport::HeadroomV1);
    ASSERT_TRUE(session.set_headroom_policy(kPolicy));
    session.grade.clip_fill = session.anchor = session.carry_chroma = false;
    auto bridge = HeadroomPresentation::create({4 * kFrameBytes, 1}); ASSERT_TRUE(bridge);
    ASSERT_TRUE((*bridge)->set_session(session, false, false, false));
    ASSERT_TRUE((*bridge)->open(movie, ffmpeg, {0, 2}));
    auto first = (*bridge)->render_next(5s); ASSERT_TRUE(first);
    auto second = (*bridge)->render_next(5s); ASSERT_TRUE(second);
    int shown = -1;
    auto display = [&](const auto& f) { shown = f.index(); };
    ASSERT_TRUE((*bridge)->present(*second, display));
    EXPECT_FALSE((*bridge)->present(*first, display)); EXPECT_EQ(shown, 1);
    session.toggle_clip_fill();
    EXPECT_FALSE((*bridge)->set_session(session, false, false, false));
    EXPECT_FALSE((*bridge)->present(*second, display)); EXPECT_EQ(shown, 1);
    EXPECT_FALSE((*bridge)->render_next());
}

TEST_F(HeadroomFrames, FinalPresentationAndSeekAreSerialized) {
    Session session(Session::SourceSupport::HeadroomV1);
    ASSERT_TRUE(session.set_headroom_policy(kPolicy));
    session.grade.clip_fill = session.anchor = session.carry_chroma = false;
    auto bridge = HeadroomPresentation::create({4 * kFrameBytes, 1}); ASSERT_TRUE(bridge);
    ASSERT_TRUE((*bridge)->set_session(session, false, false, false));
    ASSERT_TRUE((*bridge)->open(movie, ffmpeg, {0, 1}));
    auto frame = (*bridge)->render_next(5s); ASSERT_TRUE(frame);
    std::promise<void> entered, release;
    auto released = release.get_future();
    auto presenting = std::async(std::launch::async, [&] {
        return (*bridge)->present(*frame, [&](const auto&) {
            entered.set_value(); released.wait_for(3s);
        });
    });
    ASSERT_EQ(entered.get_future().wait_for(3s), std::future_status::ready);
    auto seeking = std::async(std::launch::async, [&] { return (*bridge)->seek({4, 1}); });
    EXPECT_EQ(seeking.wait_for(25ms), std::future_status::timeout);
    release.set_value();
    EXPECT_TRUE(presenting.get()); EXPECT_TRUE(seeking.get());
    EXPECT_FALSE((*bridge)->present(*frame, [](const auto&) {}));
}

TEST_F(HeadroomFrames, PresentationLateFailureRevokesAlreadyRenderedPreview) {
    Session session(Session::SourceSupport::HeadroomV1);
    ASSERT_TRUE(session.set_headroom_policy(kPolicy));
    session.grade.clip_fill = session.anchor = session.carry_chroma = false;
    auto bridge = HeadroomPresentation::create({4 * kFrameBytes, 1}); ASSERT_TRUE(bridge);
    ASSERT_TRUE((*bridge)->set_session(session, false, false, false));
    const auto bad = dir / "delayed-late-failure.mkv"; fs::copy_file(movie, bad);
    ASSERT_TRUE((*bridge)->open(bad, RUDRA_HEADROOM_PRODUCER_PATH, {0, 2}));
    auto frame = (*bridge)->render_next(5s); ASSERT_TRUE(frame) << frame.error().message;
    bool failed = false;
    const auto until = std::chrono::steady_clock::now() + 2s;
    while (!failed && std::chrono::steady_clock::now() < until) {
        auto next = (*bridge)->render_next(10ms);
        failed = !next && next.error().code == ErrorCode::IoError;
    }
    ASSERT_TRUE(failed);
    bool shown = false;
    EXPECT_FALSE((*bridge)->present(*frame, [&](const auto&) { shown = true; }));
    EXPECT_FALSE(shown);
}

TEST_F(HeadroomFrames, RealPlaybackAndPausedCacheMatchFullDecodeExactly) {
    const auto before_open = std::chrono::steady_clock::now();
    auto source = open(2, true);
    ASSERT_TRUE(source) << source.error().message;
    RecordProperty("source_open_ms", std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - before_open).count());
    EXPECT_EQ((*source)->video().clock.frames, 7);
    EXPECT_TRUE((*source)->cache_id().starts_with("headroom-float-v1-"));
    auto reference = FloatVideoDecoder::open(movie, (*source)->video(), dir / "reference.log");
    ASSERT_TRUE(reference);
    ASSERT_TRUE((*source)->request({0, 7}));
    MasterRequest grade;
    grade.analytic = true; grade.source_curve = "bt1886"; grade.source_space = "rec709";
    grade.clip_fill = grade.anchor = grade.carry_chroma = grade.settle_grain = false;
    grade.exposure_ev = 1;
    const auto before_playback = std::chrono::steady_clock::now();
    for (int i = 0; i < 7; ++i) {
        auto expected = (*reference)->next(); ASSERT_TRUE(expected);
        auto got = (*source)->take(i); ASSERT_TRUE(got) << got.error().message;
        EXPECT_EQ((*got)->source_contract, expected->source_contract);
        const auto samples = (*got)->encoded_rgb.span();
        EXPECT_TRUE(std::equal(samples.begin(), samples.end(), expected->encoded_rgb.span().begin()));
        auto actual_master = render_headroom_frame(**got, grade, kPolicy);
        auto reference_master = render_headroom_frame(*expected, grade, kPolicy);
        ASSERT_TRUE(actual_master); ASSERT_TRUE(reference_master);
        EXPECT_EQ(actual_master->linear2020.buffer().vector(), reference_master->linear2020.buffer().vector());
        EXPECT_GT(samples.back(), 1);
        EXPECT_LE((*source)->stats().live_pixel_bytes, 2 * kFrameBytes);
    }
    ASSERT_TRUE((*reference)->finish()); ASSERT_TRUE(complete(**source));
    RecordProperty("seven_frame_feed_ms", std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - before_playback).count());
    const auto decoded = (*source)->stats().decoded_frames;
    ASSERT_EQ(decoded, 7u);
    auto original = (*source)->take(6); ASSERT_TRUE(original);
    ASSERT_TRUE((*source)->request({6, 1}));
    auto cached = (*source)->take(6); ASSERT_TRUE(cached);
    EXPECT_EQ(cached->get(), original->get());
    ASSERT_TRUE(complete(**source)); EXPECT_EQ((*source)->stats().decoded_frames, decoded);
    original = make_error(ErrorCode::Cancelled, "release"); cached = make_error(ErrorCode::Cancelled, "release");
    ASSERT_TRUE((*source)->request({0, 1}));
    auto first = (*source)->take(0); ASSERT_TRUE(first) << first.error().message;
    EXPECT_LT((*first)->encoded_rgb.at(0, 0, 0), 1);
}

TEST_F(HeadroomFrames, ConsumerRetentionIsIncludedInBudgetAndEvictionAllowsProgress) {
    auto source = open(1); ASSERT_TRUE(source);
    ASSERT_TRUE((*source)->request({0, 7}));
    auto first = (*source)->take(0); ASSERT_TRUE(first);
    EXPECT_EQ((*first)->encoded_rgb.at(1, 0, 0), -.125f);
    EXPECT_EQ((*first)->encoded_rgb.at(0, 0, 0), 1.25f);
    auto busy = (*source)->take(1, 40ms);
    ASSERT_FALSE(busy); EXPECT_EQ(busy.error().code, ErrorCode::Busy);
    EXPECT_EQ((*source)->stats().live_pixel_bytes, kFrameBytes);
    EXPECT_EQ((*source)->stats().cached_frames, 0u); // evicted but consumer owns it
    first = make_error(ErrorCode::Cancelled, "release");
    for (int i = 1; i < 7; ++i) {
        auto frame = (*source)->take(i); ASSERT_TRUE(frame) << frame.error().message;
        EXPECT_LE((*source)->stats().live_pixel_bytes, kFrameBytes);
    }
    EXPECT_EQ((*source)->take(0).error().code, ErrorCode::NotFound);
    EXPECT_TRUE(complete(**source));
}

TEST_F(HeadroomFrames, NewRangeCancelsWaiterAndCannotPublishStaleBatch) {
    const auto stall = dir / "stall.mkv"; fs::copy_file(movie, stall);
    auto source = open(2, false, stall); ASSERT_TRUE(source);
    ASSERT_TRUE((*source)->request({0, 2}));
    std::promise<void> entered;
    auto waiter = std::async(std::launch::async, [&] { entered.set_value(); return (*source)->take(0); });
    entered.get_future().wait();
    ASSERT_EQ(waiter.wait_for(100ms), std::future_status::timeout);
    ASSERT_TRUE((*source)->request({6, 1}));
    ASSERT_EQ(waiter.wait_for(2s), std::future_status::ready);
    EXPECT_EQ(waiter.get().error().code, ErrorCode::Cancelled);
    auto latest = (*source)->take(6); ASSERT_TRUE(latest) << latest.error().message;
    ASSERT_TRUE(complete(**source));
    EXPECT_EQ((*source)->stats().decoded_frames, 1u);
    EXPECT_EQ((*source)->stats().cached_frames, 1u);
}

TEST_F(HeadroomFrames, CancelAndDestructionUnblockBothReadAndMemoryPressure) {
    const auto stall = dir / "stall.mkv"; fs::copy_file(movie, stall);
    auto source = open(1, false, stall); ASSERT_TRUE(source);
    ASSERT_TRUE((*source)->request({0, 1}));
    EXPECT_EQ((*source)->take(0, 50ms).error().code, ErrorCode::Busy);
    (*source)->cancel(); (*source)->cancel();
    EXPECT_EQ((*source)->take(0).error().code, ErrorCode::Cancelled);
    const auto start = std::chrono::steady_clock::now();
    source->reset();
    EXPECT_LT(std::chrono::steady_clock::now() - start, 2s);
    source = open(1); ASSERT_TRUE(source); ASSERT_TRUE((*source)->request({0, 7}));
    auto pinned = (*source)->take(0); ASSERT_TRUE(pinned);
    EXPECT_EQ((*source)->take(1, 40ms).error().code, ErrorCode::Busy);
    const auto blocked = std::chrono::steady_clock::now(); source->reset();
    EXPECT_LT(std::chrono::steady_clock::now() - blocked, 2s);
    EXPECT_EQ((*pinned)->encoded_rgb.at(0, 0, 0), 1.25f); // safe after source destruction
}

TEST_F(HeadroomFrames, FailedTruncatedExtraAndNonfiniteBatchesNeverEnterCache) {
    for (const std::string name : {"late-failure", "truncated", "extra", "nonfinite"}) {
        SCOPED_TRACE(name);
        auto file = dir / (name + ".mkv"); fs::copy_file(movie, file);
        auto source = open(2, false, file); ASSERT_TRUE(source);
        ASSERT_TRUE((*source)->request({0, 2}));
        auto result = (*source)->take(0); ASSERT_FALSE(result);
        EXPECT_EQ(result.error().code, ErrorCode::IoError);
        EXPECT_EQ((*source)->stats().cached_frames, 0u);
        EXPECT_EQ((*source)->stats().live_pixel_bytes, 0u);
        EXPECT_FALSE((*source)->stats().complete);
        EXPECT_EQ((*source)->take(1).error().message, result.error().message);
    }
}

TEST_F(HeadroomFrames, SourceAndDecoderAreProtectedUntilCloseAndReopenChangesIdentity) {
    const auto exe = dir / "producer.exe"; fs::copy_file(RUDRA_HEADROOM_PRODUCER_PATH, exe);
    auto source = HeadroomFrameSource::open(movie, kPolicy, exe, {kFrameBytes, 1});
    ASSERT_TRUE(source) << source.error().message;
    const auto first_id = (*source)->cache_id();
    for (const auto& path : {movie, exe}) {
        std::ofstream writer(path, std::ios::binary | std::ios::app); EXPECT_FALSE(writer.is_open());
        std::error_code ec; fs::rename(path, path.string() + ".moved", ec); EXPECT_TRUE(ec);
    }
    ASSERT_TRUE((*source)->request({0, 1})); ASSERT_TRUE((*source)->take(0));
    source->reset();
    const auto time = fs::last_write_time(movie);
    { std::ofstream out(movie, std::ios::binary | std::ios::app); ASSERT_TRUE(out); out.put('\0'); }
    fs::last_write_time(movie, time);
    source = HeadroomFrameSource::open(movie, kPolicy, exe, {kFrameBytes, 1}); ASSERT_TRUE(source);
    EXPECT_NE((*source)->cache_id(), first_id);
    const auto second_id = (*source)->cache_id(); source->reset();
    { std::ofstream out(exe, std::ios::binary | std::ios::app); ASSERT_TRUE(out); out.put('\0'); }
    source = HeadroomFrameSource::open(movie, kPolicy, exe, {kFrameBytes, 1}); ASSERT_TRUE(source);
    EXPECT_NE((*source)->cache_id(), second_id);
}

TEST_F(HeadroomFrames, InvalidPoliciesLimitsAndRangesFailWithoutReplacingActiveRequest) {
    EXPECT_FALSE(HeadroomFrameSource::open(movie, HeadroomPolicy::Disabled, ffmpeg));
    EXPECT_FALSE(HeadroomFrameSource::open(movie, kPolicy, ffmpeg, {kFrameBytes, 0}));
    EXPECT_FALSE(HeadroomFrameSource::open(movie, kPolicy, ffmpeg, {kFrameBytes, 33}));
    auto small = HeadroomFrameSource::open(movie, kPolicy, ffmpeg, {kFrameBytes - 1, 1});
    ASSERT_FALSE(small); EXPECT_EQ(small.error().code, ErrorCode::InvalidArgument);
    auto source = open(); ASSERT_TRUE(source);
    EXPECT_EQ((*source)->take(0).error().code, ErrorCode::InvalidArgument);
    ASSERT_TRUE((*source)->request({0, 1}));
    for (auto range : {FloatFrameRange{-1, 1}, {0, 0}, {7, 1}, {6, 2}, {1, INT_MAX}})
        EXPECT_EQ((*source)->request(range).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ((*source)->take(0, -1ms).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ((*source)->take(1).error().code, ErrorCode::InvalidArgument);
    EXPECT_TRUE((*source)->take(0));
}

TEST_F(HeadroomFrames, BorrowedFramesSurviveSeekAndStillBoundNewAllocations) {
    auto source = open(2); ASSERT_TRUE(source); ASSERT_TRUE((*source)->request({0, 2}));
    auto a = (*source)->take(0); ASSERT_TRUE(a);
    auto b = (*source)->take(1); ASSERT_TRUE(b);
    ASSERT_TRUE((*source)->request({6, 1}));
    auto busy = (*source)->take(6, 40ms); ASSERT_FALSE(busy); EXPECT_EQ(busy.error().code, ErrorCode::Busy);
    EXPECT_EQ((*source)->stats().live_pixel_bytes, 2 * kFrameBytes);
    a = make_error(ErrorCode::Cancelled, "release");
    auto c = (*source)->take(6); ASSERT_TRUE(c);
    EXPECT_LE((*source)->stats().live_pixel_bytes, 2 * kFrameBytes);
    EXPECT_EQ((*b)->encoded_rgb.at(0, 0, 0), 1.25f);
}

TEST_F(HeadroomFrames, ExistingWriterPreventsOpeningAnUnstableSource) {
    std::ofstream writer(movie, std::ios::binary | std::ios::app); ASSERT_TRUE(writer);
    auto source = open(); ASSERT_FALSE(source);
    EXPECT_EQ(source.error().code, ErrorCode::Busy);
    writer.close();
    source = open(); EXPECT_TRUE(source);
}

TEST_F(HeadroomFrames, ReclaimsEnoughPastFramesForABatchWhileCountingBorrowedFrames) {
    auto source = open(4); ASSERT_TRUE(source); ASSERT_TRUE((*source)->request({0, 4}));
    auto borrowed = (*source)->take(0); ASSERT_TRUE(borrowed);
    ASSERT_TRUE(complete(**source));
    const auto before = (*source)->stats().decoder_starts;
    ASSERT_EQ((*source)->stats().cached_frames, 4u);
    ASSERT_TRUE((*source)->request({4, 2}));
    ASSERT_TRUE((*source)->take(4)); ASSERT_TRUE((*source)->take(5)); ASSERT_TRUE(complete(**source));
    EXPECT_EQ((*source)->stats().decoder_starts - before, 1u);
    EXPECT_LE((*source)->stats().peak_pixel_bytes, 4 * kFrameBytes);
    EXPECT_EQ((*borrowed)->encoded_rgb.at(0, 0, 0), 1.25f);
}

TEST_F(HeadroomFrames, MetadataChangesRequireFreshPreparationEvenAfterRestoringTimestamp) {
    const auto exe = dir / "producer.exe"; fs::copy_file(RUDRA_HEADROOM_PRODUCER_PATH, exe);
    auto source = HeadroomFrameSource::open(movie, kPolicy, exe, {kFrameBytes, 1}); ASSERT_TRUE(source);
    for (const auto& file : {movie, exe}) {
        ASSERT_TRUE((*source)->request({0, 1})); ASSERT_TRUE((*source)->take(0));
        ASSERT_TRUE(complete(**source));
        const auto before = fs::last_write_time(file);
        // Windows sharing protects content writes, but attribute-only access
        // can still change metadata. Verify on hits as well as new decoding.
        std::error_code ec; fs::last_write_time(file, before + 2s, ec); ASSERT_FALSE(ec) << ec.message();
        auto stale = (*source)->take(0); ASSERT_FALSE(stale);
        EXPECT_EQ(stale.error().code, ErrorCode::IntegrityError);
        EXPECT_EQ((*source)->stats().cached_frames, 0u);
        EXPECT_EQ((*source)->request({0, 1}).error().code, ErrorCode::IntegrityError);
        fs::last_write_time(file, before);
        EXPECT_EQ((*source)->take(0).error().code, ErrorCode::IntegrityError);
        EXPECT_EQ((*source)->request({0, 1}).error().code, ErrorCode::IntegrityError);
        source->reset();
        source = HeadroomFrameSource::open(movie, kPolicy, exe, {kFrameBytes, 1}); ASSERT_TRUE(source);
        ASSERT_TRUE((*source)->request({0, 1})); EXPECT_TRUE((*source)->take(0));
    }
}

TEST_F(HeadroomFrames, ActiveSourceInvalidationKeepsItsErrorAfterDecoderCancellation) {
    const auto file = dir / "stall.mkv"; fs::copy_file(movie, file);
    auto source = open(2, false, file); ASSERT_TRUE(source);
    ASSERT_TRUE((*source)->request({0, 2}));
    EXPECT_EQ((*source)->take(0, 100ms).error().code, ErrorCode::Busy);
    const auto before = fs::last_write_time(file);
    fs::last_write_time(file, before + 2s);
    EXPECT_EQ((*source)->take(0).error().code, ErrorCode::IntegrityError);
    fs::last_write_time(file, before);
    const auto until = std::chrono::steady_clock::now() + 2s;
    while ((*source)->stats().live_pixel_bytes && std::chrono::steady_clock::now() < until)
        std::this_thread::sleep_for(2ms);
    EXPECT_EQ((*source)->stats().live_pixel_bytes, 0u);
    EXPECT_EQ((*source)->stats().cached_frames, 0u);
    EXPECT_EQ((*source)->take(0).error().code, ErrorCode::IntegrityError);
    EXPECT_EQ((*source)->request({6, 1}).error().code, ErrorCode::IntegrityError);
    source->reset(); source = open(2, false, file); ASSERT_TRUE(source);
    ASSERT_TRUE((*source)->request({6, 1})); EXPECT_TRUE((*source)->take(6));
}

TEST(FloatDecodeLifecycle, InterruptAlsoUnblocksFinishBeforeAndAfterPipeEof) {
    for (const std::string mode : {"wait-before-eof", "wait-after-eof"}) {
        auto d = FloatVideoDecoder::open_command({RUDRA_HEADROOM_PRODUCER_PATH, mode}, 1, 1, 0, {}, {});
        ASSERT_TRUE(d);
        auto end = std::async(std::launch::async, [&] { return (*d)->finish(); });
        ASSERT_EQ(end.wait_for(100ms), std::future_status::timeout);
        (*d)->interrupt();
        ASSERT_EQ(end.wait_for(2s), std::future_status::ready);
        auto result = end.get(); ASSERT_FALSE(result); EXPECT_EQ(result.error().code, ErrorCode::Cancelled);
        EXPECT_EQ((*d)->finish().error().code, ErrorCode::Cancelled);
    }
}

static_assert(!std::is_default_constructible_v<HeadroomDecodeReceipt>);
static_assert(!std::is_default_constructible_v<HeadroomPreviewFrame>);
static_assert(!std::is_convertible_v<HeadroomPreviewFrame, HeadroomFrameSource::Frame>);

TEST_F(HeadroomFrames, StreamRealRangesMatchFullDecodeAndGradeWithOneDecoder) {
    auto reference_video = open_video_source(movie, VideoArgs{}); ASSERT_TRUE(reference_video);
    auto reference = FloatVideoDecoder::open(movie, *reference_video, dir / "full.log"); ASSERT_TRUE(reference);
    std::vector<ExtendedVideoFrame> expected;
    for (int i = 0; i < 7; ++i) { auto f = (*reference)->next(); ASSERT_TRUE(f); expected.push_back(std::move(*f)); }
    ASSERT_TRUE((*reference)->finish());
    MasterRequest grade;
    grade.analytic = true; grade.source_curve = "bt1886"; grade.source_space = "rec709";
    grade.clip_fill = grade.anchor = grade.carry_chroma = grade.settle_grain = false;
    grade.exposure_ev = 1;
    for (auto range : {FloatFrameRange{0, 7}, {2, 3}, {6, 1}}) {
        auto source = stream(1, true); ASSERT_TRUE(source);
        ASSERT_TRUE((*source)->start(range));
        for (int i = range.first; i < range.first + range.count; ++i) {
            auto frame = (*source)->next(); ASSERT_TRUE(frame) << frame.error().message;
            EXPECT_EQ(frame->index(), i); ASSERT_TRUE((*source)->validate_preview(*frame));
            const auto pixels = frame->pixels().encoded_rgb.span();
            const auto wanted_pixels = expected[std::size_t(i)].encoded_rgb.span();
            EXPECT_TRUE(std::equal(pixels.begin(), pixels.end(), wanted_pixels.begin(), wanted_pixels.end()));
            auto actual = render_headroom_frame(frame->pixels(), grade, kPolicy);
            auto wanted = render_headroom_frame(expected[std::size_t(i)], grade, kPolicy);
            ASSERT_TRUE(actual); ASSERT_TRUE(wanted);
            EXPECT_EQ(actual->linear2020.buffer().vector(), wanted->linear2020.buffer().vector());
            EXPECT_FALSE((*source)->stats().complete);
            EXPECT_LE((*source)->stats().live_pixel_bytes, kFrameBytes);
        }
        auto receipt = (*source)->finish(); ASSERT_TRUE(receipt) << receipt.error().message;
        EXPECT_TRUE((*source)->validate_completion(*receipt)); EXPECT_TRUE((*source)->finish());
        EXPECT_EQ((*source)->stats().decoder_starts, 1u);
        EXPECT_EQ((*source)->stats().decoded_frames, std::uint64_t(range.count));
    }
}

TEST_F(HeadroomFrames, StreamRetentionBoundsMemoryAndOnlyCompletionAllowsReuse) {
    auto source = stream(1); ASSERT_TRUE(source); ASSERT_TRUE((*source)->start({0, 7}));
    auto pinned = (*source)->next(); ASSERT_TRUE(pinned);
    EXPECT_EQ((*source)->next().error().code, ErrorCode::Busy);
    EXPECT_EQ((*source)->stats().decoded_frames, 1u);
    EXPECT_EQ((*source)->finish().error().code, ErrorCode::InvalidArgument);
    pinned = make_error(ErrorCode::Cancelled, "release");
    for (int i = 1; i < 6; ++i) { auto f = (*source)->next(); ASSERT_TRUE(f); EXPECT_EQ(f->index(), i); }
    auto last = (*source)->next(); ASSERT_TRUE(last);
    auto receipt = (*source)->finish(); ASSERT_TRUE(receipt);
    auto verified = (*source)->verified_frame(*last, *receipt); ASSERT_TRUE(verified);
    EXPECT_EQ(verified->get(), &last->pixels());
    EXPECT_EQ((*source)->stats().peak_pixel_bytes, kFrameBytes);
    EXPECT_EQ((*source)->stats().decoder_starts, 1u);
    (*source)->cancel();
    EXPECT_EQ((*source)->validate_completion(*receipt).error().code, ErrorCode::Cancelled);
    EXPECT_EQ((*source)->verified_frame(*last, *receipt).error().code, ErrorCode::Cancelled);
    EXPECT_EQ((*source)->validate_preview(*last).error().code, ErrorCode::Cancelled);
    EXPECT_FALSE((*source)->stats().complete);
    source->reset();
    EXPECT_EQ(last->pixels().encoded_rgb.at(0, 0, 0), 1.25f); // lifetime, not trust
}

TEST_F(HeadroomFrames, StreamLateFailureNeverCompletesStagedOutputOrPromotesPreview) {
    for (const std::string name : {"late-failure", "extra", "truncated", "nonfinite"}) {
        SCOPED_TRACE(name);
        const auto file = dir / (name + ".mkv"); fs::copy_file(movie, file);
        auto source = stream(2, false, file); ASSERT_TRUE(source); ASSERT_TRUE((*source)->start({0, 7}));
        auto first = (*source)->next(); ASSERT_TRUE(first); ASSERT_TRUE((*source)->validate_preview(*first));
        int staged = 1;
        for (int i = 1; i < 7; ++i) { auto f = (*source)->next(); if (!f) break; ++staged; }
        auto receipt = (*source)->finish(); ASSERT_FALSE(receipt);
        EXPECT_EQ(receipt.error().code, ErrorCode::IoError);
        EXPECT_EQ(staged, (name == "truncated" || name == "nonfinite") ? 6 : 7);
        // A staging consumer must discard its temporary output on this error.
        // Seven finite frames alone must never be taken as export success.
        EXPECT_FALSE((*source)->stats().complete);
        EXPECT_EQ((*source)->validate_preview(*first).error().message, receipt.error().message);
        EXPECT_EQ((*source)->finish().error().message, receipt.error().message);
        EXPECT_EQ((*source)->next().error().message, receipt.error().message);
        EXPECT_EQ((*source)->stats().decoder_starts, 1u);
    }
}

TEST_F(HeadroomFrames, StreamReceiptsAndFramesCannotCrossOwnersOrRestart) {
    auto a = stream(), b = stream(); ASSERT_TRUE(a); ASSERT_TRUE(b);
    EXPECT_EQ((*a)->cache_id(), (*b)->cache_id()); // same content is not same generation
    ASSERT_TRUE((*a)->start({0, 1})); ASSERT_TRUE((*b)->start({1, 1}));
    auto fa = (*a)->next(), fb = (*b)->next(); ASSERT_TRUE(fa); ASSERT_TRUE(fb);
    auto ra = (*a)->finish(); ASSERT_TRUE(ra);
    EXPECT_EQ((*b)->validate_preview(*fa).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ((*b)->validate_completion(*ra).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ((*b)->verified_frame(*fb, *ra).error().code, ErrorCode::InvalidArgument);
    auto rb = (*b)->finish(); ASSERT_TRUE(rb);
    EXPECT_EQ((*a)->verified_frame(*fb, *ra).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ((*a)->start({0, 1}).error().code, ErrorCode::InvalidArgument);
    (*a)->cancel();
    EXPECT_EQ((*a)->start({0, 1}).error().code, ErrorCode::Cancelled);
    EXPECT_TRUE((*b)->verified_frame(*fb, *rb));
}

TEST_F(HeadroomFrames, StreamCancellationUnblocksReadAndTerminalWaits) {
    for (const std::string name : {"stall", "finish-wait-before", "finish-wait-after"}) {
        SCOPED_TRACE(name);
        const auto file = dir / (name + ".mkv"); fs::copy_file(movie, file);
        auto source = stream(1, false, file); ASSERT_TRUE(source); ASSERT_TRUE((*source)->start({0, 1}));
        if (name != "stall") ASSERT_TRUE((*source)->next());
        auto reader = std::async(std::launch::async, [&]() -> Result<void> {
            if (name == "stall") { auto f = (*source)->next(); if (!f) return f.error(); }
            else { auto r = (*source)->finish(); if (!r) return r.error(); }
            return {};
        });
        ASSERT_EQ(reader.wait_for(100ms), std::future_status::timeout);
        EXPECT_EQ((*source)->next().error().code, ErrorCode::Busy);
        EXPECT_EQ((*source)->finish().error().code, ErrorCode::Busy);
        (*source)->cancel(); (*source)->cancel();
        ASSERT_EQ(reader.wait_for(2s), std::future_status::ready);
        EXPECT_EQ(reader.get().error().code, ErrorCode::Cancelled);
        EXPECT_EQ((*source)->stats().live_pixel_bytes, 0u);
        EXPECT_FALSE((*source)->stats().complete);
    }
}

TEST_F(HeadroomFrames, StreamLeasesProtectContentAndInvalidateReceiptsOnMetadataChange) {
    const auto exe = dir / "producer.exe"; fs::copy_file(RUDRA_HEADROOM_PRODUCER_PATH, exe);
    for (const auto& file : {movie, exe}) {
        auto source = HeadroomStream::open(movie, kPolicy, exe, kFrameBytes); ASSERT_TRUE(source);
        { std::ofstream writer(file, std::ios::binary | std::ios::app); EXPECT_FALSE(writer.is_open()); }
        std::error_code ec; fs::rename(file, file.string() + ".moved", ec); EXPECT_TRUE(ec);
        ASSERT_TRUE((*source)->start({0, 1})); auto f = (*source)->next(); ASSERT_TRUE(f);
        auto receipt = (*source)->finish(); ASSERT_TRUE(receipt);
        const auto before = fs::last_write_time(file); fs::last_write_time(file, before + 2s);
        EXPECT_EQ((*source)->validate_completion(*receipt).error().code, ErrorCode::IntegrityError);
        fs::last_write_time(file, before);
        EXPECT_EQ((*source)->verified_frame(*f, *receipt).error().code, ErrorCode::IntegrityError);
        (*source)->cancel();
        EXPECT_EQ((*source)->finish().error().code, ErrorCode::IntegrityError);
    }
}

TEST_F(HeadroomFrames, StreamIntegrityFailureInterruptsActiveReadAndRemainsLatched) {
    const auto file = dir / "read-wait.mkv"; fs::copy_file(movie, file);
    auto source = stream(2, false, file); ASSERT_TRUE(source); ASSERT_TRUE((*source)->start({0, 7}));
    auto first = (*source)->next(); ASSERT_TRUE(first);
    auto reader = std::async(std::launch::async, [&] { return (*source)->next(); });
    ASSERT_EQ(reader.wait_for(100ms), std::future_status::timeout);
    const auto before = fs::last_write_time(file); fs::last_write_time(file, before + 2s);
    EXPECT_EQ((*source)->validate_preview(*first).error().code, ErrorCode::IntegrityError);
    ASSERT_EQ(reader.wait_for(2s), std::future_status::ready);
    EXPECT_EQ(reader.get().error().code, ErrorCode::IntegrityError);
    fs::last_write_time(file, before);
    EXPECT_EQ((*source)->start({0, 1}).error().code, ErrorCode::IntegrityError);
}

TEST_F(HeadroomFrames, StreamMetadataChangeBetweenFramesPreventsFurtherPublication) {
    auto source = stream(); ASSERT_TRUE(source); ASSERT_TRUE((*source)->start({0, 7}));
    auto f = (*source)->next(); ASSERT_TRUE(f);
    const auto before = fs::last_write_time(movie); fs::last_write_time(movie, before + 2s);
    EXPECT_EQ((*source)->next().error().code, ErrorCode::IntegrityError);
    EXPECT_EQ((*source)->validate_preview(*f).error().code, ErrorCode::IntegrityError);
    fs::last_write_time(movie, before);
    EXPECT_EQ((*source)->finish().error().code, ErrorCode::IntegrityError);
}

TEST_F(HeadroomFrames, StreamRejectsInvalidInputsAndPrematureCompletion) {
    EXPECT_FALSE(HeadroomStream::open(movie, HeadroomPolicy::Disabled, ffmpeg));
    EXPECT_FALSE(HeadroomStream::open(movie, kPolicy, ffmpeg, 0));
    EXPECT_FALSE(HeadroomStream::open(movie, kPolicy, ffmpeg, kFrameBytes - 1));
    auto source = stream(); ASSERT_TRUE(source);
    EXPECT_EQ((*source)->next().error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ((*source)->finish().error().code, ErrorCode::InvalidArgument);
    for (auto r : {FloatFrameRange{-1, 1}, {0, 0}, {7, 1}, {6, 2}, {1, INT_MAX}})
        EXPECT_EQ((*source)->start(r).error().code, ErrorCode::InvalidArgument);
    ASSERT_TRUE((*source)->start({6, 1})); ASSERT_TRUE((*source)->next());
    EXPECT_EQ((*source)->next().error().code, ErrorCode::InvalidArgument);
    EXPECT_TRUE((*source)->finish());
}

TEST_F(HeadroomFrames, RealStreamCancellationAndDestructionReleaseBlockedOutput) {
    const auto file = dir / "backpressure.mkv";
    auto made = run_process({ffmpeg.string(), "-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i",
        "testsrc2=size=256x144:rate=24,format=yuv444p10le,"
        "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
        "-frames:v", "96", "-c:v", "ffv1", "-threads", "2", "-pix_fmt", "yuv444p10le",
        "-color_range", "tv", "-color_trc", "bt709", "-color_primaries", "bt709", "-colorspace", "bt709",
        file.string()});
    ASSERT_TRUE(made); ASSERT_EQ(made->exit_code, 0) << made->err;
    constexpr std::size_t bytes = 256 * 144 * 3 * sizeof(float);
    for (bool explicit_cancel : {false, true}) {
        auto source = HeadroomStream::open(file, kPolicy, ffmpeg, bytes, dir / "backpressure.log");
        ASSERT_TRUE(source); ASSERT_TRUE((*source)->start({0, 96}));
        auto held = (*source)->next(); ASSERT_TRUE(held);
        const auto sample = held->pixels().encoded_rgb.at(0, 0, 0);
        // Remaining output is ~40 MiB, larger than the pipe. The consumer holds
        // its sole pixel slot while FFmpeg runs into downstream backpressure.
        std::this_thread::sleep_for(100ms);
        EXPECT_EQ((*source)->next().error().code, ErrorCode::Busy);
        EXPECT_EQ((*source)->stats().live_pixel_bytes, bytes);
        const auto before = std::chrono::steady_clock::now();
        if (explicit_cancel) {
            (*source)->cancel();
            EXPECT_EQ((*source)->validate_preview(*held).error().code, ErrorCode::Cancelled);
            EXPECT_EQ((*source)->finish().error().code, ErrorCode::Cancelled);
        }
        source->reset();
        EXPECT_LT(std::chrono::steady_clock::now() - before, 2s);
        EXPECT_EQ(held->pixels().encoded_rgb.at(0, 0, 0), sample);
    }
}

TEST_F(HeadroomFrames, PreparedSourceReusesProbeIdentityAndLeasesThroughLastOwner) {
    const auto file = probe_fixture("reuse");
    const auto exe = dir / "producer.exe"; fs::copy_file(RUDRA_HEADROOM_PRODUCER_PATH, exe);
    auto prepared = PreparedHeadroomSource::prepare(file, kPolicy, exe, {{}, RUDRA_HEADROOM_PRODUCER_PATH});
    ASSERT_TRUE(prepared) << prepared.error().message;
    auto expected_id = headroom_source_cache_id(file, (*prepared)->video(), kPolicy, exe);
    ASSERT_TRUE(expected_id); EXPECT_EQ(*expected_id, (*prepared)->cache_id());
    auto first = HeadroomStream::open(*prepared, kFrameBytes); ASSERT_TRUE(first);
    auto second = HeadroomStream::open(*prepared, kFrameBytes); ASSERT_TRUE(second);
    auto batch = HeadroomFrameSource::open(*prepared, {kFrameBytes, 1}); ASSERT_TRUE(batch);
    EXPECT_EQ(&(*first)->video(), &(*prepared)->video());
    EXPECT_EQ(&(*batch)->video(), &(*prepared)->video());
    EXPECT_EQ((*second)->cache_id(), *expected_id);
    prepared->reset(); first->reset(); second->reset();
    for (const auto& path : {file, exe}) {
        std::ofstream writer(path, std::ios::app); EXPECT_FALSE(writer.is_open());
        std::error_code ec; fs::rename(path, path.string() + ".moved", ec); EXPECT_TRUE(ec);
    }
    ASSERT_TRUE((*batch)->request({6, 1})); ASSERT_TRUE((*batch)->take(6));
    EXPECT_EQ(probe_calls(file), "streams\nframes\n");
    batch->reset();
    for (const auto& path : {file, exe}) { std::ofstream writer(path, std::ios::app); EXPECT_TRUE(writer.is_open()); }
}

TEST_F(HeadroomFrames, SharedPreparationDoesNotShareRangeCancellationOrReceiptIdentity) {
    rudra::stop_source stop;
    auto prepared = PreparedHeadroomSource::prepare(movie, kPolicy, RUDRA_HEADROOM_PRODUCER_PATH, {stop.get_token()});
    ASSERT_TRUE(prepared);
    auto first = HeadroomStream::open(*prepared, kFrameBytes); ASSERT_TRUE(first);
    auto second = HeadroomStream::open(*prepared, kFrameBytes); ASSERT_TRUE(second);
    stop.request_stop(); // token only belongs to completed preparation
    ASSERT_TRUE((*first)->start({0, 1})); auto a = (*first)->next(); ASSERT_TRUE(a);
    auto receipt = (*first)->finish(); ASSERT_TRUE(receipt);
    ASSERT_TRUE((*second)->start({6, 1})); auto b = (*second)->next(); ASSERT_TRUE(b);
    EXPECT_EQ((*second)->validate_completion(*receipt).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ((*second)->validate_preview(*a).error().code, ErrorCode::InvalidArgument);
    (*first)->cancel(); EXPECT_TRUE((*prepared)->verify());
    auto own_receipt = (*second)->finish(); ASSERT_TRUE(own_receipt);
    EXPECT_TRUE((*second)->verified_frame(*b, *own_receipt));
    // Explicitly per-owner: sharing metadata does not promise a combined cap.
    EXPECT_EQ((*first)->stats().live_pixel_bytes, kFrameBytes);
    EXPECT_EQ((*second)->stats().live_pixel_bytes, kFrameBytes);
}

TEST_F(HeadroomFrames, SharedIdentityFailureInvalidatesAllOwnersAndCannotBeResetByRestoredMetadata) {
    const auto exe = dir / "producer.exe"; fs::copy_file(RUDRA_HEADROOM_PRODUCER_PATH, exe);
    for (const auto& file : {movie, exe}) {
        auto prepared = PreparedHeadroomSource::prepare(movie, kPolicy, exe); ASSERT_TRUE(prepared);
        auto first = HeadroomStream::open(*prepared, kFrameBytes); ASSERT_TRUE(first);
        auto second = HeadroomStream::open(*prepared, kFrameBytes); ASSERT_TRUE(second);
        auto batch = HeadroomFrameSource::open(*prepared, {kFrameBytes, 1}); ASSERT_TRUE(batch);
        ASSERT_TRUE((*first)->start({0, 1})); auto a = (*first)->next(); ASSERT_TRUE(a);
        auto receipt = (*first)->finish(); ASSERT_TRUE(receipt);
        ASSERT_TRUE((*batch)->request({0, 1})); ASSERT_TRUE((*batch)->take(0));
        const auto before = fs::last_write_time(file); fs::last_write_time(file, before + 2s);
        EXPECT_EQ((*second)->start({6, 1}).error().code, ErrorCode::IntegrityError);
        fs::last_write_time(file, before);
        EXPECT_EQ((*prepared)->verify().error().code, ErrorCode::IntegrityError);
        EXPECT_EQ((*first)->validate_completion(*receipt).error().code, ErrorCode::IntegrityError);
        EXPECT_EQ((*first)->validate_preview(*a).error().code, ErrorCode::IntegrityError);
        EXPECT_EQ((*batch)->take(0).error().code, ErrorCode::IntegrityError);
        EXPECT_EQ(HeadroomStream::open(*prepared).error().code, ErrorCode::IntegrityError);
        EXPECT_EQ(HeadroomFrameSource::open(*prepared).error().code, ErrorCode::IntegrityError);
    }
}

TEST_F(HeadroomFrames, PreparationCancellationInterruptsBothProbePassesBeforeEofAndAfterEof) {
    for (const std::string pass : {"stream", "frame"}) for (const std::string phase : {"before", "after"}) {
        const auto file = probe_fixture("probe-wait-" + pass + "-" + phase);
        rudra::stop_source stop;
        auto work = std::async(std::launch::async, [&] {
            return PreparedHeadroomSource::prepare(file, kPolicy, RUDRA_HEADROOM_PRODUCER_PATH,
                                                   {stop.get_token(), RUDRA_HEADROOM_PRODUCER_PATH});
        });
        const auto until = std::chrono::steady_clock::now() + 2s;
        const std::string expected = pass == "stream" ? "streams\n" : "streams\nframes\n";
        while (probe_calls(file) != expected && std::chrono::steady_clock::now() < until)
            std::this_thread::sleep_for(2ms);
        EXPECT_EQ(probe_calls(file), expected);
        { std::ofstream writer(file, std::ios::app); EXPECT_FALSE(writer.is_open()); }
        const auto before = std::chrono::steady_clock::now(); stop.request_stop();
        ASSERT_EQ(work.wait_for(2s), std::future_status::ready);
        auto result = work.get(); ASSERT_FALSE(result); EXPECT_EQ(result.error().code, ErrorCode::Cancelled);
        RecordProperty("cancel_" + pass + "_" + phase + "_ms",
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - before).count());
        { std::ofstream writer(file, std::ios::app); EXPECT_TRUE(writer.is_open()); }
        EXPECT_EQ(probe_calls(file), expected); // no later pass started after cancellation
    }
}

TEST_F(HeadroomFrames, CancelledPreparationAndInvalidSharedInputsNeverCreateOwners) {
    rudra::stop_source stop; stop.request_stop();
    auto prepared = PreparedHeadroomSource::prepare(movie, kPolicy, RUDRA_HEADROOM_PRODUCER_PATH, {stop.get_token()});
    ASSERT_FALSE(prepared); EXPECT_EQ(prepared.error().code, ErrorCode::Cancelled);
    EXPECT_EQ(HeadroomStream::open(movie, kPolicy, ffmpeg, kFrameBytes, {}, {stop.get_token()}).error().code,
              ErrorCode::Cancelled);
    EXPECT_EQ(HeadroomFrameSource::open(movie, kPolicy, ffmpeg, {}, {}, {stop.get_token()}).error().code,
              ErrorCode::Cancelled);
    EXPECT_EQ(HeadroomStream::open(PreparedHeadroomSource::Handle{}).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ(HeadroomFrameSource::open(PreparedHeadroomSource::Handle{}).error().code, ErrorCode::InvalidArgument);
    std::ofstream writer(movie, std::ios::app); EXPECT_TRUE(writer.is_open());
}

TEST_F(HeadroomFrames, PreparedSourceStillRejectsFullTimingAndDimensionFailures) {
    for (const std::string mode : {"pts", "dimensions"}) {
        const auto file = probe_fixture(mode);
        const auto path = file.string() + ".frames.json";
        nlohmann::json data; { std::ifstream in(path); in >> data; }
        if (mode == "pts") data["frames"][4]["best_effort_timestamp_time"] = "99.0";
        else data["frames"][4]["width"] = 18;
        { std::ofstream out(path); out << data; }
        auto prepared = PreparedHeadroomSource::prepare(file, kPolicy, RUDRA_HEADROOM_PRODUCER_PATH,
                                                        {{}, RUDRA_HEADROOM_PRODUCER_PATH});
        ASSERT_FALSE(prepared); EXPECT_EQ(prepared.error().code, ErrorCode::Unsupported);
        EXPECT_EQ(probe_calls(file), "streams\nframes\n");
        std::ofstream writer(file, std::ios::app); EXPECT_TRUE(writer.is_open());
    }
}

TEST_F(HeadroomFrames, FileHashCancellationNeverReturnsPartialIdentity) {
    const auto file = dir / "hash-cancel.bin";
    { std::ofstream out(file, std::ios::binary); const std::string chunk(1 << 20, 'x');
      for (int i = 0; i < 64; ++i) out.write(chunk.data(), std::streamsize(chunk.size())); }
    rudra::stop_source stop;
    auto work = std::async(std::launch::async, [&] { return sha256_file(file, stop.get_token()); });
    EXPECT_EQ(work.wait_for(20ms), std::future_status::timeout);
    const auto before = std::chrono::steady_clock::now(); stop.request_stop();
    ASSERT_EQ(work.wait_for(2s), std::future_status::ready);
    auto hash = work.get(); ASSERT_FALSE(hash); EXPECT_EQ(hash.error().code, ErrorCode::Cancelled);
    RecordProperty("hash_cancel_ms", std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - before).count());
    EXPECT_EQ(sha256_file(dir / "absent.bin", stop.get_token()).error().code, ErrorCode::Cancelled);
    auto normal = sha256_file(movie); ASSERT_TRUE(normal);
    auto uncancelled = sha256_file(movie, rudra::stop_source{}.get_token()); ASSERT_TRUE(uncancelled);
    EXPECT_EQ(*normal, *uncancelled);
}

TEST_F(HeadroomFrames, PreparedRealSeekFramesMatchSequentialReference) {
    auto prepared = PreparedHeadroomSource::prepare(movie, kPolicy, ffmpeg); ASSERT_TRUE(prepared);
    auto reference = FloatVideoDecoder::open(movie, (*prepared)->video(), dir / "reference.log"); ASSERT_TRUE(reference);
    std::vector<ExtendedVideoFrame> expected;
    for (int i = 0; i < 7; ++i) { auto f = (*reference)->next(); ASSERT_TRUE(f); expected.push_back(std::move(*f)); }
    ASSERT_TRUE((*reference)->finish());
    for (int index : {6, 0, 3, 6}) {
        auto owner = HeadroomStream::open(*prepared, kFrameBytes); ASSERT_TRUE(owner);
        ASSERT_TRUE((*owner)->start({index, 1})); auto f = (*owner)->next(); ASSERT_TRUE(f);
        auto receipt = (*owner)->finish(); ASSERT_TRUE(receipt);
        auto verified = (*owner)->verified_frame(*f, *receipt); ASSERT_TRUE(verified);
        EXPECT_TRUE(std::ranges::equal((*verified)->encoded_rgb.span(), expected[std::size_t(index)].encoded_rgb.span()));
        EXPECT_EQ((*owner)->stats().peak_pixel_bytes, kFrameBytes);
        EXPECT_EQ((*owner)->stats().decoder_starts, 1u);
    }
}

TEST_F(HeadroomFrames, SharedPixelBudgetCountsPromotedPixelsAfterTheirOwnerIsDestroyed) {
    auto prepared = PreparedHeadroomSource::prepare(movie, kPolicy, RUDRA_HEADROOM_PRODUCER_PATH); ASSERT_TRUE(prepared);
    auto budget = std::make_shared<HeadroomPixelBudget>(kFrameBytes);
    auto first = HeadroomStream::open(*prepared, kFrameBytes, {}, budget); ASSERT_TRUE(first);
    ASSERT_TRUE((*first)->start({0, 1})); auto preview = (*first)->next(); ASSERT_TRUE(preview);
    auto receipt = (*first)->finish(); ASSERT_TRUE(receipt);
    auto held = (*first)->verified_frame(*preview, *receipt); ASSERT_TRUE(held);
    preview = make_error(ErrorCode::Cancelled, "release"); first->reset();
    EXPECT_EQ(budget->stats().live, kFrameBytes);
    auto second = HeadroomStream::open(*prepared, kFrameBytes, {}, budget); ASSERT_TRUE(second);
    ASSERT_TRUE((*second)->start({6, 1}));
    EXPECT_EQ((*second)->next().error().code, ErrorCode::Busy);
    EXPECT_EQ((*second)->stats().decoded_frames, 0u);
    held = make_error(ErrorCode::Cancelled, "release");
    auto last = (*second)->next(); ASSERT_TRUE(last); EXPECT_EQ(last->index(), 6);
    EXPECT_TRUE((*second)->finish()); EXPECT_EQ(budget->stats().peak, kFrameBytes);
    second->reset(); EXPECT_EQ(budget->stats().live, kFrameBytes);
    last = make_error(ErrorCode::Cancelled, "release"); EXPECT_EQ(budget->stats().live, 0u);
}

TEST_F(HeadroomFrames, PlaybackRealSequenceSeekAndAnalyticMasterMatchReference) {
    auto prepared = PreparedHeadroomSource::prepare(movie, kPolicy, ffmpeg); ASSERT_TRUE(prepared);
    auto reference = FloatVideoDecoder::open(movie, (*prepared)->video(), dir / "reference.log"); ASSERT_TRUE(reference);
    std::vector<ExtendedVideoFrame> expected;
    for (int i = 0; i < 7; ++i) { auto frame = (*reference)->next(); ASSERT_TRUE(frame); expected.push_back(std::move(*frame)); }
    ASSERT_TRUE((*reference)->finish());
    auto player = playback(); ASSERT_TRUE(player);
    auto generation = (*player)->open(*prepared, {0, 7}); ASSERT_TRUE(generation);
    auto metadata = (*player)->video(*generation, 2s); ASSERT_TRUE(metadata); EXPECT_EQ(metadata->clock.frames, 7);
    MasterRequest grade; grade.analytic = true; grade.source_curve = "bt1886"; grade.source_space = "rec709";
    grade.clip_fill = grade.anchor = grade.carry_chroma = grade.settle_grain = false; grade.exposure_ev = 1;
    for (int i = 0; i < 7; ++i) {
        auto frame = (*player)->take(*generation, 2s); ASSERT_TRUE(frame) << frame.error().message;
        EXPECT_EQ(frame->index(), i); ASSERT_TRUE((*player)->validate_preview(*frame));
        EXPECT_TRUE(std::ranges::equal(frame->pixels().encoded_rgb.span(), expected[std::size_t(i)].encoded_rgb.span()));
        auto master = render_headroom_frame(frame->pixels(), grade, kPolicy); ASSERT_TRUE(master);
        auto control = render_headroom_frame(expected[std::size_t(i)], grade, kPolicy); ASSERT_TRUE(control);
        EXPECT_EQ(master->linear2020.buffer().vector(), control->linear2020.buffer().vector());
    }
    auto done = (*player)->completion(*generation, 2s); ASSERT_TRUE(done);
    EXPECT_TRUE((*player)->validate_completion(*done));
    EXPECT_EQ((*player)->take(*generation).error().code, ErrorCode::NotFound);
    for (int index : {6, 0, 3, 6}) {
        auto seek = (*player)->seek({index, 1}); ASSERT_TRUE(seek);
        EXPECT_EQ((*player)->validate_completion(*done).error().code, ErrorCode::Cancelled);
        auto frame = (*player)->take(*seek, 2s); ASSERT_TRUE(frame); EXPECT_EQ(frame->index(), index);
        auto receipt = (*player)->completion(*seek, 2s); ASSERT_TRUE(receipt);
        auto verified = (*player)->verified_frame(*frame, *receipt); ASSERT_TRUE(verified);
        EXPECT_TRUE(std::ranges::equal((*verified)->encoded_rgb.span(), expected[std::size_t(index)].encoded_rgb.span()));
        EXPECT_LE((*player)->stats().pixels.peak, 2 * kFrameBytes);
    }
    EXPECT_EQ((*player)->stats().decoded_frames, 11u); EXPECT_EQ((*player)->stats().decoder_starts, 5u);
}

TEST_F(HeadroomFrames, PlaybackRapidSeeksCancelBlockedReaderAndOldWaiter) {
    const auto file = dir / "stall.mkv"; fs::copy_file(movie, file);
    auto prepared = PreparedHeadroomSource::prepare(file, kPolicy, RUDRA_HEADROOM_PRODUCER_PATH); ASSERT_TRUE(prepared);
    auto player = playback(); ASSERT_TRUE(player);
    auto first = (*player)->open(*prepared, {0, 2}); ASSERT_TRUE(first);
    auto waiter = std::async(std::launch::async, [&] { return (*player)->take(*first, 3s); });
    ASSERT_EQ(waiter.wait_for(100ms), std::future_status::timeout);
    const auto before = std::chrono::steady_clock::now();
    for (int i = 0; i < 30; ++i) ASSERT_TRUE((*player)->seek({1 + i % 6, 1}));
    auto latest = (*player)->seek({6, 1}); ASSERT_TRUE(latest);
    const auto dispatch_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - before).count();
    RecordProperty("31_seek_requests_ms", dispatch_ms); EXPECT_LT(dispatch_ms, 100);
    ASSERT_EQ(waiter.wait_for(2s), std::future_status::ready);
    EXPECT_EQ(waiter.get().error().code, ErrorCode::Cancelled);
    auto frame = (*player)->take(*latest, 2s); ASSERT_TRUE(frame); EXPECT_EQ(frame->index(), 6);
    EXPECT_EQ(frame->generation(), *latest); EXPECT_TRUE((*player)->completion(*latest, 2s));
    EXPECT_LE((*player)->stats().pixels.peak, 2 * kFrameBytes);
}

TEST_F(HeadroomFrames, PlaybackRetiredBorrowedAndPromotedFramesBlockNewAllocations) {
    auto prepared = PreparedHeadroomSource::prepare(movie, kPolicy, RUDRA_HEADROOM_PRODUCER_PATH); ASSERT_TRUE(prepared);
    auto player = playback(1); ASSERT_TRUE(player);
    auto first = (*player)->open(*prepared, {0, 1}); ASSERT_TRUE(first);
    auto old = (*player)->take(*first, 2s); ASSERT_TRUE(old);
    auto receipt = (*player)->completion(*first, 2s); ASSERT_TRUE(receipt);
    auto promoted = (*player)->verified_frame(*old, *receipt); ASSERT_TRUE(promoted);
    auto latest = (*player)->seek({6, 1}); ASSERT_TRUE(latest);
    EXPECT_EQ((*player)->validate_preview(*old).error().code, ErrorCode::Cancelled);
    EXPECT_EQ((*player)->validate_completion(*receipt).error().code, ErrorCode::Cancelled);
    EXPECT_EQ((*player)->take(*latest, 50ms).error().code, ErrorCode::Busy);
    EXPECT_EQ((*player)->stats().decoder_starts, 1u); // no new child until space exists
    old = make_error(ErrorCode::Cancelled, "release preview; keep promoted pixels");
    EXPECT_EQ((*player)->take(*latest, 30ms).error().code, ErrorCode::Busy);
    promoted = make_error(ErrorCode::Cancelled, "release final pixel owner");
    auto frame = (*player)->take(*latest, 2s); ASSERT_TRUE(frame); EXPECT_EQ(frame->index(), 6);
    EXPECT_TRUE((*player)->completion(*latest, 2s));
    EXPECT_EQ((*player)->stats().pixels.peak, kFrameBytes);
}

TEST_F(HeadroomFrames, PlaybackCancellationAndDestructionWakeBudgetWaitWithoutReleasingBorrowedPixels) {
    auto prepared = PreparedHeadroomSource::prepare(movie, kPolicy, RUDRA_HEADROOM_PRODUCER_PATH); ASSERT_TRUE(prepared);
    auto player = playback(1); ASSERT_TRUE(player);
    auto first = (*player)->open(*prepared, {0, 1}); ASSERT_TRUE(first);
    auto held = (*player)->take(*first, 2s); ASSERT_TRUE(held);
    EXPECT_TRUE((*player)->completion(*first, 2s));
    auto second = (*player)->seek({6, 1}); ASSERT_TRUE(second);
    EXPECT_EQ((*player)->take(*second, 30ms).error().code, ErrorCode::Busy);
    auto waiter = std::async(std::launch::async, [&] { return (*player)->take(*second, 3s); });
    const auto before = std::chrono::steady_clock::now(); (*player)->cancel();
    ASSERT_EQ(waiter.wait_for(2s), std::future_status::ready);
    EXPECT_EQ(waiter.get().error().code, ErrorCode::Cancelled);
    player->reset(); EXPECT_LT(std::chrono::steady_clock::now() - before, 2s);
    EXPECT_EQ(held->pixels().encoded_rgb.at(0, 0, 0), 1.25f);
}

TEST_F(HeadroomFrames, PlaybackCancelsPreparationBeforeAndAfterBothProbePasses) {
    for (const std::string pass : {"stream", "frame"}) for (const std::string phase : {"before", "after"}) {
        const auto file = probe_fixture("probe-wait-" + pass + "-" + phase);
        auto player = playback(); ASSERT_TRUE(player);
        const auto before_open = std::chrono::steady_clock::now();
        auto first = (*player)->open(file, kPolicy, RUDRA_HEADROOM_PRODUCER_PATH, {0, 1}, RUDRA_HEADROOM_PRODUCER_PATH);
        ASSERT_TRUE(first); EXPECT_LT(std::chrono::steady_clock::now() - before_open, 100ms);
        const auto until = std::chrono::steady_clock::now() + 2s;
        const std::string expected = pass == "stream" ? "streams\n" : "streams\nframes\n";
        while (probe_calls(file) != expected && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(2ms);
        ASSERT_EQ(probe_calls(file), expected);
        EXPECT_EQ((*player)->video(*first).error().code, ErrorCode::Busy);
        const auto before_cancel = std::chrono::steady_clock::now();
        (*player)->cancel(); player->reset();
        EXPECT_LT(std::chrono::steady_clock::now() - before_cancel, 2s);
        EXPECT_EQ(probe_calls(file), expected);
        std::ofstream writer(file, std::ios::app); EXPECT_TRUE(writer.is_open());
    }
}

TEST_F(HeadroomFrames, PlaybackSourceReplacementDuringPreparationPublishesOnlyNewSource) {
    const auto slow = probe_fixture("probe-wait-frame-before");
    auto player = playback(); ASSERT_TRUE(player);
    auto first = (*player)->open(slow, kPolicy, RUDRA_HEADROOM_PRODUCER_PATH, {0, 1}, RUDRA_HEADROOM_PRODUCER_PATH);
    ASSERT_TRUE(first);
    const auto until = std::chrono::steady_clock::now() + 2s;
    while (probe_calls(slow) != "streams\nframes\n" && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(2ms);
    ASSERT_EQ(probe_calls(slow), "streams\nframes\n");
    auto latest = (*player)->open(movie, kPolicy, RUDRA_HEADROOM_PRODUCER_PATH, {6, 1}); ASSERT_TRUE(latest);
    EXPECT_EQ((*player)->video(*first).error().code, ErrorCode::Cancelled);
    auto frame = (*player)->take(*latest, 2s); ASSERT_TRUE(frame); EXPECT_EQ(frame->index(), 6);
    EXPECT_TRUE((*player)->completion(*latest, 2s));
    std::ofstream writer(slow, std::ios::app); EXPECT_TRUE(writer.is_open());
}

TEST_F(HeadroomFrames, PlaybackSeekDuringPreparationRestartsWithLatestRange) {
    const auto slow = probe_fixture("probe-wait-frame-before");
    auto player = playback(); ASSERT_TRUE(player);
    auto first = (*player)->open(slow, kPolicy, RUDRA_HEADROOM_PRODUCER_PATH, {0, 2}, RUDRA_HEADROOM_PRODUCER_PATH);
    ASSERT_TRUE(first);
    const auto until = std::chrono::steady_clock::now() + 2s;
    while (probe_calls(slow) != "streams\nframes\n" && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(2ms);
    ASSERT_EQ(probe_calls(slow), "streams\nframes\n");
    { std::ofstream allow(slow.string() + ".release-probe"); allow << "allow next probe process"; }
    auto latest = (*player)->seek({6, 1}); ASSERT_TRUE(latest);
    auto frame = (*player)->take(*latest, 2s); ASSERT_TRUE(frame); EXPECT_EQ(frame->index(), 6);
    EXPECT_EQ((*player)->take(*first).error().code, ErrorCode::Cancelled);
    EXPECT_TRUE((*player)->completion(*latest, 2s));
}

TEST_F(HeadroomFrames, PlaybackRealDecoderBackpressureCancelsAndReapsWithBorrowedFrameAlive) {
    const auto file = dir / "async-backpressure.mkv";
    auto made = run_process({ffmpeg.string(), "-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i",
        "testsrc2=size=256x144:rate=24,format=yuv444p10le,"
        "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
        "-frames:v", "96", "-c:v", "ffv1", "-level", "3", "-threads", "4", "-pix_fmt", "yuv444p10le",
        "-color_range", "tv", "-color_trc", "bt709", "-color_primaries", "bt709", "-colorspace", "bt709", file.string()});
    ASSERT_TRUE(made); ASSERT_EQ(made->exit_code, 0) << made->err;
    constexpr std::size_t bytes = 256 * 144 * 3 * sizeof(float);
    auto player = HeadroomPlayback::create({bytes, 1}); ASSERT_TRUE(player);
    auto epoch = (*player)->open(file, kPolicy, ffmpeg, {0, 96}); ASSERT_TRUE(epoch);
    auto held = (*player)->take(*epoch, 2s); ASSERT_TRUE(held);
    const auto sample = held->pixels().encoded_rgb.at(0, 0, 0);
    std::this_thread::sleep_for(100ms); // real FFmpeg fills its output pipe
    EXPECT_EQ((*player)->stats().pixels.live, bytes);
    EXPECT_EQ((*player)->stats().pixels.peak, bytes);
    const auto before = std::chrono::steady_clock::now(); (*player)->cancel();
    EXPECT_LT(std::chrono::steady_clock::now() - before, 100ms);
    player->reset(); EXPECT_LT(std::chrono::steady_clock::now() - before, 2s);
    EXPECT_EQ(held->pixels().encoded_rgb.at(0, 0, 0), sample);
}

TEST_F(HeadroomFrames, PlaybackLateFailureRevokesPreviewAndNeverIssuesReceipt) {
    const auto file = dir / "delayed-late-failure.mkv"; fs::copy_file(movie, file);
    auto player = playback(); ASSERT_TRUE(player);
    auto epoch = (*player)->open(file, kPolicy, RUDRA_HEADROOM_PRODUCER_PATH, {0, 2}); ASSERT_TRUE(epoch);
    auto provisional = (*player)->take(*epoch, 2s); ASSERT_TRUE(provisional);
    auto done = (*player)->completion(*epoch, 2s); ASSERT_FALSE(done); EXPECT_EQ(done.error().code, ErrorCode::IoError);
    EXPECT_EQ((*player)->validate_preview(*provisional).error().code, ErrorCode::IoError);
    EXPECT_EQ((*player)->take(*epoch).error().code, ErrorCode::IoError);
    EXPECT_EQ((*player)->stats().queued_frames, 0u); EXPECT_FALSE((*player)->stats().complete);
}

TEST_F(HeadroomFrames, PlaybackMalformedOutputCannotBecomeCompletedPixels) {
    for (const std::string mode : {"extra", "truncated", "nonfinite", "late-failure"}) {
        const auto file = dir / (mode + ".mkv"); fs::copy_file(movie, file);
        auto player = playback(2, 2); ASSERT_TRUE(player);
        auto epoch = (*player)->open(file, kPolicy, RUDRA_HEADROOM_PRODUCER_PATH, {0, 2}); ASSERT_TRUE(epoch);
        auto done = (*player)->completion(*epoch, 2s);
        ASSERT_FALSE(done); EXPECT_NE(done.error().code, ErrorCode::Busy);
        EXPECT_FALSE((*player)->stats().complete);
        EXPECT_EQ((*player)->stats().queued_frames, 0u);
    }
}

TEST_F(HeadroomFrames, PlaybackSeekAndDestructionInterruptReadEofAndExitWaits) {
    for (const std::string mode : {"read-wait", "finish-wait-before", "finish-wait-after"}) {
        const auto file = dir / (mode + ".mkv"); fs::copy_file(movie, file);
        for (bool destroy : {false, true}) {
            auto player = playback(3, 2); ASSERT_TRUE(player);
            auto first = (*player)->open(file, kPolicy, RUDRA_HEADROOM_PRODUCER_PATH, {0, 2}); ASSERT_TRUE(first);
            auto frame = (*player)->take(*first, 2s); ASSERT_TRUE(frame);
            EXPECT_EQ((*player)->completion(*first, 50ms).error().code, ErrorCode::Busy);
            const auto before = std::chrono::steady_clock::now();
            if (destroy) player->reset();
            else {
                auto latest = (*player)->open(movie, kPolicy, RUDRA_HEADROOM_PRODUCER_PATH, {6, 1}); ASSERT_TRUE(latest);
                auto last = (*player)->take(*latest, 2s); ASSERT_TRUE(last); EXPECT_EQ(last->index(), 6);
                EXPECT_TRUE((*player)->completion(*latest, 2s));
                EXPECT_EQ((*player)->validate_preview(*frame).error().code, ErrorCode::Cancelled);
            }
            EXPECT_LT(std::chrono::steady_clock::now() - before, 2s);
        }
    }
}

TEST_F(HeadroomFrames, PlaybackIdentityChangeInvalidatesReceiptAndReusedPreparation) {
    auto player = playback(); ASSERT_TRUE(player);
    auto epoch = (*player)->open(movie, kPolicy, RUDRA_HEADROOM_PRODUCER_PATH, {0, 1}); ASSERT_TRUE(epoch);
    auto frame = (*player)->take(*epoch, 2s); ASSERT_TRUE(frame);
    auto receipt = (*player)->completion(*epoch, 2s); ASSERT_TRUE(receipt);
    const auto time = fs::last_write_time(movie); fs::last_write_time(movie, time + 2s);
    EXPECT_EQ((*player)->validate_completion(*receipt).error().code, ErrorCode::IntegrityError);
    fs::last_write_time(movie, time);
    auto seek = (*player)->seek({6, 1}); ASSERT_TRUE(seek);
    EXPECT_EQ((*player)->take(*seek, 2s).error().code, ErrorCode::IntegrityError);
    auto reopened = (*player)->open(movie, kPolicy, RUDRA_HEADROOM_PRODUCER_PATH, {6, 1}); ASSERT_TRUE(reopened);
    auto valid = (*player)->take(*reopened, 2s); ASSERT_TRUE(valid);
}

TEST_F(HeadroomFrames, PlaybackInvalidRequestsAndForeignTokensLeaveCurrentGenerationIntact) {
    EXPECT_FALSE(HeadroomPlayback::create({0, 1})); EXPECT_FALSE(HeadroomPlayback::create({kFrameBytes, 0}));
    EXPECT_FALSE(HeadroomPlayback::create({kFrameBytes, 33}));
    auto prepared = PreparedHeadroomSource::prepare(movie, kPolicy, RUDRA_HEADROOM_PRODUCER_PATH); ASSERT_TRUE(prepared);
    auto a = playback(); auto b = playback(); ASSERT_TRUE(a); ASSERT_TRUE(b);
    EXPECT_EQ((*a)->seek({0, 1}).error().code, ErrorCode::InvalidArgument);
    auto epoch = (*a)->open(*prepared, {0, 1}); ASSERT_TRUE(epoch);
    auto foreign_epoch = (*b)->open(*prepared, {0, 1}); ASSERT_TRUE(foreign_epoch);
    auto frame = (*a)->take(*epoch, 2s); ASSERT_TRUE(frame);
    auto receipt = (*a)->completion(*epoch, 2s); ASSERT_TRUE(receipt);
    EXPECT_EQ((*b)->validate_preview(*frame).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ((*b)->validate_completion(*receipt).error().code, ErrorCode::InvalidArgument);
    for (auto range : {FloatFrameRange{-1, 1}, {0, 0}, {7, 1}, {6, 2}, {1, INT_MAX}})
        EXPECT_EQ((*a)->seek(range).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ((*a)->open({}, kPolicy, ffmpeg, {0, 1}).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ((*a)->open(movie, HeadroomPolicy::Disabled, ffmpeg, {0, 1}).error().code, ErrorCode::Unsupported);
    EXPECT_EQ((*a)->take(*epoch, -1ms).error().code, ErrorCode::InvalidArgument);
    EXPECT_EQ((*a)->stats().generation, *epoch); EXPECT_TRUE((*a)->validate_completion(*receipt));
}
