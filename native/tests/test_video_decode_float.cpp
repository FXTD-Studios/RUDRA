#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "rudra/media/video_decode.hpp"
#include "rudra/media/video_decode_float.hpp"
#include "rudra/deliver/exr.hpp"
#include "rudra/platform/process.hpp"
#include "rudra/video/headroom_baseline.hpp"
#include "rudra/video/headroom_render.hpp"
#include "rudra/video/headroom_source.hpp"

using namespace rudra;
namespace {
namespace fs = std::filesystem;
const InputContract kContract{"rec709", "rec709", "bt709", "limited"};

class FloatDecode : public ::testing::Test {
protected:
    fs::path dir;
    void SetUp() override {
        dir = fs::temp_directory_path() / ("rudra-float-decode-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(dir);
    }
    fs::path write_bytes(const std::string& name, std::span<const std::byte> data) {
        const auto path = dir / name;
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
        return path;
    }
    fs::path write_floats(const std::string& name, const std::vector<float>& data) {
        return write_bytes(name, std::as_bytes(std::span(data)));
    }
    Result<std::unique_ptr<FloatVideoDecoder>> cat(const fs::path& path, int frames = 1) {
        return FloatVideoDecoder::open_command({RUDRA_CMAKE_COMMAND, "-E", "cat", path.string()},
                                               2, 1, frames, kContract, dir / "decode.log");
    }
};

VideoSource valid_source() {
    VideoSource s;
    s.stream.width = 4;
    s.stream.height = 2;
    s.stream.pix_fmt = "yuv444p10le";
    s.stream.color_transfer = "bt709";
    s.stream.color_primaries = "bt709";
    s.stream.color_space = "bt709";
    s.stream.color_range = "tv";
    s.contract = kContract;
    s.clock.frames = 1;
    return s;
}
}  // namespace

TEST_F(FloatDecode, PlaneOrderSignedHeadroomAndContractSurvive) {
    static_assert(!std::is_convertible_v<ExtendedVideoFrame, SdrImage>);
    // G then B then R, two samples each; intentionally includes negatives
    // and values above the proposed (incorrect) universal 1.087 ceiling.
    auto decoder = cat(write_floats("planes.raw", {.25f, .5f, -.1f, 1.75f, 1.09f, .75f}));
    ASSERT_TRUE(decoder);
    auto frame = (*decoder)->next();
    ASSERT_TRUE(frame) << frame.error().message;
    EXPECT_EQ(frame->source_contract, kContract);
    EXPECT_EQ(frame->encoded_rgb.width(), 2);
    EXPECT_EQ(frame->encoded_rgb.height(), 1);
    EXPECT_EQ(frame->encoded_rgb.at(0, 0, 0), 1.09f);
    EXPECT_EQ(frame->encoded_rgb.at(0, 0, 1), .75f);
    EXPECT_EQ(frame->encoded_rgb.at(1, 0, 0), .25f);
    EXPECT_EQ(frame->encoded_rgb.at(1, 0, 1), .5f);
    EXPECT_EQ(frame->encoded_rgb.at(2, 0, 0), -.1f);
    EXPECT_EQ(frame->encoded_rgb.at(2, 0, 1), 1.75f);
    EXPECT_EQ((*decoder)->next_index(), 1);
    EXPECT_FALSE((*decoder)->next());  // cannot consume an unadvertised frame
    EXPECT_TRUE((*decoder)->finish());
    EXPECT_FALSE((*decoder)->next());
    EXPECT_FALSE((*decoder)->finish());
    (*decoder)->close();
}

TEST_F(FloatDecode, EarlyFinishDoesNotConsumeData) {
    auto decoder = cat(write_floats("two.raw", std::vector<float>(12, 1.25f)), 2);
    ASSERT_TRUE(decoder);
    EXPECT_EQ((*decoder)->finish().error().code, ErrorCode::InvalidArgument);
    ASSERT_TRUE((*decoder)->next());
    EXPECT_EQ((*decoder)->finish().error().code, ErrorCode::InvalidArgument);
    ASSERT_TRUE((*decoder)->next());
    EXPECT_TRUE((*decoder)->finish());
}

TEST_F(FloatDecode, TruncationAtEveryPlaneBoundaryIsTerminal) {
    const std::vector<float> samples(6, .5f);
    const auto all = std::as_bytes(std::span(samples));
    for (std::size_t bytes : {0u, 1u, 7u, 8u, 9u, 15u, 16u, 17u, 23u}) {
        SCOPED_TRACE(bytes);
        auto decoder = cat(write_bytes("partial.raw", all.first(bytes)));
        ASSERT_TRUE(decoder);
        auto frame = (*decoder)->next();
        ASSERT_FALSE(frame);
        EXPECT_EQ(frame.error().message, bytes == 0 ? "Float decoder ended before expected frame count"
                                                   : "Truncated float decoded frame");
        EXPECT_EQ((*decoder)->next_index(), 0);
        EXPECT_FALSE((*decoder)->next());
        auto end = (*decoder)->finish();
        ASSERT_FALSE(end);
        EXPECT_EQ(end.error().message, frame.error().message);
    }
}

TEST_F(FloatDecode, MissingAndExtraFramesAreRefused) {
    const auto one = write_floats("one.raw", std::vector<float>(6, .5f));
    auto missing = cat(one, 2);
    ASSERT_TRUE(missing);
    ASSERT_TRUE((*missing)->next());
    EXPECT_FALSE((*missing)->next());
    EXPECT_FALSE((*missing)->finish());
    const auto extra_path = write_floats("extra.raw", std::vector<float>(7, .5f));
    auto extra = cat(extra_path);
    ASSERT_TRUE(extra);
    ASSERT_TRUE((*extra)->next());
    EXPECT_EQ((*extra)->finish().error().message, "Float decoder produced unexpected extra frames");
    EXPECT_FALSE((*extra)->finish());
}

TEST_F(FloatDecode, RejectsNonfiniteValuesInEveryPlane) {
    for (float invalid : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity()}) {
        for (int plane = 0; plane < 3; ++plane) {
            std::vector<float> values(6, .5f);
            values[std::size_t(plane * 2)] = invalid;
            auto decoder = cat(write_floats("nonfinite.raw", values));
            ASSERT_TRUE(decoder);
            auto frame = (*decoder)->next();
            ASSERT_FALSE(frame);
            EXPECT_EQ(frame.error().message, "Nonfinite sample in float decoded frame");
            EXPECT_FALSE((*decoder)->finish());
        }
    }
}

TEST_F(FloatDecode, RejectsBadShapeBeforeLaunchingProcess) {
    for (auto shape : {std::array{0, 2, 1}, std::array{2, -1, 1}, std::array{2, 2, -1},
                       std::array{std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), 1}}) {
        auto d = FloatVideoDecoder::open_command({"no-such-float-test-program"}, shape[0], shape[1], shape[2],
                                                 kContract, {});
        ASSERT_FALSE(d);
        EXPECT_EQ(d.error().code, ErrorCode::InvalidArgument);
    }
    EXPECT_FALSE(FloatVideoDecoder::open_command({}, 2, 2, 1, kContract, {}));
    EXPECT_FALSE(FloatVideoDecoder::open_command({"no-such-float-test-program"}, 2, 2, 1, kContract, {}));
}

TEST_F(FloatDecode, ProcessFailureAndEmptyStreamAreDistinguished) {
    auto missing = cat(dir / "does-not-exist.raw", 0);
    ASSERT_TRUE(missing);
    const auto failed = (*missing)->finish();
    ASSERT_FALSE(failed);
    EXPECT_EQ(failed.error().message, "Float decoder process failed");
    EXPECT_FALSE(failed.error().detail.empty());
    EXPECT_LE(failed.error().detail.size(), 2000u);
    const auto complete = write_floats("complete.raw", std::vector<float>(6, .5f));
    auto late_failure = FloatVideoDecoder::open_command({RUDRA_CMAKE_COMMAND, "-E", "cat", complete.string(),
        (dir / "missing-after-frame.raw").string()}, 2, 1, 1, kContract, dir / "late.log");
    ASSERT_TRUE(late_failure);
    ASSERT_TRUE((*late_failure)->next());
    EXPECT_EQ((*late_failure)->finish().error().message, "Float decoder process failed");
    auto empty = cat(write_floats("empty.raw", {}), 0);
    ASSERT_TRUE(empty);
    EXPECT_TRUE((*empty)->finish());
}

TEST_F(FloatDecode, InterruptUnblocksReadAndCannotBecomeSuccess) {
    // A bounded producer writes nothing. Cancellation must unblock next()
    // without waiting for its natural 10-second exit.
    auto decoder = FloatVideoDecoder::open_command({RUDRA_CMAKE_COMMAND, "-E", "sleep", "10"},
                                                   2, 1, 1, kContract, dir / "cancel.log");
    ASSERT_TRUE(decoder);
    std::promise<void> entered;
    auto started = entered.get_future();
    auto reading = std::async(std::launch::async, [&] {
        entered.set_value();
        return (*decoder)->next();
    });
    started.wait();
    EXPECT_EQ(reading.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    (*decoder)->interrupt();
    ASSERT_EQ(reading.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    const auto result = reading.get();
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::Cancelled);
    EXPECT_EQ((*decoder)->finish().error().code, ErrorCode::Cancelled);
    (*decoder)->close();
    (*decoder)->close();
}

TEST_F(FloatDecode, PrototypeRefusesUnsupportedOrContradictorySources) {
    const auto valid = valid_source();
    ASSERT_TRUE(float_decoder_command("ffmpeg", "clip.mkv", valid));
    auto rejected = [&](VideoSource source) {
        EXPECT_FALSE(float_decoder_command("ffmpeg", "clip.mkv", source));
        // A fabricated path must never reach FFmpeg on refused source types.
        EXPECT_FALSE(FloatVideoDecoder::open("missing.mkv", source, dir / "refusal.log"));
        EXPECT_FALSE(fs::exists(dir / "refusal.log"));
    };
    auto s = valid; s.alpha = true; rejected(s);
    s = valid; s.stream.pix_fmt = "yuva444p10le"; rejected(s);
    s = valid; s.stream.pix_fmt = "yuv420p10le"; rejected(s);
    s = valid; s.stream.pix_fmt = "yuv444p12le"; rejected(s);
    s = valid; s.stream.color_range.reset(); rejected(s);
    s = valid; s.stream.color_transfer = "smpte2084"; rejected(s);
    s = valid; s.stream.color_transfer = "arib-std-b67"; rejected(s);
    s = valid; s.stream.rotation = 90; rejected(s);
    s = valid; s.stream.field_order = "tt"; rejected(s);
    s = valid; s.stream.sample_aspect_ratio = "4:3"; rejected(s);
    s = valid; s.contract.range = "full"; rejected(s);
    s = valid; s.stream.color_range = "pc"; s.contract.range = "full"; rejected(s);
    s = valid; s.clock.frames = 0; rejected(s);
}

TEST_F(FloatDecode, RealFfmpegPreservesNeutralAndColoredSamplesAgainstMatrix) {
    auto ffmpeg = find_executable("ffmpeg");
    if (!ffmpeg || !find_executable("ffprobe")) GTEST_SKIP() << "FFmpeg and ffprobe required";
    for (int bits : {8, 10}) {
        for (bool rec2020 : {false, true}) {
            SCOPED_TRACE(std::to_string(bits) + (rec2020 ? " BT.2020" : " BT.709"));
            const int k = 1 << (bits - 8), w = 1 << bits, h = 4;
            const std::string fmt = bits == 8 ? "yuv444p" : "yuv444p10le";
            std::vector<std::uint16_t> codes(std::size_t(w * h * 3));
            auto code = [&](int plane, int row, int x) -> std::uint16_t& {
                return codes[std::size_t(plane * w * h + row * w + x)];
            };
            for (int row = 0; row < h; ++row)
                for (int x = 0; x < w; ++x) {
                    code(0, row, x) = std::uint16_t(row < 2 ? x : 180 * k);
                    code(1, row, x) = std::uint16_t(row < 2 ? 128 * k : std::clamp(x, 16 * k, 240 * k));
                    code(2, row, x) = std::uint16_t(row < 2 ? 128 * k : 200 * k);
                }
            const auto stem = std::to_string(bits) + (rec2020 ? "-2020" : "-709");
            fs::path raw;
            if (bits == 8) {
                std::vector<std::uint8_t> bytes(codes.size());
                std::transform(codes.begin(), codes.end(), bytes.begin(),
                               [](std::uint16_t value) { return static_cast<std::uint8_t>(value); });
                raw = write_bytes(stem + ".yuv", std::as_bytes(std::span(bytes)));
            } else raw = write_bytes(stem + ".yuv", std::as_bytes(std::span(codes)));
            const auto movie = dir / (stem + ".mkv");
            const auto encoded = run_process({ffmpeg->string(), "-v", "error", "-nostdin", "-f", "rawvideo",
                "-pixel_format", fmt, "-video_size", std::to_string(w) + "x4", "-framerate", "1",
                "-color_range", "tv", "-colorspace", rec2020 ? "bt2020nc" : "bt709", "-color_trc", "bt709",
                "-color_primaries", rec2020 ? "bt2020" : "bt709", "-i", raw.string(),
                "-frames:v", "1", "-c:v", "ffv1", "-level", "3", "-pix_fmt", fmt, "-color_range", "tv",
                "-colorspace", rec2020 ? "bt2020nc" : "bt709", "-color_trc", "bt709", "-color_primaries",
                rec2020 ? "bt2020" : "bt709", movie.string()});
            ASSERT_TRUE(encoded);
            ASSERT_EQ(encoded->exit_code, 0) << encoded->err;
            auto source = open_video_source(movie, VideoArgs{});
            ASSERT_TRUE(source) << source.error().message;
            EXPECT_EQ(source->stream.pix_fmt, fmt);
            auto decoder = FloatVideoDecoder::open(movie, *source, dir / (stem + ".log"));
            ASSERT_TRUE(decoder) << decoder.error().message;
            auto frame = (*decoder)->next();
            ASSERT_TRUE(frame) << frame.error().message;
            ASSERT_TRUE((*decoder)->finish());
            double max_error = 0;
            const double kr = rec2020 ? .2627 : .2126, kb = rec2020 ? .0593 : .0722;
            for (int row = 0; row < h; ++row)
                for (int x = 0; x < w; ++x) {
                    const double y = (double(code(0, row, x)) - 16 * k) / (219 * k);
                    const double cb = (double(code(1, row, x)) - 128 * k) / (224 * k);
                    const double cr = (double(code(2, row, x)) - 128 * k) / (224 * k);
                    const double r = y + 2 * (1 - kr) * cr, b = y + 2 * (1 - kb) * cb;
                    const double g = (y - kr * r - kb * b) / (1 - kr - kb);
                    const double expected[] = {r, g, b};
                    for (int channel = 0; channel < 3; ++channel)
                        max_error = std::max(max_error, std::abs(frame->encoded_rgb.at(channel, row, x) - expected[channel]));
                }
            EXPECT_LT(max_error, 2e-6);
            // Decoder -> versioned baseline: the preserved signal must remain
            // distinct after reconstruction, not only inside a raw buffer.
            MasterRequest q;
            q.analytic = true;
            q.clip_fill = false;
            q.source_curve = "bt1886";
            q.source_space = source->contract.primaries;
            auto baseline = reconstruct_headroom_baseline(*frame, q, HeadroomPolicy::Bt1886V1);
            ASSERT_TRUE(baseline) << baseline.error().message;
            const auto& hdr = baseline->linear2020.buffer();
            for (int x = 235 * k + 1; x < 255 * k; ++x) {
                EXPECT_GT(hdr.at(0, 0, x), hdr.at(0, 0, x - 1));
                const double expected_nits = 203 * std::pow(double(x - 16 * k) / (219 * k), 2.4);
                EXPECT_NEAR(hdr.at(0, 0, x) * 203.0, expected_nits, 3e-4);
            }
            EXPECT_NEAR(hdr.at(0, 0, 235 * k) * 203.0, 203.0, 1e-4);
            RecordProperty("last_headroom_nits_" + stem, hdr.at(0, 0, 255 * k - 1) * 203.0);
            // Source -> shared artist grade -> real half EXR roundtrip and PQ
            // movie spool. +1 EV must survive; neither boundary may clamp at
            // nominal SDR white or apply the 10000/203 scaling twice.
            q.anchor = q.carry_chroma = q.settle_grain = false;
            q.exposure_ev = 1;
            auto master = render_headroom_frame(*frame, q, HeadroomPolicy::Bt1886V1);
            ASSERT_TRUE(master) << master.error().message;
            const auto& linear = master->linear2020.buffer();
            auto spool = master_headroom_video_frame(*master, 0, "hdr10", 1000);
            ASSERT_TRUE(spool) << spool.error().message;
            int distinct_spool = 0;
            for (int x = 235 * k + 1; x < 255 * k; ++x) {
                const auto here = std::size_t(x * 3), prev = std::size_t((x - 1) * 3);
                distinct_spool += spool->packed[here] > spool->packed[prev];
                EXPECT_GT(spool->packed[here], spool->packed[prev]);
                const double want = 406 * std::pow(double(x - 16 * k) / (219 * k), 2.4);
                EXPECT_NEAR(linear.at(0, 0, x) * 203.0, want, 6e-4);
            }
            EXPECT_EQ(distinct_spool, 20 * k - 1);
            RecordProperty("graded_pq_spool_levels_" + stem, distinct_spool);
            for (bool aces : {false, true}) {
                const auto out = dir / (stem + (aces ? "-graded-aces.exr" : "-graded-linear.exr"));
                auto written = aces ? write_aces_exr(out, linear, Primaries::Rec2020)
                                    : write_exr(out, linear, true, kRec2020Chromaticities);
                ASSERT_TRUE(written) << written.error().message;
                auto back = read_exr(out);
                ASSERT_TRUE(back) << back.error().message;
                for (int x = 235 * k + 1; x < 255 * k; ++x) {
                    EXPECT_GT(back->pixels.at(0, 0, x), back->pixels.at(0, 0, x - 1));
                    // Neutral Rec.2020 and AP0 agree, within half precision.
                    const double want = 406 * std::pow(double(x - 16 * k) / (219 * k), 2.4);
                    for (int channel = 0; channel < 3; ++channel)
                        EXPECT_NEAR(back->pixels.at(channel, 0, x) * 203.0, want, .21);
                }
            }
            EXPECT_LT(frame->encoded_rgb.at(0, 0, 0), 0);
            for (int x = 235 * k + 1; x < 255 * k; ++x)
                EXPECT_GT(frame->encoded_rgb.at(0, 0, x), frame->encoded_rgb.at(0, 0, x - 1));
            // Same source through the current delivery decoder: the control
            // documents the measured improvement without changing that path.
            auto legacy = VideoDecoder::open(movie, *source, dir / (stem + "-legacy.log"));
            ASSERT_TRUE(legacy);
            auto legacy_frame = (*legacy)->next();
            ASSERT_TRUE(legacy_frame);
            EXPECT_TRUE((*legacy)->finish());
            for (int x = 235 * k + 1; x < 255 * k; ++x)
                EXPECT_EQ(legacy_frame->data[std::size_t(x * 3)], 65535);
        }
    }
}

TEST_F(FloatDecode, ExactFrameRangesRejectInvalidBoundsBeforeLaunching) {
    auto source = valid_source(); source.clock.frames = 10;
    for (auto range : {FloatFrameRange{-1, 1}, FloatFrameRange{0, 0}, FloatFrameRange{0, -1},
                       FloatFrameRange{10, 1}, FloatFrameRange{9, 2},
                       FloatFrameRange{1, std::numeric_limits<int>::max()}}) {
        EXPECT_FALSE(float_decoder_command("ffmpeg", "missing.mkv", source, range));
        EXPECT_FALSE(FloatVideoDecoder::open("missing.mkv", source, dir / "refused.log", range));
        EXPECT_FALSE(fs::exists(dir / "refused.log"));
    }
    const auto full = float_decoder_command("ffmpeg", "source.mkv", source);
    ASSERT_TRUE(full);
    EXPECT_EQ(std::find(full->begin(), full->end(), "-frames:v"), full->end());
    const auto single = float_decoder_command("ffmpeg", "source.mkv", source, FloatFrameRange{7, 1});
    ASSERT_TRUE(single);
    EXPECT_EQ(std::find(single->begin(), single->end(), "-ss"), single->end());
    const auto filter = std::find(single->begin(), single->end(), "-vf");
    ASSERT_NE(filter, single->end());
    EXPECT_EQ(std::next(filter)->find("trim=start_frame=7:end_frame=8,zscale="), 0u);
}

TEST_F(FloatDecode, Ffv1MemoryProfileScopesInputOutputAndPreservesOtherCodecs) {
    auto source = valid_source(); source.clock.frames = 31; source.clock.fps = "25/1";
    source.streams_json = R"({"format":{"format_name":"matroska,webm","start_time":"0"}})";
    for (const std::string codec : {"ffv1", "h264", "hevc", ""}) {
        source.stream.codec_name = codec;
        for (auto command : {float_decoder_command("ffmpeg", "source.mkv", source),
                             float_decoder_command("ffmpeg", "source.mkv", source, FloatFrameRange{12, 3}),
                             float_playback_decoder_command("ffmpeg", "source.mkv", source, {12, 3})}) {
            ASSERT_TRUE(command);
            const auto input = std::find(command->begin(), command->end(), "-i");
            const auto threads = std::find(command->begin(), command->end(), "-threads");
            ASSERT_NE(threads, command->end()); EXPECT_LT(threads, input);
            EXPECT_EQ(*(threads + 1), codec == "ffv1" ? "2" : "4");
            for (const std::string flag : {"-threads:v", "-thread_queue_size"}) {
                const auto option = std::find(command->begin(), command->end(), flag);
                if (codec == "ffv1") {
                    ASSERT_NE(option, command->end()); EXPECT_GT(option, input);
                    EXPECT_EQ(*(option + 1), "1");
                } else EXPECT_EQ(option, command->end());
            }
            EXPECT_EQ(command->back(), "pipe:1");
        }
    }
}

TEST_F(FloatDecode, PausedAndPlaybackRangesMatchFullDecodeAtFractionalRateAndOffset) {
    auto ffmpeg = find_executable("ffmpeg");
    if (!ffmpeg || !find_executable("ffprobe")) GTEST_SKIP() << "FFmpeg and ffprobe required";
    constexpr int w = 16, h = 4, frames = 7;
    std::vector<std::uint16_t> bytes;
    for (int f = 0; f < frames; ++f)
        for (int c = 0; c < 3; ++c)
            for (int i = 0; i < w * h; ++i)
                bytes.push_back(std::uint16_t(c == 0 ? 930 + f * 10 + i % w : 512));
    const auto raw = write_bytes("range.yuv", std::as_bytes(std::span(bytes)));
    const auto movie = dir / "range.mkv";
    auto made = run_process({ffmpeg->string(), "-v", "error", "-nostdin", "-f", "rawvideo",
        "-pixel_format", "yuv444p10le", "-video_size", "16x4", "-framerate", "24000/1001",
        "-color_range", "tv", "-color_trc", "bt709", "-color_primaries", "bt709", "-colorspace", "bt709",
        "-i", raw.string(), "-frames:v", "7", "-c:v", "ffv1", "-pix_fmt", "yuv444p10le",
        "-color_range", "tv", "-color_trc", "bt709", "-color_primaries", "bt709", "-colorspace", "bt709",
        "-output_ts_offset", "5", movie.string()});
    ASSERT_TRUE(made); ASSERT_EQ(made->exit_code, 0) << made->err;
    auto source = open_video_source(movie, VideoArgs{});
    ASSERT_TRUE(source) << source.error().message;
    EXPECT_EQ(source->clock.frames, frames);
    EXPECT_EQ(source->clock.fps, "24000/1001");
    EXPECT_NEAR(source->clock.start, 5, .001);
    std::vector<ExtendedVideoFrame> reference;
    auto full = FloatVideoDecoder::open(movie, *source, dir / "full.log");
    ASSERT_TRUE(full);
    for (int i = 0; i < frames; ++i) {
        auto f = (*full)->next(); ASSERT_TRUE(f);
        reference.push_back(std::move(*f));
    }
    ASSERT_TRUE((*full)->finish());
    // Random paused reads, playback from the middle, and full export all
    // select exactly the same decoded frames, despite nonzero PTS/fractional fps.
    for (auto range : {FloatFrameRange{6, 1}, FloatFrameRange{0, 1}, FloatFrameRange{3, 1},
                       FloatFrameRange{2, 5}, FloatFrameRange{0, 7}}) {
        auto decoder = FloatVideoDecoder::open(movie, *source, dir / "range.log", range);
        ASSERT_TRUE(decoder);
        EXPECT_EQ((*decoder)->frames(), range.count);
        for (int i = 0; i < range.count; ++i) {
            auto frame = (*decoder)->next(); ASSERT_TRUE(frame);
            EXPECT_EQ(frame->encoded_rgb.vector(), reference[std::size_t(range.first + i)].encoded_rgb.vector());
            EXPECT_EQ(frame->source_contract, source->contract);
        }
        EXPECT_EQ((*decoder)->next_index(), range.count);
        ASSERT_TRUE((*decoder)->finish());
    }
}

TEST_F(FloatDecode, CacheIdentityIncludesContentPolicyDecoderAndDecodeContract) {
    auto source = valid_source();
    source.clock = {"24", 1, 0, 1.0 / 24, 1e-6};
    const auto input = dir / "input.mkv", decoder = dir / "decoder.exe";
    std::ofstream(input, std::ios::binary) << "source-one";
    std::ofstream(decoder, std::ios::binary) << "decoder-one";
    constexpr auto policy = HeadroomPolicy::Bt1886V1;
    auto base = headroom_source_cache_id(input, source, policy, decoder);
    ASSERT_TRUE(base) << base.error().message;
    EXPECT_EQ(base->find("headroom-float-v1-"), 0u);
    EXPECT_EQ(base->size(), std::string("headroom-float-v1-").size() + 64);
    EXPECT_EQ(*headroom_source_cache_id(dir / "." / "input.mkv", source, policy, decoder), *base);
    // Same name, byte count AND modification time: content still separates it.
    const auto stamp = fs::last_write_time(input);
    std::ofstream(input, std::ios::binary) << "source-two";
    fs::last_write_time(input, stamp);
    auto changed = headroom_source_cache_id(input, source, policy, decoder);
    ASSERT_TRUE(changed); EXPECT_NE(*changed, *base);
    std::ofstream(input, std::ios::binary) << "source-one";
    std::ofstream(decoder, std::ios::binary) << "decoder-two";
    changed = headroom_source_cache_id(input, source, policy, decoder);
    ASSERT_TRUE(changed); EXPECT_NE(*changed, *base);
    std::ofstream(decoder, std::ios::binary) << "decoder-one";
    auto changed_source = source; changed_source.stream.width = 8;
    changed = headroom_source_cache_id(input, changed_source, policy, decoder);
    ASSERT_TRUE(changed); EXPECT_NE(*changed, *base);
    changed_source = source; changed_source.clock.frames = 2;
    changed = headroom_source_cache_id(input, changed_source, policy, decoder);
    ASSERT_TRUE(changed); EXPECT_NE(*changed, *base);
    changed_source = source; changed_source.stream.codec_name = "ffv1";
    changed = headroom_source_cache_id(input, changed_source, policy, decoder);
    ASSERT_TRUE(changed); EXPECT_NE(*changed, *base);
    changed_source = source;
    changed_source.contract.matrix = "bt2020nc"; changed_source.stream.color_space = "bt2020nc";
    changed = headroom_source_cache_id(input, changed_source, policy, decoder);
    ASSERT_TRUE(changed); EXPECT_NE(*changed, *base);
    EXPECT_FALSE(headroom_source_cache_id(input, source, HeadroomPolicy::Disabled, decoder));
    EXPECT_FALSE(headroom_source_cache_id(input, source, static_cast<HeadroomPolicy>(255), decoder));
    EXPECT_FALSE(headroom_source_cache_id(dir / "missing", source, policy, decoder));
    EXPECT_FALSE(headroom_source_cache_id(input, source, policy, dir / "missing"));
    changed_source = source; changed_source.contract.range = "full";
    EXPECT_FALSE(headroom_source_cache_id(input, changed_source, policy, decoder));
    changed_source = source; changed_source.clock.start = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(headroom_source_cache_id(input, changed_source, policy, decoder));
    changed_source = source; changed_source.clock.fps = "0/1";
    EXPECT_FALSE(headroom_source_cache_id(input, changed_source, policy, decoder));
    changed_source = source; changed_source.stream.codec_name = "ffv1";
    changed_source.clock = {"24", 10, 5, 10.0 / 24, 1e-6};
    changed_source.streams_json = R"({"format":{"format_name":"matroska,webm","start_time":"5"}})";
    auto video_start_key = headroom_source_cache_id(input, changed_source, policy, decoder);
    ASSERT_TRUE(video_start_key);
    changed_source.streams_json = R"({"format":{"format_name":"matroska,webm","start_time":"0"}})";
    auto audio_start_key = headroom_source_cache_id(input, changed_source, policy, decoder);
    ASSERT_TRUE(audio_start_key); EXPECT_NE(*video_start_key, *audio_start_key);
}

TEST(FloatPlaybackCommand, OnlyValidatedUnambiguousFfv1MatroskaUsesTimestampSeek) {
    auto s = valid_source(); s.stream.codec_name = "ffv1";
    s.clock = {"24000/1001", 31, 5, 31 * 1001.0 / 24000, .0015};
    s.streams_json = R"({"format":{"format_name":"matroska,webm","start_time":"5.000000"}})";
    auto command = float_playback_decoder_command("ffmpeg", "clip.MKV", s, {12, 3});
    ASSERT_TRUE(command);
    auto at = std::find(command->begin(), command->end(), "-ss"); ASSERT_NE(at, command->end());
    EXPECT_NEAR(std::stod(*(at + 1)), 11.5 * 1001 / 24000, 1e-8);
    EXPECT_NE(std::find(command->begin(), command->end(), "-copyts"), command->end());
    at = std::find(command->begin(), command->end(), "-vf"); ASSERT_NE(at, command->end());
    EXPECT_TRUE((at + 1)->starts_with("trim=start="));
    auto indexed = [&](VideoSource source, const std::string& file = "clip.mkv", FloatFrameRange range = {12, 3}) {
        auto expected = float_decoder_command("ffmpeg", file, source, range);
        auto actual = float_playback_decoder_command("ffmpeg", file, source, range);
        ASSERT_TRUE(expected); ASSERT_TRUE(actual); EXPECT_EQ(*actual, *expected);
    };
    indexed(s, "clip.mov"); indexed(s, "clip.mkv", {0, 3});
    auto changed = s; changed.stream.codec_name = "h264"; indexed(changed);
    changed = s; changed.clock.fps = ""; indexed(changed);
    changed = s; changed.clock.fps = "0/1"; indexed(changed);
    changed = s; changed.clock.start = -1; indexed(changed);
    changed = s; changed.clock.start = 1e8; indexed(changed);
    changed = s; changed.clock.start = std::numeric_limits<double>::infinity(); indexed(changed);
    changed = s; changed.clock.tolerance = .02; indexed(changed);
    changed = s; changed.clock.tolerance = -1; indexed(changed);
    changed = s; changed.clock.tolerance = std::numeric_limits<double>::quiet_NaN(); indexed(changed);
    changed = s; changed.streams_json = "{}"; indexed(changed);
    changed = s; changed.streams_json = R"({"format":{"format_name":"mov","start_time":"5"}})"; indexed(changed);
    changed = s; changed.streams_json = R"({"format":{"format_name":"matroska,webm","start_time":"NaN"}})"; indexed(changed);
    EXPECT_FALSE(float_playback_decoder_command("ffmpeg", "clip.mkv", s, {30, 2}));
}

TEST_F(FloatDecode, PlaybackSeekMatchesEverySampleAcrossRatesOffsetsAndKeyframeBoundaries) {
    const auto ffmpeg = find_executable("ffmpeg");
    if (!ffmpeg || !find_executable("ffprobe")) GTEST_SKIP() << "FFmpeg and ffprobe required";
    constexpr int w = 32, h = 4, frames = 31;
    int fixture = 0;
    for (int bits : {8, 10}) for (const std::string rate : {"25", "24000/1001", "30000/1001"})
        for (const std::string offset : {"0", "5", "123.456"}) {
            SCOPED_TRACE(std::to_string(bits) + " " + rate + " " + offset);
            const int k = 1 << (bits - 8);
            const auto stem = "seek-" + std::to_string(fixture++);
            const auto fmt = bits == 8 ? "yuv444p" : "yuv444p10le";
            std::vector<std::uint16_t> codes(std::size_t(frames * w * h * 3), std::uint16_t(128 * k));
            for (int f = 0; f < frames; ++f) for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
                codes[std::size_t(f * w * h * 3 + y * w + x)] = std::uint16_t((210 + (x + f * 7) % 44) * k);
                codes[std::size_t(f * w * h * 3 + w * h + y * w + x)] = std::uint16_t((112 + f) * k);
            }
            fs::path raw;
            if (bits == 8) {
                std::vector<std::uint8_t> bytes(codes.size());
                std::transform(codes.begin(), codes.end(), bytes.begin(), [](std::uint16_t v) { return static_cast<std::uint8_t>(v); });
                raw = write_bytes(stem + ".yuv", std::as_bytes(std::span(bytes)));
            } else raw = write_bytes(stem + ".yuv", std::as_bytes(std::span(codes)));
            const auto movie = dir / (stem + ".mkv");
            auto encode = run_process({ffmpeg->string(), "-v", "error", "-nostdin", "-f", "rawvideo",
                "-pixel_format", fmt, "-video_size", "32x4", "-framerate", rate,
                "-color_range", "tv", "-color_trc", "bt709", "-color_primaries", "bt709", "-colorspace", "bt709",
                "-i", raw.string(), "-frames:v", "31", "-c:v", "ffv1", "-level", "3", "-g", "12",
                "-pix_fmt", fmt, "-color_range", "tv", "-color_trc", "bt709", "-color_primaries", "bt709",
                "-colorspace", "bt709", "-output_ts_offset", offset, movie.string()});
            ASSERT_TRUE(encode); ASSERT_EQ(encode->exit_code, 0) << encode->err;
            auto source = open_video_source(movie, VideoArgs{}); ASSERT_TRUE(source);
            auto decoder = FloatVideoDecoder::open(movie, *source, dir / "reference.log"); ASSERT_TRUE(decoder);
            std::vector<ExtendedVideoFrame> reference;
            for (int i = 0; i < frames; ++i) {
                auto frame = (*decoder)->next(); ASSERT_TRUE(frame); reference.push_back(std::move(*frame));
            }
            ASSERT_TRUE((*decoder)->finish());
            for (auto range : {FloatFrameRange{1, 1}, {3, 2}, {11, 3}, {12, 1}, {17, 4}, {30, 1}, {1, 30}}) {
                auto argv = float_playback_decoder_command(ffmpeg->string(), movie, *source, range); ASSERT_TRUE(argv);
                ASSERT_NE(std::find(argv->begin(), argv->end(), "-ss"), argv->end());
                auto selected = FloatVideoDecoder::open_command(*argv, w, h, range.count, source->contract, dir / "seek.log");
                ASSERT_TRUE(selected);
                for (int i = 0; i < range.count; ++i) {
                    auto actual = (*selected)->next(); ASSERT_TRUE(actual) << actual.error().message;
                    EXPECT_EQ(actual->encoded_rgb.vector(), reference[std::size_t(range.first + i)].encoded_rgb.vector());
                }
                EXPECT_TRUE((*selected)->finish());
            }
            if (bits == 10 && rate == "24000/1001" && offset == "5") {
                const auto audio_movie = dir / "video-after-audio.mkv";
                auto mux = run_process({ffmpeg->string(), "-v", "error", "-nostdin", "-copyts", "-i", movie.string(),
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "7", "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "pcm_s16le", audio_movie.string()});
                ASSERT_TRUE(mux); ASSERT_EQ(mux->exit_code, 0) << mux->err;
                auto mixed = open_video_source(audio_movie, VideoArgs{}); ASSERT_TRUE(mixed);
                EXPECT_EQ(mixed->clock.start, 5);
                auto argv = float_playback_decoder_command(ffmpeg->string(), audio_movie, *mixed, {12, 3}); ASSERT_TRUE(argv);
                auto seek = std::find(argv->begin(), argv->end(), "-ss"); ASSERT_NE(seek, argv->end());
                EXPECT_NEAR(std::stod(*(seek + 1)), 5 + 11.5 * 1001 / 24000, 1e-8);
                auto selected = FloatVideoDecoder::open_command(*argv, w, h, 3, mixed->contract, dir / "audio-seek.log");
                ASSERT_TRUE(selected);
                for (int i = 0; i < 3; ++i) {
                    auto actual = (*selected)->next(); ASSERT_TRUE(actual);
                    EXPECT_EQ(actual->encoded_rgb.vector(), reference[std::size_t(12 + i)].encoded_rgb.vector());
                }
                EXPECT_TRUE((*selected)->finish());
            }
        }
}
