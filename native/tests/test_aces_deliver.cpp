// What an ACES v2 master writes (cli/aces_v2_deliver.hpp, cli/aces_v2_log.hpp,
// 27 Oct 2026): the delivery options, the EXR's depth and gamut, the encoded
// copies (DPX, ProRes, the uncompressed 16-bit movie) and the frames read from
// one video. No model, no OCIO: a pointwise stand-in for the HDR output.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "aces_v2_deliver.hpp"
#include "aces_v2_log.hpp"
#include "aces_v2_monitor.hpp"
#include "rudra/core/hdr10.hpp"
#include "rudra/core/gamut.hpp"
#include "rudra/deliver/exr.hpp"
#include "rudra/deliver/sequence_encode.hpp"

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

namespace fs = std::filesystem;
using namespace rudra;
using nlohmann::json;

namespace {

fs::path tmp(const std::string& name) {
    static const std::string run = std::to_string(std::random_device{}());
    const fs::path d = fs::temp_directory_path() / ("rudra_aces_deliver_" + run) / name;
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}

bool have(const char* tool) {
    const std::string cmd = std::string(tool) + " -version > " +
#ifdef _WIN32
                            "NUL 2>&1";
#else
                            "/dev/null 2>&1";
#endif
    return std::system(cmd.c_str()) == 0;
}

std::string run(const std::string& cmd) {
    std::string out;
    if (FILE* p = popen(cmd.c_str(), "r")) {
        char buf[4096];
        std::size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, p)) > 0) out.append(buf, n);
        pclose(p);
    }
    return out;
}

std::string probe(const fs::path& f, const std::string& entries) {
    return run("ffprobe -v error -select_streams v:0 -show_entries stream=" + entries + " -of default=nw=1 \"" + f.string() + "\"");
}

// A scene-linear ramp in ACEScg with highlights well above diffuse white.
PlanarBuffer ramp(int h, int w) {
    PlanarBuffer p(3, h, w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const std::size_t i = std::size_t(y) * std::size_t(w) + std::size_t(x);
            const float t = float(x + y * w) / float(w * h - 1);
            p.plane(0)[i] = 0.001f + 40.0f * t * t;
            p.plane(1)[i] = 0.18f + 2.0f * t;
            p.plane(2)[i] = 0.05f * (1.0f - t);
        }
    return p;
}

// Stand-in for the ACES 2.0 HDR output: display = x / (1 + x) on clamped input.
class FakeHdr final : public aces_v2::AcesView {
public:
    void forward(float* rgb, std::size_t n) const override {
        for (std::size_t i = 0; i < n * 3; ++i) {
            const float x = std::max(rgb[i], 0.0f);
            rgb[i] = x / (1.0f + x);
        }
    }
    void inverse(float*, std::size_t) const override {}
    std::string identity() const override { return "fake hdr"; }
};

}  // namespace

TEST(AcesDeliver, OptionsDefaultToTheFloatAcescgMaster) {
    auto o = aces_deliver_from_json(json::object());
    ASSERT_TRUE(o);
    EXPECT_EQ(o->exr, "float");
    EXPECT_EQ(o->exr_gamut, "acescg");
    EXPECT_FALSE(o->copies.any());
    EXPECT_EQ(aces_exr_label(*o), "ACEScg scene-linear, signed, unclamped, float32; the reconstruction mask as the rudra.reconstructed channel");
    auto h = aces_deliver_from_json({{"exr", "half"}, {"exr_gamut", "ap0"}});
    ASSERT_TRUE(h);
    EXPECT_EQ(aces_exr_label(*h), "ACES 2065-1 (AP0), SMPTE ST 2065-4 container: half float RGB, uncompressed, "
                                  "acesImageContainerFlag 1; scene-linear, signed, unclamped; the reconstruction mask in mask/<frame>.exr");
}

TEST(AcesDeliver, OptionsRefuseWhatCannotBeWritten) {
    EXPECT_FALSE(aces_deliver_from_json({{"exr", "int8"}}));
    EXPECT_FALSE(aces_deliver_from_json({{"exr_gamut", "rec709"}}));
    EXPECT_FALSE(aces_deliver_from_json({{"exr", "none"}}));   // nothing at all
    EXPECT_FALSE(aces_deliver_from_json({{"name", "../escape"}}));
    EXPECT_FALSE(aces_deliver_from_json({{"name", "a/b"}}));
    EXPECT_FALSE(aces_deliver_from_json({{"frames", {-1}}}));
    EXPECT_FALSE(aces_deliver_from_json({{"log", "slog3"}, {"log_formats", "dpx10"}}));
    EXPECT_FALSE(aces_deliver_from_json({{"log", "logc4"}, {"log_formats", "prores422hq,mov16"}}));   // one movie
    EXPECT_FALSE(aces_deliver_from_json({{"log", "logc4"}, {"log_formats", "h264"}}));
    auto movie_only = aces_deliver_from_json({{"exr", "none"}, {"log", "pq1000"}, {"log_formats", "mov16"}});
    ASSERT_TRUE(movie_only);
    EXPECT_TRUE(movie_only->copies.pq);
    EXPECT_TRUE(movie_only->copies.mov16);
    EXPECT_EQ(movie_only->copies.name(), "pq1000");
    EXPECT_EQ(aces_exr_label(*movie_only), "none (encoded copies only)");
    auto p = parse_aces_log("acescct", "prores4444,dpx10");
    ASSERT_TRUE(p);
    EXPECT_EQ(p->prores, "4444");
    EXPECT_TRUE(p->dpx10);
    EXPECT_FALSE(parse_aces_log("logc4", "prores4444p10"));
}

TEST(AcesDeliver, NamesFramesAndTheMovie) {
    auto o = aces_deliver_from_json({{"name", "SH010_rudra"}, {"start", 1001}, {"log", "logc4"}, {"log_formats", "mov16"}});
    ASSERT_TRUE(o);
    EXPECT_EQ(aces_frame_stem(*o, 0, "clip_000000.exr"), "SH010_rudra.001001");
    EXPECT_EQ(aces_frame_stem(*o, 23, "clip_000023.exr"), "SH010_rudra.001024");
    EXPECT_EQ(aces_movie_file(*o, "clip"), "SH010_rudra.mov");
    o->name.clear();
    EXPECT_EQ(aces_frame_stem(*o, 3, "clip_000003.exr"), "clip_000003");
    EXPECT_EQ(aces_movie_file(*o, "clip"), "clip_logc4.mov");
}

TEST(AcesDeliver, HalfAndAp0ExrsKeepTheMask) {
    const auto d = tmp("exr");
    const PlanarBuffer px = ramp(4, 6);
    PlanarBuffer mask(1, 4, 6, 0.0f);
    mask.plane(0)[5] = 1.0f;
    auto half = aces_deliver_from_json({{"exr", "half"}, {"exr_gamut", "ap0"}});
    ASSERT_TRUE(half);
    ASSERT_TRUE(write_aces_master_exr(d / "h.exr", px, mask, *half, {}));
    auto r = read_exr(d / "h.exr");
    ASSERT_TRUE(r) << r.error().message;
    for (int t : r->pixel_types) EXPECT_EQ(t, 1);   // half
    // ST 2065-4 (review 28 Oct): RGB only, acesImageContainerFlag an int 1; the mask beside it.
    EXPECT_EQ(r->pixel_types.size(), 3u);
    EXPECT_FALSE(r->reconstruction_mask);
    bool flag = false;
    for (std::size_t i = 0; i < r->attribute_types.size(); ++i)
        if (r->attribute_types[i].first == "acesImageContainerFlag") {
            EXPECT_EQ(r->attribute_types[i].second, "int");
            for (const auto& [k, v] : r->attributes)
                if (k == "acesImageContainerFlag") flag = v == std::vector<std::uint8_t>{1, 0, 0, 0};
        }
    EXPECT_TRUE(flag);
    auto mk = read_exr(d / "mask" / "h.exr");
    ASSERT_TRUE(mk) << mk.error().message;
    EXPECT_EQ(mk->pixels.plane(0)[5], 1.0f);
    EXPECT_EQ(mk->pixels.plane(0)[4], 0.0f);
    EXPECT_FALSE(aces_deliver_from_json({{"exr", "float"}, {"exr_gamut", "ap0"}}));   // ST 2065-4 is half
    // AP0 is the ACEScg pixels through the AP1 -> AP0 matrix.
    const Mat3 m = rgb_to_rgb_matrix(Primaries::Ap1, Primaries::Ap0);
    for (std::size_t i : {std::size_t(0), std::size_t(11), std::size_t(23)}) {
        const double want = m[0][0] * px.plane(0)[i] + m[0][1] * px.plane(1)[i] + m[0][2] * px.plane(2)[i];
        EXPECT_NEAR(r->pixels.plane(0)[i], want, std::abs(want) * 2e-3 + 1e-4) << i;
    }
    auto full = aces_deliver_from_json(json::object());
    ASSERT_TRUE(write_aces_master_exr(d / "f.exr", px, mask, *full, {}));
    auto f = read_exr(d / "f.exr");
    ASSERT_TRUE(f);
    EXPECT_EQ(f->pixel_types.front(), 2);   // float
    EXPECT_EQ(f->pixels.plane(0)[23], px.plane(0)[23]);
}

TEST(AcesDeliver, SixteenBitMovieHoldsEveryCodeExactly) {
    if (!have("ffmpeg") || !have("ffprobe")) GTEST_SKIP() << "ffmpeg and ffprobe are not on the PATH";
    const auto d = tmp("mov16");
    auto o = parse_aces_log("logc4", "mov16");
    ASSERT_TRUE(o);
    auto w = AcesLogWriter::open(*o, d, d / "shot.mov", 24000.0 / 1001.0);
    ASSERT_TRUE(w) << w.error().message;
    const PlanarBuffer a = ramp(6, 9), b = ramp(6, 9);   // odd width: fine uncompressed
    ASSERT_TRUE((*w)->write(a, "f1.exr"));
    ASSERT_TRUE((*w)->write(b, "f2.exr"));
    ASSERT_TRUE((*w)->finish());
    const auto info = probe(d / "shot.mov", "codec_name,codec_tag_string,pix_fmt,r_frame_rate,color_space");
    EXPECT_NE(info.find("codec_tag_string=b48r"), std::string::npos) << info;
    EXPECT_NE(info.find("pix_fmt=rgb48be"), std::string::npos) << info;
    EXPECT_NE(info.find("r_frame_rate=24000/1001"), std::string::npos) << info;
    // Decoded back to 16-bit RGB: exactly the codes the writer computed.
    const fs::path raw = d / "back.rgb";
    run("ffmpeg -v error -y -i \"" + (d / "shot.mov").string() + "\" -f rawvideo -pix_fmt rgb48le \"" + raw.string() + "\"");
    std::ifstream in(raw, std::ios::binary);
    std::vector<std::uint16_t> back(6 * 9 * 3 * 2);
    in.read(reinterpret_cast<char*>(back.data()), std::streamsize(back.size() * 2));
    ASSERT_TRUE(in);
    const PlanarBuffer want = aces_log_codes(a, LogCurve::LogC4);
    for (std::size_t i = 0; i < want.plane_size(); ++i)
        for (int c = 0; c < 3; ++c)
            ASSERT_EQ(back[i * 3 + std::size_t(c)], std::uint16_t(std::lround(want.plane(c)[i] * 65535.0f))) << i;
    const auto prov = (*w)->provenance();
    EXPECT_EQ(prov["files"][0]["bits"], 16);
    EXPECT_EQ(prov["encoding"], "logc4");
}

TEST(AcesDeliver, PqCopiesGoThroughTheHdrOutputAndAreTaggedPq) {
    if (!have("ffmpeg") || !have("ffprobe")) GTEST_SKIP() << "ffmpeg and ffprobe are not on the PATH";
    const auto d = tmp("pq");
    auto o = parse_aces_log("pq1000", "dpx10,prores422hq");
    ASSERT_TRUE(o);
    EXPECT_FALSE(AcesLogWriter::open(*o, d, d / "pq.mov", 25.0));   // no HDR output loaded: refused
    FakeHdr hdr;
    auto w = AcesLogWriter::open(*o, d, d / "pq.mov", 25.0, &hdr);
    ASSERT_TRUE(w) << w.error().message;
    const PlanarBuffer a = ramp(8, 16);
    const PlanarBuffer codes = (*w)->codes(a);
    EXPECT_NEAR(codes.plane(1)[0], 0.18f / 1.18f, 1e-6);
    ASSERT_TRUE((*w)->write(a, "SH010.001001.exr"));
    ASSERT_TRUE((*w)->finish());
    EXPECT_TRUE(fs::exists(d / "pq1000_dpx" / "SH010.001001.dpx"));
    const auto info = probe(d / "pq.mov", "codec_name,pix_fmt,color_transfer,color_primaries,color_space");
    EXPECT_NE(info.find("codec_name=prores"), std::string::npos) << info;
    EXPECT_NE(info.find("pix_fmt=yuv422p10le"), std::string::npos) << info;
    // The ProRes frame headers (what Resolve and FCP read) say PQ, BT.2020; the
    // container's colr atom is the ffmpeg build's (some leave it unknown).
    const auto frame = prores_frame_tags(d / "pq.mov");
    ASSERT_TRUE(frame) << info;
    EXPECT_EQ(tag_of(*frame, "color_primaries"), "bt2020") << info;
    EXPECT_EQ(tag_of(*frame, "color_transfer"), "smpte2084") << info;
    EXPECT_EQ(tag_of(*frame, "color_space"), "bt2020nc") << info;
    const auto prov = (*w)->provenance();
    EXPECT_EQ(prov["encoding"], "pq1000");
    EXPECT_EQ(prov["output_transform"], "fake hdr");
    EXPECT_EQ(prov["display_referred"], true);
}

TEST(AcesDeliver, MovieArgumentsMatchTheBitDepth) {
    auto args = [](const char* curve, const char* fmt) {
        auto o = parse_aces_log(curve, fmt);
        std::string s;
        for (const auto& a : AcesLogWriter::movie_arguments(*o)) s += a + " ";
        return s;
    };
    EXPECT_NE(args("logc4", "prores422hq").find("-pix_fmt yuv422p10le"), std::string::npos);
    EXPECT_NE(args("logc4", "prores4444").find("-pix_fmt yuv444p12le"), std::string::npos);
    EXPECT_NE(args("logc4", "prores4444xq").find("-profile:v 5"), std::string::npos);
    EXPECT_NE(args("acescct", "mov16").find("-c:v rawvideo -pix_fmt rgb48be"), std::string::npos);
    EXPECT_NE(args("pq1000", "prores422hq").find("prores_metadata=color_primaries=bt2020:color_trc=smpte2084"), std::string::npos);
    EXPECT_EQ(args("logc4", "prores422hq").find("prores_metadata"), std::string::npos);
    EXPECT_NE(args("logc4", "prores422hq").find("-color_trc unknown"), std::string::npos);
}

TEST(AcesDeliver, AFeedReadsOnlyTheChosenFramesOfAVideo) {
    if (!have("ffmpeg") || !have("ffprobe")) GTEST_SKIP() << "ffmpeg and ffprobe are not on the PATH";
    const auto d = tmp("feed");
    const fs::path clip = d / "clip.mp4";
    run("ffmpeg -v error -y -f lavfi -i testsrc2=s=64x36:r=24:d=0.25 -pix_fmt yuv420p -c:v libx264 \"" + clip.string() + "\"");
    ASSERT_TRUE(fs::exists(clip));
    auto feed = AcesFrameFeed::open({clip}, {4, 1}, d / "decode.log");
    ASSERT_TRUE(feed) << feed.error().message << " " << feed.error().detail;
    EXPECT_EQ((*feed)->total(), 2u);
    EXPECT_NEAR((*feed)->fps(), 24.0, 1e-9);
    auto a = (*feed)->next();
    ASSERT_TRUE(a);
    EXPECT_EQ(a->exr, aces_video_exr_name(clip, 4));
    EXPECT_EQ(a->codes.width(), 64);
    auto b = (*feed)->next();
    ASSERT_TRUE(b);
    EXPECT_EQ(b->exr, aces_video_exr_name(clip, 1));
    EXPECT_FALSE((*feed)->next());
    EXPECT_FALSE(AcesFrameFeed::open({clip}, {6}, d / "decode.log"));   // 6 frames: 0..5
    auto all = AcesFrameFeed::open({clip}, {}, d / "decode.log");
    ASSERT_TRUE(all);
    EXPECT_EQ((*all)->total(), 6u);
}

// The conform (28 Oct 2026): a movie's start timecode and reel go to every
// frame and into the movie; a typed start timecode wins; trims are checked.
TEST(AcesDeliver, TimecodeAndReelReachTheFramesAndTheMovie) {
    EXPECT_FALSE(aces_deliver_from_json({{"timecode", "1:00"}}));
    EXPECT_FALSE(aces_deliver_from_json({{"trims", {{"strength", 3.0}}}}));
    EXPECT_FALSE(aces_deliver_from_json({{"trims", {{"peak_nits", 50.0}}}}));
    auto o = aces_deliver_from_json({{"trims", {{"strength", 0.8}, {"peak_nits", 1000.0}}}});
    ASSERT_TRUE(o);
    EXPECT_DOUBLE_EQ(o->trims.strength, 0.8);
    EXPECT_FALSE(o->trims.identity());
    EXPECT_EQ(aces_trims_json(o->trims)["text"], "strength 0.80, saturation 1.00, peak 1000 nits");
    // The streams' tags: a tmcd track's timecode and reel; MXF's format tags.
    const auto tc = aces_source_timecode(
        R"({"streams":[{"codec_type":"video","tags":{}},{"codec_type":"data","tags":{"timecode":"10:00:00:00","reel_name":"A001"}}]})");
    EXPECT_EQ(tc.start, "10:00:00:00");
    EXPECT_EQ(tc.reel, "A001");
    EXPECT_EQ(aces_source_timecode(R"({"streams":[],"format":{"tags":{"timecode":"01:00:00;00"}}})").start, "01:00:00;00");
    // The stamp: the movie's own, a typed start (frame k of a movie is start + k), stills from the typed one.
    AcesFrameIn movie_frame;
    movie_frame.frame = 12;
    movie_frame.fps = 25;
    movie_frame.timecode = "10:00:00:12";
    movie_frame.reel = "A001";
    AcesDeliverOptions none;
    auto s = aces_frame_stamp(none, 0, movie_frame, 25);
    EXPECT_EQ(s.timecode, "10:00:00:12");
    EXPECT_EQ(s.reel, "A001");
    AcesDeliverOptions typed;
    typed.timecode = "01:00:00:00";
    typed.reel = "R2";
    s = aces_frame_stamp(typed, 0, movie_frame, 25);
    EXPECT_EQ(s.timecode, "01:00:00:12");
    EXPECT_EQ(s.reel, "R2");
    AcesFrameIn still;
    EXPECT_EQ(aces_frame_stamp(typed, 3, still, 24).timecode, "01:00:00:03");
    EXPECT_EQ(aces_frame_stamp(none, 3, still, 24).timecode, "");
    if (!have("ffmpeg") || !have("ffprobe")) GTEST_SKIP() << "ffmpeg and ffprobe are not on the PATH";
    // A real movie with a timecode: the feed carries it frame by frame.
    const auto d = tmp("timecode");
    const fs::path clip = d / "tc.mov";
    run("ffmpeg -v error -y -f lavfi -i testsrc2=s=64x36:r=25:d=0.4 -pix_fmt yuv420p -c:v libx264 -timecode 10:00:00:00 "
        "-metadata:s:v:0 reel_name=A001C003 \"" + clip.string() + "\"");
    ASSERT_TRUE(fs::exists(clip));
    auto feed = AcesFrameFeed::open({clip}, {4}, d / "decode.log");
    ASSERT_TRUE(feed) << feed.error().message;
    auto f = (*feed)->next();
    ASSERT_TRUE(f);
    EXPECT_EQ(f->timecode, "10:00:00:04");
    EXPECT_EQ(f->frame, 4);
    // The movie written from it carries the start timecode and the reel.
    auto lo = parse_aces_log("logc4", "prores422hq");
    ASSERT_TRUE(lo);
    auto w = AcesLogWriter::open(*lo, d, d / "out.mov", 25);
    ASSERT_TRUE(w);
    (*w)->set_movie_stamp("10:00:00:04", "A001C003");
    ASSERT_TRUE((*w)->write(ramp(36, 64), "f.exr"));
    ASSERT_TRUE((*w)->finish());
    const auto info = run("ffprobe -v error -show_entries stream_tags=timecode,reel_name -of default=nw=1 \"" +
                          (d / "out.mov").string() + "\"");
    EXPECT_NE(info.find("timecode=10:00:00:04"), std::string::npos) << info;
    EXPECT_NE((*w)->provenance()["files"][0]["timecode"].get<std::string>(), "");
}

// A shot's in and out (28 Oct 2026): only frames first..last of a movie, in
// order, with their own frame numbers and timecode; the light level.
TEST(AcesDeliver, ARangeReadsOnlyTheShotsFramesAndTheLightLevelIsMeasured) {
    EXPECT_FALSE(aces_deliver_from_json({{"range", {3}}}));
    EXPECT_FALSE(aces_deliver_from_json({{"range", {5, 2}}}));
    EXPECT_FALSE(aces_deliver_from_json({{"range", {1, 2}}, {"frames", {1}}}));
    auto o = aces_deliver_from_json({{"range", {2, 4}}});
    ASSERT_TRUE(o);
    EXPECT_EQ(o->range_first, 2);
    EXPECT_EQ(o->range_last, 4);
    // Light level: diffuse white is 203 nits; 4.0 is 812.
    PlanarBuffer px(3, 1, 2, 1.0f);
    for (int c = 0; c < 3; ++c) px.plane(c)[1] = 4.0f;
    const auto l = aces_light_level(px);
    EXPECT_NEAR(l.scene_peak, 812.0, 0.5);
    EXPECT_NEAR(l.scene_average, (203.0 + 812.0) / 2, 0.5);
    EXPECT_EQ(l.maxcll, 0.0);   // no HDR output given: no HDR10 numbers
    // HDR10 numbers are the PQ output's display nits (review 28 Oct): a view that
    // writes PQ 0.5 (92 nits) and 0.75 (~983 nits).
    class PqView final : public aces_v2::AcesView {
    public:
        void forward(float* rgb, std::size_t n) const override {
            for (std::size_t i = 0; i < n * 3; ++i) rgb[i] = rgb[i] > 2.0f ? 0.75f : 0.5f;
        }
        void inverse(float*, std::size_t) const override {}
        std::string identity() const override { return "pq"; }
    } pqv;
    const auto d10 = aces_light_level(px, &pqv);
    EXPECT_NEAR(d10.maxcll, pq_eotf(0.75f), 0.01);
    EXPECT_NEAR(d10.maxfall, (pq_eotf(0.5f) + pq_eotf(0.75f)) / 2, 0.01);
    EXPECT_LT(d10.maxcll, 1000.5);
    EXPECT_NEAR(d10.scene_peak, 812.0, 0.5);
    if (!have("ffmpeg") || !have("ffprobe")) GTEST_SKIP() << "ffmpeg and ffprobe are not on the PATH";
    const auto d = tmp("range");
    const fs::path clip = d / "r.mov";
    run("ffmpeg -v error -y -f lavfi -i testsrc2=s=64x36:r=24:d=0.25 -pix_fmt yuv420p -c:v libx264 -timecode 01:00:00:00 \"" +
        clip.string() + "\"");
    auto feed = AcesFrameFeed::open({clip}, {}, d / "decode.log", 2, 4);
    ASSERT_TRUE(feed) << feed.error().message;
    EXPECT_EQ((*feed)->total(), 3u);
    for (int want = 2; want <= 4; ++want) {
        auto f = (*feed)->next();
        ASSERT_TRUE(f) << f.error().message;
        EXPECT_EQ(f->frame, want);
        EXPECT_EQ(f->timecode, "01:00:00:0" + std::to_string(want));
        // The same picture as the frame read on its own.
        auto one = aces_read_one(clip, want, d / "one.log");
        ASSERT_TRUE(one);
        EXPECT_EQ(one->codes.span()[100], f->codes.span()[100]);
    }
    EXPECT_FALSE((*feed)->next());
    EXPECT_FALSE(AcesFrameFeed::open({clip}, {}, d / "decode.log", 4, 9));   // past the end
}

// The viewer's feed on the learned route (28 Oct 2026, aces_v2_monitor.hpp).
namespace {
class HalfView final : public aces_v2::AcesView {   // display = x / (1 + x)
public:
    void forward(float* rgb, std::size_t n) const override {
        for (std::size_t i = 0; i < n * 3; ++i) rgb[i] = std::max(rgb[i], 0.0f) / (1.0f + std::max(rgb[i], 0.0f));
    }
    void inverse(float*, std::size_t) const override {}
    std::string identity() const override { return "half"; }
};
}  // namespace

TEST(AcesMonitor, OutputsInDisplayLightProxiesAndCodes) {
    bool ok = false;
    EXPECT_EQ(aces_monitor_from("hdr1000", &ok), AcesMonitor::Hdr1000);
    EXPECT_TRUE(ok);
    aces_monitor_from("p3", &ok);
    EXPECT_FALSE(ok);
    PlanarBuffer grey(3, 1, 1, 1.0f);   // display code 0.5 through the stand-in view
    HalfView v;
    EXPECT_EQ(aces_monitor_planes(grey, v, AcesMonitor::Scene).plane(0)[0], 1.0f);
    const auto sdr = aces_monitor_planes(grey, v, AcesMonitor::Sdr100);
    EXPECT_NEAR(sdr.plane(1)[0] * 203.0, 100.0 * std::pow(0.5, 2.4), 1e-3);   // BT.1886 at 100 nits, neutral stays neutral
    const auto hdr = aces_monitor_planes(grey, v, AcesMonitor::Hdr1000);
    EXPECT_NEAR(hdr.plane(1)[0] * 203.0, pq_eotf(0.5f), 0.05);                // ST 2084: code 0.5 is ~92 nits
    PlanarBuffer big(3, 4, 8);
    for (std::size_t i = 0; i < big.plane_size(); ++i)
        for (int c = 0; c < 3; ++c) big.plane(c)[i] = float(i % 2);
    const auto small = aces_area_downscale(big, 4);
    EXPECT_EQ(small.width(), 4);
    EXPECT_EQ(small.height(), 2);
    EXPECT_FLOAT_EQ(small.plane(0)[0], 0.5f);
    EXPECT_EQ(aces_area_downscale(big, 0).width(), 8);
    PlanarBuffer codes(3, 1, 1);
    codes.plane(0)[0] = 0.2f;
    codes.plane(1)[0] = 0.9f;
    codes.plane(2)[0] = 0.4f;
    EXPECT_FLOAT_EQ(aces_code_plane(codes).plane(0)[0], 0.9f);
}

// Into an existing folder, as Nuke's Write (29 Oct 2026): padding, separator,
// create directories, and nothing replaced.
TEST(AcesDeliver, IntoAnExistingFolderNothingIsReplaced) {
    EXPECT_FALSE(aces_deliver_from_json({{"into_existing", true}}));   // needs a name
    EXPECT_FALSE(aces_deliver_from_json({{"padding", 0}}));
    EXPECT_FALSE(aces_deliver_from_json({{"separator", "-"}}));
    auto o = aces_deliver_from_json({{"into_existing", true}, {"name", "SH010_hdr"}, {"start", 1001}, {"padding", 4},
                                     {"separator", "_"}, {"log", "logc4"}, {"log_formats", "dpx10"}});
    ASSERT_TRUE(o);
    EXPECT_TRUE(o->copies.flat);
    EXPECT_EQ(aces_frame_stem(*o, 2, ""), "SH010_hdr_1003");
    const auto d = tmp("into_existing");
    // Create directories off: a missing folder is refused; on: it is made.
    o->create_dirs = false;
    EXPECT_FALSE(aces_prepare_out(*o, d / "new", 3, ""));
    o->create_dirs = true;
    EXPECT_TRUE(aces_prepare_out(*o, d / "new", 3, ""));
    EXPECT_TRUE(fs::is_directory(d / "new"));
    // Other renders in the folder are fine; one of this run's files is not.
    std::ofstream(d / "new" / "SH020.1001.exr") << "x";
    EXPECT_TRUE(aces_prepare_out(*o, d / "new", 3, ""));
    std::ofstream(d / "new" / "SH010_hdr_1002.dpx") << "x";
    auto r = aces_prepare_out(*o, d / "new", 3, "");
    ASSERT_FALSE(r);
    EXPECT_NE(r.error().detail.find("SH010_hdr_1002.dpx"), std::string::npos);
    EXPECT_EQ(aces_provenance_file(*o, d / "new").filename(), "SH010_hdr_provenance.json");
    // The copies go beside the EXRs.
    auto w = AcesLogWriter::open(o->copies, d / "flat", d / "flat" / "x.mov", 24);
    ASSERT_TRUE(w);
    EXPECT_TRUE(fs::exists(d / "flat"));
    EXPECT_FALSE(fs::exists(d / "flat" / "log_logc4_dpx"));
    // The old way still refuses an existing folder.
    auto plain = aces_deliver_from_json({{"name", "x"}});
    EXPECT_FALSE(aces_prepare_out(*plain, d / "new", 1, ""));
}

TEST(AcesDeliver, FilesAppearOnlyWhenWhole) {
    // Review 28 Oct: written as <file>.part and renamed, so a crash never leaves a
    // short frame or provenance under a finished name.
    const auto d = tmp("whole");
    const PlanarBuffer px = ramp(4, 6);
    PlanarBuffer mask(1, 4, 6, 0.0f);
    auto o = aces_deliver_from_json(json::object());
    ASSERT_TRUE(o);
    ASSERT_TRUE(write_aces_master_exr(d / "a.exr", px, mask, *o, {}));
    EXPECT_TRUE(fs::exists(d / "a.exr"));
    EXPECT_FALSE(fs::exists(d / "a.exr.part"));
    ASSERT_TRUE(aces_write_whole(d / "p.json", "{\"x\":1}"));
    EXPECT_FALSE(fs::exists(d / "p.json.part"));
    std::ifstream in(d / "p.json");
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_EQ(text, "{\"x\":1}");
    // A folder that is not there: an error, and nothing left behind.
    auto bad = aces_write_whole(d / "missing" / "p.json", "{}");
    EXPECT_FALSE(bad);
    EXPECT_FALSE(fs::exists(d / "missing"));
}

TEST(AcesDeliver, Hdr10HevcCarriesTheMeasuredLightLevelAndTheMasteringDisplay) {
    // Review 28 Oct: a learned-route PQ movie with HDR10 metadata in the file.
    EXPECT_FALSE(parse_aces_log("logc4", "hevc10"));   // PQ only
    EXPECT_FALSE(parse_aces_log("pq1000", "hevc10,mov16"));
    if (!have("ffmpeg") || !have("ffprobe")) GTEST_SKIP() << "ffmpeg and ffprobe are not on the PATH";
    if (run("ffmpeg -hide_banner -encoders").find("libx265") == std::string::npos) GTEST_SKIP() << "no libx265";
    const auto d = tmp("hevc");
    auto o = parse_aces_log("pq1000", "hevc10");
    ASSERT_TRUE(o);
    EXPECT_EQ(o->movie_extension(), ".mp4");
    FakeHdr hdr;
    auto w = AcesLogWriter::open(*o, d, d / "SH010.mp4", 24.0, &hdr);
    ASSERT_TRUE(w) << w.error().message;
    (*w)->set_movie_stamp("01:00:00:00", "");
    for (int i = 0; i < 3; ++i) ASSERT_TRUE((*w)->write(ramp(16, 32), "f.exr"));
    EXPECT_FALSE(fs::exists(d / "SH010.mp4"));   // only whole files under the final name
    ASSERT_TRUE((*w)->finish(812.4, 301.6));
    EXPECT_TRUE(fs::exists(d / "SH010.mp4"));
    EXPECT_FALSE(fs::exists(d / "SH010.part.mov"));
    EXPECT_FALSE(fs::exists(d / "SH010.part.mp4"));
    const auto info = probe(d / "SH010.mp4", "codec_name,pix_fmt,color_transfer,color_primaries,color_space");
    EXPECT_NE(info.find("codec_name=hevc"), std::string::npos) << info;
    EXPECT_NE(info.find("pix_fmt=yuv420p10le"), std::string::npos) << info;
    EXPECT_NE(info.find("color_transfer=smpte2084"), std::string::npos) << info;
    // The SEI: content light level and mastering display, read back from the first frame.
    const auto side = run("ffprobe -v error -select_streams v:0 -read_intervals %+#1 -show_frames -show_entries "
                          "frame_side_data=max_content,max_average,max_luminance,red_x \"" + (d / "SH010.mp4").string() + "\"");
    EXPECT_NE(side.find("max_content=813"), std::string::npos) << side;
    EXPECT_NE(side.find("max_average=302"), std::string::npos) << side;
    EXPECT_NE(side.find("max_luminance=10000000/10000"), std::string::npos) << side;
    EXPECT_NE(side.find("red_x=34000/50000"), std::string::npos) << side;
    const auto prov = (*w)->provenance();
    EXPECT_EQ(prov["files"][0]["maxcll"], 813);   // CTA-861.3: rounded up
    // The sidecar's numbers, for a ProRes delivery entered by hand.
    const auto j = aces_hdr10_json(812.4, 301.6);
    EXPECT_EQ(j["maxcll"], 813);
    EXPECT_EQ(j["maxfall"], 302);
    EXPECT_NE(j["x265"].get<std::string>().find("max-cll=813,302"), std::string::npos);
}

TEST(AcesDeliver, AnSrgbSourceReachesTheModelAsTheSameDisplayLight) {
    // Review 28 Oct: the model reads Rec.709 on BT.1886 (2.4); an sRGB or 2.2 master is re-encoded first.
    EXPECT_FALSE(aces_deliver_from_json({{"source_display", "gamma26"}}));
    auto o = aces_deliver_from_json({{"source_display", "srgb"}});
    ASSERT_TRUE(o);
    EXPECT_EQ(o->source_display, "srgb");
    PlanarBuffer c(3, 1, 4, 0.0f);
    const float in[4] = {0.0f, 0.02f, 0.5f, 1.0f};
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 3; ++k) c.plane(k)[i] = in[i];
    PlanarBuffer same = c;
    aces_adapt_source(same, "bt1886");
    EXPECT_EQ(same.plane(0)[2], 0.5f);
    aces_adapt_source(c, "srgb");
    EXPECT_EQ(c.plane(0)[0], 0.0f);
    EXPECT_NEAR(c.plane(0)[3], 1.0f, 1e-6);
    // Same light: sRGB EOTF(0.5) = 0.2140 = BT.1886 EOTF(code) -> code = 0.2140^(1/2.4) = 0.5258
    EXPECT_NEAR(c.plane(0)[2], std::pow(std::pow((0.5 + 0.055) / 1.055, 2.4), 1 / 2.4), 2e-5);
    EXPECT_NEAR(c.plane(1)[1], std::pow(0.02 / 12.92, 1 / 2.4), 2e-5);
    PlanarBuffer g(3, 1, 1, 0.5f);
    aces_adapt_source(g, "gamma22");
    EXPECT_NEAR(g.plane(0)[0], std::pow(0.5, 2.2 / 2.4), 2e-5);
}

TEST(AcesDeliver, HdrTargetsBeyond1000NitsHlgAndTheSdrTrimPass) {
    // Review 28 Oct: 2000 / 4000-nit PQ, HLG and an SDR Rec.709 pass of the same master.
    for (const char* id : {"pq1000", "pq2000", "pq4000", "hlg1000", "sdr709"}) {
        auto o = parse_aces_log(id, "prores4444xq");
        ASSERT_TRUE(o) << id;
        EXPECT_TRUE(o->pq);
        EXPECT_EQ(o->name(), id);
    }
    EXPECT_EQ(aces_display_target("pq4000")->peak_nits, 4000);
    EXPECT_STREQ(aces_display_target("hlg1000")->display, "Rec.2100-HLG - Display");
    EXPECT_FALSE(parse_aces_log("hlg1000", "hevc10"));   // HDR10 is PQ
    EXPECT_FALSE(parse_aces_log("sdr709", "hevc10"));
    EXPECT_TRUE(parse_aces_log("pq4000", "hevc10"));
    auto has = [](const std::vector<std::string>& a, const std::string& x) {
        for (const auto& s : a)
            if (s.find(x) != std::string::npos) return true;
        return false;
    };
    const auto hlg = AcesLogWriter::movie_arguments(*parse_aces_log("hlg1000", "prores422hq"));
    EXPECT_TRUE(has(hlg, "color_trc=arib-std-b67"));
    EXPECT_TRUE(has(hlg, "color_primaries=bt2020"));
    const auto sdr = AcesLogWriter::movie_arguments(*parse_aces_log("sdr709", "prores422hq"));
    EXPECT_TRUE(has(sdr, "color_trc=bt709"));
    EXPECT_TRUE(has(sdr, "colorspace=bt709"));
    EXPECT_FALSE(has(sdr, "bt2020"));
    const auto x = AcesLogWriter::hevc_arguments(900, 200, 4000);
    EXPECT_TRUE(has(x, "L(40000000,1)"));
    EXPECT_EQ(aces_hdr10_json(900, 200, 4000)["mastering_display"]["max_nits"], 4000);
}

TEST(AcesDeliver, ALearnedPqRenderCarriesItsDolbyVisionMetadata) {
    // Review 28 Oct: Dolby Vision CM v4.0 (L1 per shot, L6) from the learned PQ output itself.
    class PqView final : public aces_v2::AcesView {
    public:
        void forward(float* rgb, std::size_t n) const override {
            for (std::size_t i = 0; i < n * 3; ++i) rgb[i] = rgb[i] > 2.0f ? 0.75f : 0.5f;
        }
        void inverse(float*, std::size_t) const override {}
        std::string identity() const override { return "pq"; }
    } pqv;
    PlanarBuffer px(3, 8, 8, 1.0f);
    for (int c = 0; c < 3; ++c) px.plane(c)[3] = 4.0f;
    std::vector<FrameStats> stats;
    for (int i = 0; i < 3; ++i) {
        const auto l = aces_light_level(px, &pqv, i);
        EXPECT_EQ(l.stats.index, i);
        EXPECT_NEAR(l.stats.max_nits, pq_eotf(0.75f), 0.01);
        EXPECT_NEAR(l.stats.min_nits, pq_eotf(0.5f), 0.01);
        stats.push_back(l.stats);
    }
    auto dv = aces_dolby_vision(stats, 1000, "SH010", 24000.0 / 1001.0, 3840, 2160);
    ASSERT_TRUE(dv) << dv.error().message;
    EXPECT_EQ(dv->shots, 1);
    EXPECT_NE(dv->xml.find("<DolbyLabsMDF"), std::string::npos);
    EXPECT_NE(dv->xml.find("<MaxCLL>" + std::to_string(int(std::ceil(pq_eotf(0.75f)))) + "</MaxCLL>"), std::string::npos) << dv->xml;
    EXPECT_NE(dv->generate_json.find("\"profile\": \"8.1\""), std::string::npos);
    EXPECT_FALSE(aces_dolby_vision({}, 1000, "x", 24, 2, 2));
    // Named beside the render; refused when already there.
    auto o = aces_deliver_from_json({{"name", "SH010"}, {"into_existing", true}, {"log", "pq1000"}, {"log_formats", "prores422hq"}});
    ASSERT_TRUE(o) << o.error().message;
    EXPECT_EQ(aces_dovi_xml_file(*o, "/r").filename(), "SH010_dolby_vision.xml");
}

TEST(AcesDeliver, ADpxCarriesItsTimecodeRateAndReel) {
    // Review 28 Oct: the DPX header filled in for the conform.
    if (!have("ffmpeg")) GTEST_SKIP() << "ffmpeg is not on the PATH";
    const auto d = tmp("dpx_stamp");
    auto o = parse_aces_log("logc4", "dpx10");
    ASSERT_TRUE(o);
    o->flat = true;
    auto w = AcesLogWriter::open(*o, d, d / "x.mov", 25.0);
    ASSERT_TRUE(w) << w.error().message;
    (*w)->set_movie_stamp("10:00:00:00", "A001C003");
    ASSERT_TRUE((*w)->write(ramp(8, 16), "SH010.1001.exr", "10:00:00:12"));
    std::ifstream in(d / "SH010.1001.dpx", std::ios::binary);
    std::vector<unsigned char> b((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    ASSERT_GT(b.size(), 2048u);
    const bool big = b[0] == 'S';
    auto u32 = [&](std::size_t at) {
        std::uint32_t v = 0;
        for (int i = 0; i < 4; ++i) v |= std::uint32_t(b[at + std::size_t(i)]) << (big ? 24 - 8 * i : 8 * i);
        return v;
    };
    EXPECT_EQ(u32(1920), 0x10000012u);   // BCD 10:00:00:12
    float rate = 0;
    const std::uint32_t r = u32(1940);
    std::memcpy(&rate, &r, 4);
    EXPECT_FLOAT_EQ(rate, 25.0f);
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(&b[1556])), "A001C003");
    EXPECT_EQ(b[801], 0);   // a log: user-defined transfer (named in the provenance)
}
