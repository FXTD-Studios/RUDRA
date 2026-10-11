// The app's movie export (10 Oct 2026): a movie through the EXR master's
// chain with the app's settings, analytic (no model, no package) or with the
// model. Here the analytic half, which needs ffmpeg and no package: the
// export runs with no model at all, says so in its report, carries the grade
// (a Region EV push moves the measured light) and runs from a queue whose
// checkpoint is the settings file.
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "rudra/deliver/queue.hpp"
#include "rudra/platform/process.hpp"
#include "rudra/video/convert.hpp"
#include "rudra/video/queue_runner.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {

const fs::path kClip = fs::path(RUDRA_GOLDEN_DIR) / "video" / "clips" / "h264_709.mp4";

bool have_ffmpeg() { return find_executable("ffmpeg") && find_executable("ffprobe"); }

fs::path fresh(const std::string& name) {
    const fs::path d = fs::temp_directory_path() / ("rudra-video-graded-" + name);
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}

nlohmann::json read_json(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return nlohmann::json::parse(f);
}

MasterRequest analytic(double ev) {
    MasterRequest q;
    q.analytic = true;
    q.regions = {{"mids", 1.0, 400.0, ev}};
    q.region_softness_stops = 0.5;
    return q;
}

}  // namespace

TEST(VideoGraded, AnAnalyticExportRunsNoModelAndCarriesTheGrade) {
    if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    const fs::path dir = fresh("analytic");
    std::optional<double> fall_flat, fall_push;
    for (double ev : {0.0, 1.0}) {
        VideoConvertArgs a;
        a.input = kClip;
        a.output = dir / (ev == 0.0 ? "flat.mp4" : "push.mp4");
        a.format = "hdr10";
        a.audio = "none";
        a.preset = "ultrafast";
        a.grade = analytic(ev);
        auto r = convert_video(a, nullptr, nullptr);
        ASSERT_TRUE(r) << r.error().message << " " << r.error().detail;
        ASSERT_TRUE(fs::exists(r->output));
        const auto rep = read_json(r->sidecar);
        EXPECT_EQ(rep["reconstruction"], "analytic (no model)");
        EXPECT_TRUE(rep["checkpoint"].is_null());
        EXPECT_EQ(rep["recovery_mode"], "all");
        EXPECT_TRUE(rep.contains("qc"));
        (ev == 0.0 ? fall_flat : fall_push) = rep["max_fall"].get<double>();
    }
    // +1 EV on 1 to 400 nits: the frame average rises.
    ASSERT_TRUE(fall_flat && fall_push);
    EXPECT_GT(*fall_push, *fall_flat * 1.3) << *fall_flat << " -> " << *fall_push;
}

TEST(VideoGraded, TheModelPathRefusesWithoutAModel) {
    VideoConvertArgs a;
    a.input = kClip;
    a.output = fresh("refuse") / "x.mp4";
    MasterRequest q;   // not analytic
    a.grade = q;
    auto r = convert_video(a, nullptr, nullptr);
    ASSERT_FALSE(r);
    EXPECT_NE(r.error().message.find("no model package"), std::string::npos) << r.error().message;
}

TEST(VideoGraded, AnAnalyticQueueNeedsNoPackage) {
    if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    const fs::path dir = fresh("queue");
    const fs::path settings = dir / "out.mp4.settings.json";
    std::ofstream(settings) << master_request_json(analytic(0.5));
    const fs::path queue = dir / "export.queue.json";
    std::ofstream(queue) << nlohmann::json{
        {"version", 1},
        {"defaults", {{"checkpoint", settings.string()}}},
        {"jobs", {{{"input", kClip.string()}, {"output", (dir / "out.mp4").string()},
                   {"options", {{"format", "hlg"}, {"audio", "none"}, {"preset", "ultrafast"},
                                {"grade", settings.string()}}}}}}}.dump(2);
    VideoQueueOptions o;   // no package, no roots
    auto r = run_video_queue(queue, false, o);
    ASSERT_TRUE(r) << r.error().message << " " << r.error().detail;
    EXPECT_EQ(*r, 0);
    ASSERT_TRUE(fs::exists(dir / "out.mp4"));
    const auto rep = read_json(dir / "out.mp4.json");
    EXPECT_EQ(rep["reconstruction"], "analytic (no model)");
    EXPECT_EQ(rep["grade_file"], fs::weakly_canonical(settings).string());
}

// Temporal consistency (10 Oct 2026): the anchor's hold is a median and moves
// with the content near the knee, so a grainy clip pumps frame to frame with
// it. Held across the shot, the HDR's exposure follows the source's closer:
// the report's flicker measure (EV the HDR moved beyond the source) drops.
TEST(VideoGraded, TheShotHoldSteadiesTheAnchor) {
    if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    const fs::path dir = fresh("steady");
    // A second of a moving test pattern with grain, 24 fps, Rec.709.
    const fs::path clip = dir / "grain.mp4";
    auto made = run_process({find_executable("ffmpeg")->string(), "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                             "testsrc2=size=320x180:rate=24:duration=1,noise=alls=12:allf=t,format=yuv420p",
                             "-c:v", "libx264", "-crf", "18", "-color_primaries", "bt709", "-color_trc", "bt709",
                             "-colorspace", "bt709", clip.string()});
    ASSERT_TRUE(made && fs::exists(clip)) << "this ffmpeg cannot make the test clip (libx264)";
    std::vector<double> flicker;
    for (double seconds : {0.0, 0.25}) {
        VideoConvertArgs a;
        a.input = clip;
        a.output = dir / (seconds == 0.0 ? "off.mp4" : "held.mp4");
        a.audio = "none";
        a.preset = "ultrafast";
        a.temporal_smoothing = seconds;
        MasterRequest q;
        q.analytic = true;
        q.anchor = true;
        a.grade = q;
        auto r = convert_video(a, nullptr, nullptr);
        ASSERT_TRUE(r) << r.error().message << " " << r.error().detail;
        const auto rep = read_json(r->sidecar);
        EXPECT_EQ(rep["temporal_smoothing"].get<double>(), seconds);
        const auto& t = rep["temporal"];
        EXPECT_NEAR(t["retention"].get<double>(), seconds == 0.0 ? 0.0 : std::exp(-1.0 / (24.0 * 0.25)), 1e-12);
        EXPECT_GT(t["frames_scored"].get<int>(), 10);
        flicker.push_back(t["flicker_ev_mean"].get<double>());
        std::cout << "[ steady ] " << seconds << " s: flicker " << t.dump() << "\n";
    }
    EXPECT_LT(flicker[1], flicker[0] * 0.7) << flicker[0] << " -> " << flicker[1];
}

// The static hold (10 Oct 2026): on a still shot with grain, the HDR's gain
// over the source is held where the source is still, so the frame-to-frame
// change the expansion adds (the local flicker) drops; off, every frame is its
// own.
TEST(VideoGraded, TheStaticHoldSteadiesAStillShot) {
    if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    const fs::path dir = fresh("static");
    const std::string ffmpeg = find_executable("ffmpeg")->string();
    const fs::path still = dir / "still.png", clip = dir / "still.mp4";
    // A grey ramp (nothing clipped or crushed: those are never held) with
    // grain, a second of it at 24 fps.
    auto png = run_process({ffmpeg, "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                            "nullsrc=size=320x180,format=gray,geq=lum='40+200*X/W'", "-frames:v", "1", still.string()});
    auto made = run_process({ffmpeg, "-v", "error", "-nostdin", "-loop", "1", "-i", still.string(), "-vf",
                             "fps=24,format=yuv420p,noise=alls=3:allf=t", "-t", "1", "-c:v", "libx264", "-crf", "23",
                             "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", clip.string()});
    ASSERT_TRUE(png && made && fs::exists(clip)) << "this ffmpeg cannot make the test clip (libx264)";
    std::vector<double> local;
    for (const char* mode : {"off", "on"}) {
        VideoConvertArgs a;
        a.input = clip;
        a.output = dir / (std::string(mode) + ".mp4");
        a.audio = "none";
        a.preset = "ultrafast";
        a.static_hold = mode;
        MasterRequest q;
        q.analytic = true;
        a.grade = q;
        auto r = convert_video(a, nullptr, nullptr);
        ASSERT_TRUE(r) << r.error().message << " " << r.error().detail;
        const auto rep = read_json(r->sidecar);
        EXPECT_EQ(rep["static_hold"], mode);
        EXPECT_EQ(rep["temporal"]["static_hold"], mode);
        local.push_back(rep["temporal"]["local_flicker_ev_mean"].get<double>());
        local.push_back(rep["temporal"]["local_flicker_ev_max"].get<double>());
        std::cout << "[ static ] " << mode << ": " << rep["temporal"].dump() << "\n";
    }
    // (off mean, off max, on mean, on max): the worst frame most of all.
    EXPECT_LT(local[2], local[0] * 0.85) << local[0] << " -> " << local[2];
    EXPECT_LT(local[3], local[1] * 0.5) << local[1] << " -> " << local[3];
}
