// The app's movie export (10 Oct 2026): a movie through the EXR master's
// chain with the app's settings, analytic (no model, no package) or with the
// model. Here the analytic half, which needs ffmpeg and no package: the
// export runs with no model at all, says so in its report, carries the grade
// (a Region EV push moves the measured light) and runs from a queue whose
// checkpoint is the settings file.
#include <gtest/gtest.h>

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
