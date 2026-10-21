// Shot keyframes in a movie export (roadmap 4.3): each frame takes its shot's
// grade, the holds reset at shot boundaries, and the report lists the shots
// with their own flicker. Analytic, so it needs ffmpeg and no model.
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

#include "rudra/platform/process.hpp"
#include "rudra/video/convert.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {

const fs::path kClip = fs::path(RUDRA_GOLDEN_DIR) / "video" / "clips" / "h264_709.mp4";   // 6 frames

bool have_ffmpeg() { return find_executable("ffmpeg") && find_executable("ffprobe"); }

fs::path fresh(const std::string& name) {
    const fs::path d = fs::temp_directory_path() / ("rudra-shot-export-" + name);
    std::error_code ec;
    fs::remove_all(d, ec);
    fs::create_directories(d);
    return d;
}

nlohmann::json read_json(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return nlohmann::json::parse(f);
}

MasterRequest analytic(double exposure) {
    MasterRequest q;
    q.analytic = true;
    q.exposure_ev = exposure;
    return q;
}

VideoConvertArgs args(const fs::path& out) {
    VideoConvertArgs a;
    a.input = kClip;
    a.output = out;
    a.format = "hdr10";
    a.audio = "none";
    a.preset = "ultrafast";
    a.dynamic_metadata = "off";
    return a;
}

}  // namespace

TEST(ShotExport, EachShotTakesItsOwnGrade) {
    if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    const fs::path dir = fresh("grades");
    // Flat: one grade for the whole clip.
    auto flat_args = args(dir / "flat.mp4");
    flat_args.grade = analytic(0.0);
    auto flat = convert_video(flat_args, nullptr, nullptr);
    ASSERT_TRUE(flat) << flat.error().message << " " << flat.error().detail;
    // Two shots; the second is keyed 1.5 stops up.
    ShotGrades s = shot_grades_from_cuts({3}, 6, analytic(0.0));
    ASSERT_TRUE(set_key(s, 3, analytic(1.5)));
    const fs::path shots_file = dir / "clip.shots.json";
    ASSERT_TRUE(save_shot_grades(shots_file, s));
    auto keyed_args = args(dir / "keyed.mp4");
    keyed_args.shots_file = shots_file;
    keyed_args.shots = s;
    auto keyed = convert_video(keyed_args, nullptr, nullptr);
    ASSERT_TRUE(keyed) << keyed.error().message << " " << keyed.error().detail;

    const auto fr = read_json(flat->sidecar), kr = read_json(keyed->sidecar);
    EXPECT_EQ(kr["reconstruction"], "analytic (no model)");
    EXPECT_EQ(kr["shots_file"], fs::weakly_canonical(shots_file).string());
    ASSERT_EQ(kr["shots"].size(), 2u);
    EXPECT_EQ(kr["shots"][0]["name"], "sh010");
    EXPECT_EQ(kr["shots"][1]["first"], 3);
    EXPECT_EQ(kr["shots"][1]["keys"], 1);
    // The boundary is not scored for flicker: 2 frames scored in each 3-frame shot.
    EXPECT_EQ(kr["shots"][0]["frames_scored"], 2);
    EXPECT_EQ(kr["shots"][1]["frames_scored"], 2);
    EXPECT_EQ(kr["temporal"]["frames_scored"], 4);
    for (int i = 0; i < 6; ++i) {
        const double f = fr["frames"][i]["frame_average"].get<double>();
        const double k = kr["frames"][i]["frame_average"].get<double>();
        if (i < 3) EXPECT_NEAR(k, f, 1e-9) << "frame " << i;
        else EXPECT_GT(k, f * 1.8) << "frame " << i << ": " << f << " -> " << k;
    }
}

// The hold restarts at a shot's end when a gap follows it: the fresh smoother
// reports that frame as a cut, so the dynamic metadata starts a shot there too.
TEST(ShotExport, AGapAfterAShotStartsAMetadataShotWhereTheHoldRestarts) {
    if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    const fs::path dir = fresh("gap");
    ShotGrades s;
    s.base = analytic(0.0);
    s.shots = {{"sh010", 0, 1, "edl", {}}, {"sh020", 3, 5, "edl", {}}};   // frame 2 is in no shot
    auto a = args(dir / "gap.mp4");
    a.dynamic_metadata = "on";
    a.shots = s;
    auto r = convert_video(a, nullptr, nullptr);
    ASSERT_TRUE(r) << r.error().message << " " << r.error().detail;
    const auto dovi = read_json(dir / "gap.mp4_dovi_generate.json");
    std::vector<int> starts;
    for (const auto& g : dovi["shots"]) starts.push_back(g["start"].get<int>());
    for (int want : {0, 2, 3})
        EXPECT_NE(std::find(starts.begin(), starts.end(), want), starts.end()) << "no metadata shot starts at " << want;
}

TEST(ShotExport, AShotsFileMixingModelAndAnalyticIsRefused) {
    ShotGrades s = shot_grades_from_cuts({3}, 6, analytic(0.0));
    MasterRequest model;   // not analytic
    ASSERT_TRUE(set_key(s, 4, model));
    auto a = args(fresh("mixed") / "x.mp4");
    a.shots = s;
    auto r = convert_video(a, nullptr, nullptr);
    ASSERT_FALSE(r);
    EXPECT_NE(r.error().message.find("must use the model, or none may"), std::string::npos) << r.error().message;
}

TEST(ShotExport, ShotsPastTheEndOfTheClipAreRefused) {
    if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    ShotGrades s = shot_grades_from_cuts({3}, 10, analytic(0.0));   // the clip has 6 frames
    auto a = args(fresh("past") / "x.mp4");
    a.shots = s;
    auto r = convert_video(a, nullptr, nullptr);
    ASSERT_FALSE(r);
    EXPECT_NE(r.error().message.find("past the end"), std::string::npos) << r.error().message;
}
