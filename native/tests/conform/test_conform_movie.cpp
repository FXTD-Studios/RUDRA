// Conform on a real movie (roadmap 5.4): its cuts detected, an EDL through
// `rudra-native conform`, then one keyframed export of the whole programme.
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

#include "rudra/platform/process.hpp"
#include "rudra/video/convert.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {

bool have_ffmpeg() { return find_executable("ffmpeg") && find_executable("ffprobe"); }

fs::path fresh(const std::string& name) {
    const fs::path d = fs::temp_directory_path() / ("rudra-conform-movie-" + name);
    std::error_code ec;
    fs::remove_all(d, ec);
    fs::create_directories(d);
    return d;
}

std::string q(const fs::path& p) { return "\"" + p.string() + "\""; }

// cmd.exe strips the first and last quote of a command that starts with one.
int run(const std::string& cmd) {
#ifdef _WIN32
    return std::system(("\"" + cmd + "\"").c_str());
#else
    return std::system(cmd.c_str());
#endif
}

// 12 frames at 24 fps, 64x36, BT.709: 5 grey, 4 a brighter grey, 3 dark: two hard cuts.
fs::path three_shot_clip(const fs::path& dir) {
    const fs::path out = dir / "three_shots.mp4";
    const std::string cmd =
        "ffmpeg -v error -y -f lavfi -i color=c=0x606060:s=64x36:r=24:d=0.2083333 "
        "-f lavfi -i color=c=0xC8C8C8:s=64x36:r=24:d=0.1666667 "
        "-f lavfi -i color=c=0x181818:s=64x36:r=24:d=0.125 "
        "-filter_complex \"[0:v][1:v][2:v]concat=n=3:v=1,format=yuv420p,setparams=color_primaries=bt709:"
        "color_trc=bt709:colorspace=bt709:range=tv\" -frames:v 12 -c:v libx264 -preset ultrafast -crf 1 "
        "-color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range tv " + q(out);
    if (run(cmd) != 0) return {};
    return out;
}


nlohmann::json read_json(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return nlohmann::json::parse(f);
}

}  // namespace

TEST(ConformMovie, HardCutsAreDetected) {
    if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    const fs::path clip = three_shot_clip(fresh("detect"));
    if (clip.empty()) GTEST_SKIP() << "this ffmpeg could not make the test clip (libx264)";
    auto d = detect_cuts(clip);
    ASSERT_TRUE(d) << d.error().message << " " << d.error().detail;
    EXPECT_EQ(d->frames, 12);
    EXPECT_EQ(d->fps, "24");
    EXPECT_EQ(d->cuts, (std::vector<int>{5, 9}));
    EXPECT_FALSE(detect_cuts(clip, 0.0));
}

#ifdef RUDRA_CLI_PATH
TEST(ConformMovie, AnEdlConformsAndTheProgrammeExportsInOnePass) {
    if (!have_ffmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    const fs::path dir = fresh("edl");
    const fs::path clip = three_shot_clip(dir);
    if (clip.empty()) GTEST_SKIP() << "this ffmpeg could not make the test clip (libx264)";
    {
        std::ofstream edl(dir / "cut.edl");
        edl << "TITLE: THREE SHOTS\nFCM: NON-DROP FRAME\n"
               "001  R1 V C 00:00:00:00 00:00:00:05 01:00:00:00 01:00:00:05\n* FROM CLIP NAME: grey\n"
               "002  R2 V C 00:00:00:00 00:00:00:04 01:00:00:05 01:00:00:09\n* FROM CLIP NAME: bright\n"
               "003  R3 V C 00:00:00:00 00:00:00:03 01:00:00:09 01:00:00:12\n* FROM CLIP NAME: dark\n";
    }
    {
        std::ofstream base(dir / "base.json");
        base << R"({"analytic": true})";
    }
    const std::string cli = q(fs::path(RUDRA_CLI_PATH));
    ASSERT_EQ(run(cli + " conform " + q(clip) + " --edl " + q(dir / "cut.edl") + " --base " + q(dir / "base.json") +
                  " --key-each-shot --output " + q(dir / "cut.shots.json")), 0);
    // Refuses to replace it.
    EXPECT_NE(run(cli + " conform " + q(clip) + " --detect --output " + q(dir / "cut.shots.json")), 0);
    const auto shots = read_json(dir / "cut.shots.json");
    ASSERT_EQ(shots["shots"].size(), 3u);
    EXPECT_EQ(shots["shots"][1]["name"], "bright");
    EXPECT_EQ(shots["shots"][1]["first"], 5);
    EXPECT_EQ(shots["shots"][2]["last"], 11);
    ASSERT_EQ(shots["shots"][2]["keys"].size(), 1u);
    // Grade the dark shot up a stop by hand, then export the programme once.
    auto edited = shots;
    edited["shots"][2]["keys"][0]["grade"]["exposure_ev"] = 1.0;
    {
        std::ofstream o(dir / "graded.shots.json");
        o << edited.dump(2);
    }
    ASSERT_EQ(run(cli + " video - " + q(clip) + " --output " + q(dir / "programme.mp4") + " --shots " +
                  q(dir / "graded.shots.json") + " --audio none --preset ultrafast --dynamic-metadata off"), 0);
    const auto rep = read_json(dir / "programme.mp4.json");
    ASSERT_EQ(rep["shots"].size(), 3u);
    EXPECT_EQ(rep["shots"][2]["name"], "dark");
    EXPECT_EQ(rep["shots"][2]["source"], "edl");
    EXPECT_EQ(rep["frames"].size(), 12u);
    // The key reached the picture: the same clip with the base grade alone is
    // darker on the dark shot's first frame (9) than the programme graded up a stop.
    ASSERT_EQ(run(cli + " video - " + q(clip) + " --output " + q(dir / "reference.mp4") + " --grade " +
                  q(dir / "base.json") + " --audio none --preset ultrafast --dynamic-metadata off"), 0);
    const auto ref = read_json(dir / "reference.mp4.json");
    ASSERT_EQ(ref["frames"].size(), 12u);
    EXPECT_GT(rep["frames"][9]["frame_average"].get<double>(), ref["frames"][9]["frame_average"].get<double>());
    // The detected cuts and the EDL agree here.
    ASSERT_EQ(run(cli + " conform " + q(clip) + " --detect --output " + q(dir / "detected.shots.json")), 0);
    const auto det = read_json(dir / "detected.shots.json");
    ASSERT_EQ(det["shots"].size(), 3u);
    EXPECT_EQ(det["shots"][1]["first"], 5);
    EXPECT_EQ(det["shots"][2]["first"], 9);
    EXPECT_EQ(det["shots"][0]["source"], "detected");
}
#endif
