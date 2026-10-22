// Transactional headroom movie export (E2). Refusals run everywhere; the real
// decode/encode/QC/publish path needs Windows source leases and FFmpeg.
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <fstream>

#include <nlohmann/json.hpp>

#include "rudra/engine/headroom_export.hpp"
#include "rudra/media/video_probe.hpp"
#include "rudra/platform/process.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {
fs::path fresh(const std::string& name) {
    const fs::path d = fs::temp_directory_path() /
                       ("rudra-headroom-export-" + name + "-" +
                        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(d);
    return d;
}
MasterRequest v1_grade() {
    MasterRequest q;
    q.analytic = true;
    q.source_curve = "bt1886";
    q.anchor = q.carry_chroma = q.settle_grain = q.clip_fill = false;
    q.exposure_ev = 0.5;
    return q;
}
bool only_entry(const fs::path& dir, const fs::path& keep) {
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path() != keep) return false;
    return true;
}
// A small clip in the v1 contract: limited-range Rec.709, 10-bit 4:4:4 FFV1.
std::optional<fs::path> v1_clip(const fs::path& dir, int frames = 6) {
    auto ffmpeg = find_executable("ffmpeg");
    if (!ffmpeg) return std::nullopt;
    const fs::path out = dir / "source.mkv";
    auto r = run_process({ffmpeg->string(), "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                          "testsrc2=size=64x36:rate=24", "-vf",
                          "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709", "-frames:v", std::to_string(frames), "-pix_fmt",
                          "yuv444p10le", "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709",
                          "-color_trc", "bt709", "-c:v", "ffv1", "-level", "3", out.string()});
    if (!r || r->exit_code != 0) return std::nullopt;
    return out;
}
}  // namespace

TEST(HeadroomExport, RefusesUnknownTargetsBadPeaksAndRefusedGrades) {
    const auto dir = fresh("refuse");
    HeadroomExportRequest r;
    r.source = dir / "missing.mkv";
    r.output = dir / "out.mov";
    r.grade = v1_grade();
    r.target = "dnxhr";
    EXPECT_FALSE(export_headroom_movie(r));
    r.target = "hdr10";
    r.peak_nits = 0;
    EXPECT_FALSE(export_headroom_movie(r));
    r.peak_nits = 1000;
    r.grade.anchor = true;   // v1 cannot represent the SDR anchor
    auto refused = export_headroom_movie(r);
    ASSERT_FALSE(refused);
    EXPECT_TRUE(fs::is_empty(dir)) << "nothing is staged for a refused grade";
}

TEST(HeadroomExport, NeverReplacesAnExistingMovieOrSidecar) {
    const auto dir = fresh("exists");
    HeadroomExportRequest r;
    r.source = dir / "missing.mkv";
    r.output = dir / "out.mov";   // hdr10 writes .mp4: the suffix is the target's
    r.grade = v1_grade();
    { std::ofstream(dir / "out.mp4") << "keep"; }
    auto a = export_headroom_movie(r);
    ASSERT_FALSE(a);
    EXPECT_NE(a.error().message.find("already exists"), std::string::npos) << a.error().message;
    fs::remove(dir / "out.mp4");
    { std::ofstream(headroom_export_sidecar(dir / "out.mp4")) << "keep"; }
    EXPECT_FALSE(export_headroom_movie(r));
    EXPECT_TRUE(only_entry(dir, headroom_export_sidecar(dir / "out.mp4")));
}

TEST(HeadroomExport, AFailedPreparationLeavesNoStagingOrOutput) {
    const auto dir = fresh("prepare");
    auto clip = v1_clip(dir);
    if (!clip) GTEST_SKIP() << "FFmpeg with FFV1 required";
    HeadroomExportRequest r;
    r.source = *clip;
    r.output = dir / "out.mov";
    r.grade = v1_grade();
#ifdef _WIN32
    r.decoder = dir / "no-such-ffmpeg.exe";   // a decoder that cannot be used
#endif
    auto e = export_headroom_movie(r);
    ASSERT_FALSE(e);
    EXPECT_TRUE(only_entry(dir, *clip)) << "staging removed, nothing published: " << e.error().message;
}

#ifdef _WIN32
TEST(HeadroomExport, PublishesOnlyAfterDecodeEncodeAndQcWithProvenance) {
    if (!find_executable("ffmpeg") || !find_executable("ffprobe")) GTEST_SKIP() << "FFmpeg and ffprobe required";
    const auto dir = fresh("publish");
    auto clip = v1_clip(dir);
    ASSERT_TRUE(clip);
    HeadroomExportRequest r;
    r.source = *clip;
    r.output = dir / "graded.mov";
    r.grade = v1_grade();
    r.app_version = "test";
    std::vector<std::string> phases;
    auto out = export_headroom_movie(r, [&](const HeadroomExportProgress& p) {
        if (phases.empty() || phases.back() != p.phase) phases.push_back(p.phase);
    });
    ASSERT_TRUE(out) << out.error().message << " " << out.error().detail;
    EXPECT_EQ(phases, (std::vector<std::string>{"prepare", "render", "encode", "qc", "publish"}));
    EXPECT_EQ(out->frames, 6);
    EXPECT_TRUE(fs::is_regular_file(out->file));
    std::ifstream side(out->sidecar);
    const auto j = nlohmann::json::parse(side);
    EXPECT_EQ(j.at("policy"), "bt1886-headroom-v1");
    EXPECT_EQ(j.at("frames"), 6);
    EXPECT_EQ(j.at("source_sha256"), out->source_sha256);
    EXPECT_TRUE(j.at("qc").at("decoder_completion_verified").get<bool>());
    EXPECT_GT(j.at("maxcll").get<int>(), 0);
    auto probe = probe_video(out->file);
    ASSERT_TRUE(probe);
    EXPECT_EQ(probe->video_streams.at(0).color_transfer.value_or(""), "smpte2084");
    // Only the movie, its sidecar and the source: the staging is gone.
    int entries = 0;
    for ([[maybe_unused]] const auto& e : fs::directory_iterator(dir)) ++entries;
    EXPECT_EQ(entries, 3);
}

TEST(HeadroomExport, CancellationLeavesNothingAtTheOutput) {
    if (!find_executable("ffmpeg") || !find_executable("ffprobe")) GTEST_SKIP() << "FFmpeg and ffprobe required";
    const auto dir = fresh("cancel");
    auto clip = v1_clip(dir, 12);
    ASSERT_TRUE(clip);
    HeadroomExportRequest r;
    r.source = *clip;
    r.output = dir / "graded.mov";
    r.grade = v1_grade();
    std::atomic<bool> stop{false};
    auto out = export_headroom_movie(r, [&](const HeadroomExportProgress& p) {
        if (p.phase == "render" && p.done == 3) stop = true;
    }, &stop);
    ASSERT_FALSE(out);
    EXPECT_EQ(out.error().code, ErrorCode::Cancelled);
    EXPECT_TRUE(only_entry(dir, *clip));
}
#endif
