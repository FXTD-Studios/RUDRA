// Projects (.rudra, engine/project.hpp; product item 3, 10 Oct 2026): what is
// saved comes back, paths survive a move, masks travel beside the file, a
// newer or damaged file is refused, and a save never leaves half a project.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "rudra/core/masks.hpp"
#include "rudra/engine/project.hpp"
#include "rudra/engine/session.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {

fs::path fresh(const std::string& name) {
    const fs::path d = fs::temp_directory_path() / ("rudra-project-" + name);
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}

std::string read(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

Project sample(const fs::path& footage) {
    Session s;
    s.run("strength-up");
    s.run("mode-shadows");
    s.run("container-linear");
    s.peak_input(1.5);
    Project p = project_from_session(s);
    p.grade.source = "hable";
    p.grade.calibration = {{128, 18.0}, {235, 100.0}};
    if (p.grade.regions.empty()) p.grade.regions.push_back({"mids", 1.0, 100.0, 0.0});
    p.grade.regions.front().ev = 0.75;
    p.grade.reference.file = "grade.exr";
    p.grade.reference.target_log2_nits.assign(256, 2.5f);
    p.grade.reference.target_log2_nits[128] = 3.25f;
    p.grade.reference.samples_per_code.assign(256, 9);
    p.grade.reference.residual_mean = 0.02;
    p.grade.reference.residual_p95 = 0.08;
    p.grade.reference.samples = 1234;
    p.grade.reference.codes_seen = 3;
    p.anchor = false;
    p.anchor_knee = 0.8;
    p.carry_chroma = false;
    p.source_kind = "folder";
    p.sources = {footage};
    p.frame = 7;
    p.package = "/models/sdr2hdr_image_v8";
    p.backend = "onnxruntime:cpu";
    p.use_model = true;
    p.app_version = "0.9.0-beta.6";
    return p;
}

}  // namespace

TEST(Project, WhatIsSavedComesBack) {
    const fs::path dir = fresh("roundtrip");
    fs::create_directories(dir / "plates");
    const Project p = sample(dir / "plates");
    auto saved = save_project(dir / "shot", p);
    ASSERT_TRUE(saved) << saved.error().message;
    EXPECT_EQ(*saved, dir / "shot.rudra");   // the suffix is added
    EXPECT_FALSE(fs::exists(dir / "shot.rudra.tmp"));
    EXPECT_FALSE(fs::exists(dir / "shot.rudra.masks.png"));   // no masks, no file
    auto q = load_project(*saved);
    ASSERT_TRUE(q) << q.error().message;
    EXPECT_EQ(q->source_kind, "folder");
    ASSERT_EQ(q->sources.size(), 1u);
    EXPECT_EQ(q->sources[0], dir / "plates");
    EXPECT_EQ(q->frame, 7);
    EXPECT_EQ(q->package.generic_string(), "/models/sdr2hdr_image_v8");
    EXPECT_EQ(q->backend, "onnxruntime:cpu");
    EXPECT_TRUE(q->use_model);
    EXPECT_FALSE(Project().use_model);   // a project that does not say: the analytic reconstruction
    EXPECT_EQ(q->app_version, "0.9.0-beta.6");
    EXPECT_EQ(q->grade.mode, p.grade.mode);
    EXPECT_DOUBLE_EQ(q->grade.strength, p.grade.strength);
    EXPECT_EQ(q->grade.preserve, p.grade.preserve);
    EXPECT_EQ(q->grade.source, "hable");
    ASSERT_EQ(q->grade.regions.size(), p.grade.regions.size());
    for (std::size_t i = 0; i < p.grade.regions.size(); ++i) {
        EXPECT_EQ(q->grade.regions[i].label, p.grade.regions[i].label);
        EXPECT_DOUBLE_EQ(q->grade.regions[i].low_nits, p.grade.regions[i].low_nits);
        EXPECT_DOUBLE_EQ(q->grade.regions[i].high_nits, p.grade.regions[i].high_nits);
        EXPECT_DOUBLE_EQ(q->grade.regions[i].ev, p.grade.regions[i].ev);
    }
    ASSERT_EQ(q->grade.calibration.size(), 2u);
    EXPECT_EQ(q->grade.calibration[1].code, 235);
    EXPECT_DOUBLE_EQ(q->grade.calibration[1].nits, 100.0);
    EXPECT_EQ(q->grade.reference.file, "grade.exr");
    EXPECT_EQ(q->grade.reference.target_log2_nits, p.grade.reference.target_log2_nits);
    EXPECT_EQ(q->grade.reference.samples_per_code, p.grade.reference.samples_per_code);
    EXPECT_EQ(q->grade.reference.samples, 1234);
    EXPECT_DOUBLE_EQ(q->peak_ev, 1.5);
    EXPECT_FALSE(q->anchor);
    EXPECT_DOUBLE_EQ(q->anchor_knee, 0.8);
    EXPECT_FALSE(q->carry_chroma);
    EXPECT_EQ(q->container, "linear");
    // Into a session: the grade and the Deliver settings, with undo behind them.
    Session s;
    s.load_state(q->grade, q->peak_ev, q->anchor, q->carry_chroma, q->anchor_knee, q->container);
    EXPECT_EQ(s.grade.mode, p.grade.mode);
    EXPECT_EQ(s.container, "linear");
    EXPECT_DOUBLE_EQ(s.peak_ev, 1.5);
    EXPECT_EQ(s.undo_depth(), 1u);
    s.run("undo");
    EXPECT_EQ(s.grade.mode, Session().grade.mode);
}

TEST(Project, PathsUnderTheProjectFollowAMove) {
    const fs::path a = fresh("move-a"), b = fresh("move-b");
    fs::remove_all(b);
    fs::create_directories(a / "plates");
    std::ofstream(a / "plates" / "f.0001.png") << "x";
    Project p = sample(a / "plates");
    p.source_kind = "files";
    p.sources = {a / "plates" / "f.0001.png", "/elsewhere/f.0002.png"};
    ASSERT_TRUE(save_project(a / "shot.rudra", p));
    const auto j = nlohmann::json::parse(read(a / "shot.rudra"));
    EXPECT_EQ(j["shot"]["sources"][0]["relative"], "plates/f.0001.png");
    EXPECT_FALSE(j["shot"]["sources"][1].contains("relative"));   // outside: absolute only
    fs::rename(a, b);   // the project and its footage, moved together
    auto q = load_project(b / "shot.rudra");
    ASSERT_TRUE(q) << q.error().message;
    EXPECT_EQ(q->sources[0], b / "plates" / "f.0001.png");
    EXPECT_EQ(q->sources[1].generic_string(), "/elsewhere/f.0002.png");
}

TEST(Project, PaintedMasksTravelBesideIt) {
    const fs::path dir = fresh("masks");
    Project p = sample(dir);
    auto set = std::make_shared<MaskSet>();
    set->width = 64;
    set->height = 32;
    auto plane = empty_plane(64, 32);
    paint_stroke(*plane, 64, 32, 8, 8, 40, 20, Brush{});
    set->planes[1] = plane;
    p.grade.masks = set;
    std::ofstream(dir / "shot.masks.png") << "a master's masks";   // shot.exr's, beside the project
    ASSERT_TRUE(save_project(dir / "shot.rudra", p));
    ASSERT_TRUE(fs::exists(dir / "shot.rudra.masks.png"));
    EXPECT_EQ(read(dir / "shot.masks.png"), "a master's masks");   // left alone
    EXPECT_FALSE(fs::exists(dir / "shot.rudra.masks.png.tmp"));
    auto q = load_project(dir / "shot.rudra");
    ASSERT_TRUE(q) << q.error().message;
    ASSERT_TRUE(q->grade.masks);
    EXPECT_TRUE(*q->grade.masks == *set);
    EXPECT_TRUE(q->notes.empty());
    // Renamed in Explorer: the masks it names are still found.
    fs::copy_file(dir / "shot.rudra", dir / "shot v2.rudra");
    q = load_project(dir / "shot v2.rudra");
    ASSERT_TRUE(q);
    ASSERT_TRUE(q->grade.masks);
    EXPECT_TRUE(*q->grade.masks == *set);
    // Saved again without masks: the old file goes, so it is not read back.
    p.grade.masks.reset();
    ASSERT_TRUE(save_project(dir / "shot.rudra", p));
    EXPECT_FALSE(fs::exists(dir / "shot.rudra.masks.png"));
    EXPECT_EQ(read(dir / "shot.masks.png"), "a master's masks");
    q = load_project(dir / "shot.rudra");
    ASSERT_TRUE(q);
    EXPECT_FALSE(q->grade.masks);
    // Masks that have gone: the project opens without them, and says so.
    p.grade.masks = set;
    ASSERT_TRUE(save_project(dir / "shot.rudra", p));
    fs::remove(dir / "shot.rudra.masks.png");
    q = load_project(dir / "shot.rudra");
    ASSERT_TRUE(q) << q.error().message;
    EXPECT_FALSE(q->grade.masks);
    ASSERT_EQ(q->notes.size(), 1u);
    EXPECT_NE(q->notes[0].find("masks (shot.rudra.masks.png) are missing"), std::string::npos) << q->notes[0];
}

TEST(Project, ACalibrationHalfDoneSavesAndOpens) {
    // One patch clicked: the session keeps all three slots, the others at code -1.
    const fs::path dir = fresh("calibration");
    Session s;
    s.set_calibration(1, 118, 18.0);
    s.set_calibration(2, -1, 300.0);   // nits typed, not yet clicked
    ASSERT_EQ(s.grade.calibration.size(), 3u);
    ASSERT_TRUE(save_project(dir / "cal.rudra", project_from_session(s)));
    auto q = load_project(dir / "cal.rudra");
    ASSERT_TRUE(q) << q.error().message;
    EXPECT_EQ(q->grade.calibration, s.grade.calibration);
}

TEST(Project, HandEditedValuesArePulledIntoRange) {
    const fs::path dir = fresh("range");
    std::ofstream(dir / "r.rudra") << R"({"rudra_project": 1, "grade": {"strength": 9, "regions": [{"label": "a", "ev": 40}]},
        "view": {"peak_ev": 1000}, "deliver": {"anchor_knee": 3}})";
    auto q = load_project(dir / "r.rudra");
    ASSERT_TRUE(q) << q.error().message;
    EXPECT_DOUBLE_EQ(q->grade.strength, 2.0);
    EXPECT_DOUBLE_EQ(q->peak_ev, 5.0);
    EXPECT_DOUBLE_EQ(q->anchor_knee, 0.99);
    EXPECT_DOUBLE_EQ(q->grade.regions[0].ev, 4.0);
}

TEST(Project, NewerDamagedOrForeignFilesAreRefused) {
    const fs::path dir = fresh("refused");
    std::ofstream(dir / "newer.rudra") << R"({"rudra_project": 99, "shot": {}})";
    auto q = load_project(dir / "newer.rudra");
    ASSERT_FALSE(q);
    EXPECT_EQ(q.error().code, ErrorCode::ContractMismatch);
    EXPECT_NE(q.error().message.find("newer RUDRA"), std::string::npos);
    std::ofstream(dir / "other.rudra") << R"({"name": "a queue"})";
    EXPECT_FALSE(load_project(dir / "other.rudra"));
    std::ofstream(dir / "text.rudra") << "not json";
    EXPECT_FALSE(load_project(dir / "text.rudra"));
    std::ofstream(dir / "kind.rudra") << R"({"rudra_project": "1"})";
    EXPECT_FALSE(load_project(dir / "kind.rudra"));
    std::ofstream(dir / "damaged.rudra") << R"({"rudra_project": 1, "grade": {"strength": "strong"}})";
    q = load_project(dir / "damaged.rudra");
    ASSERT_FALSE(q);
    EXPECT_NE(q.error().message.find("damaged"), std::string::npos);
    EXPECT_FALSE(load_project(dir / "absent.rudra"));
    // In range, or refused: the window indexes these.
    for (const char* bad : {R"({"rudra_project": 1, "grade": {"mode": "everything"}})",
                            R"({"rudra_project": 1, "grade": {"source": "kodak"}})",
                            R"({"rudra_project": 1, "grade": {"calibration": [{"code": 300, "nits": 1}]}})",
                            R"({"rudra_project": 1, "grade": {"reference": {"target_log2_nits": [1, 2]}}})",
                            R"({"rudra_project": 1, "shot": {"kind": "tape"}})",
                            R"({"rudra_project": 1, "deliver": {"container": "p3"}})"}) {
        std::ofstream(dir / "range.rudra", std::ios::trunc) << bad;
        EXPECT_FALSE(load_project(dir / "range.rudra")) << bad;
    }
}

TEST(Project, ASaveReplacesTheOldFileWhole) {
    const fs::path dir = fresh("replace");
    Project p = sample(dir);
    ASSERT_TRUE(save_project(dir / "shot.rudra", p));
    p.frame = 42;
    ASSERT_TRUE(save_project(dir / "shot.rudra", p));
    auto q = load_project(dir / "shot.rudra");
    ASSERT_TRUE(q);
    EXPECT_EQ(q->frame, 42);
    std::size_t files = 0;
    for ([[maybe_unused]] const auto& e : fs::directory_iterator(dir)) ++files;
    EXPECT_EQ(files, 1u);   // no temporary left behind
}
