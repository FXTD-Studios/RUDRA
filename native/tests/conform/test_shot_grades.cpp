// Shot keyframes (roadmap 4.3): per-shot grades, keyed, reset at cuts.
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "rudra/deliver/shot_grades.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {

MasterRequest grade(double strength, double exposure, double ev0 = 0.0) {
    MasterRequest q;
    q.strength = strength;
    q.exposure_ev = exposure;
    q.regions = {{"highlights", 400.0, 2000.0, ev0}, {"shadows", 0.05, 12.0, 0.0}};
    return q;
}

fs::path fresh_dir(const char* name) {
    const fs::path d = fs::temp_directory_path() / (std::string("rudra-conform-") + name);
    std::error_code ec;
    fs::remove_all(d, ec);
    fs::create_directories(d);
    return d;
}

}  // namespace

TEST(ShotGrades, CutsMakeShotsInOrder) {
    const auto s = shot_grades_from_cuts({48, 12, 48, 0, 500}, 100);
    ASSERT_EQ(s.shots.size(), 3u);
    EXPECT_EQ(s.shots[0].name, "sh010");
    EXPECT_EQ(s.shots[0].first, 0);
    EXPECT_EQ(s.shots[0].last, 11);
    EXPECT_EQ(s.shots[1].first, 12);
    EXPECT_EQ(s.shots[1].last, 47);
    EXPECT_EQ(s.shots[2].name, "sh030");
    EXPECT_EQ(s.shots[2].last, 99);
    EXPECT_TRUE(validate_shot_grades(s, 100));
    EXPECT_EQ(shot_index(s, 0), 0);
    EXPECT_EQ(shot_index(s, 47), 1);
    EXPECT_EQ(shot_index(s, 48), 2);
    EXPECT_EQ(shot_index(s, 100), -1);
}

TEST(ShotGrades, AShotWithoutKeysTakesTheBase) {
    auto s = shot_grades_from_cuts({10}, 20, grade(0.7, 0.5));
    EXPECT_DOUBLE_EQ(grade_at(s, 3).strength, 0.7);
    EXPECT_DOUBLE_EQ(grade_at(s, 15).exposure_ev, 0.5);
}

TEST(ShotGrades, KeysInterpolateTheContinuousSettings) {
    auto s = shot_grades_from_cuts({}, 100);
    ASSERT_TRUE(set_key(s, 10, grade(1.0, 0.0, 0.0)));
    ASSERT_TRUE(set_key(s, 30, grade(0.0, 2.0, 1.0)));
    EXPECT_DOUBLE_EQ(grade_at(s, 0).strength, 1.0);    // before the first key: held
    EXPECT_DOUBLE_EQ(grade_at(s, 10).strength, 1.0);
    const auto mid = grade_at(s, 20);
    EXPECT_DOUBLE_EQ(mid.strength, 0.5);
    EXPECT_DOUBLE_EQ(mid.exposure_ev, 1.0);
    EXPECT_DOUBLE_EQ(mid.regions[0].ev, 0.5);
    EXPECT_DOUBLE_EQ(grade_at(s, 25).exposure_ev, 1.5);
    EXPECT_DOUBLE_EQ(grade_at(s, 90).exposure_ev, 2.0);   // after the last: held
}

TEST(ShotGrades, DiscreteSettingsStepAtTheLaterKey) {
    auto s = shot_grades_from_cuts({}, 10);
    MasterRequest a = grade(1.0, 0.0), b = grade(1.0, 0.0);
    a.recovery_mode = "all", b.recovery_mode = "highlights";
    a.source_curve = "bt1886", b.source_curve = "aces";
    b.clip_fill = false;
    ASSERT_TRUE(set_key(s, 0, a));
    ASSERT_TRUE(set_key(s, 8, b));
    EXPECT_EQ(grade_at(s, 7).recovery_mode, "all");
    EXPECT_EQ(grade_at(s, 7).source_curve, "bt1886");
    EXPECT_TRUE(grade_at(s, 7).clip_fill);
    EXPECT_EQ(grade_at(s, 8).recovery_mode, "highlights");
    EXPECT_FALSE(grade_at(s, 8).clip_fill);
}

TEST(ShotGrades, KeysNeverReachAcrossACut) {
    auto s = shot_grades_from_cuts({50}, 100, grade(0.3, 0.0));
    ASSERT_TRUE(set_key(s, 0, grade(1.0, -1.0)));
    ASSERT_TRUE(set_key(s, 49, grade(1.0, 1.0)));
    ASSERT_TRUE(set_key(s, 99, grade(0.0, 3.0)));
    EXPECT_DOUBLE_EQ(grade_at(s, 49).exposure_ev, 1.0);
    // The first frame of the second shot is its own key's, held back to the cut.
    EXPECT_DOUBLE_EQ(grade_at(s, 50).exposure_ev, 3.0);
    EXPECT_DOUBLE_EQ(grade_at(s, 75).strength, 0.0);
}

TEST(ShotGrades, BandsThatDifferStepRatherThanBlend) {
    MasterRequest a = grade(1.0, 0.0, 0.0), b = grade(1.0, 0.0, 2.0);
    b.regions.pop_back();
    const auto m = blend_grades(a, b, 0.5);
    EXPECT_EQ(m.regions.size(), 2u);
    EXPECT_DOUBLE_EQ(m.regions[0].ev, 0.0);
    MasterRequest c = a, d = a;
    c.calibration = {{40, 1.0}, {128, 20.0}};
    d.calibration = {{40, 3.0}, {128, 60.0}};
    EXPECT_DOUBLE_EQ(blend_grades(c, d, 0.5).calibration[1].nits, 40.0);
    d.calibration[1].code = 130;
    EXPECT_DOUBLE_EQ(blend_grades(c, d, 0.5).calibration[1].nits, 20.0);
}

TEST(ShotGrades, KeysCanBeReplacedAndRemoved) {
    auto s = shot_grades_from_cuts({5}, 10);
    ASSERT_TRUE(set_key(s, 2, grade(0.5, 0.0)));
    ASSERT_TRUE(set_key(s, 2, grade(0.25, 0.0)));
    EXPECT_EQ(s.shots[0].keys.size(), 1u);
    EXPECT_DOUBLE_EQ(grade_at(s, 2).strength, 0.25);
    EXPECT_FALSE(set_key(s, 10, grade(1.0, 0.0)));   // outside every shot
    EXPECT_TRUE(remove_key(s, 2));
    EXPECT_FALSE(remove_key(s, 2));
    EXPECT_TRUE(s.shots[0].keys.empty());
}

TEST(ShotGrades, ValidationRefusesBrokenTimelines) {
    ShotGrades s;
    s.shots = {{"a", 0, 9, "manual", {}}, {"b", 5, 20, "manual", {}}};
    EXPECT_FALSE(validate_shot_grades(s));
    s.shots[1].first = 10;
    EXPECT_TRUE(validate_shot_grades(s));
    EXPECT_FALSE(validate_shot_grades(s, 15));   // runs past the end
    s.shots[0].keys = {{3, {}}, {3, {}}};
    EXPECT_FALSE(validate_shot_grades(s));
    s.shots[0].keys = {{12, {}}};
    EXPECT_FALSE(validate_shot_grades(s));
}

TEST(ShotGrades, TheFileRoundTripsWithRelativeMasks) {
    const fs::path dir = fresh_dir("roundtrip");
    auto s = shot_grades_from_cuts({24}, 48, grade(0.8, 0.0));
    s.shots[1].source = "edl";
    MasterRequest k = grade(0.4, 1.25, 0.5);
    k.source_curve = "aces";
    k.masks_file = (dir / "masks" / "sh020.masks.png").string();
    ASSERT_TRUE(set_key(s, 30, k));
    const fs::path file = dir / "programme.shots.json";
    ASSERT_TRUE(save_shot_grades(file, s));
    std::ifstream in(file);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_NE(text.find("\"masks/sh020.masks.png\""), std::string::npos) << text;
    // The masks file is missing, so the grade with it cannot load; without it, it does.
    auto missing = load_shot_grades(file);
    EXPECT_FALSE(missing);
    s.shots[1].keys[0].grade.masks_file.clear();
    ASSERT_TRUE(save_shot_grades(file, s));
    auto back = load_shot_grades(file);
    ASSERT_TRUE(back) << back.error().message << " " << back.error().detail;
    ASSERT_EQ(back->shots.size(), 2u);
    EXPECT_EQ(back->shots[1].source, "edl");
    EXPECT_DOUBLE_EQ(back->base.strength, 0.8);
    ASSERT_EQ(back->shots[1].keys.size(), 1u);
    EXPECT_EQ(back->shots[1].keys[0].frame, 30);
    EXPECT_DOUBLE_EQ(back->shots[1].keys[0].grade.exposure_ev, 1.25);
    EXPECT_EQ(back->shots[1].keys[0].grade.source_curve, "aces");
    EXPECT_EQ(shot_grades_json(*back, file), shot_grades_json(s, file));
}

TEST(ShotGrades, ParseRefusesOtherFilesAndNewerVersions) {
    EXPECT_FALSE(parse_shot_grades("{}"));
    EXPECT_FALSE(parse_shot_grades("not json"));
    auto newer = parse_shot_grades(R"({"version": 99, "shots": []})");
    ASSERT_FALSE(newer);
    EXPECT_EQ(newer.error().code, ErrorCode::ContractMismatch);
    EXPECT_FALSE(parse_shot_grades(R"({"version": 1, "shots": [{"first": 5, "last": 2}]})"));
}
