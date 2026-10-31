#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "rudra/deliver/master.hpp"
#include "rudra/engine/project.hpp"

using namespace rudra;
namespace {
namespace fs = std::filesystem;
using json = nlohmann::json;
constexpr auto kRead = ProjectReadSupport::HeadroomV1;
class HeadroomProject : public ::testing::Test {
protected:
    fs::path dir, file;
    void SetUp() override {
        dir = fs::temp_directory_path() / ("rudra-headroom-project-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(dir); file = dir / "test.rudra";
    }
    Project candidate() const {
        Project p;
        p.grade.headroom_policy = HeadroomPolicy::Bt1886V1;
        p.source_kind = "movie"; p.sources = {dir / "source.mkv"}; p.frame = 2;
        p.anchor = p.carry_chroma = p.settle_grain = p.grade.clip_fill = false;
        p.grade.source = "bt1886"; p.grade.exposure_ev = 1;
        p.grade.regions = {{"highlights", 400, 2000, -.5}};
        return p;
    }
    std::string read() const { std::ifstream in(file); std::stringstream s; s << in.rdbuf(); return s.str(); }
};
}

TEST_F(HeadroomProject, VersionedRoundtripRequiresAnExplicitCapableReader) {
    auto p = candidate();
    ASSERT_TRUE(save_project(file, p));
    const auto raw = read();
    auto j = json::parse(raw);
    EXPECT_EQ(j["rudra_project"], 2);
    // Older native readers have a maximum version of 1 and refuse before
    // reading settings. This is a format invariant, not an old-binary run.
    EXPECT_GT(j["rudra_project"].get<int>(), 1);
    EXPECT_EQ(j["decode"]["headroom_policy"], "bt1886-headroom-v1");
    EXPECT_EQ(j["deliver"]["settle_grain"], false);
    EXPECT_FALSE(parse_project(raw, file));
    EXPECT_FALSE(load_project(file));
    auto restored = load_project(file, kRead);
    ASSERT_TRUE(restored) << restored.error().message;
    EXPECT_EQ(restored->grade.headroom_policy, p.grade.headroom_policy);
    EXPECT_EQ(restored->grade.exposure_ev, 1);
    EXPECT_EQ(restored->grade.regions[0].ev, -.5);
    EXPECT_FALSE(restored->settle_grain);
    EXPECT_EQ(project_json(*restored, file), raw);
    EXPECT_EQ(parse_headroom_policy("disabled"), HeadroomPolicy::Disabled);
    EXPECT_EQ(parse_headroom_policy("bt1886-headroom-v1"), p.grade.headroom_policy);
    EXPECT_FALSE(parse_headroom_policy("bt1886-headroom-v2"));
}

TEST_F(HeadroomProject, LegacyProjectsStillWriteVersionOne) {
    auto p = project_from_session(Session{});
    auto j = json::parse(project_json(p, file));
    EXPECT_EQ(j["rudra_project"], 1);
    EXPECT_FALSE(j.contains("decode"));
    EXPECT_FALSE(j["deliver"].contains("settle_grain"));
    auto restored = parse_project(j.dump(), file);
    ASSERT_TRUE(restored);
    EXPECT_EQ(restored->grade.headroom_policy, HeadroomPolicy::Disabled);
    EXPECT_TRUE(restored->settle_grain);
    EXPECT_TRUE(restored->grade.clip_fill);
}

TEST_F(HeadroomProject, UnknownMissingAndUnversionedPoliciesFailClosed) {
    const auto valid = json::parse(project_json(candidate(), file));
    for (const json& id : {json("unknown"), json("disabled"), json("bt1886-headroom-v2"), json(nullptr), json(true)}) {
        auto j = valid; j["decode"]["headroom_policy"] = id;
        EXPECT_FALSE(parse_project(j.dump(), file, kRead));
    }
    auto j = valid; j.erase("decode"); EXPECT_FALSE(parse_project(j.dump(), file, kRead));
    j = valid; j["decode"].erase("headroom_policy"); EXPECT_FALSE(parse_project(j.dump(), file, kRead));
    j = valid; j["deliver"].erase("settle_grain"); EXPECT_FALSE(parse_project(j.dump(), file, kRead));
    j = valid; j["rudra_project"] = 1; EXPECT_FALSE(parse_project(j.dump(), file, kRead));
    j = valid; j["headroom_policy"] = "bt1886-headroom-v1"; EXPECT_FALSE(parse_project(j.dump(), file, kRead));
    for (const json& version : {json(0), json(-1), json(2.0), json("2"), json(18446744073709551615ULL)}) {
        j = valid; j["rudra_project"] = version;
        EXPECT_FALSE(parse_project(j.dump(), file, kRead));
    }
}

TEST_F(HeadroomProject, IncompatibleSettingsAreNotClampedOrSavedOverExistingWork) {
    const auto good = candidate();
    ASSERT_TRUE(save_project(file, good));
    const auto before = read();
    auto refused = [&](Project p) {
        EXPECT_FALSE(parse_project(project_json(p, file), file, kRead));
        EXPECT_FALSE(save_project(file, p));
        EXPECT_EQ(read(), before);
        EXPECT_FALSE(fs::exists(file.string() + ".tmp"));
    };
    auto p = good; p.grade.exposure_ev = 3.1; refused(p);
    p = good; p.grade.regions[0].ev = 4.1; refused(p);
    p = good; p.grade.regions[0].low_nits = 0; refused(p);
    p = good; p.grade.strength = 2; refused(p);
    p = good; p.grade.source = "aces"; refused(p);
    p = good; p.grade.clip_fill = true; refused(p);
    p = good; p.use_model = true; refused(p);
    p = good; p.package = "research"; refused(p);
    p = good; p.anchor = true; refused(p);
    p = good; p.carry_chroma = true; refused(p);
    p = good; p.settle_grain = true; refused(p);
    p = good; p.source_kind = "folder"; refused(p);
    p = good; p.sources.clear(); refused(p);
    p = good; p.frame = -1; refused(p);
    p = good; p.grade.calibration = {{128, 50}}; refused(p);
    p = good; p.grade.headroom_policy = static_cast<HeadroomPolicy>(255); refused(p);
}

TEST_F(HeadroomProject, PaintedMasksMustReturnIntactOrLoadingFails) {
    auto p = candidate();
    auto m = std::make_shared<MaskSet>(); m->width = 2; m->height = 1;
    m->planes[0] = std::make_shared<MaskPlane>(MaskPlane{0, 255}); p.grade.masks = m;
    ASSERT_TRUE(save_project(file, p));
    auto back = load_project(file, kRead);
    ASSERT_TRUE(back); ASSERT_TRUE(back->grade.masks);
    EXPECT_EQ(*back->grade.masks, *m);
    auto parsed_only = parse_project(read(), file, kRead);
    ASSERT_TRUE(parsed_only);
    EXPECT_FALSE(save_project(dir / "unloaded.rudra", *parsed_only));
    EXPECT_FALSE(fs::exists(dir / "unloaded.rudra"));
    auto malformed = json::parse(read()); malformed["grade"]["masks"] = true;
    EXPECT_FALSE(parse_project(malformed.dump(), file, kRead));
    const auto masks = project_masks_path(file);
    fs::rename(masks, dir / "retained-masks.png");
    EXPECT_FALSE(load_project(file, kRead));
    std::ofstream(masks) << "invalid mask";
    EXPECT_FALSE(load_project(file, kRead));
    EXPECT_TRUE(fs::exists(file));
}

TEST_F(HeadroomProject, LegacyMasterReaderCannotDiscardThePolicy) {
    EXPECT_FALSE(master_request_from_json(project_json(candidate(), file)));
    for (const auto& text : {
            R"({"headroom_policy":"bt1886-headroom-v1"})", R"({"headroom_policy":null})",
            R"({"decode":{"headroom_policy":"unknown"}})", R"({"decode":{"headroom_policy":true}})", R"({"decode":null})"})
        EXPECT_FALSE(master_request_from_json(text));
    EXPECT_TRUE(master_request_from_json(R"({"headroom_policy":"disabled"})"));
    EXPECT_TRUE(master_request_from_json("{}"));
}
