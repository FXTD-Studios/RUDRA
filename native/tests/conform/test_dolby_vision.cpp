// Dolby Vision from RUDRA itself (roadmap 5.5): the RPU is byte for byte what
// dovi_tool 2.3.1 generates from the same config (tests/golden/dolby_vision),
// and the XML carries the same per-shot metadata.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>

#include "rudra/deliver/dolby_vision.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {

const fs::path kGolden = fs::path(RUDRA_GOLDEN_DIR) / "dolby_vision";

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string config(int max_display, int min_display, int cll = 500, int fall = 100) {
    char b[1024];
    std::snprintf(b, sizeof b,
                  R"({"cm_version": "V40", "profile": "8.1", "length": 3,
                      "level6": {"max_display_mastering_luminance": %d, "min_display_mastering_luminance": %d,
                                 "max_content_light_level": %d, "max_frame_average_light_level": %d},
                      "shots": [{"start": 0, "duration": 2, "metadata_blocks": [{"Level1": {"min_pq": 0, "max_pq": 3000, "avg_pq": 2000}}]},
                                {"start": 2, "duration": 1, "metadata_blocks": [{"Level1": {"min_pq": 12, "max_pq": 2081, "avg_pq": 1229}}]}]})",
                  max_display, min_display, cll, fall);
    return b;
}

}  // namespace

class DolbyVisionGolden : public ::testing::TestWithParam<const char*> {};

TEST_P(DolbyVisionGolden, TheRpuIsWhatDoviToolGenerates) {
    const std::string name = GetParam();
    auto m = parse_dovi_generate(read_file(kGolden / (name + ".json")));
    ASSERT_TRUE(m) << m.error().message << " " << m.error().detail;
    auto rpu = dovi_rpu_file(*m);
    ASSERT_TRUE(rpu) << rpu.error().message;
    const std::string want = read_file(kGolden / (name + ".rpu"));
    ASSERT_FALSE(want.empty());
    const std::string got(rpu->begin(), rpu->end());
    ASSERT_EQ(got.size(), want.size());
    EXPECT_TRUE(got == want) << "first difference at byte "
                             << std::distance(got.begin(), std::mismatch(got.begin(), got.end(), want.begin()).first);
}

INSTANTIATE_TEST_SUITE_P(Cases, DolbyVisionGolden,
                         ::testing::Values("two_shots", "peak600", "peak2000", "peak4000", "peak10000"));

TEST(DolbyVision, EveryFrameIsOneRpuWithItsShotsL1) {
    auto m = parse_dovi_generate(config(1000, 50));
    ASSERT_TRUE(m);
    auto rpu = dovi_rpu_file(*m);
    ASSERT_TRUE(rpu);
    int starts = 0;
    for (std::size_t i = 0; i + 4 <= rpu->size(); ++i)
        if ((*rpu)[i] == 0 && (*rpu)[i + 1] == 0 && (*rpu)[i + 2] == 0 && (*rpu)[i + 3] == 1) ++starts;
    EXPECT_EQ(starts, 3);
    // A payload ends with its CRC and the trailing byte; the shot's first frame refreshes.
    const auto first = dovi_rpu_payload(*m, m->shots[0], true), second = dovi_rpu_payload(*m, m->shots[0], false);
    EXPECT_EQ(first.front(), 0x19);
    EXPECT_EQ(first.back(), 0x80);
    EXPECT_NE(first, second);
    EXPECT_EQ(first.size(), second.size());
}

TEST(DolbyVision, ConfigsDoviToolRefusesAreRefused) {
    EXPECT_FALSE(parse_dovi_generate(config(10001, 50)));
    EXPECT_FALSE(parse_dovi_generate(config(1000, 10001)));
    EXPECT_FALSE(parse_dovi_generate(config(1000, 50, 10001)));
    EXPECT_FALSE(parse_dovi_generate(config(1000, 50, 500, 10001)));
    EXPECT_TRUE(parse_dovi_generate(config(10000, 10000, 10000, 10000)));
    // What RUDRA does not write is refused, not dropped.
    std::string other = config(1000, 50);
    other.replace(other.find("V40"), 3, "V29");
    EXPECT_FALSE(parse_dovi_generate(other));
    std::string profile5 = config(1000, 50);
    profile5.replace(profile5.find("8.1"), 3, "5");
    EXPECT_FALSE(parse_dovi_generate(profile5));
    std::string level2 = config(1000, 50);
    level2.replace(level2.find(R"({"Level1")"), 0, R"({"Level2": {"target_max_pq": 2081}}, )");
    EXPECT_FALSE(parse_dovi_generate(level2));
    std::string gap = config(1000, 50);
    gap.replace(gap.find(R"("start": 2)"), 10, R"("start": 3)");
    EXPECT_FALSE(parse_dovi_generate(gap));
}

TEST(DolbyVision, MetadataThisWriterDoesNotCarryIsRefused) {
    auto with = [](const std::string& at, const std::string& insert) {
        std::string c = config(1000, 50);
        c.insert(c.find(at), insert);
        return c;
    };
    // Top level, Level 6, a shot, a Level 1 block: an extra key is refused.
    EXPECT_FALSE(parse_dovi_generate(with(R"("length")", R"("level5": {"active_area_left_offset": 0}, )")));
    EXPECT_FALSE(parse_dovi_generate(with(R"("max_content_light_level")", R"("source_primary_index": 0, )")));
    EXPECT_FALSE(parse_dovi_generate(with(R"("duration": 1)", R"("frame_edits": [], )")));
    EXPECT_FALSE(parse_dovi_generate(with(R"("max_pq": 2081)", R"("mid_pq": 1, )")));
    // Two Level 1 blocks in one shot: which one is meant is not known.
    auto two = parse_dovi_generate(
        with(R"({"Level1": {"min_pq": 12)", R"({"Level1": {"min_pq": 0, "max_pq": 1000, "avg_pq": 500}}, )"));
    ASSERT_FALSE(two);
    EXPECT_NE(two.error().message.find("two Level 1"), std::string::npos);
    EXPECT_TRUE(parse_dovi_generate(config(1000, 50)));
}

TEST(DolbyVision, TheXmlCarriesTheSameShots) {
    auto m = parse_dovi_generate(read_file(kGolden / "two_shots.json"));
    ASSERT_TRUE(m);
    auto xml = dovi_metadata_xml(*m, "Trailer & teaser", 24000, 1001, 1920, 1080);
    ASSERT_TRUE(xml) << xml.error().message;
    EXPECT_NE(xml->find("version=\"4.0.2\""), std::string::npos);
    EXPECT_NE(xml->find("<CompositionName>Trailer &amp; teaser</CompositionName>"), std::string::npos);
    EXPECT_NE(xml->find("<n>24000</n>"), std::string::npos);
    EXPECT_NE(xml->find("<MaxCLL>879</MaxCLL>"), std::string::npos);
    EXPECT_NE(xml->find("<PeakBrightness>1000</PeakBrightness>"), std::string::npos);
    EXPECT_NE(xml->find("<MinimumBrightness>0.005</MinimumBrightness>"), std::string::npos);
    // L1 as normalised PQ, min / average / max (dovi_tool reads it back to the same codes).
    EXPECT_NE(xml->find("<ImageCharacter>0.002930 0.420269 0.662515</ImageCharacter>"), std::string::npos);
    EXPECT_NE(xml->find("<In>3</In>"), std::string::npos);
    std::size_t shots = 0;
    for (std::size_t at = 0; (at = xml->find("<Shot>", at)) != std::string::npos; ++at) ++shots;
    EXPECT_EQ(shots, 2u);
    // The same input, the same file (stable ids).
    EXPECT_EQ(*xml, *dovi_metadata_xml(*m, "Trailer & teaser", 24000, 1001, 1920, 1080));
    EXPECT_FALSE(dovi_metadata_xml(*m, "x", 0, 1, 1920, 1080));
}
