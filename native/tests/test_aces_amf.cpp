// core/aces_amf.hpp: AMF v2.0 sidecars. The goldens in tests/golden/amf are
// validated against the Academy's XSD by tests/test_aces_amf_schema.py, so
// "equal to the golden" here means "schema-valid".
#include <gtest/gtest.h>

#include <cstdlib>
#include <map>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

#include "rudra/core/aces_amf.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {
AmfRequest fixed(bool acescg) {
    AmfRequest r;
    r.description = "RUDRA learned ACES 2.0 master";
    r.created_utc = "2026-11-02T10:15:00Z";
    r.amf_uuid = "urn:uuid:11111111-2222-4333-8444-555555555555";
    r.pipeline_uuid = "urn:uuid:66666666-7777-4888-9999-aaaaaaaaaaaa";
    r.acescg = acescg;
    return r;
}
std::map<std::string, AmfRequest> cases() {
    std::map<std::string, AmfRequest> c;
    AmfRequest seq = fixed(true);
    seq.clip_name = "SH010";
    seq.sequence = AmfSequence{"SH010.######.exr", 1001, 1096};
    c["acescg_sequence.amf"] = seq;
    c["ap0_bare.amf"] = fixed(false);
    AmfRequest one = fixed(false);
    one.clip_name = "still <A&B> \"x\"";
    one.file = "still_A&B.exr";
    c["ap0_file_escaped.amf"] = one;
    AmfRequest who = fixed(true);
    who.author_name = "FXTD Studios";
    who.author_email = "studio@example.com";
    c["acescg_author.amf"] = who;
    return c;
}
fs::path golden_dir() { return fs::path(RUDRA_GOLDEN_DIR) / "amf"; }
std::string read(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}
}  // namespace

TEST(AcesAmf, MatchesTheSchemaValidatedGoldens) {
    const bool emit = std::getenv("RUDRA_EMIT_AMF_GOLDEN") != nullptr;
    for (const auto& [name, req] : cases()) {
        const std::string xml = aces_amf_xml(req);
        if (emit) {
            fs::create_directories(golden_dir());
            std::ofstream(golden_dir() / name, std::ios::binary) << xml;
            continue;
        }
        ASSERT_TRUE(fs::exists(golden_dir() / name)) << name;
        EXPECT_EQ(xml, read(golden_dir() / name)) << name;
    }
}

TEST(AcesAmf, TransformsFollowTheMasterSpace) {
    const std::string cg = aces_amf_xml(fixed(true)), ap0 = aces_amf_xml(fixed(false));
    EXPECT_NE(cg.find(kAcesCgToAcesId), std::string::npos);
    EXPECT_EQ(ap0.find("inputTransform"), std::string::npos);   // ACES2065-1 is already ACES
    for (const auto* x : {&cg, &ap0}) {
        EXPECT_NE(x->find(kAcesRec709Bt1886OutputId), std::string::npos);
        EXPECT_NE(x->find("applied=\"false\""), std::string::npos);   // scene-linear: nothing baked in
        EXPECT_EQ(x->find("applied=\"true\""), std::string::npos);
    }
}

TEST(AcesAmf, UuidsAndTimeHaveTheSchemaShape) {
    std::mt19937_64 rng(7);
    const std::regex uuid("urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}");
    for (int i = 0; i < 50; ++i) EXPECT_TRUE(std::regex_match(make_amf_uuid(rng), uuid));
    EXPECT_NE(make_amf_uuid(rng), make_amf_uuid(rng));
    EXPECT_TRUE(std::regex_match(amf_now_utc(), std::regex(R"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)")));
}
