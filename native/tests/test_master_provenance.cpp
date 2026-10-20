// Roadmap 1.3 (16 Oct 2026): every master records the build that wrote it and
// what it was made from: rudra:appVersion and rudra:buildCommit always, the
// source file's SHA-256 when known, the checkpoint's when a model ran, and
// whether the network tiled (rudra:tiled was hard-coded "False"). The sidecar
// carries the same keys.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "rudra/deliver/exr.hpp"
#include "rudra/deliver/master.hpp"
#include "rudra/platform/hash.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {

fs::path fresh(const std::string& name) {
    const fs::path root = fs::temp_directory_path() / ("rudra-provenance-" + name);
    fs::remove_all(root);
    fs::create_directories(root);
    return root;
}

// The ACES writer packs the provenance into one JSON attribute, rudra:provenance;
// the linear writer stores each key as its own attribute.
std::string attr(const ExrImage& e, const std::string& name) {
    for (const auto& [k, v] : e.attributes) {
        std::string s(v.begin(), v.end());
        while (!s.empty() && s.back() == '\0') s.pop_back();
        if (k == name) return s;
        if (k == "rudra:provenance") {
            const auto j = nlohmann::json::parse(s.substr(s.find('{')));
            if (j.contains(name)) return j.at(name).get<std::string>();
        }
    }
    return "<absent>";
}

struct Written {
    ExrImage exr;
    nlohmann::json side;
};

Written master(const fs::path& root, const MasterRequest& q) {
    PlanarBuffer sdr(3, 12, 16, 0.4f), residual(3, 12, 16, 0.0f), hi(1, 12, 16, 0.0f), sh(1, 12, 16, 0.0f);
    const SdrImage image{std::move(sdr)};
    const Fields fields{std::move(residual), std::move(hi), std::move(sh)};
    const fs::path out = root / "m.exr";
    auto r = write_master(image, 8, fields, FrameScalars{}, kAnalyticConstants, q, out);
    EXPECT_TRUE(r) << (r ? "" : r.error().message);
    auto e = read_exr(out);
    EXPECT_TRUE(e) << (e ? "" : e.error().message);
    std::ifstream f(fs::path(out).replace_extension(".json"));
    return {e ? *e : ExrImage{}, nlohmann::json::parse(f)};
}

}  // namespace

TEST(MasterProvenance, BuildIsAlwaysRecorded) {
    const fs::path root = fresh("build");
    MasterRequest q;
    q.analytic = true;
    const auto w = master(root, q);
    const std::string version = attr(w.exr, "rudra:appVersion"), commit = attr(w.exr, "rudra:buildCommit");
    EXPECT_NE(version, "<absent>");
    EXPECT_FALSE(version.empty());
    EXPECT_NE(commit, "<absent>");
    EXPECT_FALSE(commit.empty());
    EXPECT_EQ(w.side.at("app_version"), version);
    EXPECT_EQ(w.side.at("build_commit"), commit);
    // Nothing claimed that is not known.
    EXPECT_EQ(attr(w.exr, "rudra:sourceSha256"), "<absent>");
    EXPECT_EQ(attr(w.exr, "rudra:checkpointSha256"), "<absent>");
    EXPECT_FALSE(w.side.contains("source_sha256"));
    EXPECT_FALSE(w.side.contains("checkpoint_sha256"));
    EXPECT_EQ(attr(w.exr, "rudra:tiled"), "False");
    EXPECT_EQ(w.side.at("tiled"), false);
}

TEST(MasterProvenance, SourceCheckpointAndTiling) {
    const fs::path root = fresh("source");
    const fs::path src = root / "plate.png";
    { std::ofstream(src, std::ios::binary) << "not really a png, only its bytes are hashed"; }
    MasterRequest q;
    q.checkpoint = "sdr2hdr_v8";
    q.checkpoint_sha256 = std::string(64, 'a');
    q.source_file = src;
    q.tiled = true;
    const auto w = master(root, q);
    const auto want = sha256_file(src);
    ASSERT_TRUE(want);
    EXPECT_EQ(attr(w.exr, "rudra:sourceSha256"), *want);
    EXPECT_EQ(w.side.at("source_sha256"), *want);
    EXPECT_EQ(attr(w.exr, "rudra:checkpointSha256"), q.checkpoint_sha256);
    EXPECT_EQ(w.side.at("checkpoint_sha256"), q.checkpoint_sha256);
    EXPECT_EQ(attr(w.exr, "rudra:tiled"), "True");
    EXPECT_EQ(w.side.at("tiled"), true);
}

TEST(MasterProvenance, AMissingSourceIsAnErrorNotASilentGap) {
    const fs::path root = fresh("missing");
    MasterRequest q;
    q.analytic = true;
    q.source_file = root / "gone.png";
    PlanarBuffer sdr(3, 4, 4, 0.4f), residual(3, 4, 4, 0.0f), hi(1, 4, 4, 0.0f), sh(1, 4, 4, 0.0f);
    auto r = write_master(SdrImage{std::move(sdr)}, 8, Fields{std::move(residual), std::move(hi), std::move(sh)},
                          FrameScalars{}, kAnalyticConstants, q, root / "m.exr");
    EXPECT_FALSE(r);
}

TEST(MasterProvenance, LinearContainerToo) {
    const fs::path root = fresh("linear");
    MasterRequest q;
    q.analytic = true;
    q.container = "linear";
    const auto w = master(root, q);
    EXPECT_NE(attr(w.exr, "rudra:appVersion"), "<absent>");
    EXPECT_NE(attr(w.exr, "rudra:buildCommit"), "<absent>");
}

TEST(MasterProvenance, NotInTheRequestJson) {
    MasterRequest q;
    q.checkpoint_sha256 = "abc";
    q.source_file = "x.png";
    q.tiled = true;
    const std::string j = master_request_json(q);
    EXPECT_EQ(j.find("checkpoint_sha256"), std::string::npos);
    EXPECT_EQ(j.find("source_file"), std::string::npos);
    EXPECT_EQ(j.find("tiled"), std::string::npos);
}
