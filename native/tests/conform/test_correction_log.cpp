// Opt-in correction logging (roadmap 5.2): no consent, no log; a logged frame
// is a training pair the pipeline reads; a withdrawal stops it all.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>

#include <nlohmann/json.hpp>

#include "rudra/core/hdr10.hpp"
#include "rudra/deliver/correction_log.hpp"
#include "rudra/platform/hash.hpp"
#include "rudra/platform/png8.hpp"
#ifdef RUDRA_HAVE_STILL_DECODE
#include "rudra/media/png16.hpp"
#endif

using namespace rudra;
namespace fs = std::filesystem;

namespace {

fs::path fresh(const std::string& name) {
    const fs::path d = fs::temp_directory_path() / ("rudra-correction-" + name);
    std::error_code ec;
    fs::remove_all(d, ec);
    fs::create_directories(d);
    return d;
}

CorrectionConsent consent() {
    CorrectionConsent c;
    c.opted_in = true;
    c.by = "colourist@fxtdstudios.com";
    c.at = "2026-10-20T09:00:00Z";
    c.terms_version = "correction-log-terms-1";
    c.source_id = "fxtd-owned";
    c.rights = "owner";
    return c;
}

fs::path source_file(const fs::path& dir, const std::string& content) {
    const fs::path p = dir / "plate.png";
    std::ofstream(p, std::ios::binary) << content;
    return p;
}

struct Frame {
    SdrImage sdr;
    NitsFrame nits;
};

Frame frame(int h, int w) {
    PlanarBuffer code(3, h, w);
    NitsFrame nits(h, w);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                code.plane(c)[y * w + x] = float((x + y + c) % 255) / 255.0f;
                nits.at(c, y, x) = 0.5 + 40.0 * (x + 1) * (c + 1);   // up to a few thousand nits
            }
    return {SdrImage(std::move(code)), std::move(nits)};
}

std::string read_text(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

MasterRequest analytic() {
    MasterRequest q;
    q.analytic = true;
    q.exposure_ev = 0.5;
    return q;
}

}  // namespace

TEST(CorrectionLog, NothingIsLoggedWithoutAnOptIn) {
    const fs::path dir = fresh("no-consent");
    const auto f = frame(4, 6);
    const fs::path src = source_file(dir, "a");
    auto r = log_correction(dir / "log", f.sdr, f.nits, analytic(), {src, 0, ""});
    ASSERT_FALSE(r);
    CorrectionConsent c = consent();
    c.opted_in = false;
    EXPECT_FALSE(open_correction_log(dir / "log", c));
    c = consent();
    c.terms_version.clear();
    EXPECT_FALSE(open_correction_log(dir / "log", c));
    c = consent();
    c.rights = "probably";
    EXPECT_FALSE(open_correction_log(dir / "log", c));
    EXPECT_FALSE(fs::exists(dir / "log" / "meta"));
}

TEST(CorrectionLog, AnAcceptedFrameBecomesATrainingPair) {
    const fs::path dir = fresh("pair");
    const fs::path log = dir / "log";
    ASSERT_TRUE(open_correction_log(log, consent()));
    const auto sentinel = nlohmann::json::parse(read_text(log / "_ingest_config.json"));
    EXPECT_EQ(sentinel["storage"]["mode"], "pq_10000");
    EXPECT_EQ(sentinel["storage"]["diffuse_white_nits"], 203.0);
    const auto f = frame(4, 6);
    const fs::path src = source_file(dir, "plate bytes");
    auto e = log_correction(log, f.sdr, f.nits, analytic(), {src, 0, "sh010"});
    ASSERT_TRUE(e) << e.error().message << " " << e.error().detail;
    EXPECT_FALSE(e->already_logged);
    EXPECT_TRUE(fs::exists(e->sdr));
    EXPECT_TRUE(fs::exists(e->hdr));
    const auto meta = nlohmann::json::parse(read_text(e->meta));
    EXPECT_EQ(meta["source"]["sha256"], *sha256_file(src));
    EXPECT_EQ(meta["source"]["label"], "sh010");
    EXPECT_EQ(meta["consent_sha256"], *sha256_file(log / "consent.json"));
    EXPECT_EQ(meta["grade"]["exposure_ev"], 0.5);
    EXPECT_EQ(meta["target_kind"], "artist_accepted_grade");
    EXPECT_EQ(meta["sdr"]["sha256"], *sha256_file(e->sdr));
    EXPECT_EQ(meta["hdr"]["sha256"], *sha256_file(e->hdr));
    EXPECT_EQ(meta["hdr"]["clipped_fraction"], 0.0);
    // The same frame, source and grade twice: logged once.
    auto again = log_correction(log, f.sdr, f.nits, analytic(), {src, 0, "sh010"});
    ASSERT_TRUE(again);
    EXPECT_TRUE(again->already_logged);
    EXPECT_EQ(again->id, e->id);
    // Another grade of the same frame is another record, in the same scene and split.
    MasterRequest other = analytic();
    other.exposure_ev = 1.0;
    auto e2 = log_correction(log, f.sdr, f.nits, other, {src, 0, "sh010"});
    ASSERT_TRUE(e2);
    EXPECT_NE(e2->id, e->id);
    EXPECT_EQ(e2->scene_id, e->scene_id);
    EXPECT_EQ(e2->split, e->split);

    const fs::path manifest = dir / "corrections.jsonl";
    auto n = export_correction_manifest(log, manifest);
    ASSERT_TRUE(n) << n.error().message;
    EXPECT_EQ(*n, 2);
    std::ifstream in(manifest);
    int rows = 0;
    for (std::string line; std::getline(in, line);) {
        const auto row = nlohmann::json::parse(line);
        ++rows;
        EXPECT_EQ(row["source_id"], "fxtd-owned");
        EXPECT_EQ(row["sdr_kind"], "real");
        EXPECT_EQ(row["commercial_ok"], true);
        EXPECT_TRUE(fs::path(row["sdr_path"].get<std::string>()).is_absolute());
        EXPECT_EQ(row["hdr_sha256"], *sha256_file(row["hdr_path"].get<std::string>()));
        EXPECT_EQ(row["scene_id"], "correction-" + e->scene_id);
    }
    EXPECT_EQ(rows, 2);
    EXPECT_FALSE(export_correction_manifest(log, manifest));   // never replaced
}

TEST(CorrectionLog, AWithdrawalStopsLoggingAndExport) {
    const fs::path dir = fresh("withdraw");
    const fs::path log = dir / "log";
    ASSERT_TRUE(open_correction_log(log, consent()));
    const auto f = frame(2, 2);
    const fs::path src = source_file(dir, "x");
    ASSERT_TRUE(log_correction(log, f.sdr, f.nits, analytic(), {src, 0, ""}));
    ASSERT_TRUE(withdraw_correction_consent(log, "colourist@fxtdstudios.com", "2026-10-21T10:00:00Z"));
    auto c = read_correction_consent(log);
    ASSERT_TRUE(c);
    EXPECT_TRUE(c->withdrawn());
    MasterRequest other = analytic();
    other.exposure_ev = -1.0;
    auto r = log_correction(log, f.sdr, f.nits, other, {src, 0, ""});
    ASSERT_FALSE(r);
    EXPECT_NE(r.error().message.find("no active consent"), std::string::npos);
    EXPECT_FALSE(export_correction_manifest(log, dir / "m.jsonl"));
    // Opening it again with a new opt-in starts it again; the old record stays.
    ASSERT_TRUE(open_correction_log(log, consent()));
    auto n = export_correction_manifest(log, dir / "m.jsonl");
    ASSERT_TRUE(n);
    EXPECT_EQ(*n, 1);
}

TEST(CorrectionLog, UnknownRightsAreNotCommercial) {
    const fs::path dir = fresh("unknown-rights");
    CorrectionConsent c = consent();
    c.rights = "unknown";
    ASSERT_TRUE(open_correction_log(dir / "log", c));
    const auto f = frame(2, 2);
    ASSERT_TRUE(log_correction(dir / "log", f.sdr, f.nits, analytic(), {source_file(dir, "y"), 0, ""}));
    ASSERT_TRUE(export_correction_manifest(dir / "log", dir / "m.jsonl"));
    std::ifstream in(dir / "m.jsonl");
    std::string line;
    std::getline(in, line);
    EXPECT_EQ(nlohmann::json::parse(line)["commercial_ok"], false);
}

TEST(CorrectionLog, ARecordKeepsTheRightsItWasLoggedUnder) {
    const fs::path dir = fresh("rights-snapshot");
    const fs::path log = dir / "log";
    CorrectionConsent unknown = consent();
    unknown.rights = "unknown", unknown.source_id = "client-plates";
    ASSERT_TRUE(open_correction_log(log, unknown));
    const auto f = frame(2, 2);
    auto first = log_correction(log, f.sdr, f.nits, analytic(), {source_file(dir, "a"), 0, ""});
    ASSERT_TRUE(first);
    // A later consent claims ownership: the earlier record stays unknown.
    CorrectionConsent owned = consent();
    owned.at = "2026-10-22T09:00:00Z";
    ASSERT_TRUE(open_correction_log(log, owned));
    const fs::path src2 = dir / "plate2.png";
    std::ofstream(src2, std::ios::binary) << "b";
    auto second = log_correction(log, f.sdr, f.nits, analytic(), {src2, 0, ""});
    ASSERT_TRUE(second);
    ASSERT_TRUE(export_correction_manifest(log, dir / "m.jsonl"));
    std::ifstream in(dir / "m.jsonl");
    std::map<std::string, nlohmann::json> rows;
    for (std::string line; std::getline(in, line);) {
        auto j = nlohmann::json::parse(line);
        rows[j["asset_id"].get<std::string>()] = j;
    }
    ASSERT_EQ(rows.size(), 2u);
    const auto& a = rows["correction-" + first->id];
    const auto& b = rows["correction-" + second->id];
    EXPECT_EQ(a["commercial_ok"], false);
    EXPECT_EQ(a["source_id"], "client-plates");
    EXPECT_EQ(b["commercial_ok"], true);
    EXPECT_EQ(b["source_id"], "fxtd-owned");
    // Each record's consent is kept, by its hash, after it was replaced.
    for (const auto* r : {&a, &b}) {
        const std::string sha = (*r)["consent_sha256"].get<std::string>();
        ASSERT_FALSE(sha.empty());
        EXPECT_TRUE(fs::is_regular_file(log / "consent_history" / (sha + ".json"))) << sha;
    }
}

TEST(CorrectionLog, ThePngEncoderRefusesAMismatchedBuffer) {
    EXPECT_TRUE(png16_rgb_bytes(2, 2, std::vector<std::uint16_t>(11)).empty());
    EXPECT_TRUE(png16_rgb_bytes(0, 2, {}).empty());
    EXPECT_TRUE(png16_rgb_bytes(-1, 2, std::vector<std::uint16_t>(6)).empty());
    EXPECT_FALSE(png16_rgb_bytes(2, 2, std::vector<std::uint16_t>(12)).empty());
}

TEST(CorrectionLog, OnlyRec709FramesAreLogged) {
    const fs::path dir = fresh("p3");
    ASSERT_TRUE(open_correction_log(dir / "log", consent()));
    const auto f = frame(2, 2);
    MasterRequest q = analytic();
    q.source_space = "p3d65";
    auto r = log_correction(dir / "log", f.sdr, f.nits, q, {source_file(dir, "z"), 0, ""});
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().code, ErrorCode::Unsupported);
}

TEST(CorrectionLog, ThePairDecodesBackToTheAcceptedNits) {
    const fs::path dir = fresh("decode");
    ASSERT_TRUE(open_correction_log(dir / "log", consent()));
    // A neutral frame: Rec.709 to Rec.2020 leaves grey as it is.
    PlanarBuffer code(3, 2, 3, 0.5f);
    NitsFrame nits(2, 3);
    for (double& v : nits.span()) v = 203.0;
    auto e = log_correction(dir / "log", SdrImage(std::move(code)), nits, analytic(), {source_file(dir, "g"), 0, ""});
    ASSERT_TRUE(e);
    // The PNG's own structure: signature, IHDR 16-bit RGB.
    const std::string png = read_text(e->hdr);
    ASSERT_GT(png.size(), 33u);
    EXPECT_EQ(png.substr(1, 3), "PNG");
    EXPECT_EQ(int(static_cast<unsigned char>(png[24])), 16);
    EXPECT_EQ(int(static_cast<unsigned char>(png[25])), 2);
#ifdef RUDRA_HAVE_STILL_DECODE
    auto hdr = read_png16(e->hdr);   // OpenCV reads it as the training loader does
    ASSERT_TRUE(hdr) << hdr.error().message;
    ASSERT_EQ(hdr->width, 3);
    ASSERT_EQ(hdr->channels, 3);
    for (std::uint16_t v : hdr->data) EXPECT_NEAR(pq_eotf(float(v) / 65535.0f), 203.0, 0.5);
    auto sdr = read_png16(e->sdr);
    ASSERT_TRUE(sdr);
    for (std::uint16_t v : sdr->data) EXPECT_EQ(v, 32768);
#endif
}

#if defined(RUDRA_HAVE_STILL_DECODE) && defined(RUDRA_CLI_PATH)
TEST(CorrectionLog, TheCommandLineLogsAStill) {
    const fs::path dir = fresh("cli");
    const std::string cli = "\"" + std::string(RUDRA_CLI_PATH) + "\"";
    const fs::path still = fs::path(RUDRA_GOLDEN_DIR) / "decode" / "png8_rgb.png";
    const fs::path log = dir / "log";
    {
        std::ofstream(dir / "grade.json") << R"({"analytic": true, "exposure_ev": 0.25})";
    }
    // cmd.exe strips the first and last quote of a command that starts with one.
    auto run = [](const std::string& c) {
#ifdef _WIN32
        return std::system(("\"" + c + "\"").c_str());
#else
        return std::system(c.c_str());
#endif
    };
    // Refused before the opt-in.
    EXPECT_NE(run(cli + " correction-log add \"" + log.string() + "\" \"" + still.string() + "\" --params \"" +
                  (dir / "grade.json").string() + "\""), 0);
    ASSERT_EQ(run(cli + " correction-log open \"" + log.string() +
                  "\" --by colourist --terms terms-1 --source-id fxtd-owned --rights owner"), 0);
    ASSERT_EQ(run(cli + " correction-log add \"" + log.string() + "\" \"" + still.string() + "\" --params \"" +
                  (dir / "grade.json").string() + "\" --label hero"), 0);
    ASSERT_EQ(run(cli + " correction-log export \"" + log.string() + "\" --output \"" + (dir / "m.jsonl").string() + "\""), 0);
    std::ifstream in(dir / "m.jsonl");
    std::string line;
    ASSERT_TRUE(std::getline(in, line));
    const auto row = nlohmann::json::parse(line);
    EXPECT_EQ(row["source_id"], "fxtd-owned");
    const auto meta = nlohmann::json::parse(read_text(row["metadata_path"].get<std::string>()));
    EXPECT_EQ(meta["source"]["label"], "hero");
    EXPECT_EQ(meta["grade"]["analytic"], true);
    ASSERT_EQ(run(cli + " correction-log withdraw \"" + log.string() + "\" --by colourist"), 0);
    EXPECT_NE(run(cli + " correction-log export \"" + log.string() + "\" --output \"" + (dir / "m2.jsonl").string() + "\""), 0);
}
#endif
