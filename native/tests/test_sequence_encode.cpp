// Phase 4 step 8: a finished sequence to one file, against rudra/delivery/video.py
// encode_sequence and `rudra deliver` (tools/emit_sequence_encode_golden.py):
// the 16-bit codes for every target from three primaries with and without the
// shoulder, the ffmpeg command, and live, the file's tags read back and the
// deliver report byte for byte.
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "rudra/core/measure.hpp"
#include "rudra/deliver/sequence_encode.hpp"
#include "rudra/platform/npy.hpp"
#include "rudra/platform/process.hpp"

using namespace rudra;
using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {
const fs::path kDir = fs::path(RUDRA_GOLDEN_DIR) / "sequence_encode";

json index() {
    std::ifstream f(kDir / "index.json");
    std::stringstream ss;
    ss << f.rdbuf();
    return json::parse(ss.str());
}

Primaries primaries_of(const std::string& s) {
    return s == "rec709" ? Primaries::Rec709 : s == "p3d65" ? Primaries::P3D65 : Primaries::Rec2020;
}

// A ProRes MOV states its colour twice: in every frame header (what Resolve
// and FCP read, and what RUDRA checks) and in the container's colr atom (what
// ffprobe reports). Whether the atom can carry BT.2020 and PQ depends on the
// ffmpeg build: 6.1 (Homebrew's ffmpeg@6, Ubuntu 24.04) writes them; the 2025
// gyan.dev builds leave them unspecified, ffprobe says "unknown", and RUDRA
// adds a note. The goldens are recorded on one build (index.json "ffmpeg"), so
// for ProRes either form is right and the frame headers are what must match.
// macOS CI failed on exactly this after the goldens were re-recorded on Windows
// (PR #29, 9 Oct 2026).
const char* const kCompleteProres[][2] = {
    {"color_primaries", "bt2020"}, {"color_transfer", "smpte2084"}, {"color_space", "bt2020nc"}};

bool is_prores(const std::string& target) { return target.rfind("prores", 0) == 0; }

template <class Tags>
bool complete_prores_tags(const Tags& tags) {
    for (const auto& [key, want] : kCompleteProres) {
        bool found = false;
        for (const auto& [k, v] : tags)
            if (k == key) found = v == want;
        if (!found) return false;
    }
    return true;
}

// The golden deliver report with its ProRes colour_tags as a build that
// writes a complete colr atom reports them.
std::string with_complete_colour_tags(std::string report) {
    const auto at = report.find("\"colour_tags\"");
    if (at == std::string::npos) return report;
    const auto end = report.find('}', at);
    std::string block = report.substr(at, end - at);
    for (const auto& [key, want] : kCompleteProres) {
        const std::string from = std::string("\"") + key + "\": \"unknown\"";
        if (const auto i = block.find(from); i != std::string::npos)
            block.replace(i, from.size(), std::string("\"") + key + "\": \"" + want + "\"");
    }
    return report.substr(0, at) + block + report.substr(end);
}
}  // namespace

TEST(SequenceEncode, FrameCodesAreThePythons) {
    const json idx = index();
    for (const auto& c : idx["codes"]) {
        SCOPED_TRACE(c["codes"].get<std::string>());
        auto frame = load_linear_frame(kDir / c["input"].get<std::string>(), c["nits_scale"].get<double>());
        ASSERT_TRUE(frame);
        const SequenceTarget* t = find_sequence_target(c["target"].get<std::string>());
        ASSERT_NE(t, nullptr);
        auto got = encode_sequence_frame(*frame, *t, c["peak_nits"].get<double>(),
                                         primaries_of(c["source_space"].get<std::string>()), c["shoulder"].get<bool>());
        ASSERT_TRUE(got);
        auto want = read_npy(kDir / c["codes"].get<std::string>());
        ASSERT_TRUE(want);
        ASSERT_EQ(want->data.size(), got->size());
        int worst = 0;
        std::size_t off = 0;
        for (std::size_t i = 0; i < got->size(); ++i) {
            const int d = std::abs(int((*got)[i]) - int(want->data[i]));
            worst = std::max(worst, d);
            off += d != 0;
        }
        // numpy's SIMD power (PQ, float32; step 4) and log (HLG, float64) are not
        // correctly rounded; libm's are: a code lands one either side of a rounding
        // boundary now and then, never more.
        EXPECT_LE(worst, 1);
        EXPECT_LE(double(off), 0.05 * double(got->size()));
    }
}

TEST(SequenceEncode, CommandsAreThePythons) {
    const json idx = index();
    for (const auto& r : idx["runs"]) {
        SCOPED_TRACE(r["target"].get<std::string>());
        const SequenceTarget* t = find_sequence_target(r["target"].get<std::string>());
        const auto& rep = r["report"];
        const auto cmd = sequence_encode_command(*t, 128, 72, r["fps"].get<double>(), rep["peak_nits"].get<double>(),
                                                 rep["maxcll"].get<int>(), rep["maxfall"].get<int>(), 0.005,
                                                 "OUTPUT" + t->suffix, r["write_colr"].get<bool>(),
                                                 r["prores_metadata"].get<bool>());
        EXPECT_EQ(json(cmd), r["command"]);
    }
    EXPECT_EQ(expected_tags("av1").error().message, "unknown target 'av1'");
    EXPECT_EQ(tag_of(*expected_tags("hlg"), "color_transfer"), "arib-std-b67");
}

TEST(SequenceEncode, MeasuredLightIsTheReports) {
    // `deliver` measures MaxCLL and MaxFALL after Rec.709 -> Rec.2020 and the
    // shoulder, as the encoder writes it.
    std::vector<FrameStats> stats;
    auto paths = list_linear_frames(kDir / "frames");
    ASSERT_TRUE(paths);
    ASSERT_EQ(paths->size(), 3u);
    for (std::size_t i = 0; i < paths->size(); ++i) {
        auto f = load_linear_frame((*paths)[i], 203.0);
        ASSERT_TRUE(f);
        auto m = deliver_master_frame(std::move(*f), Primaries::Rec709, 1000.0);
        ASSERT_TRUE(m);
        stats.push_back(analyze_frame(*m, int(i)));
    }
    const auto sm = maxcll_maxfall(stats);
    const json idx = index();
    const auto& rep = idx["runs"][0]["report"];
    EXPECT_EQ(sm.maxcll, rep["maxcll"].get<int>());
    EXPECT_EQ(sm.maxfall, rep["maxfall"].get<int>());
    EXPECT_FALSE(list_linear_frames(kDir / "no-such").ok());
}

TEST(SequenceEncode, LiveEncodesCarryTheirTags) {
    if (!find_executable("ffmpeg") || !find_executable("ffprobe")) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    const fs::path work = fs::temp_directory_path() / "rudra-sequence-encode-test";
    fs::remove_all(work);
    const json idx = index();
    for (const auto& r : idx["runs"]) {
        SCOPED_TRACE(r["target"].get<std::string>());
        auto paths = list_linear_frames(kDir / "frames");
        ASSERT_TRUE(paths);
        std::size_t k = 0;
        SequenceEncodeOptions o;
        o.target = r["target"].get<std::string>();
        o.fps = r["fps"].get<double>();
        o.source = Primaries::Rec709;
        o.maxcll = r["report"]["maxcll"].get<int>();
        o.maxfall = r["report"]["maxfall"].get<int>();
        o.shoulder = false;
        std::string notes;
        o.note = [&](const std::string& n) { notes += n; };
        auto out = encode_sequence(
            [&]() -> std::optional<NitsFrame> {
                if (k >= paths->size()) return std::nullopt;
                auto f = load_linear_frame((*paths)[k++], 203.0);
                return *shoulder_to_peak(*f, 1000.0);
            },
            work / "shot", o);
        ASSERT_TRUE(out) << out.error().message;
        EXPECT_EQ(out->extension(), find_sequence_target(o.target)->suffix);
        auto tags = colour_tags(*out);
        ASSERT_TRUE(tags);
        const auto frame_tags = prores_frame_tags(*out);
        EXPECT_EQ(frame_tags.has_value(), !r["report"]["prores_frame_tags"].is_null());
        if (frame_tags) {
            for (const auto& [key, value] : *frame_tags)
                EXPECT_EQ(value, r["report"]["prores_frame_tags"][key].get<std::string>()) << key;
        }
        const auto colr = container_colr(*out);
        ASSERT_EQ(colr.has_value(), !r["container_colr"].is_null());
        // The colr atom is the container's second statement of the colour, and
        // what goes into it is the ffmpeg build's: the golden's build left
        // primaries and transfer unspecified there (HEVC too, whose VUI carries
        // them), 6.1 fills them in. Each field is the golden's or the stream's own.
        if (colr) {
            const auto stream = expected_tags(o.target);
            ASSERT_TRUE(stream);
            for (const auto& [key, value] : *colr) {
                const std::string golden = r["container_colr"][key].get<std::string>();
                if (key == "subtype" || value == golden) EXPECT_EQ(value, golden) << key;
                else EXPECT_EQ(value, tag_of(*stream, key)) << key << " (golden: " << golden << ")";
            }
        }
        // ProRes on a build whose colr atom carries BT.2020 and PQ: complete
        // tags and nothing to note (see above).
        if (is_prores(o.target) && complete_prores_tags(*tags)) {
            EXPECT_TRUE(notes.empty()) << notes;
            continue;
        }
        for (const auto& [key, value] : *tags) EXPECT_EQ(value, r["report"]["colour_tags"][key].get<std::string>());
        // The oracle's note, when this ffmpeg writes an incomplete colr atom for
        // a MOV (bt2020 primaries and PQ do not fit its legacy nclc), is the
        // native encode's note too. Old goldens have no "note" field: then none.
        const std::string want_note = r.value("note", std::string());
        if (want_note.empty()) EXPECT_TRUE(notes.empty()) << notes;
        else EXPECT_EQ(notes.substr(0, 5), "note:") << notes;
    }
    // Refusals, in the Python's order and words.
    SequenceEncodeOptions o;
    EXPECT_EQ(encode_sequence([] { return std::optional<NitsFrame>(); }, work / "x", o).error().message, "no frames to encode");
    o.target = "av1";
    EXPECT_EQ(encode_sequence([] { return std::optional<NitsFrame>(NitsFrame(72, 128)); }, work / "x", o).error().message,
              "unknown target 'av1'. Choose from: hdr10, hlg, prores422hq, prores4444");
    o.target = "prores422hq";
    int n = 0;
    auto r = encode_sequence(
        [&]() -> std::optional<NitsFrame> { return n++ == 0 ? NitsFrame(72, 128) : NitsFrame(64, 128); }, work / "y", o);
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().message, "frame 1 is 128x64, the first was 128x72. A sequence must not change size.");
    fs::remove_all(work);
}

TEST(SequenceEncode, DeliverReportIsThePythons) {
    if (!find_executable("ffmpeg") || !find_executable("ffprobe")) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
#ifdef RUDRA_CLI_PATH
    const fs::path work = fs::temp_directory_path() / "rudra-deliver-cli-test";
    fs::remove_all(work);
    fs::create_directories(work);
    const json idx = index();
    for (const auto& r : idx["runs"]) {
        SCOPED_TRACE(r["target"].get<std::string>());
        auto p = run_process({RUDRA_CLI_PATH, "deliver", (kDir / "frames").string(), "--output", (work / "shot").string(),
                              "--target", r["target"].get<std::string>(), "--fps", "23.976", "--source-space", "rec709"});
        ASSERT_TRUE(p);
        ASSERT_EQ(p->exit_code, 0) << p->err;
        std::string got = p->out;
        // Windows writes the console in text mode: \r\n for the oracle's \n.
        for (std::size_t at = got.find("\r\n"); at != std::string::npos; at = got.find("\r\n", at)) got.erase(at, 1);
        const std::string file = (work / ("shot" + find_sequence_target(r["target"].get<std::string>())->suffix)).string();
        std::string escaped;
        for (char c : file) escaped += c == '\\' ? std::string("\\\\") : std::string(1, c);
        got.replace(got.find(escaped), escaped.size(), "OUTPUT" + find_sequence_target(r["target"].get<std::string>())->suffix);
        const std::string want = r["report_json"].get<std::string>() + "\n";
        // ProRes: the golden's build or one whose colr atom is complete (see above).
        if (is_prores(r["target"].get<std::string>()) && got != want)
            EXPECT_EQ(got, with_complete_colour_tags(want));
        else
            EXPECT_EQ(got, want);
    }
    fs::remove_all(work);
#else
    GTEST_SKIP() << "rudra-native is not built";
#endif
}
