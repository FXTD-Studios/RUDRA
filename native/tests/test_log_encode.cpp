// ACEScct and ARRI LogC4 masters (9 Oct 2026): the curves against their
// published values, the two gamuts against their published matrices, the
// scene-referred master through encode_master, the encoder command's tags,
// the queue's format list and the export QC for an untagged log file.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "rudra/core/gamut.hpp"
#include "rudra/core/hdr10.hpp"
#include "rudra/core/log_encode.hpp"
#include "rudra/deliver/video_encode.hpp"
#include "rudra/deliver/video_qc.hpp"

using namespace rudra;

namespace {
PlanarBuffer grey(double scene) {   // scene-linear (diffuse white = 1.0) -> network units
    return PlanarBuffer(3, 2, 2, static_cast<float>(scene * 203.0 / 10000.0));
}
bool has_pair(const std::vector<std::string>& v, const std::string& a, const std::string& b) {
    for (std::size_t i = 0; i + 1 < v.size(); ++i)
        if (v[i] == a && v[i + 1] == b) return true;
    return false;
}
}  // namespace

TEST(LogEncode, AcesCctIsS2016001) {
    EXPECT_NEAR(acescct_encode(0.18), 0.4135884024924423, 1e-12);
    EXPECT_NEAR(acescct_encode(1.0), 0.5547945205479452, 1e-12);
    EXPECT_NEAR(acescct_encode(0.0), 0.0729055341958355, 1e-15);
    // The linear toe meets the log at the break, both ways.
    EXPECT_NEAR(acescct_encode(0.0078125), 0.155251141552511, 1e-9);
    EXPECT_NEAR(acescct_decode(0.155251141552511), 0.0078125, 1e-9);
    EXPECT_NEAR(acescct_decode(1.0), 222.8609442038076, 1e-9);
    for (double x : {-0.01, 0.0, 0.001, 0.0078125, 0.05, 0.18, 1.0, 16.0, 200.0})
        EXPECT_NEAR(acescct_decode(acescct_encode(x)), x, 1e-12 * std::max(1.0, x)) << x;
}

TEST(LogEncode, LogC4IsArrisSpecification) {
    EXPECT_NEAR(logc4_encode(0.18), 0.2783958365482653, 1e-12);   // ARRI: 18% grey at 0.2784
    EXPECT_NEAR(logc4_encode(0.0), 95.0 / 1023.0, 1e-12);          // black at code 95 of 1023
    EXPECT_NEAR(logc4_decode(1.0), 469.8, 1e-9);                    // the top of the range
    for (double x : {-0.0180, -0.01, 0.0, 0.001, 0.18, 1.0, 40.0, 400.0})
        EXPECT_NEAR(logc4_decode(logc4_encode(x)), x, 1e-11 * std::max(1.0, x)) << x;
}

TEST(LogEncode, GamutsAreThePublishedOnes) {
    // ACES AP1 -> XYZ (S-2014-004) and ARRI Wide Gamut 4 -> XYZ (ARRI, 2022).
    const Mat3 ap1{{{0.6624541811, 0.1340042065, 0.1561876870},
                    {0.2722287168, 0.6740817658, 0.0536895174},
                    {-0.0055746495, 0.0040607335, 1.0103391003}}};
    const Mat3 awg4{{{0.704858320407232, 0.129760295170463, 0.115837311473976},
                     {0.254524176404027, 0.781477732712002, -0.036001909116029},
                     {0.0, 0.0, 1.089057750759878}}};
    const Mat3 got1 = normalized_primary_matrix(Primaries::Ap1), got4 = normalized_primary_matrix(Primaries::Awg4);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(got1[i][j], ap1[i][j], 2e-9) << i << j;
            EXPECT_NEAR(got4[i][j], awg4[i][j], 1e-9) << i << j;
        }
    // White stays white into both (Bradford D65 -> D60 for AP1).
    for (Primaries p : {Primaries::Ap1, Primaries::Awg4}) {
        const Mat3 m = rgb_to_rgb_matrix(Primaries::Rec2020, p);
        for (int i = 0; i < 3; ++i) EXPECT_NEAR(m[i][0] + m[i][1] + m[i][2], 1.0, 1e-9);
    }
}

TEST(LogEncode, MasterIsSceneReferred) {
    const PlanarBuffer g = grey(0.18);
    const std::vector<std::pair<std::string, double>> cases = {{"prores4444_logc4", 0.2783958365482653},
                                                               {"prores422hq_logc4", 0.2783958365482653},
                                                               {"prores4444_acescct", 0.4135884024924423},
                                                               {"prores422hq_acescct", 0.4135884024924423}};
    for (const auto& [fmt, want] : cases) {
        SCOPED_TRACE(fmt);
        // Peak and knee are ignored: a 100-nit peak would crush a graded file, not a log one.
        auto r = encode_master(g, fmt, 100.0, 50.0);
        ASSERT_TRUE(r);
        for (float v : r->first.span()) EXPECT_NEAR(v, want, 2e-6);
        for (float v : r->second.span()) EXPECT_NEAR(v, 0.18 * 203.0, 1e-3);   // the scene's nits, for the report
    }
    // Above the curve's top the code clamps at 1.
    auto hot = encode_master(grey(1000.0), "prores4444_acescct");
    ASSERT_TRUE(hot);
    for (float v : hot->first.span()) EXPECT_FLOAT_EQ(v, 1.0f);
}

TEST(LogEncode, ProfilesAndTheEncoderCommand) {
    EXPECT_EQ(delivery_profiles().size(), 5u);   // the Python's five, untouched
    ASSERT_EQ(log_delivery_profiles().size(), 4u);
    const DeliveryProfile* p = find_delivery_profile("prores4444_logc4");
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->log, "logc4");
    EXPECT_EQ(p->tag, "ap4h");
    EXPECT_EQ(find_delivery_profile("prores422hq_acescct")->tag, "apch");
    EXPECT_EQ(find_delivery_profile("hdr10")->log, "");
    EXPECT_EQ(find_delivery_profile("prores4444_slog3"), nullptr);

    VideoEncodeRequest r;
    r.format = "prores4444_logc4";
    r.audio = "none";
    const VideoClock clock{"24", 24, 0.0, 1.0, 0.0};
    auto cmd = encode_command("ffmpeg", r, clock, "spool", "in.mp4", "out.mov", 100, 50);
    ASSERT_TRUE(cmd);
    EXPECT_TRUE(has_pair(*cmd, "-color_primaries", "unknown"));
    EXPECT_TRUE(has_pair(*cmd, "-color_trc", "unknown"));
    EXPECT_TRUE(has_pair(*cmd, "-colorspace", "bt2020nc"));
    EXPECT_TRUE(has_pair(*cmd, "-profile:v", "4"));
    const auto vf = std::find(cmd->begin(), cmd->end(), "-vf");
    ASSERT_NE(vf, cmd->end());
    EXPECT_NE((vf + 1)->find("transferin=linear"), std::string::npos) << *(vf + 1);
    EXPECT_NE((vf + 1)->find(":transfer=linear"), std::string::npos) << *(vf + 1);
    // A graded file is unchanged.
    r.format = "prores4444";
    auto pq = encode_command("ffmpeg", r, clock, "spool", "in.mp4", "out.mov", 100, 50);
    ASSERT_TRUE(pq);
    EXPECT_TRUE(has_pair(*pq, "-color_primaries", "bt2020"));
    EXPECT_TRUE(has_pair(*pq, "-color_trc", "smpte2084"));
}

TEST(LogEncode, QcTakesAnUntaggedLogFile) {
    const std::string src = R"({"streams":[{"codec_type":"video","width":4,"height":2}]})";
    const std::string frames = R"({"frames":[{"best_effort_timestamp_time":"0.000000"}]})";
    const VideoClock clock{"24", 1, 0.0, 1.0 / 24.0, 0.0};
    VideoQcRequest q;
    q.audio_mode = "none";
    q.format = "prores4444_logc4";
    auto run = [&](const std::string& extra) {
        const std::string out = R"({"streams":[{"codec_type":"video","codec_name":"prores","codec_tag_string":"ap4h",)"
                                R"("pix_fmt":"yuv444p12le","color_space":"bt2020nc","width":4,"height":2,"time_base":"1/24")" +
                                extra + "}]}";
        return evaluate_video_qc(src, clock, out, frames, q);
    };
    auto ok = run("");
    ASSERT_TRUE(ok) << ok.error().message;
    EXPECT_TRUE(ok->errors.empty()) << ok->errors.front();
    auto unknown = run(R"(,"color_primaries":"unknown","color_transfer":"unknown")");
    ASSERT_TRUE(unknown);
    EXPECT_TRUE(unknown->errors.empty());
    // Tagged as if it were PQ: refused.
    auto wrong = run(R"(,"color_primaries":"bt2020","color_transfer":"smpte2084")");
    ASSERT_TRUE(wrong);
    ASSERT_EQ(wrong->errors.size(), 2u);
    EXPECT_EQ(wrong->errors[0], "color_primaries: expected unknown, got bt2020");
}
