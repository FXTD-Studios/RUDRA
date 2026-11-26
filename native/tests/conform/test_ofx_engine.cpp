// The OFX plug-in's learned ACES 2.0 route (phase 2, 25 Nov 2026), against a stand-in engine
// (ofx_fake_engine.cpp) that speaks aces-v2-serve's protocol: the frame goes out as a 16-bit
// TIFF, the master comes back as ACEScg planes, the plug-in puts it in the output.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

#include "engine_client.hpp"
#include "ofx_process.hpp"
#include "rudra/core/hdr10.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {

constexpr int kW = 16, kH = 9;

void set_env(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

// A config file pointing at the stand-in engine and a package folder with a manifest.
fs::path fake_config() {
    const fs::path dir = fs::temp_directory_path() / "rudra_ofx_engine_test";
    fs::create_directories(dir / "RUDRA_ACES_2");
    std::ofstream(dir / "RUDRA_ACES_2" / "manifest.json") << "{}";
    const fs::path cfg = dir / "ofx_engine.json";
    std::ofstream(cfg) << "{\"cli\": " << fs::path(RUDRA_OFX_FAKE_ENGINE).generic_string().insert(0, "\"").append("\"")
                       << ", \"package\": \"" << (dir / "RUDRA_ACES_2").generic_string() << "\", \"runtime\": \"libtorch\", \"device\": \"cpu\"}";
    return cfg;
}

std::vector<float> grey(float code) {
    std::vector<float> p(std::size_t(kW) * kH * 4);
    for (std::size_t i = 0; i < std::size_t(kW) * kH; ++i) p[i * 4] = p[i * 4 + 1] = p[i * 4 + 2] = code, p[i * 4 + 3] = 1.0f;
    return p;
}

class OfxEngine : public ::testing::Test {
protected:
    void SetUp() override {
        ofx_engine::stop_engine();
        set_env("RUDRA_OFX_FAKE_ENGINE", "");
        set_env("RUDRA_OFX_ENGINE_CONFIG", fake_config().string());
    }
    void TearDown() override { ofx_engine::stop_engine(); }
};

}  // namespace

TEST_F(OfxEngine, TheLearnedRouteIsTheEnginesMaster) {
    for (float code : {0.2f, 0.5f, 0.9f}) {
        const auto in = grey(code);
        std::vector<float> out(in.size());
        OfxGrade g;
        g.route = "learned", g.output = "linear_acescg";
        ASSERT_TRUE(render_sdr_to_hdr(g, in.data(), kW * 4, out.data(), kW * 4, kW, kH));
        // The 16-bit frame the engine read: code rounded to 1/65535.
        const float want = 2.0f * std::pow(std::round(code * 65535.0f) / 65535.0f, 2.4f);
        EXPECT_NEAR(out[0], want, want * 1e-5f) << code;
        EXPECT_EQ(out[3], 1.0f);
    }
}

TEST_F(OfxEngine, ExposureAndOutputApplyToTheLearnedMaster) {
    const auto in = grey(0.5f);
    std::vector<float> lin(in.size()), up(in.size()), pq(in.size());
    OfxGrade g;
    g.route = "learned", g.output = "linear_acescg";
    ASSERT_TRUE(render_sdr_to_hdr(g, in.data(), kW * 4, lin.data(), kW * 4, kW, kH));
    g.exposure_ev = 1.0;
    ASSERT_TRUE(render_sdr_to_hdr(g, in.data(), kW * 4, up.data(), kW * 4, kW, kH));
    EXPECT_NEAR(up[1], 2.0f * lin[1], lin[1] * 1e-5f);
    g.exposure_ev = 0.0, g.output = "pq_rec2020";
    ASSERT_TRUE(render_sdr_to_hdr(g, in.data(), kW * 4, pq.data(), kW * 4, kW, kH));
    // A grey stays grey from ACEScg to Rec.2020: the same light, PQ-encoded.
    EXPECT_NEAR(pq_eotf(pq[1]), lin[1] * 203.0f, lin[1] * 203.0f * 2e-3f);
}

TEST_F(OfxEngine, StrengthIsTheEnginesTrim) {
    const auto in = grey(0.5f);
    std::vector<float> full(in.size()), half(in.size());
    OfxGrade g;
    g.route = "learned", g.output = "linear_acescg";
    ASSERT_TRUE(render_sdr_to_hdr(g, in.data(), kW * 4, full.data(), kW * 4, kW, kH));
    g.strength = 0.5;
    ASSERT_TRUE(render_sdr_to_hdr(g, in.data(), kW * 4, half.data(), kW * 4, kW, kH));
    EXPECT_NEAR(half[0], 0.5f * full[0], full[0] * 1e-5f);
}

TEST_F(OfxEngine, NoEngineSaysWhatIsMissing) {
    set_env("RUDRA_OFX_ENGINE_CONFIG", (fs::temp_directory_path() / "rudra_ofx_no_such_config.json").string());
    const auto in = grey(0.5f);
    std::vector<float> out(in.size());
    OfxGrade g;
    g.route = "learned";
    const auto r = render_sdr_to_hdr(g, in.data(), kW * 4, out.data(), kW * 4, kW, kH);
    ASSERT_FALSE(r);
    EXPECT_NE(r.error().message.find("learned route needs RUDRA's engine"), std::string::npos) << r.error().message;
}

TEST_F(OfxEngine, AnEngineThatCannotLoadSaysSo) {
    set_env("RUDRA_OFX_FAKE_ENGINE", "error");
    const auto in = grey(0.5f);
    std::vector<float> out(in.size());
    OfxGrade g;
    g.route = "learned";
    const auto r = render_sdr_to_hdr(g, in.data(), kW * 4, out.data(), kW * 4, kW, kH);
    ASSERT_FALSE(r);
    EXPECT_NE(r.error().message.find("could not load the model"), std::string::npos) << r.error().message;
}

TEST_F(OfxEngine, TheAnalyticRouteNeedsNoEngine) {
    set_env("RUDRA_OFX_ENGINE_CONFIG", (fs::temp_directory_path() / "rudra_ofx_no_such_config.json").string());
    const auto in = grey(0.5f);
    std::vector<float> out(in.size());
    OfxGrade g;   // route "analytic"
    EXPECT_TRUE(render_sdr_to_hdr(g, in.data(), kW * 4, out.data(), kW * 4, kW, kH));
}

// Phase 3 (26 Nov 2026): the luminance limit goes to the engine with the frame (the stand-in halves a
// limited master, so the test sees it arrive).
TEST_F(OfxEngine, TheLuminanceLimitGoesToTheEngine) {
    const auto in = grey(0.5f);
    std::vector<float> plain(in.size()), limited(in.size());
    rudra_ofx::Settings s;
    s.route = "learned", s.output = "linear_acescg";
    ASSERT_EQ(rudra_ofx::render(s, in.data(), kW * 4, plain.data(), kW * 4, kW, kH).error, "");
    s.limit = "highlights", s.highlights_from = 0.4;
    ASSERT_EQ(rudra_ofx::render(s, in.data(), kW * 4, limited.data(), kW * 4, kW, kH).error, "");
    EXPECT_NEAR(limited[0], 0.5f * plain[0], plain[0] * 1e-5f);
    EXPECT_GT(plain[0], 0.0f);
}
