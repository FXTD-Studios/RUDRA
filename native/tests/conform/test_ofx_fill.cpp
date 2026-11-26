// AI fill in the OFX plug-in (phase 3, 26 Nov 2026): Generate, the frames collected as the host renders them,
// the job beside the host (here the stand-in engine started as ofx_job.py: each stored master becomes its
// frame's "fill"), and the stored fill shown from then on, only for the same settings and the same codes.
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "engine_client.hpp"
#include "fill_store.hpp"
#include "ofx_process.hpp"
#include "rudra/deliver/exr.hpp"

using namespace rudra;
namespace fs = std::filesystem;
namespace fill = rudra::ofx_fill;

namespace {

constexpr int kW = 16, kH = 9;

void set_env(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

// A settings file whose AI fill part points at files that exist, its Python the stand-in engine.
fs::path fill_config(const fs::path& d) {
    fs::create_directories(d / "repo" / "rudra" / "genfill");
    std::ofstream(d / "repo" / "rudra" / "genfill" / "ofx_job.py") << "# stand-in\n";
    fs::create_directories(d / "base" / "transformer");
    std::ofstream(d / "base" / "transformer" / "config.json") << "{}";
    std::ofstream(d / "lora.safetensors") << "x";
    const nlohmann::json j = {{"cli", ""},
                              {"package", ""},
                              {"genfill",
                               {{"python", fs::path(RUDRA_OFX_FAKE_ENGINE).generic_string()},
                                {"repo", (d / "repo").generic_string()},
                                {"base", (d / "base").generic_string()},
                                {"lora", (d / "lora.safetensors").generic_string()}}}};
    std::ofstream(d / "ofx_engine.json") << j.dump();
    return d / "ofx_engine.json";
}

// RGBA, top row first: a ramp with a clipped patch, alpha 1. `shift` moves the ramp (other codes).
std::vector<float> pixels(float shift = 0.0f) {
    std::vector<float> p(std::size_t(kW) * kH * 4);
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) {
            float* q = &p[(std::size_t(y) * kW + std::size_t(x)) * 4];
            const bool clipped = x > 10 && y < 4;
            q[0] = clipped ? 1.0f : std::min(1.0f, float(x) / (kW - 1) + shift);
            q[1] = clipped ? 1.0f : std::min(1.0f, float(y) / (kH - 1) + shift);
            q[2] = clipped ? 1.0f : 0.3f;
            q[3] = 1.0f;
        }
    return p;
}

SdrImage codes(float level) { return SdrImage(PlanarBuffer(3, kH, kW, level)); }

fill::FillRequest request() {
    fill::FillRequest r;
    r.shot_id = "test01";
    r.first = 10, r.last = 12;
    r.condition = "bt1886";
    r.master_key = "analytic|bt1886";
    return r;
}

class OfxFill : public ::testing::Test {
protected:
    void SetUp() override {
        root_ = fs::temp_directory_path() / "rudra_ofx_fill_test";
        std::error_code ec;
        fs::remove_all(root_, ec);
        fs::create_directories(root_);
        set_env("RUDRA_OFX_FILL_ROOT", (root_ / "store").string());
        set_env("RUDRA_OFX_ENGINE_CONFIG", fill_config(root_).string());
        set_env("RUDRA_OFX_FAKE_JOB", "");
    }
    // Polls the status line until it holds `text` (the job runs beside the test): at most 20 s.
    static bool wait_for(const rudra_ofx::Settings& s, const std::string& text) {
        for (int i = 0; i < 200; ++i) {
            if (rudra_ofx::ai_status(s).find(text) != std::string::npos) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        return false;
    }
    fs::path root_;
};

}  // namespace

TEST_F(OfxFill, CollectingStoresEveryFrameOnceAndSaysWhenTheClipIsIn) {
    const auto r = request();
    EXPECT_EQ(fill::status(r).state, fill::State::None);
    ASSERT_TRUE(fill::begin_collect(r));
    auto s = fill::status(r);
    EXPECT_EQ(s.state, fill::State::Collecting);
    EXPECT_EQ(s.done, 0);
    EXPECT_EQ(s.of, 3);
    auto a = fill::store_frame(r, 10, codes(0.3f), PlanarBuffer(3, kH, kW, 1.0f));
    ASSERT_TRUE(a) << a.error().message;
    EXPECT_FALSE(*a);
    auto handle = fill::store_frame(r, 9, codes(0.3f), PlanarBuffer(3, kH, kW, 1.0f));   // outside the clip
    ASSERT_TRUE(handle);
    EXPECT_FALSE(*handle);
    ASSERT_TRUE(fill::store_frame(r, 11, codes(0.4f), PlanarBuffer(3, kH, kW, 2.0f)));
    auto last = fill::store_frame(r, 12, codes(0.5f), PlanarBuffer(3, kH, kW, 3.0f));
    ASSERT_TRUE(last);
    EXPECT_TRUE(*last);   // the clip is in
    s = fill::status(r);
    EXPECT_EQ(s.done, 3);
    const fs::path d = fill::folder(r);
    EXPECT_TRUE(fs::is_regular_file(d / "codes" / "f.000000.tif"));
    EXPECT_TRUE(fs::is_regular_file(d / "request.json"));
    auto back = read_exr(d / "master" / "f.000001.exr");   // ACEScg, half: 2.0 exactly
    ASSERT_TRUE(back);
    EXPECT_EQ(back->pixels.plane(0)[0], 2.0f);
    EXPECT_EQ(fill::frame_stem(12), "f.000012");
}

TEST_F(OfxFill, AFrameOfAnotherSizeIsRefused) {
    const auto r = request();
    ASSERT_TRUE(fill::begin_collect(r));
    ASSERT_TRUE(fill::store_frame(r, 10, codes(0.3f), PlanarBuffer(3, kH, kW, 1.0f)));
    auto proxy = fill::store_frame(r, 11, SdrImage(PlanarBuffer(3, kH / 2, kW / 2, 0.3f)), PlanarBuffer(3, kH / 2, kW / 2, 1.0f));
    ASSERT_FALSE(proxy);
    EXPECT_NE(proxy.error().message.find("full size"), std::string::npos) << proxy.error().message;
}

TEST_F(OfxFill, EverySettingThatChangesTheFillIsAnotherKey) {
    const auto r = request();
    for (auto change : {+[](fill::FillRequest& x) { x.prompt = "sky"; }, +[](fill::FillRequest& x) { x.seed = 1; },
                        +[](fill::FillRequest& x) { x.region_json = "{}"; }, +[](fill::FillRequest& x) { x.master_key = "learned"; },
                        +[](fill::FillRequest& x) { x.first = 11; }, +[](fill::FillRequest& x) { x.condition = "aces"; },
                        +[](fill::FillRequest& x) { x.steps = 20; }}) {   // 27 Nov 2026: Steps
        auto other = r;
        change(other);
        EXPECT_NE(other.key(), r.key());
    }
    EXPECT_EQ(request().key(), r.key());
}

TEST_F(OfxFill, TheJobStartsWhenTheLastFrameIsInAndItsFillIsShownForTheSameCodes) {
    rudra_ofx::Settings s;
    s.ai_fill = true, s.shot_id = "test02", s.first_frame = 0, s.last_frame = 2, s.output = "linear_acescg";
    EXPECT_EQ(rudra_ofx::ai_generate(s), "");
    EXPECT_NE(rudra_ofx::ai_status(s).find("collecting frames, 0 of 3"), std::string::npos) << rudra_ofx::ai_status(s);
    std::vector<float> out(std::size_t(kW) * kH * 4);
    std::string note;
    for (int f = 0; f < 3; ++f) {
        s.frame = f;
        const auto px = pixels(0.02f * float(f));
        const auto r = rudra_ofx::render(s, px.data(), kW * 4, out.data(), kW * 4, kW, kH);
        ASSERT_EQ(r.error, "");
        note = r.note;
    }
    EXPECT_NE(note.find("has started"), std::string::npos) << note;
    ASSERT_TRUE(wait_for(s, "ready")) << rudra_ofx::ai_status(s);

    // The stored fill (the stand-in's: the master itself, through half floats) after Exposure, in the output.
    s.frame = 1, s.exposure_ev = 1.0;
    const auto px = pixels(0.02f);
    const auto shown = rudra_ofx::render(s, px.data(), kW * 4, out.data(), kW * 4, kW, kH);
    ASSERT_EQ(shown.error, "");
    EXPECT_NE(shown.note.find("ready"), std::string::npos) << shown.note;
    rudra_ofx::Settings plain = s;
    plain.ai_fill = false;
    std::vector<float> master(out.size());
    ASSERT_EQ(rudra_ofx::render(plain, px.data(), kW * 4, master.data(), kW * 4, kW, kH).error, "");
    for (std::size_t i = 0; i < out.size(); ++i) ASSERT_NEAR(out[i], master[i], std::max(1e-3f, std::fabs(master[i]) * 2e-3f)) << i;

    // Other codes coming in (something upstream changed): the master, and why.
    const auto changed = pixels(0.3f);
    const auto stale = rudra_ofx::render(s, changed.data(), kW * 4, out.data(), kW * 4, kW, kH);
    ASSERT_EQ(stale.error, "");
    EXPECT_NE(stale.note.find("changed since Generate"), std::string::npos) << stale.note;

    // Another seed is another fill: out of date until Generate.
    s.seed = 5;
    EXPECT_NE(rudra_ofx::ai_status(s).find("out of date"), std::string::npos) << rudra_ofx::ai_status(s);
    s.seed = 0;
    // Alpha: what AI drew (the stand-in drew nothing: 0).
    s.alpha = "ai_fill";
    ASSERT_EQ(rudra_ofx::render(s, px.data(), kW * 4, out.data(), kW * 4, kW, kH).error, "");
    EXPECT_EQ(out[3], 0.0f);
}

TEST_F(OfxFill, AFailedJobSaysWhy) {
    set_env("RUDRA_OFX_FAKE_JOB", "fail");
    rudra_ofx::Settings s;
    s.ai_fill = true, s.shot_id = "test03", s.first_frame = 0, s.last_frame = 0;
    ASSERT_EQ(rudra_ofx::ai_generate(s), "");
    std::vector<float> out(std::size_t(kW) * kH * 4);
    const auto px = pixels();
    ASSERT_EQ(rudra_ofx::render(s, px.data(), kW * 4, out.data(), kW * 4, kW, kH).error, "");
    ASSERT_TRUE(wait_for(s, "failed")) << rudra_ofx::ai_status(s);
    EXPECT_NE(rudra_ofx::ai_status(s).find("the stand-in failed"), std::string::npos) << rudra_ofx::ai_status(s);
}

TEST_F(OfxFill, CancelStopsCollectingAndClearRemovesTheStoredFill) {
    const auto r = request();
    ASSERT_TRUE(fill::begin_collect(r));
    ASSERT_TRUE(fill::store_frame(r, 10, codes(0.3f), PlanarBuffer(3, kH, kW, 1.0f)));
    ASSERT_TRUE(fill::cancel(r));
    EXPECT_EQ(fill::status(r).state, fill::State::None);
    auto ignored = fill::store_frame(r, 11, codes(0.3f), PlanarBuffer(3, kH, kW, 1.0f));   // not collecting
    ASSERT_TRUE(ignored);
    EXPECT_FALSE(*ignored);
    ASSERT_TRUE(fill::begin_collect(r));   // Generate again: the frame stored before is kept
    EXPECT_EQ(fill::status(r).done, 1);
    ASSERT_TRUE(fill::clear(r.shot_id));
    EXPECT_FALSE(fs::exists(fill::root() / r.shot_id));
}

TEST_F(OfxFill, ARunningJobWithNoWordForFiveMinutesHasStopped) {
    const auto r = request();
    ASSERT_TRUE(fill::begin_collect(r));
    std::ofstream(fill::folder(r) / "status.json") << R"({"state": "running", "done": 1, "of": 3, "updated": 1000})";
    const auto s = fill::status(r);
    EXPECT_EQ(s.state, fill::State::Failed);
    EXPECT_NE(s.message.find("no word"), std::string::npos) << s.message;
}

TEST_F(OfxFill, AiFillOffIsThePictureAsItWas) {
    rudra_ofx::Settings s;
    const auto px = pixels();
    std::vector<float> a(px.size()), b(px.size());
    const auto r = rudra_ofx::render(s, px.data(), kW * 4, a.data(), kW * 4, kW, kH);
    ASSERT_EQ(r.error, "");
    EXPECT_EQ(r.note, "");
    ASSERT_EQ(rudra_ofx::render("analytic", "bt1886", 0.0, 1.0, "all", true, true, "linear_rec2020", px.data(), kW * 4, b.data(),
                                kW * 4, kW, kH),
              "");
    EXPECT_EQ(a, b);
    EXPECT_EQ(rudra_ofx::ai_status(s), "AI fill is off.");
}

// 26 Nov 2026: a frame outside the clip's range (the host's own frame numbers) is not stored, and the note says
// so with both, so a host whose numbers differ from its range shows why collecting does not move.
TEST_F(OfxFill, AFrameOutsideTheClipIsNotStoredAndSaysWhy) {
    rudra_ofx::Settings s;
    s.ai_fill = true, s.shot_id = "test07", s.first_frame = 1001, s.last_frame = 1002;
    ASSERT_EQ(rudra_ofx::ai_generate(s), "");
    std::vector<float> out(std::size_t(kW) * kH * 4);
    const auto px = pixels();
    s.frame = 0;
    const auto r = rudra_ofx::render(s, px.data(), kW * 4, out.data(), kW * 4, kW, kH);
    ASSERT_EQ(r.error, "");
    EXPECT_EQ(r.note, "AI fill: frame 0 is outside its frames (1001 to 1002), so it is not stored.");
    EXPECT_NE(rudra_ofx::ai_status(s).find("collecting frames, 0 of 2"), std::string::npos) << rudra_ofx::ai_status(s);
    s.frame = 1001;   // inside: stored
    ASSERT_EQ(rudra_ofx::render(s, px.data(), kW * 4, out.data(), kW * 4, kW, kH).error, "");
    EXPECT_NE(rudra_ofx::ai_status(s).find("collecting frames, 1 of 2"), std::string::npos) << rudra_ofx::ai_status(s);
}

// 27 Nov 2026: Generate's status says which frames and what they cost (the 823-frame clip that waited for all 823).
TEST_F(OfxFill, TheStatusSaysWhichFramesAndHowLong) {
    rudra_ofx::Settings s;
    s.ai_fill = true, s.shot_id = "test08", s.first_frame = 1, s.last_frame = 30, s.steps = 20;
    ASSERT_EQ(rudra_ofx::ai_generate(s), "");
    const std::string st = rudra_ofx::ai_status(s);
    EXPECT_NE(st.find("collecting frames, 0 of 30"), std::string::npos) << st;
    EXPECT_NE(st.find("Frames 1 to 30 (30): about 4 minutes to draw at 20 steps."), std::string::npos) << st;
}
