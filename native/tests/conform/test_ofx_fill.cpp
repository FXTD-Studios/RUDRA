// AI fill in the OFX plug-in (phase 3, 26 Nov 2026): Generate, the frames collected as the host renders them,
// the job beside the host (here the stand-in engine started as ofx_job.py: each stored master becomes its
// frame's "fill"), and the stored fill shown from then on, only for the same settings and the same codes.
#include <gtest/gtest.h>

#include <algorithm>
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

// The plug-in's logs go aside, for every test in this program (1.3.2): they went to the user's %TEMP%\rudra_ofx.log,
// where a stand-in's "Prompt suggested ..." read like a real session.
const bool kLogsAside = [] {
    const fs::path d = fs::temp_directory_path() / "rudra_ofx_test_logs";
    std::error_code ec;
    fs::create_directories(d, ec);
    set_env("RUDRA_OFX_LOG_DIR", d.string());
    return true;
}();

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
        // A prompt reading another test started (the stand-in, beside the test) ends first: still running, it kept this
        // test's Generate from reading its own prompt, and wrote the other prompt into this test's folder (27 Nov 2026).
        for (int i = 0; i < 400 && fill::prompt_embed_running(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        root_ = fs::temp_directory_path() / "rudra_ofx_fill_test";
        std::error_code ec;
        fs::remove_all(root_, ec);
        fs::create_directories(root_);
        set_env("RUDRA_OFX_FILL_ROOT", (root_ / "store").string());
        set_env("RUDRA_OFX_ENGINE_CONFIG", fill_config(root_).string());
        set_env("RUDRA_OFX_FAKE_JOB", "");
        set_env("RUDRA_OFX_FAKE_SUGGEST", "");
        set_env("RUDRA_OFX_FILL_FREE_BYTES", "");
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

// ---- 1.3.2 (26 Nov 2026): Prompt strength, the prompt read while the frames are collected, Suggest prompt ----

TEST_F(OfxFill, PromptStrengthIsInTheKeyOnlyWhenItIsNotOneAndReachesTheRequest) {
    auto r = request();
    r.prompt = "sky";
    const std::string one = r.key();
    r.prompt_strength = 1.0;
    EXPECT_EQ(r.key(), one);                       // 1, the default, keeps the keys fills were stored under
    r.prompt_strength = 1.5;
    EXPECT_NE(r.key(), one);
    rudra_ofx::Settings s;
    s.ai_fill = true, s.shot_id = "test20", s.first_frame = 1, s.last_frame = 3;
    s.prompt_strength = 1.234;                     // no prompt: nothing to steer, 1
    ASSERT_EQ(rudra_ofx::ai_generate(s), "");
    s.prompt = "sky";                              // in hundredths
    ASSERT_EQ(rudra_ofx::ai_generate(s), "");
    std::vector<double> got;
    for (const auto& d : fs::recursive_directory_iterator(root_ / "store" / "test20"))
        if (d.path().filename() == "request.json") {
            std::ifstream f(d.path());
            got.push_back(nlohmann::json::parse(f).value("prompt_strength", -1.0));
        }
    std::sort(got.begin(), got.end());
    ASSERT_EQ(got.size(), 2u);
    EXPECT_DOUBLE_EQ(got[0], 1.0);
    EXPECT_DOUBLE_EQ(got[1], 1.23);
}

TEST_F(OfxFill, TheTimeGenerateAnnouncesCountsTheSecondPassOfABlendedPrompt) {
    rudra_ofx::Settings s;
    s.ai_fill = true, s.shot_id = "test21", s.first_frame = 1, s.last_frame = 30, s.steps = 20;
    const std::string plain = rudra_ofx::ai_status(s);
    EXPECT_NE(plain.find("about 4 minutes"), std::string::npos) << plain;
    s.prompt = "sky";                              // one pass a step at strength 1: what no prompt costs
    const std::string prompted = rudra_ofx::ai_status(s);
    EXPECT_NE(prompted.find(fill::kSinglePromptPass ? "about 4 minutes" : "about 5 minutes"), std::string::npos) << prompted;
    s.prompt_strength = 1.5;                       // a strength other than 1 always blends: 4 x 1.35
    EXPECT_NE(rudra_ofx::ai_status(s).find("about 5 minutes"), std::string::npos) << rudra_ofx::ai_status(s);
}

TEST_F(OfxFill, GenerateWithAPromptReadsItWhileTheFramesAreCollected) {
    rudra_ofx::Settings s;
    s.ai_fill = true, s.shot_id = "test22", s.first_frame = 1, s.last_frame = 3, s.prompt = "my sky";
    ASSERT_EQ(rudra_ofx::ai_generate(s), "");
    const fs::path called = root_ / "store" / "prompt_cache" / "embed_called.txt";
    std::string text, device;
    for (int i = 0; i < 100 && text != "my sky"; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::ifstream f(called);
        std::getline(f, text);
        std::getline(f, device);
    }
    EXPECT_EQ(text, "my sky");
    EXPECT_EQ(device, "cpu");                      // the learned engine may hold the GPU while frames are collected
}

TEST_F(OfxFill, SuggestPromptReadsTheFrameAndAnswersWithTheFirstHighlightSuggestion) {
    rudra_ofx::Settings s;
    s.shot_id = "test23";
    const std::vector<unsigned char> rgb(std::size_t(kW) * kH * 3, 200);
    const auto r = rudra_ofx::ai_suggest(s, rgb.data(), kW, kH, 20.0);
    ASSERT_TRUE(r.ready) << r.error;
    EXPECT_EQ(r.text, "bright hazy sky, stand-in");
    EXPECT_EQ(r.label, "From this shot");
    EXPECT_EQ(r.others, std::vector<std::string>{"Overcast sky"});   // the shadow suggestion is not Prompt's
    EXPECT_NE(r.reason.find("stand-in on auto"), std::string::npos) << r.reason;
    EXPECT_TRUE(fs::is_regular_file(root_ / "store" / "test23" / "suggest" / "frame.tif"));
    const auto again = rudra_ofx::ai_suggestion(s);   // read again: the same answer
    EXPECT_TRUE(again.ready);
    EXPECT_EQ(again.text, r.text);
}

TEST_F(OfxFill, ASuggestionStillBeingMadeIsReadLater) {
    set_env("RUDRA_OFX_FAKE_SUGGEST", "slow");
    rudra_ofx::Settings s;
    s.shot_id = "test24";
    const std::vector<unsigned char> rgb(std::size_t(kW) * kH * 3, 200);
    int ticks = 0;
    const auto first = rudra_ofx::ai_suggest(s, rgb.data(), kW, kH, 0.3, [&](double) { return ++ticks < 1000; });
    EXPECT_FALSE(first.ready);
    EXPECT_TRUE(first.error.empty()) << first.error;
    EXPECT_GT(ticks, 0);
    const auto busy = rudra_ofx::ai_suggest(s, rgb.data(), kW, kH, 0.0);
    EXPECT_NE(busy.error.find("still reading a frame"), std::string::npos) << busy.error;
    rudra_ofx::PromptSuggestion later;
    for (int i = 0; i < 100 && !later.ready; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        later = rudra_ofx::ai_suggestion(s);
    }
    EXPECT_TRUE(later.ready) << later.error;
    set_env("RUDRA_OFX_FAKE_SUGGEST", "");
}

TEST_F(OfxFill, SuggestPromptSaysWhyWhenThereIsNone) {
    rudra_ofx::Settings s;
    s.shot_id = "test25";
    const std::vector<unsigned char> rgb(std::size_t(kW) * kH * 3, 200);
    EXPECT_EQ(rudra_ofx::ai_suggest(s, nullptr, kW, kH, 1.0).error, "no frame to read.");
    EXPECT_NE(rudra_ofx::ai_suggestion(s).error.find("press it first"), std::string::npos);
    set_env("RUDRA_OFX_FAKE_SUGGEST", "fail");
    auto r = rudra_ofx::ai_suggest(s, rgb.data(), kW, kH, 20.0);
    EXPECT_NE(r.error.find("could not read the frame"), std::string::npos) << r.error;
    set_env("RUDRA_OFX_FAKE_SUGGEST", "nothing");
    r = rudra_ofx::ai_suggest(s, rgb.data(), kW, kH, 20.0);
    EXPECT_NE(r.error.find("nothing is fully clipped"), std::string::npos) << r.error;
    set_env("RUDRA_OFX_FAKE_SUGGEST", "");
    set_env("RUDRA_OFX_ENGINE_CONFIG", (root_ / "missing.json").string());
    r = rudra_ofx::ai_suggest(s, rgb.data(), kW, kH, 1.0);
    EXPECT_FALSE(r.ready);
    EXPECT_FALSE(r.error.empty());
}

TEST_F(OfxFill, FlorenceIsReadFromTheSettingsElseBesideTheBaseModel) {
    auto cfg = ofx_engine::load_genfill_config();
    ASSERT_TRUE(cfg) << cfg.error().message;
    EXPECT_TRUE(cfg->florence.empty());
    fs::create_directories(root_ / "Florence-2-base");   // beside the base model (root_/base), as the app's setup puts it
    std::ofstream(root_ / "Florence-2-base" / "config.json") << "{}";
    cfg = ofx_engine::load_genfill_config();
    ASSERT_TRUE(cfg);
    EXPECT_EQ(cfg->florence, root_ / "Florence-2-base");
    fs::create_directories(root_ / "elsewhere");
    std::ofstream(root_ / "elsewhere" / "config.json") << "{}";
    std::ifstream in(root_ / "ofx_engine.json");
    nlohmann::json j = nlohmann::json::parse(in);
    in.close();
    j["genfill"]["florence"] = (root_ / "elsewhere").generic_string();
    std::ofstream(root_ / "ofx_engine.json", std::ios::trunc) << j.dump();
    cfg = ofx_engine::load_genfill_config();
    ASSERT_TRUE(cfg);
    EXPECT_EQ(fs::path(cfg->florence).lexically_normal(), (root_ / "elsewhere").lexically_normal());
    rudra_ofx::Settings s;
    s.shot_id = "test26";
    const std::vector<unsigned char> rgb(std::size_t(kW) * kH * 3, 200);
    const auto r = rudra_ofx::ai_suggest(s, rgb.data(), kW, kH, 20.0);
    ASSERT_TRUE(r.ready) << r.error;
    EXPECT_NE(r.reason.find("Florence"), std::string::npos) << r.reason;   // its folder reached the suggester
}

// 1.3.2: while frames are collected, a bar along the bottom of the picture shows how far it got (the Status line
// changes only when a button is pressed), the status says which frames are still to play, and each stored frame goes
// to the plug-in's log.
TEST_F(OfxFill, CollectingShowsABarSaysWhatIsStillToPlayAndLogsEachFrame) {
    rudra_ofx::Settings s;
    s.ai_fill = true, s.shot_id = "test27", s.first_frame = 10, s.last_frame = 13, s.output = "linear_rec2020";
    ASSERT_EQ(rudra_ofx::ai_generate(s), "");
    std::vector<float> out(std::size_t(kW) * kH * 4), plain(out.size());
    const auto px = pixels();
    s.frame = 11;
    const auto r = rudra_ofx::render(s, px.data(), kW * 4, out.data(), kW * 4, kW, kH);
    ASSERT_EQ(r.error, "");
    EXPECT_EQ(r.log, "AI fill: frame 11 stored, 1 of 4");
    EXPECT_EQ(r.guide, "RUDRA AI fill: 1 of 4 frames in. Play frames 10 to 13 once; the fill starts when all are in.");
    rudra_ofx::Settings off = s;
    off.ai_fill = false;
    ASSERT_EQ(rudra_ofx::render(off, px.data(), kW * 4, plain.data(), kW * 4, kW, kH).error, "");
    const int rows = 2;                                  // max(2, height / 120)
    for (int y = 0; y < kH - rows; ++y)                  // above the bar: the picture as without AI fill
        for (int i = 0; i < kW * 4; ++i) {
            const std::size_t k = std::size_t(y) * kW * 4 + std::size_t(i);
            ASSERT_NEAR(out[k], plain[k], std::max(1e-3f, std::fabs(plain[k]) * 2e-3f)) << y << " " << i;
        }
    const float* lit = &out[(std::size_t(kH - 1) * kW + 0) * 4];         // a quarter in: the first quarter lit, green
    const float* dark = &out[(std::size_t(kH - 1) * kW + (kW - 1)) * 4];
    EXPECT_GT(lit[1], 0.9f);
    EXPECT_LT(dark[1], 0.1f);
    EXPECT_EQ(lit[3], 1.0f);
    std::string st = rudra_ofx::ai_status(s);
    EXPECT_NE(st.find("Still to play: frame 10 and 2 more."), std::string::npos) << st;
    s.frame = 10;
    ASSERT_EQ(rudra_ofx::render(s, px.data(), kW * 4, out.data(), kW * 4, kW, kH).error, "");
    st = rudra_ofx::ai_status(s);
    EXPECT_NE(st.find("Still to play: frames 12 to 13."), std::string::npos) << st;
}

// ---- 1.3.2 (27 Nov 2026): the guide in the picture, for a user who never opens the Status line ----

namespace {

constexpr int kGW = 480, kGH = 270;   // letters of 5 x 7 pixels, 79 to a line

std::vector<float> big_pixels() {
    std::vector<float> p(std::size_t(kGW) * kGH * 4);
    for (int y = 0; y < kGH; ++y)
        for (int x = 0; x < kGW; ++x) {
            float* q = &p[(std::size_t(y) * kGW + std::size_t(x)) * 4];
            q[0] = float(x) / (kGW - 1);
            q[1] = float(y) / (kGH - 1);
            q[2] = 0.3f;
            q[3] = 1.0f;
        }
    return p;
}

double seconds_now() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }

}  // namespace

TEST_F(OfxFill, TheGuideSaysWhatToDoBeforeGenerateAndReadsTheRightWayUp) {
    rudra_ofx::Settings s;
    s.ai_fill = true, s.shot_id = "test30", s.first_frame = 0, s.last_frame = 2, s.output = "linear_rec2020";
    const auto px = big_pixels();
    std::vector<float> out(px.size()), plain(px.size());
    auto r = rudra_ofx::render(s, px.data(), kGW * 4, out.data(), kGW * 4, kGW, kGH);
    ASSERT_EQ(r.error, "");
    EXPECT_EQ(r.guide, "RUDRA AI fill: press Generate, then play the frames once.");
    rudra_ofx::Settings off = s;
    off.ai_fill = false;
    ASSERT_EQ(rudra_ofx::render(off, px.data(), kGW * 4, plain.data(), kGW * 4, kGW, kGH).error, "");
    // Above the words (a line: 11 rows) and the bar (2 rows), the picture as without AI fill.
    const int top = kGH - 2 - 11;
    for (std::size_t k = 0; k < std::size_t(top) * kGW * 4; ++k)
        ASSERT_NEAR(out[k], plain[k], std::max(1e-3f, std::fabs(plain[k]) * 2e-3f)) << k;
    // The first letter, R: its top row four pixels lit, its bottom row the two ends. The rows run top first, so a
    // guide drawn upside down fails here.
    const auto at = [&](int x, int y) { return out[(std::size_t(y) * kGW + std::size_t(x)) * 4 + 1]; };
    const int x0 = 2, y0 = top + 2;
    for (int x = 0; x < 4; ++x) EXPECT_GT(at(x0 + x, y0), 0.9f) << x;
    EXPECT_LT(at(x0 + 4, y0), 0.1f);
    EXPECT_GT(at(x0, y0 + 6), 0.9f);
    EXPECT_LT(at(x0 + 2, y0 + 6), 0.1f);
    EXPECT_GT(at(x0 + 4, y0 + 6), 0.9f);

    s.host_label = "Resolve";   // Generate there takes the frames from the playhead
    r = rudra_ofx::render(s, px.data(), kGW * 4, out.data(), kGW * 4, kGW, kGH);
    EXPECT_EQ(r.guide, "RUDRA AI fill: put the playhead on the first frame to fill, then press Generate.");

    // A Generate refused (AI fill not set up): the guide says why, in red, until a Generate starts.
    std::ofstream(root_ / "bare.json") << R"({"cli": "", "package": ""})";
    set_env("RUDRA_OFX_ENGINE_CONFIG", (root_ / "bare.json").string());
    EXPECT_NE(rudra_ofx::ai_generate(s), "");
    r = rudra_ofx::render(s, px.data(), kGW * 4, out.data(), kGW * 4, kGW, kGH);
    EXPECT_EQ(r.guide.rfind("RUDRA AI fill: Generate did not start. AI fill is not set up for the plug-in", 0), 0u) << r.guide;
    const float* corner = &out[(std::size_t(kGH - 1) * kGW + std::size_t(kGW - 1)) * 4];
    EXPECT_GT(corner[0], 0.9f);
    EXPECT_LT(corner[1], 0.5f);
    set_env("RUDRA_OFX_ENGINE_CONFIG", fill_config(root_).string());
    ASSERT_EQ(rudra_ofx::ai_generate(s), "");
    r = rudra_ofx::render(s, px.data(), kGW * 4, out.data(), kGW * 4, kGW, kGH);
    EXPECT_EQ(r.guide, "RUDRA AI fill: 1 of 3 frames in. Play the clip through once on the Color page; the fill starts when all are in.");
}

TEST_F(OfxFill, TheGuideFollowsTheFramesCollectedAndTheFillBeingDrawn) {
    rudra_ofx::Settings s;
    s.ai_fill = true, s.shot_id = "test31", s.first_frame = 0, s.last_frame = 2, s.host_label = "Nuke";
    ASSERT_EQ(rudra_ofx::ai_generate(s), "");
    std::vector<float> out(std::size_t(kW) * kH * 4);
    const auto px = pixels();
    const auto guide_at = [&](int frame, bool full = true) {
        s.frame = frame, s.full_size = full;
        const auto r = rudra_ofx::render(s, px.data(), kW * 4, out.data(), kW * 4, kW, kH);
        EXPECT_EQ(r.error, "");
        return r.guide;
    };
    EXPECT_EQ(guide_at(0), "RUDRA AI fill: 1 of 3 frames in. Play frames 0 to 2 once; the fill starts when all are in.");
    EXPECT_EQ(guide_at(7), "RUDRA AI fill: 1 of 3 frames in. Frame 7 is not one of them: play frames 0 to 2.");
    EXPECT_EQ(guide_at(1, false),
              "RUDRA AI fill: 1 of 3 frames in. Frames are taken from full-size renders only: turn the proxy and the viewer's downscale off.");
    s.host_label = "Resolve";   // its frame numbers are its source media's, which its user does not see: words instead
    EXPECT_EQ(guide_at(1), "RUDRA AI fill: 2 of 3 frames in. Play the clip through once on the Color page; the fill starts when all are in.");
    EXPECT_EQ(guide_at(-1), "RUDRA AI fill: 2 of 3 frames in. This frame comes before them: play on.");
    EXPECT_EQ(guide_at(1, false), "RUDRA AI fill: 2 of 3 frames in. This frame came at a reduced size, so it is not taken: play the clip "
                                  "through on the Color page, with no timeline proxy.");
    EXPECT_EQ(guide_at(3),
              "RUDRA AI fill: 2 of 3 frames in. 1 was missed: go back to where you pressed Generate and play through again on the Color page.");

    // The fill being drawn, its status.json as ofx_job.py writes it: how far, in blue, over the stored master.
    fs::path dir;
    for (const auto& e : fs::directory_iterator(fill::root() / s.shot_id))
        if (e.is_directory()) dir = e.path();
    ASSERT_FALSE(dir.empty());
    const auto job_says = [&](const std::string& state, const std::string& message) {
        const nlohmann::json j = {{"state", state}, {"done", 0}, {"of", 3}, {"message", message}, {"updated", seconds_now()}};
        std::ofstream(dir / "status.json", std::ios::trunc) << j.dump();
    };
    s.host_label = "Nuke";
    job_says("running", "window 2 of 4, step 15 of 30");
    EXPECT_EQ(guide_at(0), "RUDRA AI fill: drawing, 37% done. Keep Nuke open; press Refresh status to update this.");
    const float* lit = &out[(std::size_t(kH - 1) * kW + 0) * 4];
    EXPECT_GT(lit[2], 0.9f);
    EXPECT_LT(lit[0], 0.3f);
    job_says("running", "reading the prompt (started when Generate was pressed; once per new prompt)");
    EXPECT_EQ(guide_at(0), "RUDRA AI fill: reading the prompt (minutes the first time). Keep Nuke open; press Refresh status to update this.");
    job_says("running", "not enough GPU memory: drawing again in low-memory mode");
    EXPECT_EQ(guide_at(0), "RUDRA AI fill: not enough GPU memory, so it is drawn again in low-memory mode. Keep Nuke open; press Refresh "
                           "status to update this.");
    job_says("running", "starting");
    s.host_label.clear();
    EXPECT_EQ(guide_at(0), "RUDRA AI fill: starting (the model loads first). Keep the host open; press Refresh status to update this.");
    job_says("failed", "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB (log: C:/x/job.log)");
    EXPECT_EQ(guide_at(0), "RUDRA AI fill failed: not enough GPU memory. Close other programs that use the GPU, then press Generate again.");
    job_says("failed", "ValueError: no frames (log: C:/x/job.log)");
    EXPECT_EQ(guide_at(0), "RUDRA AI fill failed; press Generate to try again. Why: ValueError: no frames");
    job_says("cancelled", "cancelled");
    EXPECT_EQ(guide_at(0), "RUDRA AI fill cancelled. Press Generate to start again (the frames stored are kept).");
}

TEST_F(OfxFill, AReadyFillAndTheFramesItWasNotMadeForAreShownClean) {
    rudra_ofx::Settings s;
    s.ai_fill = true, s.shot_id = "test32", s.first_frame = 0, s.last_frame = 1, s.host_label = "Nuke";
    ASSERT_EQ(rudra_ofx::ai_generate(s), "");
    std::vector<float> out(std::size_t(kW) * kH * 4), plain(out.size());
    const auto px = pixels();
    rudra_ofx::Outcome r;
    for (int f = 0; f < 2; ++f) {
        s.frame = f;
        r = rudra_ofx::render(s, px.data(), kW * 4, out.data(), kW * 4, kW, kH);
        ASSERT_EQ(r.error, "");
    }
    EXPECT_EQ(r.guide, "RUDRA AI fill: all 2 frames in; the fill is being drawn. Keep Nuke open; press Refresh status to update this.");
    ASSERT_TRUE(wait_for(s, "ready")) << rudra_ofx::ai_status(s);
    rudra_ofx::Settings off = s;
    off.ai_fill = false;
    for (int f : {1, 5}) {   // a frame of the fill, and one it was not made for (a handle, or past To)
        s.frame = off.frame = f;
        r = rudra_ofx::render(s, px.data(), kW * 4, out.data(), kW * 4, kW, kH);
        ASSERT_EQ(r.error, "");
        EXPECT_EQ(r.guide, "") << f;
        ASSERT_EQ(rudra_ofx::render(off, px.data(), kW * 4, plain.data(), kW * 4, kW, kH).error, "");
        for (std::size_t i = 0; i < out.size(); ++i)
            ASSERT_NEAR(out[i], plain[i], std::max(1e-3f, std::fabs(plain[i]) * 2e-3f)) << f << " " << i;
    }
    s.frame = 1, s.full_size = false;
    EXPECT_EQ(rudra_ofx::render(s, px.data(), kW * 4, out.data(), kW * 4, kW, kH).guide,
              "RUDRA AI fill is ready; it is shown at full size only (turn the proxy off).");
    s.full_size = true;
    // The same picture decoded a little differently (Resolve, 27 Nov 2026: a paused frame's codes were not the played
    // one's): its fill is shown, clean.
    auto decoded = pixels();
    for (std::size_t i = 0; i < decoded.size(); ++i)
        if (i % 4 != 3) decoded[i] = std::min(1.0f, decoded[i] + 0.003f);
    EXPECT_EQ(rudra_ofx::render(s, decoded.data(), kW * 4, out.data(), kW * 4, kW, kH).guide, "");
    const auto changed = pixels(0.3f);   // something upstream changed: said, with how much
    const std::string g = rudra_ofx::render(s, changed.data(), kW * 4, out.data(), kW * 4, kW, kH).guide;
    EXPECT_EQ(g.rfind("RUDRA AI fill: the picture coming into the node has changed since Generate (by ", 0), 0u) << g;
    EXPECT_NE(g.find(" codes on average): press Generate again."), std::string::npos) << g;
    s.seed = 9;   // another setting
    EXPECT_EQ(rudra_ofx::render(s, px.data(), kW * 4, out.data(), kW * 4, kW, kH).guide,
              "RUDRA AI fill is out of date: a setting changed since Generate. Press Generate again, or set it back.");
    // A fill that found nothing to draw (ofx_job.py's stats) says so, where any other ready fill is clean.
    s.seed = 0;
    fs::path dir;
    for (const auto& e : fs::directory_iterator(fill::root() / s.shot_id))
        if (e.is_directory()) dir = e.path();
    ASSERT_FALSE(dir.empty());
    std::ofstream(dir / "status.json", std::ios::trunc)
        << R"({"state": "done", "done": 2, "of": 2, "message": "done", "stats": {"generated_fraction": 0.0}})";
    EXPECT_EQ(rudra_ofx::render(s, px.data(), kW * 4, out.data(), kW * 4, kW, kH).guide,
              "RUDRA AI fill: done, and it found nothing to draw in frames 0 to 1: nothing there clips (240 and up).");
}

// 1.3.2: a disk with no room for the clip is said at Generate or at the clip's first frame, with the numbers, not when
// the fill fails twenty minutes later; with room again, collecting goes on.
TEST_F(OfxFill, NoRoomOnTheDiskIsSaidBeforeAnythingIsStored) {
    const auto r = request();   // 3 frames of 16 x 9: about 1 GB with the spare
    set_env("RUDRA_OFX_FILL_FREE_BYTES", "500000000");
    auto refused = fill::begin_collect(r);
    ASSERT_FALSE(refused);
    EXPECT_NE(refused.error().message.find("not enough disk space: 0.5 GB is free"), std::string::npos) << refused.error().message;
    set_env("RUDRA_OFX_FILL_FREE_BYTES", "1000005000");   // room to start, not for the frames' 10 KB
    ASSERT_TRUE(fill::begin_collect(r));
    auto first = fill::store_frame(r, 10, codes(0.3f), PlanarBuffer(3, kH, kW, 1.0f));
    ASSERT_FALSE(first);
    EXPECT_NE(first.error().message.find("3 frames of 16 x 9 need about 1.0 GB, and 1.0 GB is free"), std::string::npos)
        << first.error().message;
    EXPECT_EQ(fill::status(r).done, 0);
    set_env("RUDRA_OFX_FILL_FREE_BYTES", "");
    auto again = fill::store_frame(r, 10, codes(0.3f), PlanarBuffer(3, kH, kW, 1.0f));
    ASSERT_TRUE(again) << again.error().message;
    EXPECT_EQ(fill::status(r).done, 1);
}

// 27 Nov 2026, Resolve: every frame came with nothing clipped (its brightest code under 240), so two fills drew nothing
// without a word; and To set before the playhead made a fill of one frame. Both are said now.
TEST_F(OfxFill, NothingClippedAndAToBeforeTheFirstFrameAreSaid) {
    rudra_ofx::Settings s;
    s.ai_fill = true, s.shot_id = "test33", s.first_frame = 26, s.last_frame = 96, s.frame = 30, s.host_label = "Resolve";
    auto px = pixels();
    for (std::size_t i = 0; i < px.size(); ++i)
        if (i % 4 != 3) px[i] *= 0.9f;   // the brightest code 230
    std::vector<float> out(std::size_t(kW) * kH * 4);
    auto r = rudra_ofx::render(s, px.data(), kW * 4, out.data(), kW * 4, kW, kH);
    ASSERT_EQ(r.error, "");
    EXPECT_EQ(r.guide.rfind("RUDRA AI fill: nothing in this frame is clipped (its brightest code is 230 of 255), and AI fill draws only "
                            "where the picture clips (240 and up).", 0), 0u) << r.guide;
    s.to_frame = 4, s.last_frame = 26;   // To 4 with the playhead on 26: Resolve's span, one frame
    const std::string why = rudra_ofx::ai_generate(s);
    EXPECT_EQ(why, "AI fill: To (4) is before the first frame to fill (26). Set To to 0 to fill to the clip's end, or to a frame after 26.");
    const auto clipped = pixels();
    r = rudra_ofx::render(s, clipped.data(), kW * 4, out.data(), kW * 4, kW, kH);
    EXPECT_EQ(r.guide, "RUDRA AI fill: Generate did not start. To (4) is before the first frame to fill (26). Set To to 0 to fill to the "
                       "clip's end, or to a frame after 26.");
    s.to_frame = 0, s.last_frame = 96;
    EXPECT_EQ(rudra_ofx::ai_generate(s), "");
}

// 27 Nov 2026, Resolve: a colour-managed project handed the node the clip with its clipped sky flat at code 195 of 255
// (an HDR picture), and RUDRA found nothing to recover. A flat top under 240 is told apart from a picture that is only
// dim, and said: on the node, in AI fill's guide, and by Generate.
TEST_F(OfxFill, AClippedPictureThatArrivesReencodedIsSaid) {
    constexpr int w = 64, h = 36;
    const auto picture = [&](float sky, bool flat) {
        std::vector<float> p(std::size_t(w) * h * 4);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                float* q = &p[(std::size_t(y) * w + std::size_t(x)) * 4];
                const float v = y < 8 && flat ? sky : sky * float(x + y) / float(w + h);   // a flat sky on top, else a ramp
                q[0] = q[1] = q[2] = v, q[3] = 1.0f;
            }
        return p;
    };
    const auto hdr = picture(195.0f / 255.0f, true), clipped = picture(1.0f, true), dim = picture(195.0f / 255.0f, false);
    EXPECT_EQ(rudra_ofx::flat_top_below_white(hdr.data(), w * 4, w, h), 195);
    EXPECT_EQ(rudra_ofx::flat_top_below_white(clipped.data(), w * 4, w, h), 0);   // clipped at 255: as RUDRA wants it
    EXPECT_EQ(rudra_ofx::flat_top_below_white(dim.data(), w * 4, w, h), 0);       // only dim: nothing flat at the top

    rudra_ofx::Settings s;
    s.ai_fill = true, s.shot_id = "test34", s.first_frame = 0, s.last_frame = 9, s.host_label = "Resolve", s.flat_top = 195;
    std::vector<float> out(std::size_t(w) * h * 4);
    const auto r = rudra_ofx::render(s, hdr.data(), w * 4, out.data(), w * 4, w, h);
    ASSERT_EQ(r.error, "");
    const std::string said = "the picture's highlights stop flat at code 195 of 255, not 255: it is not the SDR clip RUDRA reads, so "
                             "RUDRA finds nothing clipped to recover. In Resolve, turn on Enable OpenFX Color Management (Project "
                             "Settings > Color Management), or use a DaVinci YRGB project with the timeline color space Rec.709 Gamma 2.4.";
    EXPECT_EQ(r.guide, "RUDRA AI fill: " + said);
    EXPECT_EQ(rudra_ofx::ai_generate(s), "AI fill: " + said);
    EXPECT_EQ(rudra_ofx::flat_top_advice(195, "Nuke").substr(said.find("In Resolve")), "In Nuke, read the clip with Raw Data on.");
    EXPECT_NE(rudra_ofx::flat_top_advice(235, "Resolve").find("set the clip's Data Levels to Video"), std::string::npos);   // video white
}

// 1.3.2 (27 Nov 2026): OpenFX colour management. A host that manages colour (Resolve's "Enable OpenFX Color Management")
// is asked for the source as Rec.709 display codes and told the output's colourspace by Output; a host that hands
// something else all the same is said to.
TEST_F(OfxFill, ColourManagementAsksForTheSdrAndNamesTheOutput) {
    const std::string none = "OfxImageEffectColourManagementNone", basic = "OfxImageEffectColourManagementBasic",
                      core = "OfxImageEffectColourManagementCore";
    EXPECT_TRUE(rudra_ofx::colour_source_preferences("").empty());
    EXPECT_TRUE(rudra_ofx::colour_source_preferences(none).empty());
    EXPECT_EQ(rudra_ofx::colour_source_preferences(basic), std::vector<std::string>{"ofx_display_sdr"});
    EXPECT_EQ(rudra_ofx::colour_source_preferences(core), (std::vector<std::string>{"rec1886_rec709_display", "ofx_display_sdr"}));
    EXPECT_EQ(rudra_ofx::colour_output("linear_rec2020", core), "lin_rec2020");
    EXPECT_EQ(rudra_ofx::colour_output("linear_acescg", core), "ACEScg");
    EXPECT_EQ(rudra_ofx::colour_output("pq_rec2020", core), "rec2100_pq_display");
    EXPECT_EQ(rudra_ofx::colour_output("linear_acescg", basic), "ofx_scene_linear");
    EXPECT_EQ(rudra_ofx::colour_output("pq_rec2020", basic), "ofx_display_hdr");
    EXPECT_EQ(rudra_ofx::colour_output("linear_rec2020", ""), "");

    rudra_ofx::Settings s;
    s.host_label = "Resolve", s.colour_style = core;
    s.input_colourspace = "rec1886_rec709_display";
    EXPECT_EQ(rudra_ofx::input_problem(s), "");   // what RUDRA asked for
    s.input_colourspace = "Rec.1886 Rec.709 - Display";
    EXPECT_EQ(rudra_ofx::input_problem(s), "");   // its label
    s.input_colourspace = "ACEScct";
    EXPECT_EQ(rudra_ofx::input_problem(s),
              "the host hands RUDRA the picture as ACEScct, not the SDR Rec.709 codes it reads, so RUDRA finds nothing clipped to recover. "
              "Resolve did not hand RUDRA the Rec.709 codes it asks for: use a DaVinci YRGB project with the timeline color space "
              "Rec.709 Gamma 2.4 (Project Settings > Color Management).");
    s.colour_style.clear(), s.input_colourspace.clear(), s.flat_top = 195;   // no colour management: turn it on
    EXPECT_NE(rudra_ofx::input_problem(s).find("In Resolve, turn on Enable OpenFX Color Management"), std::string::npos)
        << rudra_ofx::input_problem(s);
}
