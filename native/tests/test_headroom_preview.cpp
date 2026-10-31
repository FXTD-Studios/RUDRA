// The app-side headroom preview driver (E1): epochs, back-pressure, failure
// reporting and lifetime, over a fake port so it runs on every platform.
// The real HeadroomPresentation port is covered on Windows by
// HeadroomFrames.AppDriverPresentsRealFramesThroughPresentation.
#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>

#include "rudra/engine/headroom_preview.hpp"

using namespace rudra;
using namespace std::chrono_literals;

namespace {

class FakePort final : public HeadroomPreviewPort {
public:
    struct Token final : Pending {
        int revision = 0;
        HeadroomShown shown;
    };
    std::mutex mu;
    std::condition_variable cv;
    int revision = 0, next = -1, end = -1;
    bool grade_ok = false, opened = false;
    int opens = 0, seeks = 0, cancels = 0, presents = 0;
    std::optional<Error> fail_render, fail_present;
    std::chrono::milliseconds render_delay{0};

    Result<void> configure(const Session& s, bool, bool model, bool steady) override {
        std::lock_guard lock(mu);
        ++revision;
        grade_ok = s.grade.headroom_policy == HeadroomPolicy::Bt1886V1 && !model && !steady;
        if (!grade_ok) return make_error(ErrorCode::Unsupported, "fake: unsupported grade");
        return {};
    }
    Result<void> open(const std::filesystem::path&, FloatFrameRange r) override {
        std::lock_guard lock(mu);
        if (!grade_ok) return make_error(ErrorCode::Unsupported, "fake: no grade");
        ++opens;
        opened = true;
        set(r);
        return {};
    }
    Result<void> seek(FloatFrameRange r) override {
        std::lock_guard lock(mu);
        if (!opened) return make_error(ErrorCode::InvalidArgument, "fake: not open");
        ++seeks;
        set(r);
        return {};
    }
    Result<std::shared_ptr<Pending>> render_next(std::chrono::milliseconds wait) override {
        std::unique_lock lock(mu);
        if (!grade_ok) return make_error(ErrorCode::Unsupported, "fake: no grade");
        if (fail_render) {
            auto e = *fail_render;
            fail_render.reset();
            return e;
        }
        if (next < 0) {
            cv.wait_for(lock, wait);
            return make_error(ErrorCode::Busy, "fake: nothing yet");
        }
        if (next >= end) return make_error(ErrorCode::NotFound, "fake: range done");
        auto t = std::make_shared<Token>();
        t->index = next++;
        t->revision = revision;
        t->rendered_bytes = 400;
        t->shown.index = t->index;
        t->shown.width = 2;
        t->shown.height = 1;
        t->shown.rendered_bytes = 200;
        const auto delay = render_delay;
        lock.unlock();
        if (delay.count()) std::this_thread::sleep_for(delay);
        return std::shared_ptr<Pending>(t);
    }
    Result<void> present(const Pending& p, const std::function<void(const HeadroomShown&)>& display) override {
        std::lock_guard lock(mu);
        const auto& t = static_cast<const Token&>(p);
        if (fail_present) return *fail_present;
        if (t.revision != revision) return make_error(ErrorCode::Cancelled, "fake: stale");
        ++presents;
        display(t.shown);
        return {};
    }
    void cancel() override {
        std::lock_guard lock(mu);
        ++revision;
        ++cancels;
        next = -1;
        cv.notify_all();
    }

private:
    void set(FloatFrameRange r) {
        ++revision;
        next = r.first;
        end = r.first + r.count;
        cv.notify_all();
    }
};

// The display thread: tasks queue here and run when the test pumps.
struct Loop {
    std::mutex mu;
    std::deque<std::function<void()>> q;
    void post(std::function<void()> f) {
        std::lock_guard lock(mu);
        q.push_back(std::move(f));
    }
    void pump_once() {
        std::deque<std::function<void()>> now;
        {
            std::lock_guard lock(mu);
            now.swap(q);
        }
        for (auto& f : now) f();
    }
    template <class Pred>
    bool pump_until(Pred pred, std::chrono::milliseconds limit = 3000ms) {
        const auto end = std::chrono::steady_clock::now() + limit;
        while (std::chrono::steady_clock::now() < end) {
            pump_once();
            if (pred()) return true;
            std::this_thread::sleep_for(1ms);
        }
        pump_once();
        return pred();
    }
};

Session headroom_session() {
    Session s(Session::SourceSupport::HeadroomV1);
    EXPECT_TRUE(s.set_headroom_policy(HeadroomPolicy::Bt1886V1));
    return s;
}

struct Rig {
    Loop loop;
    FakePort* port = nullptr;
    std::vector<int> shown;
    std::vector<Error> failures;
    int finished = 0;
    std::unique_ptr<HeadroomPreviewDriver> driver;
    explicit Rig(int depth = 1) {
        auto p = std::make_unique<FakePort>();
        port = p.get();
        HeadroomPreviewDriver::Callbacks cb;
        cb.post = [this](std::function<void()> f) { loop.post(std::move(f)); };
        cb.shown = [this](const HeadroomShown& f) { shown.push_back(f.index); };
        cb.failed = [this](const Error& e) { failures.push_back(e); };
        cb.finished = [this] { ++finished; };
        auto d = HeadroomPreviewDriver::create(std::move(p), cb, depth);
        EXPECT_TRUE(d);
        driver = std::move(*d);
    }
};

}  // namespace

TEST(HeadroomPreviewDriver, RequiresCallbacksAndPort) {
    HeadroomPreviewDriver::Callbacks cb;
    EXPECT_FALSE(HeadroomPreviewDriver::create(std::make_unique<FakePort>(), cb));
    cb.post = [](std::function<void()>) {};
    cb.shown = [](const HeadroomShown&) {};
    EXPECT_FALSE(HeadroomPreviewDriver::create(nullptr, cb));
    EXPECT_FALSE(HeadroomPreviewDriver::create(std::make_unique<FakePort>(), cb, 0));
    EXPECT_TRUE(HeadroomPreviewDriver::create(std::make_unique<FakePort>(), cb));
}

TEST(HeadroomPreviewDriver, PausedShowPresentsExactlyThatFrame) {
    Rig r;
    ASSERT_TRUE(r.driver->configure(headroom_session(), false, false, false));
    ASSERT_TRUE(r.driver->show("a.mkv", 3, 10, false));
    ASSERT_TRUE(r.loop.pump_until([&] { return !r.shown.empty(); }));
    r.loop.pump_until([&] { return r.driver->stats().finished; }, 300ms);
    EXPECT_EQ(r.shown, std::vector<int>{3});
    const auto st = r.driver->stats();
    EXPECT_EQ(st.presented_index, 3);
    EXPECT_EQ(st.presented, 1u);
    EXPECT_EQ(st.shown_rendered_bytes, 200u);
    EXPECT_EQ(r.port->opens, 1);
    EXPECT_TRUE(r.failures.empty());
}

TEST(HeadroomPreviewDriver, SameSourceSeeksInsteadOfReopening) {
    Rig r;
    ASSERT_TRUE(r.driver->configure(headroom_session(), false, false, false));
    ASSERT_TRUE(r.driver->show("a.mkv", 0, 10, false));
    ASSERT_TRUE(r.loop.pump_until([&] { return r.shown.size() == 1; }));
    ASSERT_TRUE(r.driver->show("a.mkv", 5, 10, false));
    ASSERT_TRUE(r.loop.pump_until([&] { return r.shown.size() == 2; }));
    EXPECT_EQ(r.shown.back(), 5);
    EXPECT_EQ(r.port->opens, 1);
    EXPECT_EQ(r.port->seeks, 1);
    ASSERT_TRUE(r.driver->show("b.mkv", 2, 4, false));   // another source opens again
    ASSERT_TRUE(r.loop.pump_until([&] { return r.shown.size() == 3; }));
    EXPECT_EQ(r.port->opens, 2);
    EXPECT_EQ(r.driver->source(), std::filesystem::path("b.mkv"));
}

TEST(HeadroomPreviewDriver, PlayingPresentsInOrderThenFinishes) {
    Rig r;
    ASSERT_TRUE(r.driver->configure(headroom_session(), false, false, false));
    ASSERT_TRUE(r.driver->show("a.mkv", 1, 5, true));
    ASSERT_TRUE(r.loop.pump_until([&] { return r.finished == 1; }));
    EXPECT_EQ(r.shown, (std::vector<int>{1, 2, 3, 4}));
    EXPECT_FALSE(r.driver->stats().active);
}

TEST(HeadroomPreviewDriver, BackPressureKeepsOneTokenInFlight) {
    Rig r;
    ASSERT_TRUE(r.driver->configure(headroom_session(), false, false, false));
    ASSERT_TRUE(r.driver->show("a.mkv", 0, 50, true));
    std::this_thread::sleep_for(150ms);   // the display thread is "slow": nothing pumped
    auto st = r.driver->stats();
    EXPECT_EQ(st.rendered, 1u);
    EXPECT_EQ(st.pending, 1);
    EXPECT_EQ(st.pending_rendered_bytes, 400u);
    ASSERT_TRUE(r.loop.pump_until([&] { return r.shown.size() >= 3; }));
    st = r.driver->stats();
    EXPECT_LE(st.pending, 1);
}

TEST(HeadroomPreviewDriver, SeekDropsTokensOfTheEarlierRequest) {
    Rig r;
    r.port->render_delay = 5ms;
    ASSERT_TRUE(r.driver->configure(headroom_session(), false, false, false));
    ASSERT_TRUE(r.driver->show("a.mkv", 0, 40, true));
    ASSERT_TRUE(r.loop.pump_until([&] { return r.shown.size() >= 2; }));
    // Let a token of the playing request reach the queue, then seek before it is pumped.
    const auto before = r.driver->stats().rendered;
    const auto deadline = std::chrono::steady_clock::now() + 1s;
    while (r.driver->stats().rendered == before && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(1ms);
    const std::size_t shown_before = r.shown.size();
    ASSERT_TRUE(r.driver->show("a.mkv", 30, 40, false));
    ASSERT_TRUE(r.loop.pump_until([&] { return r.shown.size() > shown_before && r.shown.back() == 30; }));
    EXPECT_EQ(r.shown.size(), shown_before + 1) << "a frame of the superseded range was presented";
    EXPECT_GE(r.driver->stats().discarded, 1u);
}

TEST(HeadroomPreviewDriver, GradeChangeRejectsFramesRenderedUnderTheOldGrade) {
    Rig r;
    ASSERT_TRUE(r.driver->configure(headroom_session(), false, false, false));
    ASSERT_TRUE(r.driver->show("a.mkv", 4, 10, false));
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (r.driver->stats().pending == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(1ms);
    ASSERT_EQ(r.driver->stats().pending, 1);
    auto s = headroom_session();
    s.nudge_exposure(1);
    ASSERT_TRUE(r.driver->configure(s, false, false, false));   // e.g. an undo landing now
    r.loop.pump_until([&] { return r.driver->stats().pending == 0; });
    EXPECT_TRUE(r.shown.empty());
    ASSERT_TRUE(r.driver->show("a.mkv", 4, 10, false));          // paused: the app seeks again
    ASSERT_TRUE(r.loop.pump_until([&] { return r.shown.size() == 1; }));
    EXPECT_EQ(r.shown[0], 4);
}

TEST(HeadroomPreviewDriver, InvalidGradeRevokesAndRefusesToShow) {
    Rig r;
    Session legacy(Session::SourceSupport::HeadroomV1);   // policy left Disabled
    EXPECT_FALSE(r.driver->configure(legacy, false, false, false));
    EXPECT_FALSE(r.driver->configured());
    auto shown = r.driver->show("a.mkv", 0, 3, false);
    ASSERT_FALSE(shown);
    EXPECT_EQ(shown.error().code, ErrorCode::Unsupported);
    EXPECT_FALSE(r.driver->configure(headroom_session(), false, true, false)) << "model must be off";
    EXPECT_FALSE(r.driver->configure(headroom_session(), false, false, true)) << "temporal hold must be off";
    r.loop.pump_until([] { return false; }, 50ms);
    EXPECT_TRUE(r.shown.empty());
    EXPECT_EQ(r.port->opens, 0);
}

TEST(HeadroomPreviewDriver, RenderFailureIsReportedAndNothingIsShown) {
    Rig r;
    r.port->fail_render = make_error(ErrorCode::IoError, "fake: decoder died");
    ASSERT_TRUE(r.driver->configure(headroom_session(), false, false, false));
    ASSERT_TRUE(r.driver->show("a.mkv", 0, 3, true));
    ASSERT_TRUE(r.loop.pump_until([&] { return !r.failures.empty(); }));
    EXPECT_EQ(r.failures[0].code, ErrorCode::IoError);
    EXPECT_TRUE(r.shown.empty());
    EXPECT_FALSE(r.driver->stats().active);
}

TEST(HeadroomPreviewDriver, PresentationIntegrityFailureStopsAndReports) {
    Rig r;
    r.port->fail_present = make_error(ErrorCode::IntegrityError, "fake: source changed");
    ASSERT_TRUE(r.driver->configure(headroom_session(), false, false, false));
    ASSERT_TRUE(r.driver->show("a.mkv", 0, 3, true));
    ASSERT_TRUE(r.loop.pump_until([&] { return !r.failures.empty(); }));
    EXPECT_EQ(r.failures[0].code, ErrorCode::IntegrityError);
    EXPECT_TRUE(r.shown.empty());
    EXPECT_FALSE(r.driver->stats().active);
    EXPECT_GE(r.driver->stats().rejected, 1u);
}

TEST(HeadroomPreviewDriver, PauseStopsRenderingAndKeepsTheShownFrame) {
    Rig r;
    r.port->render_delay = 5ms;
    ASSERT_TRUE(r.driver->configure(headroom_session(), false, false, false));
    ASSERT_TRUE(r.driver->show("a.mkv", 0, 100, true));
    ASSERT_TRUE(r.loop.pump_until([&] { return r.shown.size() >= 2; }));
    r.driver->pause();
    r.loop.pump_until([] { return false; }, 100ms);
    const auto n = r.shown.size();
    r.loop.pump_until([] { return false; }, 100ms);
    EXPECT_EQ(r.shown.size(), n);
    EXPECT_FALSE(r.driver->stats().active);
    EXPECT_GE(r.port->cancels, 1);
    EXPECT_EQ(r.driver->stats().presented_index, r.shown.back());
}

TEST(HeadroomPreviewDriver, DestructionWithQueuedTokensIsSafe) {
    auto r = std::make_unique<Rig>();
    ASSERT_TRUE(r->driver->configure(headroom_session(), false, false, false));
    ASSERT_TRUE(r->driver->show("a.mkv", 0, 10, true));
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (r->driver->stats().pending == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(1ms);
    r->driver.reset();   // joins the worker; queued tasks become no-ops
    r->loop.pump_once();
    EXPECT_TRUE(r->shown.empty());
}

TEST(HeadroomPreviewDriver, ViewImageConvertsUnitsAndPrimariesWithoutClamping) {
    Image<space::SceneLinear2020> img(1, 3);
    for (int c = 0; c < 3; ++c) {
        img.buffer().at(c, 0, 0) = 1.0f;   // 203-nit white
        img.buffer().at(c, 0, 1) = 8.0f;   // a super-white at 1624 nits
    }
    img.buffer().at(1, 0, 2) = 1.0f;       // pure Rec.2020 green
    const auto v = headroom_view_image(img);
    for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(v.buffer().at(c, 0, 0), 0.0203f, 1e-5f);
        EXPECT_NEAR(v.buffer().at(c, 0, 1), 8.0f * 0.0203f, 1e-4f) << "highlights must not be clamped";
    }
    EXPECT_LT(v.buffer().at(0, 0, 2), 0.0f) << "out-of-709 green keeps its negative red";
    EXPECT_GT(v.buffer().at(1, 0, 2), 0.0203f);
}

TEST(HeadroomPreviewDriver, SuperWhiteDetailIsToldApartFromCeilingClipping) {
    PlanarBuffer px(3, 1, 5, 0.5f);
    for (int c = 0; c < 3; ++c) {
        px.at(c, 0, 1) = 1.0f;                        // at reference white
        px.at(c, 0, 2) = 1.04f;                       // recorded super-white detail
        px.at(c, 0, 3) = (1019.0f - 64.0f) / 876.0f;  // 10-bit ceiling: clipped
    }
    px.at(0, 0, 4) = 1.02f;                           // one channel above white is detail too
    const auto s = super_white_stats(px);
    EXPECT_DOUBLE_EQ(s.at_white_pct, 20.0);
    EXPECT_DOUBLE_EQ(s.detail_pct, 40.0);
    EXPECT_DOUBLE_EQ(s.ceiling_pct, 20.0);
    EXPECT_NEAR(s.max_code, 1.0902f, 1e-3f);
    PlanarBuffer eight(3, 1, 1, (254.0f - 16.0f) / 219.0f);   // 8-bit ceiling
    EXPECT_DOUBLE_EQ(super_white_stats(eight).ceiling_pct, 100.0);
}
