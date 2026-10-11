#include <gtest/gtest.h>

#include "rudra/engine/scheduling.hpp"
#include "rudra/render/viewer_backend.hpp"

using namespace rudra;

TEST(Generation, ASeekMakesOlderResultsStale) {
    Generation g;
    const auto requested = g.current();
    EXPECT_TRUE(g.is_current(requested));
    g.bump();   // the user seeks while the request is in flight
    EXPECT_FALSE(g.is_current(requested));
    EXPECT_TRUE(g.is_current(g.current()));
}

TEST(OutputScale, EachSwapchainGetsItsOwnUnit) {
    const Nits thousand{1000.0f}, sdr_white{203.0f};
    EXPECT_FLOAT_EQ(output_scale(OutputPath::ScRgb, thousand, sdr_white), 12.5f);        // 1.0 = 80 nits
    EXPECT_FLOAT_EQ(output_scale(OutputPath::Edr, thousand, sdr_white), 1000.0f / 203.0f);
    EXPECT_FLOAT_EQ(output_scale(OutputPath::Hdr10, thousand, sdr_white), 0.1f);          // then PQ
}

// ---- the frame path (Phase 2 step 10) ----------------------------------------

#include <atomic>
#include <cmath>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include "rudra/engine/frame_engine.hpp"
#include "rudra/engine/playback_hold.hpp"

namespace {

// A frame whose every pixel says which frame it is, and fields that say it too:
// a result delivered for the wrong frame is caught by value, not by bookkeeping.
struct Fakes {
    std::chrono::milliseconds decode_time{1}, infer_time{8};
    std::atomic<int> decoded{0}, inferred{0};
    FrameLoader loader() {
        return [this](int i) -> Result<SdrImage> {
            ++decoded;
            std::this_thread::sleep_for(decode_time);
            return SdrImage(PlanarBuffer(3, 2, 2, float(i)));
        };
    }
    FrameInfer infer() {
        return [this](const SdrImage& s) -> Result<FrameResult> {
            ++inferred;
            std::this_thread::sleep_for(infer_time);
            FrameResult r;
            const float tag = s.buffer().span()[0];
            r.fields = Fields{PlanarBuffer(3, 2, 2, tag), PlanarBuffer(1, 2, 2, tag), PlanarBuffer(1, 2, 2, tag)};
            return r;
        };
    }
};

struct Seen {
    std::mutex mu;
    std::vector<ReadyFrame> frames;
    void add(const ReadyFrame& f) {
        std::lock_guard lk(mu);
        frames.push_back(f);
    }
    std::vector<ReadyFrame> all() {
        std::lock_guard lk(mu);
        return frames;
    }
};

bool wait_for(const std::function<bool()>& done, int ms = 3000) {
    for (int t = 0; t < ms && !done(); t += 2) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    return done();
}

}  // namespace

TEST(FrameEngine, DeliversOnlyTheFrameItWasAskedFor) {
    Fakes fk;
    FrameEngine engine(fk.loader(), fk.infer(), {32, 4});
    Seen seen;
    engine.on_ready([&](const ReadyFrame& f) { seen.add(f); });
    engine.set_sequence(240);
    // A fast scrub: 0, 50, 120, 180 without waiting.
    for (int i : {0, 50, 120, 180}) engine.show(i);
    ASSERT_TRUE(wait_for([&] { auto v = seen.all(); return !v.empty() && v.back().index == 180; }));
    for (const auto& f : seen.all()) {
        ASSERT_FALSE(f.error.has_value());
        EXPECT_EQ(f.sdr->buffer().span()[0], float(f.index)) << "a frame showed another frame's pixels";
        EXPECT_EQ(f.fields->fields.residual.span()[0], float(f.index)) << "a frame showed another frame's fields";
    }
    // Only the last request may be delivered once the scrub settles.
    EXPECT_EQ(seen.all().back().index, 180);
    const auto st = engine.stats();
    EXPECT_GE(st.cancelled, 1u);   // queued work of older generations never ran
}

TEST(FrameEngine, ScrubbingAFolderNeverShowsAStaleFrame) {
    Fakes fk;
    fk.infer_time = std::chrono::milliseconds(3);
    FrameEngine engine(fk.loader(), fk.infer(), {32, 12});
    Seen seen;
    // `pending` is set before show(i) and `asked` once it returned: a frame
    // delivered while show(i) is being called may be either, and after it
    // returned nothing older may arrive (FrameEngine::deliver_mu_).
    std::atomic<int> pending{-1}, asked{-1};
    std::atomic<int> wrong{0};
    engine.on_ready([&](const ReadyFrame& f) {
        if (f.index != asked.load() && f.index != pending.load()) ++wrong;   // a request no longer current
        seen.add(f);
    });
    engine.set_sequence(240);
    for (int i = 0; i < 240; ++i) {
        pending = i;
        engine.show(i);
        asked = i;
        std::this_thread::sleep_for(std::chrono::microseconds(700));   // faster than inference
    }
    ASSERT_TRUE(wait_for([&] { auto v = seen.all(); return !v.empty() && v.back().index == 239; }));
    EXPECT_EQ(wrong.load(), 0);
    for (const auto& f : seen.all()) EXPECT_EQ(f.sdr->buffer().span()[0], float(f.index));
    const auto st = engine.stats();
    EXPECT_EQ(st.shown, 240u);
    EXPECT_LE(st.cached, 32);
    EXPECT_LT(fk.inferred.load(), 240 + 12 + 1);   // cancelled work was not run
}

TEST(FrameEngine, ReadAheadMakesTheNextFramesFreeAndTheCacheIsBounded) {
    Fakes fk;
    FrameEngine engine(fk.loader(), fk.infer(), {8, 5});
    Seen seen;
    engine.on_ready([&](const ReadyFrame& f) { seen.add(f); });
    engine.set_sequence(20);
    engine.show(0);
    ASSERT_TRUE(wait_for([&] { return engine.is_cached(5); }));   // 1..5 read ahead
    const auto before = fk.inferred.load();
    engine.show(3);   // already read ahead: delivered at once, from the cache
    auto v = seen.all();
    ASSERT_FALSE(v.empty());
    EXPECT_EQ(v.back().index, 3);
    EXPECT_TRUE(v.back().from_cache);
    EXPECT_EQ(fk.inferred.load(), before);   // no inference for it
    for (int i = 0; i < 20; ++i) engine.show(i);
    ASSERT_TRUE(wait_for([&] { auto w = seen.all(); return w.back().index == 19; }));
    EXPECT_LE(engine.stats().cached, 8);
    EXPECT_TRUE(engine.is_cached(19));   // the frame on screen is never evicted
}

TEST(FrameEngine, AFailedFrameIsReportedNotShownAsAnother) {
    Fakes fk;
    FrameLoader bad = [&](int i) -> Result<SdrImage> {
        if (i == 7) return make_error(ErrorCode::InvalidArgument, "frame 7 is not an image");
        return SdrImage(PlanarBuffer(3, 2, 2, float(i)));
    };
    FrameEngine engine(bad, fk.infer(), {8, 0});
    Seen seen;
    engine.on_ready([&](const ReadyFrame& f) { seen.add(f); });
    engine.set_sequence(10);
    engine.show(7);
    ASSERT_TRUE(wait_for([&] { return !seen.all().empty(); }));
    const auto f = seen.all().back();
    EXPECT_EQ(f.index, 7);
    ASSERT_TRUE(f.error.has_value());
    EXPECT_EQ(f.error->message, "frame 7 is not an image");
    EXPECT_EQ(engine.stats().errors, 1u);
}

// Playback with the model (10 Oct 2026): ticks no longer cancel the frames
// ahead of the playhead, so inference runs continuously in front of it and the
// ticks are served from the cache; a proxy frame the playhead stops on is shown
// at once and inferred again at full size.
TEST(FrameEngine, PlaybackKeepsInferenceAheadOfThePlayhead) {
    Fakes fk;
    fk.infer_time = std::chrono::milliseconds(4);
    FrameEngine engine(fk.loader(), fk.infer(), {64, 4, 16});
    Seen seen;
    engine.on_ready([&](const ReadyFrame& f) { seen.add(f); });
    engine.set_sequence(120);
    engine.set_playing(true);
    engine.show(0);
    ASSERT_TRUE(wait_for([&] { return engine.stats().inferred >= 17; }));   // the frame and 16 ahead
    // 12 ticks: every one from the cache, nothing computed twice.
    const auto before = fk.inferred.load();
    for (int i = 1; i <= 12; ++i) {
        engine.show(i);
        ASSERT_TRUE(wait_for([&] { auto v = seen.all(); return !v.empty() && v.back().index == i; })) << i;
    }
    EXPECT_GE(engine.stats().cache_hits, 12u);
    ASSERT_TRUE(wait_for([&] { return engine.is_cached(12 + 16); }));   // and it kept going
    EXPECT_LE(fk.inferred.load() - before, 12 + 2);   // the new frames ahead only
    // Without playback the same ticks cancel what was queued.
    engine.set_playing(false);
    for (int i = 60; i < 70; ++i) engine.show(i);
    EXPECT_GE(engine.stats().cancelled, 1u);
}

TEST(FrameEngine, AProxyUnderAStoppedPlayheadIsInferredAgainAtFullSize) {
    Fakes fk;
    std::atomic<bool> playing{true};
    std::atomic<int> full{0};
    FrameInfer infer = [&](const SdrImage& s) -> Result<FrameResult> {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        FrameResult r;
        const float tag = s.buffer().span()[0];
        r.fields = Fields{PlanarBuffer(3, 2, 2, tag), PlanarBuffer(1, 2, 2, tag), PlanarBuffer(1, 2, 2, tag)};
        r.proxy = playing.load();
        if (!r.proxy) ++full;
        return r;
    };
    FrameEngine engine(fk.loader(), infer, {32, 2, 4});
    Seen seen;
    engine.on_ready([&](const ReadyFrame& f) { seen.add(f); });
    engine.set_sequence(30);
    engine.set_playing(true);
    engine.show(5);
    ASSERT_TRUE(wait_for([&] { return engine.is_cached(9); }));
    playing = false;
    engine.set_playing(false);
    engine.show(7);   // stopped on a proxy
    ASSERT_TRUE(wait_for([&] {
        auto v = seen.all();
        return !v.empty() && v.back().index == 7 && v.back().fields && !v.back().fields->proxy;
    }));
    const auto v = seen.all();
    ASSERT_GE(v.size(), 2u);
    EXPECT_TRUE(v[v.size() - 2].index == 7 && v[v.size() - 2].fields->proxy);   // the proxy first, at once
    EXPECT_EQ(full.load(), 1);
    // Shown again: the full frame, from the cache, not inferred a third time.
    engine.show(7);
    ASSERT_TRUE(wait_for([&] { return seen.all().size() == v.size() + 1; }));
    EXPECT_FALSE(seen.all().back().fields->proxy);
    EXPECT_EQ(full.load(), 1);
}

// Temporal consistency while playing (10 Oct 2026): the shadow weight and the
// anchor's hold are held across the shot as an export holds them; skipped
// frames count as elapsed, a cut or a step back starts from the frame's own.
namespace {
SdrImage flat_sdr(float v) { return SdrImage(PlanarBuffer(3, 18, 32, v)); }
}  // namespace

TEST(PlaybackHold, HoldsTheWeightAcrossTheShotAndResetsAtACut) {
    PlaybackHold h;
    h.configure(24.0, kShotHoldSeconds);
    const double r = std::exp(-1.0 / 6.0);
    EXPECT_DOUBLE_EQ(h.retention(), r);
    EXPECT_DOUBLE_EQ(h.weight(0, flat_sdr(0.5f), 1.0), 1.0);                 // a shot starts at its own
    EXPECT_DOUBLE_EQ(h.weight(1, flat_sdr(0.5f), 0.0), r);                   // r*1 + (1-r)*0
    EXPECT_DOUBLE_EQ(h.weight(1, flat_sdr(0.5f), 0.7), r);                   // the same frame again
    EXPECT_DOUBLE_EQ(h.weight(4, flat_sdr(0.5f), 0.0), r * r * r * r);      // two frames skipped: r^3
    EXPECT_EQ(h.shot_start(), 0);
    EXPECT_DOUBLE_EQ(h.weight(5, flat_sdr(0.9f), 0.3), 0.3);                 // a cut
    EXPECT_EQ(h.shot_start(), 5);
    EXPECT_DOUBLE_EQ(h.weight(2, flat_sdr(0.9f), 0.6), 0.6);                 // back to the start
    h.configure(24.0, 0.0);                                                    // off
    h.weight(0, flat_sdr(0.5f), 1.0);
    EXPECT_DOUBLE_EQ(h.weight(1, flat_sdr(0.5f), 0.2), 0.2);
    EXPECT_EQ(h.anchor(1, 3.3), 3.3);                                         // exactly its own
}

TEST(PlaybackHold, HoldsTheAnchorInStops) {
    PlaybackHold h;
    h.configure(24.0, kShotHoldSeconds);
    const double r = std::exp(-1.0 / 6.0);
    h.weight(0, flat_sdr(0.5f), 1.0);
    h.weight(1, flat_sdr(0.5f), 1.0);
    h.weight(2, flat_sdr(0.5f), 1.0);
    EXPECT_DOUBLE_EQ(h.anchor(0, 2.0), 2.0);
    EXPECT_NEAR(h.anchor(2, 8.0), std::exp2(r * r * 1.0 + (1.0 - r * r) * 3.0), 1e-12);   // log2: 1 -> 3
    EXPECT_NEAR(h.anchor(2, 1.0), std::exp2(r * r * 1.0 + (1.0 - r * r) * 3.0), 1e-12);   // measured again
    h.weight(3, flat_sdr(0.9f), 1.0);                                                       // a cut at 3
    EXPECT_DOUBLE_EQ(h.anchor(2, 5.0), 5.0);    // a late measure from the shot before: its own, nothing held
    EXPECT_DOUBLE_EQ(h.anchor(3, 4.0), 4.0);    // the new shot starts at its own
    EXPECT_DOUBLE_EQ(h.anchor(4, 0.0), 0.0);    // no hold to hold
}
