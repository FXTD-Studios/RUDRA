#pragma once
// The application-side driver for the opt-in headroom preview (E1, 17 Oct 2026).
// It owns one render worker over a HeadroomPreviewPort (normally the
// HeadroomPresentation bridge) and hands accepted frames to the display thread.
//
// Threading contract (archive/docs/HEADROOM_APP_INTEGRATION_2026-10-17.md):
// - configure/show/pause/stop and destruction run on the display (UI) thread.
// - The worker calls only render_next; it never reads Session or widgets.
// - Rendered frames travel to the display thread as an opaque pending token
//   through the injected `post` function, never as naked pixels: the port
//   validates the token again (present) on the display thread, under the
//   bridge lock, before `shown` sees any pixels.
// - At most `max_pending` tokens are in flight; the worker waits for the
//   display thread before rendering further, so a slow display bounds memory.
// - Every request bumps an epoch; tokens from an older epoch are released
//   without presentation. The port's own revision checks remain authoritative.
// - Errors are reported, never replaced with legacy pixels (no silent fallback).
//
// No Qt dependency: the app passes a `post` that queues onto its event loop.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include "rudra/core/image.hpp"
#include "rudra/engine/session.hpp"
#include "rudra/media/video_decode_float.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

// E4: what the source carries above reference white, per frame, from the
// decoded code values (normalized: 0 = black, 1 = reference white 235/940).
// `detail`: some channel above white but below the code ceiling, i.e. measured
// super-white information the headroom policy keeps. `ceiling`: some channel at
// the format's highest legal code (8-bit 254, 10-bit 1019, within half a code):
// true clipping, nothing above it was recorded. `at_white`: the maximum channel
// within half a code of reference white, which is where a source clipped at
// white piles up (also legitimate white; reported, not judged).
struct SuperWhiteStats {
    double detail_pct = 0.0, ceiling_pct = 0.0, at_white_pct = 0.0;
    float max_code = 0.0f;   // the largest normalized channel value
};
SuperWhiteStats super_white_stats(const PlanarBuffer& encoded_rgb);

// What the display thread receives for an accepted frame. The view images
// are the graded master and the baseline converted once, on the worker, to
// the viewer's precomposited convention (linear Rec.709, 1.0 = 10 000 nits).
// They are derived copies: the source token stays with the port until released.
struct HeadroomShown {
    int index = -1;
    int width = 0, height = 0;
    std::shared_ptr<const NetworkLinearImage> model, baseline;
    SuperWhiteStats source_whites;   // E4, of the decoded source
    // CPU memory of the two view images (outside the decoder's pixel allowance).
    std::size_t rendered_bytes = 0;
};

// Converts a scene-linear Rec.2020 (1.0 = 203 nits) image to the viewer's
// precomposited convention. Out-of-709-gamut values stay negative here; the
// display pass decides how they are shown. Never clamps highlights.
NetworkLinearImage headroom_view_image(const Image<space::SceneLinear2020>& linear2020);

// The seam the driver renders through. The real implementation wraps
// HeadroomPresentation; tests use a fake.
class HeadroomPreviewPort {
public:
    struct Pending {
        virtual ~Pending() = default;
        int index = -1;
        std::size_t rendered_bytes = 0;   // CPU memory this token holds outside the decoder allowance
    };
    virtual ~HeadroomPreviewPort() = default;
    virtual Result<void> configure(const Session& session, bool settle_grain, bool use_model, bool steady_shots) = 0;
    virtual Result<void> open(const std::filesystem::path& source, FloatFrameRange range) = 0;
    virtual Result<void> seek(FloatFrameRange range) = 0;
    // Worker thread. Busy: nothing ready yet; NotFound: the range is exhausted;
    // Cancelled: superseded by a newer request or grade. Anything else is fatal.
    virtual Result<std::shared_ptr<Pending>> render_next(std::chrono::milliseconds wait) = 0;
    // Display thread: validates under the bridge lock and calls `display` once.
    virtual Result<void> present(const Pending& pending, const std::function<void(const HeadroomShown&)>& display) = 0;
    virtual void cancel() = 0;
};

// The production port over HeadroomPresentation. `decoder` is the FFmpeg used
// for float decoding; an empty path is resolved with require_executable.
Result<std::unique_ptr<HeadroomPreviewPort>> make_headroom_presentation_port(
    const std::filesystem::path& decoder = {}, const std::filesystem::path& log = {});

struct HeadroomPreviewStats {
    std::uint64_t epoch = 0;
    std::uint64_t rendered = 0, presented = 0, discarded = 0, rejected = 0;
    int pending = 0;                 // tokens posted but not yet presented or released
    int presented_index = -1;
    std::size_t pending_rendered_bytes = 0;   // rendered CPU memory held by in-flight tokens
    std::size_t shown_rendered_bytes = 0;     // view images of the last presented frame
    bool active = false, playing = false, finished = false;
};

class HeadroomPreviewDriver {
public:
    struct Callbacks {
        // Runs `task` later on the display thread (required).
        std::function<void(std::function<void()>)> post;
        // Display thread, inside the port's present: must not call the driver.
        std::function<void(const HeadroomShown&)> shown;
        // Display thread, after the driver stopped rendering this request.
        std::function<void(const Error&)> failed;
        // Display thread: a playing range reached its end.
        std::function<void()> finished;
    };
    static Result<std::unique_ptr<HeadroomPreviewDriver>> create(std::unique_ptr<HeadroomPreviewPort> port,
                                                                 Callbacks callbacks, int max_pending = 1);
    ~HeadroomPreviewDriver();   // cancels the port, joins the worker, drops tokens
    HeadroomPreviewDriver(const HeadroomPreviewDriver&) = delete;
    HeadroomPreviewDriver& operator=(const HeadroomPreviewDriver&) = delete;

    // Snapshot the grade (after every change, including undo/redo and restore).
    // Failure revokes the current request: nothing renders until a valid grade
    // and a new show(). Success does not re-render by itself (a paused frame
    // needs show() again).
    Result<void> configure(const Session& session, bool settle_grain, bool use_model, bool steady_shots);
    // Paused (play == false): exactly frame `index`. Playing: index..count-1.
    // Opens the source the first time (or when it changes), seeks otherwise.
    Result<void> show(const std::filesystem::path& source, int index, int count, bool play);
    // Stops decoding and drops in-flight tokens; the frame on screen stays.
    void pause();
    bool configured() const { return configured_; }
    const std::filesystem::path& source() const { return opened_; }
    HeadroomPreviewStats stats() const;

private:
    struct State;
    HeadroomPreviewDriver(std::unique_ptr<HeadroomPreviewPort> port, Callbacks callbacks, int max_pending);
    void run();
    void worker_failed(const char* what) noexcept;   // run()'s exception guard
    void deliver(std::uint64_t epoch, std::shared_ptr<HeadroomPreviewPort::Pending> pending);
    void fail(std::uint64_t epoch, Error error);
    void finish(std::uint64_t epoch);
    std::uint64_t bump(bool active, bool playing);

    std::unique_ptr<HeadroomPreviewPort> port_;
    Callbacks cb_;
    int max_pending_;
    std::filesystem::path opened_;
    bool configured_ = false;
    std::shared_ptr<State> state_;
    std::thread worker_;
};

}  // namespace rudra
