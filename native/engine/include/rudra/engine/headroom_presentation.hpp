#pragma once

#include <functional>
#include <mutex>
#include "rudra/engine/session.hpp"
#include "rudra/video/headroom_playback.hpp"
#include "rudra/video/headroom_render.hpp"

namespace rudra {

// Immutable preview, deliberately not a verified frame or export receipt.
// The source token retains its charge in the decoder's shared pixel allowance.
// The two rendered images are additional CPU memory, not part of that allowance.
class HeadroomPresentationFrame {
public:
    int index() const { return source_.index(); }
    const HeadroomMaster& master() const { return master_; }
    const HeadroomBaseline& baseline() const { return baseline_; }
    // The decoded source values (normalized code values; super-whites above 1).
    const ExtendedVideoFrame& source() const { return source_.pixels(); }
private:
    friend class HeadroomPresentation;
    HeadroomPresentationFrame(HeadroomPlaybackFrame source, HeadroomMaster master,
        HeadroomBaseline baseline, std::uint64_t revision)
        : source_(std::move(source)), master_(std::move(master)), baseline_(std::move(baseline)), revision_(revision) {}
    HeadroomPlaybackFrame source_;
    HeadroomMaster master_;
    HeadroomBaseline baseline_;
    std::uint64_t revision_;
};

// Session-to-preview bridge, shared by paused and continuous ranges. Not yet
// exposed in MainWindow: scopes and transactional exports must be integrated
// before the public source selector/default project reader can enable it.
// set_session is called on the session's owner thread after every grade/delivery
// change (including undo), before presenting. It snapshots immutable parameters;
// it never changes artist controls to make them pass. Invalid settings revoke
// previously rendered output too. A paused grade change needs a new seek.
// render_next runs off the UI thread. present runs the final synchronous display
// callback under the same lock as seek/open/cancel/grade changes, after checking
// the source token. The callback must not re-enter this object or queue a naked
// frame for later presentation. Destruction joins workers; join external callers
// first. No implicit legacy SDR conversion, model, temporal hold or export path.
class HeadroomPresentation {
public:
    using Frame = HeadroomPresentationFrame;
    using Generation = HeadroomPlayback::Generation;
    static Result<std::unique_ptr<HeadroomPresentation>> create(HeadroomPlaybackLimits limits = {},
        const std::filesystem::path& log = {});
    Result<void> set_session(const Session& session, bool settle_grain, bool use_model, bool steady_shots);
    Result<Generation> open(const std::filesystem::path& source, const std::filesystem::path& decoder,
        FloatFrameRange range, const std::filesystem::path& probe = {});
    Result<Generation> open(PreparedHeadroomSource::Handle prepared, FloatFrameRange range);
    Result<Generation> seek(FloatFrameRange range);
    Result<Frame> render_next(std::chrono::milliseconds wait = {});
    Result<void> present(const Frame& frame, const std::function<void(const Frame&)>& display);
    void cancel();
    HeadroomPlaybackStats stats() const { return playback_->stats(); }
private:
    explicit HeadroomPresentation(std::unique_ptr<HeadroomPlayback> playback) : playback_(std::move(playback)) {}
    std::unique_ptr<HeadroomPlayback> playback_;
    std::mutex mutex_;
    std::mutex render_mutex_;
    std::uint64_t revision_ = 0;
    std::uint64_t presented_revision_ = 0;
    int presented_index_ = -1;
    Generation generation_ = 0;
    std::optional<MasterRequest> grade_;
    Error grade_error_ = make_error(ErrorCode::Unsupported, "Choose a supported headroom session before previewing");
};

} // namespace rudra
