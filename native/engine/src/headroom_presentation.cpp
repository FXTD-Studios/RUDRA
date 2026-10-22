#include "rudra/engine/headroom_presentation.hpp"

namespace rudra {
namespace {
Result<MasterRequest> headroom_request(const Session& s, bool grain, bool model, bool steady) {
    if (s.grade.headroom_policy != HeadroomPolicy::Bt1886V1)
        return make_error(ErrorCode::Unsupported, "Headroom preview requires bt1886-headroom-v1");
    if (model || steady)
        return make_error(ErrorCode::Unsupported, "Headroom preview requires model and temporal hold off");
    if (s.grade.source != "bt1886" || s.grade.clip_fill || !s.grade.clip_peaks.empty() ||
        !s.grade.calibration.empty() || !s.grade.reference.empty())
        return make_error(ErrorCode::Unsupported, "Headroom preview requires BT.1886 without fill, peaks, calibration or reference");
    MasterRequest q;
    q.analytic = true;
    q.source_curve = s.grade.source;
    q.clip_fill = s.grade.clip_fill;
    q.anchor = s.anchor; q.anchor_knee = s.anchor_knee;
    q.carry_chroma = s.carry_chroma; q.settle_grain = grain;
    q.container = s.container;
    q.recovery_mode = s.grade.mode; q.strength = s.grade.strength;
    q.preserve_outside = s.grade.preserve;
    q.exposure_ev = s.grade.exposure_ev; q.masks = s.grade.masks;
    for (const auto& r : s.grade.regions) q.regions.push_back({r.label, r.low_nits, r.high_nits, r.ev});
    if (auto valid = validate_headroom_grade(q); !valid) return valid.error();
    return q;
}
Error stale() { return make_error(ErrorCode::Cancelled, "Headroom preview belongs to an earlier source, seek or grade"); }
} // namespace

Result<std::unique_ptr<HeadroomPresentation>> HeadroomPresentation::create(HeadroomPlaybackLimits limits,
    const std::filesystem::path& log) {
    auto playback = HeadroomPlayback::create(limits, log);
    if (!playback) return playback.error();
    return std::unique_ptr<HeadroomPresentation>(new HeadroomPresentation(std::move(*playback)));
}

Result<void> HeadroomPresentation::set_session(const Session& s, bool grain, bool model, bool steady) {
    // The caller owns Session; no worker reads it after this snapshot.
    auto next = headroom_request(s, grain, model, steady);
    std::lock_guard lock(mutex_);
    ++revision_;
    grade_.reset();
    if (!next) { grade_error_ = next.error(); return grade_error_; }
    grade_ = std::move(*next);
    return {};
}

Result<HeadroomPresentation::Generation> HeadroomPresentation::open(const std::filesystem::path& source,
    const std::filesystem::path& decoder, FloatFrameRange range, const std::filesystem::path& probe) {
    std::lock_guard lock(mutex_);
    if (!grade_) return grade_error_;
    auto next = playback_->open(source, HeadroomPolicy::Bt1886V1, decoder, range, probe);
    if (!next) return next.error();
    generation_ = *next; ++revision_;
    return *next;
}
Result<HeadroomPresentation::Generation> HeadroomPresentation::open(PreparedHeadroomSource::Handle prepared,
    FloatFrameRange range) {
    std::lock_guard lock(mutex_);
    if (!grade_) return grade_error_;
    auto next = playback_->open(std::move(prepared), range);
    if (!next) return next.error();
    generation_ = *next; ++revision_;
    return *next;
}
Result<HeadroomPresentation::Generation> HeadroomPresentation::seek(FloatFrameRange range) {
    std::lock_guard lock(mutex_);
    if (!grade_) return grade_error_;
    auto next = playback_->seek(range);
    if (!next) return next.error();
    generation_ = *next; ++revision_;
    return *next;
}
Result<HeadroomPresentation::Frame> HeadroomPresentation::render_next(std::chrono::milliseconds wait) {
    std::lock_guard render_lock(render_mutex_);
    MasterRequest q;
    Generation generation;
    std::uint64_t revision;
    {
        std::lock_guard lock(mutex_);
        if (!grade_) return grade_error_;
        q = *grade_; generation = generation_; revision = revision_;
    }
    auto raw = playback_->take(generation, wait);
    if (!raw) return raw.error();
    q.source_space = raw->pixels().source_contract.primaries;
    auto images = render_headroom_preview(raw->pixels(), q, HeadroomPolicy::Bt1886V1);
    if (!images) return images.error();
    if (auto valid = playback_->validate_preview(*raw); !valid) return valid.error();
    std::lock_guard lock(mutex_);
    if (revision != revision_ || generation != generation_) return stale();
    return Frame(std::move(*raw), std::move(images->master), std::move(images->baseline), revision);
}
Result<void> HeadroomPresentation::present(const Frame& frame, const std::function<void(const Frame&)>& display) {
    if (!display) return make_error(ErrorCode::InvalidArgument, "Headroom display callback is missing");
    // File identity checks are outside our state lock. Recheck the revision
    // inside it, so a concurrent seek cannot publish the validated old result.
    if (auto valid = playback_->validate_preview(frame.source_); !valid) return valid.error();
    std::lock_guard lock(mutex_);
    if (!grade_ || frame.revision_ != revision_ || frame.source_.generation() != generation_) return stale();
    if (presented_revision_ == revision_ && frame.index() < presented_index_) return stale();
    display(frame);
    presented_revision_ = revision_; presented_index_ = frame.index();
    return {};
}
void HeadroomPresentation::cancel() {
    std::lock_guard lock(mutex_);
    ++revision_; generation_ = 0;
    playback_->cancel();
}
} // namespace rudra
