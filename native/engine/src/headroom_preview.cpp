#include "rudra/engine/headroom_preview.hpp"

#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>

#include "rudra/core/gamut.hpp"
#include "rudra/engine/headroom_presentation.hpp"
#include "rudra/platform/process.hpp"

namespace rudra {

NetworkLinearImage headroom_view_image(const Image<space::SceneLinear2020>& linear2020) {
    // E3: one fused pass (matrix and unit scale in float, rows split across
    // threads) instead of a double-precision conversion and a second scale
    // pass. A view image only: masters and exports never go through here.
    const PlanarBuffer& in = linear2020.buffer();
    const int h = in.height(), w = in.width();
    PlanarBuffer out(3, h, w);
    const Mat3 m = rgb_to_rgb_matrix(Primaries::Rec2020, Primaries::Rec709);
    constexpr double k = 203.0 / 10000.0;   // 1.0 = 203 nits -> 1.0 = 10 000 nits; no clamp at white
    float f[3][3];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) f[r][c] = float(m[std::size_t(r)][std::size_t(c)] * k);
    const float *ir = in.plane(0), *ig = in.plane(1), *ib = in.plane(2);
    float *orr = out.plane(0), *og = out.plane(1), *ob = out.plane(2);
    auto rows = [&](int y0, int y1) {
        const std::size_t a = std::size_t(y0) * std::size_t(w), b = std::size_t(y1) * std::size_t(w);
        for (std::size_t i = a; i < b; ++i) {
            const float r = ir[i], g = ig[i], bl = ib[i];
            orr[i] = f[0][0] * r + f[0][1] * g + f[0][2] * bl;
            og[i] = f[1][0] * r + f[1][1] * g + f[1][2] * bl;
            ob[i] = f[2][0] * r + f[2][1] * g + f[2][2] * bl;
        }
    };
    const std::size_t pixels = std::size_t(h) * std::size_t(w);
    const int threads = pixels < (1u << 18) ? 1 : int(std::clamp(std::thread::hardware_concurrency(), 1u, 8u));
    if (threads == 1) {
        rows(0, h);
    } else {
        std::vector<std::thread> pool;
        const int step = (h + threads - 1) / threads;
        for (int t = 0; t < threads; ++t) {
            const int y0 = t * step, y1 = std::min(h, y0 + step);
            if (y0 < y1) pool.emplace_back(rows, y0, y1);
        }
        for (auto& t : pool) t.join();
    }
    return NetworkLinearImage(std::move(out));
}

SuperWhiteStats super_white_stats(const PlanarBuffer& rgb) {
    SuperWhiteStats s;
    const std::size_t n = rgb.plane_size();
    if (!n || rgb.channels() < 3) return s;
    constexpr float kCode10 = 1.0f / 876.0f;
    constexpr float kWhiteTol = 0.5f * kCode10;
    // The lowest of the two ceilings: 8-bit (254-16)/219, 10-bit (1019-64)/876.
    constexpr float kCeiling = std::min((254.0f - 16.0f) / 219.0f, (1019.0f - 64.0f) / 876.0f) - kWhiteTol;
    const float *r = rgb.plane(0), *g = rgb.plane(1), *b = rgb.plane(2);
    std::size_t detail = 0, ceiling = 0, white = 0;
    float top = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        const float m = std::max(r[i], std::max(g[i], b[i]));
        top = std::max(top, m);
        if (m >= kCeiling) ++ceiling;
        else if (m > 1.0f + kWhiteTol) ++detail;
        else if (m >= 1.0f - kWhiteTol) ++white;
    }
    const double k = 100.0 / double(n);
    s.detail_pct = double(detail) * k;
    s.ceiling_pct = double(ceiling) * k;
    s.at_white_pct = double(white) * k;
    s.max_code = top;
    return s;
}

namespace {
std::size_t image_bytes(const PlanarBuffer& b) { return b.span().size() * sizeof(float); }

class PresentationPort final : public HeadroomPreviewPort {
public:
    PresentationPort(std::unique_ptr<HeadroomPresentation> p, std::filesystem::path decoder)
        : p_(std::move(p)), decoder_(std::move(decoder)) {}
    ~PresentationPort() override { p_->cancel(); }

    Result<void> configure(const Session& s, bool grain, bool model, bool steady) override {
        return p_->set_session(s, grain, model, steady);
    }
    Result<void> open(const std::filesystem::path& source, FloatFrameRange range) override {
        auto r = p_->open(source, decoder_, range);
        if (!r) return r.error();
        return {};
    }
    Result<void> seek(FloatFrameRange range) override {
        auto r = p_->seek(range);
        if (!r) return r.error();
        return {};
    }
    Result<std::shared_ptr<Pending>> render_next(std::chrono::milliseconds wait) override {
        auto frame = p_->render_next(wait);
        if (!frame) return frame.error();
        auto out = std::make_shared<Token>(std::move(*frame));
        const auto& master = out->frame.master().linear2020;
        const auto& base = out->frame.baseline().linear2020;
        out->shown.index = out->frame.index();
        out->shown.width = master.width();
        out->shown.height = master.height();
        out->shown.model = std::make_shared<const NetworkLinearImage>(headroom_view_image(master));
        out->shown.baseline = std::make_shared<const NetworkLinearImage>(headroom_view_image(base));
        out->shown.source_whites = super_white_stats(out->frame.source().encoded_rgb);
        out->index = out->shown.index;
        out->rendered_bytes = image_bytes(master.buffer()) + image_bytes(base.buffer()) +
                              image_bytes(out->shown.model->buffer()) + image_bytes(out->shown.baseline->buffer());
        out->shown.rendered_bytes = image_bytes(out->shown.model->buffer()) + image_bytes(out->shown.baseline->buffer());
        return std::shared_ptr<Pending>(std::move(out));
    }
    Result<void> present(const Pending& pending, const std::function<void(const HeadroomShown&)>& display) override {
        const auto* token = dynamic_cast<const Token*>(&pending);
        if (!token) return make_error(ErrorCode::InvalidArgument, "Headroom preview token belongs to another port");
        return p_->present(token->frame, [&](const HeadroomPresentation::Frame&) { display(token->shown); });
    }
    void cancel() override { p_->cancel(); }

private:
    struct Token final : Pending {
        explicit Token(HeadroomPresentation::Frame f) : frame(std::move(f)) {}
        HeadroomPresentation::Frame frame;
        HeadroomShown shown;
    };
    std::unique_ptr<HeadroomPresentation> p_;
    std::filesystem::path decoder_;
};
}  // namespace

Result<std::unique_ptr<HeadroomPreviewPort>> make_headroom_presentation_port(const std::filesystem::path& decoder,
                                                                             const std::filesystem::path& log) {
    std::filesystem::path exe = decoder;
    if (exe.empty()) {
        auto found = require_executable("ffmpeg");
        if (!found) return found.error();
        exe = *found;
    }
    auto p = HeadroomPresentation::create({}, log);
    if (!p) return p.error();
    return std::unique_ptr<HeadroomPreviewPort>(new PresentationPort(std::move(*p), std::move(exe)));
}

// Shared with tasks posted to the display thread, so a task that runs after
// the driver is gone does nothing. `owner` is read and cleared on that thread only.
struct HeadroomPreviewDriver::State {
    mutable std::mutex mu;
    std::condition_variable cv;
    std::uint64_t epoch = 0;
    bool active = false, playing = false, finished = false, stop = false;
    int pending = 0;
    std::size_t pending_bytes = 0, shown_bytes = 0;
    std::uint64_t rendered = 0, presented = 0, discarded = 0, rejected = 0;
    int presented_index = -1;
    HeadroomPreviewDriver* owner = nullptr;   // display thread only
};

Result<std::unique_ptr<HeadroomPreviewDriver>> HeadroomPreviewDriver::create(std::unique_ptr<HeadroomPreviewPort> port,
                                                                             Callbacks cb, int max_pending) {
    if (!port) return make_error(ErrorCode::InvalidArgument, "Headroom preview needs a renderer");
    if (!cb.post || !cb.shown) return make_error(ErrorCode::InvalidArgument, "Headroom preview needs post and display callbacks");
    if (max_pending < 1 || max_pending > 4) return make_error(ErrorCode::InvalidArgument, "Invalid headroom preview depth");
    return std::unique_ptr<HeadroomPreviewDriver>(new HeadroomPreviewDriver(std::move(port), std::move(cb), max_pending));
}

HeadroomPreviewDriver::HeadroomPreviewDriver(std::unique_ptr<HeadroomPreviewPort> port, Callbacks cb, int max_pending)
    : port_(std::move(port)), cb_(std::move(cb)), max_pending_(max_pending), state_(std::make_shared<State>()) {
    state_->owner = this;
    worker_ = std::thread([this] { run(); });
}

HeadroomPreviewDriver::~HeadroomPreviewDriver() {
    {
        std::lock_guard lock(state_->mu);
        state_->stop = true;
        state_->active = false;
        ++state_->epoch;
        state_->owner = nullptr;   // queued display tasks become no-ops
    }
    state_->cv.notify_all();
    port_->cancel();   // unblocks a render waiting on the decoder
    if (worker_.joinable()) worker_.join();
}

std::uint64_t HeadroomPreviewDriver::bump(bool active, bool playing) {
    std::uint64_t e;
    {
        std::lock_guard lock(state_->mu);
        e = ++state_->epoch;
        state_->active = active;
        state_->playing = playing;
        state_->finished = false;
    }
    state_->cv.notify_all();
    return e;
}

Result<void> HeadroomPreviewDriver::configure(const Session& s, bool grain, bool model, bool steady) {
    auto r = port_->configure(s, grain, model, steady);
    configured_ = bool(r);
    if (!r) {
        bump(false, false);   // revoked: nothing renders until a valid grade and a new show()
        return r;
    }
    // A playing range continues with the new grade (the port rejects frames
    // rendered under the old one); a paused frame needs show() again.
    std::lock_guard lock(state_->mu);
    ++state_->epoch;
    state_->cv.notify_all();
    return {};
}

Result<void> HeadroomPreviewDriver::show(const std::filesystem::path& source, int index, int count, bool play) {
    if (!configured_)
        return make_error(ErrorCode::Unsupported, "Choose a supported headroom grade before previewing");
    if (index < 0 || count <= 0 || index >= count)
        return make_error(ErrorCode::InvalidArgument, "Headroom preview frame is outside the shot");
    const FloatFrameRange range{index, play ? count - index : 1};
    Result<void> r = opened_ == source && !opened_.empty() ? port_->seek(range) : port_->open(source, range);
    if (!r) {
        if (opened_ != source) opened_.clear();
        bump(false, false);
        return r;
    }
    opened_ = source;
    bump(true, play);
    return {};
}

void HeadroomPreviewDriver::pause() {
    bump(false, false);
    port_->cancel();
}

HeadroomPreviewStats HeadroomPreviewDriver::stats() const {
    std::lock_guard lock(state_->mu);
    HeadroomPreviewStats s;
    s.epoch = state_->epoch;
    s.rendered = state_->rendered;
    s.presented = state_->presented;
    s.discarded = state_->discarded;
    s.rejected = state_->rejected;
    s.pending = state_->pending;
    s.presented_index = state_->presented_index;
    s.pending_rendered_bytes = state_->pending_bytes;
    s.shown_rendered_bytes = state_->shown_bytes;
    s.active = state_->active;
    s.playing = state_->playing;
    s.finished = state_->finished;
    return s;
}

void HeadroomPreviewDriver::run() {
    auto s = state_;
    for (;;) {
        std::uint64_t epoch;
        {
            std::unique_lock lock(s->mu);
            s->cv.wait(lock, [&] { return s->stop || (s->active && s->pending < max_pending_); });
            if (s->stop) return;
            epoch = s->epoch;
        }
        auto r = port_->render_next(std::chrono::milliseconds(50));
        std::unique_lock lock(s->mu);
        if (s->stop) return;
        if (r) {
            // Always handed to the display thread: only the port can tell
            // whether a frame rendered across a request change belongs to the
            // new range (its present() rejects stale ones). Dropping it here
            // could lose the only frame of a paused seek.
            ++s->rendered;
            ++s->pending;
            s->pending_bytes += (*r)->rendered_bytes;
            lock.unlock();
            std::weak_ptr<State> weak = s;
            cb_.post([weak, epoch, p = std::move(*r)]() mutable {
                auto st = weak.lock();
                if (!st || !st->owner) return;   // the driver is gone; the token is released here
                st->owner->deliver(epoch, std::move(p));
            });
            continue;
        }
        const ErrorCode code = r.error().code;
        if (code == ErrorCode::Busy) continue;   // nothing decoded yet (bounded 50 ms wait)
        if (epoch != s->epoch || !s->active) continue;
        if (code == ErrorCode::Cancelled) {
            // Superseded inside the port (a grade or seek the driver has not
            // seen yet, or a cancelled decoder). Back off instead of spinning.
            s->cv.wait_for(lock, std::chrono::milliseconds(5), [&] { return s->stop || s->epoch != epoch; });
            continue;
        }
        s->active = false;
        std::weak_ptr<State> weak = s;
        if (code == ErrorCode::NotFound) {   // the range is exhausted
            s->finished = true;
            lock.unlock();
            cb_.post([weak, epoch] {
                auto st = weak.lock();
                if (st && st->owner) st->owner->finish(epoch);
            });
            continue;
        }
        lock.unlock();
        cb_.post([weak, epoch, e = r.error()]() mutable {
            auto st = weak.lock();
            if (st && st->owner) st->owner->fail(epoch, std::move(e));
        });
    }
}

void HeadroomPreviewDriver::deliver(std::uint64_t epoch, std::shared_ptr<HeadroomPreviewPort::Pending> p) {
    bool current;
    {
        std::lock_guard lock(state_->mu);
        current = epoch == state_->epoch;
    }
    const std::size_t bytes = p->rendered_bytes;
    const int index = p->index;
    std::size_t kept = 0;   // what the display keeps after the token is released
    // The port validates source, generation and grade revision under its lock.
    const Result<void> shown = port_->present(*p, [&](const HeadroomShown& f) {
        kept = f.rendered_bytes;
        cb_.shown(f);
    });
    const bool stale = !shown && shown.error().code == ErrorCode::Cancelled;
    {
        std::lock_guard lock(state_->mu);
        --state_->pending;
        state_->pending_bytes -= bytes;
        if (shown) {
            ++state_->presented;
            state_->presented_index = index;
            state_->shown_bytes = kept;
        } else if (stale) {
            ++state_->discarded;
        } else {
            ++state_->rejected;
        }
    }
    p.reset();   // the source token's pixel charge returns before the worker renders again
    state_->cv.notify_all();
    if (!shown && !stale && current) {
        // An identity or integrity failure at presentation: stop, report, show nothing older.
        bump(false, false);
        if (cb_.failed) cb_.failed(shown.error());
    }
}

void HeadroomPreviewDriver::fail(std::uint64_t epoch, Error e) {
    {
        std::lock_guard lock(state_->mu);
        if (epoch != state_->epoch) return;   // a newer request replaced the failed one
    }
    if (cb_.failed) cb_.failed(e);
}

void HeadroomPreviewDriver::finish(std::uint64_t epoch) {
    {
        std::lock_guard lock(state_->mu);
        if (epoch != state_->epoch) return;
    }
    if (cb_.finished) cb_.finished();
}

}  // namespace rudra
