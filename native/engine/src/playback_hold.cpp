#include "rudra/engine/playback_hold.hpp"

#include <cmath>
#include <utility>

#include "rudra/core/video_predict.hpp"

namespace rudra {

void PlaybackHold::configure(double fps, double seconds, double cut_threshold) {
    retention_ = fps > 0.0 && seconds > 0.0 ? std::exp(-1.0 / (fps * seconds)) : 0.0;
    cut_threshold_ = cut_threshold;
    reset();
}

void PlaybackHold::reset() {
    last_ = shot_start_ = anchor_last_ = -1;
    gap_ = 0;
    ++shot_;
    thumb_ = PlanarBuffer();
    weight_ = 1.0;
    log_hold_ = 0.0;
}

double PlaybackHold::weight(int index, const SdrImage& sdr, double w) {
    if (index == last_) return weight_;   // the same frame again
    PlanarBuffer t = area_resize(sdr.buffer(), kCutThumbHeight, kCutThumbWidth);
    bool cut = last_ < 0 || index < last_ || t.span().size() != thumb_.span().size();
    if (!cut) {   // ShadowSmoother::update's detector
        double sum = 0.0;
        const auto a = t.span();
        const auto b = std::as_const(thumb_).span();
        for (std::size_t i = 0; i < a.size(); ++i) sum += std::fabs(static_cast<double>(a[i] - b[i]));
        cut = static_cast<double>(static_cast<float>(sum / static_cast<double>(a.size()))) >= cut_threshold_;
    }
    if (cut) {
        weight_ = w;
        shot_start_ = index;
        anchor_last_ = -1;
        gap_ = 0;
        ++shot_;
    } else {
        gap_ = index - last_;
        const double r = std::pow(retention_, index - last_);
        weight_ = r * weight_ + (1.0 - r) * w;
    }
    last_ = index;
    thumb_ = std::move(t);
    return weight_;
}

PlaybackHold::PixelStep PlaybackHold::pixel_step() const {
    PixelStep s;
    if (retention_ == 0.0 || last_ < 0) return s;
    s.on = true;
    s.reset = gap_ <= 0;
    s.strength = s.reset ? retention_ : std::pow(retention_, gap_);
    return s;
}

double PlaybackHold::anchor(int index, double hold, int shot) {
    // Off, nothing to hold, or a late measure from another shot (a cut, or the
    // pass before playback looped back to the start): the frame's own.
    if (retention_ == 0.0 || !(hold > 0.0) || !std::isfinite(hold) || shot != shot_) return hold;
    if (index == anchor_last_) return std::exp2(log_hold_);
    if (anchor_last_ < 0 || index < anchor_last_) {
        log_hold_ = std::log2(hold);
    } else {
        const double r = std::pow(retention_, index - anchor_last_);
        log_hold_ = r * log_hold_ + (1.0 - r) * std::log2(hold);
    }
    anchor_last_ = index;
    return std::exp2(log_hold_);
}

}  // namespace rudra
