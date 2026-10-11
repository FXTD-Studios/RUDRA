#pragma once
// Temporal consistency while playing (10 Oct 2026): the viewer holds the
// per-frame values a movie export holds (video/predictor) -- the shadow
// weight and the anchor's hold -- across the shot as playback shows it, with
// the export's time constant, so what plays is what exports. Frames skipped to
// keep time count as elapsed (retention^frames); a cut (the export's detector
// on the SDR thumbnail) or a step backwards starts again from the frame's own
// value. Stopped, the app shows the frame's own values, as a still master has
// no shot to hold them across.

#include "rudra/core/image.hpp"

namespace rudra {

// The time constant a movie export holds with by default (seconds;
// VideoConvertArgs::temporal_smoothing).
inline constexpr double kShotHoldSeconds = 0.25;

class PlaybackHold {
public:
    // `seconds` 0: no hold (the frames' own values).
    void configure(double fps, double seconds, double cut_threshold = 0.15);
    void reset();
    // The shadow weight to show for frame `index`, whose SDR is `sdr` and whose
    // own weight is `weight`. Called once per frame shown, in playback order.
    double weight(int index, const SdrImage& sdr, double weight);
    // The anchor hold to show for frame `index` of shot `shot` (shot() when
    // that frame was shown), whose own is `hold`. Its measure may arrive after
    // later frames were shown; one from another shot, or from before playback
    // looped back to the start, is the frame's own and holds nothing.
    double anchor(int index, double hold, int shot);
    // The static hold (the viewer's hold pass, ViewerHold) for the frame last
    // passed to weight(): off with no retention; the shot's first frame starts
    // it; a later one holds with retention^(frames since the one before).
    struct PixelStep {
        bool on = false;
        double strength = 0.0;
        bool reset = true;
    };
    PixelStep pixel_step() const;
    double retention() const noexcept { return retention_; }
    int shot_start() const noexcept { return shot_start_; }
    int shot() const noexcept { return shot_; }   // counts every start of a shot, loops included

private:
    double retention_ = 0.0, cut_threshold_ = 0.15;
    int last_ = -1, shot_start_ = -1, anchor_last_ = -1, shot_ = 0;
    int gap_ = 0;   // frames between the last two weight() calls in the shot; 0: it started the shot
    PlanarBuffer thumb_;
    double weight_ = 1.0, log_hold_ = 0.0;
};

}  // namespace rudra
