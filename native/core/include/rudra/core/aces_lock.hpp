#pragma once
// The lock (stage 6b, 7 Nov 2026): RUDRA ACES 2's 0 EV fix, applied after
// constrain_colour_v2 on its output. Executable spec: training/aces_partial_tighten.py
// (lock_observed with target='dequant', dequantize_recorded, the appearance guard).
// Registered arm lock_dqs050g (lockguard_fresh_20261107), the arm the Crew holdout
// passed (docs/competitive/RESULT_CREW_HOLDOUT_2026_11_07.md).
//
// In a partially clipped pixel the clipped channel(s) show 255 whatever the model
// rebuilt; the 0 EV error is all in the channels the camera recorded. Per pixel the
// lock solves (Gauss-Newton, 3x3 Jacobian by finite differences, pseudo-inverse of the
// recorded rows, in log x) for the smallest change that puts the recorded channels'
// display back on the dequantized source, then moves `strength` of the way there
// (geometric, per channel). The appearance guard raises the strength per pixel, in
// quarter steps up to the full lock, until every recorded channel shows within 1 code
// of the source and every clipped channel stays clipped. Pixels outside partial
// support, fully clipped pixels and pixels the solve cannot bring back are returned
// bit-exact. Core stays dependency free: the view is the AcesView interface.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rudra/core/aces_v2.hpp"
#include "rudra/core/image.hpp"

namespace rudra::aces_v2 {

// LockSettings and LockStats are in aces_v2.hpp (run_frame and FrameResult carry them).

// dequantize_recorded: per channel, the smoothest field (least squared Laplacian,
// accelerated projected gradient) inside every recorded code's bin (+-half_bin); a
// clipped sample is bounded below by the clip only. Used where the content is smooth
// (mean |Laplacian| over 5x5 <= max_lap) and the channel is recorded; elsewhere the
// source code. `region` (h*w bytes, optional): the solve runs on its bounding box
// plus 24 px. Values are display codes / 255, planar, double like the Python.
struct Dequantized {
    std::vector<double> values;       // 3 * h * w, planar
    std::vector<std::uint8_t> used;   // 3 * h * w, planar
};
Dequantized dequantize_recorded(const PlanarBuffer& source, const std::vector<std::uint8_t>* region = nullptr,
                                int iters = 200, double max_lap = 1.5, double half_bin = 0.49,
                                double clip_ceiling = 4.0);

// The lock on constrain_colour_v2's output. `weight` is G1's reconstruction weight (1 ch).
PlanarBuffer lock_observed(const PlanarBuffer& source, const PlanarBuffer& constrained, const PlanarBuffer& weight,
                           const AcesView& view, const LockSettings& settings, LockStats* stats = nullptr);

}  // namespace rudra::aces_v2
