#pragma once
// What the app's viewer is fed on the learned ACES 2.0 route (28 Oct 2026):
//   - the monitoring transform: the master as an ACES 2.0 output shows it,
//     ACES 2.0 SDR 100 nits (Rec.709, BT.1886) or HDR 1000 nits (P3 D65 on
//     Rec.2100 PQ), decoded back to display light, so the viewer's own display
//     pass (HDR10 / scRGB / EDR swapchain, or SDR) puts those nits on the
//     monitor. Written in the raw outputs' convention (AP1, 1.0 = 203 nits)
//     so the app's path is the same for every mode.
//   - a playback proxy: the display codes area-averaged down before the model.
//   - the SDR's largest code per pixel, for the Invented layer.

#include <string>

#include "rudra/core/aces_v2.hpp"
#include "rudra/core/image.hpp"

namespace rudra {

enum class AcesMonitor { Scene, Sdr100, Hdr1000 };
// "", "scene", "sdr100", "hdr1000"; anything else is nullopt-like Scene with ok = false.
AcesMonitor aces_monitor_from(const std::string& name, bool* ok = nullptr);

// `view` is the transform for the mode: the SDR view for Sdr100, the HDR one for Hdr1000.
PlanarBuffer aces_monitor_planes(const PlanarBuffer& acescg, const aces_v2::AcesView& view, AcesMonitor m);

// Box (area) average down so the longer side is at most max_side (no change when it already is).
PlanarBuffer aces_area_downscale(const PlanarBuffer& p, int max_side);

// One plane: max(R, G, B) of display codes in [0, 1].
PlanarBuffer aces_code_plane(const PlanarBuffer& codes);

}  // namespace rudra
