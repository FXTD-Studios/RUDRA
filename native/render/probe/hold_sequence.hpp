#pragma once
// The static hold's parity case (11 Oct 2026): a short shot built to reach
// every branch of ShadowSmoother::hold_pixels, and the C++ reference over it.
// rudra-gpu-parity holds shaders/hold.frag (GpuCompositor::hold_sequence) to
// it, frame by frame.
//
// The frame, 64 x 48: the left half still (a gradient), the right half
// panning; the top rows clipped in every frame (the HDR there wanders, by a
// little, then by a lot); a crushed corner; a band clipped in alternate frames
// only. The HDR is a stand-in composite whose exposure flickers, so the hold
// has something to hold.

#include <algorithm>
#include <cmath>
#include <vector>

#include "rudra/core/baseline.hpp"
#include "rudra/core/image.hpp"
#include "rudra/core/video_predict.hpp"

namespace rudra::hold_case {

inline constexpr int kWidth = 64, kHeight = 48, kFrames = 8;
inline constexpr double kRetention = 0.85;

struct Shot {
    std::vector<SdrImage> sdr;
    std::vector<PlanarBuffer> hdr;
};

inline Shot make_shot() {
    Shot s;
    for (int f = 0; f < kFrames; ++f) {
        PlanarBuffer codes(3, kHeight, kWidth), hdr(3, kHeight, kWidth);
        const double flicker = 1.0 + 0.06 * std::sin(1.7 * f);
        for (int y = 0; y < kHeight; ++y)
            for (int x = 0; x < kWidth; ++x) {
                double v = 0.15 + 0.6 * double(x) / kWidth + 0.1 * double(y) / kHeight;   // still
                if (x >= kWidth / 2) v = 0.5 + 0.35 * std::sin(0.4 * (x + 3 * f) + 0.2 * y);   // panning
                const bool clipped = y < 6;
                const bool flashing = y >= 40 && y < 44 && x >= 8 && x < 24 && f % 2 == 1;
                const bool crushed = y >= 44 && x < 8;
                if (clipped || flashing) v = 1.0;
                if (crushed) v = 0.01;
                const float c[3] = {float(std::clamp(v * 1.02, 0.0, 1.0)), float(v), float(std::clamp(v * 0.95, 0.0, 1.0))};
                for (int k = 0; k < 3; ++k) {
                    codes.at(k, y, x) = c[k];
                    // The stand-in composite: the SDR's light, lifted, times the flicker;
                    // in the clipped rows a highlight whose size wanders (little before
                    // frame 4, a lot after).
                    double h = double(srgb_to_linear(c[k])) * 0.0203 * flicker;
                    if (clipped) h = 0.05 * (1.0 + (f < 4 ? 0.01 : 0.6) * std::sin(0.9 * f + 0.1 * x));
                    hdr.at(k, y, x) = float(h);
                }
            }
        s.sdr.emplace_back(std::move(codes));
        s.hdr.push_back(std::move(hdr));
    }
    return s;
}

// What the viewer's hold computes, on the CPU: the SDR's Rec.709 luma and
// largest code, its linear light by the sRGB curve, then hold_pixels.
inline std::vector<PlanarBuffer> reference(const Shot& s, double retention = kRetention) {
    ShadowSmoother sm(retention, 0.15, true);
    std::vector<PlanarBuffer> out;
    for (std::size_t f = 0; f < s.sdr.size(); ++f) {
        const PlanarBuffer& c = s.sdr[f].buffer();
        auto [luma, peak] = source_code_luma(c, "rec709");
        std::vector<double> linear(c.plane_size());
        for (std::size_t i = 0; i < linear.size(); ++i)
            linear[i] = 0.2126 * double(srgb_to_linear(c.plane(0)[i])) + 0.7152 * double(srgb_to_linear(c.plane(1)[i])) +
                        0.0722 * double(srgb_to_linear(c.plane(2)[i]));
        out.push_back(sm.hold_pixels(std::move(luma), std::move(peak), linear, s.hdr[f]));
    }
    return out;
}

}  // namespace rudra::hold_case
