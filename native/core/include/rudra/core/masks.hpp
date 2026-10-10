#pragma once
// Painted masks (roadmap 3.3): one optional 8-bit mask per Region EV band,
// painted on the viewer at the preview frame's size. A band's mask gates its
// qualifier: gain = 2^(sum ev_i * q_i(Y) * m_i(x, y)), m_i = 1 everywhere when
// the band has none, so a frame without masks is bit-identical to one before
// 3.3. Masks belong to the first kMaxMaskBands bands; later bands take none.
//
// A set samples bilinear at any frame size (a master is full size, its mask
// is the preview's). The brush lives here too (paint_stroke), so the app and
// the tests paint with the same stamps. Masks travel as one RGBA8 PNG beside
// the master (a band per channel), never inside the params JSON.

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "rudra/platform/result.hpp"

namespace rudra {

inline constexpr int kMaxMaskBands = 4;

using MaskPlane = std::vector<std::uint8_t>;   // height x width, 0 = off, 255 = on

struct MaskSet {
    int width = 0, height = 0;
    // A band's plane, or null for none. Shared so a stroke copies one plane
    // and the session's undo keeps a snapshot per stroke at the plane's cost.
    std::array<std::shared_ptr<const MaskPlane>, kMaxMaskBands> planes;

    bool empty() const noexcept;
    bool has(int band) const noexcept { return band >= 0 && band < kMaxMaskBands && planes[std::size_t(band)] != nullptr; }
    // Bands with a plane, in order.
    std::vector<int> bands() const;
    // Mean of the plane over the frame, 0..1; 0 for none.
    double coverage(int band) const noexcept;
    // The band's weight at frame pixel (x, y) of a frame_w x frame_h frame:
    // bilinear on the plane's pixel centres, 1 for none.
    float weight(int band, int x, int y, int frame_w, int frame_h) const noexcept;
    // All bands' weights at once (what the composite calls per pixel).
    void weights(int x, int y, int frame_w, int frame_h, float out[kMaxMaskBands]) const noexcept;
    // Interleaved RGBA8 at the set's size, a band per channel, 255 for a
    // band without a plane: the GPU texture and the PNG.
    std::vector<std::uint8_t> rgba8() const;
    // Content equality (planes compared by value).
    bool operator==(const MaskSet& o) const;
};

// The brush: a soft round stamp, `softness` the share of the radius that
// ramps (0 hard, 1 all ramp), `flow` the stamp's opacity. Erase takes the
// mask down the same way Add takes it up.
struct Brush {
    double size_px = 80.0;   // diameter in frame pixels
    double softness = 0.5;
    double flow = 0.6;
    bool erase = false;
};

// A new plane for a set of this size, all off.
std::shared_ptr<MaskPlane> empty_plane(int width, int height);
// Stamps along the segment (x0, y0) to (x1, y1), in frame pixels, a stamp
// every 15 % of the brush's diameter, into `plane` (w x h).
void paint_stroke(MaskPlane& plane, int width, int height, double x0, double y0, double x1, double y1,
                  const Brush& brush);
void invert_plane(MaskPlane& plane);

// The PNG beside the master: RGBA8, non-interlaced, a band per channel; a
// band without a plane is written as 255 and read back as none when the
// whole channel is 255. The reader takes 8-bit grey, grey+alpha, RGB and RGBA
// too (a mask edited elsewhere), grey into band 0.
Result<void> write_mask_set(const std::filesystem::path& path, const MaskSet& set);
Result<MaskSet> read_mask_set(const std::filesystem::path& path);

}  // namespace rudra
