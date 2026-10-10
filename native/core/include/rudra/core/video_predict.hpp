#pragma once
// The pure parts of the video predictor (Phase 4, step 3): rudra/video.py
// Predictor.predict before and after the network, and ShadowSmoother. All in
// float32 as the Python's tensors are; the smoother's weight is a Python float
// (double). The network itself runs through infer (engine/video_predictor).

#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "rudra/core/color.hpp"
#include "rudra/core/image.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

// rudra.sdr2hdr.canonicalize_sdr: code values in the clip's transfer and range
// to the model's canonical sRGB code ("srgb", "rec709", "gamma22", "gamma24";
// "full" or "limited").
Result<PlanarBuffer> canonicalize_sdr(const PlanarBuffer& rgb, std::string_view transfer, std::string_view range);

// linear_to_srgb(M @ srgb_to_linear(x)) with M = rgb_to_rgb_matrix(src,
// Rec.2020) in float32, as the predictor does when the primaries are not 2020.
PlanarBuffer srgb_code_to_rec2020(const PlanarBuffer& code, Primaries src);

// F.interpolate(x, size, mode="area"): adaptive average pooling.
PlanarBuffer area_resize(const PlanarBuffer& x, int height, int width);

// torch.linspace(start, end, steps) in float32, as the CPU kernel computes it.
std::vector<float> torch_linspace(float start, float end, int steps);

// Scalar smoothing within a shot that resets at hard cuts; never blends
// pixels. update() decides whether the frame starts a shot and smooths the
// shadow weight; hold() smooths any other per-frame value (the residual
// scale, the curve, the anchor's hold) with the same retention, reset at the
// same cuts (rudra/video.py ShadowSmoother.hold, 10 Oct 2026). update() first.
class ShadowSmoother {
public:
    explicit ShadowSmoother(double retention = 0.0, double cut_threshold = 0.15)
        : retention_(retention), cut_threshold_(cut_threshold) {}
    // (the weight to use, whether this frame starts a new shot)
    std::pair<double, bool> update(const PlanarBuffer& thumbnail, double weight);
    std::vector<double> hold(const std::string& key, const std::vector<double>& value);
    double hold(const std::string& key, double value) { return hold(key, std::vector<double>{value})[0]; }
    double retention() const noexcept { return retention_; }
    double cut_threshold() const noexcept { return cut_threshold_; }

private:
    double retention_, cut_threshold_;
    PlanarBuffer previous_;
    bool has_weight_ = false;
    double weight_ = 0.0;
    std::map<std::string, std::vector<double>> held_;
};

// exp(-1/(fps*seconds)), the retention of a hold with that time constant; 0
// for 0 seconds. `fps` as ffprobe gives it ("24", "30000/1001").
double retention_for(std::string_view fps, double seconds);

// The flicker measure (rudra/video.py, 10 Oct 2026): the mean log2 of a 36x64
// area thumbnail of a luminance plane (1.0 = diffuse white, floored at 2^-10).
// A frame's flicker is |dEV(HDR) - dEV(source)| from the frame before, not
// scored at a cut.
inline constexpr int kExposureThumbHeight = 36, kExposureThumbWidth = 64;
double exposure_ev(std::span<const double> luminance, int height, int width);
// The decoded frame (full-range code in the clip's transfer, "srgb", "rec709",
// "gamma22", "gamma24"; primaries "rec709" or "rec2020") as linear luminance.
std::vector<double> source_luminance(const PlanarBuffer& rgb, std::string_view transfer, std::string_view primaries);
// An HDR frame in network units (1.0 = 10 000 nits, Rec.2020) as luminance
// relative to diffuse white.
std::vector<double> hdr_luminance(const PlanarBuffer& hdr);

// The size of the thumbnail the cut detector compares.
inline constexpr int kCutThumbHeight = 36, kCutThumbWidth = 64;

// The video path's tile origins along one axis: every size - overlap, the last
// moved back to end at the edge.
std::vector<int> video_tile_starts(int length, int size, int overlap);

// The feather for a tile at (y, x) of (th, tw) in a frame of (h, w): th x tw,
// row-major, (wy[:, None] * wx[None, :]) with linear ramps from 0.001 to 1 on
// the sides that have a neighbour.
std::vector<float> video_tile_blend(int y, int x, int th, int tw, int h, int w, int overlap);

}  // namespace rudra
