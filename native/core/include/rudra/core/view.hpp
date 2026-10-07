#pragma once
// The viewer's display pass on the CPU (docs/view.spec.md section 2): the
// reference the display shader is held to, and the port of the DISPLAY shader
// in ui/compositor.js. It reads the two composite targets and never changes
// them; switching view or moving the wipe is a new call on the same inputs.

#include <array>
#include <memory>
#include <cstdint>
#include <vector>

#include "rudra/core/color.hpp"
#include "rudra/core/image.hpp"
#include "rudra/core/masks.hpp"

namespace rudra {

// Invented: the invented-pixel map (core/compare.hpp, roadmap 3.5), a graphic
// over a dimmed picture: magenta where the network made up values the SDR has
// no information for, cyan where it reinterpreted SDR detail.
enum class ViewMode : std::uint8_t { Image = 0, FalseColour = 1, Difference = 2, Invented = 3 };

// Which path the picture takes to the glass. The pipe bar shows it verbatim.
enum class OutputPath {
    SdrPqSimulation,   // SDR swapchain: exposure to the view peak, then clip (the browser Studio's view)
    ScRgb,             // FP16 linear Rec.709, 1.0 = 80 nits (Windows D3D12, Linux Vulkan)
    Hdr10,             // PQ Rec.2020 10-bit (Windows option)
    Edr,               // linear, 1.0 = the SDR white (macOS Metal; P3 or Rec.709 primaries)
};

// The swapchain the display pass writes for (docs/view.spec.md section 9).
struct DisplayTarget {
    OutputPath path = OutputPath::SdrPqSimulation;
    Primaries primaries = Primaries::Rec709;   // of the swapchain
    double peak_nits = 203.0;                  // what the display can show right now (EDR headroom is live)
    double unit_nits = 203.0;                  // linear paths: nits written as 1.0

    static DisplayTarget sdr() { return {}; }
    static DisplayTarget scrgb(double peak) { return {OutputPath::ScRgb, Primaries::Rec709, peak, 80.0}; }
    static DisplayTarget hdr10(double peak) { return {OutputPath::Hdr10, Primaries::Rec2020, peak, 10000.0}; }
    // EDR: 1.0 is the SDR white, which RUDRA holds at diffuse white (203 nits).
    static DisplayTarget edr(double peak, Primaries p = Primaries::P3D65) { return {OutputPath::Edr, p, peak, 203.0}; }
};
enum class ViewSource : std::uint8_t { Model, Baseline };

struct ViewParams {
    ViewMode mode = ViewMode::Image;
    double display_nits = 203.0;     // the view peak: exposure to it, then clip
    ViewSource show = ViewSource::Model;
    double wipe = -1.0;              // < 0 off; else [0, 1] across the frame, baseline on the left
    double wipe_half_width = 0.0012;
    double diff_gain = 2000.0;       // nits at which the difference ramp saturates
    bool show_changes = false;       // Image: tint what the model changed against the baseline
    DisplayTarget target;            // SDR unless the swapchain is HDR
    Primaries source = Primaries::Rec709;   // of the composite: the network keeps the input's primaries
    // Roadmap 3.3: the master's anchor stage (core/master.hpp anchor_to_sdr),
    // live on the model's picture so what is seen is what the master does.
    // Per pixel the gain that puts unclipped picture back on the SDR's level;
    // above the knee (the SDR's max code) it blends to anchor_hold, the
    // frame's median gain in the knee band, which the measure computes
    // (anchor_hold()) and the app feeds back. Off: the composite as it is.
    // Needs the frame's SDR; the baseline side of a wipe is never anchored.
    bool anchor = false;
    double anchor_knee = 0.9;
    double anchor_softness = 0.04;
    double anchor_hold = 1.0;
    // Roadmap 3.3: the painted mask being edited, tinted over the picture
    // (Image mode): the band's channel of `masks`, in `paint_tint` (an sRGB
    // colour) mixed by kMaskTintMix times the mask. -1 for none.
    int paint_band = -1;
    std::array<float, 3> paint_tint{1.0f, 0.7f, 0.25f};
    std::shared_ptr<const MaskSet> masks;
};
inline constexpr float kMaskTintMix = 0.35f;

// A view with the SDR target, for tools and tests: the positional form of
// ViewParams without its target and source.
inline ViewParams view_params(ViewMode mode, double display_nits, ViewSource show = ViewSource::Model,
                              double wipe = -1.0, double wipe_half_width = 0.0012, double diff_gain = 2000.0) {
    ViewParams v;
    v.mode = mode;
    v.display_nits = display_nits;
    v.show = show;
    v.wipe = wipe;
    v.wipe_half_width = wipe_half_width;
    v.diff_gain = diff_gain;
    return v;
}

// The tint of show_changes: amber where the model is brighter than the
// baseline, blue where it is darker, mixed in by kChangeTintMix times
// change_weight() (core/compare.hpp). SDR codes; the HDR paths take them as
// graphics at the SDR white.
inline constexpr std::array<float, 3> kChangeUp{0.95f, 0.62f, 0.28f};
inline constexpr std::array<float, 3> kChangeDown{0.32f, 0.56f, 0.95f};
inline constexpr float kChangeTintMix = 0.55f;

// The Invented layer: the picture's luma at the view exposure times kMapGrey,
// with kInventedColour or kReinterpretedColour mixed in by kMapMix times
// |change_weight()|. SDR codes, graphics on the HDR paths.
inline constexpr float kMapGrey = 0.6f;
inline constexpr float kMapMix = 0.85f;
inline constexpr std::array<float, 3> kInventedColour{0.92f, 0.30f, 0.86f};
inline constexpr std::array<float, 3> kReinterpretedColour{0.25f, 0.78f, 0.86f};

// The anchor's hold gain for a composite (network units) against its SDR:
// the median of the per-pixel gains in the knee band, or of all gains when
// the band holds fewer than 64 pixels. What anchor_to_sdr computes inside
// itself; the live view needs it as a number (ViewParams::anchor_hold).
double anchor_hold(const NetworkLinearImage& model, const SdrImage& sdr, double knee = 0.9, double softness = 0.04);

// The anchor gain of one pixel, fp32 as the display shader computes it:
// target the SDR's Rec.2020 luma of its linearised codes (diffuse white 1.0),
// actual the picture's in network units, max_code the SDR's max channel.
float anchor_gain_f(float target_luma, float actual_r, float actual_g, float actual_b, float max_code, float knee,
                    float softness, float hold) noexcept;

// 8-bit RGB, interleaved, image order (row 0 the top).
struct Rgb8Image {
    int width = 0, height = 0;
    std::vector<std::uint8_t> rgb;
};

inline constexpr std::array<float, 3> kRec2020Luma{0.2627f, 0.6780f, 0.0593f};

// The false-colour zone of a Rec.2020 luminance in nits, 0..9, and its colour.
int false_colour_zone(float nits) noexcept;
std::array<float, 3> false_colour(int zone) noexcept;

// sRGB OETF on [0, 1], fp32 as the shader computes it.
float linear_to_srgb(float x) noexcept;

// The display pass as the values the swapchain is written with (3 x H x W,
// image order): SDR codes in [0, 1] before the 8-bit conversion; scRGB and EDR
// linear in the target's unit; HDR10 PQ codes. `model` and `baseline` are
// network units.
// `sdr` (the frame's SDR codes) is read by the Invented layer only; without it
// every changed pixel reads as reinterpreted.
PlanarBuffer render_view(const NetworkLinearImage& model, const NetworkLinearImage& baseline,
                         const ViewParams& params, const SdrImage* sdr = nullptr);

// The exact reductions over m = max(R, G, B) (docs/view.spec.md section 4),
// in network units, evaluated as the GPU ladder evaluates them: 2x2 boxes in
// fp32, j then i within a box, rows and columns paired in image order, so the
// fp32 sum is the same number the shader produces, not just a close one.
struct Reductions {
    float peak = 0.0f;   // max(m)
    float sum = 0.0f;    // the ladder's fp32 sum of m
};
Reductions reduce_ladder(const PlanarBuffer& rgb);

// The probe (section 3): one pixel of each composite, round half up, clamped.
struct ProbeSample {
    int x = 0, y = 0;
    double rgb_nits[3] = {0, 0, 0};   // texel * 10000 in double, as the page
    double nits = 0.0;   // Rec.2020 luminance, in double as the page computes it
};
struct Probe {
    ProbeSample model, baseline;
};
Probe probe_pixel(const NetworkLinearImage& model, const NetworkLinearImage& baseline, double x, double y);

// The SDR picture quantised as the 8-bit framebuffer does: round(255 c).
Rgb8Image render_view_rgb8(const NetworkLinearImage& model, const NetworkLinearImage& baseline,
                           const ViewParams& params, const SdrImage* sdr = nullptr);

// The SDR's max(R, G, B) code per pixel, what the display pass reads for the
// Invented layer (the GPU gets it as the baseline texture's alpha); 0.5, a
// code with detail, where there is no SDR.
std::vector<float> sdr_max_codes(const SdrImage* sdr, int width, int height);

}  // namespace rudra
