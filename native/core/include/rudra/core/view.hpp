#pragma once
// The viewer's display pass on the CPU (docs/view.spec.md section 2): the
// reference the display shader is held to, and the port of the DISPLAY shader
// in ui/compositor.js. It reads the two composite targets and never changes
// them; switching view or moving the wipe is a new call on the same inputs.

#include <array>
#include <memory>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "rudra/core/color.hpp"
#include "rudra/core/composite.hpp"
#include "rudra/core/image.hpp"
#include "rudra/core/masks.hpp"

namespace rudra {

// Invented: the invented-pixel map (core/compare.hpp, roadmap 3.5), a graphic
// over a dimmed picture: magenta where the network made up values the SDR has
// no information for, cyan where it reinterpreted SDR detail.
// Range (revision 6, 1 Nov 2026): stops of max(R, G, B) over the SDR white
// (203 nits) on a fixed legend, with the pixels the SOURCE clamped hatched;
// independent of the view exposure, like false colour.
enum class ViewMode : std::uint8_t { Image = 0, FalseColour = 1, Difference = 2, Invented = 3, Range = 4 };

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
// Sdr (revision 6): the source plate as light, per channel, the reading the
// composite uses (sdr_light_rgba), so a clamped code stays flat at its white.
enum class ViewSource : std::uint8_t { Model = 0, Baseline = 1, Sdr = 2 };

// What the region overlay's strength shows (RUDRA 1.1, 7 Nov 2026). Off: the region's own
// weight, exact for the learned ACES route and the AI fill. analytic: the analytic composite's gate
// (composite.hpp reconstruction_gate) with this mode, strength and frame's shadow weight.
struct RegionOverlayGate {
    bool analytic = false;
    RecoveryMode mode = RecoveryMode::All;
    float strength = 1.0f;
    float shadow_weight = 1.0f;
    bool operator==(const RegionOverlayGate&) const = default;
};

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
    // The composite's Region EV gain per pixel (composite()'s grade_gain), or
    // null for an ungraded composite. The master grades AFTER the anchor
    // (since 9 Oct 2026); the anchor's target is scaled by this gain so the
    // view keeps the grade below the knee instead of dividing it back out.
    // The composite.frag twin carries it in the target (alpha) channel.
    const std::vector<float>* grade_gain = nullptr;
    // The anchor's target over the sRGB reading (composite.hpp
    // anchor_target_knots, 13 Oct 2026); empty for the sRGB reading. The
    // composite.frag twin carries it in the target channel too.
    std::vector<float> anchor_knots;
    // Roadmap 3.3: the painted mask being edited, tinted over the picture
    // (Image mode): the band's channel of `masks`, in `paint_tint` (an sRGB
    // colour) mixed by kMaskTintMix times the mask. -1 for none.
    int paint_band = -1;
    std::array<float, 3> paint_tint{1.0f, 0.7f, 0.25f};
    std::shared_ptr<const MaskSet> masks;
    // ---- revision 6 (1 Nov 2026): the viewer's own controls, as in Nuke's ----
    // Viewer exposure in stops and viewer gamma, on the Image picture only, in
    // scene-linear after the anchor: h * 2^view_ev, then x^(1/view_gamma) with
    // the view peak as 1.0, then the usual exposure to the peak and clip. Never
    // in a master, a render, a measurement, false colour, Difference, Invented
    // or Range. 0 and 1 reproduce revision 5 exactly.
    double view_ev = 0.0;
    double view_gamma = 1.0;
    // The wipe's sides (A left, B right). Revision 5 was Baseline | Model.
    ViewSource wipe_left = ViewSource::Baseline;
    ViewSource wipe_right = ViewSource::Model;
    // Image overlays. clamp_zebra: where the SOURCE is clamped (a property of
    // the plate, the same pixels on both sides, at any exposure): red stripes
    // where all three channels are at code 254 or above, thin orange ones where
    // only some are, blue where max(R, G, B) is at code 1 or below.
    // clip_zebra: where the picture clips ON SCREEN at this view exposure,
    // gamma and peak (yellow, the other diagonal).
    bool clamp_zebra = false;
    bool clip_zebra = false;
    // RUDRA 1.1 (5 Nov 2026): the reconstruction region (composite.hpp) tinted over the Image
    // picture, kRegionTintMix times its weight, as the composite gates by it. Off: no change.
    bool region_overlay = false;
    ReconstructionRegion overlay_region;
    RegionOverlayGate overlay_gate;   // what the tint's strength shows (below)
    std::array<float, 3> region_tint{0.20f, 0.85f, 0.95f};
};
inline constexpr float kRegionTintMix = 0.40f;

// The region overlay's weight per pixel as the viewer uploads it (RGBA8, the weight in R, image
// order), of the SDR's clamped Rec.709 luma, clipped to [0, 1], to 8 bits: with `gate` off (the
// default) max of region_weights(), which is what the learned ACES route and the AI fill apply;
// with `gate.analytic` the analytic composite's own reconstruction_gate() (the built-in priors
// times the region, the mode and the strength), so the tint is as strong as the reconstruction
// is (7 Nov 2026: a region from 0.5 tinted the mid-tones fully where the gate is under 5 %).
// All 0 when the region is off or there is no SDR of that size.
std::vector<std::uint8_t> region_overlay_rgba8(const SdrImage* sdr, const ReconstructionRegion& region, int width, int height,
                                               const RegionOverlayGate& gate = {});

// The Reconstruct card's histogram (RUDRA 1.1): the frame's SDR Rec.709 luma in `bins` bins,
// the region's weight at each bin's centre, and the share of pixels it reaches (weight > 0).
struct RegionHistogram {
    std::vector<double> counts;   // pixels per bin, of all pixels (0..1)
    std::vector<float> weight;    // the overlay's weight (region_overlay_rgba8, same gate) at the bin's centre
    double reached = 0.0;         // share of pixels the tint shows (weight of 1/255 or more)
    double strong = 0.0;          // share of pixels at half strength or more
};
RegionHistogram region_histogram(const SdrImage& sdr, const ReconstructionRegion& region, int bins = 64,
                                 const RegionOverlayGate& gate = {});

// Revision 6 overlays: stripe period and mix, colours as SDR codes (graphics
// at the SDR white on the HDR paths).
inline constexpr int kZebraPeriod = 8;
inline constexpr float kZebraMix = 0.6f;
inline constexpr std::array<float, 3> kClampAllColour{0.95f, 0.20f, 0.20f};
inline constexpr std::array<float, 3> kClampSomeColour{0.98f, 0.55f, 0.15f};
inline constexpr std::array<float, 3> kCrushColour{0.20f, 0.45f, 0.98f};
inline constexpr std::array<float, 3> kScreenClipColour{0.98f, 0.90f, 0.15f};
inline constexpr float kRangeHatchMix = 0.7f;

// Range legend: stops = log2(max(R, G, B) nits / 203). Below -6 one colour,
// -6..0 a grey ramp (the SDR's own range), then one colour per stop above
// the SDR white, the last for +4 and above.
inline constexpr float kSdrWhiteNits = 203.0f;
std::array<float, 3> range_colour(float stops) noexcept;

// The SDR flags packed into the light texture's alpha: bit 0..2 the R, G, B
// code at kSdrClipCode or above, bit 3 max(R, G, B) at kSdrCrushCode or below.
inline constexpr int kFlagClipR = 1, kFlagClipG = 2, kFlagClipB = 4, kFlagCrush = 8;

// The Sdr source and the clamp flags, interleaved RGBA, image order, what the
// display pass reads (binding 4): rgb the SDR's light per channel in network
// units, the anchor's reading (sRGB decode times the Source picker's knots,
// composite.hpp anchor_target_knots) with the code's white at 203 nits; a the
// flags above, as a small integer. Empty without an SDR.
std::vector<float> sdr_light_rgba(const SdrImage* sdr, std::span<const float> target_knots = {});

// Whether a view reads the SDR light (the Sdr source on any side, the clamp
// zebra, the Range layer): the CPU pass and the GPU's binding 4 both ask this.
inline bool view_needs_sdr_light(const ViewParams& p) noexcept {
    const bool wiping = p.wipe >= 0.0;
    return p.show == ViewSource::Sdr || (wiping && (p.wipe_left == ViewSource::Sdr || p.wipe_right == ViewSource::Sdr)) ||
           (p.mode == ViewMode::Image && p.clamp_zebra) || p.mode == ViewMode::Range;
}
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
// With `grade_gain` (composite()'s), the hold of the ungraded picture, as the
// master computes it before its grade.
double anchor_hold(const NetworkLinearImage& model, const SdrImage& sdr, double knee = 0.9, double softness = 0.04,
                   const std::vector<float>* grade_gain = nullptr, std::span<const float> target_knots = {});

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

// ---- RUDRA 1.1 priority 4 (5 Nov 2026): exposure brackets and the display signal ----

// The EV brackets RUDRA renders beside a master: -2, 0 and +2 stops.
inline constexpr std::array<double, 3> kBracketStops{-2.0, 0.0, 2.0};

// One bracket: `picture` (network units, primaries `source`) through the SDR view with the
// diffuse white (203 nits) at code 1.0, the viewer exposure at `ev` stops, clipped, sRGB,
// rounded to 8 bits: what the viewer shows on an SDR screen at view peak 203 and EV `ev`.
// -2 shows the light up to 812 nits, +2 the shadows.
Rgb8Image exposure_bracket(const NetworkLinearImage& picture, double ev, Primaries source = Primaries::Rec709);

// How far apart two 8-bit pictures of one size are: the largest code difference over every
// channel, and the share of pixels (0..1) with any difference. size mismatch: {256, 1}.
struct CodeDelta {
    int max_code_delta = 0;
    double pixels_differing = 0.0;
};
CodeDelta code_delta(const Rgb8Image& a, const Rgb8Image& b);

// The signal the viewer writes, in one line, e.g. "HDR10 · PQ (ST 2084) · Rec.2020 · reference
// white 203 nits · display peak 1,000 nits". The SDR path names the view peak it simulates.
std::string display_signal_text(const DisplayTarget& target, double view_peak_nits);

// The SDR's max(R, G, B) code per pixel, what the display pass reads for the
// Invented layer (the GPU gets it as the baseline texture's alpha); 0.5, a
// code with detail, where there is no SDR.
std::vector<float> sdr_max_codes(const SdrImage* sdr, int width, int height);

}  // namespace rudra
