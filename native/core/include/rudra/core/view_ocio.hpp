#pragma once
// The colour-managed viewer (17 Nov 2026, docs/view.spec.md revision 7): the picture
// through an OpenColorIO display and view, as Nuke's viewer shows it, in a working
// space of ACEScg or ACES2065-1, with the original SDR brought in through an input
// transform, and the range checks (out of gamut, invalid values, clipped by the view).
//
// OpenColorIO-free on purpose: infer/ocio_viewer.cpp makes these with OpenColorIO,
// render/ runs each stage's GLSL in the display pass, and core/view.cpp runs the same
// stage on the CPU (render_view), so every layer below infer stays as it was.

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "rudra/core/color.hpp"
#include "rudra/core/image.hpp"

namespace rudra {

// The scene-linear space the viewer hands OpenColorIO, and the space the probe and
// the invalid-value check read: what a render to that space would hold. In both,
// 1.0 is the diffuse white, 203 nits (core/view.hpp kSdrWhiteNits), as the ACES
// masters write it (app/aces_master.cpp aces_view_image).
enum class WorkingSpace : std::uint8_t { AcesCg = 0, Aces2065 = 1 };

inline Primaries working_primaries(WorkingSpace w) noexcept {
    return w == WorkingSpace::Aces2065 ? Primaries::Ap0 : Primaries::Ap1;
}
inline const char* working_name(WorkingSpace w) noexcept {
    return w == WorkingSpace::Aces2065 ? "ACES2065-1" : "ACEScg";
}

// One texture a stage samples, as OpenColorIO hands it over.
struct OcioTexture {
    std::string sampler;          // the sampler's name in the stage's GLSL
    int binding = 0;              // its binding (set 0)
    int width = 0, height = 1, depth = 1;   // depth > 1: a 3D texture
    int channels = 3;             // 1 or 3 values per texel in `values`
    bool linear = true;           // linear filtering, else nearest
    std::vector<float> values;    // width x height x depth x channels, x fastest
};

// One transform as the GPU and the CPU run it. `glsl` is Vulkan-flavoured GLSL
// (set 0, explicit bindings) that defines `vec4 <function>(vec4)` and declares the
// samplers in `textures`; `cpu` is the same transform on packed RGB, in place.
struct OcioStage {
    std::string function;
    std::string glsl;
    std::vector<OcioTexture> textures;
    std::function<void(float* rgb, std::size_t pixels)> cpu;
};

// A display and view of a config, from the working space. `forward` gives the
// display's code values (what Nuke writes to an SDR framebuffer); `decode` gives
// those codes back as light, CIE XYZ (D65), so an HDR swapchain shows them at their
// real level: hdr-video displays at 100 nits per Y = 1 (PQ 1000 lands at 1000 nits),
// SDR ones with their peak at the SDR white (203 nits, BT.2408).
struct OcioViewTransform {
    std::string id;               // config | display | view | working: what caches key on
    std::string config, display, view;
    WorkingSpace working = WorkingSpace::AcesCg;
    OcioStage forward;
    bool decodes = false;         // decode is set; else HDR swapchains show the codes as SDR graphics
    OcioStage decode;
    double nits_per_y = 203.0;
    bool hdr = false;             // the display is an HDR one (hdr-video, or edr-video: Apple's extended range)
    // A code at or above this is clipped by the view (the clip zebra). EDR displays have
    // codes above 1 and no clip: huge.
    double clip_code = 1.0 - 0.5 / 1023.0;
};

// How the original SDR enters the working space (the Sdr source of the viewer): its
// codes through an OpenColorIO transform, e.g. the inverse of an SDR view (the
// original then looks as it was graded through that view) or a texture space.
struct OcioInputTransform {
    std::string id, name;
    WorkingSpace working = WorkingSpace::AcesCg;
    std::function<void(float* rgb, std::size_t pixels)> cpu;   // SDR codes -> working-space light
};

// ---- the range checks (Image layer overlays) ----
// Out of gamut: a channel in the checked primaries below -(kCheckFloor + kCheckRelative
// x the largest), so rounding never flags a colour that is inside.
inline constexpr float kCheckFloor = 1e-7f;      // network units (0.001 nits)
inline constexpr float kCheckRelative = 1e-3f;
// Invalid: not a number or infinite, not a colour at all (outside AP0, which holds
// every colour the eye sees), or above what a half-float EXR holds in the working space.
inline constexpr float kHalfMax = 65504.0f;
// Clipped by the view: a code value at the top 10-bit code or above.
inline constexpr float kViewClipCode = 1.0f - 0.5f / 1023.0f;
inline constexpr std::array<float, 3> kGamutColour{0.90f, 0.25f, 0.95f};
inline constexpr std::array<float, 3> kInvalidColour{0.15f, 1.00f, 0.35f};

// Rows of the viewer's matrices as the display pass reads them (fp32, row major),
// made once so the GPU and core/view.cpp multiply by the same numbers.
struct ViewMatrices {
    std::array<float, 9> work{};   // source -> working space, x 10000 / 203 (viewer units to 1.0 = diffuse white)
    std::array<float, 9> dec{};    // CIE XYZ (D65) -> linear Rec.709, x nits per Y
    std::array<float, 9> chk{};    // source -> the primaries the gamut check reads
    std::array<float, 9> ap0{};    // source -> AP0, for the invalid check
};
ViewMatrices view_matrices(Primaries source, WorkingSpace working, Primaries gamut, double nits_per_y);

// The checks of one pixel (viewer units, source primaries), as the overlays decide them.
bool out_of_gamut(const float h[3], const std::array<float, 9>& chk) noexcept;
bool invalid_value(const float h[3], const std::array<float, 9>& work, const std::array<float, 9>& ap0) noexcept;

// One pixel for the probe: its working-space values, the view's code values (empty
// without a view), and the checks with a word for each.
struct PixelReading {
    std::array<double, 3> working{};
    std::array<float, 3> code{};
    bool has_code = false;
    bool out_of_gamut = false;
    bool invalid = false;
    std::string why;   // "", "NaN", "infinite", "outside AP0", "above half float"
};
PixelReading read_pixel(const float rgb[3], Primaries source, WorkingSpace working, Primaries gamut,
                        const OcioViewTransform* view = nullptr);

// The original's light through an input transform, laid out as sdr_light_rgba (rgb in
// viewer units in `source` primaries, the clamp flags in a): the codes through `t`
// to the working space, then to the source primaries at 203 nits per 1.0.
std::vector<float> sdr_light_through(const SdrImage& sdr, const OcioInputTransform& t, Primaries source);

// "OCIO · sRGB - Display · ACES 2.0 - SDR 100 nits (Rec.709) · ACEScg": the pipe bar's words.
std::string ocio_view_text(const OcioViewTransform& v);

}  // namespace rudra
