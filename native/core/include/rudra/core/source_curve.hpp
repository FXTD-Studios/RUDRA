#pragma once
// The source curve (roadmap 3.1): the tone curve the artist says made the SDR.
// The baseline becomes the analytic inverse of that curve instead of the ACES
// inverse every checkpoint assumed. Expressed as the CurveHead's own output, a
// per-code log2 correction over the ACES inverse (baseline.hpp
// curve_correction_log2), one knot per 8-bit code, so the composite, the
// shader, master, measure and the parity probes carry it without a new path.
// Unknown is the ACES inverse unchanged: bit-identical to every master so far.
//
// The forward curves are pipeline/sdr_render.py's, the ones the corpus was
// rendered with (tests/test_core.cpp pins the port); the inverses are numeric
// on a dense table in log2 exposure, built once per process.

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace rudra {

enum class SourceCurve : std::uint8_t { Unknown, Aces, Hable, Agx, CameraLog, Clip };

inline constexpr int kSourceCurveKnots = 256;   // one knot per 8-bit code

// The id in params(), the sidecar and the CLI: unknown, aces, hable, agx,
// camera_log, clip. Round-trips through parse_source_curve.
std::string_view source_curve_id(SourceCurve c) noexcept;
std::optional<SourceCurve> parse_source_curve(std::string_view id) noexcept;
// Every curve, in the order the picker shows them.
std::span<const SourceCurve> all_source_curves() noexcept;
// The picker's label and sub-label.
std::string_view source_curve_label(SourceCurve c) noexcept;
std::string_view source_curve_detail(SourceCurve c) noexcept;

// pipeline/sdr_render.py curve_<name>: scene-linear (diffuse white 1.0) ->
// display-linear [0, 1]. Unknown is ACES.
float source_curve_forward(SourceCurve c, float scene_linear) noexcept;

// Its inverse: display-linear (clamped to 0.995, as inverse_aces_approx) ->
// scene-linear. Unknown and Aces are inverse_aces_approx exactly.
float source_curve_inverse(SourceCurve c, float display_linear) noexcept;

// The correction over the ACES inverse: [exposure 0, then kSourceCurveKnots
// values of log2(inverse_c / inverse_aces) at codes 0..255]. Empty for
// Unknown and Aces. Built once, then a view into a static table.
std::span<const float> source_curve_params(SourceCurve c);

// What the composite feeds corrected_baseline: the model's own curve params
// (FrameScalars.curve_params, the CurveHead if it has one) plus the source's,
// summed in log2 after resampling the model's knots to the source's. Returns
// the model's params unchanged when the source adds nothing, so a frame with
// Unknown is the same bytes as before 3.1.
std::vector<float> effective_curve_params(std::span<const float> model_params, SourceCurve source);

}  // namespace rudra
