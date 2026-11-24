#pragma once
// The source curve (roadmap 3.1): the tone curve the artist says made the SDR.
// The baseline becomes the analytic inverse of that curve instead of the ACES
// inverse every checkpoint assumed. Expressed as the CurveHead's own output, a
// per-code log2 correction over the ACES inverse (baseline.hpp
// curve_correction_log2), one knot per 8-bit code, so the composite, the
// shader, master, measure and the parity probes carry it without a new path.
// Unknown is the ACES inverse unchanged: bit-identical to every master so far.
//
// BT.1886 (13 Oct 2026) is not a render curve: it is how a display shows SDR
// video, code^2.4 with 100 % at diffuse white (ITU-R BT.1886 at zero black,
// placed per BT.2408), the code being the BT.709 video's: RUDRA reads video
// through the inverse BT.709 OETF, which this undoes. On real SDR (Meridian,
// HDRTV1K) it measured far above the ACES inverse (release row R3,
// docs/RESULTS.md), and the app's sessions start on it. The app holds a shot's
// own codes (a still; a movie's frame through PNG): source_curve_params' own_codes
// reads them as video codes, code^2.4 (24 Nov 2026; before, they came out 0.36
// stop dark at code 128 and 2.7 at code 26).
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

// BT.1886 shadows (14 Oct 2026, opt-in) is BT.1886 from SDR code 40 up and,
// below it, the median HDR the real Netflix Open Content trims hold per code
// (docs/ai/SHADOW_CORRECTION_RESULT.md): real masters keep blacks well above
// code^2.4. Appended last so every stored id and index keeps its meaning.
enum class SourceCurve : std::uint8_t { Unknown, Aces, Hable, Agx, CameraLog, Clip, Bt1886, Bt1886Shadows };

inline constexpr int kSourceCurveKnots = 256;   // one knot per 8-bit code

// The id in params(), the sidecar and the CLI: unknown, aces, hable, agx,
// camera_log, clip, bt1886, bt1886_shadows. Round-trips through parse_source_curve.
std::string_view source_curve_id(SourceCurve c) noexcept;
std::optional<SourceCurve> parse_source_curve(std::string_view id) noexcept;
// Every curve, in the order the picker shows them.
std::span<const SourceCurve> all_source_curves() noexcept;
// The picker's label and sub-label.
std::string_view source_curve_label(SourceCurve c) noexcept;
std::string_view source_curve_detail(SourceCurve c) noexcept;

// pipeline/sdr_render.py curve_<name>: scene-linear (diffuse white 1.0) ->
// display-linear [0, 1]. Unknown is ACES. BT.1886 is the display's decode
// run backwards: display-linear = inverse BT.709 OETF of scene^(1/2.4), so
// that its inverse of a video's code is code^2.4.
float source_curve_forward(SourceCurve c, float scene_linear) noexcept;

// Its inverse: display-linear (clamped to 0.995, as inverse_aces_approx) ->
// scene-linear. Unknown and Aces are inverse_aces_approx exactly; BT.1886 is
// bt709_oetf(display)^2.4 exactly (no table). BT.1886 shadows is BT.1886 at
// and above code 40 and its measured table below (log2-linear between codes).
float source_curve_inverse(SourceCurve c, float display_linear) noexcept;

// The correction over the ACES inverse: [exposure 0, then kSourceCurveKnots
// values of log2(inverse_c / inverse_aces) at codes 0..255]. Empty for
// Unknown and Aces. Built once, then a view into a static table.
//
// 24 Nov 2026: `own_codes` is true when the SDR holds the source's own code values: a still (PNG, TIFF, JPEG,
// DPX...), and a movie as the app reads it (a frame through ffmpeg to PNG, then decoded as a still). Only the
// command-line video path re-encodes a movie's BT.709 light as sRGB codes (canonicalize_sdr "rec709"). BT.1886 and
// BT.1886 shadows say the SDR is video on a BT.1886 display, so own codes are video codes: with `own_codes` they
// read each code as the BT.709 code it is, and BT.1886 gives code^2.4 as on the video path. Before, own codes went
// through the sRGB decode and then the BT.709 OETF: 0.36 stop dark at code 128, 1.3 at 51, 2.8 at 26 (09V's frames:
// 1.95 stops at the median, 3.6 in the darkest 5 %). The other curves are unchanged by it.
std::span<const float> source_curve_params(SourceCurve c, bool own_codes = false);
// BT.1886 and BT.1886 shadows: the curves that read own codes as video codes.
bool reads_video_codes(SourceCurve c) noexcept;

// What the composite feeds corrected_baseline: the model's own curve params
// (FrameScalars.curve_params, the CurveHead if it has one) plus the source's,
// summed in log2 after resampling the model's knots to the source's. Returns
// the model's params unchanged when the source adds nothing, so a frame with
// Unknown is the same bytes as before 3.1.
std::vector<float> effective_curve_params(std::span<const float> model_params, SourceCurve source, bool own_codes = false);

}  // namespace rudra
