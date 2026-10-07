#pragma once
// Three-click calibration (roadmap 3.2): the artist clicks a black, an 18%
// grey and a known highlight on the frame and says what each should be in
// nits. The fit is a correction in log2 over the source curve's inverse
// (source_curve.hpp), piecewise-linear in SDR code between the anchors and
// flat outside them: one anchor shifts the exposure, two or three bend the
// curve through them. Exact at the anchors by construction. Expressed as the
// same 1 + 256 knot vector as the source curve, so the composite, the shader,
// master and QC carry it with no new path.

#include <span>
#include <string>
#include <vector>

#include "rudra/core/source_curve.hpp"

namespace rudra {

struct CalibrationPoint {
    int code = 0;        // the SDR code under the click, 0..255
    double nits = 0.0;   // what it should be, > 0
    bool operator==(const CalibrationPoint&) const = default;
};

inline constexpr int kMaxCalibrationPoints = 3;

// nits of an SDR code under the source's inverse at the model's exposure
// (the picker's readouts and the anchors' deltas use it).
double source_code_nits(SourceCurve source, int code, float corpus_ev) noexcept;

// The correction: empty when no usable point (nits <= 0 or code out of
// range are skipped). Two points on one code keep the last.
std::vector<float> calibration_params(std::span<const CalibrationPoint> points, SourceCurve source, float corpus_ev);

// False when the calibrated curve (source inverse times the correction)
// would go down anywhere over the codes; the app refuses such a set and
// says so, the composite never sees it.
bool calibration_is_monotone(std::span<const CalibrationPoint> points, SourceCurve source, float corpus_ev);

// What the fit does, for the panel: "" for no points.
std::string calibration_summary(std::span<const CalibrationPoint> points, SourceCurve source, float corpus_ev);

}  // namespace rudra
