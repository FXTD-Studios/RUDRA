#pragma once
// Reference match (roadmap 3.4): the artist loads the graded HDR of the frame
// on screen and the baseline is fitted to it. The fit is a calibration at
// every code: for each pixel and channel the SDR code and the reference's
// linear value give one sample of log2 nits; per code, the median of its
// samples is the target. The target is the curve the reference says an SDR
// code is worth, independent of the source picker and of the model, so it is
// stored as log2 nits per code and the correction over any source curve is
// target - log2(source nits), the same 1 + 256 knot vector as the picker and
// the anchors (calibration.hpp). The composite, the shader, master and QC
// carry it with no new path.
//
// Codes with too few samples are interpolated from their neighbours, flat
// beyond the outermost seen code; the target is smoothed over a few codes
// and made monotone by isotonic regression (pool adjacent violators,
// weighted by sample count), so it never refuses: the residual says what the
// curve could not carry. The residual is the per-sample error after the fit,
// mean and p95 in stops over codes 1..254; a grade with a window or a key
// shows there.

#include <array>
#include <span>
#include <string>
#include <vector>

#include "rudra/core/image.hpp"
#include "rudra/core/source_curve.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

inline constexpr int kReferenceMinSamples = 16;    // per code, below it the code is interpolated
inline constexpr int kReferenceSmoothRadius = 3;   // codes, triangular

struct ReferenceFit {
    std::string file;                        // the reference's file name, for the panel and the sidecar
    std::vector<float> target_log2_nits;     // 256 values, one per code; empty for no reference
    std::vector<int> samples_per_code;       // 256 counts, what the fit saw (interpolated codes hold 0)
    double residual_mean = 0.0, residual_p95 = 0.0;   // stops, after the fit, codes 1..254
    long long samples = 0;                   // the samples the residual is over
    int codes_seen = 0;                      // codes with at least kReferenceMinSamples
    bool empty() const noexcept { return target_log2_nits.empty(); }
    bool operator==(const ReferenceFit&) const = default;
};

// The fit. `reference` is scene-linear Rec.709 with 1.0 = 203 nits (what
// read_reference gives), 3 planes, the sdr's size; another size is refused.
// Values <= 0 and non-finite are skipped. Refused when fewer than
// kReferenceMinSamples samples land on any code.
Result<ReferenceFit> fit_reference(const SdrImage& sdr, const PlanarBuffer& reference, std::string file = "");

// The correction over a source curve: [0, then 256 values of
// target - log2(source_code_nits)]. Empty for an empty fit.
std::vector<float> reference_params(const ReferenceFit& fit, SourceCurve source, float corpus_ev);

// The exposure the fit asks for over a source curve: the count-weighted
// median of the per-code correction, in stops. 0 for an empty fit.
double reference_exposure(const ReferenceFit& fit, SourceCurve source, float corpus_ev);

// What the fit did, for the panel: "" for an empty fit.
std::string reference_summary(const ReferenceFit& fit, SourceCurve source, float corpus_ev);

// The fit as JSON (params(), a master's request) and back; the JSON carries
// the file, the counts, the residual and the target, nothing that depends on
// the source or the model. Refused when the target is not 256 finite numbers.
std::string reference_fit_json(const ReferenceFit& fit);
Result<ReferenceFit> reference_fit_from_json(const std::string& json_text);

}  // namespace rudra
