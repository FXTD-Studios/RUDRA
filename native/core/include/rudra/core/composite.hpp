#pragma once
// The composite: picture = composite(sdr, fields, scalars, params). A pure
// function, the one every user control lives in (NATIVE_ARCHITECTURE.md 4 and
// docs/composite.spec.md). It exists once as a spec and twice as code: this
// file (fp32, the CPU path and the reference) and the render layer's shader.
//
// Port of the tail of rudra/sdr2hdr.py SDR2HDRNet.forward (as fed by
// training/infer_sdr2hdr.py predict_fields) plus Region EV from
// rudra/delivery/controls.py, which is also what ui/compositor.js runs.

#include <span>
#include <vector>

#include "rudra/core/fields.hpp"
#include "rudra/core/image.hpp"
#include "rudra/core/calibration.hpp"
#include "rudra/core/masks.hpp"
#include "rudra/core/reference_fit.hpp"
#include "rudra/core/source_curve.hpp"

namespace rudra {

enum class RecoveryMode : std::uint8_t { All, Highlights, Shadows, Off };

// The three constants a model package carries (manifest.json).
struct ModelConstants {
    float log_scale = 16.0f;
    float max_hdr = 4.0f;       // network units: 4.0 = 40 000 nits
    float corpus_ev = -1.0f;
};

// The analytic reconstruction's constants, whatever package is installed: the
// inverse at the plate's own exposure (corpus EV 0, as sdr2hdr_image_v8's) and
// the network convention's 40 000-nit ceiling.
inline constexpr ModelConstants kAnalyticConstants{16.0f, 4.0f, 0.0f};

// One luminance-qualified EV band, in absolute nits (Rec.2020 luma).
struct RegionBand {
    double low_nits = 0.0;
    double high_nits = 0.0;
    double ev = 0.0;
};

// The bands the Region EV panel opens with. Neutral: ev 0 everywhere.
std::vector<RegionBand> default_region_bands();

bool any_graded(std::span<const RegionBand> bands) noexcept;

struct CompositeParams {
    RecoveryMode mode = RecoveryMode::All;
    float strength = 1.0f;
    bool preserve_outside = true;
    std::vector<RegionBand> regions;       // empty or all ev 0: no grade
    double region_softness_stops = 1.0;
    // Roadmap 3.1: the curve the artist says made the SDR. The baseline is
    // its inverse; Unknown is the ACES inverse every master so far used.
    SourceCurve source = SourceCurve::Unknown;
    // Roadmap 3.2: the artist's anchors over that curve (calibration.hpp);
    // empty for none. Applied only when the result is monotone.
    std::vector<CalibrationPoint> calibration;
    // Roadmap 3.4: the reference match (reference_fit.hpp); empty for none.
    // A reference is an anchor at every code, so when set it replaces the
    // calibration.
    ReferenceFit reference;
    // Roadmap 3.3: painted masks gating the bands' qualifiers (masks.hpp);
    // null or empty for none.
    std::shared_ptr<const MaskSet> masks;
};

// The curve params corrected_baseline takes for a frame under these settings:
// the model's CurveHead output plus the source curve (source_curve.hpp) plus
// the calibration (calibration.hpp, which needs the model's exposure because
// its anchors are in nits), or the reference fit in the calibration's place
// (reference_fit.hpp). Every consumer of the baseline (composite, master,
// measure, the viewer and its probes) goes through this, so they agree on
// what the baseline is.
std::vector<float> baseline_curve_params(const FrameScalars& scalars, const CompositeParams& params, float corpus_ev);

// qualifier_mask for one pixel's Rec.2020 luminance in nits. float32 result,
// exactly as the Python casts it before the gain sums it.
float qualifier_mask(double luma_nits, double low_nits, double high_nits, double softness_stops) noexcept;

// region_ev_gain for one pixel in absolute nits: 2^(sum ev * mask), each
// band's term times its painted weight when `weights` is given (one per band
// index up to kMaxMaskBands, 1 beyond; masks.hpp).
double region_ev_gain(const double rgb_nits[3], std::span<const RegionBand> bands, double softness_stops,
                      const float* weights = nullptr) noexcept;

// The full composite into the network convention (1.0 = 10 000 nits). With a
// grade, Region EV is applied and the result clamped to [0, max_hdr].
// `grade_gain`, when given, receives each pixel's Region EV gain (1 without a
// grade): the live anchor scales its target by it, so a grade survives the
// anchor in the view as it does in the master (ViewParams::grade_gain).
NetworkLinearImage composite(const SdrImage& sdr, const Fields& fields, const FrameScalars& scalars,
                             const ModelConstants& model, const CompositeParams& params,
                             std::vector<float>* grade_gain = nullptr);

}  // namespace rudra
