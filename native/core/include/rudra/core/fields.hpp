#pragma once
// What inference hands the rest of the program (NATIVE_ARCHITECTURE.md 4):
// three fields per pixel and a few scalars per frame. Everything after this is
// a pure function of (sdr, fields, scalars, params), so a slider never runs the
// network. Defined in core because the composite reads them and core may not
// depend on infer.

#include <vector>

#include "rudra/core/image.hpp"

namespace rudra {

// Once per frame, from the whole frame. A model without a head reports the
// value that head's absence means: scale 1, weight 1, a one-element zero curve.
struct FrameScalars {
    float residual_scale = 1.0f;
    float shadow_weight = 1.0f;
    std::vector<float> curve_params{0.0f};
};

// The fields for a tile or a whole frame. Invariant to every user control.
struct Fields {
    PlanarBuffer residual;    // 3 x h x w, log domain, residual scale folded in
    PlanarBuffer highlight;   // 1 x h x w, in [0,1]
    PlanarBuffer shadow;      // 1 x h x w, in [0,1]
};

// The analytic reconstruction (no model): no residual and no recovery masks,
// so the composite is its analytic baseline (source curve, calibration and
// reference included) and nothing learned enters it. With the default
// FrameScalars no learned curve does either.
inline Fields analytic_fields(int height, int width) {
    return {PlanarBuffer(3, height, width), PlanarBuffer(1, height, width), PlanarBuffer(1, height, width)};
}

}  // namespace rudra
