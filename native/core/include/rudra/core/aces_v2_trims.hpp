#pragma once
// A colorist's trims on the learned ACES 2.0 master (28 Oct 2026), applied
// after the model and its appearance constraint, the same in the viewer and in
// every render, and written into the provenance:
//   strength      0..1.5: the master moved from the plain ACES inverse
//                 (0) to the model (1) or past it, everywhere
//   saturation    0..2: the reconstructed pixels' colour about their own
//                 luminance (1 = as the model made it); outside the
//                 rudra.reconstructed mask nothing changes
//   exposure_ev   -3..3 stops on the whole master (28 Oct 2026 review), after strength
//   contrast      0.5..1.5 about 18 % grey, on luminance (hue kept), the whole master
//   peak_nits     a soft limit on the reconstructed pixels' luminance, in nits
//                 (diffuse white 1.0 = 203 nits); 0 = none. Below 75 % of the
//                 limit nothing moves; above, luminance rolls off toward it.
// The baseline is the plain ACES inverse of the same frame (aces_v2::view_inverse).
// Defaults change nothing: trims_identity() is true and the master is bit-identical.

#include <string>

#include "rudra/core/image.hpp"

namespace rudra::aces_v2 {

struct AcesTrims {
    double strength = 1.0;
    double saturation = 1.0;
    double peak_nits = 0.0;
    double exposure_ev = 0.0;
    double contrast = 1.0;
    bool identity() const {
        return strength == 1.0 && saturation == 1.0 && peak_nits <= 0.0 && exposure_ev == 0.0 && contrast == 1.0;
    }
    bool operator==(const AcesTrims&) const = default;
};

inline constexpr double kAcesDiffuseNits = 203.0;

// The trimmed master (3 planes, ACEScg). `mask` is the rudra.reconstructed plane.
PlanarBuffer apply_trims(const PlanarBuffer& master, const PlanarBuffer& baseline, const PlanarBuffer& mask,
                         const AcesTrims& t);

// "strength 0.80, saturation 0.90, peak 1000 nits" (for logs and provenance).
std::string trims_text(const AcesTrims& t);

}  // namespace rudra::aces_v2
