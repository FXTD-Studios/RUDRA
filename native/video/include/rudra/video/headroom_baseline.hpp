#pragma once
// Opt-in CPU baseline for ExtendedVideoFrame. Not wired to app or delivery.
// Policy and integration gates: docs/SUPER_WHITES_REVIEW_2026-10-16.md.

#include "rudra/core/headroom_policy.hpp"
#include "rudra/deliver/master.hpp"
#include "rudra/media/video_decode_float.hpp"

namespace rudra {

struct HeadroomBaseline {
    // Linear Rec.2020, 1 = 203 nits, using the existing master unit tag.
    // An inverse display policy, not a claim to recover original scene light.
    Image<space::SceneLinear2020> linear2020;
    HeadroomPolicy policy = HeadroomPolicy::Disabled;
    InputContract source_contract;
};

// Before grade/anchor/chroma/grain/shoulder/temporal processing. The request
// checks reconstruction eligibility only; it is NOT applied as a master grade.
// Requires analytic BT.1886, clip fill/peaks off, no calibration/reference and
// matching source primaries. The original limited-range Rec.709 transfer
// contract is required, but values are already normalized by FloatVideoDecoder.
// v1 applies max(encoded, 0)^2.4 in source primaries, then converts linear light
// to Rec.2020. No clamp at white; negative source samples remain in the input.
// Disabled/unknown policies refuse rather than silently selecting a fallback.
Result<HeadroomBaseline> reconstruct_headroom_baseline(const ExtendedVideoFrame& frame,
    const MasterRequest& request, HeadroomPolicy policy = HeadroomPolicy::Disabled);

}  // namespace rudra
