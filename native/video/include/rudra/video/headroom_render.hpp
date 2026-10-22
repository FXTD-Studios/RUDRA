#pragma once
// Shared CPU rendering for the opt-in headroom path. Callers must keep this
// separate from SdrImage, the legacy composite, and its PNG source cache.

#include "rudra/deliver/video_master.hpp"
#include "rudra/video/headroom_baseline.hpp"

namespace rudra {

struct HeadroomMaster {
    // After exposure and Region EV/masks, before delivery shoulder or encoding.
    Image<space::SceneLinear2020> linear2020;
    HeadroomPolicy policy = HeadroomPolicy::Disabled;
    InputContract source_contract;
};

// Validates artist controls without allocating/decoding a frame. Reconstruction
// and source-contract eligibility are additionally checked by the baseline.
// v1 requires anchor, carry_chroma and settle_grain OFF: those legacy stages
// require canonical SDR, which cannot represent this source. Recovery controls
// must be neutral (all/1/preserve), and model provenance must be absent.
// Exposure is [-8,+8] EV. Regions/masks use the existing Rec.2020 luminance
// qualifiers, with exposure first. No reconstruction ceiling or display shoulder
// is applied here; invalid/nonfinite/overflowing settings fail, never fall back.
Result<void> validate_headroom_grade(const MasterRequest& request);
Result<HeadroomMaster> render_headroom_frame(const ExtendedVideoFrame& frame,
    const MasterRequest& request, HeadroomPolicy policy = HeadroomPolicy::Disabled);

struct HeadroomPreviewImages {
    HeadroomMaster master;
    HeadroomBaseline baseline;
};

// Reconstruct once for the graded/baseline comparison. Uses the same double
// precision grade as the master, without a frame-sized double intermediate.
// Both images own their pixels; source pixels and masks remain unchanged.
// Still full resolution: this is not a proxy or an export-completion receipt.
Result<HeadroomPreviewImages> render_headroom_preview(const ExtendedVideoFrame& frame,
    const MasterRequest& request, HeadroomPolicy policy = HeadroomPolicy::Disabled);

// Explicit unit bridge to the existing movie delivery code. Converts 203-nit
// linear units to its 10000-nit convention once; never canonicalizes SDR or
// runs a network/composite. Peak/knee remain delivery controls (log profiles
// keep the existing scene-linear convention). Stateless: no temporal smoothing.
// These primitives do not enable the path in the app, CLI or project format.
Result<SpoolFrame> master_headroom_video_frame(const HeadroomMaster& frame, int index,
    const std::string& format, double peak_nits, std::optional<double> knee_nits = std::nullopt,
    bool analyze = false);

}  // namespace rudra
