#include "rudra/video/headroom_render.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>

#include "rudra/core/hdr10.hpp"

namespace rudra {

Result<void> validate_headroom_grade(const MasterRequest& q) {
    if (q.anchor || q.carry_chroma || q.settle_grain)
        return make_error(ErrorCode::Unsupported,
            "Headroom v1 requires SDR anchoring, chroma carry and grain settling off");
    if (q.recovery_mode != "all" || q.strength != 1.0 || !q.preserve_outside)
        return make_error(ErrorCode::Unsupported, "Headroom v1 requires neutral recovery controls (all, strength 1, preserve)");
    if (!q.checkpoint.empty() || !q.checkpoint_sha256.empty() || q.tiled)
        return make_error(ErrorCode::ContractMismatch, "Headroom v1 cannot record a model checkpoint or tiled inference");
    if (q.container != "aces" && q.container != "linear")
        return make_error(ErrorCode::InvalidArgument, "container must be aces or linear", q.container);
    if (!std::isfinite(q.exposure_ev) || q.exposure_ev < -8 || q.exposure_ev > 8)
        return make_error(ErrorCode::InvalidArgument, "Headroom exposure must be finite and between -8 and +8 EV");
    if (!std::isfinite(q.region_softness_stops) || q.region_softness_stops < 0)
        return make_error(ErrorCode::InvalidArgument, "Headroom region softness must be finite and nonnegative");
    for (const auto& b : q.regions)
        if (!std::isfinite(b.low_nits) || !std::isfinite(b.high_nits) || !std::isfinite(b.ev) ||
            b.low_nits <= 0 || b.high_nits <= b.low_nits || b.ev < -8 || b.ev > 8)
            return make_error(ErrorCode::InvalidArgument,
                "Headroom regions require 0 < low < high and finite EV between -8 and +8", b.label);
    if (!q.masks_file.empty() && !q.masks)
        return make_error(ErrorCode::InvalidArgument, "Headroom masks_file requires loaded masks");
    if (q.masks && !q.masks->empty()) {
        const auto& m = *q.masks;
        if (m.width <= 0 || m.height <= 0 ||
            std::size_t(m.width) > std::numeric_limits<std::size_t>::max() / std::size_t(m.height))
            return make_error(ErrorCode::InvalidArgument, "Invalid headroom mask dimensions");
        const auto bands = q.regions.empty() ? default_region_bands().size() : q.regions.size();
        for (std::size_t b = 0; b < m.planes.size(); ++b)
            if (m.planes[b] && (b >= bands || m.planes[b]->size() != std::size_t(m.width) * m.height))
                return make_error(ErrorCode::InvalidArgument, "Headroom mask does not match its dimensions or region band");
    }
    return {};
}

namespace {
// Call only with a freshly reconstructed baseline and validated controls.
// Keep apply_region_ev's arithmetic order and existing gain/mask functions;
// a pixel's double working values do not need a whole NitsFrame allocation.
Result<HeadroomMaster> grade_baseline(HeadroomBaseline baseline, const MasterRequest& q) {
    try {
        std::vector<RegionBand> bands;
        for (const auto& b : q.regions) bands.push_back({b.low_nits, b.high_nits, b.ev});
        // Empty regions are neutral, just as the existing master request is.
        const bool banded = any_graded(bands);
        if (q.exposure_ev != 0.0 || banded) {
            auto& pixels = baseline.linear2020.buffer();
            const double exposure = exposure_gain(q.exposure_ev);
            const bool active = banded || exposure != 1.0;
            const bool masked = banded && q.masks && !q.masks->empty();
            const auto width = std::size_t(pixels.width());
            for (std::size_t i = 0; i < pixels.plane_size(); ++i) {
                double px[3];
                for (int c = 0; c < 3; ++c)
                    px[c] = (double(pixels.plane(c)[i]) * 203.0) * exposure;
                float weights[kMaxMaskBands];
                if (masked) q.masks->weights(int(i % width), int(i / width),
                                            pixels.width(), pixels.height(), weights);
                const double gain = banded ? region_ev_gain(px, bands, q.region_softness_stops,
                                                            masked ? weights : nullptr) : 1.0;
                for (int c = 0; c < 3; ++c) {
                    // apply_region_ev is a no-op when a sub-epsilon exposure
                    // rounds to unity and no band is graded (including clamp).
                    const double graded = active ? std::clamp(px[c] * gain, 0.0, std::numeric_limits<double>::max()) : px[c];
                    const double value = graded / 203.0;
                    if (!std::isfinite(value) || value > std::numeric_limits<float>::max() || value < 0)
                        return make_error(ErrorCode::InvalidArgument, "Headroom grade exceeds finite float output range");
                    pixels.plane(c)[i] = static_cast<float>(value);
                }
            }
        }
        return HeadroomMaster{std::move(baseline.linear2020), baseline.policy, std::move(baseline.source_contract)};
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::IoError, "Could not allocate the headroom grade");
    } catch (const std::length_error&) {
        return make_error(ErrorCode::InvalidArgument, "Headroom grade exceeds the buffer size limit");
    }
}
} // namespace

Result<HeadroomMaster> render_headroom_frame(const ExtendedVideoFrame& frame,
    const MasterRequest& q, HeadroomPolicy policy) {
    if (auto valid = validate_headroom_grade(q); !valid) return valid.error();
    auto baseline = reconstruct_headroom_baseline(frame, q, policy);
    if (!baseline) return baseline.error();
    return grade_baseline(std::move(*baseline), q);
}

Result<HeadroomPreviewImages> render_headroom_preview(const ExtendedVideoFrame& frame,
    const MasterRequest& q, HeadroomPolicy policy) {
    if (auto valid = validate_headroom_grade(q); !valid) return valid.error();
    auto baseline = reconstruct_headroom_baseline(frame, q, policy);
    if (!baseline) return baseline.error();
    try {
        auto master = grade_baseline(*baseline, q);
        if (!master) return master.error();
        return HeadroomPreviewImages{std::move(*master), std::move(*baseline)};
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::IoError, "Could not allocate the headroom preview");
    } catch (const std::length_error&) {
        return make_error(ErrorCode::InvalidArgument, "Headroom preview exceeds the buffer size limit");
    }
}

Result<SpoolFrame> master_headroom_video_frame(const HeadroomMaster& frame, int index,
    const std::string& format, double peak_nits, std::optional<double> knee_nits, bool analyze) {
    if (frame.policy != HeadroomPolicy::Bt1886V1)
        return make_error(ErrorCode::Unsupported, "Headroom movie frame requires bt1886-headroom-v1");
    // encode_master historically treats unknown names as PQ. This new entry
    // point fails closed so a typo cannot quietly select a delivery format.
    if (!find_delivery_profile(format))
        return make_error(ErrorCode::Unsupported, "Unknown headroom delivery format", format);
    const auto& in = frame.linear2020.buffer();
    if (in.channels() != 3 || in.width() <= 0 || in.height() <= 0 ||
        std::size_t(in.width()) > std::numeric_limits<std::size_t>::max() / std::size_t(in.height()) / 3 ||
        in.span().size() != in.plane_size() * 3)
        return make_error(ErrorCode::InvalidArgument, "Headroom movie frame requires a complete three-channel image");
    try {
        PlanarBuffer units(3, in.height(), in.width());
        for (std::size_t i = 0; i < in.span().size(); ++i) {
            const double v = double(in.span()[i]) * 203.0;
            // The existing encoder measures absolute nits in float32.
            if (!std::isfinite(v) || v < 0 || v > std::numeric_limits<float>::max())
                return make_error(ErrorCode::InvalidArgument, "Headroom movie frame exceeds finite float nits range");
            units.span()[i] = static_cast<float>(v / 10000.0);
        }
        return master_video_frame(NetworkLinearImage(std::move(units)), index, format, peak_nits, knee_nits, {}, analyze);
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::IoError, "Could not allocate the headroom movie frame");
    } catch (const std::length_error&) {
        return make_error(ErrorCode::InvalidArgument, "Headroom movie frame exceeds the buffer size limit");
    }
}

}  // namespace rudra
