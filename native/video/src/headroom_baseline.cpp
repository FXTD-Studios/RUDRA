#include "rudra/video/headroom_baseline.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>

#include "rudra/core/gamut.hpp"

namespace rudra {

Result<HeadroomBaseline> reconstruct_headroom_baseline(const ExtendedVideoFrame& frame,
    const MasterRequest& q, HeadroomPolicy policy) {
    if (policy != HeadroomPolicy::Bt1886V1)
        return make_error(ErrorCode::Unsupported, policy == HeadroomPolicy::Disabled
            ? "Headroom baseline is disabled" : "Unsupported headroom baseline policy");
    if (!q.analytic || q.source_curve != "bt1886")
        return make_error(ErrorCode::Unsupported, "Headroom v1 requires analytic BT.1886 reconstruction");
    if (q.clip_fill || !q.clip_peaks.empty())
        return make_error(ErrorCode::Unsupported, "Headroom v1 requires clipped-highlight fill and peak overrides off");
    if (!q.calibration.empty() || !q.reference.empty())
        return make_error(ErrorCode::Unsupported, "Headroom v1 does not support calibration or reference fitting");
    const auto& contract = frame.source_contract;
    if (contract.transfer != "rec709" || contract.range != "limited" ||
        (contract.matrix != "bt709" && contract.matrix != "bt2020nc") ||
        (contract.primaries != "rec709" && contract.primaries != "rec2020"))
        return make_error(ErrorCode::Unsupported, "Headroom v1 requires a supported limited-range Rec.709-transfer contract");
    if (q.source_space != contract.primaries)
        return make_error(ErrorCode::ContractMismatch, "Headroom source primaries disagree with the reconstruction request");

    const auto& in = frame.encoded_rgb;
    if (in.channels() != 3 || in.height() <= 0 || in.width() <= 0 ||
        std::size_t(in.width()) > std::numeric_limits<std::size_t>::max() / std::size_t(in.height()) / 3 ||
        in.span().size() != in.plane_size() * 3)
        return make_error(ErrorCode::InvalidArgument, "Headroom baseline requires a complete three-channel frame");
    const auto src = contract.primaries == "rec709" ? Primaries::Rec709 : Primaries::Rec2020;
    const auto matrix = rgb_to_rgb_matrix(src, Primaries::Rec2020);
    PlanarBuffer out;
    try {
        out = PlanarBuffer(3, in.height(), in.width());
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::IoError, "Could not allocate the headroom baseline");
    } catch (const std::length_error&) {
        return make_error(ErrorCode::InvalidArgument, "Headroom baseline exceeds the buffer size limit");
    }
    for (std::size_t i = 0; i < in.plane_size(); ++i) {
        double linear[3];
        for (int c = 0; c < 3; ++c) {
            const double code = in.plane(c)[i];
            if (!std::isfinite(code))
                return make_error(ErrorCode::InvalidArgument, "Nonfinite sample in headroom baseline input");
            // v1's zero-black 2.4 power policy. Units are relative to 203 nits;
            // no additional transfer decoding or range expansion belongs here.
            linear[c] = std::pow(std::max(code, 0.0), 2.4);
        }
        for (int c = 0; c < 3; ++c) {
            const auto& row = matrix[std::size_t(c)];
            const double value = row[0] * linear[0] + row[1] * linear[1] + row[2] * linear[2];
            if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
                return make_error(ErrorCode::InvalidArgument, "Headroom baseline exceeds finite float output range");
            out.plane(c)[i] = static_cast<float>(value);
        }
    }
    return HeadroomBaseline{Image<space::SceneLinear2020>(std::move(out)), policy, contract};
}

}  // namespace rudra
